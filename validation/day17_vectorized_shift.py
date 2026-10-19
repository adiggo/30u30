"""Day 17 check: the vectorized circular convolution (Eq 8) changes nothing but the speed (2026-10-18).

NTMHead._addressing used to compute Eq 8 with a Python loop over the N memory slots, re-rolling the whole
(B, N) weighting twice per slot. It is now one expression,
    w_s = s[:, 0:1] * roll(w_g, -1) + s[:, 1:2] * w_g + s[:, 2:3] * roll(w_g, +1).
This script
  1. rebuilds the old loop (copied from git history) as a subclass and compares it with the new
     code in float64: addressing weights, erase/add vectors and the gradients of a scalar loss
     w.r.t. every parameter and input (tolerance 1e-12; the forward is the same arithmetic);
  2. runs one copy-task training iteration (forward, BCE, backward, clipping, RMSprop step) with
     both versions from the same weights and data, and compares the parameters afterwards;
  3. times forward + backward of one copy-task iteration (batch 32, N = 128, M = 20,
     controller 100, length 10 -> 21 steps) for both versions (informational).
Path-independent; prints PASS/FAIL; exits 0.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "17_neural_turing_machines")
sys.path.insert(0, DAY)
import implementation as impl  # noqa: E402
from train_minimal import generate_copy_instance  # noqa: E402

torch.set_num_threads(1)
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


class LoopHead(impl.NTMHead):
    """The pre-2026-10-18 addressing: identical except Eq 8, which loops over the N slots."""

    def _addressing(self, controller_out, prev_w, memory):
        out = self.fc(controller_out)
        ptr = 0
        k = out[:, ptr:ptr + self.M]; ptr += self.M
        beta = F.softplus(out[:, ptr:ptr + 1]); ptr += 1
        g = torch.sigmoid(out[:, ptr:ptr + 1]); ptr += 1
        s = F.softmax(out[:, ptr:ptr + 3], dim=1); ptr += 3
        gamma = 1 + F.softplus(out[:, ptr:ptr + 1]); ptr += 1
        erase = add = None
        if not self.is_read:
            erase = torch.sigmoid(out[:, ptr:ptr + self.M]); ptr += self.M
            add = out[:, ptr:ptr + self.M]
        norm_k = k.norm(p=2, dim=1, keepdim=True) + 1e-8
        norm_mem = memory.norm(p=2, dim=2) + 1e-8
        sim = torch.matmul(memory, k.unsqueeze(2)).squeeze(2) / (norm_k * norm_mem)
        w_c = F.softmax(beta * sim, dim=1)
        w_g = g * w_c + (1 - g) * prev_w
        w_s = torch.zeros_like(w_g)
        for i in range(self.N):                                 # the old O(N^2) loop
            w_s[:, i] = (s[:, 0] * torch.roll(w_g, shifts=-1, dims=1)[:, i] +
                         s[:, 1] * w_g[:, i] +
                         s[:, 2] * torch.roll(w_g, shifts=1, dims=1)[:, i])
        w_pow = (w_s / (w_s.max(dim=1, keepdim=True)[0] + 1e-12)) ** gamma
        w = w_pow / (w_pow.sum(dim=1, keepdim=True) + 1e-8)
        return w, erase, add


def loop_twin(model):
    """Same weights, loop heads."""
    twin = impl.NTM(model.input_size, model.output_size, model.controller_size, model.N, model.M)
    twin.read_heads = nn.ModuleList([LoopHead(twin.memory, model.controller_size, True)])
    twin.write_heads = nn.ModuleList([LoopHead(twin.memory, model.controller_size, False)])
    twin = twin.to(next(model.parameters()).dtype)
    twin.load_state_dict(model.state_dict())
    return twin


# ---- 1. addressing + gradients, float64 -------------------------------------------------------------
torch.manual_seed(0)
worst_fwd, worst_grad = 0.0, 0.0
for (B, N, M, H) in [(1, 5, 3, 4), (3, 16, 6, 12), (2, 128, 20, 100)]:
    for is_read in (True, False):
        mem = impl.NTMMemory(N, M)
        new = impl.NTMHead(mem, H, is_read).double()
        old = LoopHead(mem, H, is_read).double()
        old.load_state_dict(new.state_dict())
        with torch.no_grad():
            for prm in new.parameters():
                prm.copy_(torch.randn_like(prm) * 0.5)
        old.load_state_dict(new.state_dict())
        h = torch.randn(B, H, dtype=torch.float64, requires_grad=True)
        prev = torch.softmax(torch.randn(B, N, dtype=torch.float64), 1).requires_grad_(True)
        memory = torch.randn(B, N, M, dtype=torch.float64, requires_grad=True)
        outs, grads = [], []
        for head in (new, old):
            head.zero_grad()
            w, e, a = head._addressing(h, prev, memory)
            parts = [w] + ([] if is_read else [e, a])
            # a fixed random linear functional of the outputs, the same for both heads
            loss = sum((p_ * torch.randn(p_.shape, dtype=p_.dtype,
                                         generator=torch.Generator().manual_seed(7 + j))).sum()
                       for j, p_ in enumerate(parts))
            gs = torch.autograd.grad(loss, [h, prev, memory] + list(head.parameters()))
            outs.append(parts); grads.append(gs)
        worst_fwd = max(worst_fwd, max((x - y).abs().max().item() for x, y in zip(*outs)))
        worst_grad = max(worst_grad, max((x - y).abs().max().item() for x, y in zip(*grads)))
report("vectorized Eq 8 vs the old loop (float64)", worst_fwd <= 1e-12 and worst_grad <= 1e-12,
       f"max |diff| forward {worst_fwd:.1e}, gradients (inputs and all head parameters) {worst_grad:.1e} "
       f"(tol 1e-12); N = 5, 16, 128, read and write heads")

# ---- 2. one full training iteration ---------------------------------------------------------------
torch.manual_seed(1)
model = impl.NTM(9, 8, 100, 128, 20).double()
twin = loop_twin(model)
x, y = generate_copy_instance(8, 6, 8)
x, y = x.double(), y.double()
for net in (model, twin):
    opt = torch.optim.RMSprop(net.parameters(), lr=1e-4, momentum=0.9, alpha=0.95)
    st = net.reset(8); outs = []
    for t in range(x.size(0)):
        o, st = net(x[t], st); outs.append(o)
    loss = nn.BCEWithLogitsLoss()(torch.stack(outs)[7:], y[7:])
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_value_(net.parameters(), 10.0)
    opt.step()
diff = max((p - q).abs().max().item() for p, q in zip(model.parameters(), twin.parameters()))
report("one training iteration, vectorized vs loop (float64)", diff <= 1e-12,
       f"max |parameter diff| after an RMSprop step = {diff:.1e} (tol 1e-12)")

# ---- 3. timing ------------------------------------------------------------------------------------------
torch.manual_seed(2)
model = impl.NTM(9, 8, 100, 128, 20)
twin = loop_twin(model)
x, y = generate_copy_instance(32, 10, 8)


def fwd_bwd(net):
    st = net.reset(32); outs = []
    for t in range(x.size(0)):
        o, st = net(x[t], st); outs.append(o)
    nn.BCEWithLogitsLoss()(torch.stack(outs)[11:], y[11:]).backward()


times = {}
for name, net in (("vectorized", model), ("loop", twin)):
    fwd_bwd(net)                                   # warm-up
    t0 = time.perf_counter()
    reps = 5 if name == "vectorized" else 2
    for _ in range(reps):
        fwd_bwd(net)
    times[name] = (time.perf_counter() - t0) / reps
print(f"[info] forward+backward, batch 32, N = 128, length 10 (21 steps), 1 thread: loop {times['loop']:.3f} s, "
      f"vectorized {times['vectorized']:.3f} s -> {times['loop'] / times['vectorized']:.1f}x faster")

print("RESULT:", "PASS" if all(results) else "FAIL")
sys.exit(0)
