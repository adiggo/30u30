"""Day 21: the paper's set2set readout, and which readout suits the repo's synthetic target (2026-10-18).

Gilmer et al. Sec. 5.3: set2set "first applies a linear projection to each tuple (h_v^T, x_v)" and then runs
M steps of the Vinyals et al. (2016) process block: q_t = LSTM(q*_{t-1}); e_v = m_v . q_t; a = softmax_v(e);
r_t = sum_v a_v m_v; q*_t = [q_t, r_t]; q*_0 = 0; output = MLP(q*_M).

  1. implementation.Set2SetReadout (with node_dim) == an independent float64 NumPy implementation of the
     formulas above (tolerance 1e-12); outputs invariant to relabelling the nodes.
  2. Readout comparison on the synthetic data, train_minimal.train_model with its settings (500 molecules,
     80/20 split, seed 42, hidden 64, T = 3, Adam 1e-3, L1, 50 epochs): sum, set2set as it was before
     this change (attend-then-LSTM order, reads h_v^T only; rebuilt below), set2set fixed (this file's
     subject), and set2set fixed with T = 6 rounds. Baselines from validation/day21_worked_example.py:
     predict-the-mean 2.98, least squares on [atom counts, #bonds, 1] 0.089.
Prints PASS/FAIL for part 1 (part 2 is informational); exits 0. ~3 min on one core.

Usage: OMP_NUM_THREADS=1 python validation/day21_readout_fix.py [seed ...]   (default: 42; the README also
quotes seeds 1 and 2, ~2 min each -- note the seed also changes the generated molecules)
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "21_Neural_Message_Passing")
sys.path.insert(0, DAY)
import implementation as I  # noqa: E402
import train_minimal as TM  # noqa: E402

torch.set_num_threads(1)
ok = True


def sig(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- 1. set2set vs NumPy (float64)
torch.manual_seed(0)
d, nd, steps = 6, 5, 4
s2s = I.Set2SetReadout(hidden_dim=d, output_dim=3, processing_steps=steps, node_dim=nd).double().eval()
H = torch.randn(11, d, dtype=torch.float64)                 # final node states h_v^T
X = torch.randn(11, nd, dtype=torch.float64)                # input features x_v
bidx = torch.tensor([0] * 4 + [1] * 7)                      # two graphs: 4 and 7 nodes
with torch.no_grad():
    y_code = s2s(H, bidx, X).numpy()

P = s2s.project
Wp, bp = P.weight.detach().numpy(), P.bias.detach().numpy()
lstm = s2s.lstm
Wi, Wh = lstm.weight_ih.detach().numpy(), lstm.weight_hh.detach().numpy()
bl = (lstm.bias_ih + lstm.bias_hh).detach().numpy()
l1, l2 = s2s.mlp[0], s2s.mlp[2]


def np_lstm(x, h, c):
    z = x @ Wi.T + h @ Wh.T + bl                            # gate order i, f, g, o (PyTorch)
    i_, f_, g_, o_ = sig(z[:, :d]), sig(z[:, d:2 * d]), np.tanh(z[:, 2 * d:3 * d]), sig(z[:, 3 * d:])
    c = f_ * c + i_ * g_
    return o_ * np.tanh(c), c


Hn, Xn, b = H.numpy(), X.numpy(), bidx.numpy()
m = np.concatenate([Hn, Xn], 1) @ Wp.T + bp                 # projected tuples m_v = W [h_v ; x_v] + b
q_star = np.zeros((2, 2 * d)); q = np.zeros((2, d)); c = np.zeros((2, d))
for _ in range(steps):
    q, c = np_lstm(q_star, q, c)                            # q_t = LSTM(q*_{t-1})
    r = np.zeros((2, d))
    for g in range(2):
        mg = m[b == g]                                      # nodes of graph g
        e = mg @ q[g]                                       # e_v = m_v . q_t
        a = np.exp(e - e.max()); a /= a.sum()               # softmax over the graph's nodes
        r[g] = a @ mg                                       # r_t = sum_v a_v m_v
    q_star = np.concatenate([q, r], 1)                      # q*_t = [q_t ; r_t]
y_ref = np.maximum(q_star @ l1.weight.detach().numpy().T + l1.bias.detach().numpy(), 0) \
    @ l2.weight.detach().numpy().T + l2.bias.detach().numpy()
err = np.abs(y_code - y_ref).max()
perm = torch.randperm(11)
with torch.no_grad():
    y_perm = s2s(H[perm], bidx[perm], X[perm]).numpy()
inv = np.abs(y_perm - y_code).max()
print(f"[set2set] code vs NumPy of Sec. 5.3 + Vinyals Eqs. 3-7 (float64): max|delta| = {err:.2e}; "
      f"node relabelling: max|delta| = {inv:.2e}")
ok &= err < 1e-12 and inv < 1e-12

# also through the full MPNN: the readout receives x_v
model = I.MPNN(node_dim=5, edge_dim=4, hidden_dim=8, output_dim=2, readout_type="set2set").double()
seen = {}
hook = model.readout.register_forward_hook(lambda mod, inp, out: seen.update(n=len(inp)))
batch = I.batch_graphs([I.generate_synthetic_molecule(n_targets=2, seed=s) for s in range(3)])
batch = I.BatchedGraph(batch.node_features.double(), batch.edge_index, batch.edge_features.double(),
                       batch.batch_indices, batch.targets.double())
with torch.no_grad():
    model(batch)
hook.remove()
print(f"[MPNN] set2set readout called with (h, batch, x): {seen.get('n') == 3}")
ok &= seen.get("n") == 3
print(f"RESULT (part 1): {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------- 2. readout comparison
class OldSet2Set(nn.Module):
    """The readout as it was before 2026-10-18: attend with the previous LSTM state, then
    LSTM([h, r]); reads h_v^T only; outputs [h_M, r_M] with r_M from h_{M-1}."""

    def __init__(self, hidden_dim, output_dim, processing_steps=6):
        super().__init__()
        self.hidden_dim, self.processing_steps = hidden_dim, processing_steps
        self.lstm = nn.LSTMCell(2 * hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, node_embeddings, batch_indices, node_features=None):
        n_graphs = batch_indices.max().item() + 1
        h = node_embeddings.new_zeros(n_graphs, self.hidden_dim)
        c = node_embeddings.new_zeros(n_graphs, self.hidden_dim)
        r = node_embeddings.new_zeros(n_graphs, self.hidden_dim)
        for _ in range(self.processing_steps):
            e = (node_embeddings * h[batch_indices]).sum(dim=-1)
            a = I.scatter_softmax(e, batch_indices, dim_size=n_graphs)
            r = I.scatter_sum(a.unsqueeze(-1) * node_embeddings, batch_indices, dim_size=n_graphs)
            h, c = self.lstm(torch.cat([h, r], dim=-1), (h, c))
        return self.mlp(torch.cat([h, r], dim=-1))


_MPNN = I.MPNN


def mpnn_with_old_readout(*a, **k):
    model = _MPNN(*a, **k)
    model.readout = OldSet2Set(model.hidden_dim, k.get("output_dim", 1), k.get("set2set_steps", 6))
    return model


base = dict(hidden_dim=64, n_messages=3, message_type="edge_network", readout="sum", set2set_steps=6,
            epochs=50, lr=1e-3, batch_size=32, n_molecules=500, n_targets=3, compare_messages=False,
            seed=42, output_dir=tempfile.gettempdir())  # train_model never writes there
seeds = [int(a) for a in sys.argv[1:]] or [42]
print("\n== readout comparison (test MAE on 100 held-out molecules, 3 targets; "
      "seed-42 baselines: predict the mean 2.98, least squares on counts 0.089)")
for seed in seeds:
    runs = [("sum", "sum", {}), ("set2set, as it was", "set2set", {"old": True}),
            ("set2set, fixed (Sec. 5.3)", "set2set", {})]
    if seed == seeds[0]:
        runs.append(("set2set, fixed, T = 6", "set2set", {"n_messages": 6}))
    for name, ro, extra in runs:
        args = argparse.Namespace(**{**base, "seed": seed, **{k: v for k, v in extra.items() if k != "old"}})
        TM.MPNN = mpnn_with_old_readout if extra.get("old") else _MPNN
        t0 = time.time()
        _, losses, metrics = TM.train_model(args, readout_type=ro, verbose=False)
        maes = [mm["mae"] for mm in metrics]
        print(f"  seed {seed:2d} {name:27s} final {maes[-1]:.3f}  best {min(maes):.3f}  "
              f"final train L1 {losses[-1]:.3f}  ({time.time() - t0:.0f} s)", flush=True)
TM.MPNN = _MPNN
sys.exit(0)
