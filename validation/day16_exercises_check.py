"""Day 16 exercises <-> solutions.

Exercises 1-2 ship test functions: run them with the solution classes swapped in.
Exercises 3-5 are scaffolds whose solutions only define training functions (their
__main__ just prints "... loaded."), so this script exercises those functions directly:
  - solution_3: a few epochs of sorting (N=5) with the repo PointerNetwork -> accuracy rises?
  - solution_4: one convex-hull batch -> is the loss finite? (padded hull labels)
    [2026-10-18: STOP pointer + canonical labels; now also a short run of train_convex_hull()]
  - solution_5: a few TSP batches -> finite, decreasing loss?
  - exercise_5.compute_tour_length vs an independent closed-tour length; nearest-neighbour
    label quality vs brute-force optimal tours (n = 7).
  - exercise_4.compute_hull_accuracy sanity.

Run: OMP_NUM_THREADS=1 python validation/day16_exercises_check.py
"""
import contextlib
import importlib.util
import io
import itertools
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, "papers", "16_order_matters")
EX = os.path.join(DAY, "exercises")
sys.path.insert(0, DAY)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


results = {}
# ---------------------------------------------------------------- ex 1, 2: shipped tests
for k, cls in [(1, "SimplePointerAttention"), (2, "SetEncoder")]:
    ex = load(f"{EX}/exercise_{k}.py", f"ex{k}")
    sol = load(f"{EX}/solutions/solution_{k}.py", f"sol{k}")
    setattr(ex, cls, getattr(sol, cls))
    torch.manual_seed(0)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for name in dir(ex):
                if name.startswith("test_"):
                    getattr(ex, name)()
            for name in dir(sol):                     # the solution's own asserting test
                if name.startswith("test_"):
                    getattr(sol, name)()
        results[f"exercise_{k}"] = "pass"
    except Exception as e:  # noqa: BLE001
        results[f"exercise_{k}"] = f"FAIL {type(e).__name__}: {e}"
    print(f"exercise_{k} with solution {cls}: {results[f'exercise_{k}']}")
    print("   " + " | ".join(l.strip() for l in buf.getvalue().splitlines() if "weight" in l.lower() or "difference" in l.lower())[:300])

# ---------------------------------------------------------------- ex 3: sorting actually trains?
import implementation as imp  # noqa: E402

sol3 = load(f"{EX}/solutions/solution_3.py", "sol3")
torch.manual_seed(0)
model = imp.PointerNetwork(input_dim=1, hidden_dim=64, dropout=0.0)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(sol3.SortingDataset(2000, 5), batch_size=32, shuffle=True)
t0 = time.time()
hist = []
for ep in range(8):
    loss, acc = sol3.train_epoch(model, loader, opt, "cpu")
    hist.append((round(loss, 3), round(acc, 3)))
model.eval()
with torch.no_grad():
    xs = torch.rand(500, 5, 1) * 10
    ptr, _, _ = model(xs, torch.full((500,), 5), 5)
    greedy_acc = (ptr == xs[..., 0].argsort(1)).all(1).float().mean().item()
print(f"solution_3 sorting N=5, 8 epochs x 63 batches ({time.time() - t0:.0f}s): (loss, teacher-forced acc) {hist}")
print(f"   greedy (no teacher forcing) exact-sort accuracy on 500 fresh sets: {greedy_acc:.1%}")
results["solution_3"] = f"trains: final loss {hist[-1][0]}, greedy acc {greedy_acc:.1%}"

# ---------------------------------------------------------------- ex 4: convex hull loss finite?
sol4 = load(f"{EX}/solutions/solution_4.py", "sol4")
torch.manual_seed(0)
pts, tgt = next(iter(DataLoader(sol4.ConvexHullDataset(32, 10), batch_size=32)))
# [PATCH 2026-10-18] labels: canonical hull + STOP (= 10) + IGNORE_INDEX; the model needs the STOP slot
rpw = imp.ReadProcessWrite(input_dim=2, hidden_dim=32, stop_token=True)
_, lp, _ = rpw(pts, torch.full((32,), 10), 11, teacher_forcing=tgt)
hull_loss = imp.compute_loss(lp, tgt)
print(f"solution_4 convex hull: one batch loss = {hull_loss.item():.4f} "
      f"({int(torch.isinf(lp).sum())} of {lp.numel()} teacher-forced steps have log-prob -inf)")
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    hist4, exact4 = sol4.train_convex_hull(num_epochs=3, num_samples=640, hidden_dim=64)
print(f"solution_4.train_convex_hull(3 epochs x 640 sets, hidden 64): losses {[round(v, 3) for v in hist4]}, "
      f"greedy exact hulls {exact4:.1%} ({time.time() - t0:.0f}s)")
results["solution_4"] = (f"finite loss {hull_loss.item():.3f}; trains: {hist4[0]:.3f} -> {hist4[-1]:.3f}, "
                         f"exact hulls {exact4:.1%}")

ex4 = load(f"{EX}/exercise_4.py", "ex4")
em, ov = ex4.compute_hull_accuracy(torch.tensor([3, 1, 2]), torch.tensor([1, 2, 3]))
print(f"exercise_4.compute_hull_accuracy(rotated/reordered hull): exact={em}, overlap={ov}  "
      "(a cyclic rotation of the same hull scores exact=False)")

# ---------------------------------------------------------------- ex 5: TSP
sol5 = load(f"{EX}/solutions/solution_5.py", "sol5")
ex5 = load(f"{EX}/exercise_5.py", "ex5")
torch.manual_seed(0)
cities = torch.rand(10, 2)
tour = torch.randperm(10)
ref = sum(torch.dist(cities[tour[i]], cities[tour[(i + 1) % 10]]).item() for i in range(10))
print(f"exercise_5.compute_tour_length vs independent closed-tour sum: |diff| = {abs(ex5.compute_tour_length(tour, cities) - ref):.2e}")


def brute_force(c):
    n = len(c)
    best = float("inf")
    for p in itertools.permutations(range(1, n)):
        t = (0,) + p
        best = min(best, sum(torch.dist(c[t[i]], c[t[(i + 1) % n]]).item() for i in range(n)))
    return best


torch.manual_seed(1)
ratios = []
for _ in range(20):
    c = torch.rand(7, 2)
    nn_tour = sol5.TSPDataset(1, 7)._greedy_tour(c) if hasattr(sol5.TSPDataset, "_greedy_tour") else None
    if nn_tour is None:      # solution_5 builds the NN tour inside __getitem__; reuse exercise_5's helper
        nn_tour = ex5.TSPDataset(1, 7)._greedy_tour(c)
    ratios.append(ex5.compute_tour_length(nn_tour, c) / brute_force(c))
print(f"nearest-neighbour label tours (n=7, 20 instances): mean length / optimal = {sum(ratios) / len(ratios):.3f}, "
      f"max = {max(ratios):.3f}")

torch.manual_seed(0)
model5 = imp.ReadProcessWrite(input_dim=2, hidden_dim=32)  # [PATCH 2026-10-18] new constructor
opt5 = torch.optim.Adam(model5.parameters(), lr=1e-3)
losses = []
for b, (cty, tg) in enumerate(DataLoader(sol5.TSPDataset(320, 10), batch_size=32, shuffle=True)):
    _, lp5, _ = model5(cty, torch.full((cty.size(0),), 10), 10, teacher_forcing=tg)
    loss5 = -lp5.mean()
    opt5.zero_grad()
    loss5.backward()
    opt5.step()
    losses.append(round(loss5.item(), 3))
print(f"solution_5-style TSP loop, 10 batches: losses {losses}")
results["solution_5"] = f"runs, loss {losses[0]} -> {losses[-1]}"
print("\nSUMMARY:", results)
