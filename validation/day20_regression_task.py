"""Day 20: the paper-faithful RMC on the repo's OLD task, against its trivial baselines.

Until 2026-10-18 train_minimal.py regressed "the vector farthest from vector 0" (15 vectors of 16 dims, MSE,
batch 32, Adam lr 1e-3, 2000 steps; read here from git commit ORIG_REF). With the old simplified cell, both
models ended near the trivial baselines (RMC val MSE 0.381, LSTM 0.372 vs predict-0 0.395 and the best fixed
-c * v0 predictor 0.345). This script trains the new RMC (and the same LSTM baseline) on that task with the
old settings and reports the validation MSE next to those baselines. It is a record, not a pass/fail check.

Usage: OMP_NUM_THREADS=1 python validation/day20_regression_task.py [steps]
"""
import ast
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, "papers", "20_Relational_RNNs"))
from implementation import RelationalRNN, StandardLSTM, count_parameters  # noqa: E402

ORIG_REF = "dfcaad3"   # last commit with the regression task
src = subprocess.run(["git", "-C", REPO, "show", f"{ORIG_REF}:papers/20_Relational_RNNs/train_minimal.py"],
                     capture_output=True, text=True, check=True).stdout
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "generate_nth_farthest_batch")
ns = {"torch": torch}
exec(compile(ast.Module(body=[fn], type_ignores=[]), f"train_minimal.py@{ORIG_REF}", "exec"), ns)
gen = ns["generate_nth_farthest_batch"]           # (batch, seq_len, dim) -> inputs, farthest-from-vector-0 targets

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEQ, DIM, BATCH, LR = 15, 16, 32, 1e-3             # the old defaults

torch.manual_seed(123)
val_x, val_y = gen(2000, SEQ, DIM)
mse_zero = (val_y ** 2).mean().item()
v0 = val_x[:, 0]
c = -(val_y * v0).sum() / (v0 * v0).sum()          # least-squares c for the predictor -c * v0 (fit on this set)
mse_v0 = ((val_y + c * v0) ** 2).mean().item()
print(f"old task: {SEQ} vectors of {DIM} dims, target = the vector farthest from vector 0 (MSE)")
print(f"baselines on 2000 validation sequences: predict 0 -> {mse_zero:.3f}; best -c*v0 (c = {c:.2f}) -> {mse_v0:.3f}")

models = {
    "RMC (paper-faithful)": RelationalRNN(DIM, 4, 32, DIM, num_heads=4),   # old sizes: 4 slots x 32, 4 heads, linear head
    "LSTM (hidden 64)": StandardLSTM(DIM, 64, DIM),
}
for name, model in models.items():
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for step in range(STEPS):
        x, y = gen(BATCH, SEQ, DIM)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        val = nn.functional.mse_loss(model(val_x), val_y).item()
    verdict = "below" if val < mse_v0 else "not below"
    print(f"{name:22s} {count_parameters(model):6,d} params, {STEPS} steps in {time.time() - t0:4.0f} s: "
          f"val MSE {val:.3f} ({verdict} the -c*v0 baseline {mse_v0:.3f})")
