"""Day 19 reference check: Relation Network (Santoro et al. 2017, arXiv 1706.01427v1).

Paper Eq 1 (Section 2) and its question-conditioned form (Section 4):
    RN(O) = f_phi( sum_{i,j} g_theta(o_i, o_j, q) )
summing over ALL ordered pairs (i, j), self-pairs included: the paper counts "n^2 feedforward passes per
object set" and describes the input as a complete directed graph.

Checks against implementation.py:
  1. RelationNetwork.forward == explicit double loop over the n^2 ordered pairs (float64), with/without q,
     for the sum / mean / max aggregators (the module's own g_theta, f_phi are reused -- they are plain
     nn.Linear/ReLU stacks; what is checked is pairing, conditioning, aggregation and reshapes).
  2. generate_pairs: row i*N + j is exactly [o_i ; o_j ; q]; there are N^2 rows (i = j included).
  3. permutation invariance of the output under shuffling objects (float64 and float32).
  4. feature_map_to_objects == explicit loop over the k x k grid cells.
  5. train_minimal.RelationalDataset labels recomputed independently ('count' and 'furthest').
  6. 'furthest' label vs invariance: the label is an INDEX into the set, so it moves when the set is
     permuted while the RN output cannot -> no permutation-invariant model can beat 1/N on that mode.

Usage: OMP_NUM_THREADS=1 python validation/day19_relation_network.py [path/to/19_relational_reasoning]
"""
import os
import sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "19_relational_reasoning")
sys.path.insert(0, DAY)
from implementation import RelationNetwork, feature_map_to_objects  # noqa: E402
from train_minimal import RelationalDataset  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
ok = True

# ---------------------------------------------------------------- 1. forward vs explicit double loop
worst = 0.0
for agg in ["sum", "mean", "max"]:
    for qdim in [None, 5]:
        B, N, D = 3, 6, 4
        rn = RelationNetwork(object_dim=D, relation_dim=32, output_dim=7, question_dim=qdim,
                             aggregator=agg).double().eval()
        O = torch.randn(B, N, D, dtype=torch.float64)
        q = torch.randn(B, qdim, dtype=torch.float64) if qdim else None
        with torch.no_grad():
            got = rn(O, q)
            ref = []
            for b in range(B):
                rel = []
                for i in range(N):            # all ordered pairs, i = j included
                    for j in range(N):
                        parts = [O[b, i], O[b, j]] + ([q[b]] if qdim else [])
                        rel.append(rn.g_theta(torch.cat(parts)))
                rel = torch.stack(rel)
                pooled = {"sum": rel.sum(0), "mean": rel.mean(0), "max": rel.max(0)[0]}[agg]
                ref.append(rn.f_phi(pooled))
            ref = torch.stack(ref)
        err = (got - ref).abs().max().item()
        worst = max(worst, err)
        print(f"[forward] agg={agg:4s} q={'yes' if qdim else 'no '}  max|delta| vs double loop = {err:.2e}")
ok &= worst < 1e-12

# ---------------------------------------------------------------- 2. pair layout
rn = RelationNetwork(object_dim=3, relation_dim=8, output_dim=2, question_dim=2).double()
O = torch.randn(2, 4, 3, dtype=torch.float64)
q = torch.randn(2, 2, dtype=torch.float64)
P = rn.generate_pairs(O, q)
layout_err = max((P[b, i * 4 + j] - torch.cat([O[b, i], O[b, j], q[b]])).abs().max().item()
                 for b in range(2) for i in range(4) for j in range(4))
print(f"[pairs] shape {tuple(P.shape)} (N^2 = 16 rows incl. 4 self-pairs); row i*N+j == [o_i;o_j;q]: "
      f"max|delta| = {layout_err:.1e}")
ok &= P.shape[1] == 16 and layout_err == 0.0

# ---------------------------------------------------------------- 3. permutation invariance
for dtype, tol in [(torch.float64, 1e-12), (torch.float32, 1e-4)]:
    rn = RelationNetwork(object_dim=8, relation_dim=64, output_dim=5, question_dim=3).to(dtype).eval()
    O = torch.randn(4, 10, 8, dtype=dtype)
    q = torch.randn(4, 3, dtype=dtype)
    with torch.no_grad():
        base = rn(O, q)
        d = max((rn(O[:, torch.randperm(10)], q) - base).abs().max().item() for _ in range(20))
    print(f"[invariance] {str(dtype):13s} max|delta| over 20 random permutations = {d:.2e}")
    ok &= d < tol

# ---------------------------------------------------------------- 4. feature map -> objects
fm = torch.randn(2, 5, 3, 3)
objs = feature_map_to_objects(fm)
fm_err = max((objs[b, r * 3 + c] - fm[b, :, r, c]).abs().max().item()
             for b in range(2) for r in range(3) for c in range(3))
print(f"[feature_map_to_objects] (2,5,3,3) -> {tuple(objs.shape)}; cell (r,c) -> object r*k+c: "
      f"max|delta| = {fm_err:.1e}")
ok &= fm_err == 0.0

# ---------------------------------------------------------------- 5. dataset labels
ds_c = RelationalDataset(mode="count", num_samples=300, num_objects=10)
bad_c = sum(int(t) != int(((p[:, 0] > 0) & (p[:, 1] > 0)).sum()) for p, t in ds_c.samples)
ds_f = RelationalDataset(mode="furthest", num_samples=300, num_objects=10)
bad_f = sum(int(t) != int(np.argmax(np.sqrt((p ** 2).sum(1)))) for p, t in ds_f.samples)
print(f"[labels] count mode: {bad_c}/300 mismatches; furthest mode: {bad_f}/300 mismatches")
ok &= bad_c == 0 and bad_f == 0

# ---------------------------------------------------------------- 6. furthest-mode ceiling
rn = RelationNetwork(object_dim=2, relation_dim=32, output_dim=10).eval()
p, t = ds_f.samples[0]
x = torch.tensor(p).float().unsqueeze(0)
labels, preds = [], []
with torch.no_grad():
    for shift in range(10):
        perm = np.roll(np.arange(10), shift)
        labels.append(int(np.argmax(np.linalg.norm(p[perm], axis=1))))
        preds.append(int(rn(x[:, perm]).argmax()))
print(f"[furthest ceiling] same set under 10 cyclic shifts: labels {labels}; RN argmax {preds} "
      f"-> at most 1 of 10 orderings can be right, so accuracy <= 1/N = 0.10 without position features")

print(f"\nRESULT: {'PASS' if ok else 'FAIL'} (worst forward max|delta| = {worst:.2e})")
