"""Day 21 worked example + baselines.
(a) One message-passing step on a 3-atom graph C1-C2 (single bond), C2=O3 (double bond), d=2, with hand-picked
    edge matrices A(single)=0.5*I, A(double)=[[0,1],[1,0]], update h' = h + m (a GRU in the real model), sum readout;
    and the same after relabelling the atoms.
(b) Chemical-accuracy conversion MAE = error ratio x chemical accuracy (paper Table 2 enn-s2s, Supp. Table 5).
(c) Baselines on the repo's synthetic data (train_minimal defaults: 500 molecules, seed 42, 3 targets, 400/100 split):
    predict-the-train-mean, and least squares on [atom-type counts, #bonds, 1]."""
import os, sys, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "21_Neural_Message_Passing")
sys.path.insert(0, DAY)
A = {"single": 0.5 * np.eye(2), "double": np.array([[0., 1.], [1., 0.]])}
def step(h, edges):
    m = {v: np.zeros(2) for v in h}
    for (w, v, t) in edges:                       # directed edge w -> v with bond type t
        m[v] = m[v] + A[t] @ h[w]                 # Eq 1: sum over neighbours of A(e_vw) h_w
    return {v: h[v] + m[v] for v in h}, m
h = {1: np.array([1., 0.]), 2: np.array([0., 1.]), 3: np.array([1., 1.])}
und = [(1, 2, "single"), (2, 3, "double")]
edges = und + [(b, a, t) for (a, b, t) in und]
h1, m = step(h, edges)
print("messages", {v: m[v].tolist() for v in m}); print("h'", {v: h1[v].tolist() for v in h1})
print("sum readout", sum(h1.values()).tolist())
perm = {1: 3, 2: 2, 3: 1}                         # relabel atoms 1 <-> 3
hp = {perm[v]: h[v] for v in h}; ep = [(perm[a], perm[b], t) for (a, b, t) in edges]
h1p, _ = step(hp, ep)
print("relabelled readout", sum(h1p.values()).tolist(), " node 3 after relabel == old node 1:", h1p[3].tolist())
print("U0 MAE  = 0.45 x 0.043 eV =", round(0.45 * 0.043, 4), "eV;  gap MAE = 1.60 x 0.043 eV =", round(1.60 * 0.043, 4), "eV")
import implementation as I
data = I.generate_dataset(n_molecules=500, n_targets=3, seed=42)
def feats(g):
    counts = g.node_features.numpy().sum(0)                       # atom-type counts (5,)
    return np.concatenate([counts, [g.edge_index.shape[1] / 2, 1.0]])   # + undirected bond count + bias
X = np.stack([feats(g) for g in data]); Y = np.stack([g.target.numpy() for g in data])
Xtr, Xte, Ytr, Yte = X[:400], X[400:], Y[:400], Y[400:]
print("target mean/std (test):", np.round(Yte.mean(), 2), np.round(Yte.std(), 2))
print("baseline MAE, predict train mean:", round(float(np.abs(Yte - Ytr.mean(0)).mean()), 3))
coef, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
print("baseline MAE, least squares on [atom counts, #bonds, 1]:", round(float(np.abs(Yte - Xte @ coef).mean()), 3))
