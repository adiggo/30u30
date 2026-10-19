"""Day 21 reference check: MPNN (Gilmer et al. 2017, arXiv 1704.01212v2).

Paper (Section 2, Eqs 1-3; Section 5.1; Section 5.3):
  m_v^{t+1} = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_vw)       edge network: M = A(e_vw) h_w, A: R^{d_e} -> R^{d x d}
  h_v^{t+1} = U_t(h_v^t, m_v^{t+1})                        U = GRU, weights tied across t (Section 5)
  y = R({h_v^T})                                           R = set2set (Vinyals et al. 2015)
Set2set (Vinyals et al. 2015, Section 4.2 "process block"):
  q_t = LSTM(q*_{t-1});  e_i = <m_i, q_t>;  a = softmax(e);  r_t = sum_i a_i m_i;  q*_t = [q_t, r_t];  q*_0 = 0

Checks
  1. T rounds of edge-network message passing + GRU == independent float64 NumPy per-node loop
     (edge MLP and GRU written out from their equations; GRU equations first checked against nn.GRUCell).
  2. Permutation equivariance of node states and invariance of the readout under node relabelling + edge
     reordering (sum and set2set readouts).
  3. Batch independence: a graph's prediction is identical alone or batched with others.
  4. Set2SetReadout == NumPy re-implementation of the Vinyals/PyG ordering (since 2026-10-18; before, the
     code attended before the LSTM step -- the old recurrence is still printed for comparison). The (h_v, x_v)
     projection added at the same time is checked in day21_readout_fix.py.
  5. scatter_softmax == per-group torch.softmax; MatrixMessage einsum == explicit per-type matrices.

Usage: OMP_NUM_THREADS=1 python validation/day21_mpnn.py [path/to/21_Neural_Message_Passing]
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "21_Neural_Message_Passing")
sys.path.insert(0, DAY)
import implementation as I  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
ok = True


def sig(z):
    return 1 / (1 + np.exp(-z))


def np_gru(x, h, cell):
    """PyTorch GRUCell equations (gate order r, z, n):
    r = s(W_ir x + b_ir + W_hr h + b_hr); z = s(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn)); h' = (1 - z) * n + z * h"""
    Wi, Wh = cell.weight_ih.detach().numpy(), cell.weight_hh.detach().numpy()
    bi, bh = cell.bias_ih.detach().numpy(), cell.bias_hh.detach().numpy()
    d = h.shape[-1]
    gi, gh = x @ Wi.T + bi, h @ Wh.T + bh
    r = sig(gi[..., :d] + gh[..., :d]); z = sig(gi[..., d:2 * d] + gh[..., d:2 * d])
    n = np.tanh(gi[..., 2 * d:] + r * gh[..., 2 * d:])
    return (1 - z) * n + z * h


def graphs(n, seed):
    return [I.generate_synthetic_molecule(n_targets=2, seed=seed + i) for i in range(n)]


def capture_nodes(model, batch):
    store = {}
    hook = model.readout.register_forward_hook(lambda m, inp, out: store.update(h=inp[0].detach().clone()))
    with torch.no_grad():
        y = model(batch)
    hook.remove()
    return store["h"], y


def to64(b):
    return I.BatchedGraph(b.node_features.double(), b.edge_index, b.edge_features.double(), b.batch_indices,
                          b.targets.double())


# ---------------------------------------------------------------- 0. GRU equations vs nn.GRUCell
cell = nn.GRUCell(6, 6)
xg, hg = torch.randn(5, 6), torch.randn(5, 6)
e_gru = np.abs(cell(xg, hg).detach().numpy() - np_gru(xg.numpy(), hg.numpy(), cell)).max()
print(f"[GRU] NumPy GRU equations vs nn.GRUCell: max|delta| = {e_gru:.2e}")
ok &= e_gru < 1e-12

# ---------------------------------------------------------------- 1. message passing vs NumPy loop
T, d = 3, 16
model = I.MPNN(node_dim=5, edge_dim=4, hidden_dim=d, output_dim=2, n_messages=T,
               message_type="edge_network", readout_type="set2set").double().eval()
batch = to64(I.batch_graphs(graphs(4, 100)))
h_code, y_code = capture_nodes(model, batch)

X = batch.node_features.numpy(); E = batch.edge_features.numpy()
src, dst = batch.edge_index[0].numpy(), batch.edge_index[1].numpy()
We, be = model.node_encoder.weight.detach().numpy(), model.node_encoder.bias.detach().numpy()
L1, L2 = model.message_fn.nn[0], model.message_fn.nn[2]
W1, b1 = L1.weight.detach().numpy(), L1.bias.detach().numpy()
W2, b2 = L2.weight.detach().numpy(), L2.bias.detach().numpy()
h = X @ We.T + be                                   # h_v^0: (n_nodes, d) learned projection of x_v
for t in range(T):
    m = np.zeros_like(h)
    for v in range(h.shape[0]):                     # Eq 1: sum over in-neighbours w of A(e_wv) h_w
        for k in np.where(dst == v)[0]:
            A = (np.maximum(E[k] @ W1.T + b1, 0) @ W2.T + b2).reshape(d, d)   # edge network -> d x d
            m[v] += A @ h[src[k]]
    h = np_gru(m, h, model.gru)                     # Eq 2: GRU(h_v, m_v), message = input, h = hidden
e_mp = np.abs(h_code.numpy() - h).max()
print(f"[message passing] node states after T={T} rounds vs NumPy per-node loop: max|delta| = {e_mp:.2e}")
ok &= e_mp < 1e-10

# ---------------------------------------------------------------- 2. relabelling
for readout in ["sum", "set2set"]:
    model = I.MPNN(node_dim=5, edge_dim=4, hidden_dim=d, output_dim=2, n_messages=T,
                   message_type="edge_network", readout_type=readout).double().eval()
    batch = to64(I.batch_graphs(graphs(5, 200)))
    h0, y0 = capture_nodes(model, batch)
    n = batch.node_features.shape[0]
    perm = torch.randperm(n)                        # new node i is old node perm[i]
    inv = torch.empty_like(perm); inv[perm] = torch.arange(n)
    eperm = torch.randperm(batch.edge_index.shape[1])   # also shuffle the edge list
    b2_ = I.BatchedGraph(batch.node_features[perm], inv[batch.edge_index][:, eperm],
                         batch.edge_features[eperm], batch.batch_indices[perm], batch.targets)
    h1, y1 = capture_nodes(model, b2_)
    e_eq = (h1 - h0[perm]).abs().max().item()
    e_inv = (y1 - y0).abs().max().item()
    print(f"[relabel, {readout:7s}] node states equivariant: max|delta| = {e_eq:.2e}; "
          f"graph outputs invariant: max|delta| = {e_inv:.2e}")
    ok &= e_eq < 1e-12 and e_inv < 1e-12

# ---------------------------------------------------------------- 3. batch independence
model = I.MPNN(node_dim=5, edge_dim=4, hidden_dim=d, output_dim=2, n_messages=T).double().eval()
gs = graphs(6, 300)
with torch.no_grad():
    yb = model(to64(I.batch_graphs(gs)))
    ys = torch.cat([model(to64(I.batch_graphs([g]))) for g in gs])
e_bi = (yb - ys).abs().max().item()
print(f"[batching] 6 graphs batched vs one at a time: max|delta| = {e_bi:.2e}")
ok &= e_bi < 1e-12

# ---------------------------------------------------------------- 4. set2set
s2s = I.Set2SetReadout(hidden_dim=d, output_dim=3, processing_steps=6).eval()
Hn = torch.randn(11, d)
bidx = torch.tensor([0] * 4 + [1] * 7)
with torch.no_grad():
    y_code = s2s(Hn, bidx).numpy()
lstm = s2s.lstm
Wi, Wh = lstm.weight_ih.detach().numpy(), lstm.weight_hh.detach().numpy()
bsum = (lstm.bias_ih + lstm.bias_hh).detach().numpy()


def np_lstm(x, hh, cc):
    z = x @ Wi.T + hh @ Wh.T + bsum                 # gate order i, f, g, o
    i_, f_, g_, o_ = sig(z[:, :d]), sig(z[:, d:2 * d]), np.tanh(z[:, 2 * d:3 * d]), sig(z[:, 3 * d:])
    cc = f_ * cc + i_ * g_
    return o_ * np.tanh(cc), cc


def attend(q, Hm, b):
    r = np.zeros((2, d))
    for gidx in range(2):
        Hg = Hm[b == gidx]                          # nodes of graph g
        e = Hg @ q[gidx]                            # dot-product scores, one per node
        a = np.exp(e - e.max()); a /= a.sum()       # softmax within the graph
        r[gidx] = a @ Hg                            # attention-weighted sum of node vectors
    return r


def mlp(z):
    l1, l2 = s2s.mlp[0], s2s.mlp[2]
    return np.maximum(z @ l1.weight.detach().numpy().T + l1.bias.detach().numpy(), 0) @ \
        l2.weight.detach().numpy().T + l2.bias.detach().numpy()


Hm, b = Hn.numpy(), bidx.numpy()
hh = np.zeros((2, d)); cc = np.zeros((2, d))
for _ in range(6):                                  # code order: attend with h, then LSTM([h, r])
    r = attend(hh, Hm, b)
    hh, cc = np_lstm(np.concatenate([hh, r], 1), hh, cc)
y_ref_code = mlp(np.concatenate([hh, r], 1))
qs = np.zeros((2, 2 * d)); hh = np.zeros((2, d)); cc = np.zeros((2, d))
for _ in range(6):                                  # Vinyals / PyG order: q = LSTM(q*), attend with q
    hh, cc = np_lstm(qs, hh, cc)
    r = attend(hh, Hm, b)
    qs = np.concatenate([hh, r], 1)
y_ref_vinyals = mlp(qs)
e_s2s = np.abs(y_code - y_ref_code).max()
d_vin = np.abs(y_code - y_ref_vinyals).max()
print(f"[set2set] code vs NumPy of the Vinyals/PyG ordering: max|delta| = {d_vin:.2e}; "
      f"vs the pre-2026-10-18 attend-then-LSTM recurrence: max|delta| = {e_s2s:.2e} (different function)")
ok &= d_vin < 1e-12  # [2026-10-18] the code now follows Vinyals et al.

# ---------------------------------------------------------------- 5. scatter_softmax, MatrixMessage
logits = torch.randn(9); grp = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2])
sm = I.scatter_softmax(logits, grp)
ref = torch.cat([torch.softmax(logits[grp == g], 0) for g in range(3)])
e_sm = (sm - ref).abs().max().item()
mm = I.MatrixMessage(hidden_dim=5, edge_dim=3)
oh = torch.eye(3)[torch.tensor([0, 2, 1, 2])]
hw = torch.randn(4, 5)
with torch.no_grad():
    got = mm(None, hw, oh)
    exp = torch.stack([mm.edge_matrices[k] @ hw[j] for j, k in enumerate([0, 2, 1, 2])])
e_mm = (got - exp).abs().max().item()
print(f"[scatter_softmax] vs per-group torch.softmax: {e_sm:.2e}; [MatrixMessage] vs A_k h_w: {e_mm:.2e}")
ok &= e_sm < 1e-12 and e_mm < 1e-12

print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
