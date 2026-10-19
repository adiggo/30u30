"""Day 17 worked example: one NTM addressing step (Eqs 5-9) by hand in NumPy, cross-checked against the
(patched) implementation's NTMHead._addressing by forcing its parameter layer to emit the same k, beta, g, s, gamma.
Usage: OMP_NUM_THREADS=1 python validation/day17_worked_example.py [day_dir]"""
import os, sys, numpy as np, torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "papers", "17_neural_turing_machines")
sys.path.insert(0, DAY)
from implementation import NTM
Mem = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [0., -1.]])   # N=5 slots, M=2
k, beta, g, s, gamma = np.array([1., 0.]), 5.0, 0.9, np.array([0.0, 0.1, 0.9]), 3.0   # s over shifts (-1, 0, +1)
prev = np.eye(5)[3]                                                     # previous focus on slot 3
K = Mem @ k / (np.linalg.norm(Mem, axis=1) * np.linalg.norm(k))         # Eq 6: cosine of k with each row -> (5,)
wc = np.exp(beta * K) / np.exp(beta * K).sum()                          # Eq 5: softmax over slots
wg = g * wc + (1 - g) * prev                                            # Eq 7: interpolate with previous weighting
wt = s[0] * np.roll(wg, -1) + s[1] * wg + s[2] * np.roll(wg, 1)         # Eq 8: np.roll(wg, 1)[i] = wg[i-1] (shift +1)
w = wt ** gamma / (wt ** gamma).sum()                                   # Eq 9: sharpen
for name, v in [("K", K), ("w_c", wc), ("w_g", wg), ("w_tilde", wt), ("w", w)]:
    print(f"{name:8s}", np.round(v, 4))
ntm = NTM(input_size=3, output_size=2, controller_size=4, N=5, M=2)
head = ntm.read_heads[0]
with torch.no_grad():
    head.fc.weight.zero_()
    raw = torch.tensor([1.0, 0.0,                                     # k
                        float(np.log(np.expm1(beta))),                # softplus^-1(5)
                        float(np.log(g / (1 - g))),                   # logit(0.9)
                        -40.0, float(np.log(0.1)), float(np.log(0.9)),  # softmax -> (0, 0.1, 0.9)
                        float(np.log(np.expm1(gamma - 1)))])          # 1 + softplus(.) = 3
    head.fc.bias.copy_(raw)
    w_impl, _, _ = head._addressing(torch.zeros(1, 4), torch.tensor(prev, dtype=torch.float32)[None],
                                    torch.tensor(Mem, dtype=torch.float32)[None])
print("implementation w", np.round(w_impl[0].numpy(), 4), " max|delta| =", f"{np.abs(w_impl[0].numpy() - w).max():.1e}")
