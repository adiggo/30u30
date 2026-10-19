"""Day 20 worked example: paper Eq. 2 with 2 slots + 1 input row, identity projections, one head."""
import numpy as np
M = np.array([[1., 0.], [0., 1.]]); x = np.array([[2., 0.]])
KV = np.vstack([M, x])                                   # [M; x]: row-wise concat -> (3, 2)
S = M @ KV.T / np.sqrt(2)                                # queries from M only -> (2, 3) scores
P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)   # row softmax
print("scores\n", np.round(S, 4)); print("attention\n", np.round(P, 4)); print("M_tilde\n", np.round(P @ KV, 4))
print("forget bias +1: sigma(1) =", round(1 / (1 + np.exp(-1)), 4), " half-life =", round(np.log(0.5) / np.log(1 / (1 + np.exp(-1))), 2), "steps")
