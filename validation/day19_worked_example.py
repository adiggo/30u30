"""Day 19 worked example: hand-written relation g(o_i, o_j) = [|o_i - o_j|, 1] summed / averaged over pairs."""
import itertools, numpy as np
def rn(objs, agg="sum", diag=True):
    rel = [[abs(a - b), 1.0] for (i, a), (j, b) in itertools.product(enumerate(objs), repeat=2) if diag or i != j]
    rel = np.array(rel)
    return rel.sum(0) if agg == "sum" else rel.mean(0)
print("sum (1,4,6)        ", rn([1, 4, 6]))
print("mean (1,4,6)       ", np.round(rn([1, 4, 6], "mean"), 3))
print("off-diagonal sum   ", rn([1, 4, 6], diag=False))
print("shuffled (6,1,4)   ", rn([6, 1, 4]))
print("sum (1,4,4,6)      ", rn([1, 4, 4, 6]))
print("mean (1,4,4,6)     ", np.round(rn([1, 4, 4, 6], "mean"), 3))
print("f: mean distinct-pair distance = 20/(n^2-n) with n=sqrt(9):", 20 / (9 - 3))
