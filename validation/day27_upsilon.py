"""Day 27 check: universal intelligence Upsilon (Legg 2008, Def. 4.1.3, p. 82).

    Upsilon(pi) = sum_{mu in E} 2^{-K(mu)} V_mu^pi,   V_mu^pi = E[sum_i r_i] <= 1
over computable reward-summable environments E (total reward bounded by 1, p. 78).  With
sum 2^{-K} <= 1 (Kraft, prefix-free K) this gives Upsilon <= 1.

The repo reports upsilon_normalized = sum_mu w_mu V_mu / sum_mu w_mu (prior renormalised to
the 4-environment suite).  Values > 1 (1.2454, 1.1563 before the fix) can only come from V_mu > 1.
1. ORIGINAL code (git, ORIG_REF = the commit before the fix): show which environment has V > 1.
2. PATCHED code: every V in [0,1]; normalized <= 1; raw <= sum w <= 1.
3. evaluate() vs closed form with scripted optimal agents:
     GridWorld(s): optimal return = 1 - 0.01 * (2(s-1) - 1)  (2(s-1) moves, last one pays +1)
     PatternSequence(p): oracle predicting the next symbol earns 10 * 0.1 = 1.0
"""
import importlib.util
import os
import subprocess
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
# Pre-fix code for the original-vs-patched comparison: the commit the fixes on this branch were
# made on (read with `git show`, no checkout; needs the repo's git history).
ORIG_REF = "195dcf2"

def load(name, src_path=None, src_text=None):
    if src_text is not None:
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False); f.write(src_text); f.close(); src_path = f.name
    spec = importlib.util.spec_from_file_location(name, src_path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

orig = load("msi_orig", src_text=subprocess.run(["git", "-C", REPO, "show", ORIG_REF + ":papers/27_MSI/implementation.py"],
                                                 capture_output=True, text=True, check=True).stdout)
new = load("msi_new", os.path.join(REPO, "papers", "27_MSI", "implementation.py"))

def suite(m):  # same suite as train_minimal.py
    return [m.GridWorld(size=3), m.GridWorld(size=7), m.PatternSequence([1, 0, 1]), m.PatternSequence([1, 2, 3, 4, 5])]

ok = True
for label, m in (("original", orig), ("patched", new)):
    np.random.seed(0)
    res = m.UniversalIntelligenceMeasure(suite(m)).evaluate(m.SimpleRLAgent(), episodes=100)
    w = np.array([d["weight"] for d in res["details"]]); V = np.array([d["expected_reward"] for d in res["details"]])
    norm = (w * V).sum() / w.sum()
    print(f"{label:8s} RL agent: V = {np.round(V, 3).tolist()}  weights = {w.tolist()}")
    print(f"         raw = {res['upsilon_raw']:.4f} (sum w = {w.sum():.4f}), normalized = {res['upsilon_normalized']:.4f} "
          f"(recomputed {norm:.4f}, |delta| = {abs(norm - res['upsilon_normalized']):.1e})")
    if label == "patched":
        ok &= bool((V >= 0).all() and (V <= 1).all() and res["upsilon_normalized"] <= 1 and res["upsilon_raw"] <= w.sum() <= 1)

class GridOracle:                       # right (3) until x = s-1, then up (0) until y = s-1
    def __init__(self, env): self.env = env
    def act(self, obs): return 3 if obs[0] < self.env.size - 1 else 0
    def observe(self, *a): pass

class PatternOracle:                    # always predicts the true next symbol
    def __init__(self, env): self.env = env
    def act(self, obs): return self.env.pattern[(self.env.idx + 1) % len(self.env.pattern)]
    def observe(self, *a): pass

deltas = []
for env, Agent, closed in ((new.GridWorld(3), GridOracle, 1 - 0.01 * (2 * 2 - 1)),
                           (new.GridWorld(7), GridOracle, 1 - 0.01 * (2 * 6 - 1)),
                           (new.PatternSequence([1, 0, 1]), PatternOracle, 1.0)):
    r = new.UniversalIntelligenceMeasure([env]).evaluate(Agent(env), episodes=3)
    d = r["details"][0]; deltas.append(abs(d["expected_reward"] - closed))
    expect_raw = 2.0 ** -env.complexity * closed
    deltas.append(abs(r["upsilon_raw"] - expect_raw))
    print(f"closed form {type(env).__name__}(K={env.complexity}): V = {d['expected_reward']:.4f} vs {closed:.4f}; "
          f"raw Upsilon = {r['upsilon_raw']:.6f} vs 2^-K V = {expect_raw:.6f}")
print(f"max |delta| (evaluate vs closed form) = {max(deltas):.1e}")
ok &= max(deltas) < 1e-12
print("RESULT:", "PASS" if ok else "FAIL")
