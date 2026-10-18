"""Day 04 exercises: run the exercise tests against the reference solutions.

- Exercise 1 has a test function (test_gaussian_sampling); swap in the solution functions and run
  it, plus the asserts that are left commented in the exercise.
- Exercises 2-5 import implementation.BayesianLinearNetwork (absent before the 2026-10-18 patch).
  Check they import, and drive the class exactly the way their TODO hints do.
"""
import importlib.util
import io
import os
import sys
import contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '04_Minimizing_Description_Length')
EX = os.path.join(DAY, 'Exercises')
sys.path.insert(0, DAY)
sys.path.insert(0, EX)


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EX, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


np.random.seed(0)
sol = load('solutions')
results = {}

# ---- Exercise 1: swap solution functions into the exercise module and run its test
ex1 = load('exercise_01_reparameterization')
ex1.softplus = sol.softplus_solution
ex1.sample_gaussian = sol.sample_gaussian_solution
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ex1.test_gaussian_sampling()
samples, sigma = ex1.sample_gaussian(5.0, 0.0, 100_000)
ok1 = abs(samples.mean() - 5.0) < 0.05 and abs(samples.std() - sigma) < 0.05 and abs(sigma - np.log(2)) < 1e-12
results['ex1 (test + commented asserts, solution swapped in)'] = ok1

# ---- Exercises 2-5: import (ImportError before the patch) and exercise the hinted API
for name in ('exercise_02_gap_experiment', 'exercise_03_beta_parameter',
             'exercise_04_monte_carlo', 'exercise_05_advanced_mdl'):
    try:
        load(name)
        results[f'{name} imports'] = True
    except Exception as e:  # noqa: BLE001
        results[f'{name} imports'] = False
        print(name, type(e).__name__, e)

from implementation import BayesianLinearNetwork  # noqa: E402
X_train = np.random.uniform(0, 4 * np.pi, 200)[:, np.newaxis]     # (200,) -> (200, 1)
y_train = np.sin(X_train.flatten()) + np.random.randn(200) * 0.15  # (200,)
# The hints use lr=0.01; with two hidden layers and unscaled x in [0, 4*pi] plain SGD diverges
# (initial loss ~10^2), so record that and test the class at lr=0.001.
with np.errstate(all='ignore'):
    hint = BayesianLinearNetwork(1, [32, 16], 1, kl_weight=0.01).train(X_train, y_train, epochs=300, lr=0.01)
print(f"hint settings (lr=0.01, hidden [32,16]): loss {hint[0]:.3g} -> {hint[-1]:.3g}  (diverges)")
model = BayesianLinearNetwork(input_dim=1, hidden_dims=[32, 16], output_dim=1, kl_weight=0.01)
losses = model.train(X_train, y_train, epochs=300, lr=0.001, verbose=False)
X_test = np.linspace(0, 4 * np.pi, 500)[:, np.newaxis]            # (500, 1)
y_test = np.sin(X_test.flatten())                                  # (500,)
preds = np.stack([model.forward(X_test).flatten() for _ in range(50)], axis=1)  # (500, 50)
kl = model.compute_kl_divergence()
results['adapter: loss decreases, shapes and KL finite'] = (
    losses[-1] < losses[0] and preds.shape == (500, 50) and np.isfinite(kl))
print(f"adapter: loss {losses[0]:.3f} -> {losses[-1]:.3f}, KL={kl:.1f}, MC preds {preds.shape}")

# The exercise-5 hint computes np.mean((predictions - y_test)**2) with predictions (500,1) and
# y_test (500,): broadcasting silently produces a (500, 500) matrix.
p = model.forward(X_test)
print("ex5 hint broadcasting shape:", (p - y_test).shape, "-> mean over 250000 entries, not 500")

for k, v in results.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
print("PASS" if all(results.values()) else "FAIL")
