"""Day 02: call the reference-solution functions (not just their print-only __main__ blocks)
against the repo's implementation.LSTM.

Run:  OMP_NUM_THREADS=1 python validation/day02_exercise_solutions.py
"""
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '02_Understanding_LSTM')
sys.path.insert(0, DAY)
os.environ.setdefault('MPLBACKEND', 'Agg')
from implementation import LSTM  # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DAY, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attempt(label, fn):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            out = fn()
        print('  PASS  %-58s %s' % (label, out if out is not None else ''))
    except Exception as e:  # noqa: BLE001
        print('  FAIL  %-58s %s: %s' % (label, type(e).__name__, str(e).splitlines()[0][:100]))


np.random.seed(0)
text = open(os.path.join(DAY, 'data', 'input.txt')).read()
chars = sorted(set(text))
c2i = {c: i for i, c in enumerate(chars)}
lstm = LSTM(len(chars), 16, seq_length=10)

print('exercise_02_solution (gate analysis):')
s2 = load('s2', 'exercises/solutions/exercise_02_solution.py')
attempt('capture_gate_activations(repo LSTM)',
        lambda: 'forget mean=%.3f over %d steps' % (
            np.mean(s2.capture_gate_activations(lstm, text[:60], c2i)['forget']), 60))
attempt('compare_gates_across_contexts(repo LSTM)', lambda: s2.compare_gates_across_contexts(lstm, c2i))

print('exercise_03_solution (ablation):')
s3 = load('s3', 'exercises/solutions/exercise_03_solution.py')
attempt('run_ablation_study(...) -> trains 5 variants?',
        lambda: s3.run_ablation_study(text, c2i, hidden_size=8, seq_length=10, num_iterations=5,
                                      sample_every=5))
src3 = open(os.path.join(DAY, 'exercises/solutions/exercise_03_solution.py')).read()
print('        contains a backward/update call in its training loop:',
      'backward' in src3.split('def run_ablation_study')[1].split('def analyze_results')[0])

print('exercise_04_solution (long-range memory):')
s4 = load('s4', 'exercises/solutions/exercise_04_solution.py')
ds = s4.generate_delayed_xor_task(delay=5, num_samples=4)
attempt('train_on_memory_task(repo LSTM)',
        lambda: s4.train_on_memory_task(LSTM(3, 8, 8), ds, epochs=1))
src4 = open(os.path.join(DAY, 'exercises/solutions/exercise_04_solution.py')).read()
print('        evaluate_memory_capacity uses a random placeholder prediction:',
      'np.random.randint(0, 2)  # Placeholder' in src4)
attempt('analyze_cell_state_evolution(repo LSTM)',
        lambda: '%d cell states' % len(s4.analyze_cell_state_evolution(LSTM(3, 8, 8), ds[0])[0]))

print('exercise_05_solution (LSTM vs GRU):')
s5 = load('s5', 'exercises/solutions/exercise_05_solution.py')
attempt('compare_lstm_gru(...) with repo LSTM',
        lambda: s5.compare_lstm_gru(text, c2i, hidden_size=8, seq_length=10, num_iterations=2))
