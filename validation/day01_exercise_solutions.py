"""Day 01: call the reference-solution functions that the solutions' __main__ blocks never run,
against the repo's implementation.CharRNN, and the exercise-1 harness with the solution class.

Run from anywhere:  OMP_NUM_THREADS=1 python validation/day01_exercise_solutions.py
"""
import importlib.util
import os
import sys
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '01_Unreasonable_Effectiveness')
sys.path.insert(0, DAY)
os.environ.setdefault('MPLBACKEND', 'Agg')
from implementation import CharRNN, prepare_data  # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DAY, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attempt(label, fn):
    try:
        out = fn()
        print('  PASS  %-60s %s' % (label, out if out is not None else ''))
        return True
    except Exception as e:  # noqa: BLE001
        print('  FAIL  %-60s %s: %s' % (label, type(e).__name__, str(e).splitlines()[0][:110]))
        return False


np.random.seed(0)
text = open(os.path.join(DAY, 'data', 'tiny_shakespeare.txt')).read()
c2i, i2c, data = prepare_data(text)
rnn = CharRNN(len(c2i), hidden_size=16, learning_rate=0.1, seq_length=10)

print('exercise_02_solution:')
s2 = load('s2', 'exercises/solutions/exercise_02_solution.py')


def gen():
    txt = s2.generate_with_temperature(rnn, i2c, 0, 20, temperature=0.8)
    assert isinstance(txt, str) and len(txt) == 20
    return repr(txt[:20])


attempt('generate_with_temperature(repo CharRNN)', gen)

print('exercise_04_solution:')
s4 = load('s4', 'exercises/solutions/exercise_04_solution.py')


def trainlog():
    log = s4.train_with_logging(rnn, text[:300], c2i, epochs=2, seq_length=10, learning_rate=0.01)
    return 'losses=%s' % np.round(log['losses'], 3)


attempt('train_with_logging(repo CharRNN)', trainlog)

print('exercise_05_solution:')
s5 = load('s5', 'exercises/solutions/exercise_05_solution.py')


def tam():
    r, _, _ = s5.train_author_model(text[:200], 'shakespeare', hidden_size=8, epochs=1)
    assert abs(r.learning_rate - 0.1) < 1e-12 or r.learning_rate < 1, \
        'learning_rate=%r (vocab size passed positionally)' % r.learning_rate
    return 'learning_rate=%r' % r.learning_rate


attempt('train_author_model -> sane learning rate', tam)
attempt('generate_author_samples(repo CharRNN)',
        lambda: repr(s5.generate_author_samples(rnn, i2c, num_samples=1, length=15)[0]))
attempt('extract_rnn_features(repo CharRNN) shape == (3H,)',
        lambda: (lambda f: (_ for _ in ()).throw(AssertionError('shape %s' % (f.shape,)))
                 if f.shape != (3 * rnn.hidden_size,) else 'shape %s' % (f.shape,))(
            s5.extract_rnn_features(text[:50], rnn, c2i)))

print('exercise_01 harness with the solution class swapped in:')
e1 = load('e1', 'exercises/exercise_01_build_rnn.py')
s1 = load('s1', 'exercises/solutions/exercise_01_solution.py')
e1.CharRNN = s1.CharRNN
attempt('exercise_01.train_rnn() with solution CharRNN', lambda: e1.train_rnn())
print('  note: exercise skeleton forward(inputs, h_prev) asks for the loss but has no targets;'
      ' the solution uses forward(inputs, targets, h_prev).')
