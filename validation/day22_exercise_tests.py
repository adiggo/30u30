"""Day 22: run each exercise's own test function with the reference solution swapped in.

Run:  OMP_NUM_THREADS=1 python validation/day22_exercise_tests.py
"""
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '22_deep_speech_2')
EX = DAY + '/exercises'
sys.path.insert(0, DAY)
os.environ.setdefault('MPLBACKEND', 'Agg')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


plan = [
    ('exercise_01_spectrogram.py', 'solution_01_spectrogram.py', ['compute_spectrogram'], 'test_spectrogram'),
    ('exercise_02_ctc_loss.py', 'solution_02_ctc_loss.py', ['CharEncoder', 'compute_ctc_loss'], 'test_ctc'),
    ('exercise_03_rnn_batchnorm.py', 'solution_03_rnn_batchnorm.py',
     ['SequenceBatchNorm', 'BidirectionalGRULayer'], 'test_rnn_batchnorm'),
    ('exercise_04_sortagrad.py', 'solution_04_sortagrad.py',
     ['sortagrad_sampler', 'create_length_sorted_batches'], 'test_sortagrad'),
    ('exercise_05_end_to_end.py', 'solution_05_end_to_end.py', ['ClippedReLU', 'MiniDeepSpeech2'], 'test_pipeline'),
]
passed = 0
for ex_file, sol_file, names, test in plan:
    ex = load('ex_' + test, os.path.join(EX, ex_file))
    sol = load('sol_' + test, os.path.join(EX, 'solutions', sol_file))
    for n in names:
        setattr(ex, n, getattr(sol, n))
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            getattr(ex, test)()
        passed += 1
        print('  PASS  %s.%s with solution %s' % (ex_file, test, ', '.join(names)))
    except Exception as e:  # noqa: BLE001
        print('  FAIL  %s.%s: %s: %s' % (ex_file, test, type(e).__name__, str(e).splitlines()[0][:100]))
print('%d/%d exercise test functions pass with the reference solutions swapped in' % (passed, len(plan)))
