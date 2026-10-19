"""Day 01: short, time-capped training run of the repo's CharRNN on the REAL Tiny Shakespeare
(1,115,394 chars; karpathy/char-rnn data/tinyshakespeare/input.txt), with a held-out validation
loss and a unigram baseline -- the bundled 1,322-char file cannot show learning.

The data file is NOT in the repo. Fetch it first, e.g.:
  curl -L -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Run:
  OMP_NUM_THREADS=1 python validation/day01_real_data_run.py /path/to/input.txt [seconds]
"""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '01_Unreasonable_Effectiveness')
sys.path.insert(0, DAY)
from implementation import CharRNN, prepare_data  # noqa: E402

path = sys.argv[1]
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
text = open(path, encoding='utf-8').read()
c2i, i2c, data = prepare_data(text)
V = len(c2i)
split = int(0.9 * len(data))
train, val = data[:split], data[split:split + 20000]           # 20k held-out characters

counts = np.bincount(np.array(train), minlength=V) / len(train)  # unigram frequencies over the vocab
H_unigram = -np.sum(counts[counts > 0] * np.log(counts[counts > 0]))


def val_loss(model):
    """Mean -log p(next char) over the held-out text, carrying the hidden state across chunks."""
    h = np.zeros((model.hidden_size, 1))
    tot, n = 0.0, 0
    for s in range(0, len(val) - 26, 25):
        _, hs, _, ps = model.forward(val[s:s + 25], h)
        tot += model.loss(ps, val[s + 1:s + 26]) * 25
        n += 25
        h = hs[24]
    return tot / n


np.random.seed(0)
m = CharRNN(V, hidden_size=100, learning_rate=1e-1, seq_length=25)   # train_minimal defaults
print('chars=%d vocab=%d | uniform ln V = %.3f nats/char | unigram entropy = %.3f nats/char'
      % (len(data), V, np.log(V), H_unigram))
print('iter      0 | val %.3f nats/char' % val_loss(m))
n, it, smooth = 0, 0, np.log(V)
hprev = np.zeros((100, 1))
t0 = time.time()
while time.time() - t0 < budget:
    if n + 26 >= len(train):
        n, hprev = 0, np.zeros((100, 1))
    inp, tgt = train[n:n + 25], train[n + 1:n + 26]
    xs, hs, ys, ps = m.forward(inp, hprev)
    smooth = 0.999 * smooth + 0.001 * m.loss(ps, tgt)
    m.update_parameters(m.backward(xs, hs, ps, tgt))
    hprev = hs[24]
    n += 25
    it += 1
    if it % 5000 == 0:
        print('iter %6d | train(smoothed) %.3f | val %.3f nats/char | %.0fs'
              % (it, smooth, val_loss(m), time.time() - t0))
vl = val_loss(m)
print('final: %d iterations (%.2f passes over %d training chars) in %.0fs | val %.3f nats/char '
      '(%.2f bits/char) vs unigram %.3f' % (it, it * 25 / len(train), len(train), time.time() - t0,
                                            vl, vl / np.log(2), H_unigram))
seed = c2i['\n']
print('sample (T=0.8):\n' + ''.join(i2c[i] for i in m.sample(np.zeros((100, 1)), seed, 300, temperature=0.8)))
