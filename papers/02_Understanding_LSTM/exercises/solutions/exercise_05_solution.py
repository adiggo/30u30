"""
Solution to Exercise 5: LSTM vs GRU
===================================

A GRU written from scratch next to the repo's LSTM, both trained the same way on the day's text
(data/input.txt; the last 10% held out): one 25-character window per update, BPTT, and the same
Adagrad rule, learning rate, initial-weight scale, windows and number of updates. The comparison
reports what is measured: parameter count, time per update, training loss and held-out loss.

GRU equations (as drawn in Colah's post; biases added):
    z_t = sigma(W_z [h_{t-1}, x_t] + b_z)              update gate
    r_t = sigma(W_r [h_{t-1}, x_t] + b_r)              reset gate
    h~_t = tanh(W_h [r_t * h_{t-1}, x_t] + b_h)        candidate (reset applied before the matmul)
    h_t = (1 - z_t) * h_{t-1} + z_t * h~_t
Two gates plus a candidate, versus the LSTM's three gates plus a candidate, and no separate cell state.
Cho et al. (2014, Eqs. 5-8) write the same model with z and 1 - z swapped (z keeps the old state);
PyTorch's nn.GRU also applies r after the recurrent matmul, r * (W_hn h + b_hn), a slightly different
model. validation/day02_solution_grads.py checks GRU.backward() against torch.autograd.

Run (about 1 minute on one CPU core: 2 models x 3 seeds x 1200 updates):
    python exercises/solutions/exercise_05_solution.py [--iterations 1200] [--seeds 0,1,2] [--plot PATH]
"""

import argparse
import os
import sys
import time

import numpy as np

# [PATCH 2026-10-18] import path from this file's location, so the script runs from any directory
DAY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, DAY_DIR)

from implementation import LSTM, sigmoid, softmax  # noqa: E402


# ============================================================================
# GRU IMPLEMENTATION
# [PATCH 2026-10-18] interface of the exercise skeleton (backward() stores self.grads,
# update_weights(learning_rate)); Adagrad and init scale as in implementation.LSTM for a fair race
# ============================================================================

class GRU:
    """Gated Recurrent Unit with a softmax output layer, trained like implementation.LSTM."""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        self.input_size = input_size              # one-hot input size (the vocabulary)
        self.vocab_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        scale = np.sqrt(2.0 / (hidden_size + input_size))   # same init scale as implementation.LSTM
        cols = hidden_size + input_size
        self.Wz = np.random.randn(hidden_size, cols) * scale   # update gate, acts on [h; x]
        self.bz = np.zeros((hidden_size, 1))
        self.Wr = np.random.randn(hidden_size, cols) * scale   # reset gate, acts on [h; x]
        self.br = np.zeros((hidden_size, 1))
        self.Wh = np.random.randn(hidden_size, cols) * scale   # candidate, acts on [r * h; x]
        self.bh = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))
        self.memory = {name: np.zeros_like(p) for name, p in self.params().items()}   # Adagrad sums
        self.cache = None
        self.grads = None

    def params(self):
        return {'Wz': self.Wz, 'bz': self.bz, 'Wr': self.Wr, 'br': self.br,
                'Wh': self.Wh, 'bh': self.bh, 'Why': self.Why, 'by': self.by}

    def forward(self, inputs, targets, h_prev):
        """Returns (summed cross-entropy loss, last hidden state); keeps a cache for backward()."""
        xs, hs, zs, rs, h_tildes, combs, rcombs, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0.0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.input_size, 1))
            xs[t][inputs[t]] = 1                                          # one-hot column (V, 1)
            combs[t] = np.vstack([hs[t - 1], xs[t]])                      # (H+V, 1) = [h_{t-1}; x_t]
            zs[t] = sigmoid(self.Wz @ combs[t] + self.bz)                 # (H, H+V) @ (H+V, 1) -> (H, 1)
            rs[t] = sigmoid(self.Wr @ combs[t] + self.br)
            rcombs[t] = np.vstack([rs[t] * hs[t - 1], xs[t]])             # (H+V, 1) = [r * h_{t-1}; x_t]
            h_tildes[t] = np.tanh(self.Wh @ rcombs[t] + self.bh)
            hs[t] = (1 - zs[t]) * hs[t - 1] + zs[t] * h_tildes[t]         # elementwise interpolation
            ys[t] = self.Why @ hs[t] + self.by                            # (V, 1) logits
            ps[t] = softmax(ys[t])
            loss += -np.log(ps[t][targets[t], 0])
        self.cache = {'xs': xs, 'hs': hs, 'zs': zs, 'rs': rs, 'h_tildes': h_tildes, 'combs': combs,
                      'rcombs': rcombs, 'ys': ys, 'ps': ps, 'targets': list(targets)}
        return loss, hs[len(inputs) - 1]

    def backward(self):
        """BPTT through the last forward(); stores and returns grads keyed like params()."""
        c = self.cache
        H = self.hidden_size
        grads = {name: np.zeros_like(p) for name, p in self.params().items()}
        dh_next = np.zeros((H, 1))
        for t in reversed(range(len(c['targets']))):
            dy = np.copy(c['ps'][t])
            dy[c['targets'][t]] -= 1                                      # softmax + cross-entropy
            grads['Why'] += dy @ c['hs'][t].T                             # (V, 1) @ (1, H) -> (V, H)
            grads['by'] += dy
            dh = self.Why.T @ dy + dh_next                                # (H, 1)

            h_prev, z, r, h_tilde = c['hs'][t - 1], c['zs'][t], c['rs'][t], c['h_tildes'][t]
            # h_t = (1 - z) * h_prev + z * h~
            da_h = dh * z * (1 - h_tilde ** 2)                            # through tanh of the candidate
            da_z = dh * (h_tilde - h_prev) * z * (1 - z)                  # through sigmoid of z
            grads['Wh'] += da_h @ c['rcombs'][t].T                        # (H, 1) @ (1, H+V)
            grads['bh'] += da_h
            d_rcomb = self.Wh.T @ da_h                                    # (H+V, 1): grad of [r*h; x]
            da_r = d_rcomb[:H] * h_prev * r * (1 - r)                     # first H rows hold d(r*h)
            grads['Wz'] += da_z @ c['combs'][t].T
            grads['bz'] += da_z
            grads['Wr'] += da_r @ c['combs'][t].T
            grads['br'] += da_r
            dh_next = (dh * (1 - z)                                       # direct path through (1 - z)
                       + d_rcomb[:H] * r                                  # through the candidate's r*h
                       + (self.Wz.T @ da_z)[:H]                           # through the update gate
                       + (self.Wr.T @ da_r)[:H])                          # through the reset gate
        for g in grads.values():
            np.clip(g, -5, 5, out=g)                                      # same clipping as the LSTM
        self.grads = grads
        return grads

    def update_weights(self, learning_rate=None):
        """Adagrad step with the gradients from backward(), the same rule as LSTM.update_weights."""
        lr = self.learning_rate if learning_rate is None else learning_rate
        for name, p in self.params().items():
            self.memory[name] += self.grads[name] * self.grads[name]
            p -= lr * self.grads[name] / np.sqrt(self.memory[name] + 1e-8)   # in place

    def sample(self, h, seed_idx, n, temperature=1.0):
        """Generate n character indices, one forward() step at a time."""
        idx, out = seed_idx, []
        for _ in range(n):
            _, h = self.forward([idx], [idx], h)                          # target only feeds unused loss
            p = softmax(self.cache['ys'][0] / temperature).ravel()        # (V, 1) -> (V,)
            idx = int(np.random.choice(self.output_size, p=p))
            out.append(idx)
        return out


def count_parameters(model):
    """Number of trainable numbers in an implementation.LSTM or a GRU."""
    # [PATCH 2026-10-18] the old code called a parameter_count() method the repo LSTM does not have
    if isinstance(model, GRU):
        arrays = model.params().values()
    else:
        arrays = [model.Wf, model.Wi, model.Wc, model.Wo, model.Why,
                  model.bf, model.bi, model.bc, model.bo, model.by]
    return int(sum(a.size for a in arrays))


# ============================================================================
# COMPARISON STUDY
# ============================================================================

def _evaluate(model, ids, seq_length):
    """Held-out mean cross-entropy (nats/char): non-overlapping windows from a zero state."""
    total, count = 0.0, 0
    H = model.hidden_size
    for p in range(0, len(ids) - 1, seq_length):
        targets = ids[p + 1:p + seq_length + 1].tolist()
        inputs = ids[p:p + len(targets)].tolist()
        if isinstance(model, GRU):
            loss, _ = model.forward(inputs, targets, np.zeros((H, 1)))
        else:
            loss, _, _, _ = model.forward(inputs, targets, np.zeros((H, 1)), np.zeros((H, 1)))
        total += loss
        count += len(targets)
    return total / count


def compare_lstm_gru(data, char_to_idx, hidden_size=32, seq_length=25,
                     num_iterations=1200, learning_rate=0.05, seed=0, val_fraction=0.1, verbose=True):
    """
    Train an LSTM and a GRU on the same windows with the same optimizer; measure both.

    Returns (results, lstm, gru) with results[name] = {'params', 'losses' (nats/char per update),
    'times' ([seconds]), 'val_loss' (nats/char)} for name in ('LSTM', 'GRU').
    """
    # [PATCH 2026-10-18] the old loop called lstm.backward() without arguments, read gradient keys
    # that do not exist and used plain SGD for one model and Adagrad-free updates for the other
    log = print if verbose else (lambda *a, **k: None)
    vocab_size = len(char_to_idx)
    ids = np.array([char_to_idx[ch] for ch in data])
    n_val = max(seq_length + 1, int(len(ids) * val_fraction))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]           # last n_val characters held out

    np.random.seed(seed)
    lstm = LSTM(vocab_size, hidden_size, seq_length, learning_rate=learning_rate)
    np.random.seed(seed)
    gru = GRU(vocab_size, hidden_size, vocab_size, learning_rate=learning_rate)

    results = {}
    for name, model in (('LSTM', lstm), ('GRU', gru)):
        window_rng = np.random.RandomState(seed)               # same windows for both models
        losses = []
        start = time.time()
        for iteration in range(num_iterations):
            p = window_rng.randint(0, len(train_ids) - seq_length)
            inputs = train_ids[p:p + seq_length].tolist()
            targets = train_ids[p + 1:p + seq_length + 1].tolist()
            h0 = np.zeros((hidden_size, 1))
            if name == 'LSTM':
                loss, _, _, cache = model.forward(inputs, targets, h0, np.zeros((hidden_size, 1)))
                model.update_weights(model.backward(inputs, targets, cache))   # BPTT + Adagrad
            else:
                loss, _ = model.forward(inputs, targets, h0)
                model.backward()                                                # BPTT
                model.update_weights(learning_rate)                             # Adagrad
            losses.append(loss / seq_length)
        elapsed = time.time() - start
        results[name] = {'params': count_parameters(model), 'losses': losses, 'times': [elapsed],
                         'val_loss': _evaluate(model, val_ids, seq_length)}
        log(f"  seed {seed} {name:4s}: train loss (last 100) {np.mean(losses[-100:]):.3f}, "
            f"held-out {results[name]['val_loss']:.3f} nats/char, "
            f"{1000 * elapsed / num_iterations:.2f} ms/update")
    return results, lstm, gru


def analyze_comparison(results):
    """Print the measured comparison; `results` is one compare_lstm_gru() result or a list (seeds)."""
    # [PATCH 2026-10-18] measured numbers only (the old summary asserted '25-30% fewer parameters,
    # 10-20% faster' without measuring)
    runs = results if isinstance(results, list) else [results]
    lstm_p, gru_p = runs[0]['LSTM']['params'], runs[0]['GRU']['params']
    print("\n" + "=" * 70)
    print(f"LSTM vs GRU over {len(runs)} seed(s)")
    print("=" * 70)
    print(f"parameters: LSTM {lstm_p:,}, GRU {gru_p:,} (GRU has {100 * (1 - gru_p / lstm_p):.1f}% fewer: "
          f"3 instead of 4 matrices acting on [h; x])")
    for name in ('LSTM', 'GRU'):
        val = [r[name]['val_loss'] for r in runs]
        ms = [1000 * r[name]['times'][0] / len(r[name]['losses']) for r in runs]
        last = [np.mean(r[name]['losses'][-100:]) for r in runs]
        print(f"  {name:4s} held-out {' '.join(f'{v:.3f}' for v in val)} (mean {np.mean(val):.3f}), "
              f"train last 100 {np.mean(last):.3f}, {np.mean(ms):.2f} ms/update")
    d = np.array([r['GRU']['val_loss'] - r['LSTM']['val_loss'] for r in runs])   # per seed
    verdict = ('GRU better in every seed' if d.max() < 0 else
               'LSTM better in every seed' if d.min() > 0 else 'mixed: within seed noise')
    speed = (np.mean([r['LSTM']['times'][0] for r in runs]) /
             np.mean([r['GRU']['times'][0] for r in runs]))
    print(f"held-out GRU - LSTM: mean {d.mean():+.3f} nats/char (min {d.min():+.3f}, max {d.max():+.3f})"
          f": {verdict}")
    print(f"time per update: LSTM / GRU = {speed:.2f} (at batch size 1 NumPy time is mostly per-operation "
          f"overhead, so this tracks the number of operations more than the parameter count)")
    return d


def visualize_comparison(results, save_path='lstm_vs_gru.png'):
    """Training curves and the measured bars for one compare_lstm_gru() result."""
    # [PATCH 2026-10-18] the old figure had made-up parameter bars and a fixed 'winner' table
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name in ('LSTM', 'GRU'):
        smoothed = np.convolve(results[name]['losses'], np.ones(50) / 50, mode='valid')  # moving average
        axes[0].plot(smoothed, label=name, linewidth=2)
    axes[0].set_xlabel('Update')
    axes[0].set_ylabel('Training loss (nats/char, smoothed)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    names = ['LSTM', 'GRU']
    axes[1].bar(names, [results[n]['params'] for n in names])
    axes[1].set_title('Parameters')
    axes[2].bar(names, [1000 * results[n]['times'][0] / len(results[n]['losses']) for n in names])
    axes[2].set_title('ms per update')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # [PATCH 2026-10-18] runs the comparison (it used to print general claims only)
    parser = argparse.ArgumentParser(description='Exercise 5 solution: LSTM vs GRU')
    parser.add_argument('--iterations', type=int, default=1200, help='updates per model and seed')
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.05, help='Adagrad learning rate for both models')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--plot', metavar='PATH', help="also save a figure of the first seed's run")
    args = parser.parse_args()

    with open(os.path.join(DAY_DIR, 'data', 'input.txt'), encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    t0 = time.time()
    runs, models = [], None
    for k, seed in enumerate(int(s) for s in args.seeds.split(',')):
        results, lstm, gru = compare_lstm_gru(text, char_to_idx, hidden_size=args.hidden_size,
                                              num_iterations=args.iterations, learning_rate=args.lr,
                                              seed=seed)
        runs.append(results)
        if k == 0:
            models = (lstm, gru)
    analyze_comparison(runs)

    np.random.seed(1)
    H = args.hidden_size
    lstm_text = ''.join(idx_to_char[i] for i in models[0].sample(np.zeros((H, 1)), np.zeros((H, 1)),
                                                                  char_to_idx['T'], 80, temperature=0.7))
    np.random.seed(1)
    gru_text = ''.join(idx_to_char[i] for i in models[1].sample(np.zeros((H, 1)), char_to_idx['T'], 80,
                                                                 temperature=0.7))
    print(f"\nSamples from the first seed (temperature 0.7):\n  LSTM | {lstm_text!r}\n  GRU  | {gru_text!r}")
    if args.plot:
        visualize_comparison(runs[0], args.plot)
    print(f"\nTotal time {time.time() - t0:.0f} s.")
    print("Caveat: a 4k-character corpus and about 1k updates per model compare early learning, not")
    print("converged models. Chung et al. (2014) found the two comparable; Jozefowicz et al. (2015)")
    print("found that a forget-gate bias of 1, as in implementation.LSTM, closes most of the LSTM-GRU gap.")
