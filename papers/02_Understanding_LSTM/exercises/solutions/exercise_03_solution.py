"""
Solution to Exercise 3: Ablation Study
======================================

Trains five recurrent language models on the same characters and compares what they measure:

    1. Full LSTM        implementation.LSTM, unchanged
    2. Vanilla RNN      h_t = tanh(W_h [h_{t-1}, x_t] + b_h): no gates, no cell state
    3. No Forget Gate   f_t = 1 always: C_t = C_{t-1} + i_t * C~_t
    4. No Input Gate    i_t = 1 always: C_t = f_t * C_{t-1} + C~_t
    5. No Output Gate   o_t = 1 always: h_t = tanh(C_t)

Every model trains with the same calls as train_minimal.py, forward() -> backward() (BPTT) ->
update_weights() (Adagrad), but on random 25-character windows from a zero state (as the exercise
asks), with the same seed, initial weights (the four LSTMs), learning rate, windows and number of
updates. The text is the day's data/input.txt (4,264 characters); the last 10% is held out.

Why the ablated LSTMs can reuse LSTM.backward() unchanged: backward() differentiates each gate through
sigma'(a) = g * (1 - g), computed from the stored gate value g. A gate stored as the constant 1 gives
exactly 0 there (its unused weights get zero gradient), and every other formula is already right for a
constant gate (e.g. dC_{t-1} = dC_t * f_t becomes dC_t when f_t = 1). validation/day02_solution_grads.py
checks this, and VanillaRNN.backward(), against torch.autograd and central differences.

Run (about 1 minute on one CPU core: 5 variants x 3 seeds x 1200 updates):
    python exercises/solutions/exercise_03_solution.py [--iterations 1200] [--seeds 0,1,2] [--plot PATH]

Each seed gives every variant the same initial weights and training windows, so the variants are
compared seed by seed; a difference counts only if it has the same sign in every seed. The corpus is
tiny (426 held-out characters), so this compares early learning, not converged models.
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
# MODEL VARIANTS
# [PATCH 2026-10-18] the old classes had forward() only (no backward, no update), so nothing trained.
# The ablated LSTMs now subclass implementation.LSTM and reuse its backward() and Adagrad update;
# the vanilla RNN has its own BPTT, written in the same style as LSTM.backward().
# ============================================================================

class VanillaRNN:
    """
    Vanilla RNN: h_t = tanh(W_h [h_{t-1}, x_t] + b_h), y_t = W_hy h_t + b_y.

    Colah's "single tanh layer". W_h acts on the stacked [h_{t-1}; x_t] like the LSTM's gate matrices;
    that is the same as Day 1's W_hh h + W_xh x with the two matrices placed side by side.

    It has the same interface as implementation.LSTM (forward / backward / update_weights / sample),
    so one training loop serves all five variants. The cell-state arguments are accepted and passed
    through unchanged (as None): a vanilla RNN has no cell state.
    """

    def __init__(self, vocab_size, hidden_size, seq_length, learning_rate=0.001):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        scale = np.sqrt(2.0 / (hidden_size + vocab_size))   # same init scale as implementation.LSTM
        self.Wh = np.random.randn(hidden_size, hidden_size + vocab_size) * scale
        self.bh = np.zeros((hidden_size, 1))
        self.Why = np.random.randn(vocab_size, hidden_size) * scale
        self.by = np.zeros((vocab_size, 1))
        self.memory = {name: np.zeros_like(p) for name, p in self.params().items()}   # Adagrad sums

    def params(self):
        return {'Wh': self.Wh, 'bh': self.bh, 'Why': self.Why, 'by': self.by}

    def forward(self, inputs, targets, h_prev, C_prev=None):
        """Returns (summed cross-entropy loss, last h, None, cache), like LSTM.forward."""
        xs, hs, zs, ys, ps = {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0.0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1                                  # one-hot column (V, 1)
            zs[t] = np.vstack([hs[t - 1], xs[t]])                 # stack rows: (H+V, 1) = [h_{t-1}; x_t]
            hs[t] = np.tanh(self.Wh @ zs[t] + self.bh)            # (H, H+V) @ (H+V, 1) -> (H, 1)
            ys[t] = self.Why @ hs[t] + self.by                    # (V, 1) logits
            ps[t] = softmax(ys[t])
            loss += -np.log(ps[t][targets[t], 0])
        return loss, hs[len(inputs) - 1], C_prev, {'xs': xs, 'hs': hs, 'zs': zs, 'ys': ys, 'ps': ps}

    def backward(self, inputs, targets, cache):
        """BPTT; same structure and +/-5 clipping as LSTM.backward."""
        hs, zs, ps = cache['hs'], cache['zs'], cache['ps']
        dWh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.bh)
        dWhy, dby = np.zeros_like(self.Why), np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1                                   # softmax + cross-entropy: p - onehot
            dWhy += dy @ hs[t].T                                  # (V, 1) @ (1, H) -> (V, H) outer product
            dby += dy
            dh = self.Why.T @ dy + dh_next                        # from the output and from step t+1
            da = (1 - hs[t] ** 2) * dh                            # through tanh: tanh' = 1 - tanh^2
            dWh += da @ zs[t].T                                   # (H, 1) @ (1, H+V) -> (H, H+V)
            dbh += da
            dh_next = (self.Wh.T @ da)[:self.hidden_size]         # first H rows of d[h; x] belong to h_{t-1}
        grads = {'Wh': dWh, 'bh': dbh, 'Why': dWhy, 'by': dby}
        for g in grads.values():
            np.clip(g, -5, 5, out=g)                              # in-place elementwise clip, as in the LSTM
        return grads

    def update_weights(self, grads):
        """Adagrad, the same rule as LSTM.update_weights."""
        for name, p in self.params().items():
            self.memory[name] += grads[name] * grads[name]
            p -= self.learning_rate * grads[name] / np.sqrt(self.memory[name] + 1e-8)   # in place

    def sample(self, h, C, seed_idx, n, temperature=1.0):
        return _sample_by_forward(self, h, C, seed_idx, n, temperature)

    def get_description(self):
        return "Vanilla RNN: h_t = tanh(W_h [h_{t-1}, x_t] + b_h)"


class _GateAblatedLSTM(LSTM):
    """
    implementation.LSTM with one gate replaced by the constant 1 (subclasses set FIXED_GATE).

    forward() is LSTM.forward with that one change and the same cache layout, so the inherited
    backward() and update_weights() apply unchanged (see the module docstring).
    """
    FIXED_GATE = None          # 'f', 'i' or 'o'
    DESCRIPTION = ''

    def _gate(self, name, W, b, z):
        if name == self.FIXED_GATE:
            return np.ones((self.hidden_size, 1))                 # gate removed: always fully open
        return sigmoid(W @ z + b)                                 # (H, H+V) @ (H+V, 1) -> (H, 1)

    def forward(self, inputs, targets, h_prev, C_prev):
        xs, hs, Cs, ys, ps = {}, {}, {}, {}, {}
        fs, i_s, c_cands, os_, zs = {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        Cs[-1] = np.copy(C_prev)
        loss = 0.0
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            zs[t] = np.vstack([hs[t - 1], xs[t]])                 # (H+V, 1) = [h_{t-1}; x_t]
            fs[t] = self._gate('f', self.Wf, self.bf, zs[t])
            i_s[t] = self._gate('i', self.Wi, self.bi, zs[t])
            c_cands[t] = np.tanh(self.Wc @ zs[t] + self.bc)
            Cs[t] = fs[t] * Cs[t - 1] + i_s[t] * c_cands[t]       # elementwise: C_t = f*C_{t-1} + i*C~
            os_[t] = self._gate('o', self.Wo, self.bo, zs[t])
            hs[t] = os_[t] * np.tanh(Cs[t])
            ys[t] = self.Why @ hs[t] + self.by
            ps[t] = softmax(ys[t])
            loss += -np.log(ps[t][targets[t], 0])
        cache = {'xs': xs, 'hs': hs, 'Cs': Cs, 'ys': ys, 'ps': ps,
                 'fs': fs, 'i_s': i_s, 'c_cands': c_cands, 'os': os_, 'zs': zs}
        return loss, hs[len(inputs) - 1], Cs[len(inputs) - 1], cache

    def sample(self, h, C, seed_idx, n, temperature=1.0):
        # LSTM.sample() recomputes all three gates, so it would ignore the ablation.
        return _sample_by_forward(self, h, C, seed_idx, n, temperature)

    def get_description(self):
        return self.DESCRIPTION


class LSTMNoForget(_GateAblatedLSTM):
    """LSTM with the forget gate fixed to 1 (always remember)."""
    FIXED_GATE = 'f'
    DESCRIPTION = "LSTM without forget gate: C_t = C_{t-1} + i_t * C~_t"


class LSTMNoInput(_GateAblatedLSTM):
    """LSTM with the input gate fixed to 1 (always write the full candidate)."""
    FIXED_GATE = 'i'
    DESCRIPTION = "LSTM without input gate: C_t = f_t * C_{t-1} + C~_t"


class LSTMNoOutput(_GateAblatedLSTM):
    """LSTM with the output gate fixed to 1 (always expose the whole cell)."""
    FIXED_GATE = 'o'
    DESCRIPTION = "LSTM without output gate: h_t = tanh(C_t)"


class FullLSTM(LSTM):
    """implementation.LSTM unchanged (only adds a description for the printout)."""

    def get_description(self):
        return "Full LSTM: forget, input and output gates"


def _sample_by_forward(model, h, C, seed_idx, n, temperature):
    """Sample n characters by running model.forward() one step at a time (works for every variant)."""
    idx, out = seed_idx, []
    for _ in range(n):
        _, h, C, cache = model.forward([idx], [idx], h, C)       # the target only feeds the unused loss
        p = softmax(cache['ys'][0] / temperature).ravel()        # (V, 1) -> (V,) for np.random.choice
        idx = int(np.random.choice(model.vocab_size, p=p))
        out.append(idx)
    return out


VARIANTS = {
    'Full LSTM': FullLSTM,
    'Vanilla RNN': VanillaRNN,
    'No Forget Gate': LSTMNoForget,
    'No Input Gate': LSTMNoInput,
    'No Output Gate': LSTMNoOutput,
}


# ============================================================================
# MEASUREMENT HELPERS
# [PATCH 2026-10-18] new: held-out loss and the unigram baseline, so the comparison is measured
# ============================================================================

def evaluate_loss(model, ids, seq_length):
    """Mean cross-entropy (nats/char) on ids, in non-overlapping windows from a zero state, as in training."""
    total, count = 0.0, 0
    H = model.hidden_size
    for p in range(0, len(ids) - 1, seq_length):
        targets = ids[p + 1:p + seq_length + 1].tolist()
        inputs = ids[p:p + len(targets)].tolist()
        loss, _, _, _ = model.forward(inputs, targets, np.zeros((H, 1)), np.zeros((H, 1)))
        total += loss
        count += len(targets)
    return total / count


def unigram_entropy(ids, vocab_size):
    """Cross-entropy (nats/char) of always predicting the character frequencies: the no-context baseline."""
    freq = np.bincount(ids, minlength=vocab_size) / len(ids)    # count of each index -> frequencies
    freq = freq[freq > 0]                                         # boolean mask drops unseen characters
    return float(-(freq * np.log(freq)).sum())




# ============================================================================
# ABLATION STUDY
# ============================================================================

def run_ablation_study(data, char_to_idx, hidden_size=32, seq_length=25,
                       num_iterations=1200, sample_every=400, learning_rate=0.05, seed=0,
                       val_fraction=0.1, sample_length=80, verbose=True):
    """
    Train all five variants with identical settings (one seed) and measure them.

    Returns {variant name: {'losses': training loss per update (nats/char), 'times': [seconds],
    'val_loss': held-out loss (nats/char), 'sample': generated text, 'model': trained model}}
    plus results['_baselines'] = {'uniform': ln V, 'unigram': character-frequency entropy} (nats/char).
    """
    # [PATCH 2026-10-18] the loop now trains (backward + Adagrad update) and measures held-out loss
    log = print if verbose else (lambda *a, **k: None)
    vocab_size = len(char_to_idx)
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    ids = np.array([char_to_idx[ch] for ch in data])
    n_val = max(seq_length + 1, int(len(ids) * val_fraction))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]               # slicing: last n_val characters held out
    baselines = {'uniform': float(np.log(vocab_size)), 'unigram': unigram_entropy(train_ids, vocab_size)}
    log(f"seed {seed}: train {len(train_ids):,} chars, held-out {len(val_ids):,} chars, vocabulary "
        f"{vocab_size}; hidden {hidden_size}, window {seq_length}, {num_iterations} updates, "
        f"Adagrad lr {learning_rate}")

    results = {}
    for name, cls in VARIANTS.items():
        np.random.seed(seed)                     # same initial weights for the four LSTM variants
        model = cls(vocab_size, hidden_size, seq_length, learning_rate=learning_rate)
        window_rng = np.random.RandomState(seed)  # same training windows, in the same order
        log(f"  {name}: {model.get_description()}")

        losses = []
        start_time = time.time()
        for iteration in range(num_iterations):
            p = window_rng.randint(0, len(train_ids) - seq_length)   # random window start
            inputs = train_ids[p:p + seq_length].tolist()
            targets = train_ids[p + 1:p + seq_length + 1].tolist()  # next character at every position
            h0 = np.zeros((hidden_size, 1))
            C0 = np.zeros((hidden_size, 1))

            loss, _, _, cache = model.forward(inputs, targets, h0, C0)
            grads = model.backward(inputs, targets, cache)          # BPTT through the window
            model.update_weights(grads)                              # Adagrad step
            losses.append(loss / seq_length)

            if (iteration + 1) % sample_every == 0:
                log(f"    update {iteration + 1:5d}: train loss (last {sample_every}) "
                    f"{np.mean(losses[-sample_every:]):.3f} nats/char")
        elapsed = time.time() - start_time

        np.random.seed(seed + 1)
        sample_ids = model.sample(np.zeros((hidden_size, 1)), np.zeros((hidden_size, 1)),
                                  char_to_idx.get('T', 0), sample_length, temperature=0.7)
        results[name] = {
            'losses': losses,
            'times': [elapsed],
            'val_loss': evaluate_loss(model, val_ids, seq_length),
            'sample': ''.join(idx_to_char[i] for i in sample_ids),
            'model': model,
        }
        log(f"    held-out loss {results[name]['val_loss']:.3f} nats/char, "
            f"{1000 * elapsed / num_iterations:.1f} ms/update")

    results['_baselines'] = baselines
    return results


def analyze_results(results):
    """
    Print the measured comparison. `results` is one run_ablation_study() result or a list of them
    (one per seed); with several seeds each ablation is compared with the full LSTM seed by seed
    (same initial weights and windows), and the min..max of that difference shows the noise.
    """
    # [PATCH 2026-10-18] measured table (the old version labelled losses 'Fast'/'Slow' by a fixed cutoff)
    runs = results if isinstance(results, list) else [results]
    names = [k for k in runs[0] if not k.startswith('_')]
    base = runs[0]['_baselines']
    val = {n: np.array([r[n]['val_loss'] for r in runs]) for n in names}   # (seeds,) per variant

    print("\n" + "=" * 78)
    print(f"RESULTS over {len(runs)} seed(s), nats per character (lower is better)")
    print("=" * 78)
    print(f"baselines: uniform guess {base['uniform']:.3f}, character frequencies (unigram) "
          f"{base['unigram']:.3f}")
    print(f"\n{'variant':16s} {'held-out (mean)':>15s} {'per seed':>22s} {'train 1st 400':>14s} "
          f"{'train last 100':>15s} {'ms/update':>10s}")
    ranking = sorted(names, key=lambda n: val[n].mean())
    for n in ranking:
        first = np.mean([np.mean(r[n]['losses'][:400]) for r in runs])
        last = np.mean([np.mean(r[n]['losses'][-100:]) for r in runs])
        ms = np.mean([1000 * r[n]['times'][0] / len(r[n]['losses']) for r in runs])
        per_seed = ' '.join(f"{v:.3f}" for v in val[n])
        print(f"{n:16s} {val[n].mean():15.3f} {per_seed:>22s} {first:14.3f} {last:15.3f} {ms:10.1f}")

    print("\nDifference from the full LSTM (held-out, paired by seed):")
    for n in ranking:
        if n == 'Full LSTM':
            continue
        d = val[n] - val['Full LSTM']                                   # elementwise, one value per seed
        verdict = ('worse in every seed' if d.min() > 0 else
                   'better in every seed' if d.max() < 0 else 'mixed: within seed noise')
        print(f"  {n:16s} mean {d.mean():+.3f}  (min {d.min():+.3f}, max {d.max():+.3f})  {verdict}")

    print("\nSamples from the first seed (temperature 0.7, seeded with 'T'):")
    for n in ranking:
        print(f"  {n:16s} | {runs[0][n]['sample']!r}")
    return ranking


def visualize_comparison(results, save_path='ablation_comparison.png'):
    """Training curves, held-out loss and time per update for every variant (one seed's results)."""
    # [PATCH 2026-10-18] the old fourth panel printed fixed 'key findings'; every panel is now data
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    variants = {k: v for k, v in results.items() if not k.startswith('_')}
    names = list(variants)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, r in variants.items():
        smoothed = np.convolve(r['losses'], np.ones(50) / 50, mode='valid')   # 50-update moving average
        axes[0].plot(smoothed, label=name, linewidth=2)
    axes[0].axhline(results['_baselines']['unigram'], color='gray', linestyle='--', label='unigram')
    axes[0].set_xlabel('Update')
    axes[0].set_ylabel('Training loss (nats/char, smoothed)')
    axes[0].set_title('Training loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(names)), [variants[n]['val_loss'] for n in names])
    axes[1].axhline(results['_baselines']['unigram'], color='gray', linestyle='--')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha='right')
    axes[1].set_ylabel('Held-out loss (nats/char)')
    axes[1].set_title('Held-out loss')

    axes[2].bar(range(len(names)),
                [1000 * variants[n]['times'][0] / len(variants[n]['losses']) for n in names])
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=30, ha='right')
    axes[2].set_ylabel('ms per update')
    axes[2].set_title('Cost per update')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # [PATCH 2026-10-18] runs the study (it used to print expected findings without training anything)
    parser = argparse.ArgumentParser(description='Exercise 3 solution: LSTM gate ablation study')
    parser.add_argument('--iterations', type=int, default=1200, help='updates per variant and seed')
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Adagrad learning rate (implementation.LSTM defaults to 0.001, too slow here)')
    parser.add_argument('--seeds', default='0,1,2', help='comma-separated seeds (default 0,1,2)')
    parser.add_argument('--plot', metavar='PATH', help="also save a figure of the first seed's run")
    args = parser.parse_args()

    with open(os.path.join(DAY_DIR, 'data', 'input.txt'), encoding='utf-8') as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    print("=" * 78)
    print("LSTM ABLATION STUDY")
    print("=" * 78)
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(',')]
    runs = [run_ablation_study(text, char_to_idx, hidden_size=args.hidden_size,
                               num_iterations=args.iterations, learning_rate=args.lr, seed=seed,
                               verbose=(k == 0))
            for k, seed in enumerate(seeds)]
    analyze_results(runs)
    if args.plot:
        visualize_comparison(runs[0], args.plot)
    print(f"\nTotal time {time.time() - t0:.0f} s.")
    print("Caveats: a 4k-character corpus, 426 held-out characters and about 1k updates per model, so")
    print("this compares early learning on a tiny dataset, not converged models; trust only differences")
    print("that have the same sign in every seed.")
