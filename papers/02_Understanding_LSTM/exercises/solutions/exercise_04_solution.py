"""
Solution to Exercise 4: Long-Range Dependencies
===============================================

How many steps can the repo's LSTM carry a piece of information? Every model here trains with
implementation.LSTM's own forward() -> backward() (BPTT) -> update_weights() (Adagrad), one sequence
per update, from a zero state, and is then tested on fresh sequences it never saw.

1. The exercise's task, "remember the first character":
       input   A  n_1 ... n_d  ?      A and the noise n_k are drawn from K symbols; ? = query token K
       target  _  _   ...  _   A      _ = class K, "no answer yet"
   A fresh model is trained at each distance d and seed. Chance is 1/K. The vanilla RNN from
   solution 3 is trained on the same sequences (the exercise's bonus).
2. Delayed XOR (an extra task, --xor): input b1 b2 n_1 ... n_d ?, with noise bits n_k, target
   b1 XOR b2 at the query (chance 50%). One LSTM is trained at one delay, and
   evaluate_memory_capacity() then tests that same model at shorter and longer delays.

Why there is a target at every step: implementation.LSTM puts a softmax and a loss at every position,
so positions before the query get the target "no answer yet". Accuracy is measured at the query only.

Run (1-2 minutes on one CPU core: 2 models x 3 distances x 3 seeds):
    python exercises/solutions/exercise_04_solution.py [--distances 5,10,20,50] [--seeds 0,1,2]
                                                       [--xor] [--plot PREFIX]
Measured with the defaults (3,000 training sequences, K = 10, hidden 32, Adagrad lr 0.05), test
accuracy for seeds 0/1/2: LSTM 100/100/93% at d = 5, 89/100/88% at d = 10, 53/45/49% at d = 20;
vanilla RNN 100/100/100%, 54/99/50% and 90/8/9%. The LSTM degrades steadily with distance; the RNN
swings with the seed. With seed 0 only, d = 50 gave 8% (LSTM) and 12% (RNN): chance, so neither
learns 50 steps within this budget.
"""

import argparse
import os
import sys
import time

import numpy as np

# [PATCH 2026-10-18] import paths from this file's location, so the script runs from any directory;
# solution 3's directory gives the vanilla RNN baseline
SOLUTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
DAY_DIR = os.path.abspath(os.path.join(SOLUTIONS_DIR, '..', '..'))
sys.path.insert(0, DAY_DIR)
sys.path.insert(0, SOLUTIONS_DIR)

from implementation import LSTM  # noqa: E402
from exercise_03_solution import VanillaRNN  # noqa: E402


# ============================================================================
# TASK 1: REMEMBER THE FIRST CHARACTER (the exercise's task)
# [PATCH 2026-10-18] new: the solution did not implement the exercise's task
# ============================================================================

def generate_memory_task(distance, num_samples=1000, vocab_size=10, rng=None):
    """
    "Remember the first character": [A, noise x distance, ?] -> A.

    Returns X (num_samples, distance + 2) int array and y (num_samples,): the first symbol.
    Symbols are 0..vocab_size-1; the query token is vocab_size, so the model needs vocab_size + 1 inputs.
    """
    rng = rng if rng is not None else np.random
    first = rng.randint(0, vocab_size, size=num_samples)                    # (N,) the symbol to remember
    noise = rng.randint(0, vocab_size, size=(num_samples, distance))        # (N, d) distractors
    query = np.full((num_samples, 1), vocab_size)                           # (N, 1) filled with token K
    X = np.hstack([first[:, None], noise, query])   # first[:, None] is (N, 1); hstack -> (N, d + 2)
    return X, first


def step_targets(sequence_length, answer, no_answer):
    """Per-step targets for implementation.LSTM: 'no answer yet' everywhere except the query (last) step."""
    return [no_answer] * (sequence_length - 1) + [int(answer)]


def predict_at_query(model, inputs):
    """Forward pass from a zero state; returns the arg-max class at the last (query) step."""
    H = model.hidden_size
    dummy = [0] * len(inputs)                     # forward() also computes a loss; it is not used here
    _, _, _, cache = model.forward(list(inputs), dummy, np.zeros((H, 1)), np.zeros((H, 1)))
    return int(np.argmax(cache['ps'][len(inputs) - 1]))


def train_on_distance(distance, vocab_size=10, hidden_size=32, num_iterations=3000, num_test=300,
                      learning_rate=0.05, model_cls=LSTM, seed=0, verbose=True, return_model=False):
    """
    Train a fresh model on the first-character task at one distance; return test accuracy in percent
    (and the trained model if return_model).

    num_iterations sequences, one Adagrad update each; num_test fresh sequences for the test.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)                                                    # initial weights
    X_train, y_train = generate_memory_task(distance, num_iterations, vocab_size, rng)
    X_test, y_test = generate_memory_task(distance, num_test, vocab_size, rng)

    model = model_cls(vocab_size + 1, hidden_size, distance + 2, learning_rate=learning_rate)
    H = hidden_size
    start = time.time()
    for x, y in zip(X_train, y_train):
        targets = step_targets(len(x), y, no_answer=vocab_size)
        loss, _, _, cache = model.forward(x.tolist(), targets, np.zeros((H, 1)), np.zeros((H, 1)))
        grads = model.backward(x.tolist(), targets, cache)                  # BPTT over d + 2 steps
        model.update_weights(grads)                                         # Adagrad
    correct = sum(predict_at_query(model, x) == y for x, y in zip(X_test, y_test))
    accuracy = 100.0 * correct / num_test
    if verbose:
        print(f"  {model_cls.__name__:10s} seed {seed} distance {distance:3d}: test accuracy "
              f"{accuracy:5.1f}% (chance {100 / vocab_size:.0f}%), {time.time() - start:.0f} s")
    return (accuracy, model) if return_model else accuracy


def test_all_distances(distances=(5, 10, 20), model_cls=LSTM, **kwargs):
    """Train and test one fresh model per distance."""
    accuracies = [train_on_distance(d, model_cls=model_cls, **kwargs) for d in distances]
    return list(distances), accuracies


# ============================================================================
# TASK 2: COPY TASK (generator only; not trained in this solution)
# ============================================================================

def generate_copy_task(seq_length=10, num_samples=1000, vocab_size=8):
    """
    Generate copy task: remember a sequence and reproduce it later.

    Format: [sequence] [blanks] [delimiter] [sequence]

    Example (seq_length=3):
        Input:  3 1 4 - - - GO ? ? ?
        Target: - - - - - - -  3 1 4
    """
    dataset = []

    for _ in range(num_samples):
        sequence = np.random.randint(0, vocab_size - 2, seq_length)
        blank_token = vocab_size - 2
        delimiter_token = vocab_size - 1

        input_seq = list(sequence)
        input_seq += [blank_token] * seq_length  # Blank period
        input_seq += [delimiter_token]           # Signal to start copying
        input_seq += [blank_token] * seq_length  # Query markers

        target_seq = [blank_token] * (2 * seq_length + 1)
        target_seq += list(sequence)

        dataset.append({'input': input_seq, 'target': target_seq, 'sequence': sequence})

    return dataset


# ============================================================================
# TASK 3: DELAYED XOR
# [PATCH 2026-10-18] the query used to be the same token as the filler, so the model could not
# tell when to answer; it now has its own token, and the filler is random bits (distractors)
# ============================================================================

XOR_QUERY = 2          # input token 2 = "answer now"; output class 2 = "no answer yet"


def generate_delayed_xor_task(delay=20, num_samples=1000, rng=None):
    """
    Delayed XOR: remember two bits across `delay` random distractor bits, answer XOR at the query.

    Input:  b1 b2 n_1 ... n_delay ?        (tokens 0/1 are bits, 2 is the query)
    Target: _  _  _   ...  _      b1^b2    (class 2 = "no answer yet")
    Three tokens, so it needs LSTM(vocab_size=3, ...).
    """
    rng = rng if rng is not None else np.random
    dataset = []
    for _ in range(num_samples):
        bit1, bit2 = int(rng.randint(0, 2)), int(rng.randint(0, 2))
        noise = [int(b) for b in rng.randint(0, 2, delay)]
        input_seq = [bit1, bit2] + noise + [XOR_QUERY]
        dataset.append({
            'input': input_seq,
            'target': step_targets(len(input_seq), bit1 ^ bit2, no_answer=XOR_QUERY),
            'bit1': bit1,
            'bit2': bit2,
            'result': bit1 ^ bit2,
        })
    return dataset


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_on_memory_task(lstm, dataset, epochs=1, learning_rate=0.05):
    """
    Train an implementation.LSTM on a list of {'input', 'target'} sequences.

    One forward / backward / Adagrad update per sequence, from a zero state. Returns per-epoch
    (mean loss per step, accuracy at the last step measured before each update).
    """
    # [PATCH 2026-10-18] the old loop never called backward() or updated a weight
    lstm.learning_rate = learning_rate           # LSTM.update_weights() reads self.learning_rate
    H = lstm.hidden_size
    losses, accuracies = [], []
    for epoch in range(epochs):
        total_loss, total_steps, correct = 0.0, 0, 0
        for k in np.random.permutation(len(dataset)):                       # shuffled order
            inputs, targets = dataset[k]['input'], dataset[k]['target']
            loss, _, _, cache = lstm.forward(inputs, targets, np.zeros((H, 1)), np.zeros((H, 1)))
            correct += int(np.argmax(cache['ps'][len(inputs) - 1]) == targets[-1])
            grads = lstm.backward(inputs, targets, cache)
            lstm.update_weights(grads)
            total_loss += loss
            total_steps += len(inputs)
        losses.append(total_loss / total_steps)
        accuracies.append(correct / len(dataset))
    return losses, accuracies


def accuracy_on(model, dataset):
    """Fraction of sequences whose arg-max at the query step equals the target there."""
    return float(np.mean([predict_at_query(model, s['input']) == s['target'][-1] for s in dataset]))


def evaluate_memory_capacity(lstm, max_delay=50, step=5, num_samples=100, seed=1):
    """
    Test one trained delayed-XOR model at delays step, 2*step, ..., max_delay.

    Returns [{'delay': d, 'accuracy': fraction correct}], measured on fresh sequences.
    """
    # [PATCH 2026-10-18] the old version returned np.random.randint(0, 2) instead of a prediction
    rng = np.random.RandomState(seed)
    results = []
    for delay in range(step, max_delay + 1, step):
        test_set = generate_delayed_xor_task(delay=delay, num_samples=num_samples, rng=rng)
        results.append({'delay': delay, 'accuracy': accuracy_on(lstm, test_set)})
    return results


# ============================================================================
# CELL STATE ANALYSIS
# ============================================================================

def analyze_cell_state_evolution(lstm, sample):
    """
    Run one sequence through the LSTM and return its cell and hidden states at every step.

    Same equations as implementation.LSTM.forward (gate order f, i, C~, o over [h; x]).
    """
    inputs = sample['input']
    h = np.zeros((lstm.hidden_size, 1))
    c = np.zeros((lstm.hidden_size, 1))
    cell_states, hidden_states = [], []
    for inp in inputs:
        x = np.zeros((lstm.vocab_size, 1))
        x[inp] = 1
        combined = np.vstack([h, x])                                        # (H+V, 1) = [h_{t-1}; x_t]
        f = sigmoid(np.dot(lstm.Wf, combined) + lstm.bf)
        i = sigmoid(np.dot(lstm.Wi, combined) + lstm.bi)
        c_tilde = np.tanh(np.dot(lstm.Wc, combined) + lstm.bc)
        c = f * c + i * c_tilde
        o = sigmoid(np.dot(lstm.Wo, combined) + lstm.bo)
        h = o * np.tanh(c)
        cell_states.append(c.copy())
        hidden_states.append(h.copy())
    return cell_states, hidden_states


def forget_gates_during_delay(lstm, sequences, first_noise_step):
    """
    Per-unit mean forget gate over the noise steps (first_noise_step .. last step - 1) of sequences.

    Returns an (H,) array. Units with values near 1 keep their cell content across the delay
    (the "highway"); along that path the gradient is scaled by the product of these values.
    """
    # [PATCH 2026-10-18] new: measures which units learned to hold memory
    H = lstm.hidden_size
    per_sequence = []
    for inputs in sequences:
        inputs = list(inputs)
        _, _, _, cache = lstm.forward(inputs, [0] * len(inputs), np.zeros((H, 1)), np.zeros((H, 1)))
        noise_steps = range(first_noise_step, len(inputs) - 1)
        # each cache['fs'][t] is (H, 1); ravel -> (H,); mean over steps (axis 0) -> (H,)
        per_sequence.append(np.mean([cache['fs'][t].ravel() for t in noise_steps], axis=0))
    return np.mean(per_sequence, axis=0)            # average over sequences -> (H,)


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


# ============================================================================
# VISUALIZATION (only with --plot)
# ============================================================================

def visualize_memory_capacity(results, save_path='memory_capacity.png', trained_delay=None):
    """Plot delayed-XOR accuracy vs test delay for one trained model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    delays = [r['delay'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delays, accuracies, 'o-', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', label='chance')
    if trained_delay is not None:
        ax.axvline(trained_delay, color='gray', linestyle=':', label='trained delay')
    ax.set_xlabel('Test delay (distractor bits)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1.05])
    ax.set_title('Delayed XOR: one model tested at other delays')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    return fig


def plot_distance_sweep(distances, lstm_acc, rnn_acc, vocab_size, save_path='memory_distance.png'):
    """Plot first-character accuracy vs distance for the LSTM and the vanilla RNN."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, lstm_acc, 'o-', label='LSTM')
    ax.plot(distances, rnn_acc, 's-', label='vanilla RNN')
    ax.axhline(100 / vocab_size, color='red', linestyle='--', label='chance')
    ax.set_xscale('log')
    ax.set_xlabel('Distance (noise symbols between the first character and the query)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Remember the first character')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # [PATCH 2026-10-18] runs the experiments (it used to print expected findings only)
    parser = argparse.ArgumentParser(description='Exercise 4 solution: long-range memory tests')
    parser.add_argument('--distances', default='5,10,20',
                        help='first-character distances (the exercise also lists 50 and 100; each costs '
                             'about 30 s and 60 s per model and seed)')
    parser.add_argument('--seeds', default='0,1,2', help='seeds per setting (default 0,1,2)')
    parser.add_argument('--train-seqs', type=int, default=3000, help='training sequences per model')
    parser.add_argument('--xor', action='store_true',
                        help='also run the delayed-XOR experiment (about 30 s more)')
    parser.add_argument('--xor-delay', type=int, default=5)
    parser.add_argument('--plot', metavar='PREFIX', help='save figures as PREFIX_*.png')
    args = parser.parse_args()
    distances = [int(d) for d in args.distances.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    K, H, LR = 10, 32, 0.05
    t0 = time.time()

    print("=" * 74)
    print("1. REMEMBER THE FIRST CHARACTER")
    print("=" * 74)
    print(f"K = {K} symbols (chance {100 / K:.0f}%), hidden {H}, Adagrad lr {LR}, {args.train_seqs} "
          f"training sequences per model, 300 test sequences, seeds {seeds}")
    acc = {}                                          # (model name, distance) -> [accuracy per seed]
    probe = None                                      # an LSTM to look inside afterwards
    for cls in (LSTM, VanillaRNN):
        for d in distances:
            for seed in seeds:
                a, model = train_on_distance(d, vocab_size=K, hidden_size=H, num_iterations=args.train_seqs,
                                             learning_rate=LR, model_cls=cls, seed=seed, return_model=True)
                acc.setdefault((cls.__name__, d), []).append(a)
                if cls is LSTM and d == distances[-1] and seed == seeds[0]:
                    probe = model

    print(f"\n{'distance':>8s} | {'LSTM: per seed':>20s} {'mean':>6s} | {'vanilla RNN: per seed':>22s} "
          f"{'mean':>6s}")
    for d in distances:
        row = []
        for name in ('LSTM', 'VanillaRNN'):
            a = acc[(name, d)]
            row.append(f"{' '.join(f'{x:5.1f}' for x in a):>20s} {np.mean(a):6.1f}")
        print(f"{d:8d} | {row[0]} | {row[1]:>29s}")

    d = distances[-1]
    X_probe, _ = generate_memory_task(d, 200, K, np.random.RandomState(99))
    f_units = np.sort(forget_gates_during_delay(probe, X_probe, first_noise_step=1))[::-1]  # descending
    print(f"\nInside the seed-{seeds[0]} LSTM at distance {d}: mean forget gate per unit over the noise "
          f"steps,\n  top 3 units {np.round(f_units[:3], 3)}, median {np.median(f_units):.3f} "
          f"(at initialisation about sigma(1) = 0.731)")
    print(f"  -> along the cell path, {d} steps scale the gradient by about {f_units[0] ** d:.2g} for the "
          f"most persistent unit and {np.median(f_units) ** d:.1g} for the median unit")

    if args.xor:
        print("\n" + "=" * 74)
        print(f"2. DELAYED XOR: trained at delay {args.xor_delay}, tested at other delays (chance 50%)")
        print("=" * 74)
        print("(below 50% means the model often answers 'no answer yet' at the query)")
        for seed in seeds:
            np.random.seed(seed)
            xor_lstm = LSTM(3, H, args.xor_delay + 3, learning_rate=LR)
            train_set = generate_delayed_xor_task(args.xor_delay, args.train_seqs,
                                                  rng=np.random.RandomState(seed))
            _, train_acc = train_on_memory_task(xor_lstm, train_set, epochs=1, learning_rate=LR)
            capacity = evaluate_memory_capacity(xor_lstm, max_delay=40, step=5, num_samples=200)
            print(f"  seed {seed}: accuracy during training {100 * train_acc[0]:4.1f}%; test: " +
                  "  ".join(f"d={r['delay']}: {100 * r['accuracy']:.0f}%" for r in capacity))
            if args.plot and seed == seeds[0]:
                visualize_memory_capacity(capacity, f"{args.plot}_xor_capacity.png", args.xor_delay)

    if args.plot:
        plot_distance_sweep(distances, [np.mean(acc[('LSTM', d)]) for d in distances],
                            [np.mean(acc[('VanillaRNN', d)]) for d in distances], K,
                            f"{args.plot}_distance.png")
    print(f"\nTotal time {time.time() - t0:.0f} s.")
    print("Caveats: a fixed budget of updates and a few seeds. A model at chance has not learned the task")
    print("within this budget; that does not prove it cannot (more updates, or a larger forget-gate bias")
    print("as in the Day 2 worked example, push the limit further).")
