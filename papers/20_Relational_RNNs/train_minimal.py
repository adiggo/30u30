"""
Day 20: Relational RNNs - Minimal Training Script (the Nth Farthest task)
=========================================================================

[PATCH 2026-10-18] Rewritten. The old script regressed "the vector farthest from vector 0" with MSE;
neither model beat a fixed -0.39 * v0 predictor. This script trains the paper's task instead:
Santoro et al. (2018), Sec. 4.1 and App. A.1, "What is the n-th farthest vector from vector m?",
answered as a classification over the vectors' labels with softmax cross-entropy.

Usage:
    python train_minimal.py                           # RMC then LSTM, ~3-5 min on one CPU core
    python train_minimal.py --save_plot --visualize   # + accuracy curves and the RMC's attention
    python train_minimal.py --curriculum_steps 0      # the paper's protocol: every n from step 1
    python train_minimal.py --full                    # opt-in: 4x the steps (~15-20 min)
    python train_minimal.py --num_objects 8 --num_features 16 --batch_size 1600 --lr 1e-4 \\
        --curriculum_steps 0 --steps ...              # the paper's setting: hours on a GPU

Paper vs this script (CPU-sized; README "The Task We Run" has the measured numbers):
    paper (App. A.1)                         default here
    8 vectors of 16 dims (40-d inputs)       3 vectors of 2 dims (11-d inputs)     --num_objects, --num_features
    n, m uniform for every sequence          same, after a curriculum: the first    --curriculum_steps
                                             1500 steps ask only for the farthest
                                             vector (n = 0)
    RMC: 1/8/16 memories, 2048 units,        4 memories x 64 units, 4 heads,       --slots, --mem_size, --heads
         1/8/16 heads, 1 block                1 block, 'unit' gating
    4-layer MLP (256, ReLU) on the output    the same MLP for both models          --head_hidden
    LSTM hidden 64 ... 4096                  hidden size matched to the RMC's      --lstm_hidden (0 = match)
                                             total parameter count
    Adam, batch 1600, lr 1e-4, hours         Adam, batch 128, lr 1e-3, 3000 steps  --batch_size, --lr, --steps
    RMC 91%, LSTM and DNC < 30%              see README

Why the curriculum. With n uniform, a model first learns a free 2/K: when n = K-1 (the K-th farthest,
counting from 0) the answer is m itself, which is in the input. Past that point nothing about the vectors
alone helps: averaged over the uniform n, every label other than m is equally likely whatever the
vectors are, so a model must combine n with the distances before its loss can move. Gradient descent
finds that slowly: hours on a GPU in the paper (a public PyTorch reproduction, L0SG/relational-rnn-pytorch,
reports the same "25% barrier"), and here neither model left 2/K within 1.3M training sequences. Asking
only for the farthest vector first removes the cancellation; the RMC then learns the distances, and the
switch to the full question mix takes a few hundred steps. Evaluation always uses the paper's uniform n.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F

from implementation import RelationalRNN, StandardLSTM, count_parameters
from visualization import plot_attention_heatmap, plot_comparison


# [PATCH 2026-10-18] the paper's labelled Nth Farthest task (App. A.1) replaces the MSE regression
def generate_nth_farthest_batch(batch_size, num_objects=8, num_features=16, generator=None, fixed_n=None):
    """A batch of the Nth Farthest task (Sec. 4.1, App. A.1; Sonnet examples/dataset_nth_farthest.py).

    Each sequence shows K = num_objects vectors x ~ U(-1, 1)^D, one per time step. Every vector has
    a label (one-hot; the labels are a random permutation, so a label says nothing about when the
    vector arrives), and every step repeats the question (n, m), also one-hot. The target is the label
    of the n-th farthest vector (Euclidean distance) from the vector labelled m.
    Input per step: (x_t; l_t; n; m) -> D + 3K numbers (40 for the paper's K = 8, D = 16).

    Counting is 0-based here: n = 0 asks for the farthest vector, n = K-1 for the nearest, which is
    vector m itself (distance 0), so those questions can be answered by copying m from the input.
    (Sonnet's dataset ranks nearest-first, so its nth = 0 is m itself: the same task with n -> K-1-n.)
    fixed_n: ask every sequence for this rank (used by the curriculum); None = uniform, as in the paper.

    Returns inputs (B, K, D + 3K) float32 and targets (B,) int64 labels in [0, K).
    """
    B, K, D = batch_size, num_objects, num_features
    vectors = torch.rand(B, K, D, generator=generator) * 2 - 1   # (B, K, D); object k has label k until the shuffle
    n = torch.randint(0, K, (B,), generator=generator)            # rank asked for, 0 = farthest
    if fixed_n is not None:
        n = torch.full_like(n, fixed_n)
    m = torch.randint(0, K, (B,), generator=generator)            # label of the reference vector
    # (B, K, 1, D) - (B, 1, K, D) broadcasts to (B, K, K, D): the difference of every pair of objects
    diff = vectors.unsqueeze(2) - vectors.unsqueeze(1)
    sq_dist = (diff ** 2).sum(-1)                                 # (B, K, K) squared distances (same ranking as distances)
    rows = torch.arange(B)
    dist_to_ref = sq_dist[rows, m]                                # (B, K): row m[b] of sequence b = distances to the reference
    ranking = dist_to_ref.argsort(dim=-1, descending=True)        # (B, K): labels from farthest to nearest (m itself last)
    targets = ranking[rows, n]                                    # (B,): label at rank n[b]

    eye = torch.eye(K)
    labels = eye.expand(B, K, K)                                  # (B, K, K): object k carries one-hot label k
    n_hot = eye[n].unsqueeze(1).expand(B, K, K)                   # (B, K, K): the same n one-hot at every step
    m_hot = eye[m].unsqueeze(1).expand(B, K, K)                   # (B, K, K): the same m one-hot at every step
    inputs = torch.cat([vectors, labels, n_hot, m_hot], dim=-1)   # (B, K, D + 3K): (x; l; n; m) per object
    # Present the objects in a random order: argsort of random numbers is a random permutation per row,
    # and gather(1, idx) picks inputs[b, idx[b, t], :] for every step t.
    order = torch.rand(B, K, generator=generator).argsort(dim=-1)                   # (B, K)
    inputs = inputs.gather(1, order.unsqueeze(-1).expand(-1, -1, inputs.size(-1)))  # (B, K, D + 3K)
    return inputs, targets


def question_n(inputs, num_objects, num_features):
    """Read n (0-based) back from the inputs: the argmax of the n one-hot at step 0."""
    D, K = num_features, num_objects
    return inputs[:, 0, D + K:D + 2 * K].argmax(-1)


def evaluate(model, inputs, targets, num_objects, num_features):
    """Accuracy overall, per question rank n, on the relational questions (n < K-1: the distances
    are needed) and on the n = K-1 questions (answer = m, copied from the input)."""
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        correct = logits.argmax(-1) == targets
        loss = F.cross_entropy(logits, targets).item()
    model.train()
    n = question_n(inputs, num_objects, num_features)
    copy_m = n == num_objects - 1
    return {'loss': loss, 'acc': correct.float().mean().item(),
            'acc_by_n': [correct[n == k].float().mean().item() for k in range(num_objects)],
            'acc_relational': correct[~copy_m].float().mean().item(),
            'acc_reference': correct[copy_m].float().mean().item()}


def matched_lstm_hidden(target_params, input_size, output_size, head_hidden):
    """Smallest LSTM hidden size whose total parameter count reaches the RMC's (fair comparison)."""
    for h in range(4, 2049):
        if count_parameters(StandardLSTM(input_size, h, output_size, head_hidden=head_hidden)) >= target_params:
            return h
    return 2048


def train_model(name, model, args, eval_set):
    """Adam on fresh batches every step (as in Sonnet); the first args.curriculum_steps batches ask only
    for the farthest vector. Returns the evaluation history and the final numbers."""
    gen = torch.Generator().manual_seed(args.seed + 1)            # same training data for both models
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {'step': [], 'acc': [], 'acc_relational': [], 'loss': []}

    def log(step, t0):
        ev = evaluate(model, *eval_set, args.num_objects, args.num_features)
        for k in ('acc', 'acc_relational', 'loss'):
            history[k].append(ev[k])
        history['step'].append(step)
        phase = "farthest only" if step <= args.curriculum_steps else "all n"
        print(f"[{name}] step {step:5d} ({phase:13s}) eval: loss {ev['loss']:.3f}  acc {ev['acc']:.3f}  "
              f"by n {[round(a, 2) for a in ev['acc_by_n']]}  ({time.time() - t0:4.0f} s)", flush=True)
        return ev

    print(f"\nTraining {name}: {count_parameters(model):,} parameters")
    t0 = time.time()
    before = log(0, t0)
    for step in range(1, args.steps + 1):
        fixed_n = 0 if step <= args.curriculum_steps else None
        inputs, targets = generate_nth_farthest_batch(args.batch_size, args.num_objects, args.num_features,
                                                      gen, fixed_n=fixed_n)
        loss = F.cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            after = log(step, t0)
    return {'before': before, 'after': after, 'history': history, 'seconds': time.time() - t0,
            'params': count_parameters(model)}


def build_parser():
    parser = argparse.ArgumentParser(description="Day 20: RMC vs LSTM on the Nth Farthest task")
    parser.add_argument('--num_objects', type=int, default=3, help='K vectors per sequence (paper: 8)')
    parser.add_argument('--num_features', type=int, default=2, help='D dims per vector (paper: 16)')
    parser.add_argument('--curriculum_steps', type=int, default=1500,
                        help='first steps ask only for the farthest vector; 0 = the paper (every n from the start)')
    parser.add_argument('--slots', type=int, default=4, help='RMC memory slots N')
    parser.add_argument('--mem_size', type=int, default=64, help='RMC slot width F')
    parser.add_argument('--heads', type=int, default=4, help='RMC attention heads')
    parser.add_argument('--gate_style', type=str, default='unit', choices=['unit', 'memory'])
    parser.add_argument('--head_hidden', type=int, nargs='*', default=[256, 256, 256, 256],
                        help='output MLP widths shared by both models (paper: 4 x 256)')
    parser.add_argument('--lstm_hidden', type=int, default=0, help='0 = match the RMC parameter count')
    parser.add_argument('--steps', type=int, default=3000, help='training steps per model (curriculum included)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_every', type=int, default=250)
    parser.add_argument('--eval_size', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--full', action='store_true', help='opt-in longer run: 4x steps and curriculum steps')
    parser.add_argument('--model', type=str, choices=['rmc', 'lstm', 'both'], default='both')
    parser.add_argument('--visualize', action='store_true', help='plot the trained RMC attention')
    parser.add_argument('--save_plot', action='store_true', help='plot accuracy curves')
    parser.add_argument('--output_dir', type=str, default='.')
    return parser


def build_models(args):
    """The RMC and an LSTM with (about) the same number of parameters and the same output MLP."""
    K, D = args.num_objects, args.num_features
    input_size, head = D + 3 * K, tuple(args.head_hidden)
    torch.manual_seed(args.seed)
    rmc = RelationalRNN(input_size, args.slots, args.mem_size, K, num_heads=args.heads,
                        head_hidden=head, gate_style=args.gate_style)
    hidden = args.lstm_hidden or matched_lstm_hidden(count_parameters(rmc), input_size, K, head)
    torch.manual_seed(args.seed)
    lstm = StandardLSTM(input_size, hidden, K, head_hidden=head)
    print(f"RMC: {args.slots} slots x {args.mem_size}, {args.heads} heads, {count_parameters(rmc):,} params; "
          f"LSTM: hidden {hidden}, {count_parameters(lstm):,} params; output MLP {head}")
    return rmc, lstm


def main():
    args = build_parser().parse_args()
    if args.full:
        args.steps *= 4
        args.curriculum_steps *= 4
    os.makedirs(args.output_dir, exist_ok=True)

    K, D = args.num_objects, args.num_features
    eval_set = generate_nth_farthest_batch(args.eval_size, K, D, torch.Generator().manual_seed(12345))
    print(f"Nth Farthest: {K} vectors of {D} dims per sequence ({D + 3 * K} inputs per step). "
          f"Training: {args.curriculum_steps} steps 'farthest only', then all n; evaluation: all n.")
    print(f"Baselines: chance {1 / K:.3f}; copying m when n = K-1 and guessing otherwise {2 / K:.3f}")
    rmc, lstm = build_models(args)

    results = {}
    if args.model in ('rmc', 'both'):
        results['RMC'] = train_model('RMC', rmc, args, eval_set)
    if args.model in ('lstm', 'both'):
        results['LSTM'] = train_model('LSTM', lstm, args, eval_set)

    print(f"\n{'model':6s} {'params':>8s} {'time':>6s} {'acc before':>11s} {'acc after':>10s} "
          f"{'relational (n<K-1)':>19s} {'copy-m (n=K-1)':>15s}")
    for name, r in results.items():
        print(f"{name:6s} {r['params']:8,d} {r['seconds']:5.0f}s {r['before']['acc']:11.3f} {r['after']['acc']:10.3f} "
              f"{r['after']['acc_relational']:19.3f} {r['after']['acc_reference']:15.3f}")
    print(f"baselines: chance {1 / K:.3f} overall and {1 / (K - 1):.3f} on relational questions; "
          f"copying m when n = K-1: {2 / K:.3f} overall")

    if args.save_plot and results:
        plot_comparison({name: r['history'] for name, r in results.items()},
                        os.path.join(args.output_dir, "accuracy_comparison.png"),
                        baselines={'acc_relational': {'chance 1/(K-1)': 1 / (K - 1)},
                                   'acc': {'chance 1/K': 1 / K, 'copy m when n = K-1: 2/K': 2 / K}},
                        marker_step=args.curriculum_steps or None)
    if args.visualize and 'RMC' in results:
        inputs, targets = eval_set[0][:1], eval_set[1][:1]
        rmc.eval()
        with torch.no_grad():
            _, attention = rmc(inputs, return_attention=True)            # (1, T, H, N, N+1)
        labels = inputs[0, :, D:D + K].argmax(-1)                          # label shown at each step
        m = inputs[0, 0, D + 2 * K:].argmax().item()
        n = question_n(inputs, K, D)[0].item()
        ref_step = (labels == m).nonzero().item()                          # when vector m arrived
        plot_attention_heatmap(attention[0], os.path.join(args.output_dir, "attention_heatmap.png"),
                               highlight_step=ref_step,
                               title=f"Which vector is rank n = {n} (0 = farthest) from vector {m}? "
                                     f"Answer: vector {targets[0].item()}")


if __name__ == "__main__":
    main()
