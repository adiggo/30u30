"""
NTM Training Script: The Copy Task
==================================

This script implements the "Copy Task" defined in section 4.1 of
Graves et al. (2014). It serves as the primary benchmark for verifying
that the NTM has learned a simple algorithmic procedure.

Tasks:
1. Input a sequence of random 8-bit vectors.
2. Input a unique delimiter.
3. Observe the NTM outputting the exact same sequence in order.

A standard LSTM often fails this task on long sequences because it tries
to compress the whole history into a fixed-size vector. The NTM solves it
by using its memory like a tape (Section 4.1).

Usage:
    python train_minimal.py             # paper's copy task, 3,000 iterations (~5-6 min on one CPU core)
    python train_minimal.py --paper     # the same task for 10,001 iterations (320k sequences, ~17 min)
    python train_minimal.py --iters 300 --max-len 5    # quick smoke run

[PATCH 2026-10-18] Settings follow Table 2 / Sec. 4.1 / Sec. 4.6 of the paper (NTM with an LSTM
controller of 100 units, one read and one write head, 128 x 20 memory, RMSProp lr 1e-4 with
momentum 0.9, sequences of 1-20 random 8-bit vectors, every gradient component clipped to
(-10, 10)). With the vectorized shift in implementation.py an iteration takes ~0.1 s instead of
~1.3 s, so the default run learns the task in about 5-6 minutes; progress is measured on a fixed
held-out set (per-bit BCE, bit error rate, and bits per sequence, the paper's unit).
NTM training is seed-sensitive: the held-out error sits at 30-40% for ~1,000-1,750 iterations and
then collapses, and some runs later destabilize (README, "Results in this repo").
Differences from the paper: batches of 32 sequences (the paper does not give a batch size), the
loss is BCE averaged per bit (the paper's cost is cross-entropy per sequence), and PyTorch's
uncentred RMSprop (the paper uses the centred form of Graves, 2013).
"""

import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from implementation import NTM

def generate_copy_instance(batch_size, seq_len, width):
    """
    Generates data for a single training step.

    Format (Section 4.1):
    - Input phase: [Data bits] followed by [1 at index Width+1]
    - Output phase: Model must output [Data bits] given zero input.
    """
    # Create random bit vectors {0, 1}^width
    data = torch.randn(batch_size, seq_len, width).sign().add(1).div(2)

    # Input sequence: [data] + [delimiter] + [zeros for output phase]
    # Total length: L (input) + 1 (delim) + L (output)
    full_len = 2 * seq_len + 1
    inputs = torch.zeros(batch_size, full_len, width + 1)
    targets = torch.zeros(batch_size, full_len, width)

    inputs[:, :seq_len, :width] = data
    inputs[:, seq_len, width] = 1.0 # Delimiter bit

    # Target is the data during the second L timesteps
    targets[:, seq_len+1:, :] = data

    return inputs.transpose(0, 1), targets.transpose(0, 1)


# [PATCH 2026-10-18] everything below: held-out evaluation, the paper's elementwise clipping,
# configurable run length, timing, and a generalization check on longer sequences.

def run_sequence(model, inputs):
    """Feed a (T, B, width+1) input through the NTM one time step at a time; returns logits (T, B, width)."""
    state = model.reset(inputs.size(1))
    outs = []
    for t in range(inputs.size(0)):
        out, state = model(inputs[t], state)
        outs.append(out)
    return torch.stack(outs)


def make_eval_set(lengths, width, per_length=4, seed=1234):
    """A fixed held-out set: `per_length` random sequences for each length, drawn from a separate seed
    (the global RNG state is saved and restored, so training data is unaffected)."""
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    eval_set = [generate_copy_instance(per_length, L, width) for L in lengths]
    torch.set_rng_state(rng_state)
    return eval_set


def evaluate(model, eval_set):
    """Per-bit BCE (nats), bit error rate, and cost in bits per sequence (the paper's unit) on the recall phase."""
    bce = nn.BCEWithLogitsLoss(reduction='sum')
    total, errors, bits, seqs = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in eval_set:
            L = (inputs.size(0) - 1) // 2
            logits, target = run_sequence(model, inputs)[L + 1:], targets[L + 1:]
            total += bce(logits, target).item()
            errors += ((logits > 0).float() != target).sum().item()
            bits += target.numel()
            seqs += inputs.size(1)
    return total / bits, errors / bits, total / math.log(2) / seqs


def train(iters=3000, max_seq_len=20, batch_size=32, N=128, M=20, controller_size=100, lr=1e-4,
          clip='value', seed=0, eval_every=250, save=True, loss_type='bit'):
    # Experimental Hyperparameters (Section 4, Table 2)
    width = 8
    torch.manual_seed(seed)
    random.seed(seed)

    model = NTM(width + 1, width, controller_size, N, M)

    # Section 4.6: RMSProp with momentum 0.9; Table 2: learning rate 1e-4.
    # alpha = 0.95 is the decay of Graves (2013)'s RMSProp.
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=0.9, alpha=0.95)

    # Binary Cross Entropy is ideal for bit-copy tasks.
    # 'bit': mean over every target bit of the batch; 'sequence': summed over each sequence's
    # target bits and averaged over the batch, i.e. the paper's "cost per sequence" (in nats).
    criterion = nn.BCEWithLogitsLoss(reduction='mean' if loss_type == 'bit' else 'sum')

    eval_set = make_eval_set(range(1, max_seq_len + 1), width)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Starting Copy Task training (N={N}, M={M}, Controller={controller_size}, {n_params:,} parameters, "
          f"lengths 1-{max_seq_len}, batch {batch_size}, {iters} iterations, clip={clip}, loss={loss_type})...")
    l0, e0, s0 = evaluate(model, eval_set)
    print(f"  held-out before training: BCE/bit {l0:.4f}  bit error {100 * e0:.2f}%  cost {s0:.2f} bits/sequence")

    train_time = 0.0
    history = [(0, l0, e0, s0)]
    for i in range(iters):
        t_start = time.perf_counter()
        optimizer.zero_grad()

        # Vary sequence length to force the model to learn the 'rule' of copying
        # instead of a fixed-length pattern (Section 4.1).
        cur_len = random.randint(1, max_seq_len)
        inputs, targets = generate_copy_instance(batch_size, cur_len, width)

        # Sequentially process the entire task sequence
        out_seq = run_sequence(model, inputs)

        # Only compute loss on the recall phase
        loss = criterion(out_seq[cur_len+1:], targets[cur_len+1:])
        if loss_type == 'sequence':
            loss = loss / batch_size
        loss.backward()

        # Section 4.6: "all gradient components are clipped elementwise to the range (-10, 10)".
        # (The repo used to clip the global gradient norm at 10, which is a different operation: it
        # rescales the whole gradient whenever its norm exceeds 10.) With the per-bit loss a component
        # exceeded 10 only a handful of times in our runs, while the network was discovering the copy
        # procedure; with --loss sequence (the paper's cost per sequence, ~8 L times larger) it bites
        # far more often.
        if clip == 'value':
            nn.utils.clip_grad_value_(model.parameters(), 10.0)
        else:
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        optimizer.step()
        train_time += time.perf_counter() - t_start

        if i % 100 == 0:
            with torch.no_grad():
                # Bit Error Rate (BER): % of bits that are wrong
                preds = (out_seq[cur_len+1:] > 0).float()
                error = torch.abs(preds - targets[cur_len+1:]).mean().item()
            print(f"Iter {i:5d} | Len {cur_len:2d} | Loss {loss.item():.4f} | BER {error*100:.2f}%")

        if (i + 1) % eval_every == 0 or i + 1 == iters:
            l, e, s = evaluate(model, eval_set)
            history.append((i + 1, l, e, s))
            print(f"  held-out after {i + 1:5d} iterations: BCE/bit {l:.4f}  bit error {100 * e:.2f}%  "
                  f"cost {s:.2f} bits/sequence  ({(i + 1) / train_time:.2f} it/s)")

    print(f"\nHeld-out (lengths 1-{max_seq_len}): BCE/bit {l0:.4f} -> {history[-1][1]:.4f}, "
          f"bit error {100 * e0:.2f}% -> {100 * history[-1][2]:.2f}%, "
          f"cost {s0:.2f} -> {history[-1][3]:.2f} bits/sequence; {iters / train_time:.2f} it/s")

    # Section 4.1, Fig. 4: does the learned procedure generalize to sequences longer than any seen
    # in training? (Memory has N slots, so lengths up to ~N can be stored.)
    gen_lengths = [L for L in (10, 20, 30, 50) if L <= N]
    gen_set = make_eval_set(gen_lengths, width, per_length=8, seed=4321)
    print("Generalization (bit error by length; trained on 1-%d):" % max_seq_len,
          ", ".join(f"L={L}: {100 * evaluate(model, [batch])[1]:.2f}%" for L, batch in zip(gen_lengths, gen_set)))

    if save:
        # Save weights for visualization
        # [PATCH 2026-10-18] save next to this script. The repo-root-relative path raised FileNotFoundError when the
        # script was run from its own directory (as the README's quick start does), losing the trained weights.
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ntm_copy_final.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Training complete. Weights saved to {ckpt}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="NTM copy task (Graves et al. 2014, Sec. 4.1)")
    parser.add_argument('--iters', type=int, default=3000,
                        help='training iterations (batches); default 3000, about 5-6 minutes on one CPU core')
    parser.add_argument('--paper', action='store_true',
                        help="the paper's full run length here: 10,001 iterations x 32 = 320k sequences (~17 min)")
    parser.add_argument('--max-len', type=int, default=20, help='sequence lengths 1..max-len (paper: 20)')
    parser.add_argument('--batch', type=int, default=32, help='sequences per batch (default 32)')
    parser.add_argument('--memory-slots', type=int, default=128, help='N, memory rows (paper: 128)')
    parser.add_argument('--lr', type=float, default=1e-4, help='RMSprop learning rate (paper: 1e-4)')
    parser.add_argument('--clip', choices=['value', 'norm'], default='value',
                        help="'value': clip every gradient component to (-10, 10) (paper, Sec. 4.6); "
                             "'norm': clip the global gradient norm at 10 (the repo's earlier choice)")
    parser.add_argument('--loss', choices=['bit', 'sequence'], default='bit',
                        help="'bit': BCE averaged per bit; 'sequence': BCE summed per sequence, averaged over "
                             "the batch (the paper's cost per sequence)")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=250, help='held-out evaluation interval (iterations)')
    parser.add_argument('--no-save', action='store_true', help='do not write ntm_copy_final.pth')
    args = parser.parse_args()
    iters = 10001 if args.paper else args.iters
    train(iters=iters, max_seq_len=args.max_len, batch_size=args.batch, N=args.memory_slots, lr=args.lr,
          clip=args.clip, seed=args.seed, eval_every=args.eval_every, save=not args.no_save,
          loss_type=args.loss)


if __name__ == "__main__":
    main()
