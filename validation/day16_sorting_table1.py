"""Day 16 (Vinyals, Bengio, Kudlur 2015, "Order Matters") -- a small version of Table 1 (sorting, Sec. 4.4).

Out-of-sample greedy exact-sort accuracy ("whether we succeeded in sorting all numbers or not") for N
numbers uniform in [0, 1) (the paper's setting), comparing the Ptr-Net baseline (an LSTM reads the set
in order) with Read-Process-Write (papers/16_order_matters/implementation.py) for P = 0 / 5 process
steps and 0 / 1 glimpses. Every configuration sees the same training batches and the same test sets.

Paper: 10,000 training iterations (batch size not given), N = 5, 10, 15, P = 0, 1, 5, 10, glimpses 0/1.
Here, to fit a CPU: hidden size 64, batch 128, Adam lr 1e-3, gradient-norm clip 2, one seed, and
    default:  N = 5 and N = 10, 600 steps each                     (about 2 CPU-minutes)
    --full:   N = 5 with 1500 steps, N = 10 with 2500 steps         (about 6 CPU-minutes)
Test sets: 2000 per N, so one standard error is at most 1.1 accuracy points; with one training seed,
differences of a few points between neighbouring cells are not meaningful.

Masking: by default here an element already pointed at stays selectable (only the model's own learning
stops it from repeating one), as in the original Ptr-Net training; the paper does not say what it did.
--mask masks chosen elements, as train.py does by default; then sorting reduces to "point at the
smallest remaining element", which Read-Process-Write learns almost perfectly even with P = 0.

Run: OMP_NUM_THREADS=1 python validation/day16_sorting_table1.py [--full] [--mask] [--seed S]
"""
import argparse
import os
import sys
import time

import torch

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, "papers", "16_order_matters"))
import implementation as imp  # noqa: E402

# Table 1 of the paper (percent): (Ptr-Net, P=0, P=5) x glimpses (0, 1); Ptr-Net with a glimpse is not run here
PAPER = {5: {"ptrnet": 81, "P0g0": 65, "P0g1": 84, "P5g0": 88, "P5g1": 94},
         10: {"ptrnet": 8, "P0g0": 7, "P0g1": 30, "P5g0": 17, "P5g1": 57}}
CONFIGS = [("ptrnet", None, None), ("P0g0", 0, 0), ("P0g1", 0, 1), ("P5g0", 5, 0), ("P5g1", 5, 1)]


def build(name, P, G, hidden, mask):
    if name == "ptrnet":
        return imp.PointerNetwork(1, hidden, dropout=0.0, mask_chosen=mask)
    return imp.ReadProcessWrite(1, hidden, process_steps=P, glimpses=G, mask_chosen=mask)


def run(N, steps, args):
    gen = torch.Generator().manual_seed(1234 + N)
    x_test = torch.rand(2000, N, 1, generator=gen)          # fixed test sets, (2000, N, 1)
    y_test = x_test[..., 0].argsort(dim=1)                  # sorted order = argsort of the values
    lens_test = torch.full((2000,), N)
    lens = torch.full((args.batch,), N)
    row = {}
    for name, P, G in CONFIGS:
        torch.manual_seed(args.seed)
        model = build(name, P, G, args.hidden, args.mask)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = torch.Generator().manual_seed(args.seed + 100 * N)   # same batches for every config
        t0 = time.time()
        model.train()
        for _ in range(steps):
            x = torch.rand(args.batch, N, 1, generator=data)
            y = x[..., 0].argsort(dim=1)
            _, log_probs, _ = model(x, lens, N, teacher_forcing=y)
            loss = imp.compute_loss(log_probs, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pointers, _, _ = model(x_test, lens_test, N)    # greedy decoding
        acc = (pointers == y_test).all(dim=1).float().mean().item()
        srt = pointers.sort(dim=1).values                   # (2000, N): each row's pointers in increasing order
        repeats = (srt[:, 1:] == srt[:, :-1]).any(dim=1)    # equal neighbours = an element chosen twice
        row[name] = acc
        print(f"  N={N:2d} {name:6s}: exact sorts {acc:6.1%} (paper {PAPER[N][name]}%), outputs repeating "
              f"an element {repeats.float().mean().item():5.1%}, final train NLL {loss.item():.3f}, "
              f"{time.time() - t0:4.0f}s", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", action="store_true", help="1500 steps for N=5, 2500 for N=10 (~6 CPU-minutes)")
    ap.add_argument("--mask", action="store_true", help="mask elements already pointed at (train.py default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    budget = {5: 1500, 10: 2500} if args.full else {5: 600, 10: 600}
    print(f"Sorting, greedy exact-sort accuracy on 2000 test sets; hidden {args.hidden}, batch {args.batch}, "
          f"masking of chosen elements: {'on' if args.mask else 'off'}, seed {args.seed}")
    t0 = time.time()
    rows = {N: run(N, steps, args) for N, steps in budget.items()}
    print("\n| N (steps) | Ptr-Net | P=0, 0 glimpses | P=0, 1 glimpse | P=5, 0 glimpses | P=5, 1 glimpse |")
    print("|---|---|---|---|---|---|")
    for N, row in rows.items():
        cells = " | ".join(f"{row[k]:.1%} ({PAPER[N][k]}%)" for k, _, _ in CONFIGS)
        print(f"| {N} ({budget[N]}) | {cells} |")
    print(f"(paper's Table 1 in brackets; total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
