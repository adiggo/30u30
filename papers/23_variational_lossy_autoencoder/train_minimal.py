"""
train_minimal.py - Variational Lossy Autoencoder (VLAE) Training CLI

Train a VLAE on binarized MNIST, then measure whether the latent code z is actually used.

[PATCH 2026-10-18] Trains with the paper's free bits by default (App. A: 0.01 nats per data
dimension; see free_bits_loss in implementation.py). --no-free-bits trains the plain ELBO, which
collapses: the KL to the prior stays near 0 and the decoder ignores z. The default run is
CPU-sized (200 batches of 64 images, about 4.5 minutes on one core with the plots); the paper
trains to convergence with a larger model (App. A). Each logged line shows the training objective (with
the free-bits max) next to the true KL and negative ELBO (without it), per image.

Usage:
    # Default: free bits, 200 batches
    python train_minimal.py

    # Collapse comparison: the plain ELBO
    python train_minimal.py --no-free-bits

    # Learned autoregressive-flow (AF) prior, as in the paper
    python train_minimal.py --use-flow

    # Longer opt-in run: one full epoch, 938 batches (about 16 minutes on one core)
    python train_minimal.py --max-batches 0 --eval-images 5000

    # A deeper decoder (larger receptive field: 8 layers see 144 past pixels, 3 layers see 34)
    python train_minimal.py --decoder-layers 8

At the end it reports, on test images: the KL to the prior, -log p(x|z) with each image's own z,
with another image's z and with z ~ p(z), an estimate of I(x; z), and how well a linear probe
reads the digit class from one z sample.

Reference: Chen et al. (2017) "Variational Lossy Autoencoder"
Author: 30u30 Project
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from typing import Dict, List

from implementation import VLAE, free_bits_loss, latent_usage, linear_probe_accuracy
from visualization import (
    plot_training_trajectories,
    plot_decompressions,
    plot_bits_heatmap,
    plot_latent_traversals
)

HERE = os.path.dirname(os.path.abspath(__file__))


def binarize(images_uint8: torch.Tensor) -> torch.Tensor:
    """(N, 28, 28) uint8 -> (N, 1, 28, 28) float {0, 1}; same threshold as the training transform
    (ToTensor divides by 255, then > 0.5)."""
    return (images_uint8.float() / 255 > 0.5).float().unsqueeze(1)


def train(model, train_loader, optimizer, device, epoch, free_bits=0.01, max_batches=0,
          log_every=25, history=None, step0=0):
    """[PATCH 2026-10-18] One pass over train_loader (stopping after max_batches if > 0) with the
    free-bits objective. Logs per-image averages over the last log_every batches: the objective
    that is optimised, the reconstruction term, the true KL and the true negative ELBO.
    Returns the number of batches run."""
    model.train()
    sums = {'objective': 0.0, 'recon': 0.0, 'kl': 0.0}
    n_images = 0
    n_batches = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break
        data = data.to(device)
        optimizer.zero_grad()

        logits, mu, logvar, z, log_det = model(data)
        loss, parts = free_bits_loss(model, logits, data, mu, logvar, z, free_bits=free_bits)
        loss.backward()
        optimizer.step()

        sums['objective'] += loss.item()
        sums['recon'] += parts['recon']
        sums['kl'] += parts['kl']
        n_images += len(data)
        n_batches += 1
        if n_batches % log_every == 0 or (max_batches and n_batches == max_batches):
            avg = {k: v / n_images for k, v in sums.items()}
            step = step0 + n_batches
            print(f"Epoch {epoch} batch {step:5d} | per image: objective {avg['objective']:7.2f} | "
                  f"recon {avg['recon']:7.2f} | KL {avg['kl']:6.2f} | "
                  f"-ELBO {avg['recon'] + avg['kl']:7.2f} nats", flush=True)
            if history is not None:
                history['step'].append(step)
                history['objective'].append(avg['objective'])
                history['recon'].append(avg['recon'])
                history['kl'].append(avg['kl'])
            sums = {k: 0.0 for k in sums}
            n_images = 0
    return n_batches


@torch.no_grad()
def encode_samples(model, x: torch.Tensor, batch_size: int = 500) -> torch.Tensor:
    """One z ~ q(z|x) per image, (N, D)."""
    model.eval()
    out = []
    for s in range(0, len(x), batch_size):
        mu, logvar = model.encoder(x[s:s + batch_size])
        out.append(model.reparameterize(mu, logvar))
    return torch.cat(out)


def evaluate(model, train_set, test_set, device, eval_images=2000, probe_images=10000) -> Dict:
    """[PATCH 2026-10-18] Latent-usage diagnostics on test images (see latent_usage) plus a linear
    probe for the digit class on one z sample (trained on probe_images training images, tested
    on the same number of test images)."""
    x_test = binarize(test_set.data).to(device)
    y_test = test_set.targets
    usage = latent_usage(model, x_test[:eval_images])
    n = min(probe_images, len(test_set))
    z_tr = encode_samples(model, binarize(train_set.data[:n]).to(device)).cpu()
    z_te = encode_samples(model, x_test[:n]).cpu()
    usage['probe_accuracy'] = linear_probe_accuracy(z_tr, train_set.targets[:n], z_te, y_test[:n])
    usage['probe_images'] = n
    return usage


def print_usage(u: Dict, budget: float):
    print(f"\nLatent usage on {u['n_images']} test images (nats per image):")
    note = f"free-bits budget {budget:.2f}" if budget > 0 else "free bits off"
    print(f"  KL(q(z|x) || p(z)), true             {u['kl']:7.2f}   ({note})")
    print(f"  I(x; z) estimate                     {u['mi']:7.2f}   (cannot exceed log N = {u['mi_cap']:.2f})")
    print(f"  -log p(x|z), z from the same image   {u['nll_own']:7.2f}")
    print(f"  -log p(x|z), z from another image    {u['nll_shuffled']:7.2f}")
    print(f"  -log p(x|z), z ~ p(z)                {u['nll_prior']:7.2f}")
    print(f"  gap (another image - same image)     {u['gap']:7.2f}   (~0 when z is ignored)")
    print(f"  negative ELBO                        {u['neg_elbo']:7.2f}   (= {u['bpd']:.4f} bits/dim)")
    print(f"  digit class read from one z sample   {100 * u['probe_accuracy']:6.1f} %  "
          f"(linear probe, {u['probe_images']} train / test images; chance 10 %)")


def main():
    parser = argparse.ArgumentParser(description='VLAE Training')
    # [PATCH 2026-10-18] CPU-sized defaults (was 10 full epochs, > 1 h per epoch on one core),
    # free-bits flags, evaluation and data options.
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')
    parser.add_argument('--max-batches', type=int, default=200,
                        help='stop after this many batches in total (0 = run all epochs in full)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--latent-dim', type=int, default=32, help='latent dimension size')
    parser.add_argument('--use-flow', action='store_true', help='use the learned AF prior p(z)')
    parser.add_argument('--decoder-layers', type=int, default=3, help='number of gated pixelcnn layers')
    parser.add_argument('--free-bits', type=float, default=0.01,
                        help='free bits in nats per data dimension (paper App. A: 0.01)')
    parser.add_argument('--no-free-bits', action='store_true',
                        help='train the plain ELBO (posterior-collapse comparison)')
    parser.add_argument('--eval-images', type=int, default=2000, help='test images for the latent-usage check')
    parser.add_argument('--log-every', type=int, default=25, help='batches per log line')
    parser.add_argument('--no-plots', action='store_true', help='skip the plots (they take about a minute)')
    parser.add_argument('--data-dir', type=str, default=os.path.join(HERE, 'data'), help='MNIST root')
    parser.add_argument('--output-dir', type=str, default='results', help='directory for results')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    free_bits = 0.0 if args.no_free_bits else args.free_bits

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   # [PATCH 2026-10-18] MADE (flow prior) draws its hidden-unit degrees with np.random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    budget = free_bits * 28 * 28
    print("=" * 72)
    print(f"Training VLAE (Flow: {args.use_flow}, Decoder Layers: {args.decoder_layers})")
    if free_bits > 0:
        print(f"Free bits: {free_bits} nats/pixel x 784 = {budget:.2f} nats per image, split over "
              f"{args.latent_dim} groups (one per latent dim): {budget / args.latent_dim:.3f} nats each")
    else:
        print("Free bits: off (plain ELBO)")
    print(f"Device: {device}")
    print("=" * 72)

    # Data Loader (Binarized MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: (x > 0.5).float()
    ])
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Model & Optimizer
    model = VLAE(input_dim=1, latent_dim=args.latent_dim, n_layers=args.decoder_layers, use_flow=args.use_flow).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history: Dict[str, List[float]] = {'step': [], 'objective': [], 'recon': [], 'kl': []}
    history['kl_budget'] = budget

    start_time = time.time()
    steps = 0
    for epoch in range(1, args.epochs + 1):
        remaining = args.max_batches - steps if args.max_batches else 0
        steps += train(model, train_loader, optimizer, device, epoch, free_bits=free_bits,
                       max_batches=remaining, log_every=args.log_every, history=history, step0=steps)
        if args.max_batches and steps >= args.max_batches:
            break
    train_time = time.time() - start_time
    print(f"\nTrained {steps} batches ({steps * args.batch_size} images) in {train_time:.1f}s")

    usage = evaluate(model, train_set, test_set, device, eval_images=args.eval_images)
    print_usage(usage, budget)
    usage.update({'free_bits': free_bits, 'use_flow': args.use_flow, 'decoder_layers': args.decoder_layers,
                  'batches': steps, 'train_seconds': train_time})
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({'usage': usage, 'history': history}, f, indent=1)

    if not args.no_plots:
        print("\nGenerating final plots...")
        plot_training_trajectories(history, args.output_dir)
        test_data = binarize(test_set.data[:8])
        # The paper's Fig. 1a: decompressions z ~ q(z|x), x ~ p(x|z), next to samples from the prior
        plot_decompressions(model, test_data, device, output_dir=args.output_dir)
        with torch.no_grad():
            _, mu, logvar, _, _ = model(binarize(test_set.data[:100]).to(device))
            plot_bits_heatmap(mu, logvar, args.output_dir)
        print("Generating latent traversals...")
        plot_latent_traversals(model, test_data[:1], device, args.latent_dim, dim_idx=0, output_dir=args.output_dir)

    # Save Model
    save_path = os.path.join(args.output_dir, "vlae_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path} (total {time.time() - start_time:.0f}s)")
    print("=" * 72)


if __name__ == "__main__":
    main()
