"""
visualization.py - Visualization Suite for Variational Lossy Autoencoder (VLAE)

Generates key plots demonstrating the core themes of Chen et al. (2017):
1. Posterior Convergence: KL Divergence vs. Reconstruction Loss over time.
2. Global vs Local: teacher-forced reconstructions, and decompressions z ~ q(z|x), x ~ p(x|z)
   (the paper's Fig. 1a).
3. Latent Sampling: Autoregressive pixel-by-pixel generation.
4. KL per latent dimension (to N(0, I), in nats).
5. Latent Traversals: Demonstrating high-level semantic capture in z.

Reference: Chen et al. (2017) - https://arxiv.org/abs/1611.02731
Author: 30u30 Project
"""

import argparse
import sys
import os
import torch
import numpy as np
from typing import Dict, List

try:
    import matplotlib
    # [PATCH 2026-10-18] Forcing Agg on import switched a notebook's inline backend off, so
    # figures never displayed; use Agg only outside Jupyter (scripts save PNGs either way).
    if 'ipykernel' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("matplotlib is required for visualization.")
    print("Install with: pip install matplotlib")
    sys.exit(1)

from implementation import VLAE


def _save(outpath: str):
    """[PATCH 2026-10-18] Save the current figure; inside Jupyter also show it before closing
    (plt.close() alone discarded every figure these helpers drew in the notebook)."""
    plt.savefig(outpath, dpi=150)
    if 'ipykernel' in sys.modules:
        plt.show()
    plt.close()
    print(f"  Saved: {outpath}")


@torch.no_grad()
def sample_pixels(model, z: torch.Tensor, height: int = 28, width: int = 28) -> torch.Tensor:
    """[PATCH 2026-10-18] x ~ p(x|z) for a batch of codes z (B, D), one pixel at a time in raster
    order. Masked convolutions never look below the current row, so row i only needs rows 0..i
    of the canvas: decoding that crop gives the same logits for row i at about half the cost."""
    x = torch.zeros(z.size(0), 1, height, width, device=z.device)
    for i in range(height):
        for j in range(width):
            logits = model.decode(x[:, :, :i + 1, :], z)        # (B, 1, i+1, W): rows 0..i only
            x[:, :, i, j] = torch.bernoulli(torch.sigmoid(logits[:, :, i, j]))
    return x


def plot_training_trajectories(history: Dict[str, List[float]], output_dir: str = '.'):
    """
    Plot 1: Training convergence. 
    Crucial for VLAE to show that KL stays non-zero (No Collapse).

    [PATCH 2026-10-18] history: 'step' (batches seen), 'recon' and 'kl' (true KL, no free-bits
    max), per image in nats, optional 'kl_budget' (free-bits budget per image). Was labelled
    'Bits' and 'Epochs' while plotting nats per logged interval.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Training batches')
    ax1.set_ylabel('Reconstruction term -log p(x|z) (nats/image)', color=color)
    ax1.plot(history['step'], history['recon'], color=color, linewidth=2, label='Recon Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('KL(q(z|x) || p(z)) (nats/image)', color=color)
    ax2.plot(history['step'], history['kl'], color=color, linewidth=2, label='KL Divergence')
    if history.get('kl_budget'):
        ax2.axhline(history['kl_budget'], color=color, linestyle='--', linewidth=1,
                    label='free-bits budget')
        ax2.legend(loc='upper right')
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("VLAE Training Trajectory: Monitoring Posterior Collapse", fontsize=14)
    fig.tight_layout()

    _save(os.path.join(output_dir, "plot1_training_trajectory.png"))


def plot_reconstructions(model, data, device, output_dir='.'):
    """
    Plot 2: Original vs Reconstruction.
    Visual assessment of the 'lossy' vs 'textured' trade-off.
    """
    model.eval()
    with torch.no_grad():
        logits, _, _, _, _ = model(data.to(device))
        recon = torch.sigmoid(logits).cpu()

    n = 8
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        # Original
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original", loc='left')
        
        # Recon
        axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("VLAE Recon", loc='left')

    # [PATCH 2026-10-18] Say what these are: each pixel is predicted from the true pixels above and
    # to its left, so they look sharp even when z carries nothing (see plot_decompressions).
    plt.suptitle("Teacher-forced one-step predictions p(x_i | x_<i, z) (not decompressions)", fontsize=14)
    plt.tight_layout()
    _save(os.path.join(output_dir, "plot2_reconstructions.png"))


def plot_decompressions(model, data, device, n_prior=8, output_dir='.',
                        filename="plot2b_decompressions.png", title=None):
    """
    [PATCH 2026-10-18] The paper's Fig. 1a: originals x, "decompressions" z ~ q(z|x), x' ~ p(x|z)
    sampled pixel by pixel, and samples from the prior z ~ p(z), x' ~ p(x|z). If z carries the
    global structure, a decompression keeps the digit and its shape while the local details
    (exact pixels, stroke texture) are redrawn; if z is ignored it is just another prior sample.
    One batched sampling pass (784 decoder passes).
    """
    print("  Decompressing (pixel-by-pixel)...")
    model.eval()
    n = len(data)
    with torch.no_grad():
        mu, logvar = model.encoder(data.to(device))
        z = torch.cat([model.reparameterize(mu, logvar), model.sample_prior(n_prior)])  # (n + n_prior, D)
        samples = sample_pixels(model, z).cpu()
    cols = max(n, n_prior)
    fig, axes = plt.subplots(3, cols, figsize=(cols * 1.6, 5.2), squeeze=False)
    rows = [("original x", data.cpu()), ("decompressed\nz~q(z|x), x~p(x|z)", samples[:n]),
            ("prior sample\nz~p(z), x~p(x|z)", samples[n:])]
    for r, (label, imgs) in enumerate(rows):
        for c in range(cols):
            axes[r, c].axis('off')
            if c < len(imgs):
                axes[r, c].imshow(imgs[c].squeeze(), cmap='gray')
        axes[r, 0].set_title(label, loc='left', fontsize=9)
    plt.suptitle(title or "Lossy code: what z keeps (decompressions) vs. what the decoder invents", fontsize=13)
    plt.tight_layout()
    _save(os.path.join(output_dir, filename))


def plot_latent_sampling(model, device, latent_dim, n_samples=16, output_dir='.'):
    """
    Plot 3: True autoregressive sampling from the latent prior.
    This is slow because it happens pixel-by-pixel.
    """
    print("  Generating autoregressive samples (pixel-by-pixel)...")
    model.eval()
    with torch.no_grad():
        # [PATCH 2026-10-18] Sample z from the model's prior: the learned AF prior when use_flow
        # (was always N(0, I)); the pixel loop moved into sample_pixels.
        z = model.sample_prior(n_samples).to(device)
        samples = sample_pixels(model, z)

    side = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(side, side, figsize=(2 * side, 2 * side), squeeze=False)
    for i in range(side * side):
        r, c = i // side, i % side
        axes[r, c].axis('off')
        if i < n_samples:
            axes[r, c].imshow(samples[i].cpu().squeeze(), cmap='gray')

    plt.suptitle("Samples from Prior $p(z)$ via Autoregressive Decoding", fontsize=14)
    plt.tight_layout()
    _save(os.path.join(output_dir, "plot3_sampling.png"))


def plot_bits_heatmap(mu, logvar, output_dir='.'):
    """
    Plot 4: Visualize how many 'bits' are used in each latent dimension.
    Based on KL = 0.5 * (exp(logvar) + mu^2 - 1 - logvar)

    [PATCH 2026-10-18] This is KL(q(z_d|x) || N(0, 1)) in nats (the axis said bits). It is the KL
    to the prior only without the flow; with the learned AF prior it is not.
    """
    kl_per_dim = 0.5 * (torch.exp(logvar) + torch.pow(mu, 2) - 1 - logvar)
    avg_kl = kl_per_dim.mean(dim=0).cpu().detach().numpy()

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(avg_kl)), avg_kl, color='skyblue', edgecolor='navy')
    plt.xlabel("Latent Dimension")
    plt.ylabel("KL to N(0, 1) (nats)")
    plt.title("KL(q(z_d|x) || N(0, 1)) per Latent Dimension, averaged over images")
    plt.grid(axis='y', alpha=0.3)

    _save(os.path.join(output_dir, "plot4_bits_per_dim.png"))


def plot_latent_traversals(model, data, device, latent_dim, dim_idx=0, output_dir='.'):
    """
    Plot 5: Latent space traversals.
    Sweeps a latent dimension to show semantic control.
    """
    model.eval()
    with torch.no_grad():
        # Encode a single sample
        mu, logvar = model.encoder(data[0:1].to(device))
        z_base = mu # Use mu for a clean baseline
        
        n_steps = 8
        traversals = []
        # Move z in the range [-3, 3]
        for val in np.linspace(-3, 3, n_steps):
            z_step = z_base.clone()
            z_step[0, dim_idx] = val
            
            # Reconstruction (Note: Using fast non-autoregressive recon for visualization)
            # [PATCH 2026-10-18] through model.decode: z now reaches the decoder as a feature map
            logits = model.decode(torch.zeros(1, 1, 28, 28).to(device), z_step)
            traversals.append(torch.sigmoid(logits).cpu().squeeze())

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 2, 2.5))
    for i, img in enumerate(traversals):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"z[{dim_idx}]={np.linspace(-3,3,n_steps)[i]:.1f}", fontsize=8)

    plt.suptitle(f"Latent Traversal: Dimension {dim_idx}", fontsize=14)
    plt.tight_layout()
    _save(os.path.join(output_dir, f"plot5_traversal_dim{dim_idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLAE Visualization Suite")
    parser.add_argument('--model-path', type=str, default='vlae_mnist.pth')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=" * 60)
    print("VLAE Visualization Suite (Chen et al. 2017)")
    print("=" * 60)
    
    # In a real scenario, we'd load the model and data here.
    # For this suite, we provide the standalone functions to be called 
    # from train_minimal.py or the notebook.
    print("Utility script loaded. Call functions with trained model/data.")
