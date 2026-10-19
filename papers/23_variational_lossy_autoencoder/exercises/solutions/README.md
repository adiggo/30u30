# Day 23 Solutions: Variational Lossy Autoencoder

This directory contains the reference implementations for the Day 23 exercises.

## Overview of Solutions

### 1. Masked Convolutions (`solution_01_masked_conv.py`)
- **Key Logic:** Implementation of spatial masking. 
- **Type A:** Masks the center pixel (0). Essential for the very first layer so the model can't "see" the pixel it's trying to predict.
- **Type B:** Includes the center pixel (1). Used for all subsequent layers to allow information flow from previous layers.

### 2. Gated PixelCNN Prior (`solution_02_pixelcnn_prior.py`)
- **Key Logic:** Standalone autoregressive density estimation.
- **Goal:** Verify that your MaskedConvs can actually generate image-like structures without any latent code $z$.

### 3. Reproducing Collapse (`solution_03_posterior_collapse.py`)
- **Key Logic:** Intentional failure mode.
- **Setup:** Uses a large receptive field (8 decoder layers) with a standard Gaussian prior and the plain ELBO, for 60 batches (CPU-sized; `MAX_BATCHES = 0` runs the full epoch).
- **Result:** The KL stays near zero from the start: the encoder begins close to $q(z|x) = \mathcal{N}(0, I)$ and the plain ELBO never pulls information into $z$. The default 3-layer decoder collapses the same way without free bits, so this run shows the optimisation side of collapse, not a receptive-field effect.

### 4. Flow Prior (`solution_04_flow_prior.py`)
- **Key Logic:** Pushes Gaussian samples through two of the repo's MADE flow blocks (the exercise text asks for a planar flow). In VLAE these blocks parameterise an autoregressive-flow (AF) prior (Sec. 3.2), not an IAF.
- **Mechanism:** Uses MADE to calculate $\mu$ and $\sigma$ for each latent dimension in a single forward pass while maintaining the autoregressive constraint.

### 5. Final Assembly (`solution_05_vlae_complete.py`)
- **Key Logic:** The complete VLAE: AF prior, restricted decoder, and the paper's free bits (App. A, 0.01 nats per pixel) via `free_bits_loss`. 150 batches by default (`MAX_BATCHES`).
- **Verification:** Prints the true KL next to the training objective, then checks on 1,000 test images that the decoder uses $z$: $-\log p(x|z)$ with each image's own $z$ vs another image's, and an estimate of $I(x; z)$. With the plain ELBO the same model collapses.

## Benchmarks (MNIST)

The paper's numbers come from fully trained models on stochastically binarized MNIST; the solutions run a few hundred CPU batches on MNIST thresholded at 0.5, so the nats are not directly comparable.

| Metric | Paper (static MNIST, converged) | `solution_05` (150 batches, CPU) |
|--------|-------------------------|-------------------------|
| **KL to the prior** | 13.3 nats = 19.2 bits (Sec. 4.1) | 4.39 nats |
| **$-\log p(x\|z)$, own vs another image's $z$** | not reported | 78.76 vs 81.40 nats |
| **NLL / negative ELBO** | 79.03 nats NLL (Table 1, importance-sampled) | 83.15 nats negative ELBO (an upper bound) |

---
**Next:** [Day 24 — GPipe](../../24_gpipe/)
