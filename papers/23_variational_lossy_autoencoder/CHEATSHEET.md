# Day 23 Cheat Sheet: Variational Lossy Autoencoder

## The Big Idea (30 seconds)

VAEs with powerful autoregressive decoders (like PixelCNN) suffer from **posterior collapse** — the model ignores the latent code $z$ because local pixel prediction is "easier." **VLAE** fixes this by:
1.  **Limiting the decoder's receptive field** (forcing it to look at $z$ for global structure).
2.  **Using a learned autoregressive-flow (AF) prior** (equal to an IAF posterior along the encoder path, Sec. 3.2; it shortens the code).
3.  **Training with free bits** (App. A: 0.01 nats per pixel), because the preference in 1 is "an asymptotic view": without it the KL collapses early in training.

**Cure the collapse:** Force the model to use $z$ by making it necessary for global coherence, and keep the optimiser from switching $z$ off before it learns to use it.

---

## Quick Start (Training & Sampling)

```bash
# Verify Implementation (Unit Tests)
python setup.py

# Train VLAE on MNIST with free bits (200 batches, ~4.5 min on one CPU core)
python train_minimal.py
python train_minimal.py --no-free-bits      # plain ELBO: collapses
python train_minimal.py --use-flow          # learned AF prior

# Qualitative Check (Decompressions, the paper's Fig. 1a)
# See results/plot2b_decompressions.png
```

---

## Architecture at a Glance

| Component | Choice | Why? |
|-----------|--------|------|
| **Encoder** | ResNet-style | Compresses image to $\mu, \sigma$ for $z$. |
| **Prior $p(z)$** | **AF (autoregressive flow) from $\mathcal{N}(0, I)$** | Allows prior to learn complex structure (Sec. 3.2). |
| **Decoder $p(x\|z)$** | **Restricted masked PixelCNN** | Handles local texture; blind to global patterns. Gets $z$ as a 28x28 feature map. |
| **Training** | **Teacher Forcing + free bits** | Feed original pixel $x_i$ to predict $x_{i+1}$; KL below $\lambda$ per group is free. |

---

## Key Hyperparameters

| Parameter | Range | Importance |
|-----------|-------|------------|
| `latent_dim` | 16 - 64 | **High**: Global information capacity. |
| `decoder_layers`| 2 - 4 | **CRITICAL**: Controls receptive field (Keep < 8). |
| `kernel_size` | 3 or 5 | Controls how quickly receptive field grows. |
| `flow_steps` | 2 - 4 | Complexity of the prior $p(z)$. |
| `free_bits` | 0.01 nats/pixel (paper) | **High**: 0 = plain ELBO, which collapses. Here 7.84 nats/image = 0.245 per latent dim. |

---

## Common Issues & Fixes

- **KL Divergence is 0:** Classic posterior collapse. **Fix:** free bits (paper App. A) or KL annealing; a small receptive field alone did not prevent it here (the 3-layer decoder collapses with the plain ELBO).
- **KL is at the budget but $z$ is unused:** free bits can be spent on shrinking $\sigma$ without encoding anything. Check $-\log p(x|z)$ with each image's own $z$ vs another image's (`latent_usage`). With a small receptive field the decoder also needs $z$ in a spatial form (a feature map), or it cannot tell where its strokes go.
- **Checkerboard Artifacts:** Often due to bad latent injection or masking bias. **Fix:** Ensure $z$ is upsampled correctly and used in every Gated layer.
- **Blurry Images:** VAE tendency. **Fix:** Increase PixelCNN layers (but don't exceed the global structure threshold).

---

## The Math (Copy-Paste Ready)

### Autoregressive flow (AF) prior
Sampling direction (each $z_i$ conditions on earlier *outputs*, so it is sequential):
$$ z_i = \mu_i(z_{<i}) + \sigma_i(z_{<i}) \cdot \epsilon_i, \qquad \epsilon \sim \mathcal{N}(0, I) $$
Density direction (parallel, what the ELBO needs): $\epsilon_i = (z_i - \mu_i(z_{<i})) / \sigma_i(z_{<i})$ and
$$ \log p(z) = \log \mathcal{N}(\epsilon; 0, I) + \log\left|\det \frac{d\epsilon}{dz}\right| = \log \mathcal{N}(\epsilon; 0, I) - \sum_i \log \sigma_i . $$
An IAF conditions on earlier *inputs* instead ($z_i = (y_i - \mu_i(y_{<i})) / \sigma_i(y_{<i})$ in the paper's notation, Sec. 3.2). The repo's `IAFBlock` parameterises the density direction directly, $\epsilon_i = (z_i - \mu_i)\,\sigma_i$, so its `log_det = sum(log(sigma))` enters with a plus sign.

### Free bits (Kingma et al. 2016, eq. 15; VLAE App. A)
$$ \tilde{\mathcal{L}}_\lambda = \mathbb{E}_{x \in M}\big[\mathbb{E}_{q(z|x)} \log p(x|z)\big] - \sum_{j=1}^{K} \max\big(\lambda,\ \mathbb{E}_{x \in M}[\mathrm{KL}(q(z_j|x)\,\|\,p(z_j))]\big) $$
$M$ is the minibatch, $j$ a group of latent units (here one latent dimension). Report the true KL, not the max.

### Gated Activation
```python
def gated_activation(x):
    f, g = torch.chunk(x, 2, dim=1)
    return torch.tanh(f) * torch.sigmoid(g)
```

---

## Visualization Snippets

### Plotting Reconstructions
```python
import matplotlib.pyplot as plt

def plot_recon(original, recon):
    n = 8
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(torch.sigmoid(recon[i]).cpu().detach().squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()
```

---

## Experiment Ideas

- **Easy (2/5):** Disable the Flow Prior (`use_flow=False`) and compare KL divergence to a `use_flow=True` run.
- **Medium (3/5):** Sweep `decoder_layers` from 1 to 15 with free bits on. Plot `KL` and the own-vs-other-$z$ gap vs `layers`. Information preference predicts less information in $z$ as the window grows (the paper shows this only qualitatively, Fig. 3).
- **Advanced (5/5):** Implement **PixelRNN** (using LSTMs) as the decoder instead of PixelCNN. Compare sampling speed vs quality.

---

## Success Criteria

1.  **$z$ is used, not just non-zero KL:** $-\log p(x|z)$ is clearly lower with each image's own $z$ than with another image's, and $I(x; z)$ is well above 0. The paper's converged model uses 13.3 nats = 19.2 bits per image (static MNIST, Sec. 4.1); the 200-batch CPU run reaches 6.8 nats with a 12-nat gap, one epoch 9.5 nats with a 35-nat gap. (This repo thresholds MNIST at 0.5; the paper binarizes stochastically, so nats are not directly comparable.)
2.  **Global Coherence:** Decompressions ($z \sim q(z|x)$, $x \sim p(x|z)$) keep the digit and its layout; samples from random $z$ should look like digits (7s, 1s, etc.), not just "furry noise", which needs far more training than the CPU runs here.
3.  **Sharp Edges:** Reconstructions should be sharper than a standard MLP-VAE (teacher-forced reconstructions always look sharp, so they prove little).

---

**Next:** [Day 24 — GPipe](../24_gpipe/)
