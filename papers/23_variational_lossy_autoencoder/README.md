# Day 23: Variational Lossy Autoencoder (VLAE)

> Xi Chen, Kingma, Salimans, et al. (2017) — [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731)

**Time:** 3-4 hours  
**Prerequisites:** VAE Essentials (Day 4/5), PixelCNN/RNN Intuition  
**Code:** PyTorch: masked-convolution decoder, autoregressive-flow (AF) prior, free bits

---

## What This Paper Is Actually About

This paper solves one of the most frustrating problems in deep generative modeling: **Posterior Collapse**. 

When you pair a Variational Autoencoder (VAE) with a powerful autoregressive decoder (like PixelCNN), the model often chooses the "lazy" path. The decoder becomes so good at predicting pixels from their immediate neighbors that it stops bothering to look at the global latent code $z$. The result? A model that ignores your input and generates generic output.

The paper first explains *why* (Sec. 2.2): read as a bits-back code, using $z$ costs extra nats whenever the encoder's $q(z|x)$ misses the true posterior, so if the decoder can model the data alone, the cheapest code ignores $z$. VLAE turns this "information preference" into a design tool: give the decoder only a small window of past pixels, and the information it cannot model locally (global structure) has to go through $z$, while the decoder handles local texture (Sec. 3.1). A learned **autoregressive-flow (AF) prior** shortens the code (Sec. 3.2). The paper calls the preference argument "an asymptotic view" and still trains with **free bits** (App. A) so that $z$ is not shut off early in training.

---

## The Core Idea

The information preference of a model depends on the capacity of its components. If the decoder can explain the data locally, it will.

```
KL (Latent Usage)
    ^
    |      * PEAK (VLAE)
    |     / \
    |    /   \
    |   /     \
    |  /       \_________________ COLLAPSE (PixelCNN VAE)
    +----------------------------------> Receptive Field Size
```

(An illustrative sketch, not a plot from the paper: its receptive-field comparison, Fig. 3, is qualitative CIFAR-10 decompressions with no KL numbers.)

VLAE operates at the **left side** of this curve. By keeping the receptive field small, the model *must* use $z$ to figure out where the major object boundaries are. In this repo the default decoder lets each pixel see 34 past pixels (rows i-4..i, with the PixelCNN blind spot); the paper's 6 masked 3x3 layers would see 69.

---

## What the Authors Actually Showed

1.  **Density estimation:** state of the art at the time on statically binarized MNIST (79.03 nats NLL, Table 1), dynamically binarized MNIST (78.53, Table 2), OMNIGLOT and Caltech-101. On CIFAR-10, 2.95 bits/dim was the best among latent-variable models, behind PixelCNN++'s 2.92 (Table 5).
2.  **Information Decomposition:** Decompressions ($z \sim q(z|x)$, then $x \sim p(x|z)$) keep the global shape and identity while the decoder redraws local details (Fig. 1a). The MNIST latent uses 13.3 nats = 19.2 bits per image, against 37.3 bits with a factorised decoder (Sec. 4.1).
3.  **Prior Flexibility:** An **autoregressive-flow (AF) prior** beats the equivalent IAF posterior: 79.30 vs 79.88 nats (Table 1). The paper shows the two are the same along the encoder path, but the AF prior gives a deeper generative path for free (eqs. 12-14).

---

## The Architecture

The VLAE consists of three "elite" components:

1.  **The Encoder $q(z|x)$:** A ResNet in the paper; two strided convolutions and a linear layer here.
2.  **The Flow Prior $p(z)$:** An autoregressive flow from Gaussian noise (paper: 4 mean-only steps of a 640-unit MADE, 64-d latent; here: 2 steps, 32-d). `IAFBlock` computes one step in the noise-ward direction $z \to \epsilon$, which is what $\log p(z)$ needs.
3.  **The Restricted Decoder $p(x|z)$:** A masked-convolution PixelCNN with a small receptive field. It receives $z$ as a 28x28 feature map (the paper: 28x28x4). The paper used a plain 6-layer masked PixelCNN with 12 filters; this repo uses 3 gated layers with 64 channels after a type-A layer.

---

## Implementation Notes

- **Autoregressive Property:** Enforced via **MaskedConv2d** (Type A and Type B masks).
- **Gated Units:** Uses `tanh * sigmoid` gating to avoid vanishing gradients in deep PixelCNNs.
- **Flow prior:** Uses **MADE** (Masked Autoencoder for Distribution Estimation) to compute mu/sigma parameters in parallel while maintaining autoregression. The density is $\log p(z) = \log \mathcal{N}(\epsilon; 0, I) + \log|\det d\epsilon/dz|$ (Sec. 3.2); `VLAE.sample_prior` runs the flow the other way, one dimension at a time.
- **Free bits** (`free_bits_loss`): the paper's App. A uses "0.01 nats/data-dim free bits". Free bits (Kingma et al. 2016, eq. 15) replaces the KL term by $\sum_j \max(\lambda, \mathbb{E}_{\text{minibatch}}[\mathrm{KL}_j])$ over groups $j$ of latent units, so pushing a group's KL below $\lambda$ earns nothing. Here each latent dimension is a group, and the budget 0.01 x 784 pixels = 7.84 nats per image is split evenly: $\lambda$ = 0.245 nats per dimension. With the AF prior the per-dimension KL uses the chain rule, $\log q(z_d|x) - \log p(z_d \mid z_{<d})$. Training reports the true KL and negative ELBO (no max) next to the objective.
- **Is $z$ used?** (`latent_usage`): the KL alone cannot tell. Free bits removes the penalty below $\lambda$, and the budget can be spent on shrinking $\sigma$ with every image mapped to the same Gaussian. So the check also compares $-\log p(x|z)$ with each image's own $z$ against another image's $z$, and estimates $I(x; z)$.

---

## What to Build

### Quick Start

```bash
# Verify the implementation
python setup.py

# Train the VLAE on binarized MNIST with free bits (200 batches, about 4.5 min on one CPU core)
python train_minimal.py

# The same run with the plain ELBO: posterior collapse
python train_minimal.py --no-free-bits

# Longer run (one full epoch, 938 batches, about 16 min), or with the learned AF prior
python train_minimal.py --max-batches 0 --eval-images 5000
python train_minimal.py --use-flow
```

### What the runs show

Test images (2,000; 5,000 for the epoch run), nats per image (seed 42, Gaussian prior, 3 decoder layers). The probe is a logistic regression reading the digit class from one sample $z \sim q(z|x)$.

| Run | KL to prior | $I(x;z)$ est. | $-\log p(x\|z)$ own $z$ | ... another image's $z$ | gap | digit probe | $-$ELBO |
|---|---|---|---|---|---|---|---|
| plain ELBO, 200 batches | 0.33 | 0.24 | 79.72 | 80.66 | 0.94 | 16.8 % | 80.05 |
| free bits, 200 batches | 6.76 | 2.53 | 74.80 | 86.93 | 12.13 | 49.0 % | 81.56 |
| free bits, 1 epoch (938 batches) | 9.47 | 5.33 | 59.40 | 94.42 | 35.02 | 76.7 % | 68.87 |

- The plain ELBO (nearly) collapses: the encoder starts near $q(z|x) = \mathcal{N}(0, I)$, the KL never rises above 0.4 nats, and the decoder gains under 1 nat from the right $z$. With the repo's original decoder (below) the KL stayed under 0.04 nats.
- With free bits, $z$ is used and carries global information (the digit class). The KL stays below the 7.84-nat budget this early: free bits only removes the penalty, it does not push information in.
- At 200 batches the free-bits model's bound is *worse* (81.56 vs 80.05): it pays 6.76 nats of KL for about 2.5 nats of information, the coding inefficiency of Sec. 2.2. The paper's claim is about the converged model (78.53 vs 87.55 nats for the decoder alone on dynamic MNIST, Table 2).
- Free bits was not enough with the repo's original decoder, which gave every pixel the same $z$ vector: a pixel on blank background cannot tell where it is from a 34-pixel window, so $z$ could not say where the strokes go. After 200 batches the KL was 4.7 nats while $I(x;z)$ was 0.05 and the gap 0.12: the budget went into shrinking $\sigma$. Feeding $z$ in as a 28x28 feature map, as the paper does (App. A), fixed this. `validation/day23_free_bits_check.py` reproduces both sides (150 batches: map 5.6-nat gap and $I(x;z)$ 1.75; with `DAY23_BROADCAST=1`, broadcast $z$: KL 4.3 nats, $I(x;z)$ 0.01).
- The notebook repeats the comparison with the learned AF prior (150 batches each): free bits KL 4.97, gap 2.48, $I(x;z)$ 0.95; plain ELBO KL 0.71 (from 29 nats at initialisation), gap 0.54, $I(x;z)$ 0.16.
- After one epoch the free-bits KL passes the budget (9.47 nats), the decoder gains 35 nats from the right $z$, the probe reads the digit 77 % of the time, and decompressions keep the digit and its layout (a 7, a 1, a 0 come back as the same digit with redrawn strokes): the paper's lossy code. There is no plain-ELBO run of that length to compare with.
- The paper's converged model uses 13.3 nats on static MNIST and reaches 79.03 nats NLL (importance-sampled). The nats here are **not comparable**: this repo binarizes MNIST by thresholding at 0.5, while the paper uses stochastically binarized MNIST, which has far more entropy (hence a $-$ELBO of 68.9 here, below the paper's NLL). The $-$ELBO is an upper bound on the NLL, and these CPU runs are far shorter.

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | `exercise_01_masked_conv.py` | Implement the Type A/B masking logic. |
| 2 | `exercise_02_pixelcnn_prior.py` | Build a standalone density estimator. |
| 3 | `exercise_03_posterior_collapse.py` | Reproduce collapse with the plain ELBO (KL stays near 0). |
| 4 | `exercise_04_flow_prior.py` | Implement a flow step (the exercise asks for a planar flow; the solution visualises the repo's MADE blocks). |
| 5 | `exercise_05_vlae_complete.py` | Final assembly with the AF prior and free bits; check that $z$ is used. |

---

## Key Takeaways

1.  **Limitation is a Feature:** Restricting a model's local expressive power forces it to learn better global representations.
2.  **Inductive Bias Matters:** Masking and receptive field control are powerful tools for controlling information flow.
3.  **VAEs + Flows = Power:** Normalizing flows remove the "Gaussian bottleneck" of standard VAEs.
4.  **The optimum is not the trajectory:** information preference says what $z$ should carry at the optimum; getting there still needs free bits (or KL annealing). And a KL above zero is not proof that $z$ is used: measure what the decoder gains from it.

---

## Files in This Directory

| File | What It Is |
|------|------------|
| `implementation.py` | The main PyTorch implementation of VLAE, the AF prior, free bits and the $z$-usage checks. |
| `train_minimal.py` | Training CLI: free bits by default, `--no-free-bits` for the collapse comparison. |
| `paper_notes.md` | Deep dive into the math and theoretical context. |
| `CHEATSHEET.md` | Quick reference, hyperparameters, and pro tips. |
| `visualization.py` | Suite for plotting KL curves and reconstructions. |
| `notebook.ipynb` | Interactive walkthrough and latent space exploration. |

---

## Further Reading

- [MADE: Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509) (Germain et al., 2015)
- [Conditional Image Generation with PixelCNN](https://arxiv.org/abs/1606.05328) (van den Oord et al., 2016)
- [Improved Variational Inference with IAF](https://arxiv.org/abs/1606.04934) (Kingma et al., 2016): IAF, and free bits in App. C.8

---

**Previous:** [Day 22-DeepSpeech2](../22_deep_speech_2/)
**Next:** [Day 24 — GPipe](../24_gpipe/)
