# Paper Notes: Variational Lossy Autoencoder

> Notes on Chen et al. (2017) ICLR paper "Variational Lossy Autoencoder"

---

## Post Overview

**Title:** Variational Lossy Autoencoder
**Authors:** Xi Chen, Diederik P. Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever, Pieter Abbeel
**Year:** 2017
**Source:** [arXiv:1611.02731](https://arxiv.org/abs/1611.02731)

**One-sentence summary:**
*"Explains posterior collapse as a bits-back coding preference, then uses it to choose what the latent learns: a decoder with a small receptive field leaves global information to the latent, a learned autoregressive-flow prior shortens the code, and free bits keeps the latent in use during training."*

---

## ELI5 (Explain Like I'm 5)

### The Story

Imagine you are a student trying to redraw a complex painting from a tiny, blurry Polaroid photo.

A **standard VAE** gives you the Polaroid (the latent code $z$) and asks you to draw the painting. If you're a bad artist (weak decoder), you look at the Polaroid for every detail.

A **VAE with a powerful decoder** (like PixelCNN) is like giving a master artist the Polaroid. The artist looks at the first pixel they drew, knows exactly what the next one should be because they've seen so many paintings, and eventually stops looking at the Polaroid entirely. They just draw a generic "beautiful painting" that has nothing to do with the specific one you wanted. This is **posterior collapse**.

**VLAE** is the fix. The teacher (the researcher) puts a small tube over the artist's eye so they can only see 5 pixels at a time. Now, the artist *cannot* draw a coherent painting just by looking at local neighbors. They are forced to look at the Polaroid (the latent $z$) to know where the big things like "heads" or "mountains" are.

> **Note:** This analogy is ours, not the authors'.

---

## What the Paper Actually Covers

The paper identifies a fundamental "information preference" problem in VAEs. Powerful autoregressive decoders (like PixelCNN or PixelRNN) are so expressive that they can model the data distribution $p(x)$ locally, making the latent code $z$ redundant. This leads to the KL divergence term in the ELBO being optimized to zero.

---

## The Core Idea (From the Paper)

### The Problem: Posterior Collapse
When the decoder $p(x|z)$ is powerful enough to model the data unconditionally ($p(x|z) \approx p(x)$), the model reaches a trivial solution where $q(z|x) \approx p(z)$. The model ignores $z$ because using a noisy latent variable is "harder" than local pixel prediction.

### The Solution: VLAE
1.  **Restrict the Decoder Receptive Field (Section 3.1):** Design $p(x|z)$ so it has a limited spatial window. It can handle local texture and noise but cannot capture global structure (e.g., the shape of a digit).
2.  **Flexible Flow Prior (Section 3.2):** Use an **autoregressive flow (AF)** from Gaussian noise for the prior $p(z)$. This reduces the "penalty" of the KL term by allowing the prior to better match the complicated aggregated posterior. The paper shows that an AF prior equals an IAF posterior along the encoder path (eqs. 12-14) but gives a deeper generative path at no extra cost; it is not an "IAF prior".
3.  **Free bits in training (App. A):** "0.01 nats/data-dim free bits (Kingma et al., 2016) was found to be effective in dealing with the problem of all the latent code being ignored early in training." Sec. 3.1 calls information preference "an asymptotic view" and says free bits and KL annealing "are still useful methods to smooth the optimization problem and used in this paper's experiments".

---

## The Math

### The Evidence Lower Bound (Eqs. 2-4)
The VAE objective is to maximize:
$$ \mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p_\theta(z)) $$

### Bits-Back Interpretation (Section 2.2)
The authors frame the ELBO as a coding cost.
- $-\log p(x|z)$ is the cost of encoding the image given the latent.
- $D_{KL}(q(z|x) || p(z))$ is the "information cost" of describing the latent.
VLAE forces the model to put "global bits" into the KL term by making $-\log p(x|z)$ high for global structures.

### Flow-Based Prior (Section 3.2, unnumbered)
By using a sequence of transformations $f_1, \dots, f_k$:
$$ z_k = f_k(z_{k-1}) $$
The log-likelihood of the latent $z$ becomes:
$$ \log p(z_k) = \log p(z_0) - \sum_{i=1}^k \log \left| \det \frac{\partial f_i}{\partial z_{i-1}} \right| $$
The paper writes the same thing from the other side: $\log p(z) = \log u(\epsilon) + \log\left|\det \frac{d\epsilon}{dz}\right|$ with $z = f(\epsilon)$ and noise density $u$. For an autoregressive flow, going from $z$ back to $\epsilon$ (the direction the ELBO needs) is one parallel MADE pass per step with a triangular Jacobian, so the determinant is a sum of log-diagonal terms. Sampling $z$ runs one dimension at a time.

### Free Bits (App. A; Kingma et al. 2016, App. C.8, eq. 15)
The latent units are split into $K$ groups and the KL term is replaced by
$$ \sum_{j=1}^{K} \max\big(\lambda,\ \mathbb{E}_{x \in \text{minibatch}}[D_{KL}(q(z_j|x)\,\|\,p(z_j))]\big), $$
so a group whose batch-average KL is below $\lambda$ costs a constant: the optimiser gains nothing by switching it off. The paper gives $\lambda$ per data dimension (0.01 nats x 784 pixels = 7.84 nats per image). This repo makes each latent dimension a group (the IAF paper's rule when no parameters are shared across dimensions) and splits the budget evenly, 0.245 nats per dimension. Under the AF prior, the per-dimension KL uses the chain rule, $\log q(z_d|x) - \log p(z_d \mid z_{<d})$. The paper's App. C also proposes a smoother "soft free bits" variant for CIFAR-10.

---

## The Experiments (Section 4)

### MNIST & CIFAR-10
The binary-image models use a PixelCNN decoder with 6 masked 3x3 layers, an AF prior and free bits; NLL is importance-sampled with 4096 samples.
- **Collapse with RNN decoders** is cited from earlier work (Bowman et al. 2015 and others), not re-measured in the paper.
- **Lossy code (Sec. 4.1):** on static MNIST the converged VLAE uses 13.3 nats = 19.2 bits per image in $z$, against 37.3 bits for the same VAE with a factorised decoder. Decompressions keep the global shape with new local details (Fig. 1a).
- **Density (Tables 1-5):** 79.03 nats on static MNIST (AF VAE 79.30, IAF VAE 79.88); 78.53 on dynamic MNIST against 87.55 for the same PixelCNN alone. On CIFAR-10, 2.95 bits/dim is the best among latent-variable models but behind PixelCNN++ (2.92).
- **Ablation (App. D):** PixelCNN alone 87.55 nats; + Gaussian latent 79.48 (KL 10.60); + AF prior 78.94 (KL 11.73).

### Visualizing Information
Fig. 3 shows CIFAR-10 decompressions for receptive fields of 4x2, 5x3 and 7x4: larger windows leave only rough shape in $z$. This is qualitative; the paper reports no KL-vs-receptive-field numbers.

---

## What the Paper Gets Right

- **Formalizes the "Information Preference"**: Provides a clear theoretical reason why collapse happens (Section 2.2).
- **Practical Fix**: Restricting the receptive field is a simple, elegant architectural constraint that works, together with free bits during training (App. A).
- **Improved Metrics**: Shows consistently better density estimation (ELBO) than standard VAEs.

## What the Paper Doesn't Cover

- **Sampling Speed**: The decoder is still autoregressive, meaning sampling an image is $O(N^2)$ (pixel-by-pixel), which is very slow.
- **Optimization Stability**: Training flows alongside VAEs can be tricky and requires careful initialization.
- **Comparison to Discretization**: VQ-VAE came a year later (Nov 2017 vs Nov 2016), so the paper could not compare to it -- VQ-VAE solves collapse via a different mechanism (vector quantization).

---

## What This Repo's Run Shows

> **[Our Addition]** `python train_minimal.py` vs `--no-free-bits`, 200 batches of 64 on one CPU core, test images, nats per image.

- **Plain ELBO:** KL to the prior 0.33, $I(x;z)$ estimate 0.24, and $-\log p(x|z)$ is the same whether the decoder gets the image's own $z$ or another image's (gap 0.94): posterior collapse.
- **Free bits:** KL 6.76 (budget 7.84), $I(x;z)$ 2.53, own-$z$ 74.80 vs another image's $z$ 86.93 (gap 12.1); a linear probe reads the digit class from one $z$ sample 49% of the time (chance 10%).
- **Free bits is not sufficient on its own.** When every pixel received the same $z$ vector (the repo's original decoder), free bits held the KL at 4.7 nats while $I(x;z)$ was 0.05: the free budget went into shrinking $\sigma$, with every image mapped to the same Gaussian. With a 34-pixel window and zero padding, a pixel on blank background cannot tell where on the canvas it is, so a broadcast $z$ cannot say where the strokes go. The paper's decoder receives $z$ as a 28x28x4 feature map (App. A); with a feature map (here 28x28x32), $z$ is used.
- **One epoch with free bits** (938 batches): KL 9.47, $I(x;z)$ 5.33, own-$z$ 59.4 vs another image's $z$ 94.4, probe 77%. Decompressions keep the digit and redraw its strokes, the paper's Fig. 1a behaviour.
- So: a KL above zero (or at the budget) is not evidence that $z$ is used. Measure what the decoder gains from it.
- These nats are not comparable with the paper's: the repo thresholds MNIST at 0.5 instead of binarizing it stochastically, which removes much of the per-pixel entropy.

---

## Looking Back (Our Retrospective)

> **[Our Addition: Retrospective - written 2024]**
VLAE was a key "stepping stone" in the generative model wars of 2016-2018. It highlighted that the bottleneck isn't just capacity, but **inductive bias**. Today, we see this principle everywhere: MAE (Masked Autoencoders) use masking to force a model to learn high-level representations. The idea that "limitation breeds representation" is now a core tenet of self-supervised learning.

---

## Questions Worth Thinking About

1. If we use a Transformer decoder (Global attention) instead of PixelCNN (local), can we still use the VLAE principle? (Hint: how would you "restrict" a Transformer?)
2. Why is a Gaussian prior $p(z) = \mathcal{N}(0, I)$ considered "restrictive"? How does a Flow prior make the KL term less punitive?
3. In a world with Diffusion Models (which are much faster to sample), is the VLAE architecture still relevant?
4. **[Our Addition]** Can you imagine a "receptive field" for audio or text? What would it mean to "cripple" a GPT-layer to force it to use a latent code?

---

**Next:** [Day 24 — GPipe](../24_gpipe/) - Scaling Giant Models.
