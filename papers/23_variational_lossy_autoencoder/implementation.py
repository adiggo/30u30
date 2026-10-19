"""
Variational Lossy Autoencoder (VLAE): Bridging VAEs and Autoregressive Models

Implements the core ideas from Chen et al. "Variational Lossy Autoencoder" (ICLR 2017):

1. Masked Convolutions (Type A/B) to enforce autoregressive property.
2. Gated Activation Units (tanh/sigmoid) for powerful local modeling.
3. A PixelCNN-style decoder with a restricted receptive field.
4. A learned autoregressive-flow (AF) prior p(z) (Sec. 3.2), built from MADE blocks.
5. Evidence Lower Bound (ELBO) objective with bits-back interpretation.
6. Free bits (Kingma et al. 2016), which the paper trains with (App. A), plus diagnostics
   that measure whether the decoder actually uses z.

[PATCH 2026-10-18] The central demonstration: a decoder that only sees a small window of past
pixels cannot model global structure (digit identity, stroke layout), so that information has
to go through z while the decoder models local texture: a lossy code (Sec. 3.1). The paper
calls this preference "an asymptotic view" and still trains with free bits (App. A). Here,
too, the plain ELBO collapses (KL -> ~0 nats, z unused); free bits keeps z in use (see
free_bits_loss and latent_usage below).

Reference: https://arxiv.org/abs/1611.02731
Author: 30u30 Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional

LOG_2PI = float(np.log(2 * np.pi))  # [PATCH 2026-10-18] shared by the Gaussian log-densities below


# ============================================================================
# SECTION 1: MASKED CONVOLUTIONS & PIXELCNN COMPONENTS
# ============================================================================
# The VLAE decoder is a Gated PixelCNN. Autoregression is enforced via 
# spatial masking. Type A masks the center pixel (first layer), Type B 
# includes it (subsequent layers).
# ============================================================================

class MaskedConv2d(nn.Conv2d):
    """
    Convolution with a masked weight matrix to enforce autoregressive property.
    Pixels can only see previous pixels (top-to-bottom, left-to-right).
    """
    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kH, kW = self.weight.size()
        
        # Mask out the future (center + right, and all rows below)
        # Assuming kernel has odd size, center is at kH // 2, kW // 2
        
        # 1. Mask rows below center
        self.mask[:, :, kH // 2 + 1:, :] = 0
        # 2. Mask future columns in center row
        self.mask[:, :, kH // 2, kW // 2 + 1:] = 0
        
        # Type A: cannot see itself (center pixel masked) -> used in first layer
        # Type B: can see itself (reusing its own pixel from previous layer)
        if mask_type == 'A':
            self.mask[:, :, kH // 2, kW // 2] = 0
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask
        return super().forward(x)


class GatedActivation(nn.Module):
    """
    Gated PixelCNN activation: y = tanh(W_f * x) * sigmoid(W_g * x).
    Ref: van den Oord et al. (2016) "Conditional Image Generation with PixelCNN Decoders".
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f, g = torch.chunk(x, 2, dim=1)
        return torch.tanh(f) * torch.sigmoid(g)


class GatedPixelCNNLayer(nn.Module):
    """
    One masked 3x3 conv + tanh*sigmoid gate + 1x1 projection + residual; the latent enters every
    layer through a 1x1 projection of h, the z feature map (VLAE.decode).

    [PATCH 2026-10-18] This is the repo's own design, not the paper's: VLAE's MNIST decoder
    (App. A) is a plain masked PixelCNN (6 masked 3x3 layers with 12 filters in ResNet blocks,
    weight-tied, ELU, weight norm) fed with the image concatenated to a 28x28x4 feature map
    computed from z. The gate here has no vertical/horizontal stacks (van den Oord et al. 2016b),
    so the PixelCNN blind spot remains.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 mask_type: str, conditional_dim: int = 0):
        super().__init__()
        self.mask_type = mask_type
        # Double output channels for the gated activation split (f, g)
        self.conv = MaskedConv2d(mask_type, in_channels, out_channels * 2, 
                                kernel_size, padding=kernel_size // 2)
        
        # Condition from latent z (upsampled)
        if conditional_dim > 0:
            self.cond_proj = nn.Conv2d(conditional_dim, out_channels * 2, kernel_size=1)
        else:
            self.cond_proj = None
            
        self.activation = GatedActivation()
        self.out_proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_in = x
        x = self.conv(x)
        
        if h is not None and self.cond_proj is not None:
            x = x + self.cond_proj(h)
            
        x = self.activation(x)
        x = self.out_proj(x)
        
        # Residual connection (only for Type B layers where channels match)
        if self.mask_type == 'B' and x.shape == x_in.shape:
            x = x + x_in
            
        return x


# ============================================================================
# SECTION 2: AUTOREGRESSIVE FLOW PRIOR (MADE BLOCKS)
# ============================================================================
# [PATCH 2026-10-18] VLAE's prior is an autoregressive flow (AF) from Gaussian noise eps to z
# (Sec. 3.2); the paper shows it equals an IAF posterior along the encoder path (eqs. 12-14).
# Each IAFBlock below computes one step of the inverse map z -> eps in parallel, which is what
# the ELBO needs: log p(z) = log N(eps; 0, I) + log|det d eps / dz|. Sampling z ~ p(z) runs the
# AF direction, one dimension at a time (VLAE.flow_inverse).
# The autoregressive structure is implemented with MADE (Germain et al. 2015).
# ============================================================================

class MaskedLinear(nn.Linear):
    """Linear layer with a mask to preserve autoregressive property in MADE."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones_like(self.weight))

    def set_mask(self, mask: torch.Tensor):
        self.mask.data.copy_(mask.t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation.
    Used as the coupling layer for IAF.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            MaskedLinear(input_dim, hidden_dim),
            nn.ReLU(),
            MaskedLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            MaskedLinear(hidden_dim, output_dim)
        )
        self.m = {}
        self.create_masks()

    def create_masks(self):
        """Standard MADE mask generation logic."""
        # Find linear layers (skipping ReLUs)
        layers = [m for m in self.net if isinstance(m, MaskedLinear)]
        L = len(layers) - 1
        
        # 1. Assign ranks to hidden units
        self.m[-1] = np.arange(self.input_dim)
        for l in range(L):
            # Each hidden unit gets a rank between 0 and D-2
            self.m[l] = np.random.randint(0, self.input_dim - 1, size=layers[l].out_features)
        
        # 2. Assign ranks to output units (must be 1:1 or N:1 mapping to input dim)
        # [PATCH 2026-10-18] IAFBlock splits the outputs with torch.chunk into contiguous halves
        # (mu, log_sigma). np.repeat gave degrees [0,0,1,1,...], so log_sigma_i had degree
        # (D+i)//2 and depended on z_j for j >= i: Jacobian not triangular, sum(log sigma) not the
        # log-determinant. np.tile gives each half the degrees [0..D-1].
        self.m[L] = np.tile(np.arange(self.input_dim), layers[L].out_features // self.input_dim)

        # 3. Apply masks to each layer
        layer_idx = 0
        for m in self.net:
            if isinstance(m, MaskedLinear):
                if layer_idx < L:
                    # Hidden layers: rank_prev <= rank_curr
                    mask = self.m[layer_idx-1][:, None] <= self.m[layer_idx][None, :]
                else:
                    # Output layer: rank_prev < rank_curr (STRICT for autoregression)
                    mask = self.m[layer_idx-1][:, None] < self.m[layer_idx][None, :]
                
                m.set_mask(torch.from_numpy(mask.astype(np.float32)))
                layer_idx += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IAFBlock(nn.Module):
    """
    One autoregressive step z -> z_new with z_new_i = (z_i - mu_i(z_<i)) * sigma_i(z_<i).

    [PATCH 2026-10-18] In VLAE this is the inverse (noise-ward) direction of one step of the AF
    prior: it maps a latent z towards the noise eps in one parallel pass, so log p(z) can be
    evaluated (Sec. 3.2). Its Jacobian is lower triangular with diagonal sigma, so
    log|det| = sum_i log sigma_i (checked in validation/day23_kl_and_flow_prior.py).
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.made = MADE(dim, hidden_dim, dim * 2)

    def forward_terms(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[PATCH 2026-10-18] Same transform as forward(), but returns log sigma per dimension,
        shape (B, D), instead of its sum. Free bits needs the log-density split by dimension."""
        stats = self.made(z)
        mu, log_sigma = torch.chunk(stats, 2, dim=1)
        sigma = torch.sigmoid(log_sigma) # Use sigmoid for stability in IAF

        # Transformations: z_new = (z - mu) * sigma (the z -> eps direction of the AF prior)
        z_new = (z - mu) * sigma
        return z_new, torch.log(sigma)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_new, log_sigma = self.forward_terms(z)
        # Log determinant of the Jacobian (triangular, diagonal sigma)
        log_det = torch.sum(log_sigma, dim=1)
        return z_new, log_det


# ============================================================================
# SECTION 3: ENCODER & COMPLETE VLAE MODEL
# ============================================================================
# VLAE combines a standard VAE encoder, the IAF prior, and the restricted
# PixelCNN decoder.
# ============================================================================

class Encoder(nn.Module):
    """ResNet-style encoder for VLAE."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VLAE(nn.Module):
    """
    Variational Lossy Autoencoder: encoder q(z|x), optional AF prior p(z), local PixelCNN decoder.

    [PATCH 2026-10-18] The small receptive field decides *what* z should carry (global
    structure), but on its own it does not stop posterior collapse in training: train with
    free_bits_loss (the paper's free bits, App. A) rather than the plain loss_function. The
    decoder receives z as a spatial feature map (z_to_map), as in the paper.
    """
    def __init__(self, input_dim: int = 1, latent_dim: int = 32, 
                 hidden_dim: int = 64, n_layers: int = 3, use_flow: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_flow = use_flow
        
        self.encoder = Encoder(input_dim, latent_dim)
        
        # Restricted Receptive Field Decoder
        # [PATCH 2026-10-18] was "(7x7)". One type-A then n_layers type-B 3x3 masked convs: with
        # n_layers=3 each pixel sees 34 past pixels (rows i-4..i, 9/8/7/6/4 per row, with the
        # PixelCNN blind spot), of the up to 783 before it. The paper's 6 masked layers
        # (n_layers=5 here) would see 69 (rows i-6..i).
        self.initial_conv = MaskedConv2d('A', input_dim, hidden_dim, 3, padding=1)
        self.decoder_layers = nn.ModuleList([
            GatedPixelCNNLayer(hidden_dim, hidden_dim, 3, 'B', conditional_dim=latent_dim)
            for _ in range(n_layers)
        ])
        self.final_conv = nn.Conv2d(hidden_dim, input_dim, 1)
        
        # [PATCH 2026-10-18] z reaches the decoder as a 28x28 feature map, as in the paper (App. A:
        # the VAE decoder "outputs a 28x28x4 spatial feature map" that the masked PixelCNN conditions
        # on). The repo used to give every pixel the same z vector. With a 34-pixel window and zero
        # padding, a pixel on blank background then cannot tell where on the canvas it is, so z
        # cannot say where the strokes go: with free bits the KL budget went into shrinking
        # sigma instead (KL 4.7 nats per image but I(x; z) ~0.05 after 200 batches). This
        # upsampling network mirrors the encoder (FC, then two stride-2 transposed convs; ELU as
        # in the paper). It has latent_dim channels (paper: 4), so GatedPixelCNNLayer keeps
        # conditional_dim=latent_dim. The map depends only on z, so every pixel may see all of
        # it; the masks still keep x autoregressive.
        self.z_to_map = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7), nn.ELU(),
            nn.Unflatten(1, (64, 7, 7)),                                    # (B, 3136) -> (B, 64, 7, 7)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ELU(),   # -> (B, 32, 14, 14)
            nn.ConvTranspose2d(32, latent_dim, 4, stride=2, padding=1),     # -> (B, latent_dim, 28, 28)
        )

        # Flow Prior Steps
        if use_flow:
            self.flow_steps = nn.ModuleList([
                IAFBlock(latent_dim, 128) for _ in range(2)
            ])

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Encode
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # 2. Prior Flow (for log p(z) calculation)
        log_det_total = torch.zeros(z.size(0), device=z.device)
        z_flow = z
        if self.use_flow:
            for flow in self.flow_steps:
                z_flow, log_det = flow(z_flow)
                log_det_total += log_det
            # [PATCH 2026-10-18] AF prior (paper Sec. 3.2): the flow maps z to eps = z_flow, so
            # log p(z) = log N(z_flow; 0, I) + log|det dz_flow/dz| = log N(z_flow) + sum log sigma.
            # loss_function previously used log N(z) - sum log sigma: not a normalised density, and
            # unbounded below as sigma -> 0. The 5th output is now the flow's correction to
            # log N(z; 0, I) (zero without flow), which loss_function adds.
            log_det_total = log_det_total - 0.5 * (z_flow ** 2).sum(1) + 0.5 * (z ** 2).sum(1)

        # 3. Decode with restricted receptive field
        logits = self.decode(x, z)
        return logits, mu, logvar, z, log_det_total

    # [PATCH 2026-10-18] decode / log_prior_terms / flow_inverse / sample_prior: the decoder path
    # as its own method, so a latent can be swapped in (latent_usage) or sampled from the prior
    # (visualization); and log p(z) split by dimension for free bits.
    def decode(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Logits of p(x|z), one per pixel. Pixel (i, j) sees the z feature map and its masked
        window of x. x may be a top crop (rows 0..i) of the canvas, as in pixel-by-pixel sampling."""
        # (B, D) -> (B, D, 28, 28), cropped to x's height and width
        h_latent = self.z_to_map(z)[:, :, :x.size(2), :x.size(3)]
        h = self.initial_conv(x)
        for layer in self.decoder_layers:
            h = layer(h, h_latent)
        return self.final_conv(h)

    def log_prior_terms(self, z: torch.Tensor) -> torch.Tensor:
        """Per-dimension terms of log p(z), shape (B, D); they sum to log p(z).

        No flow: log N(z_d; 0, 1). AF prior: the map z -> eps (all steps) is triangular, so
        log p(z) = sum_d [log N(eps_d; 0, 1) + sum_steps log sigma_d] and the d-th term is the
        conditional log p(z_d | z_<d) (chain rule). Its sum equals log N(z; 0, I) plus the
        correction forward() returns (checked in validation/day23_free_bits_check.py).
        """
        if not self.use_flow:
            return -0.5 * (LOG_2PI + z ** 2)
        z_flow, log_sigma_total = z, torch.zeros_like(z)
        for flow in self.flow_steps:
            z_flow, log_sigma = flow.forward_terms(z_flow)
            log_sigma_total = log_sigma_total + log_sigma        # (B, D): diagonal of the Jacobian, per step
        return -0.5 * (LOG_2PI + z_flow ** 2) + log_sigma_total

    @torch.no_grad()
    def flow_inverse(self, eps: torch.Tensor) -> torch.Tensor:
        """AF direction eps -> z: undo the flow steps in reverse order. Each step needs D sequential
        MADE passes: z_d = y_d / sigma_d(z_<d) + mu_d(z_<d), and mu_d, sigma_d only see z_<d."""
        y = eps
        for flow in reversed(self.flow_steps):
            u = torch.zeros_like(y)
            for d in range(self.latent_dim):
                mu, log_sigma = torch.chunk(flow.made(u), 2, dim=1)
                u = u.clone()                                   # keep earlier values; overwrite column d
                u[:, d] = y[:, d] / torch.sigmoid(log_sigma[:, d]) + mu[:, d]
            y = u
        return y

    @torch.no_grad()
    def sample_prior(self, n: int) -> torch.Tensor:
        """n samples z ~ p(z): N(0, I) without the flow, the learned AF prior with it."""
        eps = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        return self.flow_inverse(eps) if self.use_flow else eps


def loss_function(logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                  logvar: torch.Tensor, z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
    """
    VLAE ELBO Loss Function (negative ELBO, nats, summed over the batch).
    Equivalent to bits-back coding cost: Cost = Reconstruction + KL

    [PATCH 2026-10-18] This is the plain objective, which collapses in training (KL -> ~0).
    For training use free_bits_loss (the paper's setting); with free_bits=0 it equals this.
    """
    # 1. Reconstruction Loss (Binary Cross Entropy for binarized MNIST)
    recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')
    
    # 2. KL Divergence: log q(z|x) - log p(z)
    # log q(z|x) = sum_i log N(z_i; mu_i, var_i)
    log_q = -0.5 * torch.sum(np.log(2*np.pi) + logvar + torch.pow(z - mu, 2) / torch.exp(logvar))
    
    # log p(z) with Flow (Jacobian change of variables)
    # Start with base log p(z_base) = N(0, I)
    # log p(z) = log p(base) - log |det Jacobian|
    log_p_base = -0.5 * torch.sum(np.log(2*np.pi) + torch.pow(z, 2))
    # [PATCH 2026-10-18] log_det is the flow-prior correction returned by VLAE.forward
    # (log p(z) - log N(z; 0, I)); it is added, not subtracted.
    log_p = log_p_base + log_det.sum()
    
    kl_loss = (log_q - log_p).sum()
    
    return recon_loss + kl_loss


# ============================================================================
# SECTION 4: FREE BITS AND LATENT-USAGE DIAGNOSTICS
# ============================================================================
# [PATCH 2026-10-18] Added. Without free bits the repo's training collapsed (KL -> ~0 nats).
#
# Paper, App. A: "0.01 nats/data-dim free bits (Kingma et al., 2016) was found to be effective
# in dealing with the problem of all the latent code being ignored early in training."
# Free bits (Kingma et al. 2016, App. C.8, eq. 15; restated in VLAE App. C) splits the latent
# units into K groups and replaces the KL term of the negative ELBO by
#
#     sum_{j=1..K} max(lambda, E_{x in minibatch}[ KL(q(z_j|x) || p(z_j)) ])
#
# A group whose batch-average KL is below lambda costs the constant lambda, so the optimiser
# gains nothing by pushing it lower and the reconstruction term may put up to lambda nats into
# it for free. Above lambda the term is the ordinary KL.
#
# Grouping and lambda used here (the paper does not spell out the MNIST grouping):
#   * Groups: IAF App. C.8 uses "the latent feature maps, or individual dimensions if no
#     parameter are shared across dimensions". Our z is a 32-d vector from a Linear layer, so
#     each latent dimension is a group (K = latent_dim). (openai/iaf: one group per latent
#     feature map, its KL summed over positions.)
#   * Lambda: App. A states it per *data* dimension: 0.01 nats x 784 pixels = 7.84 nats per
#     image. We split that budget evenly over the K groups: 7.84 / 32 = 0.245 nats per latent
#     dimension (the paper's 64-d latent would get 0.1225 each).
#   * The AF prior is not factorised, so the per-dimension KL uses the chain rule:
#     KL(q(z|x) || p(z)) = sum_d E_q[log q(z_d|x) - log p(z_d | z_<d)]  (q is factorised).
#   * As in loss_function (and openai/iaf), each KL is a single-sample estimate, z ~ q(z|x).
# Reported KL / ELBO / bits-per-dim must use the true KL (no max): free_bits_loss returns it
# separately and train_minimal.py prints it next to the training objective.
# ============================================================================


def kl_terms(mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor,
             log_p_terms: torch.Tensor) -> torch.Tensor:
    """Single-sample KL estimate per latent dimension, shape (B, D):
    log q(z_d|x) - log p(z_d | z_<d), with log_p_terms from VLAE.log_prior_terms. Summed over d it
    equals loss_function's KL term for the same z; its mean over z ~ q(z|x) is KL(q(z|x) || p(z)).
    """
    log_q = -0.5 * (LOG_2PI + logvar + (z - mu) ** 2 / torch.exp(logvar))   # (B, D), elementwise
    return log_q - log_p_terms


def free_bits_kl(kl: torch.Tensor, lam: float, n_groups: Optional[int] = None) -> torch.Tensor:
    """Free-bits KL term (Kingma et al. 2016, eq. 15), times the batch size B so it adds to batch
    sums like loss_function's: B * sum_j max(lam, mean_b KL_bj).

    kl: (B, D) per-dimension KL estimates (kl_terms). lam: nats per group. n_groups: K, default D
    (one group per latent dimension); group j holds dimensions [j*D/K, (j+1)*D/K).
    A group whose batch-average KL is below lam adds the constant B*lam (zero gradient); above lam
    it adds sum_b KL_bj, exactly the plain KL's share.
    """
    B, D = kl.shape
    K = D if n_groups is None else n_groups
    assert D % K == 0, f"latent dim {D} does not split into {K} equal groups"
    kl_groups = kl.reshape(B, K, D // K).sum(2)   # (B, D) -> (B, K, D/K) -> sum -> (B, K)
    kl_batch = kl_groups.mean(0)                   # (K,): E_{x in minibatch}[KL_j]
    # clamp(., min=lam) = max(lam, .): gradient 0 where the batch average is below lam, 1 above
    return B * torch.clamp(kl_batch, min=lam).sum()


def free_bits_loss(model: 'VLAE', logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
                   logvar: torch.Tensor, z: torch.Tensor, free_bits: float = 0.01,
                   n_groups: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Training objective with free bits (paper App. A), summed over the batch like loss_function.

    free_bits: lambda in nats per data dimension (paper: 0.01). 0 gives the plain negative ELBO,
    equal to loss_function(logits, x, mu, logvar, z, log_det) for the same forward pass.
    Returns (objective, parts): backpropagate the objective; report parts['recon'] + parts['kl'],
    the true negative ELBO. parts (floats): recon and kl (true KL, no max) as batch sums in nats,
    kl_objective (the free-bits KL term), lambda_per_group, n_groups.
    """
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')   # -log p(x|z), summed
    kl = kl_terms(mu, logvar, z, model.log_prior_terms(z))                   # (B, D)
    K = kl.shape[1] if n_groups is None else n_groups
    lam = free_bits * x[0].numel() / K      # nats per group: 0.01 nats x 784 data dims / K groups
    kl_obj = free_bits_kl(kl, lam, K) if free_bits > 0 else kl.sum()
    parts = {'recon': recon.item(), 'kl': kl.sum().item(), 'kl_objective': kl_obj.item(),
             'lambda_per_group': lam, 'n_groups': K}
    return recon + kl_obj, parts


@torch.no_grad()
def mutual_information(mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor) -> float:
    """Estimate of I_q(x; z) = E_x E_{q(z|x)}[log q(z|x) - log q(z)] over the N images given:
    how much one z ~ q(z|x) tells about which image it came from. q(z) is the aggregate
    posterior mean_m q(z|x_m) over the same N images. Since E_x KL(q(z|x) || p(z)) =
    I_q(x; z) + KL(q(z) || p(z)) (Hoffman & Johnson 2016, "ELBO surgery"), a KL can sit at the
    free-bits budget while I_q stays ~0 (every image mapped to the same shifted Gaussian); this
    tells the two apart. mu, logvar, z: (N, D), one z sample per image. The estimate is biased
    and can never exceed log N.
    """
    mu, logvar, z = mu.double(), logvar.double(), z.double()
    N = mu.size(0)
    inv_var = torch.exp(-logvar)                                              # (N, D)
    log_q_own = (-0.5 * (LOG_2PI + logvar + (z - mu) ** 2 * inv_var)).sum(1)  # (N,): log q(z_n|x_n)
    # log q(z_n | x_m) for every pair (n, m) without building an (N, N, D) tensor: expand the square,
    # sum_d (z_nd - mu_md)^2 / var_md = sum_d z_nd^2/var_md - 2 sum_d z_nd mu_md/var_md + sum_d mu_md^2/var_md
    const = (LOG_2PI + logvar + mu ** 2 * inv_var).sum(1)                     # (N,): one value per image m
    log_q_agg = []
    for s in range(0, N, 1000):
        zc = z[s:s + 1000]                                                    # (c, D) samples
        quad = (zc ** 2) @ inv_var.T - 2 * zc @ (mu * inv_var).T              # (c, D) @ (D, N) -> (c, N)
        log_q_pairs = -0.5 * (quad + const[None, :])                          # (c, N): log q(z_n | x_m)
        log_q_agg.append(torch.logsumexp(log_q_pairs, dim=1) - np.log(N))     # (c,): log mean_m q(z_n|x_m)
    return (log_q_own - torch.cat(log_q_agg)).mean().item()


@torch.no_grad()
def latent_usage(model: 'VLAE', x: torch.Tensor, batch_size: int = 250,
                 generator: Optional[torch.Generator] = None) -> Dict[str, float]:
    """Does the decoder use z? Averages per image, in nats, over the images x (N, 1, H, W).

    kl            true KL(q(z|x) || p(z)), single-sample estimate (no free-bits max)
    nll_own       -log p(x|z) with z ~ q(z|x) of the same image (the ELBO's reconstruction term)
    nll_shuffled  -log p(x|z) with the z of another image (shuffled across the N images)
    nll_prior     -log p(x|z) with z ~ p(z)
    gap           nll_shuffled - nll_own: nats the decoder gains from the image's own code. ~0 when
                  z is ignored (posterior collapse). A diagnostic, not a bound on I(x; z).
    mi            mutual_information estimate of I_q(x; z) (at most log N, given as mi_cap)
    neg_elbo      nll_own + kl, an upper bound on -log p(x); bpd is the same in bits per pixel
    """
    was_training = model.training
    model.eval()
    N = x.size(0)
    stats = [model.encoder(x[s:s + batch_size]) for s in range(0, N, batch_size)]
    mu = torch.cat([m for m, _ in stats])                     # (N, D)
    logvar = torch.cat([lv for _, lv in stats])               # (N, D)
    z = model.reparameterize(mu, logvar)                      # (N, D): one sample per image
    kl = kl_terms(mu, logvar, z, model.log_prior_terms(z)).sum(1)   # (N,)
    perm = torch.randperm(N, generator=generator)
    src = torch.empty_like(perm)
    src[perm] = perm.roll(1)        # image perm[i] gets the code of image perm[i-1]: never its own

    def nll(codes: torch.Tensor) -> torch.Tensor:             # (N,): -log p(x_n | codes_n), nats
        out = []
        for s in range(0, N, batch_size):
            xb = x[s:s + batch_size]
            bce = F.binary_cross_entropy_with_logits(model.decode(xb, codes[s:s + batch_size]), xb,
                                                     reduction='none')       # (b, 1, H, W)
            out.append(bce.flatten(1).sum(1))
        return torch.cat(out)

    nll_own, nll_shuffled, nll_prior = nll(z), nll(z[src]), nll(model.sample_prior(N))
    model.train(was_training)
    n_pixels = x[0].numel()
    res = {'n_images': N, 'kl': kl.mean().item(), 'nll_own': nll_own.mean().item(),
           'nll_shuffled': nll_shuffled.mean().item(), 'nll_prior': nll_prior.mean().item(),
           'mi': mutual_information(mu, logvar, z), 'mi_cap': float(np.log(N))}
    res['gap'] = res['nll_shuffled'] - res['nll_own']
    res['neg_elbo'] = res['nll_own'] + res['kl']
    res['bpd'] = float(res['neg_elbo'] / (n_pixels * np.log(2)))
    return res


def linear_probe_accuracy(feats_train: torch.Tensor, y_train: torch.Tensor,
                          feats_test: torch.Tensor, y_test: torch.Tensor, steps: int = 200) -> float:
    """Test accuracy of a multinomial logistic regression trained on frozen features (N, D).
    Fed one z ~ q(z|x) per image, it asks whether z carries the digit's identity, a global
    property a small-window decoder cannot infer early in the raster scan. Use samples rather than
    the means mu(x): a collapsed encoder's mu still varies slightly with x, but the noise in z
    hides those variations, so only information that is actually transmitted counts.
    """
    mean, std = feats_train.mean(0), feats_train.std(0) + 1e-8   # (D,) each: standardise features
    a, b = (feats_train - mean) / std, (feats_test - mean) / std
    n_classes = int(y_train.max()) + 1
    W = torch.zeros(a.size(1), n_classes, requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, bias], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(a @ W + bias, y_train) + 1e-4 * (W ** 2).sum()   # (N, D) @ (D, C)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return ((b @ W + bias).argmax(1) == y_test).float().mean().item()
