"""Day 23: autoregressive property of the VLAE decoder (masked convolutions, type A then B).

1. MaskedConv2d.forward == F.conv2d with an explicitly masked weight (max |d|).
2. With z held fixed, the full Jacobian d logits[i,j] / d x[k,l] of the decoder path
   (initial_conv -> GatedPixelCNNLayer x n_layers -> final_conv) must be exactly zero for every
   input pixel at or after (i,j) in raster order. Checked for n_layers=3 (default) and 8, both
   at init and after one Adam step (the masked weights become non-zero in .data after a step;
   forward() must re-mask them).
3. Receptive field of the centre pixel (14,14): which past pixels have non-zero gradient.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '23_variational_lossy_autoencoder')
sys.path.insert(0, DAY)
from implementation import MaskedConv2d, VLAE  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(1)
ok = True
H = W = 28

# ---- 1. MaskedConv2d vs explicit masked conv
for t in ('A', 'B'):
    conv = MaskedConv2d(t, 3, 5, 3, padding=1)
    xin = torch.randn(2, 3, 9, 9)
    w_raw = conv.weight.detach().clone()
    out = conv(xin)
    ref = F.conv2d(xin, w_raw * conv.mask, conv.bias, padding=1)
    d = (out - ref).abs().max().item()
    print(f"[mask {t}] mask[0,0]=\n{conv.mask[0, 0].int().tolist()}  |MaskedConv2d - F.conv2d(W*mask)| = {d:.1e}")
    ok &= d < 1e-6


def decoder(model, x, h_latent):
    h = model.initial_conv(x)
    for layer in model.decoder_layers:
        h = layer(h, h_latent)
    return model.final_conv(h)


def jacobian(model, x1, h1, chunk=49):
    """J[p, q] = d logits_flat[p] / d x_flat[q] for one image, via a batch of copies:
    copy b only back-propagates output pixel p_b, so one backward gives 'chunk' rows."""
    J = torch.zeros(H * W, H * W)
    for s in range(0, H * W, chunk):
        idx = torch.arange(s, min(s + chunk, H * W))
        xb = x1.repeat(len(idx), 1, 1, 1).requires_grad_(True)      # (B,1,28,28)
        hb = h1.repeat(len(idx), 1, 1, 1)
        out = decoder(model, xb, hb).reshape(len(idx), -1)          # (B, 784)
        out[torch.arange(len(idx)), idx].sum().backward()
        J[idx] = xb.grad.reshape(len(idx), -1)
    return J


future = torch.triu(torch.ones(H * W, H * W), diagonal=0).bool()  # q >= p in raster order
for n_layers in (3, 8):
    model = VLAE(input_dim=1, latent_dim=32, n_layers=n_layers, use_flow=False)
    x1 = (torch.rand(1, 1, H, W) > 0.5).float()
    h1 = torch.randn(1, 32, 1, 1).expand(-1, -1, H, W)
    for phase in ('init', 'after Adam step'):
        if phase == 'after Adam step':
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            logits = decoder(model, x1, h1)
            F.binary_cross_entropy_with_logits(logits, x1).backward()
            opt.step()
            nz = (model.initial_conv.weight.data * (1 - model.initial_conv.mask)).abs().max().item()
            print(f"   masked weights in .data after the step (re-masked at next forward): max |w| = {nz:.2e}")
        J = jacobian(model, x1, h1)
        leak = J[future].abs().max().item()
        rows, cols = np.nonzero(J[14 * W + 14].reshape(H, W).numpy())  # receptive field of (14,14)
        print(f"[n_layers={n_layers}, {phase}] max |d logit_p / d x_q| over q >= p: {leak:.1e}; "
              f"RF of (14,14): rows {rows.min()}..{rows.max()}, cols {cols.min()}..{cols.max()}, "
              f"{len(rows)} pixels")
        ok &= leak == 0.0
    # receptive field picture for the centre pixel (last phase): '#' depends, 'o' = the pixel
    if n_layers == 3:
        grid = (J[14 * W + 14].reshape(H, W) != 0).numpy()
        print("   RF map rows 8..15, cols 8..21 ('#' = dependency, 'o' = predicted pixel):")
        for r in range(8, 16):
            print("   " + "".join('o' if (r, c) == (14, 14) else ('#' if grid[r, c] else '.')
                               for c in range(8, 22)))
print("PASS" if ok else "FAIL")
