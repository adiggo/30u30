# Day 17 Cheat Sheet: Neural Turing Machines (NTM)

Quick reference for training and monitoring your NTM.

---

## Quick Start

```bash
# Train on the standard Copy Task (~5 min on one CPU core; add --paper for 10,001 iterations)
python train_minimal.py

# Visualize results (requires a trained model)
# python visualization.py --checkpoint ntm_copy_final.pth
```

---

## Key Hyperparameters (Graves et al., 2014)

| Parameter | Typical Value | What It Does | Rationale (from Paper) |
|-----------|---------------|--------------|-------------------------|
| `memory_n` | 128 | Number of addressable slots | Scale independently of weights (Sec 3) |
| `memory_m` | 20 | Dimension of each slot | Enough precision for bitmask storage |
| `controller_h` | 100 | Size of RNN hidden state | The "thinking" capacity (Sec 2) |
| `learning_rate`| 1e-4 | RMSProp step size | Sec 4: RMSProp is recommended for NTM |
| `grad_clip` | (-10, 10) | Clip every gradient component (`clip_grad_value_`) | Sec 4.6: "all gradient components are clipped elementwise to the range (-10, 10)" |

---

## Common Issues & Fixes

### Loss is "NaN"
- **Reason:** Exploding gradients, or a normalization that divides by ~0 (e.g. the sharpening of Eq 9 with a diffuse focus and a large gamma).
- **Fix:** Clip gradients: `train_minimal.py` clips every component to (-10, 10) like the paper (`nn.utils.clip_grad_value_`). Keep normalizations scale-invariant, as `implementation.py` does for Eq 9.

### Bit Error Rate (BER) doesn't decrease
- **Reason:** The model failed to learn the "Sharpening" mechanism (Eq 9), resulting in blurry addresses.
- **Fix:** Check that your `gamma` parameter is properly initialized and constrained to $\gamma \ge 1$ using `1 + F.softplus`.

### Memory focus jumps randomly
- **Reason:** The Interpolation gate (Eq 7) is not learning to steady the focus.
- **Fix:** Train longer. With batch 32 on lengths 1-20 the held-out error stays at ~30-40% for the first ~1,500-1,750 iterations and then collapses within a few hundred (seed-dependent); `train_minimal.py` runs 3,000.

---

## The Addressing Equations

1. **Content:** $w_t^c = \text{Softmax}(\beta_t \cdot \text{CosineSim}(k_t, M_t))$
2. **Gating:** $w_t^g = g_t w_t^c + (1 - g_t) w_{t-1}$
3. **Shift:** $w_t^s = w_t^g * s_t$ (Circular Convolution)
4. **Sharp:** $w_t = (w_t^s)^\gamma / \sum (w_t^s)^\gamma$
