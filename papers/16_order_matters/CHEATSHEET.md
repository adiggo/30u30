# Day 16 Cheat Sheet: Sequence to Sequence for Sets

Quick reference for Read-Process-Write (the paper's model) and the Pointer Network baseline.

---

## Quick Start

```bash
# Sort 5 numbers with Read-Process-Write (5 process steps, 1 glimpse, chosen elements masked)
python train.py

# Larger sets, and the paper-like setting where chosen elements are not masked
python train.py --task sort --set-size 10 --no-mask

# The order-sensitive baseline: Pointer Network with an LSTM encoder
python train.py --model ptrnet --set-size 10

# Convex hull (a Pointer Networks task) with a STOP pointer
python train.py --task hull --set-size 10
```

---

## Key Hyperparameters

The paper's Table 1 varies only the process steps (P = 0, 1, 5, 10) and glimpses (0 or 1) and trains for 10,000 iterations; it gives no other hyperparameters. The rest are this repo's defaults.

| Parameter | Default | What It Does | Source |
|-----------|---------|--------------|--------|
| `--process-steps` | 5 | T steps of the process block (Eqs. 3-7) | Table 1: 0, 1, 5, 10 |
| `--glimpses` | 1 | One attention read before each pointer | Table 1: 0 or 1 |
| `--no-mask` | off | Off = chosen elements are masked (easier) | Paper does not say |
| `--hidden-dim` | 64 | Memory and LSTM size | Repo default |
| `--lr` | 1e-3 | Adam learning rate | Repo default |
| grad clip | 2.0 | Gradient-norm threshold | Repo default |

---

## Common Issues & Fixes

### Model selects the same element twice
- **Reason:** Nothing forbids it; the decoder must learn not to repeat itself.
- **Fix:** **Hard masking**: set the scores of chosen indices to `-inf` before the softmax (`mask_chosen=True`, the default). Without the mask, the process block and glimpses are what help (README, Results).

### Shuffling the input changes the output
- **Reason:** Something sees positions: positional encodings, or an LSTM reading the elements in order (the Ptr-Net baseline, and this folder's `ReadProcessWrite` before 2026-10-18).
- **Fix:** Per-element read block plus content-based attention only. Test with `validation/day16_process_block_check.py`.

### Variable-length outputs (convex hull)
- **Reason:** A fixed number of decoding steps does not fit hulls of different sizes.
- **Fix:** A learned **STOP** slot the decoder points at to finish (`stop_token=True`); pad targets after it with `IGNORE_INDEX` (-1), which the loss skips. Never pad by repeating an index the decoder has already masked: its log-probability is `-inf`.

### Performance drops on larger sets
- **Reason:** The model overfit to the fixed sequence length used in training.
- **Fix:** Use **Variable Length Training**. Train on a range of set sizes (e.g., sample $N \in [5, 15]$ per batch) to force the model to learn the selection rule rather than a fixed pattern.

### Loss becomes NaN
- **Reason:** Dividing by zero in attention scores or exploding gradients in the LSTM.
- **Fix:** Use `log_softmax` for NLL loss and ensure `torch.nn.utils.clip_grad_norm_` is active.

---

## Core Formulas

1. **Read:** $m_i = \mathrm{MLP}(x_i)$, the same small MLP for every element
2. **Process (Eqs. 3-7):** $q_t = \mathrm{LSTM}(q^*_{t-1})$, $e_{i,t} = m_i \cdot q_t$, $a_{i,t} = \mathrm{softmax}_i(e_{i,t})$, $r_t = \sum_i a_{i,t} m_i$, $q^*_t = [q_t, r_t]$; $q^*_T$ is invariant to the order of the set
3. **Glimpse:** $g_t = \sum_i \mathrm{softmax}_i\big(w^T \tanh(G_1 m_i + G_2 d_t)\big)\, m_i$, then point with the query $[d_t, g_t]$
4. **Pointer Attention Score:** $u_i = v^T \tanh(W_1 m_i + W_2\, \mathrm{query}_t)$
5. **Probability Distribution:** $p(y_t = i \mid y_{<t}, X) = \mathrm{softmax}_i(u_i)$, with padded (and, by default, already chosen) elements masked to $-\infty$
