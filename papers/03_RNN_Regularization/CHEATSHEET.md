# RNN Regularization Cheatsheet

Quick reference for Zaremba et al. (2014) and the regularization techniques we implement.

---

## The Paper's Contribution (30 seconds)

**Problem:** Dropout doesn't work with RNNs — it disrupts temporal memory.
**Solution:** Apply dropout ONLY to non-recurrent (vertical) connections. Leave recurrent (horizontal) connections alone.
**Result:** PTB test perplexity 114.5 (best non-regularized LSTM) -> 78.4. Previous best single models: 107.5 (Pascanu et al.) and 100.0 (Cheng et al.).

---

## Where to Apply Dropout in an LSTM

```
Dropout ON:                     Dropout OFF:
- Input embedding: D(x_t)      - Recurrent hidden state: h_{t-1}^l
- Between layers: D(h_t^{l-1}) - Cell state: c_{t-1}^l
- Before softmax: D(h_t^L)
```

Rule of thumb: if the connection goes **between layers** (vertical), apply dropout. If it goes **across time** (horizontal), don't.

---

## The Four Techniques (Paper + Our Additions)

### 1. Dropout (From the paper)

```python
# Forward (training only)
mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
out = x * mask

# Forward (testing)
out = x  # No change!

# Backward
dx = dout * mask
```

Key points:
- Scale by `1/keep_prob` during training (inverted dropout)
- NEVER apply at test time
- For RNNs: only on non-recurrent connections

### 2. Layer Normalization (Our addition — Ba et al. 2016)

```python
mean = np.mean(x, axis=-1, keepdims=True)
var = np.var(x, axis=-1, keepdims=True)
x_hat = (x - mean) / np.sqrt(var + eps)
out = gamma * x_hat + beta
```

Normalizes per-sample (not per-batch), works with any batch size.

### 3. Weight Decay / L2 (Our addition)

```python
L_total = L_model + (lambda/2) * sum(w**2)
# Gradient: dW += lambda * W
```

Penalizes large weights, encourages simpler models.

### 4. Early Stopping (Our addition)

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    counter = 0
    save_model()
else:
    counter += 1
    if counter >= patience:
        STOP and load best model
```

---

## Paper's Hyperparameters

### Non-regularized PTB Baseline (test ppl 114.5)
```python
layers = 2
hidden = 200
dropout = 0.0
init = 0.1            # uniform in [-0.1, 0.1]
lr = 1.0              # SGD
lr_decay = 1/2        # After epoch 4
bptt = 20
batch_size = 20
epochs = 13
```

### Medium PTB Model
```python
layers = 2
hidden = 650
dropout = 0.5         # 50% dropout
init = 0.05           # uniform in [-0.05, 0.05]
lr = 1.0              # SGD
lr_decay = 1/1.2      # After epoch 6
grad_clip = 5
bptt = 35
batch_size = 20
epochs = 39
```

### Large PTB Model
```python
layers = 2
hidden = 1500
dropout = 0.65        # 65% dropout
init = 0.04           # uniform in [-0.04, 0.04]
lr = 1.0
lr_decay = 1/1.15     # After epoch 14
grad_clip = 10
bptt = 35
batch_size = 20
epochs = 55
```

### Machine Translation
```python
layers = 4
hidden = 1000
dropout = 0.2         # Much less — more training data
```

---

## Hyperparameter Guide (Our Recommendations)

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| `keep_prob` | 0.5-0.9 | Paper uses 0.35-0.8 depending on model/task size |
| `weight_decay` | 0.00001-0.01 | Not in the paper; standard practice |
| `patience` | 3-10 | Not in the paper; standard practice |
| `grad_clip` | 5-10 | Global gradient norm; paper uses 5 (medium) and 10 (large) |

### Starting Point (`train_minimal.py` defaults, the paper's recipe)
```python
keep_prob = 0.5        # "50% dropout" on non-recurrent connections; 1.0 = none
weight_decay = 0.0     # the paper uses none (--weight-decay to try it)
patience = 0           # off: the paper trains a fixed number of epochs (--patience to try it)
grad_clip = 5.0        # global norm, from the paper
```

---

## Common Issues

### Still Overfitting (val loss >> train loss)
- More aggressive dropout: `keep_prob = 0.5`
- Add weight decay: `weight_decay = 0.001`
- Reduce patience: `patience = 3`

### Underfitting (both losses stay high)
- Less dropout: `keep_prob = 0.9`
- Less weight decay: `weight_decay = 0.00001`
- More patience: `patience = 10`

### NaN Loss
- Clip the global gradient norm as the paper does: `clip_grad_norm(grads, 5.0)` (rescales all gradients together; element-wise `np.clip` changes their direction)
- Check learning rate (may be too high)
- Add layer normalization (our addition, not in the paper)

---

## Quick Start

```bash
# The paper's recipe at CPU scale: 2x200 LSTM, 50% dropout, first 100k PTB words (downloads PTB once)
python train_minimal.py

# Dropout vs no dropout, everything else identical (about 10 min of CPU)
python train_minimal.py --compare

# More aggressive dropout (the large model's 65% = keep_prob 0.35)
python train_minimal.py --keep-prob 0.35

# Our additions, off by default (the paper uses neither)
python train_minimal.py --weight-decay 0.00001 --patience 2
```

---

## Formulas

### Dropout
$$y = \frac{x \odot m}{p}, \quad m_i \sim \text{Bernoulli}(p)$$

### Layer Norm
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

### Weight Decay
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2} \|W\|_2^2$$

---

## Debugging Checklist

- [ ] Dropout OFF during evaluation? (Must be!)
- [ ] Dropout scaled by 1/keep_prob? (Inverted dropout)
- [ ] Dropout only on non-recurrent connections? (Paper's key insight)
- [ ] Weight decay applied to weights only, not biases?
- [ ] Saving best model, not just the last one?
- [ ] Gradient clipping enabled?

---

*For paper details, see [paper_notes.md](paper_notes.md). For implementation guide, see [README.md](README.md).*
