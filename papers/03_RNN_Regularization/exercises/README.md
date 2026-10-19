# Exercises: RNN Regularization

5 hands-on exercises to master regularization techniques. Work through them in order — each builds on the previous.

Exercise 1 is grounded in the Zaremba et al. (2014) paper. Exercises 2-5 go beyond the paper to cover complementary techniques that are standard in practice.

**Time estimates:**
- Easy (20-30 min)
- Medium (30-45 min)
- Hard (45-60 min)
- Very Hard (60-90 min)

---

## Exercise 1: Implement Dropout [Medium] — From the Paper

**Goal:** Implement a dropout layer with forward and backward passes from scratch.

**File:** `exercise_01_dropout.py`

**What you'll learn:**
- How dropout randomly disables neurons
- Why we scale outputs by 1/keep_prob (inverted dropout)
- Difference between training and inference modes
- How gradients flow through dropout

**Tasks:**

1. Implement `dropout_forward(x, keep_prob, training)`:
   - Generate binary mask: `mask = np.random.binomial(1, keep_prob, x.shape)`
   - Scale outputs: `output = x * mask / keep_prob`
   - Return (output, mask) for backward pass

2. Implement `dropout_backward(dout, mask)`:
   - Gradient only flows through active neurons
   - Return `dout * mask`

3. Handle edge cases:
   - `training=False`: return input unchanged
   - `keep_prob=1.0`: no dropout (return input)

**Success criteria:**
- Forward pass produces correct sparsity (verify ~20% zeros with keep_prob=0.8)
- Output expectation is unchanged (mean stays same)
- Backward pass only propagates through active neurons
- All test cases pass

**Hints:**
- The mask is already scaled by 1/keep_prob in forward, so backward just multiplies by mask
- Use `np.random.binomial(1, keep_prob, size)` for mask generation
- Test with known seeds for reproducibility
- Common mistake: applying dropout at test time

---

## Exercise 2: Layer Normalization [Hard] — Our Addition

*Not from the paper. Layer normalization (Ba et al., 2016) is a complementary technique widely used with RNNs.*

**Goal:** Implement layer normalization with learnable parameters (gamma, beta).

**File:** `exercise_02_normalization.py`

**What you'll learn:**
- How normalization stabilizes training
- Computing statistics per sample (not per batch)
- Learnable scale and shift parameters
- Backward pass through normalization

**Tasks:**

1. Implement `layer_norm_forward(x, gamma, beta, eps)`:
   - Compute mean and variance across features (axis=-1)
   - Normalize: `x_hat = (x - mean) / sqrt(var + eps)`
   - Scale and shift: `y = gamma * x_hat + beta`
   - Cache values for backward pass

2. Implement `layer_norm_backward(dout, cache)`:
   - Compute `dgamma = sum(dout * x_hat)`
   - Compute `dbeta = sum(dout)`
   - Compute `dx` using chain rule (this is the tricky part)

3. Use `eps=1e-5` for numerical stability

**Success criteria:**
- Forward output has zero mean, unit variance (per sample)
- Backward gradients match numerical gradient check
- Learnable parameters (gamma, beta) update correctly
- All test cases pass

**Hints:**
- Use `keepdims=True` when computing mean/var for proper broadcasting
- Layer norm normalizes across features, NOT across batch
- For backward, the derivative is complex — see solution if stuck
- Test with gamma=1, beta=0 first (identity transform)

---

## Exercise 3: Weight Decay (L2 Regularization) [Easy] — Our Addition

*Not from the paper. L2 regularization is a standard technique included here because it pairs well with dropout.*

**Goal:** Implement L2 regularization that penalizes large weights.

**File:** `exercise_03_weight_decay.py`

**What you'll learn:**
- Why simpler models generalize better (Occam's Razor)
- How weight decay pushes weights toward zero
- Adding regularization to the loss
- Gradient of L2 penalty

**Tasks:**

1. Implement `compute_l2_penalty(weights_list, weight_decay)`:
   - Sum squared weights: `sum(w^2)`
   - Multiply by coefficient: `lambda/2 * sum(w^2)`
   - Return scalar penalty

2. Implement `regularized_loss(model_loss, weights, weight_decay)`:
   - Combine: `total = model_loss + l2_penalty`

3. Implement gradient contribution:
   - Weight decay adds `lambda * W` to gradient

**Success criteria:**
- Penalty increases with larger weights
- Penalty increases with higher weight_decay
- Regularized loss = original loss when weight_decay=0
- Gradient includes regularization term

**Hints:**
- Don't forget the 1/2 factor: `0.5 * lambda * sum(w^2)` — it makes the gradient cleanly `lambda * W`
- Weight decay should NOT apply to biases
- Start with weight_decay=0.0001 (very small)
- Test: penalty should be proportional to weight magnitude

---

## Exercise 4: Early Stopping [Medium] — Our Addition

*Not from the paper. Early stopping is a standard training technique included here to complete the regularization toolkit.*

**Goal:** Implement an early stopping monitor that tracks validation loss and stops training when improvement plateaus.

**File:** `exercise_04_early_stopping.py`

**What you'll learn:**
- Why we stop before training loss is minimized
- The patience mechanism
- Saving and restoring best model state
- Detecting overfitting in real-time

**Tasks:**

1. Implement `EarlyStoppingMonitor.__init__`:
   - Store patience value
   - Initialize `best_loss = infinity`
   - Initialize `no_improve_count = 0`
   - Initialize `best_model_state = None`

2. Implement `check(val_loss, epoch, model_state)`:
   - If `val_loss < best_loss`: update best_loss, best_epoch, reset counter, save state
   - Else: increment counter
   - Return `True` to continue, `False` to stop

3. Implement `should_stop()`:
   - Return `no_improve_count >= patience`

**Success criteria:**
- Monitor correctly tracks best loss
- Counter resets on improvement
- Counter increments on plateau
- Stops after `patience` consecutive non-improvements
- Best model state is saved correctly

**Hints:**
- Use `float('inf')` for initial best_loss
- Common mistake: not resetting counter on improvement
- Test with artificial loss sequences to verify behavior
- Patience=3 means stop after 3 consecutive bad epochs

---

## Exercise 5: Full Pipeline [Very Hard] — Our Addition

*Not from the paper as a whole: dropout is the paper's; layer norm, weight decay and early stopping are our additions. Part B trains the paper's own model.*

**Goal:** Combine all four regularization techniques in a complete training loop that really trains.

**File:** `exercise_05_full_pipeline.py`

**What you'll learn:**
- How regularization techniques work together
- Training vs validation mode switching
- Managing hidden states in RNNs (carried across windows, never dropped)
- Real-world training workflow: forward, backward, weight decay, clipping, SGD, early stopping

**Tasks:**

Part A: `RegularizedRNNCell` (TODOs 1-12)
1. Store sizes and initialize weights (Xavier) and layer-norm parameters
2. Implement `layer_norm()` and `dropout()` (inverted scaling, training only)
3. Implement `forward()`: drop the input `x_t` only, never the hidden state carried to the next step (the paper's rule)
4. Implement `get_weights()` / `set_weights()` (snapshots for early stopping) and `compute_l2_penalty()`

Part B: `train_regularized_rnn()` (TODOs 13-15), a real training loop on a Penn Treebank slice. The model is the paper's 2-layer `LSTMLanguageModel` from `../implementation.py`, whose `keep_prob` sets the dropout. For each epoch:
1. Training (dropout on): for each window, `forward` → `backward` (BPTT) → add `weight_decay * W` to each weight gradient → `clip_grad_norm` → `sgd_update`
2. Validation (dropout off): `evaluate()` from `../train_minimal.py` returns the perplexity; record its log as the validation loss
3. Early stopping: snapshot the weights when validation improves, stop after `patience` epochs without improvement, and always finish with the best weights restored
4. Track `train_loss`, `val_loss`, `epochs_trained`, `stopped_early` and `best_epoch`

**Success criteria:**
- Every parameter array changes, and the validation loss falls well below its untrained value (about ln 10,000 = 9.2 nats)
- Validation loss tracked separately, with dropout off
- Early stopping triggers correctly
- The model ends with its best-validation weights
- All techniques work together

Penn Treebank is read from `../data/ptb/` (or `$PTB_DIR`) and downloaded (~6 MB) on first use. The tests take a few seconds; the solution's demo (`python solutions/05_full_pipeline.py`) trains three small models in about 2 minutes of CPU. It compares no regularization, dropout, and dropout plus weight decay. Without regularization, the validation perplexity turns upward after a few epochs and early stopping restores the best weights; the dropout runs end lower.

**Hints:**
- Don't forget `training=False` during validation (`evaluate()` does it for you)
- Reset the hidden state at the start of each epoch; carry it across windows within an epoch
- `grads[k] += ...` modifies the gradient in place; skip biases (keys starting with `b`)
- Use exercises 1-4 as building blocks

---

## How to Run

```bash
# Run a single exercise
cd exercises
python exercise_01_dropout.py

# Compare with solution
python solutions/01_dropout.py
```

---

## Tips

**General:**
1. Read the docstring first — it explains exactly what to implement
2. Check test cases — they show expected behavior
3. Start simple — get the basic case working before edge cases
4. Print intermediate values — helps debug shape/value issues

**When stuck:**
1. Re-read the relevant section in [../README.md](../README.md)
2. Check [../CHEATSHEET.md](../CHEATSHEET.md) for formulas
3. Look at the solution for just the function you need
4. Understanding > completing — take your time

**Common mistakes by exercise:**

Exercise 1 (Dropout): Applying dropout at test time. Forgetting to scale by 1/keep_prob.

Exercise 2 (Layer Norm): Using wrong axis for statistics (should be -1). Forgetting epsilon.

Exercise 3 (Weight Decay): Missing the 0.5 factor in penalty. Applying to biases.

Exercise 4 (Early Stopping): Not resetting counter on improvement. Using <= instead of <.

Exercise 5 (Full Pipeline): Dropout during validation (should be off). Not resetting hidden states. Forgetting to restore the best weights at the end.

---

## Exercise Summary

| # | Exercise | Difficulty | Time | Key Concept | Source |
|---|----------|-----------|------|-------------|--------|
| 1 | Dropout | Medium | 30 min | Random neuron deactivation | Paper |
| 2 | Layer Norm | Hard | 45 min | Activation normalization | Our addition |
| 3 | Weight Decay | Easy | 20 min | L2 regularization | Our addition |
| 4 | Early Stopping | Medium | 30 min | Validation monitoring | Our addition |
| 5 | Full Pipeline | Very Hard | 60 min | Integration | Our addition |

**Total time:** 3-4 hours

---

## Completion Checklist

- [ ] Exercise 1: Dropout — Can implement forward and backward
- [ ] Exercise 2: Layer Norm — Understand normalization math
- [ ] Exercise 3: Weight Decay — Know how L2 regularization works
- [ ] Exercise 4: Early Stopping — Can monitor and stop training
- [ ] Exercise 5: Full Pipeline — Can combine all techniques

Completed all 5? Move on to [solutions/](solutions/README.md) for detailed solution explanations.

---

*Try each exercise for at least 20 minutes before checking solutions. The struggle is where the learning happens.*
