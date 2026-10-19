# Day 16 Solutions: Order Matters (Pointer Networks)

Complete solutions for all 5 exercises with detailed explanations.

---

## Solution Files

Each solution is a fully working implementation that you can run directly:

- `solution_1.py` - Basic Pointer Attention
- `solution_2.py` - Set Encoder  
- `solution_3.py` - Sorting with Pointer Networks
- `solution_4.py` - Convex Hull
- `solution_5.py` - Traveling Salesman Problem

---

## How to Use Solutions

### When to Look at Solutions

**Good reasons:**
- You've tried for 30+ minutes and are stuck
- You want to verify your implementation
- You want to see alternative approaches
- You finished and want to compare

**Avoid:**
- Looking before attempting the exercise
- Copying without understanding
- Skipping the learning process

### How to Learn from Solutions

1. **Try First** - Spend serious time on the exercise
2. **Compare Approaches** - Even if yours works, see the solution
3. **Understand Differences** - Why did the solution do X differently?
4. **Run Both** - Test your version vs the solution
5. **Experiment** - Modify the solution, break it, fix it.

---

## Key Concepts Covered

### Solution 1: Pointer Attention
- Attention score computation
- Masking mechanism
- Softmax for probability distribution
- Argmax for selection

### Solution 2: Set Encoder
- Self-attention without positional encoding
- Permutation equivariance (per element) and invariance (after mean pooling)
- Layer normalization
- Residual connections

### Solution 3: Sorting
- Training loop implementation
- Teacher forcing
- Negative log-likelihood loss
- Exact match accuracy

### Solution 4: Convex Hull
- Geometric reasoning with neural nets
- Variable-length outputs: a STOP pointer, IGNORE_INDEX padding, one canonical hull order
- Visualization of predictions
- Evaluation metrics for geometry

### Solution 5: TSP
- Combinatorial optimization
- Beam search (optional)
- Tour quality metrics
- Comparison with classical algorithms

---

## Performance Expectations

Measured on one CPU core on 2026-10-18 (the table that used to be here was not measured):

| Task | Run | Result |
|------|-----|--------|
| **Sorting** (N = 5) | `python train.py` (Read-Process-Write, chosen elements masked) | 100% exact sorts |
| **Sorting** (N = 5) | solution 3's `train_epoch` with the LSTM `PointerNetwork`, 8 epochs of 2,000 sets | 91.6% exact sorts |
| **Convex Hull** (10 points) | `python train.py --task hull --set-size 10 --epochs 20` | 74.8% exact hulls |
| **Convex Hull** (10 points) | `python solution_4.py` (8 epochs of 1,000 sets) | 56.0% exact hulls |
| **TSP** (10 cities) | `python train.py --task tsp --set-size 10 --epochs 3` | 57.4% of tours match the nearest-neighbour teacher |

Accuracy counts a set only if the whole output matches (for hulls: every vertex in order, then STOP).
The README's "Results" section has the paper's sorting comparison (Table 1).

---

## Common Issues & Fixes

### Issue 1: NaN Loss
**Symptom:** Loss becomes NaN after a few iterations

**Solution:**
```python
# Check your masking - masked values should be -inf BEFORE softmax
scores = scores.masked_fill(mask == 0, float('-inf'))
probs = F.softmax(scores, dim=-1)  # This will handle -inf correctly

# Also add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
```

### Issue 2: Model Doesn't Learn
**Symptom:** Accuracy stuck at random (~10% for 10 elements)

**Solutions:**
- Start with smaller sets (5 elements)
- Increase model capacity (hidden_dim=128 or 256)
- Use teacher forcing during training
- Check learning rate (try 1e-3 or 1e-4)
- Verify loss is decreasing

### Issue 3: Out of Memory
**Symptom:** CUDA out of memory error

**Solutions:**
```python
# Reduce batch size
batch_size = 32  # or 16

# Reduce model size
hidden_dim = 64  # instead of 128

# Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = compute_loss(...)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue 4: Slow Training
**Symptom:** Each epoch takes forever

**Solutions:**
- Use GPU if available
- Reduce dataset size for testing
- Use DataLoader with `num_workers > 0`
- Profile your code to find bottlenecks

---

## Solution Highlights

### Elegant Attention Implementation (Solution 1)

```python
# Clean, vectorized attention computation
keys_proj = self.W_key(keys)  # [batch, seq_len, hidden]
query_proj = self.W_query(query).unsqueeze(1)  # [batch, 1, hidden]
combined = torch.tanh(keys_proj + query_proj)  # Broadcasting!
scores = self.v(combined).squeeze(-1)  # [batch, seq_len]
```

### Permutation Invariance Test (Solution 2)

```python
# Prove order doesn't matter
points_shuffled = points[:, torch.randperm(set_size)]
enc1 = encoder(points)
enc2 = encoder(points_shuffled)
# enc1.mean(dim=1) should equal enc2.mean(dim=1)!
```

### Teacher Forcing Strategy (Solution 3)

```python
# Use ground truth during training
if training and random.random() < teacher_forcing_ratio:
    next_input = inputs[target_pointer]  # Ground truth
else:
    next_input = inputs[predicted_pointer]  # Model prediction
```

---

## Going Beyond

### Extensions to Try

1. **Beam Search** - Instead of greedy decoding, keep top-k candidates
2. **Attention Visualization** - Plot heatmaps of attention weights
3. **Larger Problems** - Test generalization to 50+ elements
4. **Different Tasks** - Try other set-to-sequence problems
5. **Curriculum Learning** - Start small, gradually increase difficulty

### Research Directions

- **Set Transformers** - Modern extension with full attention
- **Graph Neural Networks** - Sets with edges/relationships
- **Slot Attention** - Object-centric representations
- **Neural Combinatorial Optimization** - Better TSP solvers

---

## Additional Reading

- Original Paper: https://arxiv.org/abs/1511.06391
- Pointer Networks Explained: https://arxiv.org/abs/1506.03134
- Set Transformers: https://arxiv.org/abs/1810.00825
- Attention Is All You Need: https://arxiv.org/abs/1706.03762

---

## Solution Checklist

Use this to track which solutions you've reviewed:

- [ ] Solution 1: Pointer Attention
- [ ] Solution 2: Set Encoder
- [ ] Solution 3: Sorting
- [ ] Solution 4: Convex Hull
- [ ] Solution 5: TSP

---

**Solutions are in the files listed above.**

*Remember: The goal is understanding, not just getting the right answer.*
