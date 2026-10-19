# Solutions

Complete solutions for all exercises. **Try the exercises first.**

## Files

| Solution | Exercise | Description |
|----------|----------|-------------|
| `solution_1.py` | Additive Attention | Complete `AdditiveAttention` class |
| `solution_2.py` | Bidirectional Encoder | Complete `BidirectionalEncoder` class |
| `solution_3.py` | Attention Decoder | Complete `AttentionDecoder` class |
| `solution_4.py` | Training Loop | Full training script that achieves >90% |
| `solution_5.py` | Visualization | Attention heatmaps and analysis |

## Running Solutions

```bash
# Test individual components
python solution_1.py  # Tests attention
python solution_2.py  # Tests encoder
python solution_3.py  # Tests decoder

# Full training
python solution_4.py  # Trains and evaluates

# Visualization
python solution_5.py --demo  # Demo with synthetic data
```

## Expected Results

Measured output of `solution_4.py` (CPU, seed 42; about 6 minutes on one busy core):

```
Device: cpu
Creating datasets...
Creating model...
Parameters: 399,922

Training...
==================================================
Epoch  1 | Train: 3.0682 | Val: 1.2632
Epoch  5 | Train: 0.0882 | Val: 0.0220 | Acc: 96.00%
Epoch 10 | Train: 0.3995 | Val: 0.0043 | Acc: 100.00%
Epoch 15 | Train: 0.0508 | Val: 0.0167 | Acc: 99.50%
Epoch 20 | Train: 0.0617 | Val: 0.0029 | Acc: 100.00%
Epoch 25 | Train: 0.0161 | Val: 0.1237 | Acc: 94.00%
Epoch 30 | Train: 0.0009 | Val: 0.0004 | Acc: 100.00%
==================================================

Final Accuracy: 100.00%
Success: Model achieves >90% accuracy.
```

(The script prints every epoch; accuracy, on 200 validation sequences, every 5th. The occasional
dip, as at epoch 25, comes from Adam with batch size 32 and no learning-rate decay.)

## Key Implementation Details

### Solution 1: Attention
- The paper's alignment model (Appendix A.2.2): `query_size` = n (decoder state), `key_size` = 2n
  (bidirectional annotations), `hidden_size` = n'
- Uses `nn.Linear` without bias for cleaner math
- Xavier initialization for stable training
- Proper masking with `-inf` before softmax

### Solution 2: Encoder
- Packs sequences for efficient RNN processing
- Returns the 2n-dim annotations h_j = [forward; backward] (Sec. 3.2, Eq. 7) and sums the two
  directions' final states for the decoder (the paper uses s_0 = tanh(W_s backward h_1))
- Handles variable-length sequences correctly

### Solution 3: Decoder
- Attention computed BEFORE GRU step (Bahdanau style)
- Combines hidden + context + embedding for output
- Supports both step-by-step and full sequence modes

### Solution 4: Training
- CrossEntropyLoss with `ignore_index=0` for padding
- Gradient clipping with `max_norm=1.0`
- Greedy decoding for inference

### Solution 5: Visualization
- Uses matplotlib for heatmaps
- Computes attention entropy (lower = more focused)
- Checks for reversed diagonal pattern

## Learning Points

After completing all solutions, you should understand:

1. **Why attention works** - Dynamic focus on relevant input parts
2. **The alignment problem** - Matching output positions to input positions
3. **Bidirectional encoding** - Context from both directions
4. **Teacher forcing** - Using ground truth during training
5. **Attention visualization** - Interpreting what the model learned

## Next Steps

Once you've mastered Bahdanau attention:

1. **Try different tasks** - Sorting, copying, arithmetic
2. **Scale up** - Real translation datasets
3. **Compare** - Implement Luong attention (dot-product)
4. **Move forward** - Study the Transformer (self-attention)

That completes Day 15.
