# Exercise Solutions: Understanding LSTM Networks

Complete, well-commented solutions for all 5 exercises.

---

## How to Use These Solutions

### Try First, Check Later!

These solutions are here to help you **learn**, not to copy-paste. Here's the recommended approach:

1. **Attempt the exercise first** (spend at least 30 minutes)
2. **Get stuck?** Review the relevant section in README.md
3. **Still stuck?** Look at just the function you need help with
4. **Compare your solution** with ours after completing the exercise

**Remember:** Struggling is part of learning.

---

## Solutions Index

### Exercise 1: Build LSTM from Scratch
**File:** `solution_01_lstm_implementation.py`  
**Difficulty:** Hard  
**What's included:**
- Complete LSTM class with all 4 gates
- Forward pass with detailed comments
- Backward pass (BPTT) with gradient flow
- Weight update with Adagrad
- Test function to verify implementation

**Key learning points:**
- How to implement forget gate with bias = 1
- Cell state update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
- Gradient flow through cell state (why LSTMs work!)
- Proper gradient clipping

**Lines of code:** ~400 lines  
**Study time:** 1-2 hours

---

### Exercise 2: Gate Activation Analysis
**File:** `exercise_02_solution.py`  
**Difficulty:** Medium  
**What's included:**
- `capture_gate_activations()` - capture all gates during forward pass
- `analyze_gate_patterns()` - statistical analysis with correlations
- `visualize_gates()` - 4-panel visualization
- `compare_gates_across_contexts()` - behavior in different scenarios

**Key learning points:**
- When forget gates activate (context switches, sentence boundaries)
- When input gates activate (new important information)
- Gate complementarity (negative correlation between forget & input)
- Pattern recognition in gate behaviors

**Lines of code:** ~300 lines  
**Study time:** 1-2 hours

---

### Exercise 3: Ablation Study
**File:** `exercise_03_solution.py`  
**Difficulty:** Medium  
**What's included:**
- `VanillaRNN` class (baseline, with its own BPTT `backward()`)
- `LSTMNoForget`, `LSTMNoInput`, `LSTMNoOutput`: `implementation.LSTM` with one gate fixed to 1; they reuse its `backward()` and Adagrad update
- `run_ablation_study()` - trains all 5 variants with the same seed, windows and optimizer, and measures held-out loss
- `analyze_results()` - per-seed comparison with the full LSTM
- `visualize_comparison()` - loss curves and measured bars (`--plot PATH`)

**Measured (3 seeds, 1,200 updates each, about 1 minute):** removing the forget gate costs about
+0.14 nats/char of held-out loss in every seed; removing the input or the output gate, or using a
vanilla RNN, is within seed noise on this 4k-character corpus. Gradients are checked by
`validation/day02_solution_grads.py`.

**Lines of code:** ~430 lines  
**Study time:** 2-3 hours

---

### Exercise 4: Long-Range Dependencies
**File:** `exercise_04_solution.py`  
**Difficulty:** Medium  
**What's included:**
- `generate_memory_task()` - the exercise's "remember the first character" task
- `train_on_distance()` / `test_all_distances()` - a fresh model per distance, tested on held-out sequences
- `generate_delayed_xor_task()`, `train_on_memory_task()`, `evaluate_memory_capacity()` - delayed XOR with distractor bits, one model tested at other delays (`--xor`)
- `analyze_cell_state_evolution()`, `forget_gates_during_delay()` - what the trained cell holds
- `generate_copy_task()` - a copy-task generator (not trained here)

**Measured (3 seeds, 3,000 training sequences per model, 1-2 minutes):** the LSTM is near-perfect at
5 steps, about 92% at 10 and about 49% at 20 (chance 10%); the vanilla RNN learns 5 steps, but at 10
and 20 its accuracy swings with the seed (54/99/50% and 90/8/9%). Neither learns 50 steps in this
budget. The trained LSTM keeps a few units with forget gates near 1 (the memory) while most forget.

**Lines of code:** ~420 lines  
**Study time:** 2-3 hours

---

### Exercise 5: LSTM vs GRU Comparison
**File:** `exercise_05_solution.py`  
**Difficulty:** Hard  
**What's included:**
- `GRU` class from scratch (forward, BPTT backward, the same Adagrad rule as the LSTM)
- `count_parameters()` for both models
- `compare_lstm_gru()` - both models on the same windows with the same optimizer
- `analyze_comparison()` and `visualize_comparison()` - measured parameters, speed and loss

**Measured (hidden 32, 3 seeds, 1,200 updates each, under a minute):** the GRU has 21.6% fewer
parameters (2 gates plus a candidate, versus 3 gates plus a candidate), is about 1.2x faster per
update in NumPy, and reached a lower held-out loss in every seed (by 0.02-0.09 nats/char) at this
early stage of training. Larger studies found the two comparable (Chung et al. 2014).

**Lines of code:** ~320 lines  
**Study time:** 3-4 hours

---

## Solution Features

All solutions include:

- **Detailed comments** explaining every step
- **Type hints** in docstrings
- **Error handling** for edge cases
- **Test functions** to verify correctness
- **Print statements** for debugging
- **Clear variable names** (no single letters)
- **Mathematical notation** in comments

---

## Learning Tips

### If You're Comparing Your Solution

Look for these aspects:

1. **Correctness** - Does it produce the right output?
2. **Efficiency** - Are there unnecessary loops?
3. **Readability** - Would someone else understand your code?
4. **Comments** - Did you explain the "why", not just the "what"?
5. **Edge cases** - Did you handle empty inputs, NaNs, etc.?

### Common Mistakes to Check

- Forget gate bias not initialized to 1
- Mixing up cell state (C) and hidden state (h)
- Not clipping gradients (causes explosion)
- Using wrong activation (sigmoid vs tanh)
- Shape mismatches in matrix operations
- Not detaching states between sequences

---

## Comparison with Your Implementation

### Use this checklist:

- [ ] Does your forward pass match the equations?
- [ ] Are gradients flowing correctly backward?
- [ ] Do gate values stay in [0, 1] range?
- [ ] Is cell state updating via addition (not multiplication)?
- [ ] Are you clipping gradients to prevent explosion?
- [ ] Does your model train (loss decreases)?
- [ ] Can it generate reasonable text?

---

## Going Further

After completing the exercises, try:

1. **Optimize for speed** - Vectorize operations
2. **Add features** - Peephole connections, layer norm
3. **Scale up** - Larger hidden size, deeper networks
4. **Real tasks** - Sentiment analysis, NER, time series
5. **Compare frameworks** - Implement in PyTorch/TensorFlow

---

## Additional Resources

- **Original paper:** Hochreiter & Schmidhuber (1997)
- **Colah's blog:** "Understanding LSTM Networks" (well-known visualizations)
- **Day 2 README:** Complete LSTM explanation
- **Day 2 notebook:** Interactive walkthrough

---

## Need Help?

If you're stuck after reviewing the solution:

1. **Re-read the relevant section** in README.md
2. **Check the notebook** for interactive examples
3. **Print intermediate values** to debug
4. **Open an issue** on GitHub with your question
5. **Join the discussion** to ask the community

---

## Solution Status

| Exercise | Solution Available | Verified | Study Time |
|----------|-------------------|----------|------------|
| 1. Build LSTM | Available | Tested | 1-2 hours |
| 2. Gate Analysis | Available | Tested | 1-2 hours |
| 3. Ablation Study | Available | Tested | 2-3 hours |
| 4. Long-Range Deps | Available | Tested | 2-3 hours |
| 5. LSTM vs GRU | Available | Tested | 3-4 hours |

**Total:** All 5 solutions are complete and ready for study!

---

## Completion Checklist

Once you've completed all 5 exercises, you will have:

- Built an LSTM from scratch with all 4 gates
- Analyzed gate behavior and activation patterns
- Tested component importance via ablation study
- Measured memory capacity with synthetic tasks
- Compared LSTM with GRU architectures
- Understood mathematical foundations deeply
- Debugged common training issues
- Visualized internal states and gradients

You're ready for Day 3.

---

*Remember: The best way to learn is to struggle through it yourself first. Use these solutions as a guide, not a shortcut!*
