# Day 17: Neural Turing Machines (NTM)

> Alex Graves, Greg Wayne, Ivo Danihelka (2014) - [Original Paper](https://arxiv.org/abs/1410.5401)

**Time:** 3-5 hours  
**Prerequisites:** PyTorch, LSTM intuition, Linear Algebra (matrix products)  
**Code:** PyTorch (Implementation + Visualization)

---

## What This Paper Is Actually About

The Neural Turing Machine (NTM) addresses a fundamental limitation of recurrent neural networks (RNNs): the coupling of processing power and memory capacity. 

In a standard LSTM, the "memory" is the hidden state vector. If you want to store more information, you have to increase the size of that vector, which also increases the number of weights in the network (the "thinking" part). The NTM breaks this coupling by connecting a neural network **Controller** to a separate **Memory Bank**. 

This allows the network to store large amounts of data without becoming computationally unmanageable. It successfully learns *algorithms* (like sorting and copying) rather than just statistical mappings.

---

## The Core Idea: Differentiable Addressing

The central problem Graves et al. address is: **How do you make a pointer differentiable?**

In a standard computer, a pointer is a discrete address (e.g., "0x7ffd"). You can't take a derivative of a discrete address. To solve this, Graves et al. introduced **Weightings**. 

Instead of reading from one address, the NTM "reads" from every address at once, using a weighting vector $w_t$ that sums to 1. If the weight is 1.0 at address $A$ and 0.0 everywhere else, it's a discrete pointer. If it's spread out, it's a "blurry" focus.

### The Addressing Pipeline (Section 3.3.1)

To determine where to focus, the NTM uses a four-stage process:

1.  **Content Addressing (Eq 5):** The model emits a "search key" $k_t$. It looks for rows in memory that are similar to $k_t$ using cosine similarity.
2.  **Interpolation (Eq 7):** A gate $g_t$ decides whether to use the new search result or stick with the previous focus $w_{t-1}$.
3.  **Convolutional Shift (Eq 8):** A shift weighting $s_t$ performs circular convolution, allowing the "pointer" to move (e.g., "stay put", "move +1", "move -1").
4.  **Sharpening (Eq 9):** A factor $\gamma_t$ "squeezes" the distribution to make the focus more precise.

---

## What the NTM Actually Showed

### 1. The Copy Task (Section 4.1)
The model is shown a sequence of random bits, a delimiter, and is then asked to repeat the sequence. 

- **LSTM Result:** Copies well up to the training length (20) but degrades rapidly beyond it (Section 4.1, Figures 4-5).
- **NTM Result:** Learns to increment its memory pointer as it writes, and then increment it as it reads. It generalizes to sequences much longer than those it was trained on because it learned the *algorithm* of copying, not just the data.

### 2. The Sorting Task (Section 4.5)
The model is shown a list of vectors, each with a priority. It is then asked to output them in sorted order.

- **Outcome:** The NTM writes each vector to a location given (approximately linearly) by its priority, then reads the locations in increasing order (Section 4.5, Figure 17).

## The Architecture

### 1. The Addressing Pipeline (Section 3.3.1)
The NTM uses a four-stage process to decide where to focus its "blurry" pointer:

1. **Content Addressing**: Emits a "search key" $k_t$. Looks for rows in memory similar to $k_t$ using cosine similarity (Eq 5).
2. **Interpolation**: A gate $g_t$ decides whether to use the new search result or stick with the previous focus (Eq 7).
3. **Convolutional Shift**: A shift weighting $s_t$ performs circular convolution, allowing the pointer to move (e.g., stay put, +1, -1) (Eq 8).
4. **Sharpening**: A factor $\gamma_t$ "squeezes" the distribution to make the focus more precise (Eq 9).

### 2. Controller & Heads
- **Memory Matrix ($N \times M$):** The "scratchpad" where data is stored.
- **Read Head**: Computes a weighted average of memory: $r_t = \sum w_t(i) M_t(i)$ (Section 3.1).
- **Write Head**: Performs an **Erase** followed by an **Add** (Section 3.2):
  - `M_t(i) = M_{t-1}(i) * [1 - w_t(i) * e_t]` (Erase)
  - `M_t(i) = M_t(i) + w_t(i) * a_t` (Add)

**Variables & Dimensions:**
- `w_t` (N): The weighting vector (the pointer).
- `e_t` (M): The erase vector (0 to 1).
- `a_t` (M): The add vector (data to write).
- `k_t` (M): The content search key.

---

## Implementation Notes

When implementing an NTM, pay attention to these stability details:

- **Gradient Clipping (Section 4.6)**: "all gradient components are clipped elementwise to the range (-10, 10)". `train_minimal.py` does exactly that (`clip_grad_value_`); the repo used to clip the global gradient *norm* at 10, a different operation that rescales the whole gradient. With the per-bit BCE loss used here, a component exceeded 10 only a handful of times, during the few hundred iterations in which the network discovers the copy procedure (gradient norms then reach ~100-300). Both clips learned the task equally well in our runs.
- **Numerical Stability**: Operations like sharpening ($w^\gamma$) can generate extremely large or small numbers. Use `torch.clamp` and small epsilons ($1e-8$) to prevent `NaN`.
- **Initialization**: The paper resets the controller state, the previous read vectors and the memory to learned bias values at the start of every sequence (Section 4). This implementation uses fixed constants instead (registered buffers, not learned): memory 1e-6, head weightings one-hot at slot 0, controller state and read vectors zero.

**Things that will bite you:**
- **Circular Convolution**: Implementing Eq 8 correctly requires a specialized 1D convolution or a sequence of `torch.roll` operations: with shifts -1, 0, +1 it is three rolled copies of the weighting, weighted by s. A Python loop over the N slots (what this repo used to do) costs 2N rolls per head per time step and made a training iteration ~13x slower (about 1 s instead of 0.08 s at length 10).
- **Softplus for Positivity**: Parameters like $\beta$ (key strength) and $\gamma$ (sharpening) must be positive. Use `1 + F.softplus(x)` to ensure stability.
- **Memory Footprint**: A $128 \times 20$ memory bank is small, but the addressing logic (calculating scores for every slot) adds significant overhead per step.

---

## What to Build

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the copy task training (paper's settings, 3,000 iterations: ~5 minutes on one CPU core)
python train_minimal.py

# The same task for 10,001 iterations (320k sequences), as the script used to run
python train_minimal.py --paper
```

### Results in This Repo (2026-10-18)

`python train_minimal.py` uses the paper's settings (Table 2, Sections 4.1 and 4.6), batch 32, seed 0 and 3,000 iterations, and ran at 10.4 iterations/s (296 s) on one CPU core. On a fixed held-out set of lengths 1-20:

| Metric | Before training | After 3,000 iterations |
|---|---|---|
| bit error | 50.25% | 0.67% |
| BCE per bit | 0.694 | 0.023 |
| cost (bits per sequence, the paper's unit) | 84.2 | 2.8 |

On lengths never seen in training the bit error is 0.42% at length 30 and 0.75% at length 50: the learned procedure generalizes, as in Figure 4. Before the vectorized shift, the same script managed ~1 iteration/s (1.06 s per iteration at length 10; `validation/day17_vectorized_shift.py`) and had done ~1,150 of its 10,001 iterations after 25 minutes.

Training is seed-sensitive. The held-out error sits at 30-40% for the first 1,000-1,750 iterations, then collapses within a few hundred, and a run can later destabilize. Bit error after 3,000 iterations:

| Seed | Per-bit loss, clip each component (default) | Per-bit loss, clip the global norm | Cost per sequence (`--loss sequence`), clip each component |
|---|---|---|---|
| 0 | 0.67% | not run | 25.6% (still on the plateau) |
| 1 | 0.00% | not run | 0.00% |
| 2 | 18.3% (reached 3%, then destabilized) | 27.7% (reached 0.5%, then destabilized) | 0.21% |

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|---------------------------|
| 1 | Content-based addressing | Implement cosine similarity weighting (Eq 5) |
| 2 | Convolutional shift | Build circular convolution for location-based addressing (Eq 8) |
| 3 | Memory erase and add | Implement the write head update operations (Eq 3, 4) |
| 4 | Controller input preparation | Concatenate input with previous read vectors (Section 2) |
| 5 | Sharpening mechanism | Apply gamma exponentiation to prevent blurry focus (Eq 9) |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **Decoupling memory from computation is the core contribution.** Standard RNNs store everything in the hidden state, coupling memory capacity with processing power. The NTM separates these by connecting a controller to an external memory bank (Section 1).

2. **Differentiable addressing makes discrete pointer operations learnable.** By using soft weightings that sum to 1, the NTM can read and write to memory locations while remaining fully differentiable for backpropagation (Section 3.1).

3. **The four-stage addressing pipeline is necessary, not decorative.** Content addressing alone cannot implement simple algorithms like incrementing a pointer. Interpolation, shift, and sharpening together allow learned sequential access patterns (Section 3.3.1).

4. **NTMs learn algorithms, not just statistical mappings.** On the copy task, the model generalizes to sequences longer than those in training because it learns to increment a pointer, not memorize sequence patterns (Section 4.1).

---

## Files in This Directory

| File | What It Is |
|------|------------|
| `implementation.py` | Full NTM with controller, memory, and read/write heads |
| `train_minimal.py` | Training script for the copy task (Section 4.1) |
| `visualization.py` | Memory state and addressing weight visualizations |
| `notebook.ipynb` | Interactive walkthrough of addressing mechanisms |
| `exercises/` | 5 exercises with solutions |
| `paper_notes.md` | Condensed notes on the original paper |
| `CHEATSHEET.md` | Quick reference for architecture dimensions and training tips |
| `requirements.txt` | Python dependencies |

---

## Further Reading

- [Original Paper - Neural Turing Machines (arXiv:1410.5401)](https://arxiv.org/abs/1410.5401)
- [Differentiable Neural Computers (Graves et al., 2016)](https://www.nature.com/articles/nature20101) - the successor to NTMs with improved memory addressing
- [Learning to Transduce with Unbounded Memory (Grefenstette et al., 2015)](https://arxiv.org/abs/1506.02516) - alternative neural stack/queue architectures

---

**Next:** [Day 18 - Pointer Networks](../18_pointer_networks/)
