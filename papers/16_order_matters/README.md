# Day 16: Order Matters: Sequence to Sequence for Sets

> Vinyals, Bengio, Kudlur (2015) - [Original Paper](https://arxiv.org/abs/1511.06391)

**Time:** 4-5 hours  
**Prerequisites:** PyTorch, LSTM intuition, Linear Algebra, Attention mechanisms  
**Code:** PyTorch (Implementation + Visualization)

---

## What This Paper Is Actually About

Vinyals, Bengio and Kudlur (ICLR 2016) show that the order in which a sequence-to-sequence model reads its inputs and writes its outputs changes what it learns, even though the chain rule makes every order valid (Sections 3, 4.1, 5.1). For inputs that are sets, they propose the **Read-Process-Write** model, whose summary of the set cannot depend on the order the elements arrive in (Sections 4.2-4.3). For outputs with no natural order, they let the model search over output orders during training (Section 5.2, Eq. 9).

---

## What the Authors Actually Showed

- **Input order matters (Section 4.1):** reversing the source sentence gave 5.0 BLEU in translation and 0.5% F1 in parsing; sorting convex-hull points by angle before a Pointer Network raised accuracy by up to 10% absolute.
- **Sorting a set of numbers (Section 4.4, Table 1):** exact-sort accuracy for N numbers in [0, 1], each N trained and tested separately (there is no train-on-short, test-on-long experiment). Without / with one glimpse; P = number of process steps:

| N | Ptr-Net (LSTM encoder) | P = 0 | P = 1 | P = 5 | P = 10 |
|---|---|---|---|---|---|
| 5 | 81% / 90% | 65% / 84% | 84% / 92% | 88% / 94% | 90% / 94% |
| 10 | 8% / 28% | 7% / 30% | 14% / 44% | 17% / 57% | 19% / 50% |
| 15 | 0% / 4% | 1% / 2% | 0% / 5% | 2% / 4% | 0% / 10% |

  With P = 0 and no glimpse the writer points "blindly" and is worst; one or more process steps beat the LSTM-encoder baseline; glimpses help every model.
- **Output order matters (Section 5.1):** Penn Treebank language modelling: natural and reversed word order both reach 86 perplexity, reversing every 3 words gives 96. Parsing: depth-first tree linearisation 89.5% F1, breadth-first 81.5%. Sorting trained on random orderings of the target "never reached the same performance".
- **Letting the model choose the output order (Section 5.2, Eq. 9):** maximise the likelihood over orderings, approximated by pretraining with a uniform prior over orderings and then sampling orderings from the model; on 5-grams it settles on natural-like orders (perplexity 225, vs 280 for a bad fixed order).

Convex hull and the travelling salesman problem (TSP) are **Pointer Networks** experiments (Vinyals, Fortunato, Jaitly 2015; Day 18), not results of this paper. This folder uses them as exercises and as extra `train.py` tasks.

---

## The Core Idea: Read-Process-Write

```
Input set {x_1, ..., x_n}             (in any order)
   |
[ READ ]    m_i = MLP(x_i)                         the same small MLP for every element
   |                                               -> one memory vector per element
[PROCESS]   for t = 1..T  (an LSTM with no data input, Eqs. 3-7):
               q_t     = LSTM(q*_{t-1})
               a_{i,t} = softmax_i(m_i . q_t)      content-based attention
               r_t     = sum_i a_{i,t} m_i
               q*_t    = [q_t, r_t]                -> q*_T: the same for every order of the set
   |
[ WRITE ]   pointer-network LSTM started from q*_T: at each output step
            (optional glimpse = one more attention read), point at one memory;
            the chosen memory is the next input    -> output indices
```

Why the order cannot leak in: the read block treats each element on its own; the process block touches the memories only through a softmax-weighted sum, and permuting the memories permutes the scores and weights together while only reordering the terms of the sum; the write block scores each memory on its own and is fed memories by content. So shuffling the input set shuffles the model's output distributions over elements in exactly the same way, and nothing else changes.

Until 2026-10-18 this folder's `ReadProcessWrite` had no process block: a Transformer encoder (as read block) fed a Pointer Network whose LSTM re-read the encoded set in order, so the whole model depended on the input order. It is now the paper's model.

---

## The Architecture (as implemented in `implementation.py`)

### 1. READ (`ReadBlock`)
$m_i = W_2\,\mathrm{ReLU}(W_1 x_i + b_1) + b_2$, the same MLP for every element ("a small multilayer perceptron for each number", Section 4.4). No element sees any other, so shuffling the inputs just shuffles the memories (permutation *equivariance*).

### 2. PROCESS (`ProcessBlock`, Eqs. 3-7)
An LSTM cell whose only input is its own previous read-out $q^*_{t-1}$ ($q^*_0 = 0$), with $f(m_i, q_t) = m_i \cdot q_t$ (a dot product, the paper's example). After $T$ steps $q^*_T = [q_T, r_T]$ summarises the set and is permutation-*invariant*, because the memories enter only through $r_t = \sum_i a_{i,t} m_i$. With $T = 0$, $q^*_0 = 0$ says nothing about the set.

### 3. WRITE (`PointerDecoder`)
- $q^*_T$ sets the decoder's initial state: $(h_0, c_0) = W q^*_T + b$.
- Inputs: a learned "go" vector at step 1, then the memory of the element chosen at the previous step.
- Glimpse (`glimpses=1`, Section 4.3): before pointing, the decoder state $d_t$ reads the memories once more, $g_t = \sum_i \mathrm{softmax}_i(w^T \tanh(G_1 m_i + G_2 d_t))\, m_i$, and the pointer uses the query $[d_t, g_t]$.
- Pointer (Pointer Networks):

```
u_i^t = v^T * tanh(W_1 * m_i + W_2 * query_t)
p(y_t = i | y_1, ..., y_{t-1}, X) = softmax_i(u^t)
```

  Padding slots and, by default, elements already chosen get $u = -\infty$ (`mask_chosen=True`).
- STOP slot (`stop_token=True`): one extra learned memory at index N; pointing at it ends a variable-length output (Pointer Networks' end-of-sequence symbol). Used for convex hulls.

**Choices the paper leaves open** (it promised exact equations in an appendix that the published version does not have): $f$ = dot product; $q^*_T$ entering through the decoder's initial state; the glimpse formula (the pointer's additive form, query $[d_t, g_t]$, like $q^*_t = [q_t, r_t]$); masking of chosen elements (the paper does not say; Pointer Networks did not mask during training); the STOP slot.

The order-sensitive baseline, `PointerNetwork` (an LSTM reads the elements in the order given), is the "Ptr-Net" of Table 1. `SetEncoder` (a Transformer without positional encodings) is kept for exercise 2 and for rebuilding the old model in the validation scripts; it is not used by `ReadProcessWrite`.

---

## Results in this folder

Measured on one CPU core (2026-10-18).

**Default `python train.py`** (sorting N = 5 numbers in [0, 10); 5 epochs of 5,000 fresh sets, batch 32, hidden 64, chosen elements masked; greedy exact sorts on 1,000 fresh sets):

| Model | Exact sorts | CPU time |
|---|---|---|
| Before 2026-10-18 (Transformer read, LSTM re-encoding, no process block) | 95.3% | 13 s |
| Read-Process-Write, P = 5, 1 glimpse | 0.0% untrained, 100.0% from epoch 1 on | 13 s |

**Order invariance** (`validation/day16_process_block_check.py`): shuffling the set changes $q^*_T$ by at most 2e-16 in float64 (2e-7 in float32), and the output distributions are the original ones permuted, to 1e-16 (16 configurations, with padding). After 800 training steps on sorting, the old model gave a different answer for 4.9% of 2,000 test sets once they were shuffled; Read-Process-Write for 0%.

**A small Table 1** (`validation/day16_sorting_table1.py --full`: numbers in [0, 1], hidden 64, batch 128, one seed, 2,000 test sets per N, chosen elements **not** masked; the paper's numbers in brackets):

| N (training steps) | Ptr-Net | P = 0, no glimpse | P = 0, 1 glimpse | P = 5, no glimpse | P = 5, 1 glimpse |
|---|---|---|---|---|---|
| 5 (1,500) | 85.5% (81%) | 71.7% (65%) | 97.5% (84%) | 95.5% (88%) | 99.6% (94%) |
| 10 (2,500) | 33.2% (8%) | 32.4% (7%) | 92.8% (30%) | 57.1% (17%) | 93.2% (57%) |

The paper's pattern appears. Without process steps or a glimpse, Read-Process-Write is no better than the LSTM-encoder baseline (71.7% vs 85.5% at N = 5; 32.4% vs 33.2% at N = 10); five process steps lift it above the baseline (95.5%; 57.1% at N = 10); a glimpse helps most. With a glimpse, P = 5 and P = 0 differ by less than the noise of a single seed at N = 10. Our absolute numbers are higher than the paper's after far fewer steps; the paper gives no batch size, hidden size or optimiser, so they are not directly comparable. (The `--full` run took 6 CPU-minutes; the default, 600 steps per N, about 2.)

**Masking changes the picture.** With chosen elements masked (`train.py`'s default), sorting reduces to "point at the smallest remaining element", which a per-element read block plus a content-based pointer can do without knowing anything else about the set. In the same comparison with masking (`day16_sorting_table1.py --mask`, 600 steps per N), every Read-Process-Write variant, P = 0 without a glimpse included, sorts 100% of the 2,000 test sets for N = 5 and N = 10, while the LSTM-encoder Ptr-Net reaches 90.6% and 68.2%. It can happen without any training: for 2 of 6 random initialisations an *untrained* Read-Process-Write (hidden 32, masking on) sorted 99.9% and 100% of 2,000 test sets of 5 numbers, and for 2 others it reverse-sorted 98.8% and 100% of them, because its random pointer score happened to be monotone in the value (`validation/day16_process_block_check.py`, item 8). Without masking the decoder must also keep track of what it has already output, and that is where the process block and glimpses earn their keep.

**Convex hull** (a Pointer Networks task): `python train.py --task hull --set-size 10 --epochs 20` (100,000 fresh sets of 10 points, hidden 64, targets in canonical order ending with STOP; 84 s of CPU) reproduces the exact hull, every vertex in order and then STOP, for 74.8% of 1,000 validation sets with greedy decoding (validation NLL 0.12). Pointer Networks reported 87.0% at n = 10 with a 256- or 512-unit model trained on 1M examples of 5-50 points, counting any output that describes the same polygon. Before 2026-10-18 `--task hull` refused to run: the padded labels gave an infinite loss.

**TSP** (also a Pointer Networks task): with tours that start at the leftmost city, `python train.py --task tsp --set-size 10 --epochs 3` reproduces the nearest-neighbour teacher tour exactly for 57.4% of 500 validation sets.

---

## Implementation Notes

When implementing Read-Process-Write and Pointer Networks, pay attention to these details:

- **Handling sets**: anything that shows the model positions (positional encodings, an LSTM reading the elements in order) breaks permutation invariance. Test it: shuffle the set and compare (`validation/day16_process_block_check.py`).
- **Masking chosen elements**: setting a chosen element's score to `-inf` before the softmax guarantees the output is a permutation. It is a modelling choice the paper does not mention, and it makes sorting much easier (Results).
- **Variable lengths**: Pointer Networks handle any input size because the "vocabulary" is the input set. For outputs of varying length, add a STOP slot; pad targets after it with `IGNORE_INDEX` (-1), which `compute_loss` skips. Never pad by repeating an index the decoder has already masked: its log-probability is `-inf`.
- **Canonical output order** (Section 5.1.3): give each example one target ordering. Hulls start at the leftmost point and go counter-clockwise; nearest-neighbour TSP tours start at the leftmost city. Pointer Networks started at the lowest *index*, but an order-invariant model never sees indices, so the start has to be defined by content.

**Things that will bite you:**
- **Teacher forcing**: during training the decoder is fed the correct previous element; measure accuracy with greedy decoding (or the model's own per-step argmax), never with the fed targets.
- **Infinite loops**: without a STOP pointer (or a fixed output length), the model might point forever.
- **Numerical stability**: use `log_softmax` for the NLL rather than taking the log of a softmax.

---

## What to Build

### Quick Start

```bash
# Install dependencies
pip install torch numpy matplotlib scipy tqdm

# Sort 5 numbers with Read-Process-Write (about 20 s on one CPU core)
python train.py

# Paper-like setting (no masking): process steps and glimpses vs the LSTM-encoder baseline
python train.py --set-size 10 --no-mask --process-steps 0 --glimpses 0
python train.py --set-size 10 --no-mask --process-steps 5 --glimpses 1
python train.py --set-size 10 --no-mask --model ptrnet

# Convex hull with a STOP pointer (a Pointer Networks task)
python train.py --task hull --set-size 10 --epochs 20

# Shuffle tests and the small Table 1
python ../../validation/day16_process_block_check.py
python ../../validation/day16_sorting_table1.py
```

`visualization.py` holds plotting helpers (`plot_attention_heatmap`, `visualize_sorting`) to call from Python; it has no command line.

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|---------------------------|
| 1 | Pointer attention mechanism | Understand how decoder state selects input elements |
| 2 | Self-attention set encoder | Per-element outputs are equivariant, their mean is invariant: f(permute(X)) = f(X) |
| 3 | Train Pointer Network for sorting | End-to-end training loop for mapping sets to sorted indices |
| 4 | Convex hull with a STOP pointer | Variable-length outputs and a canonical output order (a Pointer Networks task) |
| 5 | TSP approximation | Learn a heuristic for NP-Hard combinatorial optimization (a Pointer Networks task) |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **Input order matters for standard seq2seq models.** The same set of elements produces different representations depending on the order the encoder reads them, and it changes results (Section 4.1).

2. **Content-based attention gives an order-invariant summary of a set.** The process block reads per-element memories only through attention-weighted sums (Eqs. 3-7), so $q^*_T$ is the same for every order of the set (Section 4.2). Self-attention without positional encodings is only *equivariant*; a symmetric pooling such as this attention read-out is what makes the summary invariant.

3. **Pointer Networks output indices, not vocabulary tokens.** The decoder "points" back to input elements using attention scores as a selection mechanism, which allows variable-size output dictionaries (Pointer Networks; the write block here).

4. **Output order matters too.** Among the many equivalent ways to write an output, one consistent (canonical) order trains much better than random ones (Section 5.1), and when no good order is known the model can search for one while training (Section 5.2, Eq. 9).

---

## Files in This Directory

| File | What It Is |
|------|------------|
| `implementation.py` | Read-Process-Write (`ReadBlock`, `ProcessBlock`, `PointerDecoder`), the Ptr-Net baseline, `SetEncoder` |
| `train.py` | Training script: tasks sort, hull, tsp; models rpw, ptrnet |
| `visualization.py` | Attention heatmaps and pointer trajectory plots |
| `notebook.ipynb` | Pointer attention, a sorting Ptr-Net, a Read-Process-Write shuffle test |
| `exercises/` | 5 exercises with solutions |
| `PAPER_NOTES.md` | Condensed notes on the original paper |
| `CHEATSHEET.md` | Quick reference for architecture and hyperparameters |
| `requirements.txt` | Python dependencies |

---

## Further Reading

- [Original Paper - Order Matters (arXiv:1511.06391)](https://arxiv.org/abs/1511.06391)
- [Pointer Networks (Vinyals et al., 2015)](https://arxiv.org/abs/1506.03134) - the predecessor that introduced the pointer mechanism
- [Set Transformer (Lee et al., 2019)](https://arxiv.org/abs/1810.00825) - a later approach to permutation-invariant set encoding

---

**Next:** [Day 17 - Neural Turing Machines](../17_neural_turing_machines/)
