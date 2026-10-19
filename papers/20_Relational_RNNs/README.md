# Day 20: Relational Recurrent Neural Networks

> Adam Santoro et al. (2018) — [Relational recurrent neural networks](https://arxiv.org/abs/1806.01822)

**Time:** 4-6 hours
**Prerequisites:** Day 19 (Relation Networks), basic Attention mechanism
**Code:** PyTorch

---

## What This Paper Is Actually About

Memory-based networks are good at storing information for a long time, but can they *reason about* what they stored? An LSTM packs everything into one hidden vector, which makes it hard to keep separate facts apart. Slot memories such as the NTM and DNC keep facts in separate rows, but once written the rows have no direct way to interact (Sec. 2).

This paper introduces the **Relational Memory Core (RMC)**. Memory is a small matrix of **slots** (rows). At every time step the slots run multi-head dot-product attention over each other *and over the new input*, so the model can relate new information to old information before deciding what to keep.

It sits between the fixed-size state of an RNN and the relational power of a Transformer: the same attention, but over a constant number of slots rather than over the whole history.

---

## The Core Idea

**Recurrent attention over a fixed set of slots** (Sec. 3):

1.  **Memory matrix**: the state is $M$ with $N$ slots of width $F$.
2.  **Interact and write in one attention**: the input is projected to one row $\tilde{x}$ and appended to the memory; the slots attend over the $N+1$ rows $[M; \tilde{x}]$ (Eq. 2). Queries come only from the slots, so the memory keeps its size, and the extra column of the $N \times (N+1)$ attention matrix says how much each slot reads the input.
3.  **Gated update**: the attended memory goes through a row-wise MLP and is gated into $M_t$ like an LSTM cell state (Eqs. 3-9).

Because all weights are shared across slots, $N$ sets the capacity ($N \times F$ numbers) without changing the parameter count.

---

## What the Authors Actually Showed

1.  **Nth Farthest** (Sec. 4.1, App. A.1, Sec. 5.1): 8 random 16-dim vectors with shuffled labels; every input also carries the question "what is the $n$-th farthest vector from vector $m$?" ($n$, $m$ sampled per sequence); the answer is a label. The RMC "consistently" reached **91%**; LSTM and DNC stayed **below 30%** best batch accuracy. Note that 25% is free: in the dataset code the paper links to (Sonnet), vector $m$ itself is one of the 8 ranks, so when $n$ asks for the 8th farthest vector the answer is $m$, which is in the input.
2.  **Program evaluation** (Learning to Execute, Table 1): best or tied on five of six tasks, e.g. 79.0% vs 66.1% (LSTM) per-character accuracy on full programs; on double memorization the DNC's 100.0 edges its 99.8.
3.  **Reinforcement learning** (Sec. 5.3): Mini PacMan with a 5x5 viewport, 677 vs 550 points for an LSTM agent.
4.  **Language modeling** (Table 2): test perplexity **31.6** on WikiText-103 (LSTM 34.3, QRNN 33), 42.0 on Project Gutenberg, 38.3 on GigaWord. No fewer-parameters claim is made.

---

## The Architecture (what `implementation.py` computes)

`RelationalMemory` follows Sec. 3 and Fig. 1, and DeepMind's Sonnet code (`relational_memory.py`, the paper's footnote 2) for the details the paper leaves out:

1.  **Input row**: $\tilde{x} = W_{in} x_t$, one row of width $F$.
2.  **MHDPA (Eq. 2)**: one linear map gives $q, k, v$ for every head and every row of $[M; \tilde{x}]$ (Sonnet also LayerNorms them); queries from the $N$ slots, keys/values from all $N+1$ rows; scaled dot-product softmax; heads concatenated back to width $F$. Attention weights: $(B, H, N, N+1)$.
3.  **Residual + LayerNorm**, then a **row-wise MLP** $g_\psi$ (2 layers, ReLU) with another **residual + LayerNorm** (Fig. 1a; "a row/memory-wise MLP with layer normalisation", Sec. 3.3). Call the result $G$.
4.  **Gates (Eqs. 4-5, 7)**, one set per row: $[i; f] = W\tilde{x} + U\tanh(M_{t-1})$, $i = \sigma(i)$, $f = \sigma(f + 1)$. The $\tanh(M_{t-1})$ is the paper's $h_{t-1}$ once the output gate is dropped ("we did not find output gates necessary").
5.  **Update**: $M_t = i \odot \tanh(G) + f \odot M_{t-1}$. The core's output is $M_t$ flattened to $N \cdot F$ numbers.

`RelationalRNN` unrolls the core over a sequence and feeds the final flattened memory to an output MLP; `StandardLSTM` feeds its final hidden state to the same kind of MLP.

---

## Implementation Notes

-   **Initial memory**: the $N \times N$ identity, zero-padded to $N \times F$, not trained (Sonnet). The rows must differ: the update treats rows symmetrically, so identical rows would stay identical. `validation/day20_rmc_paper_check.py` checks this equivariance: permuting the slots permutes the new memory and the attention.
-   **Weight initialisation**: every `nn.Linear` in the core starts like Sonnet's: truncated-normal weights with std $1/\sqrt{\text{fan-in}}$ and zero biases, so the constant forget bias (+1) is the only gate bias at the start.
-   **`num_blocks` > 1**: each block re-reads the same input row. Sonnet runs later blocks over all $N+1$ rows (updating the input row too); the two agree for `num_blocks=1`, the setting the paper used for Nth Farthest.
-   **`gate_style`**: `'unit'` (one gate per unit, Sonnet's default) or `'memory'` (one scalar gate per slot, the paper's "memory gating").
-   **Earlier version**: until 2026-10-18 this file held a simplified cell (input tiled onto every slot as extra features, $N \times N$ attention, no MLP, a residual + LayerNorm that was computed and then discarded). `validation/day20_rmc.py` still checks that version, read from git.

---

## The Task We Run, and What a CPU Run Shows

`train_minimal.py` trains the RMC and an LSTM on the paper's Nth Farthest task, generated as in App. A.1 and Sonnet's `dataset_nth_farthest.py`: $K$ vectors $x \sim U(-1,1)^D$, a random permutation of labels, $(n, m)$ one-hot at every step, input $(x_t; l_t; n; m)$ of $D + 3K$ numbers, cross-entropy over the $K$ labels, Adam, fresh batches every step. Both models get the same output MLP (the paper's 4 layers of 256), and the LSTM's hidden size is chosen to match the RMC's total parameter count (302,787 vs 303,363 parameters by default).

Counting is 0-based in the code: $n = 0$ asks for the farthest vector, and $n = K-1$ for the nearest, which is $m$ itself. Those questions are answered by copying $m$ from the input, so **$2/K$ accuracy needs no relational reasoning at all**. The script reports accuracy per $n$ and on the other, "relational" questions (chance $1/(K-1)$).

| | Paper (App. A.1) | `train_minimal.py` default |
|---|---|---|
| Task | 8 vectors x 16 dims (40-d inputs) | 3 vectors x 2 dims (11-d inputs) |
| Questions | $n$, $m$ uniform for every sequence | the same, after a curriculum: the first 1500 of 3000 steps ask only for the farthest vector ($n = 0$). Evaluation always uses uniform $n$ |
| RMC | 1/8/16 slots, 2048 total units, 1/8/16 heads, 1 block | 4 slots x 64, 4 heads, 1 block, 'unit' gating |
| LSTM | hidden 64 ... 4096 | hidden 128 (matched to the RMC's parameter count) |
| Training | Adam, batch 1600, lr 1e-4, hours on a GPU | Adam, batch 128, lr 1e-3, about 2 minutes on one CPU core |

**Why the curriculum (a deviation from the paper).** With $n$ uniform, both models learn the free $2/K$ within a few hundred steps and then stop: averaged over $n$, every label other than $m$ is equally likely whatever the vectors are, so no feature of the vectors improves the loss until the model combines it with $n$. This is why the paper's LSTM and DNC stay below 30% on 8 vectors, where 25% is free (Sec. 5.1, Fig. 5); the paper's RMC got past it after hours on a GPU, and so did a public PyTorch reproduction ([L0SG/relational-rnn-pytorch](https://github.com/L0SG/relational-rnn-pytorch)), which calls it "the 25% barrier". On a CPU budget neither of our models broke it (below). Asking only for the farthest vector first removes the cancellation; once the RMC ranks distances, switching to every $n$ takes a few hundred steps. Use `--curriculum_steps 0` for the paper's protocol.

### Results (one CPU core; accuracy on 3000 held-out sequences with uniform $n$)

| Run | RMC | LSTM | Time |
|---|---|---|---|
| Default, seeds 0 / 1 / 2 | 94.7 / 95.4 / 94.5% (relational 92.1 / 93.2 / 91.7%) | 67.7 / 67.3 / 67.5% (relational 51.7 / 51.1 / 51.5%) | 2.0 / 2.2 / 1.7 min |
| `--curriculum_steps 0` (paper's protocol), seed 0 | 67.7% (relational 51.7%) | 67.7% (relational 51.7%) | 2.2 min |
| `--full` (12000 steps, the first 6000 farthest only), seed 0 | 97.2% (relational 95.8%) | 67.3% (relational 51.1%) | 12.0 min |

Baselines: chance 33.3%; copying $m$ when $n = K-1$ and guessing otherwise 66.7%; chance on the relational questions 50%. Both models answer every copy-$m$ question correctly.

What this shows, and what it does not:
- **The gap the paper reports appears at small scale.** With the curriculum, the RMC ranks the distances (about 92-93% on the relational questions) while a parameter-matched LSTM, trained on the same batches, never gets past the copy-$m$ level: during the farthest-only phase it stays at "never answer $m$" (50% between the two candidates), even for 6000 steps in the `--full` run.
- **But only with a curriculum the paper did not use.** Under the paper's protocol both models sit at $2/K$ for the whole budget, the same barrier as the paper's baselines; in an exploratory run the RMC stayed there for 1.3M training sequences (5000 steps of batch 256, then with PyTorch's default initialisation instead of Sonnet's). This is a CPU-sized demonstration on 3 vectors of 2 dims, not a reproduction of 91% vs 30% on 8 vectors of 16 dims.
- **Size matters a lot.** At 4 vectors (2 dims) the RMC did not learn even the farthest-only question within 3000 steps; at the paper's 8 x 16 both models reached the 25% shortcut within a minute and made no further progress in our short runs.

The repo's previous task (regress the vector farthest from vector 0 with MSE) is gone: a fixed predictor $-c\,v_0$ ($c \approx 0.39$) beat both models on it in the first-round run (old cell 0.381, LSTM 0.372). `validation/day20_regression_task.py` reruns it with the old settings on 2000 validation sequences: the new RMC reaches MSE 0.341, just below the best $-c\,v_0$ (0.349; predicting 0 gives 0.398), and the LSTM 0.380.

---

## What to Build

### Quick Start

```bash
cd papers/20_Relational_RNNs

# Verify your RMC pieces
python exercises/exercise_01_attention.py
python exercises/exercise_02_gating.py

# RMC vs LSTM on the Nth Farthest task (plots: --save_plot, --visualize)
python train_minimal.py --save_plot --visualize
```

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|--------------------------|
| 1 | **Dot-product attention** (`exercise_01_attention.py`) | N queries against N+1 keys: the attention of Eq. 2, checked against PyTorch's built-in |
| 2 | **RMC gating** (`exercise_02_gating.py`) | Input and forget gates from $\tilde{x}$ and $\tanh(M)$, forget bias +1, and their open/closed limits |
| 3 | **One RMC step** (`exercise_03_rmc_step.py`) | Append the input as a row, attend, MLP, gate; must match `implementation.RelationalMemory` |
| 4 | **Nth Farthest data** (`exercise_04_nth_farthest.py`) | The labelled task of App. A.1, checked against a decode-and-sort of its own inputs |
| 5 | **Training loop** (`exercise_05_train.py`) | Train the RMC with cross-entropy and the short curriculum, and read accuracy against the right baselines |

Solutions are in `exercises/solutions/`. Try to get stuck first!

---

## Key Takeaways

1.  **One attention both relates and writes**: appending the input as a row lets the same softmax mix the slots and decide how much each slot reads the observation (Eq. 2).
2.  **Attention inside a recurrent cell**: a constant-size state with Transformer-style interactions, at a cost per step that does not grow with the sequence.
3.  **Read accuracy against the right baselines**: on Nth Farthest, $2/K$ comes free from the question itself; only the remaining questions measure relational reasoning.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | `RelationalMemory` (the RMC core), `RelationalRNN`, `StandardLSTM` |
| `train_minimal.py` | Nth Farthest data generator and the RMC-vs-LSTM training run |
| `visualization.py` | Attention heatmaps ($N \times (N+1)$, input column labelled) and accuracy curves |
| `paper_notes.md` | Notes and equations from the paper |
| `notebook.ipynb` | Interactive walkthrough: build the RMC, look at its attention, a short training run |
| `exercises/` | Step-by-step implementation tasks |

Checks live in the repo's `validation/` folder: `day20_rmc_paper_check.py` (this core and task), `day20_exercises.py`, `day20_worked_example.py`, `day20_regression_task.py` (the old task), and `day20_rmc.py` (the earlier simplified cell, from git).

---

## Further Reading

- [Original Paper (Santoro et al., 2018)](https://arxiv.org/abs/1806.01822): Sec. 3 for the model, App. A.1 for the task
- [Sonnet `relational_memory.py`](https://github.com/google-deepmind/sonnet/blob/v1/sonnet/python/modules/relational_memory.py): the reference implementation the paper links to
- [DeepMind blog: Relational deep reinforcement learning](https://deepmind.google/discover/blog/relational-deep-reinforcement-learning/): companion work from the same group (BoxWorld), not an overview of this paper
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): the origin of the attention mechanism used here

---

**Previous:** [Day 19: Relational Reasoning](../19_relational_reasoning/)
**Next:** [Day 21: Neural Message Passing](../21_Neural_Message_Passing/)
