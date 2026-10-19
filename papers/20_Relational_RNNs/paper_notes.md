# Paper Notes: Relational Recurrent Neural Networks

## ELI5 (Explain Like I'm 5)

### The Memory Cocktail Party

Imagine a standard RNN (like an LSTM) as a person trying to remember a long story. They have a notebook (the hidden state), and at each step, they write down new information and cross out old stuff. But they only look at *one page* at a time. If they need to connect a detail from Page 1 to Page 50, they have to hope they copied that detail forward over and over again for 49 pages.

The **Relational Memory Core (RMC)** is like a cocktail party inside your head. instead of one single notebook page, you have distinct "memory slots"—like different people at a party. At every time step, before taking in new information, these memory slots *talk to each other*.
- Memory A says: "I'm holding the color red."
- Memory B says: "I'm holding the shape square."
- They interact (via attention) and realize: "Oh, we're looking for a red square!"

By letting memories interact *at each step*, the network doesn't just store facts; it reasons about how those facts relate to each other, even if they happened thousands of steps apart.

> **Note:** This analogy is ours, not the authors'.

---

## What the Paper Actually Covers

The paper (Santoro et al., 2018) introduces the **Relational Memory Core (RMC)**. It targets a weakness of memory-based networks: LSTMs pack everything into one hidden vector (hard to keep facts apart), while slot memories such as the NTM and DNC keep facts apart but give the stored slots no direct way to interact (Sec. 2).

- **Section 3 (Model):** 3.1 lets memories interact with multi-head dot-product attention (Eq. 1); 3.2 writes new input with the same attention by appending it as an extra row (Eq. 2); 3.3 adds recurrence by embedding the update in an LSTM-style gated cell (Eqs. 3-9).
- **Section 4 (Experiments):** Nth Farthest and program evaluation (Learning to Execute) as supervised tasks; BoxWorld and Mini PacMan with a viewport as RL tasks; language modeling on WikiText-103, Project Gutenberg and GigaWord. The appendix gives the exact configurations.
- **Section 5 (Results):** Nth Farthest 91% for the RMC vs below 30% for LSTM and DNC (5.1); Table 1 for program evaluation; Mini PacMan 677 vs 550 for an LSTM with the viewport (5.3); Table 2 for language modeling (5.4).
- **Section 6 (Discussion):** fixed-size recurrent memory vs growing buffers of past states, and slots vs heads as two ways to compartmentalize.

---

## The Core Idea

The Big Idea is **attention between memories, inside the recurrent update**.

In a Transformer, attention runs over the whole history (all past tokens). In an RMC it runs **among the memory slots, plus the current input, at the current time step** ("we apply attention between memories at a single time step, and not across all previous representations", Sec. 3).

1.  **Fixed slots**: Memory is a matrix $M$ of size $N \times F$ ($N$ rows/slots, $F$ features per slot).
2.  **Interaction and writing in one step**: The slots attend over $[M; x]$, the memory with the input appended as one extra row; queries come from the slots only, so the result keeps $M$'s shape.
3.  **Gated update**: The attended memory goes through a row-wise MLP and is gated into $M_t$ like an LSTM cell state.

---

## The Math

### 1. Multi-Head Dot Product Attention (Sec. 3.1, Eq. 1)
Standard Transformer attention, applied to the rows of $M$:

$$ A_\theta(M) = \text{softmax}\Big(\frac{MW^q (MW^k)^T}{\sqrt{d_k}}\Big) MW^v $$

Each head has its own $W^q, W^k, W^v$; the heads' outputs are concatenated column-wise back to width $F$.

### 2. Writing the input (Sec. 3.2, Eq. 2)

$$ \tilde{M} = \text{softmax}\Big(\frac{MW^q ([M; x]W^k)^T}{\sqrt{d_k}}\Big) [M; x]W^v $$

$[M; x]$ is the **row-wise** concatenation, $(N+1) \times F$. Queries come from $M$ only, so the score matrix is $N \times (N+1)$ and $\tilde{M}$ is $N \times F$; the extra column says how much each slot reads the input.

### 3. The gated update (Sec. 3.3, Eqs. 3-9)
Each row $m_i$ is treated like an LSTM cell state, with $h_{i,t}$ its output:

$$ f_{i,t} = W^f x_t + U^f h_{i,t-1} + b^f, \qquad i_{i,t} = W^i x_t + U^i h_{i,t-1} + b^i $$
$$ m_{i,t} = \sigma(f_{i,t} + \tilde{b}^f) \circ m_{i,t-1} + \sigma(i_{i,t}) \circ g_\psi(\tilde{m}_{i,t}) $$
$$ h_{i,t} = \sigma(o_{i,t}) \circ \tanh(m_{i,t}) $$

The change from an LSTM is the candidate: $g_\psi(\tilde{m}_{i,t})$, the attended row passed through a row-wise MLP, instead of a fresh $\tanh(Wx + Uh)$. "In practice we did not find output gates necessary"; without one, $h_{i,t-1} = \tanh(m_{i,t-1})$. The paper points to its Sonnet code for "the exact formulation", which `implementation.py` follows:

- $\tilde{x} = W_{in} x_t$ is the input projected to one memory-sized row; it is the row appended in Eq. 2.
- $g_\psi$: residual + LayerNorm after the attention, then a 2-layer MLP with residual + LayerNorm ("a row/memory-wise MLP with layer normalisation", Sec. 3.3). Sonnet also LayerNorms the q, k, v projections.
- Gates: $[i; f] = W\tilde{x} + U \tanh(M_{t-1})$, input bias 0, forget bias $+1$.
- $M_t = \sigma(i) \circ \tanh(G) + \sigma(f) \circ M_{t-1}$, with $G$ the MLP block's output (Sonnet adds the $\tanh$).
- 'unit' gating gives one gate per unit; 'memory' gating gives one scalar per row (Sec. 3.3).
- The initial memory is the identity matrix padded to $N \times F$, and the core's output is the flattened memory.

---

## The Experiments

### Nth Farthest (Sec. 4.1, App. A.1, results Sec. 5.1)
- **Setup**: 8 vectors of 16 dims, $x \sim U(-1, 1)$, each with a randomly assigned label 1-8 (one-hot, so labels do not match time steps). Every step also carries $n$ and $m$ as one-hots: a 40-dim input $(x_t; l_t; n; m)$. The question is "what is the $n$-th farthest vector from vector $m$?" and the answer is a label, trained with softmax cross-entropy; $n$ and $m$ are sampled per sequence. All models feed their output through the same 4-layer MLP (256 units). Adam, batch 1600, learning rate $10^{-4}$.
- **Why it matters**: vector $m$ may arrive at any step, even last, so the model has to keep every vector and later rank all distances to $m$: it sorts *distance relations*, not the vectors themselves.
- **Result**: the RMC "consistently" reached 91%, while LSTM and DNC never passed 30% best batch accuracy. With 32-dim vectors the RMC got similar accuracy but less robustly. There is no sequence-length sweep.
- **A free 25%**: in Sonnet's dataset code (the implementation the paper links to) vector $m$ itself is one of the 8 ranks, so when the question asks for the 8th farthest vector the answer is $m$, which is in the input. Copying $m$ for those questions and guessing among the other 7 labels otherwise gives $1/8 + 7/8 \cdot 1/7 = 2/8 = 25\%$, so "below 30%" means the baselines learned almost nothing beyond that.
- **This repo**: `train_minimal.py` runs the same task with 3 vectors of 2 dims. Under the paper's protocol both the RMC and a parameter-matched LSTM stop at the free $2/K$ (67.7%) in a CPU budget. With a short curriculum that first asks only for the farthest vector (not in the paper), the RMC reaches about 95% on every $n$ while the LSTM stays at 67.5%; see the README.

### Language modeling (Sec. 4.3, results Sec. 5.4, Table 2)
- **Setup**: word-level language modeling on WikiText-103, Project Gutenberg and GigaWord v5. The RMC used a single, large memory slot here.
- **Result**: test perplexity 31.6 on WikiText-103, against 34.3 for a tuned LSTM and 33 for the QRNN; 42.0 on Gutenberg and 38.3 on GigaWord. The paper makes no fewer-parameters claim.

---

## Going Beyond the Paper (Our Retrospective)

> **Note:** This section is our analysis, looking back from 2024.

The RMC is a fascinating "missing link" between RNNs and Transformers.
- **RNNs**: State is a vector, update is linear/nonlinear.
- **Transformers**: State is the entire history, update is attention.
- **RMC**: State is a matrix (fixed size), update is attention.

It represents the attempt to bring the power of Attention (O(N^2) interactions) into a Recurrent layout (O(1) memory size). While pure Transformers eventually won the "scale" war (because keeping *all* history is better than compressing it, if you have the compute), architectures like RMC are seeing a resurgence in **Linear Attention** and **State Space Models (Mamba/S4)**, which try to compress history into a fixed state efficiently.

---

## Questions Worth Thinking About

1.  **Why fixed slots?** Why not let the number of memory slots grow? (Hint: Computational cost and memory bounds).
2.  **Is it truly relational?** Does attention *guarantee* relational reasoning, or just correlation?
3.  **The "Binding Problem"**: How does the model know that "Slot 1" contains the "Subject" and "Slot 2" contains the "Object"?
