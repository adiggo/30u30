# Day 20 Cheat Sheet: Relational Recurrent Neural Networks

## The Big Idea (30 seconds)

**Problem:** An LSTM packs everything into one vector, so separate facts blur; slot memories (NTM, DNC) keep facts apart but give the slots no direct way to interact once written.
**Insight:** Keep a small matrix of memory slots and let the slots attend to each other *and to the new input* at every step (multi-head dot-product attention), then gate the result into memory like an LSTM.
**Result:** On the Nth Farthest task the RMC reached 91% while LSTM and DNC stayed below 30% (Sec. 5.1); it also set the then state of the art on WikiText-103 (test perplexity 31.6 vs 34.3 for the LSTM, Table 2).

**The one picture to remember** (Fig. 1a; what `implementation.py` computes):

```text
 M_{t-1}: N slots x F          x_t
   [ m_1 ]                       |  Linear -> x~ (one row of width F)
   [ ... ]                       |
   [ m_N ]   [M; x~]: N+1 rows <-+   append the input as an extra ROW
      |
      |  MHDPA: queries from the N slots, keys/values from all N+1 rows
      |         -> N x (N+1) attention per head (last column = "input")
      v
   LN(M + attention)            residual + LayerNorm
   LN(. + MLP(.))               row-wise MLP g_psi + residual + LayerNorm   -> G
      |
   M_t = i * tanh(G) + f * M_{t-1}     i, f = sigmoid(W x~ + U tanh(M_{t-1}) + [0; +1])
      |
   output = flatten(M_t)  (N*F numbers) -> output MLP -> logits
```

---

## Quick Start

```bash
cd papers/20_Relational_RNNs

# 1. RMC vs a parameter-matched LSTM on the reduced Nth Farthest task, with plots (~2 min)
python train_minimal.py --save_plot --visualize
python train_minimal.py --curriculum_steps 0     # the paper's protocol: both stay at 2/K

# 2. Unit checks of the core (shapes, equivariance, gate limits, gradients)
python ../../validation/day20_rmc_paper_check.py

# 3. Step-by-step implementation guide
python exercises/exercise_01_attention.py
```

---

## Key Parameters for the RMC (`RelationalMemory`)

| Parameter | Paper / Sonnet | Here (default) | What It Does |
|-----------|----------------|----------------|--------------|
| `mem_slots` (N) | 1, 8 or 16 (Nth Farthest) | 4 | Rows of memory. Weights are shared across rows, so N changes capacity, not the parameter count |
| `mem_size` (F) | 2048 / N total units | 64 | Width of each slot; `num_heads * head_size` |
| `num_heads` | 1, 8 or 16 | 4 | Each head has its own q/k/v projections; head outputs are concatenated back to width F |
| `num_blocks` | 1 | 1 | Rounds of attention + MLP per time step |
| `attention_mlp_layers` | 2 (Sonnet) | 2 | Depth of the row-wise MLP g_psi |
| `gate_style` | 'unit' or 'memory' | 'unit' | One gate per unit, or one scalar gate per slot |
| `forget_bias` | +1 | +1 | Added inside the forget-gate sigmoid: sigma(1) = 0.73, so memory persists by default |
| `qkv_layer_norm` | Sonnet: yes | True | LayerNorm on the q, k, v projections (not in the paper's equations) |

---

## The Math (Copy-Paste Ready)

### 1. Attention over [M; x~] (Eq. 2)

```python
mem_plus_input = torch.cat([memory, x_tilde], dim=1)        # (B, N+1, F): ROW-wise concat
q, k, v = split_heads(qkv_norm(qkv(mem_plus_input)))        # each (B, H, N+1, d)
q = q[:, :, :N]                                              # queries from the N memory rows only
weights = softmax(q @ k.transpose(-2, -1) / sqrt(d))         # (B, H, N, N+1)
attended = merge_heads(weights @ v)                          # (B, N, F): memory keeps its shape
```

### 2. MLP and gated update (Fig. 1a, Eq. 7, Sonnet's gates)

```python
m1 = norm_attention(memory + attended)                       # residual + LayerNorm
G = norm_mlp(m1 + mlp(m1))                                   # row-wise MLP + residual + LayerNorm
gates = gate_memory_proj(torch.tanh(memory)) + gate_input_proj(x_tilde)   # x~ broadcasts over rows
i, f = gates.chunk(2, dim=-1)
i, f = torch.sigmoid(i + 0.0), torch.sigmoid(f + 1.0)       # input bias 0, forget bias +1
next_memory = i * torch.tanh(G) + f * memory                 # no output gate
```

Why `tanh(memory)` in the gates: the paper's gates read the previous output h_{t-1}; with the output gate removed, Eq. 8 makes h_{t-1} = tanh(m_{t-1}).

---

## Common Issues & Fixes

### Accuracy stuck at 2/K (25% for 8 vectors, 50% for 4)
```text
That is the free part of the task: when the question asks for the K-th farthest vector
(n = K-1) the answer is m itself, which is in the input. Look at the accuracy on the other
questions (train_minimal.py prints it per n); chance there is 1/(K-1).
Why it sticks: with n uniform, averaged over n every label other than m is equally likely
whatever the vectors are, so nothing improves the loss until the model combines n with the
distances. The paper's RMC needed hours on a GPU; on a CPU, ask only for the farthest vector
first (train_minimal.py --curriculum_steps, default 1500 of 3000 steps).
```

### Attention matrix is N x N
```python
# Problem: the input was tiled onto every slot as extra features (the old simplified cell)
# Fix: append it as a ROW and take queries from the memory rows only
mem_plus_input = torch.cat([memory, x_tilde], dim=1)   # dim=1 (rows), not dim=-1 (features)
```

### Parameters that never get a gradient
```python
# Dead code passes shape tests. After one backward pass:
dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
```

### All slots end up identical
```python
# The step treats rows symmetrically (shared weights), so identical rows stay identical.
# Sonnet starts from the identity matrix (padded to N x F): every row is different.
```

---

## Comparison: RMC vs. Others

| Feature | LSTM | Transformer | Relational RNN (RMC) |
| :--- | :--- | :--- | :--- |
| **State** | One vector | Every past token (KV cache) | N slots (fixed) |
| **Mixing** | Gates | Attention over the history | Attention among slots + the current input |
| **Cost per step** | O(h^2) for hidden size h | t scores per head, grows with t | N(N+1) scores per head, constant |
| **Weak spot** | Relating stored facts | Memory grows with the sequence | Lossy: a slot that dropped a fact cannot get it back |

---

*For paper details, see [paper_notes.md](paper_notes.md). For implementation guide, see [README.md](README.md).*
