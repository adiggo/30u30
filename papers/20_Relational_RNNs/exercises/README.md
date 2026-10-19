# Exercises: Day 20 - Relational RNNs

Build the Relational Memory Core of Santoro et al. (2018) piece by piece, then train it on the paper's Nth Farthest task.

---

## Overview

| Exercise | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 1 | **Dot-Product Attention** | Easy (2/5) | 20 min |
| 2 | **RMC Gating** | Medium (3/5) | 30 min |
| 3 | **One RMC Step** | Medium (3/5) | 45 min |
| 4 | **Nth Farthest Data** | Easy (2/5) | 25 min |
| 5 | **Training Loop** | Medium (3/5) | 30 min |

---

## Exercise 1: Dot-Product Attention

Write `scaled_dot_product_attention` by hand. In the RMC the queries come from the N memory slots and the keys and values from the N+1 rows of [M; x~] (the memory plus the input row, Eq. 2), so the weights are N x (N+1). The test compares your output with PyTorch's `F.scaled_dot_product_attention`.

## Exercise 2: RMC Gating

Implement the RMC's LSTM-style update in Sonnet's form: input and forget gates from the projected input x~ and tanh(M_{t-1}), a +1 forget bias, and M_t = i * tanh(G) + f * M_{t-1}, where G is the attended memory. The tests check the zero-weight value and the open/closed gate limits.

## Exercise 3: One RMC Step

Put it together: append the input as an extra memory row, attend with queries from the memory rows only, add the residual + LayerNorm, the row-wise MLP + residual + LayerNorm, and the gates. The test loads your weights into `implementation.RelationalMemory` and requires the same output, and checks that permuting the slots permutes the result.

## Exercise 4: Nth Farthest Data

Generate the labelled task of App. A.1: K random vectors with shuffled labels, the question (n, m) at every step, and the label of the n-th farthest vector from vector m as the target. The test recomputes every answer from the inputs alone.

## Exercise 5: Training Loop

Train the RMC with cross-entropy on 3 vectors of 2 dims and read the accuracy per question type. Copying m when n = K-1 is worth 2/K for free; the relational questions are the ones that count. The loop starts with a short curriculum (only the farthest vector is asked for) because with every n from the first step both the RMC and an LSTM stay at 2/K for a very long time.

---

## How to Use

1.  Read the exercise file — each has detailed instructions.
2.  Find the TODO sections — these are what you implement (delete the `NotImplementedError` guard when you are done).
3.  Run the file — it contains a test function.
4.  Check solutions — compare with `solutions/solution_0X_*.py`.

## Tips

-   **Shapes**: `(batch, heads, slots, head_dim)` for queries, `(batch, heads, slots + 1, head_dim)` for keys and values.
-   **Rows vs features**: the input is appended along the row dimension (`dim=1`), not the feature dimension.
-   **Gating**: `sigmoid` for gates (0 to 1) and `tanh` for the candidate (-1 to 1); the forget gate gets +1 inside its sigmoid.

## Common Issues

-   **Dimension mismatch**: `mem_size` must be divisible by `num_heads`.
-   **Accuracy stuck at 2/K**: that is the copy-m shortcut; look at the relational questions and see Exercise 5's notes.
