# Day 3: Recurrent Neural Network Regularization

> Zaremba, Sutskever & Vinyals (2014) — [arXiv:1409.2329](https://arxiv.org/abs/1409.2329)

**Time:** 2-4 hours
**Prerequisites:** Day 1 (RNN basics), Day 2 (LSTM internals)
**Code:** NumPy (builds on Day 2's LSTM)

---

## What This Paper Is Actually About

This paper answers one specific question: **how do you apply dropout to LSTMs without breaking them?**

Standard dropout was a huge success for feedforward networks (Hinton et al., 2012), but people struggled to make it work with RNNs. Bayer et al. (2013) even published results claiming dropout doesn't help recurrent networks. Zaremba, Sutskever, and Vinyals show that's wrong — the problem wasn't dropout itself, but *where* you apply it.

Their solution: apply dropout only to the **non-recurrent** (vertical) connections, never to the **recurrent** (horizontal) connections. This simple rule lets you use dropout to regularize LSTMs while preserving their ability to remember across long sequences.

The result: PTB language modeling perplexity drops from 114.5 (no regularization) to 78.4 (with dropout), well below the best previous single models in the paper's Table 1 (107.5 for Pascanu et al. 2013, 100.0 for Cheng et al.).

---

## The Core Idea

### Why Naive Dropout Fails on RNNs

In a multi-layer LSTM, information flows in two directions:
- **Vertically** (between layers): `h_t^{l-1}` feeds into layer `l` at the same timestep
- **Horizontally** (across time): `h_{t-1}^l` carries memory forward within layer `l`

```
          Timestep 1      Timestep 2      Timestep 3

Layer 2:    h_1^2  ------->  h_2^2  ------->  h_3^2   (horizontal = recurrent)
              ^                ^                ^
              | (vertical)     | (vertical)     |
Layer 1:    h_1^1  ------->  h_2^1  ------->  h_3^1   (horizontal = recurrent)
              ^                ^                ^
              |                |                |
Input:       x_1              x_2              x_3
```

If you apply dropout to the horizontal arrows (recurrent connections), you corrupt the memory at every timestep. Over a 35-step sequence, the signal gets randomly perturbed 35 times — making it nearly impossible for the LSTM to learn long-range dependencies.

### The Fix: Dropout Only on Vertical Connections

The paper's regularized LSTM equations differ from the standard ones by exactly one thing — the dropout operator `D()` wrapping the non-recurrent input:

**Standard:**
```
gates = W * [h_t^{l-1}, h_{t-1}^l] + b
```

**Regularized (this paper):**
```
gates = W * [D(h_t^{l-1}), h_{t-1}^l] + b
```

That's the entire contribution. `D()` is applied to:
- The input embedding at the bottom: `D(x_t)`
- The inter-layer connections: `D(h_t^{l-1})`
- The output to softmax at the top: `D(h_t^L)`

It is NOT applied to:
- The recurrent hidden state: `h_{t-1}^l` passes through clean

### The L+1 Property

A piece of information traveling from input to output encounters dropout exactly **L+1** times (L layers + output), regardless of how many timesteps it persists. This bounded corruption is what makes it work — the noise level doesn't scale with sequence length.

---

## What the Paper Showed

### Penn Treebank Results

| Model | Hidden | Dropout | Test Perplexity |
|-------|--------|---------|-----------------|
| Non-regularized LSTM | 200 | 0% | 114.5 |
| **Medium regularized** | **650** | **50%** | **82.7** |
| **Large regularized** | **1500** | **65%** | **78.4** |
| Previous single models (Pascanu 2013 / Cheng et al.) | -- | -- | 107.5 / 100.0 |

Key observations from the paper:
- Without dropout, bigger models overfit and get WORSE (200 units was the best non-regularized size)
- With dropout, bigger models consistently improve
- The large model beats the best previous single model (100.0) by ~22 perplexity points

### Other Tasks

The paper also demonstrates the technique on:
- **Speech recognition** (Icelandic, 93k utterances): validation frame accuracy 68.9% -> 70.5% (training accuracy fell 71.6% -> 69.4%; the paper reports frame accuracy, not word error rate)
- **Machine translation** (WMT'14 EN-FR): BLEU 29.03 vs 25.9 without dropout (4-layer, 1000 hidden, 20% dropout)
- **Image captioning** (MS COCO): BLEU 23.5 -> 24.3 for a single model; an ensemble of 10 non-regularized models reaches 24.4

---

## The Architecture

### Forward Pass (Single Timestep, Layer l)

```python
# Standard LSTM gates
combined = np.vstack([h_prev_layer, h_prev_time])  # [D(h_t^{l-1}), h_{t-1}^l]

f = sigmoid(Wf @ combined + bf)     # Forget gate
i = sigmoid(Wi @ combined + bi)     # Input gate
g = tanh(Wc @ combined + bc)        # Candidate
o = sigmoid(Wo @ combined + bo)     # Output gate

c_t = f * c_prev + i * g            # Cell state update
h_t = o * tanh(c_t)                 # Hidden state
```

The critical difference from Day 2's LSTM: the input `h_prev_layer` has dropout applied to it BEFORE entering the gate computations. The recurrent input `h_prev_time` does not.

### Dropout Application Points

```python
# Layer 1: dropout on input embedding
x_dropped = dropout(x_t, keep_prob=0.5)
h_1 = lstm_layer_1(x_dropped, h_prev_1, c_prev_1)

# Layer 2: dropout on layer 1 output (non-recurrent connection)
h_1_dropped = dropout(h_1, keep_prob=0.5)
h_2 = lstm_layer_2(h_1_dropped, h_prev_2, c_prev_2)

# Output: dropout on top layer output
h_2_dropped = dropout(h_2, keep_prob=0.5)
y = softmax(Why @ h_2_dropped + by)
```

Notice: dropout is applied 3 times total (L+1 = 2+1 = 3) for a 2-layer LSTM.

### Training Hyperparameters (From the Paper)

**Medium model (recommended starting point):**
```python
num_layers = 2
hidden_size = 650
dropout_keep_prob = 0.5
init_scale = 0.05         # all parameters uniform in [-0.05, 0.05]
learning_rate = 1.0       # SGD
lr_decay = 1.2            # Decay after epoch 6
gradient_clip = 5.0       # global norm of the gradient (loss normalized by minibatch size)
bptt_steps = 35
batch_size = 20
epochs = 39
```

**Large model:**
```python
num_layers = 2
hidden_size = 1500
dropout_keep_prob = 0.35   # More aggressive dropout
init_scale = 0.04
learning_rate = 1.0
lr_decay = 1.15            # Gentler decay, starts epoch 14
gradient_clip = 10.0
bptt_steps = 35
batch_size = 20
epochs = 55
```

---

## Implementation Notes

`implementation.py` has two LSTM classes:

- **`LSTMLanguageModel`** is the paper's model (Sec. 3 and 4.1): word embeddings → 2 LSTM layers → softmax over the 10k-word vocabulary, dropout on the three non-recurrent connections (embedding → layer 1, layer 1 → layer 2, layer 2 → softmax) and never on `h_{t-1} → h_t`, with a hand-written backward pass (truncated backpropagation through time). `train_minimal.py` trains it on Penn Treebank with the paper's recipe: 35-step windows with the hidden state carried from one minibatch to the next, global gradient-norm clipping at 5, plain SGD at learning rate 1 with per-epoch decay. Its gradients match `torch.autograd` to about 1e-15 in float64 (`validation/day03_bptt_check.py`).
- **`RegularizedLSTM`** is the original single-layer teaching model (batch size 1, one-hot inputs, column vectors, optional layer norm). It is **forward-only**: it shows step by step where dropout goes, but nothing trains it.

The file also includes regularization techniques that are commonly used alongside dropout in practice (layer normalization, weight decay, early stopping). These are clearly labeled as our additions; the paper's model uses none of them (`train_minimal.py` offers weight decay and early stopping as opt-in flags, and no layer norm).

Key decisions:
- **Pure NumPy**: No framework — you see every operation, including the backward pass
- **Inverted dropout**: We scale by `1/keep_prob` during training so test time needs no modification (the paper's Torch `nn.Dropout` does the same)
- **Gate order (i, f, o, g)** as in the paper's equations; PyTorch's `nn.LSTM` uses (i, f, g, o), so the gradient check reorders the blocks when it copies weights
- **Layer norm and weight decay**: Not from this paper, but useful pedagogical additions

Things to watch for:
- **Don't apply dropout during evaluation.** The paper is explicit: dropout is training-only. At test time, all neurons are active.
- **Dropout mask is fresh each timestep** (for non-recurrent connections). Gal & Ghahramani (2016) later showed that using the same mask across timesteps works even better (variational dropout), but this paper doesn't do that.
- **The keep_prob values are per-connection.** 50% dropout means each non-recurrent connection independently has a 50% chance of being zeroed at each timestep.

---

## Our CPU Reproduction

The paper's models trained for hours to a day each on a 2014 GPU; our NumPy code runs on one CPU core. So `train_minimal.py` keeps the paper's method and recipe and shrinks the experiment. (`--paper-config small|medium|large` runs the paper's exact settings on all of PTB: correct, but hours to days in NumPy.)

| | Paper (Sec. 4.1) | `python train_minimal.py` | `python train_minimal.py --compare` |
|---|---|---|---|
| Training data | all 929k PTB words | first 100k words (11%) | first 100k words |
| Model | 2×200 without dropout; 2×650 with 50%; 2×1500 with 65% | 2×200, 50% dropout | 2×200, without and with 50% dropout |
| Init (uniform) | ±0.1 / ±0.05 / ±0.04 | ±0.1 | ±0.1 |
| Unroll, batch | 20 (no dropout) or 35 steps, 20 | 35, 20 | 35, 20 |
| Epochs, learning rate | 13 / 39 / 55; lr 1, then ÷2 / ÷1.2 / ÷1.15 per epoch after epoch 4 / 6 / 14 | 6; lr 1, then ÷2 per epoch after epoch 4 | 10; lr 1, then ÷2 per epoch after epoch 6 |
| Gradient-norm clip | 5 (10 for large) | 5 | 5 |
| Evaluation | full validation and test sets | same | same |
| Time | 2-3 h to 1 day on an NVIDIA K20 | about 3 min of CPU | about 10 min of CPU |

Recorded results (seed 0). Perplexity, lower is better. Validation uses all 73,760 words; test uses all 82,430 words as one continuous stream, with the best-validation parameters (the paper's code also evaluates test as one stream). "Final train" is measured in eval mode (no dropout) on the first 20,000 training words after the last epoch.

| Run | Best validation (epoch) | Final validation | Final train | Test |
|---|---|---|---|---|
| Default: 50% dropout, 6 epochs | 296.3 (6) | 296.3 | 201.9 | 287.7 |
| `--compare`: no dropout | 315.4 (7) | 333.9 | 69.8 | 307.8 |
| `--compare`: 50% dropout | 266.7 (10) | 266.7 | 138.8 | 258.8 |
| Paper, full PTB: non-regularized 2×200 | 120.7 | | | 114.5 |
| Paper, full PTB: medium 2×650, 50% dropout | 86.2 | | | 82.7 |

What the comparison shows is the paper's claim in miniature. On 100k words a 4.65M-parameter LSTM is "large" for its data. Without dropout it overfits: by the last epoch its training perplexity (69.8) is about a fifth of its validation perplexity (333.9), and validation got worse in each of the last three epochs (315.4 at epoch 7, then 326.7, 331.4, 333.9). With dropout on the non-recurrent connections the gap stays under 2× (138.8 vs 266.7), validation improves in every epoch (it was still improving when the schedule ended), and the best validation and test perplexities are about 15% lower (266.7 vs 315.4; 258.8 vs 307.8). Even perfect early stopping would not rescue the unregularized model: its best epoch is still worse than the dropout model. The absolute numbers are far from the paper's because we train on 11% of the data for at most 10 epochs with a 200-unit model; matching them needs the full data and the 650/1500-unit models.

Gradient check (`validation/day03_bptt_check.py`, float64, the same dropout masks given to NumPy and PyTorch): every parameter's gradient, and the gradient with respect to the carried-in state, matches `torch.autograd` through `nn.LSTM` to a relative error of 1.2e-15, and fourth-order central finite differences agree to 3.5e-9.

---

## What to Build

### Quick Start

```bash
python train_minimal.py             # the paper's recipe at CPU scale: 2x200 LSTM, 50% dropout, 100k PTB words (~3 min of CPU)
python train_minimal.py --compare   # the same model with and without dropout (about 10 min of CPU)
python train_minimal.py --help      # every knob; --paper-config small|medium|large = the paper's exact settings
```

Penn Treebank (Mikolov's split, from the paper's repository) is downloaded to `data/ptb/` on first use, or read from `--data-dir` / `$PTB_DIR`. `--keep-prob` is the keep probability (the paper's "50% dropout" is 0.5). Weight decay (`--weight-decay`) and early stopping (`--patience`) are our additions and are off by default, as in the paper. See [Our CPU Reproduction](#our-cpu-reproduction) for what to expect.

To check the backward pass yourself: `python ../../validation/day03_bptt_check.py` compares every gradient with `torch.autograd` (float64, identical dropout masks) and with finite differences.

### Exercises (in `exercises/`)

| # | Task | What You'll Learn |
|---|------|-------------------|
| 1 | Implement dropout forward/backward | Core of the paper: how dropout works mechanically |
| 2 | Implement layer normalization | Our addition: stabilizing activations (Ba et al. 2016) |
| 3 | Implement weight decay (L2) | Our addition: penalizing large weights |
| 4 | Implement early stopping | Our addition: stopping before overfitting |
| 5 | Full regularized pipeline | Combine all techniques into one training loop |

Exercises 1 is directly from what the paper covers. Exercises 2-4 are standard regularization techniques that complement dropout. Exercise 5 brings everything together.

Solutions are in `exercises/solutions/`. Try first.

---

## Key Takeaways

1. **Dropout works for LSTMs — but only on non-recurrent connections.** This is the paper's single contribution, and it's an important one. Applying dropout to recurrent connections destroys the LSTM's ability to carry information across time.

2. **The corruption is bounded.** Information encounters dropout exactly L+1 times regardless of sequence length. This property is what makes the approach work for long sequences.

3. **Regularization unlocks scale.** Without dropout, making the LSTM bigger hurts (overfitting). With dropout, bigger is better. This was one of the early demonstrations that regularization and scale work together.

4. **Simple ideas can be powerful.** The entire contribution is one line of change in the LSTM equations. But it takes PTB test perplexity from 114.5 to 78.4 and generalizes across language modeling, translation, speech, and captioning.

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | NumPy: the paper's 2-layer `LSTMLanguageModel` with BPTT, the forward-only `RegularizedLSTM` teaching model, dropout, layer norm, weight decay, early stopping |
| `train_minimal.py` | Trains `LSTMLanguageModel` on Penn Treebank and reports perplexity; `--compare` runs dropout vs no dropout |
| `visualization.py` | Learning curves, dropout effects, weight distributions |
| `notebook.ipynb` | Interactive walkthrough of regularization techniques |
| `exercises/` | 5 exercises: dropout (from paper), layer norm, weight decay, early stopping, full pipeline |
| `paper_notes.md` | Detailed notes on the actual Zaremba et al. paper |
| `CHEATSHEET.md` | Quick reference for regularization techniques and hyperparameters |

---

## Further Reading

- [Zaremba et al. (2014)](https://arxiv.org/abs/1409.2329) — this paper
- [Gal & Ghahramani (2016)](https://arxiv.org/abs/1512.05287) — variational dropout: same mask across timesteps, theoretically grounded
- [Merity et al. (2018)](https://arxiv.org/abs/1708.02182) — AWD-LSTM: combines many regularization tricks, PTB perplexity ~57
- [Srivastava et al. (2014)](https://jmlr.org/papers/v15/srivastava14a.html) — the original dropout paper
- [Ba et al. (2016)](https://arxiv.org/abs/1607.06450) — layer normalization

---

**Next:** [Day 4 -- Sequence to Sequence Learning](../04_Sequence_to_Sequence/)
