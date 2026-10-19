# Paper Notes: Recurrent Neural Network Regularization

## ELI5 (Explain Like I'm 5)

Imagine you have a friend who repeats everything back to you, word by word — that's an RNN. Now imagine during your conversations, you sometimes randomly cover your ears for a moment and miss what your friend says. If you do this during casual practice chats, you get better at filling in gaps and understanding context. But if you do it while your friend is telling you a long multi-part story, you lose the thread completely.

That's the problem this paper solves. Dropout (the "ear covering" trick) works great for regular neural networks but breaks RNNs because it disrupts the memory that flows from one timestep to the next. The solution: only apply dropout to the "vertical" connections (between layers), never to the "horizontal" connections (across time).

Note: This analogy is ours, not from the paper. But it captures the core insight — the recurrent connections need to stay clean for the LSTM to maintain long-term memory.

---

## What This Paper Actually Is

**"Recurrent Neural Network Regularization"** by Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals (2014). [arXiv:1409.2329](https://arxiv.org/abs/1409.2329).

This is a short, focused paper about ONE thing: **how to correctly apply dropout to LSTMs**. That's it. It's not a general survey of regularization techniques. The entire contribution is showing where to place the dropout operator in a multi-layer LSTM so that it helps (reduces overfitting) rather than hurts (destroys temporal memory).

At the time, there was genuine confusion about whether dropout could work with RNNs at all. Bayer et al. (2013) had published results suggesting dropout simply doesn't work for recurrent networks. This paper shows it does — you just have to do it the right way.

---

## The Problem: Dropout Breaks RNNs

Standard dropout randomly zeroes activations during training. For feedforward networks, this works beautifully — it acts as an implicit ensemble and prevents co-adaptation of neurons.

But with RNNs, the hidden state flows across timesteps:

```
h_0 --> h_1 --> h_2 --> h_3 --> ... --> h_T
```

If you apply dropout to these recurrent connections, you're randomly corrupting the model's memory at every single timestep. Over a long sequence, this cumulative disruption makes it nearly impossible for the LSTM to learn to store information for extended periods.

The paper states this directly: "Standard dropout perturbs the recurrent connections, which makes it difficult for the LSTM to learn to store information for long periods of time."

---

## The Solution: Dropout Only on Non-Recurrent Connections

The paper's key insight is simple but crucial: apply dropout only to the **non-recurrent** (vertical/inter-layer) connections, not to the **recurrent** (horizontal/temporal) connections.

### The LSTM Equations (From Section 3.1 of the Paper)

For a multi-layer LSTM with L layers, the standard (non-regularized) operation at layer l, timestep t:

```
                ( sigmoid )       (     ( h_t^{l-1} ) )
( i )           ( sigmoid )       ( W * (           ) )
( f )     =     ( sigmoid )       (     ( h_{t-1}^l ) )
( o )           (  tanh   )       (                   )
( g )

c_t^l  =  f (.) c_{t-1}^l  +  i (.) g
h_t^l  =  o (.) tanh(c_t^l)
```

Where:
- `h_t^{l-1}` = input from the layer BELOW (non-recurrent / vertical)
- `h_{t-1}^l` = hidden state from the PREVIOUS timestep (recurrent / horizontal)
- `i, f, o` = input, forget, output gates
- `g` = candidate cell update
- `c_t^l` = cell state
- `(.)` = element-wise multiplication

### The Regularized Version

The ONLY change is wrapping the non-recurrent input with the dropout operator D:

```
                ( sigmoid )       (     ( D(h_t^{l-1}) ) )
( i )           ( sigmoid )       ( W * (              ) )
( f )     =     ( sigmoid )       (     ( h_{t-1}^l    ) )
( o )           (  tanh   )       (                       )
( g )
```

That's it. `D(h_t^{l-1})` applies dropout to the input from below. `h_{t-1}^l` (the recurrent connection) is left untouched.

For the bottom layer (l=1), the input is the word embedding: `D(x_t)`.
For the top layer output going to softmax: `D(h_t^L)`.

### Why This Works: The L+1 Property

The paper makes an important observation about information flow. In this scheme, a piece of information passing from the bottom of the network to the top is affected by dropout exactly **L+1** times (once at each of the L inter-layer boundaries, plus once at the softmax output). This count is the same regardless of how many timesteps the information persists in any layer's hidden state.

This is the key property: the "corruption level" is bounded and doesn't grow with sequence length. Standard dropout gets worse with longer sequences because each timestep adds more corruption; this approach does not.

---

## The Experiments

### Penn Treebank Language Modeling (The Main Result)

The paper uses the Penn Treebank (PTB) dataset (929K training words, 73K validation, 82K test, 10K vocabulary) as the primary benchmark. Results in perplexity (lower is better):

| Model | Parameters | Validation | Test |
|-------|-----------|-----------|------|
| Non-regularized LSTM (2 x 200) | ~4.65M | 120.7 | 114.5 |
| **Medium regularized LSTM (2 x 650)** | **~19.8M** | **86.2** | **82.7** |
| **Large regularized LSTM (2 x 1500)** | **~66.0M** | **82.2** | **78.4** |
| Previous single models: Pascanu et al. (2013) / Cheng et al. | -- | -- | 107.5 / 100.0 |

The jump from 114.5 to 78.4 test perplexity is massive. The best previous single model in Table 1 (Cheng et al., 100.0) is beaten by about 22 points.

Without dropout, larger models overfit badly. With the paper's dropout scheme, bigger models keep improving — demonstrating that the regularization actually works.

### Architecture Details (From Section 4.1)

All models: unrolled for 35 steps (20 for the non-regularized one), minibatch 20, hidden state initialized to zero and then carried from one minibatch to the next (successive minibatches traverse the training set in order).

**Non-regularized LSTM (the baseline; "larger networks overfit"):**
- 2 layers, 200 hidden units per layer, weights uniform in [-0.1, 0.1]
- No dropout
- 4 epochs at learning rate 1, then the rate is halved after each epoch, 13 epochs in total

**Medium LSTM:**
- 2 layers, 650 hidden units per layer, parameters uniform in [-0.05, 0.05]
- 50% dropout (keep_prob = 0.5)
- Trained for 39 epochs
- SGD with learning rate 1.0
- After epoch 6: learning rate decayed by 1.2 each epoch
- Gradient clipping at 5
- BPTT unrolled for 35 steps
- Batch size 20

**Large LSTM:**
- 2 layers, 1500 hidden units per layer, parameters uniform in [-0.04, 0.04]
- 65% dropout (keep_prob = 0.35)
- Trained for 55 epochs
- SGD with learning rate 1.0
- After epoch 14: learning rate decayed by 1.15 each epoch
- Gradient clipping at 10
- BPTT unrolled for 35 steps
- Batch size 20

### Speech Recognition (Section 4.2)

Tested on an internal Google Icelandic speech dataset (93k utterances, small enough that overfitting is a concern). The paper reports frame accuracy, not word error rate: validation frame accuracy rose from 68.9% to 70.5% with dropout, while training accuracy fell from 71.6% to 69.4%, the usual signature of a regularizer (Table 2).

### Machine Translation (Section 4.3)

Applied to WMT'14 English-to-French translation (an LSTM trained on concatenated source and target sentences):
- 4 layers, 1000 hidden units per layer
- 20% dropout
- BLEU score: **29.03** (up from 25.9 without dropout)

This is notable — a 3-point BLEU improvement from dropout alone, on a task that already had a lot of training data (the "selected" WMT'14 subset: 340M French and 304M English words).

### Image Caption Generation (Section 4.4)

Also tested on MS COCO image captioning (Table 4). Dropout improved a single model (BLEU 23.5 to 24.3), but an ensemble of 10 non-regularized models reached 24.4: "the main effect of dropout is to produce a single model that is as good as an ensemble."

---

## Generated Text Sample (Figure 4)

The paper's Figure 4 shows samples from a large regularized model conditioned on "The meaning of life is" (with "unk", "N" and "$" removed from the permissible words), for example:

> "the meaning of life is that only if an end would be of the whole supplier. widespread rules are regarded as the companies of refuses to deliver. in balance of the nation 's information and loan growth associated with the carrier thrifts are in the process of slowing the seed and commercial paper."

It's not coherent, but the grammar and vocabulary are remarkably fluid for a word-level language model from 2014. The paper notes this as a qualitative demonstration.


---

## Figures Worth Understanding

### Figure 2 (The Key Diagram)

This is the most cited figure from the paper. It shows a regularized multi-layer RNN where:
- **Dashed arrows** = connections where dropout IS applied (non-recurrent: between layers and at input/output)
- **Solid arrows** = connections where dropout is NOT applied (recurrent: across timesteps within the same layer)

This single diagram captures the entire contribution. If you understand why the dashed lines have dropout and the solid lines don't, you understand the paper.

### Figure 3 (Information Flow)

Shows a path from the bottom layer to the top layer through the network. The information encounters exactly L+1 dropout operations regardless of how many timesteps it persists. This illustrates why the corruption doesn't grow with sequence length.

---

## Prior Work and Context

- **Pham et al. (2013)** independently found the same result: dropout on non-recurrent connections works for handwriting recognition. The Zaremba paper cites this and confirms the finding on language modeling, translation, and other tasks.
- **Bayer et al. (2013)** had claimed dropout simply doesn't work with RNNs. This paper shows that's only true if you apply dropout to recurrent connections.
- **Pachitariu and Sahani (2013)** proposed a different dropout variant for RNNs where the mask updates slowly over time. The Zaremba paper's approach is simpler.

---

## What the Paper Doesn't Cover

- **Variational dropout** (Gal & Ghahramani, 2016) — using the SAME dropout mask at every timestep for the non-recurrent connections. This came 2 years later and further improves results.
- **Layer normalization** (Ba et al., 2016) — normalizing activations within each layer. Also came later.
- **Weight decay / L2 regularization** — not discussed in this paper at all.
- **Early stopping** — not formally analyzed (though they clearly did stop training at specific epochs).
- **Why it works theoretically** — the paper is entirely empirical. No formal proof of why dropout on non-recurrent connections is better.
- **Weight tying** (Press & Wolf, 2017) — sharing input/output embeddings.
- **AWD-LSTM** (Merity et al., 2018) — which combines many regularization techniques and gets PTB perplexity down to ~57.

---

## Our Implementation (Not From the Paper)

Our code goes beyond what the paper covers. We implement:
1. **Dropout** — the paper's contribution (on non-recurrent connections)
2. **Layer normalization** — not in the paper (Ba et al. 2016), but commonly paired with dropout in modern practice
3. **Weight decay** — not in the paper, but a standard technique
4. **Early stopping** — not in the paper, but a standard technique

This is deliberate: the paper solved one specific problem, but a practical regularized RNN uses multiple techniques together. The exercises cover all four because they're all worth understanding.

The paper's own model is implemented too: `LSTMLanguageModel` in `implementation.py` (word embeddings, 2 LSTM layers, softmax; dropout on the three non-recurrent connections; truncated BPTT; global gradient-norm clipping; plain SGD with the paper's learning-rate decay) uses none of the three additions. `train_minimal.py` trains it on Penn Treebank at CPU scale (2 x 200 units on a slice of the training set instead of 2 x 650 / 2 x 1500 on all of it); the README's "Our CPU Reproduction" section lists what the paper used, what we use, and the perplexities we got.

---

## Questions Worth Thinking About

1. The paper applies a FRESH dropout mask independently at each timestep for non-recurrent connections. Gal & Ghahramani (2016) later showed that using the same mask across timesteps works even better. Why would repeating the same mask help?

2. The large model uses 65% dropout (only keeping 35% of neurons). That's aggressive. Why might larger models tolerate — or even benefit from — more aggressive dropout?

3. The paper shows that without regularization, making the model bigger HURTS performance (more overfitting). With their dropout, bigger consistently helps. What does this say about model capacity vs. regularization?

4. For machine translation, they use only 20% dropout (much less than the 50-65% for language modeling). Why might the dropout rate need to be different for different tasks?

---

**Next:** [Day 4 -- Sequence to Sequence Learning](../04_Sequence_to_Sequence/)

