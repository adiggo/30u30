# Paper Notes: Order Matters: Sequence to Sequence for Sets

## ELI5 (Explain Like I'm 5)

### The Grocery List Problem
Imagine you give a friend a shopping list. Does it matter if "Milk" is at the top or the bottom? No-the ingredients for the cake are the same regardless of the list's order. This represents a **Set**. 

However, if you give them a recipe, the order matters immensely. You can't frost the cake before you bake it. This represents a **Sequence**.

Historically, neural networks (like RNNs) were only good at the "Recipe" part. They got confused if you gave them a "Grocery List" in a different order, even if the items were identical. This paper teaches the network how to handle a Bag (Set) and turn it into a Line (Sequence).

> **Note:** This analogy is ours, intended to clarify the difference between permutation-invariant inputs and ordered outputs.

---

## What the Paper Actually Covers (Traceability)

The paper **"Order Matters: Sequence to Sequence for Sets" (Vinyals, Bengio, Kudlur, ICLR 2016)** asks whether the order in which a seq2seq model reads its inputs and writes its outputs matters, and what to do when the data have no natural order.

### 1. Order matters even though the chain rule allows any order (Sections 3, 4.1)
Seq2seq factorises $P(Y \mid X) = \prod_t P(y_t \mid y_{<t}, X)$, which is valid for any ordering, and an LSTM can in principle encode any input order. In practice the order changes results, which the authors attribute to non-convex optimisation and priors: reversing the source sentence gave 5.0 BLEU in translation and 0.5% F1 in parsing, and sorting convex-hull points by angle before a Pointer Network raised accuracy by up to 10% absolute (Section 4.1).

### 2. Content-based attention (Section 4.2, Eqs. 3-7)
The process block is an LSTM with no data input that, for $T$ steps, queries a set of memory vectors $m_i$:
$q_t = \mathrm{LSTM}(q^*_{t-1})$, $e_{i,t} = f(m_i, q_t)$, $a_{i,t} = \mathrm{softmax}_i(e_{i,t})$, $r_t = \sum_i a_{i,t} m_i$, $q^*_t = [q_t, r_t]$.
Because memory is addressed by content, "the vector retrieved from our memory would not change if we randomly shuffled the memory": $q^*_T$ is permutation-invariant.

### 3. The Read-Process-Write model (Section 4.3, Figure 1)
- **Read:** a small neural network (for sorting, "a small multilayer perceptron for each number", Section 4.4) embeds each element $x_i$ into a memory $m_i$, the same network for every element.
- **Process:** $T$ steps of Eqs. 3-7 over the memories; the result $q^*_T$ is an order-invariant embedding of the set.
- **Write:** an LSTM **Pointer Network** (Vinyals, Fortunato, Jaitly 2015) that takes in $q^*_T$ and points at the memories one step at a time. The authors add a **glimpse**, an attention read just before each pointer; process steps and glimpses "complement each other".

The pointer mechanism itself comes from Pointer Networks (Day 18): instead of choosing a token from a fixed dictionary, the decoder outputs an index into its input, so the output vocabulary grows with the input.

### 4. Output order matters (Section 5.1)
- Language modelling (PTB): natural and reversed word order both reach 86 perplexity; reversing every 3 words gives 96.
- Parsing: emitting the tree depth-first gives 89.5% F1, breadth-first 81.5%.
- Combinatorial outputs have many equivalent orderings (the $n!$ orders of a set of argsort indices); training on random ones spreads probability over all of them. For sorting with $n = 5$, it "never reached the same performance". Restricting outputs to one canonical order (Pointer Networks' TSP tours started at the first input city and went counter-clockwise) gave at least 5% absolute in earlier work.

### 5. Letting the model choose the output order (Section 5.2, Eq. 9)
$\theta^* = \arg\max_\theta \sum_i \max_{\pi(X_i)} \log p(Y_{\pi(X_i)} \mid X_i; \theta)$. Searching $n!$ orders is impossible, and taking the max greedily locks in whatever order the initialisation prefers, so the authors pretrain with a uniform prior over orderings for 1,000 steps and then sample orderings in proportion to $p(Y_\pi \mid X)$. On 5-grams treated as sets, this finds natural-like orders (perplexity 225, vs 280 for a bad fixed order).

---

## The Experiments

### Sorting numbers (Section 4.4, Table 1)
- **Task:** $N$ numbers uniform in $[0, 1]$, output their indices in sorted order. Accuracy counts a set only if it is sorted completely. Each $N$ is trained and tested separately (no length generalisation), 10,000 training iterations.
- **Finding (without / with one glimpse):** $N = 5$: Ptr-Net 81% / 90%, $P = 0$ 65% / 84%, $P = 1$ 84% / 92%, $P = 5$ 88% / 94%, $P = 10$ 90% / 94%. $N = 10$: Ptr-Net 8% / 28%, $P = 5$ 17% / 57%, $P = 10$ 19% / 50%. $N = 15$: every model at or below 10%.
- With $P = 0$ and no glimpse the writer is "effectively unconditioned on X and has to blindly point", and is worst; one or more process steps beat the LSTM-encoder Ptr-Net; glimpses more than double accuracy in the hardest cases.

### Language modelling, parsing, 5-grams, graphical models (Section 5)
See items 4 and 5 above. A toy experiment with star-shaped graphical models (Section 5.1.4) shows that putting the "head" variable first is easier to learn unless data are plentiful or the distributions nearly deterministic.

**Not in this paper:** the convex hull and travelling salesman experiments belong to Pointer Networks (Vinyals, Fortunato, Jaitly 2015; Day 18). This folder uses them as exercises.

---

## What the Paper Doesn't Cover

- **Dynamic Sets:** The paper assumes the set is provided all at once. It doesn't explore sets that grow or change over time.
- **Exact equations and implementation:** Section 4.3 promises them for an appendix that is not in the published paper (arXiv v4); $f$, the glimpse and how $q^*_T$ enters the writer are left open (this repo's choices are listed in the README).
- **Masking:** the paper does not say whether elements already pointed at are masked; this changes how hard sorting is (README, Results).
- **Larger sets:** accuracy collapses by $N = 15$; scaling set models came later (e.g. Set Transformer).

---

## Our Additions (Not from the Paper)

### Implementation Stability: The Log-Sum-Exp Trick
When implementing the Pointer Network's attention mechanism, you will often encounter numerical instability if you calculate raw exponentials. In our `implementation.py`, we use PyTorch's `log_softmax` to stay in log-space, preventing `inf` values during training.

### The Transformer Connection
The process block is **attention pooling**: a query vector reads the whole set by content, so the result ignores order. A Transformer without positional encodings is related but different: its per-element outputs are permutation-*equivariant* (they are shuffled along with the inputs), and only a symmetric pooling of them (a mean, or an attention read-out like the process block) is invariant. That is also why Transformers *must* add positional encodings to care about word order. Set Transformer's pooling by multihead attention (Lee et al. 2019), where learned seed vectors attend over the set, is the same idea.

### Masking Used Elements
In our code, once the model points at an element we mask it (score $-\infty$), so the output is a permutation and the model never picks an element twice. The paper does not say whether it did this; in Pointer Networks the decoder was unconstrained, and only TSP's beam search at inference was restricted to valid tours. With masking, sorting becomes "point at the smallest remaining element", which Read-Process-Write learns perfectly even without process steps; `train.py --no-mask` turns it off.

### Content-defined canonical orders
For an order-invariant model, a canonical output order must be defined by content, not by input index: our hull targets start at the leftmost point (not at the lowest index, as in Pointer Networks), because the model cannot see indices.
