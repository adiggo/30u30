# Day 21: Neural Message Passing for Quantum Chemistry

> Gilmer, Schoenholz, Riley, Vinyals, Dahl (2017) — [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)

**Time:** 4-6 hours
**Prerequisites:** Day 19-20 (relational reasoning), basic graph theory (nodes, edges, adjacency), PyTorch
**Code:** PyTorch

---

## What This Paper Is Actually About

By 2017, several neural network models for learning on graphs had been proposed independently — Convolutional Networks on Graphs (Duvenaud et al. 2015), Gated Graph Neural Networks (Li et al. 2015), Interaction Networks (Battaglia et al. 2016), and others. Each had its own notation, its own message-passing scheme, and its own readout mechanism. But they all did roughly the same thing: propagate information along edges, update node states, and aggregate into a graph-level prediction.

Gilmer et al. noticed this and did something simple but important: they unified all of these models into a single framework called **Message Passing Neural Networks (MPNNs)**. The framework has exactly two phases — a message passing phase and a readout phase — and each existing model is just a specific instantiation of the message function, update function, and readout function.

They then used this framework to explore new variants and applied them to the QM9 benchmark — a dataset of ~130,000 small organic molecules with 13 quantum-chemical properties computed via Density Functional Theory (DFT). The result: predictions reaching chemical accuracy on 11 of the 13 properties (Section 8, Table 2).

The paper matters beyond chemistry. The MPNN framework became the standard way to think about graph neural networks. If you understand message passing, you understand the backbone of nearly every GNN that followed.

---

## The Core Idea

Every graph neural network does three things:

1. **Message**: Each node collects information from its neighbors.
2. **Update**: Each node updates its own state based on the collected messages.
3. **Readout**: All node states are aggregated into a single graph-level output.

```
Step 1: MESSAGE              Step 2: UPDATE              Step 3: READOUT
                                                          
  h_w ---e_vw---> m_vw        m_v + h_v ---> h_v'         {h_v'} ---> y_hat
  h_u ---e_vu---> m_vu                                     
  (neighbors send messages)   (GRU updates state)         (Set2Set aggregates)
```

The MPNN framework (Section 2) defines this precisely:

**Message passing phase** (T steps):

$$m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw})$$

$$h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$$

**Readout phase**:

$$\hat{y} = R(\{h_v^T \mid v \in G\})$$

Where $M_t$ is the message function, $U_t$ is the update function, $R$ is the readout function, $h_v^t$ is the hidden state of node $v$ at step $t$, $e_{vw}$ is the edge feature between $v$ and $w$, and $N(v)$ is the set of neighbors of $v$.

Different choices of $M$, $U$, and $R$ recover different existing models. The paper's contribution is making this explicit and finding which choices work best for molecular property prediction.

---

## What the Authors Actually Showed

### The QM9 Benchmark (Section 4)

QM9 contains ~130,000 small organic molecules (H, C, N, O, F atoms, up to 9 heavy atoms). Each molecule has 13 quantum-chemical properties computed via DFT at the B3LYP/6-31G(2df,p) level. The properties include:

- Atomization energies (U0, U, H, G)
- Electronic properties (HOMO, LUMO, gap)
- Dipole moment (mu)
- Electronic spatial extent (R2)
- Zero-point vibrational energy (ZPVE)
- Heat capacity (Cv)
- Vibrational frequencies

### Results (Section 8, Table 2)

The best MPNN variant achieved chemical accuracy on 11 out of 13 targets. "Chemical accuracy" is a predefined error threshold established by the chemistry community for each property — it's the point where DFT predictions are accurate enough to be useful.

Key results from Table 2:
- Best overall: MPNN with edge network message function + GRU update + Set2Set readout
- The two targets that did NOT reach chemical accuracy: the HOMO-LUMO gap (error ratio 1.60) and ZPVE (1.27); mu (0.30) and alpha (0.92) did reach it
- With only graph topology (no 3D coordinates): chemical accuracy on 5 of the 13 targets with a set2set output, using a DFT-computed partial-charge input feature (Section 8)

### Unification (Section 2)

The paper shows that existing models map to specific MPNN instantiations:

| Model | Message Function | Update Function | Readout |
|-------|-----------------|----------------|---------|
| Convolutional Networks (Duvenaud) | $M = h_w$ (neighbor state) | $U$ = neural net | Sum + neural net |
| Gated Graph NN (Li et al.) | $M = A_{e_{vw}} h_w$ (matrix per edge type) | GRU | Gated sum |
| Interaction Networks (Battaglia) | $M = f(h_v, h_w, e_{vw})$ | $U = g(h_v, m_v)$ | Sum |
| MPNN (this paper, best variant) | Edge Network: $M = A(e_{vw}) h_w$ | GRU | Set2Set |

---

## The Architecture

### 1. Message Function — Edge Network (Section 5.1)

The best-performing message function uses a neural network to process edge features:

$$M(h_v, h_w, e_{vw}) = A(e_{vw}) \cdot h_w$$

where $A(e_{vw})$ is a neural network that maps the edge feature vector $e_{vw}$ to a $d \times d$ matrix. This matrix is then applied to the neighbor's hidden state $h_w$.

This is more general than having a fixed matrix per edge type (as in GG-NNs). It handles continuous edge features (bond length, bond angle) naturally.

### 2. Update Function — GRU (Section 2)

The node state is updated using a Gated Recurrent Unit:

$$h_v^{t+1} = \text{GRU}(h_v^t, m_v^{t+1})$$

The GRU treats the aggregated message $m_v^{t+1}$ as the "input" and the previous hidden state $h_v^t$ as the "recurrent state." All weights are shared across message passing steps.

### 3. Readout Function — Set2Set (Section 5.3)

For graph-level predictions, the individual node states need to be combined into a single vector. The paper compares GG-NN's gated sum (Eq. 4) with **Set2Set** (Vinyals et al. 2016), an attention-based method that iteratively queries the set of node vectors using an LSTM. Set2set "first applies a linear projection to each tuple $(h_v^T, x_v)$" (Section 5.3), giving one vector $m_v$ per atom, and then runs $M$ processing steps from $q^*_0 = 0$:

1. $q_t = \mathrm{LSTM}(q^*_{t-1})$: the LSTM's hidden state is the query
2. Attention over the atoms of the molecule: $a_v = \mathrm{softmax}_v(m_v \cdot q_t)$, read $r_t = \sum_v a_v m_v$
3. $q^*_t = [q_t ; r_t]$; after $M$ steps, $q^*_M$ goes through an MLP

Set2Set is permutation-invariant (order of nodes doesn't matter). Its read, though, is a weighted *average*: the weights $a_v$ of a molecule sum to 1, so giving every atom a twin leaves $r_t$ unchanged. On its own it reports the composition of a molecule, not its size, and it cannot represent an *extensive* target (one that grows with the number of atoms, like a total energy) unless the node vectors carry that information.

**In this repo** (2026-10-18), `Set2SetReadout` follows the paper: projection of $(h_v^T, x_v)$, Vinyals' order (LSTM step, then attention). The earlier version attended before the LSTM step and read $h_v^T$ only. The *default* readout, however, is now **sum**. The synthetic target here is extensive: a sum of per-atom constants plus 0.1 x (number of bonds) x (target index + 1), plus noise. Test MAE after 50 epochs (`validation/day21_readout_fix.py`; seed 42, plus seeds 1 and 2, which also change the generated molecules; for seed 42, predicting the mean gives 2.98 and least squares on atom and bond counts 0.089):

| Readout | Seed 42 | Mean of 3 seeds (range) |
|---|---|---|
| sum | 0.24 | 0.19 (0.17-0.24) |
| set2set as it was | 2.06 | 2.26 (1.98-2.75) |
| set2set as in the paper (projection + Vinyals order) | 1.80 | 2.17 (1.80-2.62) |
| same, with T = 6 message-passing rounds | 1.89 | not run |

Making set2set paper-faithful changes its accuracy here by less than the seed-to-seed spread: the bottleneck is the weighted average, not the details of the recurrence. The sum readout is ~10x better. Use `--readout set2set` to train the paper's readout.

### 4. Virtual Graph Elements (Section 5.2)

The paper also experiments with adding "virtual edges" between all pairs of atoms (a latent fully-connected graph on top of the real molecular graph). This allows information to travel between distant atoms in fewer message passing steps. The virtual edges have their own learned features distinct from real bonds.

---

## Implementation Notes

Key decisions in `implementation.py`:

- **Graph representation**: Each molecule is a tuple of `(node_features, edge_index, edge_features, target)`. Node features are one-hot atom type + other atomic properties. Edge features encode bond type, distance, etc.
- **Edge network**: A 2-layer MLP maps edge features to a $d \times d$ matrix. The matrix is reshaped from the MLP output. This is the most parameter-heavy part.
- **GRU update**: Standard PyTorch GRU cell. The aggregated message is the input, previous node state is the hidden state.
- **Readout**: sum by default (see "Readout Function" above). **Set2Set** (`--readout set2set`): projection of $(h_v^T, x_v)$, then $M = 6$ processing steps (the paper picked $M$ by random search over 1-12); the output $q^*_M$ is $2d$-dimensional (query and attention read).
- **Batching**: Multiple graphs are batched by concatenating their node/edge lists and tracking which nodes belong to which graph. This is the standard approach in PyTorch Geometric, but we implement it from scratch here.

Things that will bite you:
- **Edge network memory**: The $d \times d$ matrix per edge is expensive. For large molecules or large $d$, this dominates memory. ($d = 200$ appears in the paper's GG-NN/towers runs; the best model's $d$ is not reported.)
- **Message passing steps**: More steps = larger receptive field but diminishing returns and training instability. The paper picked $T$ by random search over 3-8 per target (Section 7); this repo uses $T = 3$.
- **Node feature initialization**: Atom type is one-hot encoded. If you forget to include hydrogen atoms (which QM9 has explicitly), your graph is wrong.
- **Numerical targets**: QM9 properties span wildly different scales (e.g., dipole moment in Debye vs. atomization energy in eV). Normalize each target independently.

---

## What to Build

### Quick Start

```bash
cd papers/21_Neural_Message_Passing
pip install -r requirements.txt
python train_minimal.py --epochs 50 --hidden-dim 64 --n-messages 3
# the paper's set2set readout instead of the default sum readout:
python train_minimal.py --readout set2set
```

This trains an MPNN (edge network + GRU + sum readout) on synthetic molecular data in about 30 seconds and reports the test MAE.

### Exercises (in `exercises/`)

| # | Task | What You'll Get Out of It |
|---|------|-----------------------------|
| 1 | Message functions (`exercise_01_message_functions.py`) | Build three message function variants and compare them |
| 2 | Graph construction (`exercise_02_graph_construction.py`) | Convert molecular data into graph tensors with proper batching |
| 3 | Readout functions (`exercise_03_readout_functions.py`) | Implement sum, mean, and Set2Set readouts and compare expressiveness |
| 4 | Edge networks (`exercise_04_edge_networks.py`) | Build the edge network that maps features to transformation matrices |
| 5 | Property prediction (`exercise_05_property_prediction.py`) | Train a full MPNN on multi-target regression and evaluate per-property MAE |

Solutions are in `exercises/solutions/`. Try to get stuck first.

---

## Key Takeaways

1. **MPNNs unify existing graph neural networks.** Convolutional Networks on Graphs, Gated Graph NNs, Interaction Networks, and others are all specific instantiations of the same message/update/readout framework. The paper makes this explicit. (Section 2)

2. **The edge network is the key architectural choice for molecular data.** Using a neural network to map edge features to transformation matrices (rather than having a fixed matrix per edge type) allows handling continuous features like bond length and angle. This was the best-performing message function. (Section 5.1, Section 8)

3. **The readout must fit the target.** The paper's best QM9 model reads out with set2set (Section 5.3; it has no set2set-vs-sum ablation in Table 2). Set2set's attention read is a weighted average, so on an extensive target (a sum over atoms) a plain sum readout is far better: test MAE 0.19 vs 2.17 on this repo's synthetic data (means of 3 seeds).

4. **Chemical accuracy is achievable from data.** On 11 of 13 QM9 targets, the MPNN matches the accuracy threshold where DFT predictions become practically useful — learned entirely from data, without hand-crafted quantum chemistry features. (Section 8, Table 2)

5. **The framework transcends chemistry.** While the experiments are on molecules, the MPNN framework applies to any domain with graph-structured data: social networks, knowledge graphs, physical simulations, program analysis. The message passing abstraction is domain-agnostic. (Section 1)

---

## Files in This Directory

| File | What It Is |
|------|-----------|
| `implementation.py` | Core MPNN: message functions, GRU update, Set2Set readout, graph batching |
| `train_minimal.py` | CLI script — train on synthetic molecular data, report per-property MAE |
| `visualization.py` | Graph visualization, message flow, training curves, per-property error bars |
| `notebook.ipynb` | Interactive walkthrough — build MPNN step by step |
| `exercises/` | 5 exercises: message functions, graph construction, readout, edge networks, full pipeline |
| `paper_notes.md` | Detailed notes on Gilmer et al. 2017 |
| `CHEATSHEET.md` | Quick reference for MPNN components and hyperparameters |

---

## Further Reading

- [Gilmer et al. 2017 — Original Paper](https://arxiv.org/abs/1704.01212) — the MPNN framework paper
- [Li et al. 2015 — Gated Graph Neural Networks](https://arxiv.org/abs/1511.05493) — the GG-NN that MPNN extends
- [Vinyals et al. 2015 — Order Matters / Set2Set](https://arxiv.org/abs/1511.06391) — the readout function used here (Day 16)
- [Battaglia et al. 2018 — Relational Inductive Biases](https://arxiv.org/abs/1806.01261) — comprehensive survey of graph networks
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/) — production-grade GNN library

---

**Previous:** [Day 20 — Relational RNNs](../20_Relational_RNNs/)
**Next:** [Day 22 — Deep Speech 2](../22_Deep_Speech_2/)
