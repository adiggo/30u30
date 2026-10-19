"""
Message Passing Neural Networks (MPNNs) for Molecular Property Prediction

Implements the MPNN framework from Gilmer et al. (2017):
"Neural Message Passing for Quantum Chemistry" (arXiv:1704.01212)

The framework has two phases:
1. Message Passing: For T rounds, each node collects messages from neighbors
   via M(h_v, h_w, e_vw), aggregates them, and updates its state via U(h_v, m_v).
   (Section 2, Equations 1-2)
2. Readout: All node states are aggregated into a graph-level prediction via R.
   (Section 2, Equation 3)

This implementation provides:
- Three message function variants (Sections 2 and 5.1)
- GRU update function (Section 2)
- Sum and Set2Set readout functions (Section 5.3)
- Edge network for continuous edge features (Section 5.1)
- Graph batching utilities for training
- Synthetic molecular data generator for demonstration

The paper's best QM9 model uses GRU update, edge network messages, and a
Set2Set readout (Section 8, Table 2). [PATCH 2026-10-18] This repo's synthetic
target is extensive (a sum over atoms and bonds), so MPNN defaults to the sum
readout here; set2set reads an attention-weighted average and cannot represent
a total by itself (README, "Readout").
"""

import math
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SECTION 1: GRAPH DATA STRUCTURES
# ============================================================================
# A graph is represented as:
#   - node_features: (n_nodes, node_dim) tensor
#   - edge_index: (2, n_edges) tensor of [source, target] pairs
#   - edge_features: (n_edges, edge_dim) tensor
#   - target: scalar or vector of properties to predict
#
# For batching multiple graphs, we concatenate all node/edge tensors and
# track which graph each node belongs to via a batch_indices tensor.
# This is the standard approach used by PyTorch Geometric.
# ============================================================================


class MolecularGraph:
    """
    Container for a single molecular graph.

    Attributes:
        node_features: (n_nodes, node_dim) — atom features (one-hot type, etc.)
        edge_index: (2, n_edges) — [source_indices, target_indices]
        edge_features: (n_edges, edge_dim) — bond features
        target: (n_targets,) — property values to predict
    """
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        target: torch.Tensor
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.target = target

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)


class BatchedGraph:
    """
    Multiple molecular graphs batched into a single set of tensors.

    When batching, we concatenate all node features and offset edge indices
    so that each graph's node indices are unique within the batch. The
    batch_indices tensor tracks which graph each node belongs to.
    """
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch_indices: torch.Tensor,
        targets: torch.Tensor
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.batch_indices = batch_indices
        self.targets = targets

    @property
    def num_graphs(self) -> int:
        return self.targets.size(0)

    def to(self, device: torch.device) -> 'BatchedGraph':
        """Move all tensors to the specified device."""
        return BatchedGraph(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            batch_indices=self.batch_indices.to(device),
            targets=self.targets.to(device)
        )


def batch_graphs(graphs: List[MolecularGraph]) -> BatchedGraph:
    """
    Batch multiple MolecularGraph instances into a single BatchedGraph.

    Edge indices are offset by the cumulative node count so that each graph's
    indices remain valid after concatenation.

    Args:
        graphs: List of MolecularGraph objects.

    Returns:
        BatchedGraph with all graphs concatenated.
    """
    node_features_list = []
    edge_index_list = []
    edge_features_list = []
    batch_indices_list = []
    target_list = []

    node_offset = 0
    for i, g in enumerate(graphs):
        node_features_list.append(g.node_features)
        # Offset edge indices by cumulative node count
        edge_index_list.append(g.edge_index + node_offset)
        edge_features_list.append(g.edge_features)
        batch_indices_list.append(torch.full((g.num_nodes,), i, dtype=torch.long))
        target_list.append(g.target)
        node_offset += g.num_nodes

    return BatchedGraph(
        node_features=torch.cat(node_features_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_features=torch.cat(edge_features_list, dim=0),
        batch_indices=torch.cat(batch_indices_list, dim=0),
        targets=torch.stack(target_list, dim=0)
    )


# ============================================================================
# SECTION 2: MESSAGE FUNCTIONS
# ============================================================================
# The message function M(h_v, h_w, e_vw) computes a message from neighbor w
# to node v. Different choices of M give different models (Section 2).
#
# We implement three variants:
# 1. SimpleMessage: M = h_w (just pass neighbor state — Duvenaud et al.)
# 2. MatrixMessage: M = A_{e_vw} * h_w (one matrix per edge type — Li et al.)
# 3. EdgeNetwork: M = A(e_vw) * h_w (NN maps features to matrix — Section 5.1)
# ============================================================================


class SimpleMessage(nn.Module):
    """
    Simplest message function: just pass the neighbor's hidden state.
    M(h_v, h_w, e_vw) = h_w

    This is the message function used in Convolutional Networks on Graphs
    (Duvenaud et al. 2015), as described in Section 2 of the paper. (The paper
    writes Duvenaud's message as the concatenation (h_w, e_vw); this variant drops e_vw.)
    """
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        # No parameters needed — messages are just neighbor states
        self.hidden_dim = hidden_dim

    def forward(
        self,
        h_v: torch.Tensor,
        h_w: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_v: (n_edges, hidden_dim) — receiving node states
            h_w: (n_edges, hidden_dim) — sending node states
            edge_features: (n_edges, edge_dim) — edge features (ignored here)

        Returns:
            messages: (n_edges, hidden_dim)
        """
        return h_w


class MatrixMessage(nn.Module):
    """
    Matrix-based message function: one learned matrix per edge type.
    M(h_v, h_w, e_vw) = A_{e_vw} * h_w

    This is the message function used in Gated Graph Neural Networks
    (Li et al. 2015), as described in Section 2. Requires discrete edge types.

    Note: edge_dim here is treated as the number of discrete edge types.
    The edge features should be integer indices, not continuous vectors.
    For continuous features, use EdgeNetwork instead.
    """
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # One transformation matrix per edge type
        self.edge_matrices = nn.Parameter(
            torch.randn(edge_dim, hidden_dim, hidden_dim) * 0.01
        )

    def forward(
        self,
        h_v: torch.Tensor,
        h_w: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_v: (n_edges, hidden_dim) — receiving node states
            h_w: (n_edges, hidden_dim) — sending node states
            edge_features: (n_edges, edge_dim) — one-hot edge type encoding

        Returns:
            messages: (n_edges, hidden_dim)
        """
        # edge_features are one-hot: (n_edges, n_types)
        # Compute weighted sum of matrices: A = sum_k(e_k * A_k)
        # A: (n_edges, hidden_dim, hidden_dim)
        A = torch.einsum('ek,khd->ehd', edge_features, self.edge_matrices)
        # Matrix-vector multiply: messages = A * h_w
        messages = torch.bmm(A, h_w.unsqueeze(-1)).squeeze(-1)
        return messages


class EdgeNetwork(nn.Module):
    """
    Edge network message function: a neural network maps edge features
    to a transformation matrix.
    M(h_v, h_w, e_vw) = A(e_vw) * h_w

    This is the paper's key contribution (Section 5.1). The neural network
    A: R^{edge_dim} -> R^{d x d} maps the edge feature vector to a d x d
    matrix. This handles continuous edge features (bond lengths, angles)
    naturally, unlike MatrixMessage which requires discrete types.

    Note: For hidden_dim=200, the edge network outputs 40,000 values per edge.
    This is the most parameter-heavy component. For constrained settings,
    consider reducing hidden_dim.
    """
    def __init__(self, hidden_dim: int, edge_dim: int, edge_hidden: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 2-layer MLP: edge features -> d*d matrix
        self.nn = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, hidden_dim * hidden_dim)
        )

    def forward(
        self,
        h_v: torch.Tensor,
        h_w: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_v: (n_edges, hidden_dim) — receiving node states
            h_w: (n_edges, hidden_dim) — sending node states
            edge_features: (n_edges, edge_dim) — continuous edge features

        Returns:
            messages: (n_edges, hidden_dim)
        """
        # Map edge features to d x d transformation matrix
        A = self.nn(edge_features)  # (n_edges, d*d)
        A = A.view(-1, self.hidden_dim, self.hidden_dim)  # (n_edges, d, d)
        # Apply matrix to neighbor state
        messages = torch.bmm(A, h_w.unsqueeze(-1)).squeeze(-1)
        return messages


# ============================================================================
# SECTION 3: READOUT FUNCTIONS
# ============================================================================
# The readout function R aggregates all final node states into a graph-level
# prediction. It must be permutation-invariant (node ordering doesn't matter).
#
# We implement two variants:
# 1. SumReadout: R = MLP(sum(h_v)) — simple but baseline
# 2. Set2SetReadout: R = Set2Set({h_v}) — LSTM-attention, more expressive
#    (Section 5.3, from Vinyals et al. 2016)
# ============================================================================


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None
) -> torch.Tensor:
    """
    Sum elements of src according to index groupings.

    This is the core operation for aggregating per-node values into per-graph
    values. For each unique index i, the output[i] = sum of src[j] where
    index[j] == i.

    Args:
        src: (N, D) — source tensor
        index: (N,) — group indices
        dim: dimension to scatter along
        dim_size: number of groups (inferred from index if not provided)

    Returns:
        (dim_size, D) — aggregated tensor
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(dim, index_expanded, src)
    return out


def scatter_softmax(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: Optional[int] = None
) -> torch.Tensor:
    """
    Per-group softmax: for each group defined by index, compute softmax
    over the elements in that group.

    Used in Set2Set to compute attention weights per graph.

    Args:
        src: (N,) — logits
        index: (N,) — group indices
        dim_size: number of groups

    Returns:
        (N,) — softmax values (sum to 1.0 within each group)
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    # Subtract max per group for numerical stability
    max_vals = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    max_vals.scatter_reduce_(0, index, src, reduce='amax', include_self=False)
    src_shifted = src - max_vals[index]
    exp_src = torch.exp(src_shifted)
    # Sum per group
    sum_exp = scatter_sum(exp_src.unsqueeze(-1), index, dim_size=dim_size).squeeze(-1)
    return exp_src / (sum_exp[index] + 1e-16)


class SumReadout(nn.Module):
    """
    Simple sum readout: sum all node states, then pass through MLP.
    R = MLP(sum_v h_v^T)

    This is the simplest permutation-invariant aggregation. It captures
    the total "mass" of node features but loses information about their
    distribution. [PATCH 2026-10-18] It is not one of the paper's two readouts
    (Sec. 5.3 compares GG-NN's gated sum, Eq. 4, with set2set) but it is the
    right inductive bias for an EXTENSIVE target (one that grows with the number
    of atoms, like the synthetic target here or a total energy): a sum of per-atom
    contributions is linear in exactly what matters.
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch_indices: torch.Tensor,
        node_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: (total_nodes, hidden_dim)
            batch_indices: (total_nodes,) — which graph each node belongs to
            node_features: unused here; accepted so that every readout has the
                same call signature ([PATCH 2026-10-18], see Set2SetReadout)

        Returns:
            (n_graphs, output_dim) — per-graph predictions
        """
        n_graphs = batch_indices.max().item() + 1
        # Sum node embeddings per graph
        graph_embeddings = scatter_sum(node_embeddings, batch_indices, dim_size=n_graphs)
        return self.mlp(graph_embeddings)


class Set2SetReadout(nn.Module):
    """
    Set2Set readout (Gilmer et al. Sec. 5.3; Vinyals, Bengio, Kudlur 2016,
    "Order Matters", Sec. 4.2, the "process block", Eqs. 3-7).

    Paper, Sec. 5.3: "This model first applies a linear projection to each
    tuple (h_v^T, x_v) and then takes as input the set of projected tuples
    T = {(h_v^T, x_v)}. Then, after M steps of computation, the set2set model
    produces a graph level embedding q*_t which is invariant to the order of
    the tuples T. We feed this embedding q*_t through a neural network to
    produce the output."

    With m_v = W [h_v^T ; x_v] + b (the projection) and q*_0 = 0, each of the
    M processing steps does (Vinyals et al., Eqs. 3-7):
        q_t     = LSTM(q*_{t-1})            the LSTM's hidden state is the query
        e_{v,t} = m_v . q_t                 one dot-product score per node
        a_{v,t} = softmax of e over the nodes of the same graph
        r_t     = sum_v a_{v,t} m_v         attention read
        q*_t    = [q_t ; r_t]
    and the prediction is MLP(q*_M).

    Because the weights a_{v,t} of a graph sum to 1, r_t is a weighted
    AVERAGE of the m_v: giving every node a twin leaves it unchanged. So
    set2set on its own reports proportions, not totals, and cannot represent
    an extensive target (one that grows with the number of atoms) unless the
    node vectors already carry that information. See README ("Readout").

    [PATCH 2026-10-18] Two changes to match the paper: (1) the optional
    projection of (h_v^T, x_v) (node_dim given; MPNN always passes it), and
    (2) Vinyals' order: LSTM step first, then attend with the new query. The
    previous version attended with the old LSTM state before each step (so
    its first read was a plain mean), fed [h, r] to the LSTM, and output the
    last state with the read from one step earlier. Standalone use without
    node_dim (as in the exercises) applies the process block to the h_v^T.
    """
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        processing_steps: int = 6,
        node_dim: Optional[int] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.processing_steps = processing_steps
        # Linear projection of the tuple (h_v^T, x_v) -> m_v (Sec. 5.3)
        self.project = nn.Linear(hidden_dim + node_dim, hidden_dim) if node_dim else None
        # LSTM: input q*_{t-1} = [q_{t-1} ; r_{t-1}] (2d), hidden state q_t (d)
        self.lstm = nn.LSTMCell(2 * hidden_dim, hidden_dim)
        # Final prediction from q*_M (2 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch_indices: torch.Tensor,
        node_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: (total_nodes, hidden_dim) — final node states h_v^T
            batch_indices: (total_nodes,) — which graph each node belongs to
            node_features: (total_nodes, node_dim) — input features x_v; required
                when the readout was built with node_dim

        Returns:
            (n_graphs, output_dim) — per-graph predictions
        """
        n_graphs = batch_indices.max().item() + 1

        if self.project is not None:
            if node_features is None:
                raise ValueError("this Set2SetReadout projects (h_v^T, x_v): pass node_features")
            # cat along features: (total_nodes, hidden_dim + node_dim) -> Linear -> (total_nodes, hidden_dim)
            m = self.project(torch.cat([node_embeddings, node_features], dim=-1))
        else:
            m = node_embeddings

        # q*_0 = 0; LSTM state (q, c) starts at 0. new_zeros keeps m's dtype and device.
        q_star = m.new_zeros(n_graphs, 2 * self.hidden_dim)
        q = m.new_zeros(n_graphs, self.hidden_dim)
        c = m.new_zeros(n_graphs, self.hidden_dim)

        for _ in range(self.processing_steps):
            # q_t = LSTM(q*_{t-1}): (n_graphs, 2d) -> (n_graphs, d)
            q, c = self.lstm(q_star, (q, c))

            # Score every node against its own graph's query: q[batch_indices] repeats
            # graph g's query once per node of g -> (total_nodes, d); dot product -> (total_nodes,)
            e = (m * q[batch_indices]).sum(dim=-1)

            # Softmax per graph
            a = scatter_softmax(e, batch_indices, dim_size=n_graphs)

            # r_t: attention-weighted sum (= weighted average, weights sum to 1) -> (n_graphs, d)
            r = scatter_sum(
                a.unsqueeze(-1) * m,
                batch_indices,
                dim_size=n_graphs
            )

            q_star = torch.cat([q, r], dim=-1)  # q*_t: (n_graphs, 2d)

        return self.mlp(q_star)


# ============================================================================
# SECTION 4: THE FULL MPNN MODEL
# ============================================================================
# Combines message function, GRU update, and readout into the complete
# framework described in Section 2 of the paper.
#
# Architecture:
#   1. Linear projection of node features to hidden dimension
#   2. T rounds of message passing:
#      a. Compute messages from all edges: m_vw = M(h_v, h_w, e_vw)
#      b. Aggregate messages per node: m_v = sum of m_vw over neighbors w
#      c. Update node states: h_v = GRU(h_v, m_v)
#   3. Readout: y_hat = R({h_v})
# ============================================================================


class MPNN(nn.Module):
    """
    Message Passing Neural Network (Section 2).

    Args:
        node_dim: Dimension of input node features (e.g., 5 for one-hot atom types)
        edge_dim: Dimension of edge features (e.g., 4 for one-hot bond types)
        hidden_dim: Dimension of node hidden states (d = 200 appears in the paper's
            GG-NN/towers runs; the best model's d is not reported)
        output_dim: Number of target properties to predict (QM9 has 13; the paper
            trains one model per target)
        n_messages: Number of message passing rounds T (paper: random search over
            3 <= T <= 8, Sec. 7)
        message_type: 'simple', 'matrix', or 'edge_network' (paper's best: edge_network)
        readout_type: 'sum' (default here) or 'set2set' (paper's best QM9 model)
        set2set_steps: Processing steps M for Set2Set (paper: random search over
            1 <= M <= 12, Sec. 7)
    [PATCH 2026-10-18] corrected the "paper uses" values above (they were T = 6,
    M = 6, d = 200; Sec. 7 searched T and M per target).
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_messages: int = 3,
        message_type: str = 'edge_network',
        readout_type: str = 'sum',
        set2set_steps: int = 6
    ):
        super().__init__()
        # [PATCH 2026-10-18] default readout 'sum' (was 'set2set'): the repo's synthetic target is
        # extensive, which an attention-weighted average cannot represent (see Set2SetReadout)
        self.hidden_dim = hidden_dim
        self.n_messages = n_messages

        # Project node features to hidden dimension
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        # Message function (Sections 2 and 5.1)
        message_classes = {
            'simple': SimpleMessage,
            'matrix': MatrixMessage,
            'edge_network': EdgeNetwork
        }
        if message_type not in message_classes:
            raise ValueError(
                f"Unknown message_type '{message_type}'. "
                f"Choose from: {list(message_classes.keys())}"
            )
        self.message_fn = message_classes[message_type](hidden_dim, edge_dim)

        # Update function: GRU (Section 2)
        # The aggregated message is the "input", previous node state is "hidden"
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # Readout function (Section 5.3)
        if readout_type == 'sum':
            self.readout = SumReadout(hidden_dim, output_dim)
        elif readout_type == 'set2set':
            # [PATCH 2026-10-18] node_dim: project the tuple (h_v^T, x_v) as in Sec. 5.3
            self.readout = Set2SetReadout(hidden_dim, output_dim, set2set_steps, node_dim=node_dim)
        else:
            raise ValueError(
                f"Unknown readout_type '{readout_type}'. Choose from: 'sum', 'set2set'"
            )

    def forward(self, batch: BatchedGraph) -> torch.Tensor:
        """
        Forward pass: message passing followed by readout.

        Args:
            batch: BatchedGraph containing all graphs in the batch.

        Returns:
            (n_graphs, output_dim) — predicted properties for each graph.
        """
        # Initialize node hidden states from features
        h = self.node_encoder(batch.node_features)  # (total_nodes, hidden_dim)

        # Message passing phase (Section 2, T rounds)
        source_idx = batch.edge_index[0]  # neighbor (sender) indices
        target_idx = batch.edge_index[1]  # node (receiver) indices

        for t in range(self.n_messages):
            # Gather sender and receiver states for each edge
            h_w = h[source_idx]  # (n_edges, hidden_dim) — neighbor states
            h_v = h[target_idx]  # (n_edges, hidden_dim) — receiver states

            # Compute messages: M(h_v, h_w, e_vw) for each edge
            messages = self.message_fn(h_v, h_w, batch.edge_features)

            # Aggregate messages per node: m_v = sum of messages to v
            m = scatter_sum(messages, target_idx, dim_size=h.size(0))

            # Update node states: h_v = GRU(h_v, m_v)
            h = self.gru(m, h)

        # Readout phase: aggregate node states to graph-level prediction.
        # [PATCH 2026-10-18] the input features x_v go along: set2set reads (h_v^T, x_v) (Sec. 5.3)
        return self.readout(h, batch.batch_indices, batch.node_features)


# ============================================================================
# SECTION 5: SYNTHETIC MOLECULAR DATA
# ============================================================================
# For demonstration purposes, we generate synthetic molecular graphs that
# mimic the structure of QM9 data. Real QM9 data requires downloading the
# dataset and parsing molecular files (not included here to keep the
# implementation self-contained).
#
# Each synthetic molecule:
#   - Has 3-12 atoms (random)
#   - Atom types: H, C, N, O, F (one-hot encoded, 5 features)
#   - Bond types: single, double, triple, aromatic (one-hot, 4 features)
#   - Molecular graph is connected (tree + random extra edges)
#   - Target: synthetic scalar property (sum of atomic contributions + noise)
#
# Note: This synthetic data is for testing the implementation only. It does
# not capture real QM9 physics. For actual molecular prediction, use the
# real QM9 dataset from http://quantum-machine.org/datasets/
# ============================================================================


# Atom types from QM9 (Section 4)
ATOM_TYPES = ['H', 'C', 'N', 'O', 'F']
BOND_TYPES = ['single', 'double', 'triple', 'aromatic']

# Synthetic "atomic contributions" for generating fake targets
ATOM_CONTRIBUTIONS = {
    'H': 0.5, 'C': 1.0, 'N': 1.2, 'O': 1.5, 'F': 2.0
}


def generate_synthetic_molecule(
    min_atoms: int = 3,
    max_atoms: int = 12,
    n_targets: int = 1,
    seed: Optional[int] = None
) -> MolecularGraph:
    """
    Generate a random synthetic molecular graph.

    The molecule is a connected graph with random atom and bond types.
    The target property is a synthetic function of atom types (not physically
    meaningful — for testing only).

    Args:
        min_atoms: Minimum number of atoms in the molecule.
        max_atoms: Maximum number of atoms.
        n_targets: Number of target properties.
        seed: Random seed for reproducibility.

    Returns:
        MolecularGraph instance.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_atoms = random.randint(min_atoms, max_atoms)

    # Generate atom types (one-hot encoded)
    # Bias towards C and H (like real organic molecules)
    atom_weights = [0.3, 0.35, 0.15, 0.15, 0.05]  # H, C, N, O, F
    atom_indices = np.random.choice(len(ATOM_TYPES), size=n_atoms, p=atom_weights)
    node_features = np.zeros((n_atoms, len(ATOM_TYPES)), dtype=np.float32)
    for i, idx in enumerate(atom_indices):
        node_features[i, idx] = 1.0

    # Generate edges: start with a spanning tree (ensures connectivity)
    # then add random extra edges (10-30% of remaining possible edges)
    edges = set()
    nodes = list(range(n_atoms))
    random.shuffle(nodes)
    for i in range(1, n_atoms):
        # Connect node i to a random previously-connected node
        j = random.choice(nodes[:i])
        edges.add((min(nodes[i], j), max(nodes[i], j)))

    # Add some extra edges
    max_extra = max(1, n_atoms // 3)
    for _ in range(random.randint(0, max_extra)):
        i, j = random.sample(range(n_atoms), 2)
        edges.add((min(i, j), max(i, j)))

    # Convert to edge index (bidirectional — undirected graph)
    edge_list = []
    for i, j in edges:
        edge_list.append((i, j))
        edge_list.append((j, i))

    if len(edge_list) == 0:
        # Fallback: at least one edge
        edge_list = [(0, 1), (1, 0)]

    edge_index = np.array(edge_list, dtype=np.int64).T  # (2, n_edges)

    # Generate bond type features (one-hot encoded)
    n_edges = edge_index.shape[1]
    bond_indices = np.random.choice(len(BOND_TYPES), size=n_edges)
    edge_features = np.zeros((n_edges, len(BOND_TYPES)), dtype=np.float32)
    for i, idx in enumerate(bond_indices):
        edge_features[i, idx] = 1.0

    # Generate synthetic target: sum of atom contributions + noise
    target_values = []
    for t in range(n_targets):
        value = sum(ATOM_CONTRIBUTIONS[ATOM_TYPES[idx]] for idx in atom_indices)
        value += np.random.normal(0, 0.1)  # small noise
        # Add interaction term for variety
        value += len(edges) * 0.1 * (t + 1)
        target_values.append(value)

    return MolecularGraph(
        node_features=torch.tensor(node_features),
        edge_index=torch.tensor(edge_index),
        edge_features=torch.tensor(edge_features),
        target=torch.tensor(target_values, dtype=torch.float32)
    )


def generate_dataset(
    n_molecules: int = 500,
    n_targets: int = 1,
    seed: int = 42
) -> List[MolecularGraph]:
    """
    Generate a dataset of synthetic molecular graphs.

    Args:
        n_molecules: Number of molecules to generate.
        n_targets: Number of target properties per molecule.
        seed: Random seed.

    Returns:
        List of MolecularGraph instances.
    """
    random.seed(seed)
    np.random.seed(seed)
    return [
        generate_synthetic_molecule(n_targets=n_targets, seed=seed + i)
        for i in range(n_molecules)
    ]


# ============================================================================
# SECTION 6: TRAINING UTILITIES
# ============================================================================


def train_epoch(
    model: MPNN,
    data: List[MolecularGraph],
    optimizer: torch.optim.Optimizer,
    batch_size: int = 32,
    device: torch.device = torch.device('cpu')
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: MPNN model.
        data: List of training MolecularGraph instances.
        optimizer: Optimizer (paper: Adam, initial lr drawn from [1e-5, 5e-4] with
            linear decay, Sec. 7).
        batch_size: Number of graphs per batch (paper uses 20).
        device: Device to train on.

    Returns:
        Average loss over the epoch.
    """
    model.train()
    random.shuffle(data)
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(data), batch_size):
        batch_graphs_list = data[i:i + batch_size]
        batch = batch_graphs(batch_graphs_list).to(device)

        optimizer.zero_grad()
        predictions = model(batch)
        # L1 loss (MAE), the metric the paper reports. [PATCH 2026-10-18] The paper itself
        # trains with MSE on targets normalized to mean 0, variance 1 (Sec. 7) and only
        # evaluates MAE; this repo trains directly on the unnormalized L1.
        loss = F.l1_loss(predictions, batch.targets)
        loss.backward()
        # Gradient clipping (common practice, 1.0 is a standard default)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: MPNN,
    data: List[MolecularGraph],
    batch_size: int = 32,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.

    Args:
        model: MPNN model.
        data: List of MolecularGraph instances.
        batch_size: Number of graphs per batch.
        device: Device.

    Returns:
        Dictionary with 'mae' (mean absolute error) and 'mse' keys.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for i in range(0, len(data), batch_size):
        batch_graphs_list = data[i:i + batch_size]
        batch = batch_graphs(batch_graphs_list).to(device)
        predictions = model(batch)
        all_preds.append(predictions.cpu())
        all_targets.append(batch.targets.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    mae = F.l1_loss(preds, targets).item()
    mse = F.mse_loss(preds, targets).item()

    return {'mae': mae, 'mse': mse}


def summarize_model(model: MPNN) -> str:
    """Print a summary of the model architecture and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        f"MPNN Model Summary",
        f"  Message function: {model.message_fn.__class__.__name__}",
        f"  Update function: GRUCell",
        f"  Readout function: {model.readout.__class__.__name__}",
        f"  Hidden dim: {model.hidden_dim}",
        f"  Message passing rounds: {model.n_messages}",
        f"  Total parameters: {total_params:,}",
        f"  Trainable parameters: {trainable_params:,}",
    ]
    return '\n'.join(lines)


# ============================================================================
# SECTION 7: CONVENIENCE ENTRY POINT
# ============================================================================


if __name__ == '__main__':
    print("MPNN Implementation — Gilmer et al. 2017")
    print("=" * 50)

    # Generate synthetic data
    print("\nGenerating synthetic molecular data...")
    dataset = generate_dataset(n_molecules=200, n_targets=3, seed=42)
    train_data = dataset[:160]
    test_data = dataset[160:]
    print(f"  Train: {len(train_data)} molecules")
    print(f"  Test: {len(test_data)} molecules")
    print(f"  Example molecule: {train_data[0].num_nodes} atoms, "
          f"{train_data[0].num_edges} bonds")

    # Create model
    model = MPNN(
        node_dim=len(ATOM_TYPES),
        edge_dim=len(BOND_TYPES),
        hidden_dim=64,
        output_dim=3,
        n_messages=3,
        message_type='edge_network',
        readout_type='sum',  # [PATCH 2026-10-18] was 'set2set'; see the module docstring
        set2set_steps=6
    )
    print(f"\n{summarize_model(model)}")

    # Quick training demo
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("\nTraining for 5 epochs...")
    for epoch in range(5):
        loss = train_epoch(model, train_data, optimizer, batch_size=32)
        metrics = evaluate(model, test_data, batch_size=32)
        print(f"  Epoch {epoch+1}: train_loss={loss:.4f}, "
              f"test_mae={metrics['mae']:.4f}")

    print("\nDone. See train_minimal.py for full training pipeline.")
