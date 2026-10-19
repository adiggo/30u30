"""
Exercise 3: Readout Functions (Medium, 3/5)

Implement two readout functions that aggregate node embeddings into a
single graph-level representation:

  1. SumReadout: R = MLP(sum_v h_v^T) — simple permutation-invariant baseline
  2. Set2SetReadout: LSTM-attention based aggregation (Section 4.3)

The readout function must be permutation-invariant: reordering the nodes
should not change the output. Sum achieves this trivially. Set2Set achieves
it through attention weights that depend on content, not position.

Reference: Gilmer et al. 2017, Section 4.3 and Table 2
"""

import torch
import torch.nn as nn
from typing import Optional


def scatter_sum(src, index, dim=0, dim_size=None):
    """Sum elements of src grouped by index. Helper function."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(dim, index_expanded, src)
    return out


class SumReadout(nn.Module):
    """
    Sum readout: aggregate by summing all node embeddings per graph,
    then pass through an MLP for prediction.

    R = MLP(sum_v h_v^T)
    """
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        # TODO: Create a 2-layer MLP:
        #   Linear(hidden_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> output_dim)
        self.mlp = None  # Replace with your implementation

    def forward(self, node_embeddings, batch_indices):
        """
        Args:
            node_embeddings: (total_nodes, hidden_dim)
            batch_indices: (total_nodes,) — which graph each node belongs to

        Returns:
            (n_graphs, output_dim) — one prediction per graph
        """
        # TODO: Implement sum readout
        # Step 1: Sum node embeddings per graph using scatter_sum
        # Step 2: Pass through MLP
        raise NotImplementedError("Implement SumReadout.forward")


class Set2SetReadout(nn.Module):
    """
    Set2Set readout (Section 5.3; Vinyals et al. 2016, "Order Matters", Eqs. 3-7):
    attention-based aggregation.

    Uses an LSTM to iteratively attend to different subsets of nodes.
    After processing_steps iterations, outputs [query, weighted_sum].

    The paper's best QM9 model reads out with set2set (number of steps M chosen
    by random search over 1-12), applied to a linear projection of (h_v^T, x_v);
    here you implement the process block on the node embeddings alone.
    [PATCH 2026-10-18] This docstring used to say the paper reports set2set
    "consistently outperforms sum pooling across all QM9 targets (Table 2)": the
    paper has no such comparison. Note that the attention read is a weighted
    AVERAGE (weights sum to 1), so set2set alone cannot count nodes.
    """
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        processing_steps: int = 6
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.processing_steps = processing_steps

        # TODO: Create the LSTM cell and output MLP
        # LSTM input dim = 2 * hidden_dim (concatenation of query and weighted sum)
        # LSTM hidden dim = hidden_dim
        # Output MLP: Linear(2 * hidden_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> output_dim)
        self.lstm = None  # Replace: nn.LSTMCell(2 * hidden_dim, hidden_dim)
        self.mlp = None   # Replace with your implementation

    def forward(self, node_embeddings, batch_indices):
        """
        Args:
            node_embeddings: (total_nodes, hidden_dim)
            batch_indices: (total_nodes,) — which graph each node belongs to

        Returns:
            (n_graphs, output_dim) — one prediction per graph

        Algorithm (Vinyals et al. 2016, Eqs. 3-7; used in Section 5.3):
            1. Initialize q_star = zeros(n_graphs, 2 * hidden_dim) and
               h, c = zeros(n_graphs, hidden_dim)
            2. For each processing step:
               a. h, c = LSTM(q_star, (h, c))   # q_t = LSTM(q*_{t-1}); h is the query
               b. e_i = dot(h[batch[i]], node_embeddings[i]) for all nodes
               c. a_i = softmax(e_i) per graph (attention weights)
               d. r = sum(a_i * node_embeddings[i]) per graph (weighted sum)
               e. q_star = cat(h, r)            # q*_t = [q_t ; r_t]
            3. Return MLP(q_star)
            [PATCH 2026-10-18] order fixed: the LSTM step comes first, then attention
            with the new query (before, it attended with the previous state).
        """
        # TODO: Implement Set2Set readout
        # This is the hardest part of the exercises. Follow the algorithm above.
        #
        # Hint for per-graph softmax:
        #   max_vals = scatter_max(e, batch_indices)  # or compute manually
        #   e_shifted = e - max_vals[batch_indices]   # numerical stability
        #   exp_e = torch.exp(e_shifted)
        #   sum_exp = scatter_sum(exp_e, batch_indices)
        #   a = exp_e / sum_exp[batch_indices]
        raise NotImplementedError("Implement Set2SetReadout.forward")


# ============================================================================
# Tests
# ============================================================================


def test_readout_functions():
    """Test readout function implementations."""
    hidden_dim = 8
    output_dim = 3
    torch.manual_seed(42)

    # Create fake batched graph data: 2 graphs, 5+4=9 nodes
    node_embeddings = torch.randn(9, hidden_dim)
    batch_indices = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

    passed = True

    # Test SumReadout
    print("Testing SumReadout...")
    try:
        readout = SumReadout(hidden_dim, output_dim)
        assert readout.mlp is not None, "self.mlp must be defined"
        out = readout(node_embeddings, batch_indices)
        assert out.shape == (2, output_dim), f"Wrong shape: {out.shape}"

        # Check permutation invariance: shuffle nodes within graph 0
        perm = torch.tensor([3, 1, 4, 0, 2, 5, 6, 7, 8])
        out_perm = readout(node_embeddings[perm], batch_indices[perm])
        assert torch.allclose(out, out_perm, atol=1e-5), \
            "SumReadout should be permutation-invariant"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    # Test Set2SetReadout
    print("Testing Set2SetReadout...")
    try:
        readout = Set2SetReadout(hidden_dim, output_dim, processing_steps=3)
        assert readout.lstm is not None, "self.lstm must be defined"
        assert readout.mlp is not None, "self.mlp must be defined"
        out = readout(node_embeddings, batch_indices)
        assert out.shape == (2, output_dim), f"Wrong shape: {out.shape}"

        # Check that different graphs give different outputs
        assert not torch.allclose(out[0], out[1], atol=1e-3), \
            "Different graphs should produce different predictions"
        print("  PASSED")
    except NotImplementedError:
        print("  NOT IMPLEMENTED")
        passed = False
    except Exception as e:
        print(f"  FAILED: {e}")
        passed = False

    if passed:
        print("\nAll readout function tests PASSED")
    else:
        print("\nSome tests failed or are not implemented yet")


if __name__ == '__main__':
    test_readout_functions()
