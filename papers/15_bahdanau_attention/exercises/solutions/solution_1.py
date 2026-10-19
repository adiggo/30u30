"""
Solution 1: Additive (Bahdanau) Attention

Reference implementation of the alignment model (Bahdanau et al. 2014, Appendix A.2.2):

    e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)       alpha_ij = softmax_j(e_ij)       c_i = sum_j alpha_ij h_j

with v_a in R^{n'}, W_a in R^{n' x n} and U_a in R^{n' x 2n}: the query s_{i-1} is a decoder state of
n units, and each key h_j is a 2n-dim annotation [forward h_j; backward h_j] from the bidirectional
encoder (Sec. 3.2, Eq. 7). The paper uses n = n' = 1000 (A.2.3).

[PATCH 2026-10-18] Constructor and forward() now follow exercise_1.py (hidden_size = n',
query_size = n, key_size = 2n; forward(query, keys, values, mask)), so the exercise's own test passes
with this class swapped in. The old version took (encoder_hidden_size, decoder_hidden_size).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention.

    "Additive" because the projected query and key are ADDED inside the tanh, unlike dot-product
    attention, which multiplies them. The tanh layer plus v_a is a one-hidden-layer MLP that scores
    how well the source annotation h_j matches the decoder state s_{i-1}.

    Args:
        hidden_size: n', units of the alignment MLP
        query_size: n, size of the decoder state s_{i-1}
        key_size: 2n, size of the encoder annotations h_j (forward and backward states concatenated)
    """

    def __init__(self, hidden_size, query_size, key_size):
        super().__init__()
        self.query_proj = nn.Linear(query_size, hidden_size, bias=False)   # W_a: n -> n'
        self.key_proj = nn.Linear(key_size, hidden_size, bias=False)       # U_a: 2n -> n'
        self.energy = nn.Linear(hidden_size, 1, bias=False)                # v_a^T: n' -> 1
        self._init_weights()

    def _init_weights(self):
        """Xavier init (the paper, App. B.1, draws W_a, U_a from N(0, 0.001^2) and sets v_a = 0)."""
        for module in [self.query_proj, self.key_proj, self.energy]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, query, keys, values=None, mask=None):
        """
        Compute attention weights and the context vector.

        Args:
            query: decoder state s_{i-1} (batch, query_size)
            keys: annotations h_j (batch, src_len, key_size)
            values: vectors to average (batch, src_len, value_size); default keys, as in the paper,
                where the same annotations are scored and averaged
            mask: True where a position must be ignored, e.g. padding (batch, src_len)

        Returns:
            context: sum_j alpha_ij values_j (batch, value_size)
            attention_weights: alpha_i (batch, src_len)
        """
        if values is None:
            values = keys
        keys_proj = self.key_proj(keys)                     # (batch, src_len, n'): U_a h_j for every j
        query_proj = self.query_proj(query).unsqueeze(1)    # (batch, 1, n'): W_a s_{i-1}
        # broadcasting adds the same query to every source position: (batch, src_len, n')
        scores = self.energy(torch.tanh(query_proj + keys_proj)).squeeze(-1)   # (batch, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))   # exp(-inf) = 0 after the softmax
        attention_weights = F.softmax(scores, dim=-1)          # normalise over source positions

        # (batch, 1, src_len) @ (batch, src_len, value_size) -> (batch, 1, value_size)
        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        return context, attention_weights


# ============================================================================
# Testing
# ============================================================================

def test_additive_attention():
    """Comprehensive test of the AdditiveAttention implementation."""
    print("Testing AdditiveAttention...")

    batch_size, src_len = 2, 5
    n = 128                      # decoder units; the annotations have 2n = 256
    attention = AdditiveAttention(hidden_size=64, query_size=n, key_size=2 * n)

    decoder_hidden = torch.randn(batch_size, n)
    annotations = torch.randn(batch_size, src_len, 2 * n)

    context, weights = attention(decoder_hidden, annotations)
    assert context.shape == (batch_size, 2 * n), f"Wrong context shape: {context.shape}"
    assert weights.shape == (batch_size, src_len), f"Wrong weights shape: {weights.shape}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5), \
        f"Weights don't sum to 1: {weights.sum(dim=-1)}"
    print("  Shape tests passed")
    print("  Weights sum to 1")

    mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
    mask[:, -2:] = True  # Mask last 2 positions
    context_masked, weights_masked = attention(decoder_hidden, annotations, mask=mask)
    assert (weights_masked[:, -2:] == 0).all(), "Masked positions have non-zero weights!"
    assert torch.allclose(weights_masked.sum(dim=-1), torch.ones(batch_size), atol=1e-5), \
        "Masked weights don't sum to 1"
    print("  Masking works correctly")

    context.sum().backward()
    assert attention.key_proj.weight.grad is not None, "No gradients for U_a!"
    assert attention.query_proj.weight.grad is not None, "No gradients for W_a!"
    assert attention.energy.weight.grad is not None, "No gradients for v_a!"
    print("  Gradients flow correctly")

    print("Test passed.")
    return attention


if __name__ == '__main__':
    test_additive_attention()
