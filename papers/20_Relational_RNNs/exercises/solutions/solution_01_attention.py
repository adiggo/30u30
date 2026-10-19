import torch
import torch.nn.functional as F
import math

# [PATCH 2026-10-18] N queries vs N+1 keys/values (paper Eq. 2); test checks values against PyTorch's SDPA


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute 'Scaled Dot Product Attention': softmax(q k^T / sqrt(d_k)) v  (Santoro et al. 2018, Eq. 1).

    In the RMC (Eq. 2) the queries come from the N memory slots only, while keys and values come from
    the N + 1 rows of [M; x~] (the memory plus the projected input row).

    Args:
        q (torch.Tensor): Queries (batch, heads, N, head_dim)
        k (torch.Tensor): Keys (batch, heads, N + 1, head_dim)
        v (torch.Tensor): Values (batch, heads, N + 1, head_dim)
        mask (torch.Tensor): Optional mask broadcastable to (batch, heads, N, N + 1); 0 = blocked

    Returns:
        output (torch.Tensor): (batch, heads, N, head_dim)
        attention_weights (torch.Tensor): (batch, heads, N, N + 1), rows sum to 1
    """
    d_k = q.size(-1)

    # Calculate attention scores: (B, H, N, d) @ (B, H, d, N+1) -> (B, H, N, N+1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax over the keys (last dimension)
    attn_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values: (B, H, N, N+1) @ (B, H, N+1, d) -> (B, H, N, d)
    output = torch.matmul(attn_weights, v)

    return output, attn_weights


def test_attention():
    print("Testing Scaled Dot Product Attention...")
    batch, heads, slots, dim = 2, 4, 8, 16
    q = torch.randn(batch, heads, slots, dim)          # queries: the N memory slots
    k = torch.randn(batch, heads, slots + 1, dim)      # keys: N slots + the input row
    v = torch.randn(batch, heads, slots + 1, dim)

    out, weights = scaled_dot_product_attention(q, k, v)

    assert out.shape == (batch, heads, slots, dim), f"Expected {(batch, heads, slots, dim)}, got {out.shape}"
    assert weights.shape == (batch, heads, slots, slots + 1), \
        f"Expected {(batch, heads, slots, slots + 1)}, got {weights.shape}"
    assert torch.allclose(weights.sum(-1), torch.ones(batch, heads, slots), atol=1e-5), "rows must sum to 1"
    expected = F.scaled_dot_product_attention(q, k, v)  # PyTorch's built-in, same formula
    assert torch.allclose(out, expected, atol=1e-5), "output differs from F.scaled_dot_product_attention"

    print("Test Passed!")


if __name__ == "__main__":
    test_attention()
