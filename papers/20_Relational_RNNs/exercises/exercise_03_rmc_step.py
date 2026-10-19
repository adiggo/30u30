import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# [PATCH 2026-10-18] rewritten for the paper's RMC step: the input is appended as an extra memory ROW
# (Eq. 2), queries come from the memory rows only (N x (N+1) attention), then residual + LayerNorm,
# a row-wise MLP with residual + LayerNorm, and gates from x~ and tanh(M_{t-1}). The old version tiled
# the input onto every slot as extra features (N x N attention) and had no MLP.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # for the test


class RelationalMemoryStep(nn.Module):
    """One step of the Relational Memory Core (Santoro et al. 2018, Sec. 3, Fig. 1a; Sonnet).

        x~  = input_projector(x_t)                    one memory-sized row          (B, 1, F)
        A   = MHDPA: queries from M, keys/values from [M; x~]   (Eq. 2)             (B, N, F)
        M1  = LN(M + A);  G = LN(M1 + MLP(M1))        residuals + LayerNorms (Fig. 1a)
        i,f = sigmoid(gate_memory_proj(tanh M) + gate_input_proj(x~) + [0; 1])
        M_t = i * tanh(G) + f * M
    """

    def __init__(self, mem_slots, mem_size, input_size, num_heads=4):
        super().__init__()
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.num_heads = num_heads
        self.head_size = mem_size // num_heads
        self.forget_bias, self.input_bias = 1.0, 0.0

        self.input_projector = nn.Linear(input_size, mem_size)
        # q, k, v of every head from one Linear; per head the layout is [q | k | v], each head_size wide
        self.qkv = nn.Linear(mem_size, 3 * mem_size)
        self.qkv_norm = nn.LayerNorm(3 * mem_size)        # Sonnet normalises q, k, v together
        self.norm_attention = nn.LayerNorm(mem_size)
        self.mlp = nn.Sequential(nn.Linear(mem_size, mem_size), nn.ReLU(), nn.Linear(mem_size, mem_size))
        self.norm_mlp = nn.LayerNorm(mem_size)
        self.gate_input_proj = nn.Linear(mem_size, 2 * mem_size)
        self.gate_memory_proj = nn.Linear(mem_size, 2 * mem_size)

    def forward(self, input_step, memory_state):
        """input_step: (B, input_size); memory_state: (B, N, F). Returns next memory (B, N, F) and the
        attention weights (B, H, N, N+1)."""
        B, N, H, d = input_step.size(0), self.mem_slots, self.num_heads, self.head_size

        # TODO 1: project the input to one memory row x~ of shape (B, 1, F)
        x_tilde = None  # YOUR CODE HERE
        if x_tilde is None:  # [PATCH 2026-10-18] delete this guard once done
            raise NotImplementedError("Exercise 3: fill in the TODOs in RelationalMemoryStep.forward")

        # TODO 2: [M; x~] - concatenate along the ROW dimension -> (B, N+1, F)
        mem_plus_input = None  # YOUR CODE HERE

        # q, k, v for all N+1 rows, split into heads: (B, N+1, 3F) -> (B, H, N+1, 3d)
        qkv = self.qkv_norm(self.qkv(mem_plus_input))
        qkv = qkv.view(B, N + 1, H, 3 * d).transpose(1, 2)
        q, k, v = qkv.split([d, d, d], dim=-1)

        # TODO 3: keep the queries of the N memory rows only (Eq. 2), then
        # weights = softmax(q k^T / sqrt(d)) over the N+1 keys -> (B, H, N, N+1), attended = weights @ v
        weights = None  # YOUR CODE HERE
        attended = None  # YOUR CODE HERE

        # merge heads column-wise: (B, H, N, d) -> (B, N, H*d) = (B, N, F)
        attended = attended.transpose(1, 2).reshape(B, N, H * d)

        # TODO 4: residual + LayerNorm (norm_attention), then MLP + residual + LayerNorm (norm_mlp) -> G
        candidate = None  # YOUR CODE HERE

        # TODO 5: gates from tanh(memory_state) and x~ (see exercise 2): chunk into input/forget halves,
        # add input_bias / forget_bias, sigmoid; then next_memory = i * tanh(G) + f * M
        next_memory = None  # YOUR CODE HERE

        return next_memory, weights


def test_rmc_step():
    print("Testing One RMC Step...")
    from implementation import RelationalMemory      # the day's reference implementation

    batch, slots, mem_dim, input_dim = 2, 4, 32, 10
    step = RelationalMemoryStep(slots, mem_dim, input_dim)
    x = torch.randn(batch, input_dim)
    m = torch.randn(batch, slots, mem_dim)
    m_new, weights = step(x, m)
    assert m_new.shape == (batch, slots, mem_dim)
    assert weights.shape == (batch, 4, slots, slots + 1), f"attention should be N x (N+1), got {tuple(weights.shape)}"

    # Same weights in implementation.RelationalMemory -> same next memory and attention
    ref = RelationalMemory(slots, mem_dim, input_dim, num_heads=4)
    pairs = {'input_projector': ref.input_projector, 'qkv': ref.qkv[0], 'qkv_norm': ref.qkv_norm[0],
             'norm_attention': ref.norm_attention[0], 'mlp': ref.mlp, 'norm_mlp': ref.norm_mlp[0],
             'gate_input_proj': ref.gate_input_proj, 'gate_memory_proj': ref.gate_memory_proj}
    for name, module in pairs.items():
        module.load_state_dict(getattr(step, name).state_dict())
    m_ref, w_ref = ref(x, m)
    assert torch.allclose(m_new, m_ref, atol=1e-5), "next memory differs from implementation.RelationalMemory"
    assert torch.allclose(weights, w_ref, atol=1e-5), "attention differs from implementation.RelationalMemory"

    # Permuting the memory slots permutes the new memory the same way (shared weights per row)
    perm = torch.tensor([3, 1, 0, 2])
    assert torch.allclose(step(x, m[:, perm])[0], m_new[:, perm], atol=1e-5), "step must be slot-equivariant"
    print("Test Passed!")


if __name__ == "__main__":
    test_rmc_step()
