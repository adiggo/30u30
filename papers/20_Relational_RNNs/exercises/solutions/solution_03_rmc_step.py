import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# [PATCH 2026-10-18] the paper's RMC step (input as an extra memory row, N x (N+1) attention, MLP, gates)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # for the test


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

        x_tilde = self.input_projector(input_step).unsqueeze(1)            # (B, 1, F)
        mem_plus_input = torch.cat([memory_state, x_tilde], dim=1)          # [M; x~] row-wise: (B, N+1, F)

        qkv = self.qkv_norm(self.qkv(mem_plus_input))                       # (B, N+1, 3F)
        qkv = qkv.view(B, N + 1, H, 3 * d).transpose(1, 2)                  # (B, H, N+1, 3d)
        q, k, v = qkv.split([d, d, d], dim=-1)

        q = q[:, :, :N]                                                     # queries: memory rows only (Eq. 2)
        weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / d ** 0.5, dim=-1)   # (B, H, N, N+1)
        attended = torch.matmul(weights, v)                                 # (B, H, N, d)
        attended = attended.transpose(1, 2).reshape(B, N, H * d)            # heads side by side: (B, N, F)

        m1 = self.norm_attention(memory_state + attended)                   # residual + LayerNorm
        candidate = self.norm_mlp(m1 + self.mlp(m1))                        # row-wise MLP + residual + LayerNorm

        gates = self.gate_memory_proj(torch.tanh(memory_state)) + self.gate_input_proj(x_tilde)  # (B, N, 2F)
        input_gate, forget_gate = gates.chunk(2, dim=-1)
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)
        next_memory = input_gate * torch.tanh(candidate) + forget_gate * memory_state

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
