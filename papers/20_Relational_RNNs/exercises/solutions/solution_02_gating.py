import torch
import torch.nn as nn

# [PATCH 2026-10-18] the RMC's gates (Santoro et al. 2018, Eqs. 4-5 and 7, Sonnet form) instead of a plain LSTM update


class RMCGates(nn.Module):
    """The RMC's LSTM-style gated memory update, one memory row at a time.

        [i; f] = U tanh(M_{t-1}) + W x~          U, W: Linear(F -> 2F), shared by all rows
        i = sigmoid(i + input_bias)              input gate  (input_bias = 0)
        f = sigmoid(f + forget_bias)             forget gate (forget_bias = 1: remember by default)
        M_t = i * tanh(G) + f * M_{t-1}          G = the attended memory after the MLP (the candidate)

    Why tanh(M_{t-1})? The paper's gates read the row's previous output h_{t-1}; with no output gate
    ("we did not find output gates necessary"), Eq. 8 gives h_{t-1} = tanh(m_{t-1}).
    There is no separate candidate layer: unlike an LSTM, the new content is the attended memory G.
    """

    def __init__(self, mem_size, forget_bias=1.0, input_bias=0.0):
        super().__init__()
        self.forget_bias = forget_bias
        self.input_bias = input_bias
        self.input_proj = nn.Linear(mem_size, 2 * mem_size)    # W^i, W^f applied to the projected input x~
        self.memory_proj = nn.Linear(mem_size, 2 * mem_size)   # U^i, U^f applied to tanh(M_{t-1})

    def forward(self, x_tilde, memory, candidate):
        """
        Args:
            x_tilde: (batch, 1, mem_size) - the projected input row (the same for every memory row)
            memory: (batch, slots, mem_size) - M_{t-1}, the previous memory
            candidate: (batch, slots, mem_size) - G, the attended memory after the MLP
        Returns:
            next_memory, input_gate, forget_gate - each (batch, slots, mem_size)
        """
        # (B, N, 2F) + (B, 1, 2F): the input term broadcasts over the N rows
        gates = self.memory_proj(torch.tanh(memory)) + self.input_proj(x_tilde)
        input_gate, forget_gate = gates.chunk(2, dim=-1)        # first half input, second half forget
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)
        next_memory = input_gate * torch.tanh(candidate) + forget_gate * memory
        return next_memory, input_gate, forget_gate


def test_gating():
    print("Testing RMC gating...")
    batch, slots, dim = 2, 4, 16
    x_t = torch.randn(batch, 1, dim)
    m = torch.randn(batch, slots, dim)
    g = torch.randn(batch, slots, dim)

    gates = RMCGates(dim)
    m_new, i, f = gates(x_t, m, g)
    assert m_new.shape == i.shape == f.shape == (batch, slots, dim)
    assert ((i > 0) & (i < 1) & (f > 0) & (f < 1)).all(), "gates must lie in (0, 1)"

    # Zero weights: every gate is a constant, sigmoid(0) = 0.5 and sigmoid(1) = 0.731
    with torch.no_grad():
        for p in gates.parameters():
            p.zero_()
    m_new, _, _ = gates(x_t, m, g)
    expected = 0.5 * torch.tanh(g) + torch.sigmoid(torch.tensor(1.0)) * m
    assert torch.allclose(m_new, expected, atol=1e-6), "with zero weights: 0.5 tanh(G) + 0.731 M"

    # Limits: forget gate forced to 1 and input gate to 0 keeps the memory; the reverse overwrites it
    gates.forget_bias, gates.input_bias = 1e4, -1e4
    assert torch.allclose(gates(x_t, m, g)[0], m), "closed input gate + open forget gate must keep M"
    gates.forget_bias, gates.input_bias = -1e4, 1e4
    assert torch.allclose(gates(x_t, m, g)[0], torch.tanh(g)), "open input gate + closed forget gate gives tanh(G)"

    print("Test Passed!")


if __name__ == "__main__":
    test_gating()
