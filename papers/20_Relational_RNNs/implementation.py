"""
Day 20: Relational Recurrent Neural Networks - the Relational Memory Core (RMC)
==============================================================================

Paper: Adam Santoro et al. (2018), "Relational recurrent neural networks", arXiv 1806.01822.
Where the paper leaves a detail out, we follow DeepMind's reference implementation, which the
paper's footnote 2 points to: Sonnet v1, sonnet/python/modules/relational_memory.py.

[PATCH 2026-10-18] This file used to hold a simplified cell (input tiled onto every slot as extra
features, N x N attention, no MLP, residual + LayerNorm computed and then discarded, gates from a
linear map of [M; x]) that the docs presented as the paper's model. It is replaced by the RMC of
Sec. 3 / Fig. 1. The old cell is still in git history (commit dfcaad3; validation/day20_rmc.py
checks it from there).

One RMC step, for memory M (N slots x F features) and input x_t (Fig. 1a, read top to bottom):

    x~  = W_in x_t                               project the input to ONE memory-sized row (Sonnet)
    A   = MHDPA(M, [M; x~])                      Eq. 2: queries from the N memory rows, keys and values
                                                 from the N+1 rows of [M; x~] -> N x (N+1) attention
    M1  = LN(M + A)                              residual + LayerNorm             (Fig. 1a; LN: Sonnet)
    G   = LN(M1 + MLP(M1))                       row-wise MLP g_psi + residual + LayerNorm (Sec. 3.3)
    i,f = sigma(W x~ + U tanh(M) + b + bias)     per-row input / forget gates      (Eqs. 4-5, Sonnet form)
    M'  = i * tanh(G) + f * M                    Eq. 7, no output gate ("we did not find output gates
                                                 necessary", Sec. 3.3)
    out = flatten(M')                            the core's output (Sonnet: BatchFlatten(next_memory))

Names: M~ (paper) = the attended memory before the gating. Shapes use B = batch, N = memory slots,
F = mem_size = num_heads * head_size, H = heads, T = time steps.
"""

import torch
import torch.nn as nn


# [PATCH 2026-10-18] Sonnet's Linear initialisation, a detail the paper leaves out
def sonnet_linear_init(module):
    """Initialise every nn.Linear inside `module` like Sonnet v1's snt.Linear: weights from a normal
    distribution with std 1/sqrt(fan_in), truncated at 2 std, and zero biases. (PyTorch's default is
    uniform with a sqrt(3) smaller std and random biases.)"""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            std = layer.in_features ** -0.5
            nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(layer.bias)
    return module


class RelationalMemory(nn.Module):
    """Relational Memory Core, one time step per call (Santoro et al. 2018, Sec. 3, Fig. 1).

    Args:
        mem_slots: N, the number of memory rows ("memories"). Parameters are shared across rows,
            so N changes capacity (N * F units) but not the parameter count (Sec. 3.3).
        mem_size: F, the width of each row; must be divisible by num_heads (Sonnet builds it as
            head_size * num_heads).
        input_size: width of the raw input x_t.
        num_heads: H attention heads, each with its own query/key/value projections; head outputs
            are concatenated column-wise back to width F (Sec. 3.1).
        num_blocks: rounds of attention + MLP per time step ("num blocks", App. A). The paper used
            1 for Nth Farthest (App. A.1). See `attend_over_memory` for how >1 differs from Sonnet.
        attention_mlp_layers: depth of the row-wise MLP g_psi (Sonnet default 2, width F, ReLU).
        gate_style: 'unit' = one gate per unit (Sonnet default) or 'memory' = one scalar gate per
            memory row (the paper's "memory gating", Sec. 3.3). App. A.1 swept both.
        forget_bias, input_bias: constants added inside the gate sigmoids; forget_bias = 1 is the
            paper's b~_f in Eq. 7 and Sonnet's default (keeps memory by default at the start).
        key_size: width of queries/keys per head (Sonnet default: head_size = F / H).
        qkv_layer_norm: Sonnet applies LayerNorm to the concatenated q, k, v projections. It is
            not in the paper's Eqs. 1-2; True follows Sonnet, False gives the bare Eq. 2.
    """

    # [PATCH 2026-10-18] paper/Sonnet RMC replaces the simplified tiled-input cell (see module docstring)
    def __init__(self, mem_slots, mem_size, input_size, num_heads=4, num_blocks=1,
                 attention_mlp_layers=2, gate_style='unit', forget_bias=1.0, input_bias=0.0,
                 key_size=None, qkv_layer_norm=True):
        super().__init__()
        if mem_size % num_heads != 0:
            raise ValueError(f"mem_size ({mem_size}) must be divisible by num_heads ({num_heads})")
        if gate_style not in ('unit', 'memory'):
            raise ValueError("gate_style must be 'unit' or 'memory'")
        if num_blocks < 1 or attention_mlp_layers < 1:
            raise ValueError("num_blocks and attention_mlp_layers must be >= 1")
        self.mem_slots = mem_slots
        self.mem_size = mem_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.head_size = mem_size // num_heads            # value width per head
        self.key_size = key_size or self.head_size        # query/key width per head (d_k)
        self.gate_style = gate_style
        self.forget_bias = forget_bias
        self.input_bias = input_bias

        # x_t -> x~, one memory-sized row (Sonnet: basic.Linear(mem_size) on the flattened input)
        self.input_projector = nn.Linear(input_size, mem_size)

        # Eq. 1-2 / Fig. 1b: one Linear makes q, k, v for every head and every row (row-wise shared
        # weights W^q, W^k, W^v). Per head the layout is [q (key_size) | k (key_size) | v (head_size)].
        qkv_size = num_heads * (2 * self.key_size + self.head_size)
        self.qkv = nn.ModuleList([nn.Linear(mem_size, qkv_size) for _ in range(num_blocks)])
        self.qkv_norm = nn.ModuleList([nn.LayerNorm(qkv_size) if qkv_layer_norm else nn.Identity()
                                       for _ in range(num_blocks)])
        # Fig. 1a: residual + LayerNorm after attention, and after the row-wise MLP g_psi.
        self.norm_attention = nn.ModuleList([nn.LayerNorm(mem_size) for _ in range(num_blocks)])
        self.norm_mlp = nn.ModuleList([nn.LayerNorm(mem_size) for _ in range(num_blocks)])
        layers = []
        for k in range(attention_mlp_layers):              # Sonnet mlp.MLP([F] * layers): ReLU between
            layers.append(nn.Linear(mem_size, mem_size))   # layers, none after the last one
            if k < attention_mlp_layers - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)                  # one MLP shared by all blocks, as in Sonnet

        # Gates (Eqs. 4-5 in Sonnet's form): [i; f] = W x~ + U tanh(M_{t-1}), one row at a time.
        gate_size = mem_size if gate_style == 'unit' else 1
        self.gate_input_proj = nn.Linear(mem_size, 2 * gate_size)    # W^i, W^f (on the projected input)
        self.gate_memory_proj = nn.Linear(mem_size, 2 * gate_size)   # U^i, U^f (on tanh of the memory)
        # Sonnet's initialisation for every Linear: zero bias, so input_bias / forget_bias are the only
        # gate biases at the start (forget gate sigma(1) = 0.73).
        sonnet_linear_init(self)

        # Initial memory, as in Sonnet's initial_state: the N x N identity, zero-padded (or cut) to
        # N x F, and NOT trained. Distinct rows matter: the step treats rows symmetrically (shared
        # weights), so rows that started identical would stay identical forever.
        init = torch.eye(mem_slots, mem_size)                        # (N, F): ones at (i, i), zeros elsewhere
        self.register_buffer('initial_memory', init.unsqueeze(0))   # (1, N, F)

    def initial_state(self, batch_size):
        """(B, N, F) copy of the identity initial memory for a batch."""
        return self.initial_memory.expand(batch_size, -1, -1)

    def project_input(self, input_step):
        """x_t (B, input_size) -> x~ as one extra memory row (B, 1, F)."""
        return self.input_projector(input_step).unsqueeze(1)

    def multihead_attention(self, memory, x_tilde, block=0):
        """Eq. 2 with H heads (Sec. 3.1): M~ = softmax(M W^q ([M; x~] W^k)^T / sqrt(d_k)) [M; x~] W^v.

        memory: (B, N, F); x_tilde: (B, 1, F).
        Returns the attended memory (B, N, F), heads concatenated column-wise, and the attention
        weights (B, H, N, N+1): row i = memory slot i's query, column j < N = slot j, column N = input.
        Like Sonnet there is no output projection after the heads.
        """
        B, N, _ = memory.shape
        H, dk, dv = self.num_heads, self.key_size, self.head_size
        mem_plus_input = torch.cat([memory, x_tilde], dim=1)                 # [M; x~]: row-wise -> (B, N+1, F)
        qkv = self.qkv_norm[block](self.qkv[block](mem_plus_input))           # (B, N+1, H*(2dk+dv))
        qkv = qkv.view(B, N + 1, H, 2 * dk + dv).transpose(1, 2)              # (B, H, N+1, 2dk+dv)
        q, k, v = qkv.split([dk, dk, dv], dim=-1)                             # per-head q, k, v
        q = q[:, :, :N] * dk ** -0.5          # queries from the N memory rows only (Eq. 2), scaled by 1/sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1))                         # (B, H, N, N+1)
        weights = torch.softmax(scores, dim=-1)                               # row-wise softmax over N+1 keys
        attended = torch.matmul(weights, v)                                   # (B, H, N, dv)
        attended = attended.transpose(1, 2).reshape(B, N, H * dv)             # [M~_1 : ... : M~_H] -> (B, N, F)
        return attended, weights

    def attend_over_memory(self, memory, x_tilde):
        """Fig. 1a up to the gating: attention, residual + LN, row-wise MLP g_psi, residual + LN.

        With num_blocks > 1 the memory rows go round this loop again and re-read the same input row
        x~. Sonnet instead runs every block over all N+1 rows (so the input row is updated between
        blocks too) and drops the input row at the end; for num_blocks = 1 the two are identical.
        Returns (B, N, F) and the last block's attention weights (B, H, N, N+1).
        """
        for b in range(self.num_blocks):
            attended, weights = self.multihead_attention(memory, x_tilde, b)
            memory = self.norm_attention[b](memory + attended)          # residual + LayerNorm
            memory = self.norm_mlp[b](memory + self.mlp(memory))        # g_psi: row-wise MLP + residual + LN
        return memory, weights

    def create_gates(self, x_tilde, memory):
        """Input and forget gates, one per row (Eqs. 4-5, 7 in Sonnet's form).

        The paper computes the gates from x_t and the row's previous output h_{i,t-1}; with the output
        gate removed, Eq. 8 makes that h_{i,t-1} = tanh(m_{i,t-1}), so Sonnet uses tanh(M_{t-1})
        (and the projected input x~ in place of x_t). x~ (B, 1, F) is broadcast to every row.
        Returns input_gate, forget_gate: (B, N, F) for 'unit' gating, (B, N, 1) for 'memory' gating.
        """
        gates = self.gate_memory_proj(torch.tanh(memory)) + self.gate_input_proj(x_tilde)  # (B, N, 2g)
        input_gate, forget_gate = gates.chunk(2, dim=-1)       # first half input gate, second forget (Sonnet order)
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)   # + b~_f of Eq. 7
        return input_gate, forget_gate

    def forward(self, input_step, memory_state=None):
        """One RMC step.

        input_step: (B, input_size); memory_state: (B, N, F) or None for the identity initial memory.
        Returns next_memory (B, N, F) and the attention weights (B, H, N, N+1). The core's output is
        next_memory flattened to (B, N*F) (see RelationalRNN).
        """
        if memory_state is None:
            memory_state = self.initial_state(input_step.size(0))
        x_tilde = self.project_input(input_step)                          # (B, 1, F)
        candidate, weights = self.attend_over_memory(memory_state, x_tilde)
        input_gate, forget_gate = self.create_gates(x_tilde, memory_state)
        # Eq. 7: m_t = sigma(f + b~_f) * m_{t-1} + sigma(i) * g_psi(m~); Sonnet adds the tanh.
        next_memory = input_gate * torch.tanh(candidate) + forget_gate * memory_state
        return next_memory, weights


# [PATCH 2026-10-18] the output MLP every model shares in App. A.1
def mlp_head(in_size, hidden_sizes, output_size):
    """Output MLP: ReLU after each hidden layer, then a linear layer to the outputs/logits.
    App. A.1: every model (RMC, LSTM, DNC) had the same 4-layer MLP (256 units, ReLU) before the
    softmax; Sonnet's example uses snt.nets.MLP(activate_final=True) then snt.Linear."""
    layers, width = [], in_size
    for h in hidden_sizes:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


class RelationalRNN(nn.Module):
    """RMC unrolled over a sequence; the final memory, flattened, goes through an output MLP.

    rmc_kwargs are passed to RelationalMemory (num_blocks, gate_style, forget_bias, ...).
    """

    # [PATCH 2026-10-18] output MLP option (App. A.1) and kwargs for the paper-faithful core
    def __init__(self, input_size, mem_slots, mem_size, output_size, num_heads=4,
                 head_hidden=(), **rmc_kwargs):
        super().__init__()
        self.rmc = RelationalMemory(mem_slots, mem_size, input_size, num_heads, **rmc_kwargs)
        self.head = mlp_head(mem_slots * mem_size, head_hidden, output_size)

    def forward(self, x, return_attention=False):
        # x: (B, T, input_size)
        memory, attention_history = None, []
        for t in range(x.size(1)):
            memory, weights = self.rmc(x[:, t, :], memory)
            if return_attention:
                attention_history.append(weights)
        output = self.head(memory.reshape(x.size(0), -1))   # (B, N*F): Sonnet's core output
        if return_attention:
            return output, torch.stack(attention_history, dim=1)   # (B, T, H, N, N+1)
        return output


class StandardLSTM(nn.Module):
    """LSTM baseline: final hidden state h_T -> the same kind of output MLP as RelationalRNN."""

    # [PATCH 2026-10-18] same output-MLP option as RelationalRNN so the two differ only in the core
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, head_hidden=()):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = mlp_head(hidden_size, head_hidden, output_size)

    def forward(self, x):
        # x: (B, T, input); h_n: (num_layers, B, hidden) -> last layer's final state
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


# [PATCH 2026-10-18] shared helper (train_minimal, notebook)
def count_parameters(model):
    """Number of trainable parameters (the identity initial memory is a buffer, not counted)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
