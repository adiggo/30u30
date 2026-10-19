"""
Order Matters: models for input sets
====================================

Paper: "Order Matters: Sequence to Sequence for Sets"
       Vinyals, Bengio, Kudlur (ICLR 2016)
       https://arxiv.org/abs/1511.06391

This module contains:

1. ReadProcessWrite - the paper's model for input sets (Sec. 4.3, Figure 1):
     Read    (ReadBlock)       the same small MLP embeds every element x_i into a memory m_i
     Process (ProcessBlock)    an LSTM with no data input runs T steps of content-based
                               attention over the memories (Eqs. 3-7); its final state q*_T
                               is invariant to the order of the set
     Write   (PointerDecoder)  an LSTM pointer network that starts from q*_T and points at the
                               memories one step at a time, optionally with a "glimpse"
                               (an attention read) before each pointer (Sec. 4.3)
   Nothing in the model sees the input order except through content-based attention, so
   permuting the input set permutes the output distributions over its elements in the same
   way (validation/day16_process_block_check.py).

2. PointerNetwork - the Ptr-Net of Vinyals, Fortunato, Jaitly (2015): an LSTM reads the set
   *as a sequence* (order-sensitive) and an LSTM decoder points at inputs. This is the
   "Ptr-Net" baseline of the paper's Table 1 (without glimpses).

3. SetEncoder - a Transformer encoder without positional encodings: permutation-EQUIVARIANT
   per-element vectors (the idea of exercise 2, as in Set Transformer). It is not the paper's
   read block. Until 2026-10-18 ReadProcessWrite was SetEncoder followed by PointerNetwork,
   whose LSTM re-read the encoded set in order, so that model was not permutation-invariant;
   the validation scripts rebuild it from these two classes for comparison.

Both pointer models share one interface:
    forward(inputs (B, N, D), input_lengths (B,), max_steps, teacher_forcing=None)
        -> pointers (B, max_steps), log_probs (B, max_steps), attentions
    log_probs[:, t] = log p(pointer_t); attentions is a list of max_steps tensors (B, slots)
    holding the pointer distributions. With teacher_forcing=targets the pointer fed back at
    step t is targets[:, t] (so log_probs holds log p(target_t)); otherwise it is the argmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# [PATCH 2026-10-18] Targets of variable-length outputs (convex hull) end with a pointer to
# the STOP slot and are padded with IGNORE_INDEX; loss and accuracy skip those steps.
IGNORE_INDEX = -1


class PointerNetwork(nn.Module):
    """
    Pointer Network for selecting elements from a set.
    
    Instead of generating tokens from a vocabulary, this network "points"
    to elements in the input sequence.
    
    Key property: The output space is the indices of the input set.

    [PATCH 2026-10-18] Note: the encoder is an LSTM that reads the elements in the order
    given, so this model is NOT permutation-invariant. It is the "Ptr-Net" baseline of
    Table 1 in Order Matters (Sec. 4.4); ReadProcessWrite below is the paper's set model.
    mask_chosen=False keeps already chosen elements selectable (the original Ptr-Net was
    trained without such a mask); the default masks them, as before.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        mask_chosen: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mask_chosen = mask_chosen  # [PATCH 2026-10-18] see class docstring
        
        # Encoder: Process the input set
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder: Generate pointers
        self.decoder_cell = nn.LSTMCell(input_dim, hidden_dim)
        
        # Attention mechanism for pointing
        # Formula: e_i = v^T tanh(W1*encoder_i + W2*decoder)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_steps: int,
        teacher_forcing: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Forward pass for pointer network.
        
        Args:
            inputs: [batch, seq_len, input_dim] - The input set (possibly padded)
            input_lengths: [batch] - Actual lengths before padding
            max_steps: Number of elements to point to
            teacher_forcing: [batch, max_steps] - Ground truth pointers
            temperature: Softmax temperature
            
        Returns:
            pointers: [batch, max_steps] - Selected indices
            log_probs: [batch, max_steps] - Log probabilities
            attention_weights: List of [batch, seq_len] for visualization
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        device = inputs.device
        
        # Phase 1: Encode
        encoder_outputs, (hidden, cell) = self.encoder(inputs)
        
        # Initialize decoder state
        decoder_hidden = hidden[-1]
        decoder_cell = cell[-1]
        
        # Start with a zero vector (no previous selection)
        decoder_input = torch.zeros(batch_size, self.input_dim, device=device)
        
        # Padding mask: 1 = valid, 0 = padding
        mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < input_lengths.unsqueeze(1)).float()
        
        # Phase 2: Decode
        pointers = []
        log_probs_list = []
        attention_weights_list = []
        
        # Track selected elements to enable sampling without replacement
        selection_mask = mask.clone()
        
        for step in range(max_steps):
            decoder_hidden, decoder_cell = self.decoder_cell(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            decoder_hidden = self.dropout(decoder_hidden)
            
            # Attention Mechanism (Eq 1 in Paper)
            encoder_proj = self.W1(encoder_outputs)
            decoder_proj = self.W2(decoder_hidden).unsqueeze(1)
            combined = torch.tanh(encoder_proj + decoder_proj)
            scores = self.v(combined).squeeze(-1) / temperature
            
            # Mask padding and selected elements
            scores = scores.masked_fill(selection_mask == 0, float('-inf'))
            
            probs = F.softmax(scores, dim=-1)
            log_probs = F.log_softmax(scores, dim=-1)
            
            if teacher_forcing is not None and step < teacher_forcing.size(1):
                pointer = teacher_forcing[:, step]
            else:
                pointer = torch.argmax(probs, dim=-1)
            
            pointers.append(pointer)
            log_probs_list.append(log_probs.gather(1, pointer.unsqueeze(1)))
            attention_weights_list.append(probs)
            
            if self.mask_chosen:  # [PATCH 2026-10-18]
                selection_mask = selection_mask.scatter(1, pointer.unsqueeze(1), 0)
            decoder_input = inputs[torch.arange(batch_size, device=device), pointer]
        
        pointers = torch.stack(pointers, dim=1)
        log_probs = torch.cat(log_probs_list, dim=1)
        
        return pointers, log_probs, attention_weights_list


class SetEncoder(nn.Module):
    """
    Transformer encoder without positional encodings.

    [PATCH 2026-10-18] The original docstring called this "order-invariant". It is
    permutation-EQUIVARIANT: enc(x[perm]) == enc(x)[perm], i.e. each element gets the same
    vector wherever it sits; only a symmetric pooling of the outputs (mean, sum, attention
    read-out) is invariant. It is not the paper's read block (a small per-element MLP,
    ReadBlock below) and ReadProcessWrite does not use it; it is kept as a stand-alone
    module (exercise 2's idea) and to rebuild the pre-2026-10-18 model in validation/.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Embed inputs
        x = self.embedding(x)
        
        # Self-attention without positional encodings: permutation-equivariant
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        
        return x


# [PATCH 2026-10-18] The paper's Read-Process-Write model (Sec. 4.2-4.4). Before this patch
# ReadProcessWrite was SetEncoder (Transformer) -> PointerNetwork: no process block, and the
# pointer network's LSTM re-read the encoded set in order, so the model depended on the input
# order -- the property the paper set out to remove. Read, Process and Write below follow the
# paper; validation/day16_process_block_check.py tests the invariance.


class ReadBlock(nn.Module):
    """Read (Sec. 4.3; Sec. 4.4: "a small multilayer perceptron for each number").

    m_i = MLP(x_i) with the same MLP for every element. No element sees any other, so
    permuting the inputs permutes the memories and nothing else (equivariance).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, D_in) -> (B, N, H): nn.Linear acts on the last axis only, so each element
        # is embedded on its own.
        return self.net(x)


class ProcessBlock(nn.Module):
    """Process (Sec. 4.2, Eqs. 3-7): an LSTM with no data input that, for T steps, reads
    the memories with content-based attention:

        q_t     = LSTM(q*_{t-1})                        (3)
        e_{i,t} = f(m_i, q_t) = m_i . q_t               (4)  f = dot product (the paper's example)
        a_{i,t} = exp(e_{i,t}) / sum_j exp(e_{j,t})     (5)
        r_t     = sum_i a_{i,t} m_i                     (6)
        q*_t    = [q_t, r_t]                            (7)

    q*_0 = 0 and the LSTM state starts at 0, so the data enter only through the read-outs
    r_t. Permuting the memories permutes e_t and a_t in the same way and only reorders the
    terms of the sum in (6), so every r_t and q*_t is unchanged (Sec. 4.2: "permuting m_i
    and m_i' has no effect on the read vector r_t"). With T = 0 the block returns q*_0 = 0.

    Implementation note: nn.LSTMCell(2H -> H) whose only input is q*_{t-1}, the usual reading
    of Eq. 3 (it is the Set2Set read-out of Gilmer et al. 2017 and PyTorch Geometric). The
    cell's recurrent term W_hh q_{t-1} repeats the q_{t-1} half of q*_{t-1}; harmless.
    """

    def __init__(self, hidden_dim: int, steps: int):
        super().__init__()
        if steps < 0:
            raise ValueError("process steps T must be >= 0")
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.cell = nn.LSTMCell(2 * hidden_dim, hidden_dim)

    def forward(
        self, memories: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            memories: (B, N, H) memory vectors m_i from the read block
            mask:     (B, N) bool, True for real elements (False = padding, never attended)
        Returns:
            q_star:     (B, 2H) q*_T
            attentions: list of T tensors (B, N), the weights a_t of Eq. 5
        """
        B, _, H = memories.shape
        q_star = memories.new_zeros(B, 2 * H)                          # q*_0 = 0
        state = (memories.new_zeros(B, H), memories.new_zeros(B, H))   # LSTM (h, c) start at 0
        attentions = []
        for _ in range(self.steps):
            q, c = self.cell(q_star, state)                            # (3): q_t = LSTM(q*_{t-1})
            state = (q, c)
            # (4): e[b, i] = sum_h m[b, i, h] * q[b, h], one dot product per memory -> (B, N)
            e = torch.einsum("bnh,bh->bn", memories, q)
            e = e.masked_fill(~mask, float("-inf"))                    # padding gets weight 0
            a = torch.softmax(e, dim=-1)                               # (5): softmax over i
            # (6): r[b, h] = sum_i a[b, i] * m[b, i, h], the attention-weighted mean -> (B, H)
            r = torch.einsum("bn,bnh->bh", a, memories)
            q_star = torch.cat([q, r], dim=-1)                         # (7): (B, 2H)
            attentions.append(a)
        return q_star, attentions


class PointerDecoder(nn.Module):
    """Write (Sec. 4.3): an LSTM pointer network (Vinyals et al. 2015a) that "takes in q*_T
    ... and points at elements of m_i, one step at a time".

    - Start: (h_0, c_0) = W_init q*_T + b, split in two; this is how q*_T enters.
    - Input at step 1 is a learned "go" vector (Ptr-Net's start symbol); at step t > 1 it is
      the memory of the element pointed at in step t-1 (Ptr-Net feeds back its last output).
    - Glimpse (optional, Sec. 4.3: "an extra attention step before the pointer"): with the
      decoder state d_t as query, an attention read over the still-available memories,
          g_t = sum_i softmax_i(w_g^T tanh(G1 m_i + G2 d_t)) m_i,
      after which the pointer's query is [d_t, g_t] (the same pattern as q*_t = [q_t, r_t]).
      The paper gives no formula; this follows the additive form of the pointer itself.
    - Pointer (Ptr-Net): u_{i,t} = v^T tanh(W1 m_i + W2 query_t); padded and already chosen
      elements get u = -inf (probability exactly 0); p(y_t = i | y_<t, X) = softmax_i(u_{i,t}).

    Every score is a function of one memory and of quantities that do not depend on the input
    order, so permuting the memories permutes the distribution over them.

    mask_chosen=False leaves chosen elements selectable (only padding is masked), so the
    decoder must learn not to repeat itself, as in the original Ptr-Net training. The paper
    does not say whether it masked; masking makes sorting much easier (README, Results).
    """

    def __init__(self, hidden_dim: int, glimpses: int = 1, mask_chosen: bool = True):
        super().__init__()
        if glimpses not in (0, 1):
            raise ValueError("glimpses must be 0 or 1 (the two settings compared in Table 1)")
        H = hidden_dim
        self.hidden_dim = H
        self.glimpses = glimpses
        self.mask_chosen = mask_chosen
        self.init_state = nn.Linear(2 * H, 2 * H)       # q*_T -> (h_0, c_0)
        self.go = nn.Parameter(torch.zeros(H))          # learned first input
        self.cell = nn.LSTMCell(H, H)
        if glimpses:
            self.G1 = nn.Linear(H, H, bias=False)
            self.G2 = nn.Linear(H, H, bias=False)
            self.w_g = nn.Linear(H, 1, bias=False)
        self.W1 = nn.Linear(H, H, bias=False)
        self.W2 = nn.Linear(H * (1 + glimpses), H, bias=False)
        self.v = nn.Linear(H, 1, bias=False)

    @staticmethod
    def _additive_scores(keys: torch.Tensor, query: torch.Tensor, v: nn.Linear,
                         available: torch.Tensor) -> torch.Tensor:
        # keys (B, S, H) + query (B, 1, H) broadcasts the query over all S slots -> (B, S, H);
        # v maps H -> 1 and squeeze drops that axis -> scores (B, S)
        u = v(torch.tanh(keys + query)).squeeze(-1)
        return u.masked_fill(~available, float("-inf"))

    def forward(
        self,
        memories: torch.Tensor,
        available: torch.Tensor,
        q_star: torch.Tensor,
        max_steps: int,
        teacher_forcing: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        stop_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            memories:  (B, S, H) memories to point at (S = N, or N + 1 with a STOP slot)
            available: (B, S) bool, True where pointing is allowed (real elements, STOP)
            q_star:    (B, 2H) output of the process block
            stop_index: index of the STOP slot, or None. Pointing at STOP ends the output:
                the STOP slot is never masked, and in greedy decoding every later step
                repeats it. Teacher-forced targets may hold IGNORE_INDEX after the STOP.
        """
        B, S, H = memories.shape
        rows = torch.arange(B, device=memories.device)
        h, c = self.init_state(q_star).chunk(2, dim=-1)       # (B, 2H) -> two (B, H) halves
        x = self.go.expand(B, H)                               # same go vector for every row
        keys = self.W1(memories)                               # (B, S, H), computed once
        glimpse_keys = self.G1(memories) if self.glimpses else None
        if stop_index is not None:
            is_stop = torch.zeros(B, S, dtype=torch.bool, device=memories.device)
            is_stop[:, stop_index] = True
            finished = torch.zeros(B, dtype=torch.bool, device=memories.device)

        pointers, log_probs, attentions = [], [], []
        for t in range(max_steps):
            h, c = self.cell(x, (h, c))                        # d_t = h
            query = h
            if self.glimpses:
                g_scores = self._additive_scores(glimpse_keys, self.G2(h).unsqueeze(1),
                                                 self.w_g, available)
                g_weights = torch.softmax(g_scores, dim=-1)    # (B, S)
                # g[b, h] = sum_s w[b, s] * m[b, s, h]: attention read of the memories -> (B, H)
                glimpse = torch.einsum("bs,bsh->bh", g_weights, memories)
                query = torch.cat([h, glimpse], dim=-1)        # (B, 2H)
            scores = self._additive_scores(keys, self.W2(query).unsqueeze(1), self.v,
                                           available) / temperature
            logp = F.log_softmax(scores, dim=-1)               # (B, S)

            if teacher_forcing is not None and t < teacher_forcing.size(1):
                pointer = teacher_forcing[:, t]
                if stop_index is not None:
                    # steps after the STOP (IGNORE_INDEX) feed STOP again; the loss skips them
                    pointer = torch.where(pointer >= 0, pointer, torch.full_like(pointer, stop_index))
                elif (pointer < 0).any():
                    raise ValueError("targets contain IGNORE_INDEX but the model has no STOP slot "
                                     "(build ReadProcessWrite with stop_token=True)")
            else:
                pointer = logp.argmax(dim=-1)
                if stop_index is not None:
                    pointer = torch.where(finished, torch.full_like(pointer, stop_index), pointer)
            if stop_index is not None:
                finished = finished | (pointer == stop_index)

            pointers.append(pointer)
            log_probs.append(logp.gather(1, pointer.unsqueeze(1)))   # (B, 1): log p(pointer_t)
            attentions.append(logp.exp())
            if self.mask_chosen:
                # scatter writes False at column pointer[b] of row b: the chosen element is used up
                available = available.scatter(1, pointer.unsqueeze(1), False)
            if stop_index is not None:
                available = available | is_stop               # STOP always stays available
            x = memories[rows, pointer]                        # (B, H): memory of each row's choice

        return torch.stack(pointers, dim=1), torch.cat(log_probs, dim=1), attentions


class ReadProcessWrite(nn.Module):
    """
    The Read-Process-Write model of Order Matters (Sec. 4.3, Figure 1; Sec. 4.4 for sorting).

        Read:    m_i = MLP(x_i)                                   (ReadBlock)
        Process: T steps of Eqs. 3-7 over {m_i} -> q*_T            (ProcessBlock)
        Write:   pointer LSTM started from q*_T, points at m_i     (PointerDecoder)

    Args:
        input_dim:     size of one set element (1 for numbers, 2 for points)
        hidden_dim:    memory / LSTM size H
        process_steps: T, the number of process steps (Table 1 uses P = 0, 1, 5, 10). With
                       T = 0 (and no glimpse) the writer is "unconditioned on X" (Sec. 4.4)
                       apart from the memories it points at.
        glimpses:      0 or 1 attention reads before each pointer (Table 1's two settings)
        stop_token:    append one learned memory, the STOP slot at index N, that the decoder
                       can point at to end a variable-length output (Pointer Networks'
                       end-of-sequence symbol; used for convex hulls)
        mask_chosen:   mask elements already pointed at (default). False = only padding is
                       masked and the decoder must learn not to repeat (see PointerDecoder)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        process_steps: int = 5,
        glimpses: int = 1,
        stop_token: bool = False,
        mask_chosen: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.read = ReadBlock(input_dim, hidden_dim)
        self.process = ProcessBlock(hidden_dim, process_steps)
        self.write = PointerDecoder(hidden_dim, glimpses, mask_chosen)
        self.stop_token = stop_token
        if stop_token:
            self.stop_memory = nn.Parameter(torch.zeros(hidden_dim))
        
    def encode(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Read + Process. Returns memories (B, N, H), mask (B, N) bool (True = real
        element), q*_T (B, 2H) and the process attentions (T tensors (B, N))."""
        N = inputs.size(1)
        memories = self.read(inputs)
        # (1, N) < (B, 1) broadcasts to (B, N): True for the first input_lengths[b] slots
        mask = torch.arange(N, device=inputs.device).unsqueeze(0) < input_lengths.to(inputs.device).unsqueeze(1)
        q_star, process_attention = self.process(memories, mask)
        return memories, mask, q_star, process_attention
    
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        max_steps: int,
        teacher_forcing: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            inputs:          (B, N, input_dim), padded sets
            input_lengths:   (B,) number of real elements per set
            max_steps:       number of pointers to emit
            teacher_forcing: (B, max_steps) target pointers, or None for greedy decoding
        Returns:
            pointers (B, max_steps), log_probs (B, max_steps), attentions: list of
            (B, N) pointer distributions, (B, N + 1) with stop_token (STOP = index N)
        """
        memories, mask, q_star, _ = self.encode(inputs, input_lengths)
        stop_index = None
        if self.stop_token:
            B, N, H = memories.shape
            # append the learned STOP memory as slot N of every set: (B, N, H) -> (B, N + 1, H)
            memories = torch.cat([memories, self.stop_memory.expand(B, 1, H)], dim=1)
            mask = torch.cat([mask, mask.new_ones(B, 1)], dim=1)
            stop_index = N
        return self.write(memories, mask, q_star, max_steps, teacher_forcing,
                          temperature, stop_index)


def compute_loss(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # [PATCH 2026-10-18] PointerNetwork.forward already returns, per step, the
    # log-probability of the pointer it was fed (the target under teacher
    # forcing): log_probs is (batch, steps), not (batch, steps, set_size). The
    # original `log_probs.gather(1, targets)` re-indexed the *step* axis by
    # pointer values; that only equals the NLL when every target row is a
    # permutation of 0..steps-1 (sorting, TSP) and is wrong otherwise (e.g.
    # padded convex-hull labels). Call this on a teacher-forced pass.
    # [PATCH 2026-10-18] Steps whose target is IGNORE_INDEX (after a STOP) are left out:
    # mean NLL over the remaining steps. For targets without IGNORE_INDEX this is
    # -log_probs.mean(), as before.
    valid = targets >= 0
    # torch.where instead of multiplying by the mask: 0 * (-inf) would give NaN
    return -torch.where(valid, log_probs, torch.zeros_like(log_probs)).sum() / valid.sum()


def accuracy(pointers: torch.Tensor, targets: torch.Tensor) -> float:
    # Exact match of whole output sequences. [PATCH 2026-10-18] Steps whose target is
    # IGNORE_INDEX (after a STOP) do not count.
    correct = ((pointers == targets) | (targets < 0)).all(dim=1).float()
    return correct.mean().item()


if __name__ == "__main__":
    # [PATCH 2026-10-18] Quick demonstration: shuffle a set and compare the outputs.
    torch.manual_seed(0)
    x = torch.rand(1, 6, 1, dtype=torch.float64)
    perm = torch.randperm(6)
    inv = torch.argsort(perm)              # element i of x sits at position inv[i] of x[:, perm]
    lengths = torch.tensor([6])
    target = x[..., 0].argsort(dim=1)      # sorting target (B, N); in the shuffled set it is inv[target]
    rpw = ReadProcessWrite(1, 32, process_steps=5, glimpses=1).double().eval()
    ptr = PointerNetwork(1, 32, dropout=0.0).double().eval()
    torch.set_default_dtype(torch.float64)  # PointerNetwork builds its first input in the default dtype
    with torch.no_grad():
        q1, q2 = rpw.encode(x, lengths)[2], rpw.encode(x[:, perm], lengths)[2]
        # teacher forcing with the same elements in both runs, so every step compares like with like
        a1, a2 = rpw(x, lengths, 6, target)[2], rpw(x[:, perm], lengths, 6, inv[target])[2]
        b1, b2 = ptr(x, lengths, 6, target)[2], ptr(x[:, perm], lengths, 6, inv[target])[2]
    # p2[:, inv] reorders the probabilities over the shuffled set back to the original order
    print("ReadProcessWrite, shuffled set: max |q*_T change| =", f"{(q1 - q2).abs().max().item():.1e},",
          "pointer distributions:", f"{max((p2[:, inv] - p1).abs().max().item() for p1, p2 in zip(a1, a2)):.1e}")
    print("PointerNetwork (LSTM encoder), shuffled set: pointer distributions change by",
          f"{max((p2[:, inv] - p1).abs().max().item() for p1, p2 in zip(b1, b2)):.1e}")
