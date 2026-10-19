"""
Solution 2: Bidirectional Encoder

Reference implementation of the paper's encoder (Bahdanau et al. 2014, Sec. 3.2 and Appendix A.2.1):
a forward and a backward GRU read the source, and each position j gets the annotation

    h_j = [forward h_j ; backward h_j]          (2n numbers, Eq. 7)

which is what the alignment model scores and averages (Appendix A.2.2, U_a in R^{n' x 2n}).

[PATCH 2026-10-18] The outputs are now the 2n-dim annotations themselves, as exercise_2.py asks
(the old version projected them down to n with an extra Linear layer), the decoder's initial state
is the sum of the two directions' final states (the exercise's option 1), and the padding argument
is called padding_idx as in the exercise.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BidirectionalEncoder(nn.Module):
    """
    Bidirectional GRU encoder.

    Why bidirectional? The forward GRU summarises the words before position j, the backward GRU the
    words after it, so the annotation h_j describes the whole sentence with a focus around word j.
    That is what makes it a good thing for attention to point at.

    Args:
        vocab_size: source vocabulary size
        embed_size: embedding dimension (m in the paper)
        hidden_size: units per direction (n in the paper); annotations have 2 * hidden_size
        num_layers: stacked GRU layers (the paper uses 1)
        dropout: dropout probability
        padding_idx: padding token index
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1,
                 dropout=0.1, padding_idx=0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def forward(self, src, src_lengths):
        """
        Encode source sequence.

        Args:
            src: Source token IDs (batch, src_len)
            src_lengths: Actual lengths before padding (batch,)

        Returns:
            outputs: annotations h_j = [fwd_j; bwd_j] (batch, src_len, 2 * hidden_size); padded
                positions are exactly zero
            hidden: initial decoder state (num_layers, batch, hidden_size), see _combine_hidden()
        """
        batch_size, src_len = src.shape

        embedded = self.dropout(self.embedding(src))  # (batch, src_len, embed_size)

        # Sort by length for packing (required by pack_padded_sequence with enforce_sorted=True)
        src_lengths_cpu = src_lengths.cpu()
        sorted_lengths, sorted_indices = src_lengths_cpu.sort(descending=True)
        sorted_embedded = embedded[sorted_indices]

        # Packing makes each GRU direction stop at the true length: the backward GRU starts at the
        # last real token, not at padding
        packed = pack_padded_sequence(
            sorted_embedded,
            sorted_lengths.tolist(),
            batch_first=True,
            enforce_sorted=True
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True,
                                         total_length=src_len)

        # Unsort to restore original order
        _, unsort_indices = sorted_indices.sort()
        outputs = outputs[unsort_indices]
        # [PATCH 2026-10-18] The GRU already returns `hidden` in sorted order
        # (enforce_sorted=True), so only the unsort is needed. The original extra
        # `hidden[:, sorted_indices, :]` made the two permutations cancel, leaving
        # hidden in sorted order: each decoder was initialised from another
        # example's encoder state whenever the batch was not pre-sorted by length.
        hidden = hidden[:, unsort_indices, :]  # Unsort to original batch order

        return outputs, self._combine_hidden(hidden)

    def _combine_hidden(self, hidden):
        """
        Combine the two directions' final states into the decoder's initial state.

        hidden from nn.GRU is (num_layers * 2, batch, hidden_size), ordered layer by layer with the
        forward direction first. This sums the two directions (exercise_2's option 1, also what
        implementation.Encoder does). The paper instead uses s_0 = tanh(W_s backward h_1)
        (Appendix A.2.2): only the backward state, through a learned layer.
        """
        _, batch_size, hidden_size = hidden.shape
        # view: (num_layers * 2, B, H) -> (num_layers, 2, B, H); then sum over the direction axis
        hidden = hidden.view(self.num_layers, 2, batch_size, hidden_size)
        return hidden.sum(dim=1)                       # (num_layers, B, H)


# ============================================================================
# Testing
# ============================================================================

def test_bidirectional_encoder():
    """Comprehensive test of the BidirectionalEncoder."""
    print("Testing BidirectionalEncoder...")

    vocab_size = 100
    embed_size = 64
    hidden_size = 128
    batch_size = 3

    encoder = BidirectionalEncoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=2,
        dropout=0.1
    )

    # Create test input with varying lengths
    src = torch.tensor([
        [5, 3, 8, 2, 1, 0, 0],  # length 5
        [9, 7, 4, 0, 0, 0, 0],  # length 3
        [6, 2, 1, 8, 5, 3, 9],  # length 7
    ])
    src_lengths = torch.tensor([5, 3, 7])

    outputs, hidden = encoder(src, src_lengths)

    # Annotations are 2n wide: forward and backward states side by side
    assert outputs.shape == (batch_size, 7, 2 * hidden_size), \
        f"Wrong output shape: {outputs.shape}"
    assert hidden.shape == (2, batch_size, hidden_size), \
        f"Wrong hidden shape: {hidden.shape}"
    print("  Output shapes correct")

    # Padded positions are exactly zero (packing skips them)
    assert outputs[1, 3:].abs().max() == 0, "Padded positions should be zero"
    print("  Padded positions are zero")

    outputs.sum().backward()
    assert encoder.embedding.weight.grad is not None, "No gradient for embeddings!"
    print("  Gradients flow correctly")

    print("Test passed.")
    return encoder


if __name__ == '__main__':
    test_bidirectional_encoder()
