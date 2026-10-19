"""
Solution 3: Attention Decoder

Reference implementation of a GRU decoder with Bahdanau attention. Following the paper (Appendix
A.2.2), the decoder state s_{i-1} (n units) queries the 2n-dim encoder annotations h_j, and the
context c_i = sum_j alpha_ij h_j (2n numbers) feeds the GRU and the output layer.

[PATCH 2026-10-18] encoder_hidden_size is now the encoder's units per direction (n), and the
annotations are 2 * encoder_hidden_size wide, as exercise_3.py specifies; the old version expected
annotations already projected down to n, so the exercise's test failed with this class swapped in.
"""

import torch
import torch.nn as nn

# Import attention from solution_1
from solution_1 import AdditiveAttention


class AttentionDecoder(nn.Module):
    """
    GRU decoder with Bahdanau attention.

    At each step i:
    1. Score every annotation h_j against the previous state s_{i-1} and average them: c_i
    2. Feed [embedding of y_{i-1}; c_i] to the GRU to get s_i
    3. Predict y_i from [s_i; c_i; embedding of y_{i-1}]

    The attention uses the PREVIOUS state s_{i-1}, as in the paper (Sec. 3.1). The paper's GRU also
    takes c_i inside its gates and candidate (C, C_z, C_r in A.2.2); feeding [embedding; c_i] as the
    nn.GRU input gives the same input terms (nn.GRU applies its reset gate after the recurrent
    matmul, a small difference). The paper's output layer is a maxout layer on (s_{i-1}, y_{i-1},
    c_i); this uses one linear layer on (s_i, c_i, y_{i-1}).

    Args:
        vocab_size: target vocabulary size
        embed_size: embedding dimension (m)
        hidden_size: decoder GRU units (n)
        encoder_hidden_size: encoder units per direction; annotations have 2 * encoder_hidden_size
        num_layers: decoder GRU layers (the paper uses 1)
        dropout: dropout probability
        padding_idx: padding token index
    """

    def __init__(self, vocab_size, embed_size, hidden_size, encoder_hidden_size=None,
                 num_layers=1, dropout=0.1, padding_idx=0):
        super().__init__()

        if encoder_hidden_size is None:
            encoder_hidden_size = hidden_size

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_output_size = 2 * encoder_hidden_size     # annotation size, 2n

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)

        # Alignment model: query s_{i-1} (n), keys h_j (2n), hidden layer n' = n as in the paper
        self.attention = AdditiveAttention(hidden_size=hidden_size, query_size=hidden_size,
                                           key_size=self.encoder_output_size)

        # GRU input: [embedding; context]
        self.gru = nn.GRU(
            input_size=embed_size + self.encoder_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer: [hidden; context; embedding] -> vocabulary
        self.output_layer = nn.Linear(hidden_size + self.encoder_output_size + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward_step(self, prev_token, hidden, encoder_outputs, src_mask=None):
        """
        One decoding step.

        Args:
            prev_token: previous target token y_{i-1} (batch,)
            hidden: previous state s_{i-1} (num_layers, batch, hidden_size)
            encoder_outputs: annotations (batch, src_len, 2 * encoder_hidden_size)
            src_mask: True where a source position is padding (batch, src_len)

        Returns:
            output: vocabulary logits (batch, vocab_size)
            hidden: new state s_i (num_layers, batch, hidden_size)
            attention_weights: alpha_i (batch, src_len)
        """
        embedded = self.dropout(self.embedding(prev_token))       # (batch, embed_size)

        # Query with the top layer's previous state: (batch, hidden_size)
        context, attention_weights = self.attention(hidden[-1], encoder_outputs, mask=src_mask)

        # (batch, embed + 2n) -> (batch, 1, embed + 2n): one time step for nn.GRU
        gru_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(1)                                 # (batch, hidden_size)

        combined = torch.cat([output, context, embedded], dim=1)  # (batch, n + 2n + embed)
        return self.output_layer(combined), hidden, attention_weights

    def forward(self, trg, hidden, encoder_outputs, src_mask=None, teacher_forcing_ratio=1.0):
        """
        Full decoding pass.

        Args:
            trg: target sequence including SOS (batch, trg_len)
            hidden: initial decoder state from the encoder (num_layers, batch, hidden_size)
            encoder_outputs: annotations (batch, src_len, 2 * encoder_hidden_size)
            src_mask: source padding mask
            teacher_forcing_ratio: probability of feeding the ground-truth previous token

        Returns:
            outputs: logits for trg[:, 1:] (batch, trg_len-1, vocab_size)
            attentions: (batch, trg_len-1, src_len)
        """
        batch_size, trg_len = trg.shape
        src_len = encoder_outputs.size(1)
        outputs = torch.zeros(batch_size, trg_len - 1, self.vocab_size, device=trg.device)
        attentions = torch.zeros(batch_size, trg_len - 1, src_len, device=trg.device)

        input_token = trg[:, 0]                                     # SOS
        for t in range(trg_len - 1):
            output, hidden, attn = self.forward_step(input_token, hidden, encoder_outputs, src_mask)
            outputs[:, t] = output
            attentions[:, t] = attn
            # Teacher forcing: use ground truth or prediction?
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = trg[:, t + 1]
            else:
                input_token = output.argmax(dim=-1)

        return outputs, attentions


# ============================================================================
# Testing
# ============================================================================

def test_attention_decoder():
    """Comprehensive test of AttentionDecoder."""
    print("Testing AttentionDecoder...")

    vocab_size, embed_size, hidden_size, encoder_hidden_size = 100, 64, 128, 128
    batch_size, src_len, trg_len = 3, 7, 5

    decoder = AttentionDecoder(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        encoder_hidden_size=encoder_hidden_size,
        num_layers=2,
        dropout=0.1
    )

    trg = torch.randint(3, vocab_size, (batch_size, trg_len))
    trg[:, 0] = 1  # SOS token
    encoder_outputs = torch.randn(batch_size, src_len, 2 * encoder_hidden_size)   # annotations, 2n
    encoder_hidden = torch.randn(2, batch_size, hidden_size)

    outputs, attentions = decoder(trg, encoder_hidden, encoder_outputs)
    assert outputs.shape == (batch_size, trg_len - 1, vocab_size), \
        f"Wrong output shape: {outputs.shape}"
    assert attentions.shape == (batch_size, trg_len - 1, src_len), \
        f"Wrong attention shape: {attentions.shape}"
    print("  Output shapes correct")

    attn_sums = attentions.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), \
        "Attention weights don't sum to 1"
    print("  Attention weights sum to 1")

    input_token = torch.randint(3, vocab_size, (batch_size,))
    hidden = torch.randn(2, batch_size, hidden_size)
    output, new_hidden, attn = decoder.forward_step(input_token, hidden, encoder_outputs)
    assert output.shape == (batch_size, vocab_size), f"Wrong step output shape: {output.shape}"
    assert new_hidden.shape == hidden.shape, f"Wrong step hidden shape: {new_hidden.shape}"
    assert attn.shape == (batch_size, src_len), f"Wrong step attention shape: {attn.shape}"
    print("  Forward step shapes correct")

    outputs.sum().backward()
    assert decoder.embedding.weight.grad is not None, "No gradient for embeddings!"
    assert decoder.attention.key_proj.weight.grad is not None, "No gradient for attention!"
    print("  Gradients flow correctly")

    decoder.zero_grad()
    src_mask = torch.zeros(batch_size, src_len, dtype=torch.bool)
    src_mask[:, -2:] = True  # Mask last 2 positions
    _, attentions_masked = decoder(trg, encoder_hidden, encoder_outputs, src_mask)
    assert (attentions_masked[:, :, -2:] == 0).all(), "Masked positions have attention weight!"
    print("  Masking works correctly")

    print("Test passed.")
    return decoder


if __name__ == '__main__':
    test_attention_decoder()
