"""
Day 13: Attention Is All You Need - Implementation

A complete, from-scratch implementation of the Transformer architecture.
All core components with detailed explanations.

Paper: https://arxiv.org/abs/1706.03762

Components implemented:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Positional Encoding
4. Feed-Forward Network
5. Encoder Block
6. Decoder Block
7. Full Transformer

[PATCH 2026-10-18] Backward passes. Every component above now also has a
hand-written backward pass, so the model trains in pure NumPy (train_minimal.py):
  - forward() keeps what backward() will need in `self.cache` (inputs, softmax
    probabilities, dropout masks, LayerNorm statistics, ...);
  - backward(grad_output) takes dL/d(output) and returns dL/d(input) -- the
    chain rule applied one module at a time, in reverse order of the forward;
  - parameter gradients go to `self.grads`, a dict keyed like the parameter
    attributes (grads["W_Q"] has the shape of W_Q). backward() overwrites them:
    each module runs once per forward pass, so there is nothing to accumulate
    (PyTorch accumulates into .grad instead, hence its optimizer.zero_grad());
  - Transformer.named_params_and_grads() / parameters() / gradients() list every
    trainable array once, for the optimizer.
The gradients match torch.autograd (float64, every parameter) and central finite
differences: validation/day13_backprop_check.py.
"""

import numpy as np
from typing import Optional, Tuple, List


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (used in modern transformers)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
               eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization.
    
    Normalizes across the last dimension (features).
    
    Args:
        x: Input of shape (..., features)
        gamma: Scale parameter (features,)
        beta: Shift parameter (features,)
        eps: Small constant for numerical stability
        
    Returns:
        Normalized output of same shape
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


# [PATCH 2026-10-18] Helpers for the backward passes. Dropout used to be written
# inline in every forward(); as a function pair the forward returns its random
# mask and the backward reuses exactly that mask. (Same random numbers, drawn in
# the same order as the old inline code, so forward outputs are unchanged.)
def dropout_forward(x: np.ndarray, p: float, training: bool):
    """
    Inverted dropout (Sec. 5.4: P_drop = 0.1 on sub-layer outputs and embeddings).

    "Inverted" = the kept units are scaled by 1/(1-p) during training, so the
    expected activation is unchanged and evaluation needs no rescaling.

    Returns:
        (output, mask): mask is a bool array shaped like x (True = kept), or
        None when dropout is inactive (eval mode or p = 0).
    """
    if not training or p <= 0:
        return x, None
    mask = np.random.rand(*x.shape) > p     # uniform [0,1) draws > p -> True with probability 1-p
    return x * mask / (1 - p), mask         # bool * float -> float: dropped entries become 0.0


def dropout_backward(grad_output: np.ndarray, mask: Optional[np.ndarray], p: float) -> np.ndarray:
    """y = x * mask / (1-p) is linear in x, so dL/dx = dL/dy * mask / (1-p), with the SAME mask."""
    if mask is None:
        return grad_output
    return grad_output * mask / (1 - p)


def softmax_backward(probs: np.ndarray, grad_probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Backward pass of p = softmax(s) along `axis`: returns dL/ds given p and dL/dp.

    The softmax Jacobian is dp_i/ds_j = p_i * (delta_ij - p_j), so
        dL/ds_j = sum_i g_i * p_i * (delta_ij - p_j) = p_j * (g_j - sum_i g_i * p_i)
    where g = dL/dp. The sum is the probability-weighted average of the incoming
    gradient (one number per row): a logit's gradient is its probability times how
    much better its own g is than that average. This costs O(n) per row instead of
    building the n x n Jacobian.
    """
    # (g * p).sum(axis, keepdims=True): one number per row; keepdims=True keeps the
    # summed axis with length 1, so the subtraction broadcasts it back over the row
    return probs * (grad_probs - np.sum(grad_probs * probs, axis=axis, keepdims=True))


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding:
    """
    Sinusoidal Positional Encoding.
    
    Adds position information using sine and cosine functions of
    different frequencies. This allows the model to attend to
    relative positions, as PE(pos+k) can be represented as a
    linear function of PE(pos).
    
    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout_p: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length to precompute
            dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.training = True
        
        # Precompute positional encodings
        self.pe = self._create_encoding(max_len, d_model)
    
    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create the sinusoidal encoding matrix."""
        pe = np.zeros((max_len, d_model))
        
        # Position indices: [0, 1, 2, ..., max_len-1]
        position = np.arange(max_len)[:, np.newaxis]
        
        # Division term: 10000^(2i/d_model) using exp-log for stability
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch, seq_len, d_model)
            
        Returns:
            Embeddings + positional encoding
        """
        seq_len = x.shape[1]
        
        # Add positional encoding (broadcasts over batch)
        output = x + self.pe[:seq_len]
        
        # Apply dropout during training
        # [PATCH 2026-10-18] via dropout_forward, keeping the mask for backward().
        # The cache is a 1-tuple so that it is never None (see backward's `cache`).
        output, mask = dropout_forward(output, self.dropout_p, self.training)
        self.cache = (mask,)
        
        return output

    # [PATCH 2026-10-18] backward pass (the encoding is a fixed table: no parameters)
    def backward(self, grad_output: np.ndarray, cache: Optional[tuple] = None) -> np.ndarray:
        """
        Backward of output = dropout(x + PE[:seq_len]).

        PE is a constant, so the addition passes the gradient through unchanged;
        only the dropout mask acts on it.

        Args:
            grad_output: dL/d(output), (batch, seq_len, d_model)
            cache: the `self.cache` saved right after a particular forward() call
                (default: the most recent call). The Transformer runs this module
                twice per forward pass (source, then target), so it keeps both
                caches and passes each one back explicitly.
        Returns:
            dL/dx, (batch, seq_len, d_model)
        """
        (mask,) = self.cache if cache is None else cache
        return dropout_backward(grad_output, mask, self.dropout_p)
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# SCALED DOT-PRODUCT ATTENTION
# =============================================================================

class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention.
    
    The core attention mechanism:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Q (Query): What am I looking for?
    K (Key): What do I contain?
    V (Value): What do I give when matched?
    
    The scaling by sqrt(d_k) prevents the dot products from growing
    too large, which would push softmax into regions with tiny gradients.
    """
    
    def __init__(self, dropout_p: float = 0.1):
        self.dropout_p = dropout_p
        self.training = True
        self.attention_weights = None  # Cache for visualization
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Queries of shape (batch, heads, seq_q, d_k)
            K: Keys of shape (batch, heads, seq_k, d_k)
            V: Values of shape (batch, heads, seq_k, d_v)
            mask: Optional boolean mask, True = mask out (batch, 1, seq_q, seq_k)
            
        Returns:
            Attention output of shape (batch, heads, seq_q, d_v)
        """
        d_k = K.shape[-1]
        
        # Step 1: Compute attention scores
        # Q @ K^T -> (batch, heads, seq_q, seq_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        
        # Step 2: Scale by sqrt(d_k)
        scores = scores / np.sqrt(d_k)
        
        # Step 3: Apply mask (set masked positions to -inf)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        # Step 4: Softmax to get attention weights
        attention_weights = softmax(scores, axis=-1)
        
        # Cache for visualization
        self.attention_weights = attention_weights
        
        # Step 5: Apply dropout to attention weights
        # [PATCH 2026-10-18] via dropout_forward; keep everything backward() needs
        dropped_weights, dropout_mask = dropout_forward(attention_weights, self.dropout_p, self.training)
        self.cache = (Q, K, V, mask, attention_weights, dropped_weights, dropout_mask)
        
        # Step 6: Multiply by values
        # (batch, heads, seq_q, seq_k) @ (batch, heads, seq_k, d_v)
        # -> (batch, heads, seq_q, d_v)
        output = np.matmul(dropped_weights, V)
        
        return output

    # [PATCH 2026-10-18] backward pass
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward of output = dropout(softmax(mask(Q K^T / sqrt(d_k)))) V, in reverse order
        of forward()'s steps.

        Args:
            grad_output: dL/d(output), (batch, heads, seq_q, d_v)
        Returns:
            (dQ, dK, dV), shaped like Q, K, V
        """
        Q, K, V, mask, P, P_dropped, dropout_mask = self.cache
        d_k = K.shape[-1]
        swap = (0, 1, 3, 2)   # transpose(swap) swaps the last two axes (a batched matrix transpose)

        # Step 6: output = P_dropped @ V
        # dV = P_dropped^T @ dout: (B,h,Lk,Lq) @ (B,h,Lq,d_v) -> (B,h,Lk,d_v);
        # value j collects gradient from every query i in proportion to the weight P_ij
        dV = np.matmul(P_dropped.transpose(swap), grad_output)
        # dP_dropped = dout @ V^T: (B,h,Lq,d_v) @ (B,h,d_v,Lk) -> (B,h,Lq,Lk)
        dP_dropped = np.matmul(grad_output, V.transpose(swap))

        # Step 5: dropout on the weights -> reuse the forward mask
        dP = dropout_backward(dP_dropped, dropout_mask, self.dropout_p)

        # Step 4: softmax over keys (last axis), row by row; see softmax_backward
        d_scores = softmax_backward(P, dP, axis=-1)

        # Step 3: masked scores were REPLACED by the constant -1e9, so no gradient
        # reaches the original scores there. (Their probabilities are exactly 0, so
        # d_scores is already 0 there; np.where makes the rule explicit.) The mask
        # (B,1,1,Lk) or (1,1,Lq,Lk) broadcasts over heads/queries like in forward().
        if mask is not None:
            d_scores = np.where(mask, 0.0, d_scores)

        # Step 2: scores = raw / sqrt(d_k) -> the gradient is divided by the same constant
        d_raw = d_scores / np.sqrt(d_k)

        # Step 1: raw = Q @ K^T
        # dQ = d_raw @ K: (B,h,Lq,Lk) @ (B,h,Lk,d_k) -> (B,h,Lq,d_k)
        dQ = np.matmul(d_raw, K)
        # dK = d_raw^T @ Q: (B,h,Lk,Lq) @ (B,h,Lq,d_k) -> (B,h,Lk,d_k)
        dK = np.matmul(d_raw.transpose(swap), Q)
        return dQ, dK, dV
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# MULTI-HEAD ATTENTION
# =============================================================================

class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    Instead of computing one attention function, we project Q, K, V
    into multiple subspaces (heads) and compute attention in parallel.
    This allows the model to learn different types of relationships.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W^O
    where head_i = Attention(Q @ W^Q_i, K @ W^K_i, V @ W^V_i)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout_p: Dropout probability
        """
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.dropout_p = dropout_p
        self.training = True
        
        # Initialize projection weights
        # Each head gets d_k dimensions
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        # [PATCH 2026-10-18] parameter gradients, filled by backward()
        self.grads = {}
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout_p)
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute multi-head attention.
        
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional mask (batch, 1, seq_q, seq_k)
            
        Returns:
            Output of shape (batch, seq_q, d_model)
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]
        
        # Step 1: Linear projections
        Q = query @ self.W_Q  # (batch, seq_q, d_model)
        K = key @ self.W_K    # (batch, seq_k, d_model)
        V = value @ self.W_V  # (batch, seq_k, d_model)
        
        # Step 2: Reshape for multi-head attention
        # (batch, seq, d_model) -> (batch, seq, n_heads, d_k) -> (batch, n_heads, seq, d_k)
        Q = Q.reshape(batch_size, seq_q, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_k, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Step 3: Apply attention
        self.attention.training = self.training
        attn_output = self.attention.forward(Q, K, V, mask)
        # Shape: (batch, n_heads, seq_q, d_k)
        
        # Step 4: Concatenate heads
        # (batch, n_heads, seq_q, d_k) -> (batch, seq_q, n_heads, d_k) -> (batch, seq_q, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_q, self.d_model)
        
        # Step 5: Final linear projection
        output = attn_output @ self.W_O
        # [PATCH 2026-10-18] inputs and concatenated heads, for backward()
        self.cache = (query, key, value, attn_output)
        
        return output

    # [PATCH 2026-10-18] backward pass
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward of MultiHead(query, key, value) = Concat(head_1..head_h) W_O (Sec. 3.2.2).

        Args:
            grad_output: dL/d(output), (batch, seq_q, d_model)
        Returns:
            (d_query, d_key, d_value), shaped like the three inputs. In self-attention
            the caller passed the same x three times, so it must ADD all three; in
            cross-attention d_key + d_value flow to the encoder output.
        Fills self.grads["W_Q"], ["W_K"], ["W_V"], ["W_O"] (each (d_model, d_model)).
        """
        query, key, value, concat = self.cache
        B, Lq, D = query.shape
        Lk = key.shape[1]
        h, d_k = self.n_heads, self.d_k

        # Step 5: output = concat @ W_O. A weight is shared by every (batch, position)
        # row, so its gradient sums over all of them: flatten (B, Lq, D) -> (B*Lq, D) and
        # one 2-D matmul does the sum: (D, B*Lq) @ (B*Lq, D) -> (D, D)
        self.grads["W_O"] = concat.reshape(-1, D).T @ grad_output.reshape(-1, D)
        d_concat = grad_output @ self.W_O.T            # (B, Lq, D) @ (D, D) -> (B, Lq, D)

        # Step 4 undone: "concatenate heads" was transpose(0,2,1,3) then reshape, so
        # reshape back to (B, Lq, h, d_k), then transpose to (B, h, Lq, d_k)
        d_heads = d_concat.reshape(B, Lq, h, d_k).transpose(0, 2, 1, 3)

        # Step 3: attention, all heads at once
        dQh, dKh, dVh = self.attention.backward(d_heads)   # each (B, h, L, d_k)

        # Step 2 undone: "split heads" was reshape then transpose(0,2,1,3); that transpose
        # is its own inverse, so transpose back to (B, L, h, d_k) and merge (h, d_k) -> D
        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, Lq, D)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, Lk, D)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, Lk, D)

        # Step 1: Q = query @ W_Q (likewise K, V): weight gradient = input^T @ output-gradient,
        # summed over batch and positions; input gradient = output-gradient @ W^T
        self.grads["W_Q"] = query.reshape(-1, D).T @ dQ.reshape(-1, D)
        self.grads["W_K"] = key.reshape(-1, D).T @ dK.reshape(-1, D)
        self.grads["W_V"] = value.reshape(-1, D).T @ dV.reshape(-1, D)
        return dQ @ self.W_Q.T, dK @ self.W_K.T, dV @ self.W_V.T

    # [PATCH 2026-10-18] parameter listing for the optimizer
    def named_params_and_grads(self, prefix: str = "") -> List[tuple]:
        """[(name, parameter, gradient or None before backward)] for W_Q, W_K, W_V, W_O."""
        return [(prefix + n, getattr(self, n), self.grads.get(n)) for n in ("W_Q", "W_K", "W_V", "W_O")]
    
    def get_attention_weights(self) -> np.ndarray:
        """Get the last computed attention weights."""
        return self.attention.attention_weights
    
    def train(self):
        self.training = True
        self.attention.training = True
    
    def eval(self):
        self.training = False
        self.attention.training = False


# =============================================================================
# FEED-FORWARD NETWORK
# =============================================================================

class FeedForward:
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    
    Applied independently to each position.
    Typically d_ff = 4 * d_model.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension (usually 4 * d_model)
            dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.training = True
        
        # Initialize weights
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)
        # [PATCH 2026-10-18] parameter gradients, filled by backward()
        self.grads = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input of shape (..., d_model)
            
        Returns:
            Output of shape (..., d_model)
        """
        # First linear + ReLU
        # [PATCH 2026-10-18] keep the pre-activation z1: ReLU's backward needs its sign
        z1 = x @ self.W1 + self.b1
        hidden = relu(z1)
        
        # Dropout
        hidden, mask = dropout_forward(hidden, self.dropout_p, self.training)
        
        # Second linear
        output = hidden @ self.W2 + self.b2
        self.cache = (x, z1, hidden, mask)
        
        return output

    # [PATCH 2026-10-18] backward pass
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward of FFN(x) = dropout(ReLU(x W1 + b1)) W2 + b2 (Eq. 2).

        Args:
            grad_output: dL/d(output), (..., d_model)
        Returns:
            dL/dx, (..., d_model). Fills self.grads["W1"], ["b1"], ["W2"], ["b2"].
        """
        x, z1, hidden, mask = self.cache
        D, F = self.d_model, self.d_ff
        g = grad_output.reshape(-1, D)          # (..., D) -> (N, D): every position is one row

        # output = hidden @ W2 + b2 (the same W2, b2 at every position -> sum over rows)
        self.grads["W2"] = hidden.reshape(-1, F).T @ g     # (F, N) @ (N, D) -> (F, D)
        self.grads["b2"] = g.sum(axis=0)                   # (N, D) -> (D,)
        d_hidden = grad_output @ self.W2.T                 # (..., D) @ (D, F) -> (..., F)

        # dropout (same mask), then ReLU: gradient passes only where z1 > 0
        d_hidden = dropout_backward(d_hidden, mask, self.dropout_p)
        d_z1 = d_hidden * (z1 > 0)                         # bool mask * float -> float

        # z1 = x @ W1 + b1
        self.grads["W1"] = x.reshape(-1, D).T @ d_z1.reshape(-1, F)   # (D, N) @ (N, F) -> (D, F)
        self.grads["b1"] = d_z1.reshape(-1, F).sum(axis=0)            # (F,)
        return d_z1 @ self.W1.T                                       # (..., F) @ (F, D) -> (..., D)

    # [PATCH 2026-10-18] parameter listing for the optimizer
    def named_params_and_grads(self, prefix: str = "") -> List[tuple]:
        """[(name, parameter, gradient or None before backward)] for W1, b1, W2, b2."""
        return [(prefix + n, getattr(self, n), self.grads.get(n)) for n in ("W1", "b1", "W2", "b2")]
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


# =============================================================================
# LAYER NORMALIZATION MODULE
# =============================================================================

class LayerNorm:
    """Layer Normalization module with learnable parameters."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.d_model = d_model
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        # [PATCH 2026-10-18] parameter gradients, filled by backward()
        self.grads = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        # [PATCH 2026-10-18] Same arithmetic as layer_norm() above, written out so the
        # normalised input and the per-token std can be kept for backward().
        mean = np.mean(x, axis=-1, keepdims=True)      # (..., 1): one mean per token
        var = np.var(x, axis=-1, keepdims=True)        # (..., 1): biased variance, like the paper/PyTorch
        std = np.sqrt(var + self.eps)
        x_hat = (x - mean) / std
        self.cache = (x_hat, std)
        return self.gamma * x_hat + self.beta

    # [PATCH 2026-10-18] backward pass
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward of y = gamma * x_hat + beta, x_hat = (x - mean(x)) / sqrt(var(x) + eps),
        statistics over the last (feature) axis of each token.

        With g = dL/dx_hat = dL/dy * gamma, the result is
            dL/dx = (g - mean(g) - x_hat * mean(g * x_hat)) / std
        Intuition: LayerNorm's output does not move if you shift x by a constant
        and (up to eps) if you rescale x, so dL/dx has no component along those two
        directions: subtracting mean(g) removes the "shift" part (all-ones direction),
        subtracting x_hat * mean(g * x_hat) removes the "rescale" part (x_hat direction).

        Returns dL/dx; fills self.grads["gamma"], ["beta"] (each (d_model,)).
        """
        x_hat, std = self.cache
        D = self.d_model
        # gamma and beta are shared by every token: sum over all leading axes
        # (reshape(-1, D) flattens (batch, seq) into one axis of rows)
        self.grads["gamma"] = (grad_output * x_hat).reshape(-1, D).sum(axis=0)
        self.grads["beta"] = grad_output.reshape(-1, D).sum(axis=0)
        g = grad_output * self.gamma                   # (..., D) * (D,) broadcasts over tokens
        # mean(..., axis=-1, keepdims=True): one number per token, kept as (..., 1) to broadcast
        return (g - g.mean(axis=-1, keepdims=True)
                - x_hat * (g * x_hat).mean(axis=-1, keepdims=True)) / std

    # [PATCH 2026-10-18] parameter listing for the optimizer
    def named_params_and_grads(self, prefix: str = "") -> List[tuple]:
        """[(name, parameter, gradient or None before backward)] for gamma, beta."""
        return [(prefix + n, getattr(self, n), self.grads.get(n)) for n in ("gamma", "beta")]


# =============================================================================
# ENCODER BLOCK
# =============================================================================

class EncoderBlock:
    """
    Transformer Encoder Block.
    
    Structure:
        x -> Self-Attention -> Add & Norm -> FFN -> Add & Norm -> output
        
    Uses pre-norm in modern implementations (norm before attention).
    Original paper uses post-norm (norm after residual).
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.training = True
        
        # Components
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through encoder block.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            mask: Optional padding mask
            
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.self_attention.forward(x, x, x, mask)
        
        # Dropout on attention output
        # [PATCH 2026-10-18] via dropout_forward; both residual-branch masks are kept for backward()
        attn_output, drop_attn = dropout_forward(attn_output, self.dropout_p, self.training)
        
        x = self.norm1.forward(x + attn_output)  # Add & Norm
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward.forward(x)
        
        # Dropout on FFN output
        ff_output, drop_ff = dropout_forward(ff_output, self.dropout_p, self.training)
        
        x = self.norm2.forward(x + ff_output)  # Add & Norm
        self.cache = (drop_attn, drop_ff)
        
        return x

    # [PATCH 2026-10-18] backward pass
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward through the post-norm block (Sec. 3.1):
            x1 = LayerNorm(x  + Dropout(SelfAttn(x, x, x)))
            y  = LayerNorm(x1 + Dropout(FFN(x1)))

        A residual add z = a + b copies dL/dz to BOTH a and b unchanged, so each
        sub-layer's input gradient is "skip path + branch path".

        Args:
            grad_output: dL/dy, (batch, seq_len, d_model)
        Returns:
            dL/dx, (batch, seq_len, d_model)
        """
        drop_attn, drop_ff = self.cache
        p = self.dropout_p

        # y = norm2(x1 + dropout(ffn(x1)))
        d_sum2 = self.norm2.backward(grad_output)
        d_x1 = d_sum2 + self.feed_forward.backward(dropout_backward(d_sum2, drop_ff, p))

        # x1 = norm1(x + dropout(attn(x, x, x)))
        d_sum1 = self.norm1.backward(d_x1)
        d_q, d_k, d_v = self.self_attention.backward(dropout_backward(d_sum1, drop_attn, p))
        # x was used four times: skip path, and as query, key and value
        return d_sum1 + d_q + d_k + d_v

    # [PATCH 2026-10-18] parameter listing for the optimizer
    def named_params_and_grads(self, prefix: str = "") -> List[tuple]:
        """(name, parameter, gradient) for every parameter in this block."""
        return (self.self_attention.named_params_and_grads(prefix + "self_attention.")
                + self.feed_forward.named_params_and_grads(prefix + "feed_forward.")
                + self.norm1.named_params_and_grads(prefix + "norm1.")
                + self.norm2.named_params_and_grads(prefix + "norm2."))
    
    def train(self):
        self.training = True
        self.self_attention.train()
        self.feed_forward.train()
    
    def eval(self):
        self.training = False
        self.self_attention.eval()
        self.feed_forward.eval()


# =============================================================================
# DECODER BLOCK
# =============================================================================

class DecoderBlock:
    """
    Transformer Decoder Block.
    
    Structure:
        x -> Masked Self-Attention -> Add & Norm 
          -> Cross-Attention -> Add & Norm 
          -> FFN -> Add & Norm -> output
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_p: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout_p: Dropout probability
        """
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.training = True
        
        # Components
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout_p)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_p)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_mask: Optional[np.ndarray] = None,
                cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through decoder block.
        
        Args:
            x: Decoder input (batch, seq_tgt, d_model)
            encoder_output: Encoder output (batch, seq_src, d_model)
            self_mask: Causal mask for self-attention
            cross_mask: Padding mask for cross-attention
            
        Returns:
            Output of shape (batch, seq_tgt, d_model)
        """
        # [PATCH 2026-10-18] dropout via dropout_forward; the three masks are kept for backward()
        # Masked self-attention
        self_attn = self.self_attention.forward(x, x, x, self_mask)
        self_attn, drop_self = dropout_forward(self_attn, self.dropout_p, self.training)
        x = self.norm1.forward(x + self_attn)
        
        # Cross-attention (attend to encoder output)
        cross_attn = self.cross_attention.forward(x, encoder_output, encoder_output, cross_mask)
        cross_attn, drop_cross = dropout_forward(cross_attn, self.dropout_p, self.training)
        x = self.norm2.forward(x + cross_attn)
        
        # Feed-forward
        ff_output = self.feed_forward.forward(x)
        ff_output, drop_ff = dropout_forward(ff_output, self.dropout_p, self.training)
        x = self.norm3.forward(x + ff_output)
        self.cache = (drop_self, drop_cross, drop_ff)
        
        return x

    # [PATCH 2026-10-18] backward pass
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward through the post-norm decoder block (Sec. 3.1, 3.2.3):
            x1 = LayerNorm(x  + Dropout(SelfAttn(x, x, x, causal mask)))
            x2 = LayerNorm(x1 + Dropout(CrossAttn(x1, enc, enc)))   # Q from decoder, K/V from encoder
            y  = LayerNorm(x2 + Dropout(FFN(x2)))

        Args:
            grad_output: dL/dy, (batch, seq_tgt, d_model)
        Returns:
            (dL/dx, dL/d encoder_output): (batch, seq_tgt, d_model) and
            (batch, seq_src, d_model). The second one is only THIS layer's share:
            every decoder layer reads the same encoder output, so the caller sums it
            over all decoder layers.
        """
        drop_self, drop_cross, drop_ff = self.cache
        p = self.dropout_p

        # y = norm3(x2 + dropout(ffn(x2)))
        d_sum3 = self.norm3.backward(grad_output)
        d_x2 = d_sum3 + self.feed_forward.backward(dropout_backward(d_sum3, drop_ff, p))

        # x2 = norm2(x1 + dropout(cross_attn(x1, enc, enc)))
        d_sum2 = self.norm2.backward(d_x2)
        d_q, d_k, d_v = self.cross_attention.backward(dropout_backward(d_sum2, drop_cross, p))
        d_x1 = d_sum2 + d_q              # the decoder state was only the query
        d_encoder_output = d_k + d_v     # the encoder output was both key and value

        # x1 = norm1(x + dropout(self_attn(x, x, x)))
        d_sum1 = self.norm1.backward(d_x1)
        d_q, d_k, d_v = self.self_attention.backward(dropout_backward(d_sum1, drop_self, p))
        return d_sum1 + d_q + d_k + d_v, d_encoder_output

    # [PATCH 2026-10-18] parameter listing for the optimizer
    def named_params_and_grads(self, prefix: str = "") -> List[tuple]:
        """(name, parameter, gradient) for every parameter in this block."""
        return (self.self_attention.named_params_and_grads(prefix + "self_attention.")
                + self.cross_attention.named_params_and_grads(prefix + "cross_attention.")
                + self.feed_forward.named_params_and_grads(prefix + "feed_forward.")
                + self.norm1.named_params_and_grads(prefix + "norm1.")
                + self.norm2.named_params_and_grads(prefix + "norm2.")
                + self.norm3.named_params_and_grads(prefix + "norm3."))
    
    def train(self):
        self.training = True
        self.self_attention.train()
        self.cross_attention.train()
        self.feed_forward.train()
    
    def eval(self):
        self.training = False
        self.self_attention.eval()
        self.cross_attention.eval()
        self.feed_forward.eval()


# =============================================================================
# TOKEN EMBEDDING
# =============================================================================

class TokenEmbedding:
    """Token embedding layer."""
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Initialize embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        # [PATCH 2026-10-18] parameter gradients, filled by backward()
        self.grads = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Look up embeddings.
        
        Args:
            x: Token indices (batch, seq_len)
            
        Returns:
            Embeddings (batch, seq_len, d_model)
        """
        self.cache = x  # [PATCH 2026-10-18] token ids, for backward()
        return self.embedding[x]

    # [PATCH 2026-10-18] backward pass
    def backward(self, grad_output: np.ndarray) -> None:
        """
        A lookup copies row tokens[b, t] of the table to output[b, t], so the
        table's gradient is a scatter-add: row v receives the sum of grad_output
        over every position whose token is v (rows of unused tokens stay 0).

        Args:
            grad_output: dL/d(output), (batch, seq_len, d_model)
        Fills self.grads["embedding"], (vocab_size, d_model). Token ids are
        integers, so there is no gradient w.r.t. the input to return.
        """
        tokens = self.cache
        d_table = np.zeros_like(self.embedding)
        # np.add.at(d_table, tokens, grad_output): d_table[tokens[b, t]] += grad_output[b, t]
        # for every (b, t). It is "unbuffered": a token id that appears several times gets
        # ALL its updates. The look-alike `d_table[tokens] += grad_output` would keep only
        # one of them per repeated id, silently losing gradient.
        np.add.at(d_table, tokens, grad_output)
        self.grads["embedding"] = d_table

    # [PATCH 2026-10-18] parameter listing for the optimizer
    def named_params_and_grads(self, prefix: str = "") -> List[tuple]:
        """[(name, parameter, gradient or None before backward)] for the embedding table."""
        return [(prefix + "embedding", self.embedding, self.grads.get("embedding"))]


# =============================================================================
# FULL TRANSFORMER
# =============================================================================

class Transformer:
    """
    Complete Transformer Model (Encoder-Decoder).
    
    For sequence-to-sequence tasks like translation.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout_p: float = 0.1,
        max_len: int = 5000,
        share_embeddings: bool = False
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder blocks
            n_decoder_layers: Number of decoder blocks
            d_ff: Feed-forward hidden dimension
            dropout_p: Dropout probability
            max_len: Maximum sequence length
            share_embeddings: [PATCH 2026-10-18] Sec. 3.4 weight tying: one matrix
                serves as source embedding, target embedding and (transposed) as the
                pre-softmax projection. Needs src_vocab_size == tgt_vocab_size (the
                paper used a shared source-target BPE vocabulary). Default False keeps
                the original three separate matrices.
        """
        self.d_model = d_model
        self.training = True
        
        # Embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout_p)
        
        # Encoder stack
        self.encoder_layers = [
            EncoderBlock(d_model, n_heads, d_ff, dropout_p)
            for _ in range(n_encoder_layers)
        ]
        
        # Decoder stack
        self.decoder_layers = [
            DecoderBlock(d_model, n_heads, d_ff, dropout_p)
            for _ in range(n_decoder_layers)
        ]
        
        # Output projection
        self.output_projection = np.random.randn(d_model, tgt_vocab_size) * 0.02
    
        # [PATCH 2026-10-18] Optional weight tying (Sec. 3.4) and backward-pass state.
        self.share_embeddings = share_embeddings
        if share_embeddings:
            assert src_vocab_size == tgt_vocab_size, "weight tying needs one shared vocabulary"
            # std d_model^-0.5 (our choice; the paper gives no initialisation): the scaled
            # embedding E * sqrt(d_model) then has unit-variance entries, the scale of the
            # positional encodings, and the tied logits h . e_v start with std ~1.
            shared = np.random.randn(src_vocab_size, d_model) * d_model ** -0.5
            self.src_embedding.embedding = shared
            self.tgt_embedding.embedding = shared      # the SAME array object, not a copy
            self.output_projection = None              # logits use shared.T, see _output_weights()
        self.grads = {}   # gradient of the output projection (or of the shared matrix), set by backward()
        self.cache = {}   # what backward() needs from encode()/decode()/forward()

    # [PATCH 2026-10-18]
    def _output_weights(self) -> np.ndarray:
        """The (d_model, tgt_vocab) pre-softmax matrix: E^T when weights are tied (Sec. 3.4)."""
        return self.src_embedding.embedding.T if self.share_embeddings else self.output_projection

    def encode(self, src: np.ndarray, src_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode source sequence.
        
        Args:
            src: Source tokens (batch, src_len)
            src_mask: Optional padding mask
            
        Returns:
            Encoder output (batch, src_len, d_model)
        """
        # Embed and add positional encoding
        x = self.src_embedding.forward(src) * np.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)
        # [PATCH 2026-10-18] decode() reuses the positional-encoding module, which would
        # overwrite its dropout cache: keep this call's cache for backward()
        self.cache["src_pe"] = self.positional_encoding.cache
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer.forward(x, src_mask)
        
        return x
    
    def decode(self, tgt: np.ndarray, encoder_output: np.ndarray,
               tgt_mask: Optional[np.ndarray] = None,
               cross_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decode target sequence given encoder output.
        
        Args:
            tgt: Target tokens (batch, tgt_len)
            encoder_output: From encoder (batch, src_len, d_model)
            tgt_mask: Causal + padding mask for target
            cross_mask: Padding mask for encoder output
            
        Returns:
            Decoder output (batch, tgt_len, d_model)
        """
        # Embed and add positional encoding
        x = self.tgt_embedding.forward(tgt) * np.sqrt(self.d_model)
        x = self.positional_encoding.forward(x)
        self.cache["tgt_pe"] = self.positional_encoding.cache  # [PATCH 2026-10-18] see encode()
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer.forward(x, encoder_output, tgt_mask, cross_mask)
        
        return x
    
    def forward(self, src: np.ndarray, tgt: np.ndarray,
                src_mask: Optional[np.ndarray] = None,
                tgt_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Full forward pass.
        
        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target causal mask
            
        Returns:
            Logits (batch, tgt_len, tgt_vocab_size)
        """
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Project to vocabulary
        # [PATCH 2026-10-18] via _output_weights() (E^T when tied); keep outputs for backward()
        logits = decoder_output @ self._output_weights()
        self.cache["encoder_output"] = encoder_output
        self.cache["decoder_output"] = decoder_output
        
        return logits

    # [PATCH 2026-10-18] backward pass through the whole model
    def backward(self, grad_logits: np.ndarray) -> None:
        """
        Backpropagate dL/dlogits from the last forward() call through every layer.

        Order (reverse of forward): output projection -> decoder layers (last first)
        -> target embedding; then encoder layers (last first) -> source embedding.
        The encoder output was read by the cross-attention of EVERY decoder layer,
        so its gradient is the sum of those contributions; it is complete only
        after the whole decoder stack is done, which is why the encoder goes second.

        Args:
            grad_logits: dL/dlogits, (batch, tgt_len, tgt_vocab_size), e.g. from
                train_minimal.cross_entropy_loss_and_grad.
        Afterwards every module's .grads is filled; collect them with gradients().
        """
        D = self.d_model
        decoder_output = self.cache["decoder_output"]
        V = grad_logits.shape[-1]

        # logits = decoder_output @ W_out, W_out (D, V) shared by all (batch, position) rows:
        # flatten them, (D, B*T) @ (B*T, V) -> (D, V)
        d_W_out = decoder_output.reshape(-1, D).T @ grad_logits.reshape(-1, V)
        d_dec = grad_logits @ self._output_weights().T          # (B, T, V) @ (V, D) -> (B, T, D)

        # Decoder stack, last layer first; sum the encoder-output gradients
        d_enc = np.zeros_like(self.cache["encoder_output"])     # (B, S, D)
        for layer in reversed(self.decoder_layers):
            d_dec, d_enc_from_layer = layer.backward(d_dec)
            d_enc += d_enc_from_layer

        # x = dropout(E_tgt[tgt] * sqrt(d_model) + PE): through dropout/PE, then the constant scale
        d_tgt_emb = self.positional_encoding.backward(d_dec, self.cache["tgt_pe"]) * np.sqrt(D)
        self.tgt_embedding.backward(d_tgt_emb)

        # Encoder stack, last layer first, starting from the summed gradient
        d_x = d_enc
        for layer in reversed(self.encoder_layers):
            d_x = layer.backward(d_x)
        d_src_emb = self.positional_encoding.backward(d_x, self.cache["src_pe"]) * np.sqrt(D)
        self.src_embedding.backward(d_src_emb)

        if self.share_embeddings:
            # One matrix used in three places -> its gradient is the sum of the three uses.
            # The projection used E^T, so its (D, V) gradient is transposed back to (V, D).
            self.grads = {"shared_embedding": self.src_embedding.grads["embedding"]
                          + self.tgt_embedding.grads["embedding"] + d_W_out.T}
        else:
            self.grads = {"output_projection": d_W_out}

    # [PATCH 2026-10-18] parameter / gradient collection for the optimizer
    def named_params_and_grads(self) -> List[tuple]:
        """
        Every trainable array exactly once, as (name, parameter, gradient) in a fixed
        order; gradient is None until backward() has run. With share_embeddings the
        tied matrix appears once, as "shared_embedding", with the summed gradient.
        """
        if self.share_embeddings:
            items = [("shared_embedding", self.src_embedding.embedding, self.grads.get("shared_embedding"))]
        else:
            items = (self.src_embedding.named_params_and_grads("src_embedding.")
                     + self.tgt_embedding.named_params_and_grads("tgt_embedding.")
                     + [("output_projection", self.output_projection, self.grads.get("output_projection"))])
        for i, layer in enumerate(self.encoder_layers):
            items += layer.named_params_and_grads(f"encoder_layers.{i}.")
        for i, layer in enumerate(self.decoder_layers):
            items += layer.named_params_and_grads(f"decoder_layers.{i}.")
        return items

    def parameters(self) -> List[np.ndarray]:
        """The trainable arrays (the model's own objects: update them IN PLACE)."""
        return [param for _, param, _ in self.named_params_and_grads()]

    def gradients(self) -> List[np.ndarray]:
        """Gradients from the last backward(), in the same order as parameters()."""
        grads = [grad for _, _, grad in self.named_params_and_grads()]
        if any(g is None for g in grads):
            raise RuntimeError("no gradients yet: call forward() and backward() first")
        return grads
    
    def train(self):
        self.training = True
        self.positional_encoding.train()
        for layer in self.encoder_layers:
            layer.train()
        for layer in self.decoder_layers:
            layer.train()
    
    def eval(self):
        self.training = False
        self.positional_encoding.eval()
        for layer in self.encoder_layers:
            layer.eval()
        for layer in self.decoder_layers:
            layer.eval()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask for decoder self-attention.
    Prevents attending to future positions.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Boolean mask (1, 1, seq_len, seq_len), True = masked
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask[np.newaxis, np.newaxis, :, :]  # Add batch and head dims


def create_padding_mask(seq: np.ndarray, pad_token: int = 0) -> np.ndarray:
    """
    Create padding mask.
    
    Args:
        seq: Token sequence (batch, seq_len)
        pad_token: Padding token ID
        
    Returns:
        Boolean mask (batch, 1, 1, seq_len), True = masked
    """
    mask = (seq == pad_token)
    return mask[:, np.newaxis, np.newaxis, :]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSFORMER IMPLEMENTATION DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # ---------------------------------
    # 1. Scaled Dot-Product Attention
    # ---------------------------------
    print("\n1. Scaled Dot-Product Attention:")
    print("-" * 40)
    
    batch, heads, seq_len, d_k = 2, 4, 5, 64
    Q = np.random.randn(batch, heads, seq_len, d_k)
    K = np.random.randn(batch, heads, seq_len, d_k)
    V = np.random.randn(batch, heads, seq_len, d_k)
    
    attention = ScaledDotProductAttention(dropout_p=0.0)
    attention.eval()
    output = attention.forward(Q, K, V)
    
    print(f"  Q, K, V shapes: {Q.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attention.attention_weights.shape}")
    print(f"  Attention weights sum (per query): {attention.attention_weights[0, 0].sum(axis=-1)}")
    
    # ---------------------------------
    # 2. Multi-Head Attention
    # ---------------------------------
    print("\n2. Multi-Head Attention:")
    print("-" * 40)
    
    batch, seq_len, d_model, n_heads = 2, 10, 128, 8
    x = np.random.randn(batch, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model, n_heads, dropout_p=0.0)
    mha.eval()
    output = mha.forward(x, x, x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of heads: {n_heads}")
    print(f"  Dimension per head: {d_model // n_heads}")
    
    # ---------------------------------
    # 3. Positional Encoding
    # ---------------------------------
    print("\n3. Positional Encoding:")
    print("-" * 40)
    
    pe = PositionalEncoding(d_model=128, max_len=100)
    embeddings = np.random.randn(2, 20, 128)
    encoded = pe.forward(embeddings)
    
    print(f"  Input shape: {embeddings.shape}")
    print(f"  Output shape: {encoded.shape}")
    print(f"  PE matrix shape: {pe.pe.shape}")
    print(f"  Sample PE values (pos 0): {pe.pe[0, :4]}")
    print(f"  Sample PE values (pos 1): {pe.pe[1, :4]}")
    
    # ---------------------------------
    # 4. Encoder Block
    # ---------------------------------
    print("\n4. Encoder Block:")
    print("-" * 40)
    
    d_model, n_heads, d_ff = 128, 8, 512
    encoder = EncoderBlock(d_model, n_heads, d_ff, dropout_p=0.0)
    encoder.eval()
    
    x = np.random.randn(2, 10, d_model)
    output = encoder.forward(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  d_model: {d_model}, n_heads: {n_heads}, d_ff: {d_ff}")
    
    # ---------------------------------
    # 5. Full Transformer
    # ---------------------------------
    print("\n5. Full Transformer:")
    print("-" * 40)
    
    transformer = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        dropout_p=0.0
    )
    transformer.eval()
    
    src = np.random.randint(0, 1000, (2, 8))  # Source sequence
    tgt = np.random.randint(0, 1000, (2, 6))  # Target sequence
    
    # Create causal mask for decoder
    tgt_mask = create_causal_mask(tgt.shape[1])
    
    logits = transformer.forward(src, tgt, tgt_mask=tgt_mask)
    
    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Predicted tokens: {logits.argmax(axis=-1)[0]}")
    
    # ---------------------------------
    # 6. Attention Visualization
    # ---------------------------------
    print("\n6. Attention Weights Sample:")
    print("-" * 40)
    
    weights = transformer.decoder_layers[0].self_attention.get_attention_weights()
    if weights is not None:
        print(f"  Attention weights shape: {weights.shape}")
        print(f"  (batch, heads, query_pos, key_pos)")
        print(f"  Sample attention pattern (head 0, first 4x4):")
        sample = weights[0, 0, :4, :4]
        for row in sample:
            print(f"    [{', '.join(f'{v:.2f}' for v in row)}]")
    
    print("\n" + "=" * 60)
    print("Demo complete! Explore the components to learn more.")
    print("=" * 60)
