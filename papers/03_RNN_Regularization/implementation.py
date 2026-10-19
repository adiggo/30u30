"""
RNN Regularization Techniques in NumPy
======================================

A complete implementation of regularization strategies:
- Dropout
- Layer Normalization
- Weight Decay
- Early Stopping
- Validation Monitoring

This builds on Day 2's LSTM but adds defensive programming.

[PATCH 2026-10-18] Two LSTM classes live here:
- RegularizedLSTM: the original single-layer teaching model (batch size 1, one-hot inputs,
  column vectors, optional layer norm). FORWARD ONLY: it has no backward pass and nothing
  trains it; it is kept as a compact, one-sequence walkthrough of where dropout goes
  (validation/day03_dropout_placement.py checks it against the paper's equations).
- LSTMLanguageModel (bottom of the file): the model the paper trains (Zaremba et al. 2014,
  Sec. 3-4.1): word embeddings -> 2 LSTM layers -> softmax, dropout on every non-recurrent
  connection, truncated backpropagation through time, gradient-norm clipping and plain SGD.
  train_minimal.py trains it on Penn Treebank; validation/day03_bptt_check.py checks its
  gradients against torch.autograd.

Author: 30u30 Project
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)


def dropout_forward(x: np.ndarray, keep_prob: float, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dropout forward pass.
    
    During training: Randomly set activations to 0 with probability (1-keep_prob)
    During testing: Don't apply dropout (use full network)
    
    Args:
        x: Activations, shape (neurons,)
        keep_prob: Probability to keep each neuron (e.g., 0.8)
        training: Whether in training mode
        
    Returns:
        x_out: Dropped out activations
        mask: Binary mask used (for backward pass)
    """
    if not training or keep_prob == 1.0:
        return x, None
    
    # Create mask: 1 with probability keep_prob, 0 otherwise
    mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
    
    return x * mask, mask


def dropout_backward(dh: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """
    Dropout backward pass.
    
    Args:
        dh: Gradient of loss w.r.t. output
        mask: Mask from forward pass
        
    Returns:
        dh_out: Gradient w.r.t. input
    """
    if mask is None:
        return dh
    
    return dh * mask


def layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                       eps: float = 1e-5) -> Tuple[np.ndarray, Dict]:
    """
    Layer Normalization forward pass.
    
    Normalize each sample independently:
    1. Compute mean and variance across features
    2. Normalize to zero mean, unit variance
    3. Scale by gamma, shift by beta
    
    Args:
        x: Input, shape (batch_size, features) or (features,)
        gamma: Scale parameter, shape (features,)
        beta: Shift parameter, shape (features,)
        eps: Small constant for numerical stability

    Returns:
        x_norm: Normalized output
        cache: Data for backward pass
    """
    # [PATCH 2026-10-18] normalize over the LAST axis (the features). axis=1 raised an
    # AxisError on a 1-D input (exercise 2's first test); axis=-1 works for (features,)
    # and (batch, features) alike, and is what the exercise and its solution use.
    mean = np.mean(x, axis=-1, keepdims=True)  # (batch, 1) or (1,): keepdims keeps it broadcastable
    var = np.var(x, axis=-1, keepdims=True)    # same shape as mean
    
    # Normalize
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    # Scale and shift
    out = gamma * x_hat + beta
    
    cache = (x, x_hat, mean, var, gamma, beta, eps)
    return out, cache


def layer_norm_backward(dout: np.ndarray, cache: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer Normalization backward pass.
    
    Args:
        dout: Gradient of loss w.r.t. output
        cache: Cache from forward pass
        
    Returns:
        dx: Gradient w.r.t. input
        dgamma: Gradient w.r.t. gamma
        dbeta: Gradient w.r.t. beta
    """
    x, x_hat, mean, var, gamma, beta, eps = cache

    # [PATCH 2026-10-18] features on the last axis, any number of leading (batch) axes,
    # including none (a 1-D input), to match layer_norm_forward.
    D = x.shape[-1]
    lead = tuple(range(x.ndim - 1))   # batch axes to sum over; () for a 1-D input (sum over nothing)

    # Gradient w.r.t. scale and shift (summed over the batch, one value per feature)
    dgamma = np.sum(dout * x_hat, axis=lead)
    dbeta = np.sum(dout, axis=lead)

    # Gradient w.r.t. normalized input
    dx_hat = dout * gamma

    # Gradient w.r.t. variance and mean (statistics are per sample: sum over the feature axis)
    dx_dvar = dx_hat * (x - mean) * -0.5 * (var + eps) ** -1.5
    dvar = np.sum(dx_dvar, axis=-1, keepdims=True)

    dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=-1, keepdims=True)
    dmean += dvar * np.sum(-2 * (x - mean), axis=-1, keepdims=True) / D
    
    # Gradient w.r.t. input
    dx = dx_hat / np.sqrt(var + eps)
    dx += dvar * 2 * (x - mean) / D
    dx += dmean / D
    
    return dx, dgamma, dbeta


class RegularizedLSTM:
    """
    LSTM with regularization techniques.

    [PATCH 2026-10-18] FORWARD ONLY. This single-layer teaching model (batch size 1, one-hot
    inputs, column vectors) has no backward pass, so nothing can train it; it is kept as a
    compact, one-sequence walkthrough of where dropout goes. The trainable, paper-faithful
    model is LSTMLanguageModel at the bottom of this file.

    Features:
    - Dropout on the input x_t and on the h_t -> output path; the recurrent h_t carried
      to t+1 is never dropped (Zaremba et al. Sec. 3.2)
    - Layer normalization on the output path (our addition, not in the paper)
    - Weight decay and gradient clipping are separate helpers below (no training loop here)
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int,
                 dropout_keep_prob: float = 0.8, use_layer_norm: bool = True):
        """
        Initialize regularized LSTM.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden state size
            output_size: Output size (vocab_size for LM)
            dropout_keep_prob: Dropout keep probability
            use_layer_norm: Whether to use layer normalization
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_keep_prob = dropout_keep_prob
        self.use_layer_norm = use_layer_norm
        
        # Parameters
        scale = 0.01
        input_size = hidden_size + vocab_size
        
        # LSTM gates (same as before)
        self.Wf = np.random.randn(hidden_size, input_size) * scale
        self.Wi = np.random.randn(hidden_size, input_size) * scale
        self.Wc = np.random.randn(hidden_size, input_size) * scale
        self.Wo = np.random.randn(hidden_size, input_size) * scale
        
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        
        # Output layer
        self.Why = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))
        
        # Layer norm parameters
        if self.use_layer_norm:
            self.gamma_h = np.ones((hidden_size, 1))
            self.beta_h = np.zeros((hidden_size, 1))
    
    def forward(self, inputs: list, targets: list, h_prev: np.ndarray, 
                c_prev: np.ndarray, training: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Forward pass with regularization.
        
        Args:
            inputs: List of input indices
            targets: List of target indices
            h_prev: Previous hidden state
            c_prev: Previous cell state
            training: Whether in training mode (affects dropout)
            
        Returns:
            loss: Cross-entropy loss
            h: Final hidden state
            c: Final cell state
        """
        xs, hs, cs, ys, ps = {}, {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        cs[-1] = np.copy(c_prev)
        
        loss = 0
        
        for t, (inp, target) in enumerate(zip(inputs, targets)):
            # One-hot encode
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inp] = 1
            
            # Dropout on input
            x_dropped, x_mask = dropout_forward(xs[t], self.dropout_keep_prob, training)
            
            # LSTM forward
            combined = np.vstack([hs[t-1], x_dropped])
            
            f = sigmoid(np.dot(self.Wf, combined) + self.bf)
            i = sigmoid(np.dot(self.Wi, combined) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, combined) + self.bc)
            cs[t] = f * cs[t-1] + i * c_tilde
            
            o = sigmoid(np.dot(self.Wo, combined) + self.bo)
            hs[t] = o * np.tanh(cs[t])  # recurrent state: carried to t+1 without dropout

            # [PATCH 2026-10-18] Zaremba et al. (2014) Sec. 3.2: dropout only on non-recurrent
            # connections. The dropped (and layer-normed) copy used to overwrite hs[t], which is the
            # recurrent input at t+1, so the recurrent connection was dropped too; with
            # use_layer_norm=False the dropped copy was discarded, so h->output got no dropout.
            h_out, h_mask = dropout_forward(hs[t], self.dropout_keep_prob, training)

            # Layer norm on the output path only (our addition, not in the paper)
            if self.use_layer_norm:
                h_out, _ = layer_norm_forward(h_out.T, self.gamma_h.T, self.beta_h.T)
                h_out = h_out.T

            # Output layer
            ys[t] = np.dot(self.Why, h_out) + self.by
            ps[t] = softmax(ys[t])
            
            # Loss
            loss += -np.log(ps[t][target, 0] + 1e-8)
        
        return loss, hs[len(inputs)-1], cs[len(inputs)-1]
    
    def parameter_count(self) -> int:
        """Count total parameters."""
        count = (self.Wf.size + self.Wi.size + self.Wc.size + self.Wo.size +
                self.bf.size + self.bi.size + self.bc.size + self.bo.size +
                self.Why.size + self.by.size)
        
        if self.use_layer_norm:
            count += self.gamma_h.size + self.beta_h.size
        
        return count


class EarlyStoppingMonitor:
    """
    Monitor training and perform early stopping.
    
    Tracks validation loss and saves best model when it improves.
    Stops when validation loss doesn't improve for 'patience' epochs.
    """
    
    def __init__(self, patience: int = 5, verbose: bool = True):
        """
        Initialize early stopping monitor.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            verbose: Whether to print status
        """
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.no_improve_count = 0
        self.best_model_state = None
    
    def check(self, val_loss: float, epoch: int, model_state: Optional[Dict] = None) -> bool:
        """
        Check if should continue training.
        
        Args:
            val_loss: Validation loss
            epoch: Current epoch
            model_state: Model parameters to save
            
        Returns:
            True if should continue, False if should stop
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.no_improve_count = 0
            self.best_model_state = model_state
            
            if self.verbose:
                print(f"  [ok] Epoch {epoch}: val_loss improved to {val_loss:.4f}")
        else:
            self.no_improve_count += 1
            
            if self.verbose:
                print(f"  [FAIL] Epoch {epoch}: val_loss {val_loss:.4f} "
                      f"(no improve for {self.no_improve_count}/{self.patience})")
        
        return self.no_improve_count < self.patience
    
    def should_stop(self) -> bool:
        """Check if should stop training."""
        return self.no_improve_count >= self.patience


class RegularizationConfig:
    """Configuration for regularization strategies."""
    
    def __init__(self):
        self.dropout_keep_prob = 0.8
        self.weight_decay = 0.0001
        self.gradient_clip = 5.0
        self.use_layer_norm = True
        self.learning_rate = 0.01
        self.batch_size = 32
        self.val_split = 0.1
        self.patience = 5


# ============================================================================
# UTILITIES
# ============================================================================

def compute_l2_penalty(weights_list: list, weight_decay: float) -> float:
    """
    Compute L2 regularization penalty.
    
    L2_penalty = weight_decay * sum(w^2)
    
    Args:
        weights_list: List of weight matrices
        weight_decay: Coefficient (lambda)
        
    Returns:
        L2 penalty (scalar)
    """
    penalty = 0.0
    for w in weights_list:
        penalty += np.sum(w ** 2)
    
    return weight_decay * penalty / 2


def regularized_loss(model_loss: float, weights_list: list, 
                     weight_decay: float) -> float:
    """
    Compute total loss with L2 regularization.
    
    Total_loss = model_loss + L2_penalty
    
    Args:
        model_loss: Loss from model (e.g., cross-entropy)
        weights_list: List of weight matrices
        weight_decay: Weight decay coefficient
        
    Returns:
        Total loss
    """
    l2_penalty = compute_l2_penalty(weights_list, weight_decay)
    return model_loss + l2_penalty


def clip_gradients(grads_dict: Dict, clip_value: float) -> Dict:
    """
    Clip gradients to prevent explosion.

    Clips all gradient magnitudes to [-clip_value, clip_value].

    [PATCH 2026-10-18] Element-wise clipping changes the gradient's direction. The paper
    (Sec. 4.1) instead clips the global gradient NORM; that is clip_grad_norm below.

    Args:
        grads_dict: Dictionary of gradients
        clip_value: Clipping threshold
        
    Returns:
        Clipped gradients dictionary
    """
    clipped = {}
    for key, grad in grads_dict.items():
        clipped[key] = np.clip(grad, -clip_value, clip_value)
    return clipped


# ============================================================================
# [PATCH 2026-10-18] THE PAPER'S MODEL, WITH A BACKWARD PASS
# ============================================================================
# Zaremba, Sutskever & Vinyals (2014), "Recurrent Neural Network Regularization":
#   Sec. 3.1  LSTM equations (gate order i, f, o, g)
#   Sec. 3.2  dropout D only on non-recurrent connections
#   Sec. 4.1  PTB language model: 2 layers unrolled 35 steps, minibatch 20, hidden state carried
#             across minibatches, uniform init, SGD with lr 1 and per-epoch decay, gradient-norm
#             clipping "normalized by minibatch size".
# Everything below is plain NumPy; the backward pass is written out by hand and checked against
# torch.autograd in validation/day03_bptt_check.py.


def _sigm(x):
    """Logistic sigmoid written as sigm(x) = (1 + tanh(x/2)) / 2.

    Same function as 1 / (1 + exp(-x)), but np.tanh never overflows, so no
    RuntimeWarnings for large |x| (np.exp(88.8) already overflows in float32).
    """
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def clip_grad_norm(grads: Dict[str, np.ndarray], max_norm: float) -> float:
    """Global gradient-norm clipping (paper Sec. 4.1: "We clip the norm of the gradients
    (normalized by minibatch size) at 5"; 10 for the large model).

    One L2 norm over ALL parameters together: if it exceeds max_norm, every gradient is
    multiplied by the same factor max_norm / norm, so the update keeps its direction and only
    its length is capped. "Normalized by minibatch size" refers to the loss being differentiated:
    LSTMLanguageModel's loss divides the summed cross-entropy by the batch size B.

    Modifies grads in place and returns the norm measured BEFORE clipping.
    """
    # np.vdot(g, g) flattens g and returns sum(g * g) in one BLAS call (no temporary g * g array)
    norm = float(np.sqrt(sum(float(np.vdot(g, g)) for g in grads.values())))
    if norm > max_norm:
        scale = max_norm / norm
        for g in grads.values():
            g *= scale          # in place: the caller's arrays shrink by the same factor
    return norm


class LSTMLanguageModel:
    """
    Word-level LSTM language model as trained in Zaremba et al. (2014), Sec. 4.1, with a
    hand-written backward pass (truncated backpropagation through time).

        word ids -> embedding -D-> LSTM 1 -D-> LSTM 2 -D-> softmax over the vocabulary
                                   ^ h, c        ^ h, c
                                   (carried across time steps, NEVER dropped)

    Dropout (Sec. 3.2): D is applied on every non-recurrent (vertical) connection, i.e. on
    embedding -> layer 1, on layer l -> layer l+1 and on the top layer -> softmax; the recurrent
    h_{t-1} -> h_t and c_{t-1} -> c_t paths are never dropped. Information therefore meets D
    exactly L+1 times on its way up, however many time steps it is carried. D is inverted
    dropout: keep each unit with probability keep_prob and scale survivors by 1/keep_prob (the
    paper's Torch nn.Dropout does the same scaling). A fresh mask is drawn for every time step
    and every sequence in the minibatch, as in the paper's code, which applies nn.Dropout
    separately at each unrolled step.

    LSTM cell (Sec. 3.1; the paper's affine map T_{2n,4n} is our [W_x; W_h] plus b):
        a = D(h^{l-1}_t) @ W_x + h^l_{t-1} @ W_h + b        (B, 4H), columns = [i | f | o | g]
        i, f, o = sigm(a_i), sigm(a_f), sigm(a_o);  g = tanh(a_g)
        c^l_t = f * c^l_{t-1} + i * g
        h^l_t = o * tanh(c^l_t)
    Gate order is the paper's (i, f, o, g). torch.nn.LSTM uses (i, f, g, o) and the paper's own
    Lua code (i, g, f, o): reorder the blocks when copying weights between them.

    Shapes (row-vector convention; B sequences side by side):
        token ids x, y: (T, B) ints;  per-layer activations (T, B, H);  logits (T*B, V).
        The embedding size equals the hidden size H, as in the paper's code.

    Loss (what the paper's code differentiates): at each time step the cross-entropy is averaged
    over the minibatch, and the T steps are summed:
        loss = (1/B) * sum_{t,b} -log p(y_{t,b} | words up to x_{t,b})
    Perplexity = exp(average per-word cross-entropy) = exp(loss / T).

    Initialization (Sec. 4.1): every parameter, embeddings and biases included, uniform in
    [-init_scale, init_scale]: 0.1 for the paper's 200-unit model, 0.05 for 650, 0.04 for 1500.
    (The paper's code has two bias vectors per layer, one in each nn.Linear; one is equivalent.)
    """

    def __init__(self, vocab_size: int, hidden_size: int = 200, num_layers: int = 2,
                 keep_prob: float = 0.5, init_scale: float = 0.1, dtype=np.float32,
                 seed: int = 0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob                 # 1.0 = no dropout (the paper's non-regularized model)
        self.dtype = np.dtype(dtype)
        self.rng = np.random.default_rng(seed)     # one generator for the init and every dropout mask

        V, H = vocab_size, hidden_size
        shapes = {'embed': (V, H)}                 # row k = vector of word k (a lookup table)
        for l in range(1, num_layers + 1):
            shapes[f'W_x{l}'] = (H, 4 * H)         # non-recurrent input (from below) -> 4 gate pre-activations
            shapes[f'W_h{l}'] = (H, 4 * H)         # recurrent input h_{t-1} -> 4 gate pre-activations
            shapes[f'b{l}'] = (4 * H,)
        shapes['W_y'] = (H, V)                     # top hidden state -> one logit per vocabulary word
        shapes['b_y'] = (V,)
        # rng.uniform(lo, hi, size) draws i.i.d. U[lo, hi) samples; .astype picks float32/float64
        self.params = {k: self.rng.uniform(-init_scale, init_scale, size=s).astype(self.dtype)
                       for k, s in shapes.items()}

    # ------------------------------------------------------------------ helpers
    def parameter_count(self) -> int:
        """Total number of trainable numbers (2x200 on PTB: 4,651,600)."""
        return int(sum(p.size for p in self.params.values()))

    def init_state(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Zero (h, c) for every layer; Sec. 4.1: 'We initialize the hidden states to zero'."""
        z = lambda: np.zeros((batch_size, self.hidden_size), dtype=self.dtype)
        return [(z(), z()) for _ in range(self.num_layers)]

    def sample_masks(self, T: int, B: int) -> List[np.ndarray]:
        """L+1 inverted-dropout masks, each (T, B, H) with entries 0 or 1/keep_prob.

        masks[0]: embedding -> layer 1;  masks[l]: layer l -> layer l+1;  masks[L]: top -> softmax.
        There is deliberately no mask for h_{t-1} -> h_t (the recurrent connection).
        """
        p = self.keep_prob
        # rng.random(shape) is U[0, 1); "< p" is True with probability p -> keep that unit.
        # Casting the boolean array to float gives 1.0 / 0.0; dividing by p makes E[mask] = 1.
        return [(self.rng.random((T, B, self.hidden_size), dtype=self.dtype) < p).astype(self.dtype) / p
                for _ in range(self.num_layers + 1)]

    # ------------------------------------------------------------------ forward
    @staticmethod
    def _layer_forward(X, h0, c0, W_x, W_h, b):
        """Run one LSTM layer over a whole window (Sec. 3.1 equations), one time step at a time.

        X: (T, B, H_in) input from below, ALREADY dropped;  h0, c0: (B, H) carried-in state.
        Returns the hidden states Hs (T, B, H) and a cache for _layer_backward.
        """
        T, B = X.shape[0], X.shape[1]
        H = h0.shape[1]
        # The non-recurrent part does not depend on h, so all T steps take ONE matmul:
        # (T, B, H_in) -> reshape (T*B, H_in) @ (H_in, 4H) -> (T*B, 4H) -> reshape back (T, B, 4H).
        # "+ b" broadcasts the (4H,) bias over every row.
        Ax = (X.reshape(T * B, -1) @ W_x + b).reshape(T, B, 4 * H)
        G = np.empty((T, B, 4 * H), dtype=X.dtype)   # gate values [i | f | o | g] after sigm/tanh
        C = np.empty((T, B, H), dtype=X.dtype)       # cell states c_t
        TC = np.empty((T, B, H), dtype=X.dtype)      # tanh(c_t), reused by the backward pass
        Hs = np.empty((T, B, H), dtype=X.dtype)      # hidden states h_t
        h, c = h0, c0
        for t in range(T):
            a = Ax[t] + h @ W_h                      # (B, H) @ (H, 4H) -> (B, 4H): recurrent part, never dropped
            G[t, :, :3 * H] = _sigm(a[:, :3 * H])    # i, f, o share one sigmoid: the first 3H columns
            G[t, :, 3 * H:] = np.tanh(a[:, 3 * H:])  # g (the paper's "input modulation gate"): last H columns
            # Column slices of G[t] are views (no copies): (B, H) each
            i, f, o, g = G[t, :, :H], G[t, :, H:2 * H], G[t, :, 2 * H:3 * H], G[t, :, 3 * H:]
            c = f * c + i * g                        # elementwise: c_t = f (.) c_{t-1} + i (.) g
            C[t] = c
            TC[t] = np.tanh(c)
            h = o * TC[t]                            # h_t = o (.) tanh(c_t)
            Hs[t] = h
        return Hs, {'X': X, 'G': G, 'C': C, 'TC': TC, 'Hs': Hs, 'h0': h0, 'c0': c0}

    def forward(self, x: np.ndarray, y: np.ndarray, state, masks=None, training: bool = True):
        """Forward pass over one BPTT window.

        Args:
            x, y: (T, B) int arrays of input and target word ids (y[t] is the word after x[t]).
            state: list of (h, c) per layer, each (B, H): the state carried from the previous
                window (truncated BPTT: the forward pass uses it, the backward pass stops at it).
            masks: optional list of L+1 dropout masks (see sample_masks). If None, fresh masks
                are drawn when training; pass them explicitly to make a run reproducible (the
                gradient check gives the same masks to NumPy and PyTorch).
            training: False turns dropout off (evaluation) and skips building the cache.

        Returns:
            loss: (1/B) * summed cross-entropy over the T*B predictions (a Python float).
            new_state: list of (h_T, c_T) per layer: the UN-dropped state to carry forward.
            cache: what backward() needs (None when training=False).
        """
        p, H, L = self.params, self.hidden_size, self.num_layers
        T, B = x.shape
        if not training:
            masks = None                               # evaluation: the full network, no dropout
        elif masks is None and self.keep_prob < 1.0:
            masks = self.sample_masks(T, B)

        # Embedding lookup: indexing a (V, H) table with a (T, B) int array picks one row per id
        # -> (T, B, H). It equals one_hot(x) @ embed without building the one-hot vectors.
        inp = p['embed'][x]
        layer_caches, new_state = [], []
        for l in range(1, L + 1):
            # D(h^{l-1}_t): the vertical connection into layer l is dropped (h^0_t = embedding)
            X = inp * masks[l - 1] if masks is not None else inp
            h0, c0 = state[l - 1]
            Hs, lc = self._layer_forward(X, h0, c0, p[f'W_x{l}'], p[f'W_h{l}'], p[f'b{l}'])
            layer_caches.append(lc)
            # Carry the CLEAN last state to the next window: dropout never touches it.
            new_state.append((Hs[-1].copy(), lc['C'][-1].copy()))
            inp = Hs                                   # undropped output; the next layer drops its own copy
        top = inp * masks[L] if masks is not None else inp   # D(h^L_t) on the way to the softmax

        N = T * B
        Z = top.reshape(N, H)                          # (T, B, H) -> (T*B, H): one row per prediction
        logits = Z @ p['W_y'] + p['b_y']               # (N, H) @ (H, V) -> (N, V)
        # Softmax cross-entropy, numerically stable: softmax(z) = softmax(z - max z), so after the
        # shift every exponent is <= 0 and np.exp cannot overflow.
        logits -= logits.max(axis=1, keepdims=True)    # (N, V) minus (N, 1): broadcast per row
        e = np.exp(logits)                             # (N, V)
        sumexp = e.sum(axis=1)                         # (N,)
        yf = y.reshape(N)
        # -log softmax(z)_y = log(sum_j exp z_j) - z_y; logits[np.arange(N), yf] picks, from row r,
        # the column yf[r] (the target word's logit) -> (N,)
        nll = np.log(sumexp) - logits[np.arange(N), yf]
        loss = float(nll.sum(dtype=np.float64)) / B    # sum in float64 to limit rounding error

        if not training:
            return loss, new_state, None
        cache = {'x': x, 'y': y, 'masks': masks, 'layers': layer_caches,
                 'top': top, 'e': e, 'sumexp': sumexp}
        return loss, new_state, cache

    # ------------------------------------------------------------------ backward
    @staticmethod
    def _layer_backward(dHs, lc, W_x, W_h):
        """Backpropagation through time for one LSTM layer, newest step first.

        dHs: (T, B, H) gradient of the loss w.r.t. this layer's outputs h_t coming from ABOVE
             (the next layer or the softmax), already multiplied by that connection's dropout mask.
        Returns dW_x, dW_h, db, dX (w.r.t. the DROPPED input X), and dL/dh0, dL/dc0.
        """
        X, G, C, TC, Hs, h0, c0 = (lc[k] for k in ('X', 'G', 'C', 'TC', 'Hs', 'h0', 'c0'))
        T, B, H = dHs.shape
        dA = np.empty_like(G)              # (T, B, 4H): dL/d(gate pre-activations a) at every step
        dh_next = np.zeros_like(h0)        # dL/dh_t that arrives from step t+1 through W_h
        dc_next = np.zeros_like(c0)        # dL/dc_t that arrives from step t+1 through the forget gate
        for t in reversed(range(T)):
            i, f, o, g = G[t, :, :H], G[t, :, H:2 * H], G[t, :, 2 * H:3 * H], G[t, :, 3 * H:]
            c_prev = C[t - 1] if t > 0 else c0
            # h_t feeds two places: the layer above at step t (dHs[t]) and this layer at step t+1.
            dh = dHs[t] + dh_next
            # c_t feeds h_t = o * tanh(c_t) (derivative o * (1 - tanh^2 c_t)) and c_{t+1} (dc_next).
            dc = dc_next + dh * o * (1.0 - TC[t] * TC[t])
            # Gate derivatives. Each pre-activation goes through sigm or tanh, whose derivatives are
            # written with the stored OUTPUTS: sigm'(a) = s * (1 - s), tanh'(a) = 1 - g^2.
            dA[t, :, :H] = (dc * g) * i * (1.0 - i)                 # i: dc_t/di = g
            dA[t, :, H:2 * H] = (dc * c_prev) * f * (1.0 - f)       # f: dc_t/df = c_{t-1}
            dA[t, :, 2 * H:3 * H] = (dh * TC[t]) * o * (1.0 - o)    # o: dh_t/do = tanh(c_t)
            dA[t, :, 3 * H:] = (dc * i) * (1.0 - g * g)             # g: dc_t/dg = i
            dc_next = dc * f                       # c_{t-1} reaches c_t only through f (.) c_{t-1}
            # (B, 4H) @ (4H, H) -> (B, H): back through h_{t-1} @ W_h. No mask here: the recurrent
            # connection was never dropped, so its gradient is never dropped either.
            dh_next = dA[t] @ W_h.T
        # Weight gradients for all T steps in one matmul each: reshaping (T, B, .) to (T*B, .)
        # stacks every (step, sequence) pair as a row, and A.T @ dA sums the per-row outer products.
        dA2 = dA.reshape(T * B, 4 * H)
        # h_{t-1} for every step t: the carried-in state h0, then Hs[0] .. Hs[T-2]. h0[None] turns the
        # (B, H) array into (1, B, H) so it can be stacked in front of Hs along the time axis.
        H_prev = np.concatenate([h0[None], Hs[:-1]], axis=0)        # (T, B, H)
        dW_h = H_prev.reshape(T * B, H).T @ dA2                     # (H, T*B) @ (T*B, 4H) -> (H, 4H)
        dW_x = X.reshape(T * B, -1).T @ dA2                         # (H_in, T*B) @ (T*B, 4H) -> (H_in, 4H)
        db = dA2.sum(axis=0)                                        # (4H,): the bias is added to every row
        dX = (dA2 @ W_x.T).reshape(X.shape)                         # (T*B, 4H) @ (4H, H_in) -> (T, B, H_in)
        return dW_x, dW_h, db, dX, dh_next, dc_next

    def backward(self, cache):
        """Gradients of the forward() loss w.r.t. every parameter (same keys as self.params).

        Returns (grads, dstate). dstate holds dL/d(h0, c0) per layer: the gradient that would flow
        into the PREVIOUS window. Truncated BPTT simply discards it (the training loop ignores it);
        the gradient check compares it with autograd to test the carried-state path too.
        """
        p, H, L = self.params, self.hidden_size, self.num_layers
        x, y, masks = cache['x'], cache['y'], cache['masks']
        T, B = x.shape
        N = T * B

        # Softmax + cross-entropy: d(-log softmax(z)_y)/dz = softmax(z) - one_hot(y); /B from the loss.
        dlogits = cache['e'] / cache['sumexp'][:, None]    # (N, V) / (N, 1): probabilities per row
        dlogits[np.arange(N), y.reshape(N)] -= 1.0         # subtract 1 at each row's target column
        dlogits /= B
        grads = {}
        Z = cache['top'].reshape(N, H)
        grads['W_y'] = Z.T @ dlogits                        # (H, N) @ (N, V) -> (H, V)
        grads['b_y'] = dlogits.sum(axis=0)                  # (V,)
        dtop = (dlogits @ p['W_y'].T).reshape(T, B, H)      # (N, V) @ (V, H) -> dL/d(dropped top output)

        # Dropout backward = multiply by the SAME mask used forward: a unit that was zeroed passed
        # nothing up, so it gets no gradient; a kept unit was scaled by 1/keep_prob, and so is its
        # gradient. This is where each of the L+1 masks enters the backward pass.
        dHs = dtop * masks[L] if masks is not None else dtop
        dstate = [None] * L
        for l in range(L, 0, -1):
            dW_x, dW_h, db, dX, dh0, dc0 = self._layer_backward(
                dHs, cache['layers'][l - 1], p[f'W_x{l}'], p[f'W_h{l}'])
            grads[f'W_x{l}'], grads[f'W_h{l}'], grads[f'b{l}'] = dW_x, dW_h, db
            dstate[l - 1] = (dh0, dc0)
            # dX is w.r.t. D(h^{l-1}); route it through mask l-1 to get dL/dh^{l-1} (or the embedding)
            dHs = dX * masks[l - 1] if masks is not None else dX
        # dHs is now dL/d(embedding vectors used in this window), (T, B, H). Scatter it back into
        # the (V, H) table. np.add.at is an unbuffered scatter-add: a word that occurs k times gets
        # all k rows added (dE[ids] += rows would keep only one of them).
        dE = np.zeros_like(p['embed'])
        np.add.at(dE, x.reshape(N), dHs.reshape(N, H))
        grads['embed'] = dE
        return grads, dstate

    # ------------------------------------------------------------------ update
    def sgd_update(self, grads: Dict[str, np.ndarray], lr: float) -> None:
        """Plain SGD, as in the paper: theta <- theta - lr * grad (no momentum)."""
        for k, g in grads.items():
            self.params[k] -= lr * g              # in place, so the arrays keep their identity

    def get_state(self) -> Dict[str, np.ndarray]:
        """Copy of all parameters (e.g. to keep the best-validation model)."""
        return {k: v.copy() for k, v in self.params.items()}

    def set_state(self, state: Dict[str, np.ndarray]) -> None:
        for k, v in state.items():
            self.params[k][...] = v               # [...] = copies into the existing array


if __name__ == "__main__":
    print("Regularization techniques module loaded.")
    print("Classes: RegularizedLSTM (forward-only teaching model), LSTMLanguageModel (paper model,")
    print("         trainable: forward + BPTT backward), EarlyStoppingMonitor")
    print("Functions: dropout_forward/backward, layer_norm_forward/backward, clip_grad_norm")
