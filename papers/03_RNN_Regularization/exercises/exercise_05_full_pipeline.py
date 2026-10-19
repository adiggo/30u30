"""
Exercise 5: Full Regularization Pipeline
==========================================

Goal: Combine ALL regularization techniques into one complete training pipeline.

Your Task:
- Part A (TODOs 1-12): build a regularized RNN cell: dropout on the input, layer norm,
  an L2 penalty and weight snapshots
- Part B (TODOs 13-15): write a training loop that really trains: the paper's 2-layer LSTM
  language model (LSTMLanguageModel from ../implementation.py) on a slice of Penn Treebank,
  with dropout, weight decay, gradient clipping, SGD and early stopping
- Test your implementation with the provided tests

Learning Objectives:
1. Integrate multiple regularization techniques
2. Understand how they work together
3. Build a training loop that updates the weights (BPTT + SGD) and keeps the best ones
4. See the combined effect on overfitting

Time: 40-50 minutes
Difficulty: Hard

The Four Regularization Techniques:
    1. DROPOUT: Randomly zero neurons (training only!), on non-recurrent connections only
    2. LAYER NORM: Normalize activations for stability (our addition; Part A only, the
       paper's model has none)
    3. WEIGHT DECAY: Penalize large weights (L2 regularization; our addition)
    4. EARLY STOPPING: Stop when validation loss stops improving (our addition)

[PATCH 2026-10-18] Part B used to compute losses only (no backward pass, so nothing trained).
It now trains LSTMLanguageModel, whose backward pass is checked against torch.autograd in
validation/day03_bptt_check.py. The data helpers come from ../train_minimal.py; Penn Treebank
is read from data/ptb/ (or $PTB_DIR) and downloaded (~6 MB) on first use.
"""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

# [PATCH 2026-10-18] Part B uses the day's model and data helpers, one folder up.
# (clip_grad_norm and windows are used by the code you write in TODO 15.)
DAY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAY_DIR not in sys.path:
    sys.path.insert(0, DAY_DIR)
from implementation import LSTMLanguageModel, clip_grad_norm  # noqa: E402,F401
from train_minimal import load_ptb, batchify, windows, evaluate  # noqa: E402,F401


class RegularizedRNNCell:
    """
    A single RNN cell with dropout and layer normalization.
    
    Architecture:
        x_t = dropout(x_t)  # Only during training!  [PATCH 2026-10-18] never on the recurrent h_t
        h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
        h_t = layer_norm(h_t)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        dropout_rate: float = 0.3,
        weight_decay: float = 0.0001
    ):
        """
        Initialize the regularized RNN cell.
        
        Args:
            input_size: Dimension of input vectors
            hidden_size: Dimension of hidden state
            dropout_rate: Probability of dropping a neuron (0.0 to 1.0)
            weight_decay: L2 regularization coefficient
        """
        # TODO 1: Store parameters
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.dropout_rate = dropout_rate
        # self.weight_decay = weight_decay
        
        # TODO 2: Initialize weights (Xavier initialization)
        # Xavier: scale by sqrt(2 / (fan_in + fan_out))
        # self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        # self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (2 * hidden_size))
        # self.b_h = np.zeros(hidden_size)
        
        # TODO 3: Initialize layer norm parameters
        # self.gamma = np.ones(hidden_size)   # Scale
        # self.beta = np.zeros(hidden_size)   # Shift
        # self.epsilon = 1e-5                  # Numerical stability
        
        pass  # Remove this when you implement
    
    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        
        Formula:
            mean = mean(x)
            var = var(x)
            x_norm = (x - mean) / sqrt(var + epsilon)
            output = gamma * x_norm + beta
        """
        # TODO 4: Implement layer normalization
        # mean = np.mean(x, axis=-1, keepdims=True)
        # var = np.var(x, axis=-1, keepdims=True)
        # x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        # return self.gamma * x_norm + self.beta
        
        pass  # Remove this when you implement
    
    def dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply dropout.
        
        During training: Randomly zero elements with probability dropout_rate
                        Scale remaining by 1/(1-dropout_rate)
        During inference: Return x unchanged
        """
        # TODO 5: Implement dropout with inverted scaling
        # if not training or self.dropout_rate == 0:
        #     return x
        # 
        # mask = np.random.rand(*x.shape) > self.dropout_rate
        # return (x * mask) / (1 - self.dropout_rate)
        
        pass  # Remove this when you implement
    
    def forward(
        self, 
        x: np.ndarray, 
        h_prev: np.ndarray, 
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass of regularized RNN cell.
        
        Args:
            x: Input at current timestep (batch_size, input_size)
            h_prev: Hidden state from previous timestep (batch_size, hidden_size)
            training: Whether we're training (affects dropout)
            
        Returns:
            h: New hidden state (batch_size, hidden_size)
        """
        # TODO 8: Apply dropout to the INPUT only (only during training!)
        # [PATCH 2026-10-18] the returned h becomes the next h_prev, so dropping h would drop the
        # recurrent connection, which Zaremba et al. (2014) Sec. 3.2 says not to do.
        # x = self.dropout(x, training=training)

        # TODO 6: Compute raw hidden state
        # h_raw = np.tanh(self.W_hh @ h_prev.T + self.W_xh @ x.T + self.b_h.reshape(-1, 1))
        # h_raw = h_raw.T  # Back to (batch_size, hidden_size)

        # TODO 7: Apply layer normalization
        # h_norm = self.layer_norm(h_raw)

        # TODO 9: Return final hidden state
        # return h_norm
        
        pass  # Remove this when you implement
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return all weights as a dictionary."""
        # TODO 10: Return weights dictionary
        # return {
        #     'W_xh': self.W_xh,
        #     'W_hh': self.W_hh,
        #     'b_h': self.b_h,
        #     'gamma': self.gamma,
        #     'beta': self.beta
        # }
        pass  # Remove this when you implement
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights from a dictionary."""
        # TODO 11: Set weights from dictionary
        # self.W_xh = weights['W_xh'].copy()
        # self.W_hh = weights['W_hh'].copy()
        # self.b_h = weights['b_h'].copy()
        # self.gamma = weights['gamma'].copy()
        # self.beta = weights['beta'].copy()
        pass  # Remove this when you implement
    
    def compute_l2_penalty(self) -> float:
        """Compute L2 regularization penalty."""
        # TODO 12: Compute L2 penalty for W_xh and W_hh
        # penalty = 0.0
        # penalty += np.sum(self.W_xh ** 2)
        # penalty += np.sum(self.W_hh ** 2)
        # return (self.weight_decay / 2) * penalty
        pass  # Remove this when you implement


class EarlyStopping:
    """Early stopping handler (simplified version)."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, weights: Optional[Dict] = None) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if weights is not None:
                self.best_weights = {k: v.copy() for k, v in weights.items()}
            return False
        
        self.wait += 1
        return self.wait >= self.patience
    
    def get_best_weights(self) -> Optional[Dict]:
        return self.best_weights


# ============================================================================
# PART B: A REAL TRAINING LOOP (the paper's model on a Penn Treebank slice)
# ============================================================================

def load_ptb_slice(n_train: int = 30_000, n_valid: int = 20_000) -> Tuple[np.ndarray, np.ndarray, int]:
    """First n_train training words and first n_valid validation words of Penn Treebank as 1-D
    word-id arrays, plus the vocabulary size (10,000). Uses load_ptb from ../train_minimal.py."""
    train_ids, valid_ids, _, vocab = load_ptb()
    return train_ids[:n_train], valid_ids[:n_valid], len(vocab)


def train_regularized_rnn(
    model: LSTMLanguageModel,
    train_data: np.ndarray,
    val_ids: np.ndarray,
    learning_rate: float = 1.0,
    max_epochs: int = 10,
    patience: int = 2,
    weight_decay: float = 0.0,
    bptt: int = 20,
    max_grad_norm: float = 5.0
) -> Dict:
    """
    Train a word-level LSTM language model with all the regularizers and early stopping.

    The model is the paper's (Zaremba et al. 2014): model.keep_prob sets the dropout on the
    non-recurrent connections, applied inside model.forward(..., training=True); the recurrent
    state is never dropped. Training is truncated BPTT: the hidden state is carried from one
    bptt-step window to the next, but gradients stop at the window boundary.

    Args:
        model: LSTMLanguageModel to train (updated in place; it ends with its best weights)
        train_data: (rows, B) word ids from batchify(train_ids, B): B parallel word streams
        val_ids: 1-D array of validation word ids
        learning_rate: SGD step size (the paper uses 1.0)
        max_epochs: Maximum number of passes over train_data
        patience: Epochs without a validation improvement before stopping
        weight_decay: L2 coefficient lambda; adds lambda * W to every weight gradient
        bptt: Time steps per truncated-BPTT window
        max_grad_norm: Clip the global gradient norm at this value (the paper uses 5)
        
    Returns:
        Dictionary with training history: 'train_loss' and 'val_loss' per epoch (mean
        cross-entropy per word in nats; perplexity = exp(loss)), 'epochs_trained',
        'stopped_early' and 'best_epoch'
    """
    # TODO 13: Initialize early stopping
    # early_stop = EarlyStopping(patience=patience)
    
    # TODO 14: Initialize history tracking
    # history = {
    #     'train_loss': [],
    #     'val_loss': [],
    #     'epochs_trained': 0,
    #     'stopped_early': False,
    #     'best_epoch': 0
    # }
    # batch_size = train_data.shape[1]
    
    # TODO 15: Training loop
    # for epoch in range(max_epochs):
    #     # Training phase: dropout ON, weights updated after every window
    #     state = model.init_state(batch_size)      # zero (h, c) at the start of each pass
    #     nll_sum, n_words = 0.0, 0
    #     for x, y in windows(train_data, bptt):     # x, y: (T, B) word ids; y = the next words
    #         loss, state, cache = model.forward(x, y, state, training=True)
    #         grads, _ = model.backward(cache)       # BPTT through this window only
    #         
    #         # Weight decay: the gradient of (lambda/2) * ||W||^2 is lambda * W.
    #         # Apply it to weights and embeddings (keys not starting with 'b'), not biases.
    #         for k, g in grads.items():
    #             if not k.startswith('b'):
    #                 g += weight_decay * model.params[k]
    #         
    #         clip_grad_norm(grads, max_grad_norm)    # one global norm, as in the paper
    #         model.sgd_update(grads, learning_rate)  # W <- W - lr * grad
    #         nll_sum += loss * x.shape[1]           # loss is (1/B) * summed NLL
    #         n_words += x.size
    #     train_loss = nll_sum / n_words             # mean NLL per word, measured with dropout on
    #     
    #     # Validation phase: dropout OFF (evaluate() runs with training=False)
    #     val_loss = float(np.log(evaluate(model, val_ids, batch_size, bptt)))  # log(perplexity)
    #     
    #     # Record history
    #     history['train_loss'].append(train_loss)
    #     history['val_loss'].append(val_loss)
    #     
    #     # Check early stopping (it snapshots the weights whenever validation improves)
    #     stop = early_stop(val_loss, model.get_state())
    #     if early_stop.wait == 0:                   # this epoch was the best so far
    #         history['best_epoch'] = epoch + 1
    #     if stop:
    #         history['stopped_early'] = True
    #         history['epochs_trained'] = epoch + 1
    #         break
    # else:
    #     history['epochs_trained'] = max_epochs
    # 
    # # Restore the best weights, whether or not training stopped early
    # model.set_state(early_stop.get_best_weights())
    # return history
    
    pass  # Remove this when you implement


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_rnn_cell_initialization():
    """Test RNN cell initialization."""
    print("Testing RNN cell initialization...")
    
    cell = RegularizedRNNCell(input_size=10, hidden_size=20, dropout_rate=0.3)
    
    # Check shapes
    assert cell.W_xh.shape == (20, 10), f"W_xh shape wrong: {cell.W_xh.shape}"
    assert cell.W_hh.shape == (20, 20), f"W_hh shape wrong: {cell.W_hh.shape}"
    assert cell.b_h.shape == (20,), f"b_h shape wrong: {cell.b_h.shape}"
    assert cell.gamma.shape == (20,), f"gamma shape wrong: {cell.gamma.shape}"
    assert cell.beta.shape == (20,), f"beta shape wrong: {cell.beta.shape}"
    
    print("  [ok] All shapes correct")
    print("[ok] Initialization tests passed!\n")


def test_layer_norm():
    """Test layer normalization."""
    print("Testing layer normalization...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=10)
    
    # Test normalization
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
    x_norm = cell.layer_norm(x)
    
    # After normalization, mean should be ~0 and std ~1
    # (with gamma=1 and beta=0)
    mean = np.mean(x_norm)
    std = np.std(x_norm)
    
    assert abs(mean) < 0.01, f"Mean should be ~0, got {mean}"
    assert abs(std - 1.0) < 0.01, f"Std should be ~1, got {std}"
    
    print("  [ok] Normalization works correctly")
    print("[ok] Layer norm tests passed!\n")


def test_dropout():
    """Test dropout."""
    print("Testing dropout...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=100, dropout_rate=0.5)
    
    np.random.seed(42)
    x = np.ones((1, 100))
    
    # Test training mode
    x_dropped = cell.dropout(x, training=True)
    zeros = np.sum(x_dropped == 0)
    assert zeros > 20, f"Should have some zeros, got {zeros}"
    
    # Test inference mode
    x_kept = cell.dropout(x, training=False)
    assert np.allclose(x_kept, x), "Inference mode should not drop"
    
    print("  [ok] Training mode drops neurons")
    print("  [ok] Inference mode keeps all neurons")
    print("[ok] Dropout tests passed!\n")


def test_forward_pass():
    """Test forward pass."""
    print("Testing forward pass...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=10, dropout_rate=0.0)
    
    batch_size = 3
    x = np.random.randn(batch_size, 5)
    h_prev = np.zeros((batch_size, 10))
    
    h = cell.forward(x, h_prev, training=False)
    
    assert h.shape == (batch_size, 10), f"Output shape wrong: {h.shape}"
    
    print("  [ok] Forward pass works")
    print("[ok] Forward pass tests passed!\n")


def test_l2_penalty():
    """Test L2 penalty computation."""
    print("Testing L2 penalty...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=10, weight_decay=0.1)
    
    # Set simple weights
    cell.W_xh = np.ones((10, 5))   # Sum of squares = 50
    cell.W_hh = np.ones((10, 10))  # Sum of squares = 100
    
    penalty = cell.compute_l2_penalty()
    expected = 0.1 / 2 * (50 + 100)  # = 7.5
    
    assert abs(penalty - expected) < 1e-6, f"Expected {expected}, got {penalty}"
    
    print("  [ok] L2 penalty computed correctly")
    print("[ok] L2 penalty tests passed!\n")


def test_weight_save_restore():
    """Test weight save and restore."""
    print("Testing weight save/restore...")
    
    cell = RegularizedRNNCell(input_size=5, hidden_size=10)
    
    # Save weights
    weights = cell.get_weights()
    original_W_xh = weights['W_xh'].copy()
    
    # Modify weights
    cell.W_xh = np.zeros((10, 5))
    
    # Restore weights
    cell.set_weights(weights)
    
    assert np.allclose(cell.W_xh, original_W_xh), "Weights not restored correctly"
    
    print("  [ok] Weights saved and restored correctly")
    print("[ok] Weight save/restore tests passed!\n")


def test_full_pipeline():
    """Test the complete training pipeline: real training on a small Penn Treebank slice."""
    print("Testing full pipeline (LSTM language model on a Penn Treebank slice)...")
    
    # [PATCH 2026-10-18] the old test ran on random numbers and could not tell whether anything
    # trained. This one trains for 3 short epochs (a few seconds) and checks that it learned.
    train_ids, val_ids, vocab_size = load_ptb_slice(n_train=6000, n_valid=3000)
    model = LSTMLanguageModel(vocab_size, hidden_size=32, num_layers=2, keep_prob=0.8,
                              init_scale=0.1, seed=0)
    before = model.get_state()
    untrained = float(np.log(evaluate(model, val_ids, 10, 20)))   # about ln(10000) = 9.2 nats
    
    # Train
    history = train_regularized_rnn(
        model,
        batchify(train_ids, 10), val_ids,
        learning_rate=1.0,
        max_epochs=3,
        patience=2,
        weight_decay=1e-5,
        bptt=20
    )
    
    for key in ('train_loss', 'val_loss', 'epochs_trained', 'stopped_early', 'best_epoch'):
        assert key in history, f"Missing {key} in history"
    assert len(history['val_loss']) == history['epochs_trained'] > 0, "Expected one validation loss per epoch"
    changed = [k for k in before if not np.array_equal(before[k], model.params[k])]
    assert len(changed) == len(before), \
        f"Only {len(changed)}/{len(before)} parameter arrays changed: is the SGD update applied?"
    best = min(history['val_loss'])
    assert best < untrained - 1.0, f"Validation loss should fall well below {untrained:.2f} nats, got {best:.2f}"
    restored = float(np.log(evaluate(model, val_ids, 10, 20)))
    assert abs(restored - best) < 1e-6, "The model should end with its best-validation weights"
    
    print(f"  [ok] All {len(before)} parameter arrays updated")
    print(f"  [ok] Validation perplexity {np.exp(untrained):.0f} -> {np.exp(best):.0f} in {history['epochs_trained']} epochs")
    print(f"  [ok] Best weights (epoch {history['best_epoch']}) restored; early stopped: {history['stopped_early']}")
    print("[ok] Full pipeline tests passed!\n")


if __name__ == "__main__":
    print("="*60)
    print("Exercise 5: Full Regularization Pipeline")
    print("="*60 + "\n")
    
    try:
        test_rnn_cell_initialization()
        test_layer_norm()
        test_dropout()
        test_forward_pass()
        test_l2_penalty()
        test_weight_save_restore()
        test_full_pipeline()
        print("All tests passed!")
    except AssertionError as e:
        print(f"FAIL - Test failed: {e}")
    except Exception as e:
        print(f"FAIL - Error: {e}")
        import traceback
        traceback.print_exc()
