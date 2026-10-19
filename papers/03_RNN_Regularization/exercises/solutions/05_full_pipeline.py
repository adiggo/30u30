"""
Solution 5: Full Regularization Pipeline
==========================================

Complete solution combining ALL regularization techniques.

Part A: RegularizedRNNCell (dropout on the input, layer norm, L2 penalty, weight snapshots).
Part B: train_regularized_rnn, a training loop that really trains: the paper's 2-layer LSTM
language model (LSTMLanguageModel from ../../implementation.py) on a slice of Penn Treebank,
with dropout, weight decay, gradient-norm clipping, SGD and early stopping.

[PATCH 2026-10-18] Part B used to compute losses on random numbers without ever updating a
weight. Running this file takes about 2 minutes of CPU (the demo trains three small models).
"""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

# [PATCH 2026-10-18] Part B uses the day's model and data helpers, two folders up.
DAY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if DAY_DIR not in sys.path:
    sys.path.insert(0, DAY_DIR)
from implementation import LSTMLanguageModel, clip_grad_norm  # noqa: E402
from train_minimal import load_ptb, batchify, windows, evaluate  # noqa: E402


class RegularizedRNNCell:
    """
    SOLUTION 1: Complete RNN Cell with All Regularization
    
    This combines:
    - Dropout (on non-recurrent connections)
    - Layer Normalization (before activation)
    - Weight Decay (L2 regularization)
    
    Architecture per timestep:
    1. x_t (input) → Dropout (training only)
    2. Linear: z = W_xh @ x + W_hh @ h + b
    3. Layer Norm: z_norm = LN(z)
    4. Activation: h = tanh(z_norm)
    5. Output dropout (training only)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.3,
        weight_decay: float = 0.0001
    ):
        """
        SOLUTION 1a: Initialize all components
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        # Xavier initialization for weights
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (2 * hidden_size))
        self.b_h = np.zeros(hidden_size)
        
        # Layer norm parameters
        self.gamma = np.ones(hidden_size)
        self.beta = np.zeros(hidden_size)
        self.epsilon = 1e-5
    
    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        SOLUTION 2: Layer Normalization
        
        Normalizes across the feature dimension:
        - Compute mean and variance per sample
        - Normalize: (x - mean) / std
        - Scale and shift: gamma * x_norm + beta
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta
    
    def dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        SOLUTION 3: Inverted Dropout
        
        Training: Drop neurons randomly, scale remaining
        Inference: Return unchanged (no dropout)
        
        Why "inverted"?
        - Scale by 1/(1-rate) during training
        - No scaling needed at inference
        - Makes inference faster!
        """
        if not training or self.dropout_rate == 0:
            return x
        
        mask = np.random.rand(*x.shape) > self.dropout_rate
        return (x * mask) / (1 - self.dropout_rate)
    
    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        SOLUTION 4: Forward Pass with All Regularization
        
        Steps:
        1. Linear transformation (no dropout on recurrent!)
        2. Layer normalization
        3. Tanh activation
        4. Dropout (training only)
        """
        # [PATCH 2026-10-18] Zaremba et al. Sec. 3.2: dropout only on the non-recurrent input x_t.
        # The returned h is the caller's next h_prev, so dropping h (as before) dropped the
        # recurrent connection -- the exact thing the paper says not to do.
        x = self.dropout(x, training=training)

        # Linear transformation
        # Note: W_hh @ h_prev does NOT get dropout
        # (recurrent connections preserve memory)
        h_raw = np.tanh(
            self.W_hh @ h_prev.T +
            self.W_xh @ x.T +
            self.b_h.reshape(-1, 1)
        )
        h_raw = h_raw.T  # Back to (batch, hidden)

        # Layer normalization
        h_norm = self.layer_norm(h_raw)

        return h_norm  # [PATCH 2026-10-18] recurrent state returned without dropout
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Return all weights as dictionary."""
        return {
            'W_xh': self.W_xh,
            'W_hh': self.W_hh,
            'b_h': self.b_h,
            'gamma': self.gamma,
            'beta': self.beta
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set weights from dictionary."""
        self.W_xh = weights['W_xh'].copy()
        self.W_hh = weights['W_hh'].copy()
        self.b_h = weights['b_h'].copy()
        self.gamma = weights['gamma'].copy()
        self.beta = weights['beta'].copy()
    
    def compute_l2_penalty(self) -> float:
        """
        SOLUTION 5: L2 Regularization Penalty
        
        Penalize large weights to encourage simpler models.
        Only regularize W_xh and W_hh (not biases or layer norm).
        """
        penalty = 0.0
        penalty += np.sum(self.W_xh ** 2)
        penalty += np.sum(self.W_hh ** 2)
        return (self.weight_decay / 2) * penalty


class EarlyStopping:
    """Early stopping handler."""
    
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


def load_ptb_slice(n_train: int = 30_000, n_valid: int = 20_000) -> Tuple[np.ndarray, np.ndarray, int]:
    """First n_train training words and first n_valid validation words of Penn Treebank as 1-D
    word-id arrays, plus the vocabulary size (10,000). Penn Treebank is read from data/ptb/ (or
    $PTB_DIR) and downloaded (~6 MB) the first time, by load_ptb in ../../train_minimal.py."""
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
    SOLUTION 6: Complete Training Loop
    
    This demonstrates how all regularization techniques
    work together in a real training loop:
    
    1. DROPOUT: inside model.forward(..., training=True), on the non-recurrent connections only
       (model.keep_prob; Zaremba et al. Sec. 3.2). Validation runs with dropout off.
    2. WEIGHT DECAY: the gradient of (lambda/2)||W||^2, i.e. lambda * W, is added to every weight
       gradient before the update (our addition; the paper uses none).
    3. GRADIENT CLIPPING: one global norm over all gradients, capped at max_grad_norm (paper: 5).
    4. EARLY STOPPING: snapshot the weights whenever validation improves; stop after `patience`
       epochs without improvement; always finish with the best weights.
    (Layer norm is practiced in Part A only: the paper's model has none.)
    
    Training is truncated BPTT: the hidden state is carried across bptt-step windows, the
    gradient is not (model.backward stops at the window boundary).
    
    Returns history: 'train_loss' / 'val_loss' per epoch (mean cross-entropy per word in nats;
    perplexity = exp(loss)), 'epochs_trained', 'stopped_early', 'best_epoch'.
    """
    early_stop = EarlyStopping(patience=patience)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs_trained': 0,
        'stopped_early': False,
        'best_epoch': 0
    }
    batch_size = train_data.shape[1]         # train_data is (rows, B): B parallel word streams
    
    for epoch in range(max_epochs):
        # =====================
        # TRAINING PHASE
        # =====================
        # Dropout IS active (training=True); the weights change after every window.
        state = model.init_state(batch_size)    # zero (h, c) at the start of each pass
        nll_sum, n_words = 0.0, 0
        for x, y in windows(train_data, bptt):  # x, y: (T, B) word ids; y = the next words
            loss, state, cache = model.forward(x, y, state, training=True)
            grads, _ = model.backward(cache)    # BPTT through this window only
            
            # WEIGHT DECAY: d/dW (lambda/2)||W||^2 = lambda * W, for weights and embeddings
            # (keys not starting with 'b'), not biases. g += ... updates the array in place.
            for k, g in grads.items():
                if not k.startswith('b'):
                    g += weight_decay * model.params[k]
            
            clip_grad_norm(grads, max_grad_norm)     # rescale all gradients together if too long
            model.sgd_update(grads, learning_rate)   # W <- W - lr * grad
            nll_sum += loss * x.shape[1]             # loss is (1/B) * summed NLL
            n_words += x.size                        # T * B predictions in this window
        train_loss = nll_sum / n_words               # mean NLL per word, measured with dropout on
        
        # =====================
        # VALIDATION PHASE
        # =====================
        # Dropout is OFF: evaluate() runs model.forward(..., training=False) and returns perplexity
        val_loss = float(np.log(evaluate(model, val_ids, batch_size, bptt)))
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # =====================
        # EARLY STOPPING CHECK
        # =====================
        stop = early_stop(val_loss, model.get_state())   # saves a copy of the weights if improved
        if early_stop.wait == 0:                          # this epoch is the best so far
            history['best_epoch'] = epoch + 1
        if stop:
            history['stopped_early'] = True
            history['epochs_trained'] = epoch + 1
            break
    else:
        history['epochs_trained'] = max_epochs
    
    # Restore the best weights, whether or not training stopped early
    model.set_state(early_stop.get_best_weights())
    return history


def test_cell_components():
    """Test the Part A cell: dropout modes, layer norm, L2 penalty, weight snapshots."""
    print("Testing the regularized RNN cell (Part A)...\n")
    np.random.seed(42)
    cell = RegularizedRNNCell(input_size=3, hidden_size=8, dropout_rate=0.2, weight_decay=0.001)
    
    # Dropout: on in training mode, off in inference mode
    x = np.ones((1, 100))
    assert np.any(cell.dropout(x, training=True) == 0) and np.allclose(cell.dropout(x, training=False), x)
    print("   [ok] Dropout works (training/inference modes)")
    
    # Layer norm: bounded output even for a huge input
    h = cell.forward(np.random.randn(1, 3) * 100, np.zeros((1, 8)), training=False)
    print(f"   [ok] Layer norm works (output bounded: max={np.max(np.abs(h)):.2f})")
    
    # Weight decay penalty
    penalty = cell.compute_l2_penalty()
    assert penalty > 0
    print(f"   [ok] Weight decay works (penalty={penalty:.6f})")
    
    # Weight snapshots (what early stopping restores)
    saved = {k: v.copy() for k, v in cell.get_weights().items()}
    cell.W_xh = np.zeros_like(cell.W_xh)
    cell.set_weights(saved)
    assert np.allclose(cell.W_xh, saved['W_xh'])
    print("   [ok] Weight save/restore works\n")


def test_full_pipeline():
    """Test the complete training pipeline on a small Penn Treebank slice (a few seconds)."""
    print("Testing full regularization pipeline (Part B)...\n")
    
    print("1. Loading a small Penn Treebank slice...")
    train_ids, val_ids, vocab_size = load_ptb_slice(n_train=6000, n_valid=3000)
    print(f"   Train: {len(train_ids):,} words, Val: {len(val_ids):,} words, vocabulary {vocab_size:,}")
    
    print("\n2. Creating the paper's LSTM language model (tiny: 2 x 32 units)...")
    model = LSTMLanguageModel(vocab_size, hidden_size=32, num_layers=2, keep_prob=0.8,
                              init_scale=0.1, seed=0)
    before = model.get_state()
    untrained = float(np.log(evaluate(model, val_ids, 10, 20)))
    print(f"   Untrained validation perplexity: {np.exp(untrained):.0f}")
    
    print("\n3. Training with dropout, weight decay, clipping and early stopping (patience=2)...")
    history = train_regularized_rnn(model, batchify(train_ids, 10), val_ids, learning_rate=1.0,
                                    max_epochs=3, patience=2, weight_decay=1e-5, bptt=20)
    
    print("\n4. Results:")
    print(f"   Epochs trained: {history['epochs_trained']} (early stopped: {history['stopped_early']})")
    print(f"   Validation perplexity by epoch: {', '.join(f'{np.exp(v):.0f}' for v in history['val_loss'])}")
    
    changed = [k for k in before if not np.array_equal(before[k], model.params[k])]
    assert len(changed) == len(before), "every parameter array should change"
    best = min(history['val_loss'])
    assert best < untrained - 1.0, "validation loss should fall well below its untrained value"
    assert abs(float(np.log(evaluate(model, val_ids, 10, 20))) - best) < 1e-6, "best weights restored"
    print(f"   [ok] All {len(before)} parameter arrays updated; best weights (epoch {history['best_epoch']}) restored")
    print("\nFull pipeline test passed!")


def demonstrate_regularization_effects():
    """Show how each regularization technique helps (about 2 minutes of CPU).
    
    Same model, data, initialization and dropout seed; only the regularizers differ. Constant
    learning rate 1 so that overfitting shows up within a few epochs, and early stopping
    (patience 2) in every run.
    """
    print("\nDemonstrating regularization effects on Penn Treebank...\n")
    
    # A small slice: on 30k words a 2 x 64 LSTM (1.36M parameters, most of them in the
    # 10,000-word embedding and softmax) overfits within a few epochs.
    train_ids, val_ids, vocab_size = load_ptb_slice(n_train=30_000, n_valid=20_000)
    train_data = batchify(train_ids, 10)       # 10 streams; 20-step windows -> 150 updates/epoch
    print(f"2-layer LSTM, 64 units; first {len(train_ids):,} training and {len(val_ids):,} validation words")
    print("SGD lr 1, gradient-norm clip 5, early stopping with patience 2, at most 10 epochs\n")
    
    configs = [
        ("No regularization", 1.0, 0.0),
        ("Dropout only", 0.5, 0.0),
        ("Dropout + weight decay", 0.5, 1e-4),
    ]
    
    results = []
    for name, keep_prob, decay in configs:
        model = LSTMLanguageModel(vocab_size, hidden_size=64, num_layers=2, keep_prob=keep_prob,
                                  init_scale=0.1, seed=0)   # same seed: same initial weights
        history = train_regularized_rnn(model, train_data, val_ids, learning_rate=1.0,
                                        max_epochs=10, patience=2, weight_decay=decay, bptt=20)
        val_ppl = np.exp(history['val_loss'])
        print(f"{name:24s} validation perplexity by epoch: {', '.join(f'{v:.0f}' for v in val_ppl)}")
        results.append((name, history))
    
    print("\nConfiguration          | Best Val PPL (epoch) | Epochs | Early Stop | Train PPL at best epoch*")
    print("-" * 88)
    for name, h in results:
        b = h['best_epoch'] - 1
        print(f"{name:22} | {np.exp(h['val_loss'][b]):12.1f} ({h['best_epoch']:2d})    | {h['epochs_trained']:6} | "
              f"{'Yes' if h['stopped_early'] else 'No':10} | {np.exp(h['train_loss'][b]):8.1f}")
    print("* measured while training, so with dropout on in the dropout runs (it reads higher than the")
    print("  same model evaluated with dropout off)")
    
    best = {name: min(h['val_loss']) for name, h in results}
    winner = min(best, key=best.get)
    print(f"\nLowest validation perplexity: {winner} ({np.exp(best[winner]):.1f}).")
    for name, h in results:
        if h['stopped_early']:
            print(f"{name}: validation stopped improving after epoch {h['best_epoch']}, so early stopping "
                  f"ended training at epoch {h['epochs_trained']} and restored the epoch-{h['best_epoch']} weights.")
    print("\nDropout follows the paper (non-recurrent connections only); weight decay and early")
    print("stopping are our additions. Layer norm (Part A) is not used here: the paper's model has none.")


if __name__ == "__main__":
    print("="*60)
    print("Solution 5: Full Regularization Pipeline")
    print("="*60)
    
    test_cell_components()
    test_full_pipeline()
    demonstrate_regularization_effects()
    
    print("\nAll tests passed!")
