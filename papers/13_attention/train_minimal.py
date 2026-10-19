"""
Day 13: Minimal Transformer Training Script

Train a small Transformer on a simple sequence task (copy or reverse), in pure NumPy.

[PATCH 2026-10-18] This script now really trains. Before, it only ran forward
passes (no backward pass existed and the SGD class was never used), so the loss
stayed at ln(vocab_size) and every prediction was the same token. Now:
  - implementation.py backpropagates through every layer (checked against
    torch.autograd and finite differences: validation/day13_backprop_check.py);
  - the loss is softmax cross-entropy over non-padding targets, with optional
    label smoothing (Sec. 5.4);
  - the optimizer is Adam with beta1 = 0.9, beta2 = 0.98, eps = 1e-9 and the
    warm-up learning-rate schedule of Eq. 3 (Sec. 5.3).

Paper vs this script (CPU budget): the paper's base model has d_model = 512,
6 + 6 layers, 8 heads, warmup_steps = 4000, 100k steps on WMT translation (8 GPUs,
12 hours). Here: d_model = 64, 2 + 2 layers, 4 heads, a synthetic task, and
warmup_steps = 800, because with 4000 a run of ~2k steps would end before the
learning rate even peaks. (After the warm-up, Eq. 3 gives d_model^-0.5 * step^-0.5
whatever warmup_steps is.)

Default run (python train_minimal.py, copy task): ~53 s of CPU time on one core
(2-3 minutes of wall time on a busy shared machine); README.md ("Training It in
NumPy") has the numbers.
"""

import argparse
import time
from typing import Tuple, List, Dict, Optional

import numpy as np

from implementation import Transformer, create_causal_mask, create_padding_mask


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_copy_task_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int = 10,
    pad_token: int = 0,
    start_token: int = 1,
    end_token: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for a copy task.
    
    The model learns to copy the input sequence to the output.
    Example: Input [1, 5, 3, 7, 2] -> Output [1, 5, 3, 7, 2]
    
    This is a fundamental test for sequence-to-sequence models.
    """
    # Generate random sequences (tokens 3 to vocab_size-1)
    src = np.random.randint(3, vocab_size, (n_samples, seq_len))
    
    # Target is the same sequence with start token prepended
    tgt = np.zeros((n_samples, seq_len + 1), dtype=np.int32)
    tgt[:, 0] = start_token
    tgt[:, 1:] = src
    
    return src, tgt


def generate_reverse_task_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int = 10,
    start_token: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for a reverse task.
    
    The model learns to reverse the input sequence.
    Example: Input [5, 3, 7] -> Output [7, 3, 5]
    """
    src = np.random.randint(3, vocab_size, (n_samples, seq_len))
    
    tgt = np.zeros((n_samples, seq_len + 1), dtype=np.int32)
    tgt[:, 0] = start_token
    tgt[:, 1:] = src[:, ::-1]  # Reverse
    
    return src, tgt


# =============================================================================
# LOSS AND METRICS
# =============================================================================

# [PATCH 2026-10-18] Loss with its gradient (the backward pass starts here).
def cross_entropy_loss_and_grad(logits: np.ndarray, targets: np.ndarray,
                                ignore_index: int = -100,
                                label_smoothing: float = 0.0) -> Tuple[float, np.ndarray]:
    """
    Softmax cross-entropy averaged over the non-ignored positions, and dL/dlogits.

    Label smoothing (Sec. 5.4; eps_ls = 0.1 in the paper, after Szegedy et al. 2016):
    the one-hot target is replaced by
        q = (1 - eps) * one_hot(y) + eps / V        (V = vocabulary size)
    and the loss is the cross-entropy H(q, p) = -sum_k q_k log p_k. "Smoothing"
    because the spike at the true token is flattened a little: the model is never
    asked for probability 1, which (the paper reports) hurts perplexity but helps
    accuracy and BLEU. eps = 0 gives the usual -log p_y.

    Gradient: for softmax followed by cross-entropy the two Jacobians collapse to
        dL/dlogits = (p - q) / N
    at the N counted positions and 0 at ignored ones (padding), which is why
    frameworks fuse the two ops.

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len) integer token ids
        ignore_index: target id that is not counted (e.g. the padding token)
        label_smoothing: eps in [0, 1)
    Returns:
        (loss, dlogits); dlogits has the shape of logits
    """
    batch, seq_len, vocab_size = logits.shape
    # log-softmax with the max trick: log p = z - log(sum(exp(z))), z = logits - max.
    # keepdims=True leaves the reduced axis as length 1 so it broadcasts back over V.
    z = logits - logits.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))     # (B, T, V)

    valid = targets != ignore_index                                    # (B, T) bool, True = counted
    n_valid = max(int(valid.sum()), 1)
    safe_targets = np.where(valid, targets, 0)       # ignored slots point at class 0; zeroed by `valid` below

    # q: smoothed target distribution, (B, T, V). np.full fills every entry with eps/V;
    # np.put_along_axis then writes 1 - eps + eps/V at index safe_targets[b, t] along the
    # last axis (the index array needs the same number of dims, hence [..., None]: (B, T, 1)).
    q = np.full(logits.shape, label_smoothing / vocab_size)
    np.put_along_axis(q, safe_targets[..., None], 1.0 - label_smoothing + label_smoothing / vocab_size, axis=-1)

    loss_per_position = -(q * log_probs).sum(axis=-1)                  # (B, T)
    loss = float((loss_per_position * valid).sum() / n_valid)
    # valid[..., None] is (B, T, 1): broadcasting zeroes every logit of an ignored position
    dlogits = (np.exp(log_probs) - q) * valid[..., None] / n_valid
    return loss, dlogits


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray, 
                       ignore_index: int = -100) -> float:
    """
    Cross-entropy loss for sequence prediction.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        ignore_index: Token to ignore in loss computation
        
    Returns:
        Average loss

    # [PATCH 2026-10-18] Now computed by cross_entropy_loss_and_grad (exact
    # log-softmax; the old version clipped probabilities at 1e-10), so the loss
    # reported is exactly the one being minimised (when label smoothing is off).
    """
    return cross_entropy_loss_and_grad(logits, targets, ignore_index)[0]


def compute_accuracy(logits: np.ndarray, targets: np.ndarray,
                    ignore_index: int = -100) -> float:
    """Compute token-level accuracy."""
    predictions = logits.argmax(axis=-1)
    
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    
    return correct.sum() / mask.sum() if mask.sum() > 0 else 0.0


# =============================================================================
# OPTIMIZER: Adam with the paper's warm-up schedule (Sec. 5.3)
# =============================================================================

# [PATCH 2026-10-18] Replaces the unused SGD class.
def get_lr(step: int, d_model: int, warmup: int = 4000) -> float:
    """
    Eq. 3 (Sec. 5.3): lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5).
    
    "Warm-up": like warming an engine, the rate rises linearly for `warmup` steps
    (second branch), then decays as 1/sqrt(step) (first branch). The branches meet
    at step = warmup, where the rate peaks at (d_model * warmup)^-0.5. Early steps
    are small because Adam's moment estimates are still noisy and a post-norm
    Transformer's initial gradients near the output are large (Xiong et al. 2020).
    Often called the "Noam" schedule, after Noam Shazeer (one of the authors).
    `step` counts from 1.
    """
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


class Adam:
    """
    Adam (Kingma & Ba 2015) with the paper's settings (Sec. 5.3): beta1 = 0.9,
    beta2 = 0.98, eps = 1e-9, learning rate from get_lr (Eq. 3) unless a constant
    `lr` is given.

    The name is short for "adaptive moment estimation": per parameter entry it
    keeps a running average of the gradient (1st moment, m) and of the squared
    gradient (2nd moment, v), and steps by m / sqrt(v), so every entry gets its own
    effective step size. beta2 = 0.98 (the default elsewhere is 0.999) makes v
    adapt faster.

    Usage: opt = Adam(model.parameters(), d_model=64, warmup=800)
           ... model.backward(dlogits); opt.step(model.gradients())
    """

    def __init__(self, params: List[np.ndarray], d_model: Optional[int] = None,
                 warmup: int = 4000, lr: Optional[float] = None,
                 betas: Tuple[float, float] = (0.9, 0.98), eps: float = 1e-9):
        if lr is None and d_model is None:
            raise ValueError("give d_model (for the Eq. 3 schedule) or a constant lr")
        self.params = params            # the model's own arrays, updated in place
        self.d_model = d_model
        self.warmup = warmup
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]   # 1st-moment estimates, one per parameter
        self.v = [np.zeros_like(p) for p in params]   # 2nd-moment estimates
        self.t = 0                                     # number of steps taken
        
    def current_lr(self) -> float:
        """Learning rate of the next step."""
        if self.lr is not None:
            return self.lr
        return get_lr(self.t + 1, self.d_model, self.warmup)
    
    def step(self, grads: List[np.ndarray]) -> float:
        """One Adam update of every parameter; returns the learning rate used."""
        lr = self.current_lr()
        self.t += 1
        beta1, beta2 = self.betas
        for param, grad, m, v in zip(self.params, grads, self.m, self.v):
            # In-place updates (*=, +=, -=) modify the existing arrays. That matters for
            # `param`: the model's layers hold references to these very arrays, so
            # `param = param - ...` would create a new array the model never sees.
            m *= beta1
            m += (1 - beta1) * grad
            v *= beta2
            v += (1 - beta2) * grad * grad
            # bias correction: m and v start at 0, so early averages are too small
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)   # elementwise: each entry its own step
        return lr


# =============================================================================
# SIMPLE TRANSFORMER FOR TRAINING
# =============================================================================

class SimpleTransformer:
    """
    A simplified Transformer for training demos.

    # [PATCH 2026-10-18] Was: "Uses numerical gradients for simplicity". It now
    # trains with the analytic backward pass of implementation.py: forward()
    # builds the masks and returns logits, backward() backpropagates dL/dlogits,
    # parameters()/gradients() feed the optimizer.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 256, dropout_p: float = 0.1,
                 share_embeddings: bool = False, pad_token: int = 0):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.pad_token = pad_token
        self.training = True
        
        # Create full transformer
        self.model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_layers,
            n_decoder_layers=n_layers,
            d_ff=d_ff,
            dropout_p=dropout_p,
            share_embeddings=share_embeddings
        )
    
    def forward(self, src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Create causal mask for decoder
        tgt_mask = create_causal_mask(tgt.shape[1])
        # [PATCH 2026-10-18] Also mask padded source tokens (used as keys by the encoder
        # self-attention and the decoder cross-attention). The copy/reverse data has no
        # padding, so this changes nothing there, but variable-length data needs it.
        src_mask = create_padding_mask(src, self.pad_token)
        
        return self.model.forward(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

    # [PATCH 2026-10-18] backward pass and parameter access, delegated to the Transformer
    def backward(self, grad_logits: np.ndarray) -> None:
        """Backpropagate dL/dlogits of the last forward() call (fills the gradients)."""
        self.model.backward(grad_logits)

    def parameters(self) -> List[np.ndarray]:
        return self.model.parameters()

    def gradients(self) -> List[np.ndarray]:
        return self.model.gradients()
    
    def train(self):
        self.training = True
        self.model.train()
    
    def eval(self):
        self.training = False
        self.model.eval()


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: SimpleTransformer,
    src_data: np.ndarray,
    tgt_data: np.ndarray,
    optimizer: Adam,
    batch_size: int = 32,
    label_smoothing: float = 0.0,
    verbose: bool = False
) -> Dict[str, float]:
    """
    One epoch of training: for each mini-batch, forward -> loss -> backward -> Adam step.

    # [PATCH 2026-10-18] Real training (was forward passes only; its `lr`
    # argument was ignored). Takes the optimizer instead, so Adam's moments and
    # step count (hence the warm-up schedule) carry over from epoch to epoch.

    Returns the mean training objective (cross-entropy, label-smoothed if
    label_smoothing > 0), the teacher-forced token accuracy, measured with dropout
    on and while the weights change, and the last learning rate.
    """
    model.train()
    
    n_samples = src_data.shape[0]
    n_batches = n_samples // batch_size
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    lr = 0.0
    
    indices = np.random.permutation(n_samples)
    
    for i in range(n_batches):
        batch_idx = indices[i * batch_size:(i + 1) * batch_size]
        src_batch = src_data[batch_idx]
        tgt_batch = tgt_data[batch_idx]
        
        # Teacher forcing: input is target[:-1], predict target[1:]
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        
        # Forward pass
        logits = model.forward(src_batch, tgt_input)
        
        # Loss and its gradient; padding targets (if any) are not counted
        loss, dlogits = cross_entropy_loss_and_grad(
            logits, tgt_output, ignore_index=model.pad_token, label_smoothing=label_smoothing)
        acc = compute_accuracy(logits, tgt_output, ignore_index=model.pad_token)

        # Backward pass and parameter update
        model.backward(dlogits)
        lr = optimizer.step(model.gradients())
        
        epoch_loss += loss
        epoch_acc += acc
        
        if verbose and (i + 1) % max(1, n_batches // 5) == 0:
            print(f"    Batch {i+1}/{n_batches}: Loss={loss:.4f}, Acc={acc:.4f}, lr={lr:.2e}")
    
    return {
        'loss': float(epoch_loss / n_batches),
        'accuracy': float(epoch_acc / n_batches),
        'lr': lr
    }


def evaluate(
    model: SimpleTransformer,
    src_data: np.ndarray,
    tgt_data: np.ndarray,
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Teacher-forced evaluation (eval mode, no dropout): mean cross-entropy (no label
    smoothing) and token accuracy over all target tokens.

    # [PATCH 2026-10-18] Counts every sample (the last partial batch used to be
    # dropped) and weights batches by their number of target tokens.
    """
    model.eval()
    
    n_samples = src_data.shape[0]
    
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0
    
    for start in range(0, n_samples, batch_size):
        src_batch = src_data[start:start + batch_size]
        tgt_batch = tgt_data[start:start + batch_size]
        
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]
        
        logits = model.forward(src_batch, tgt_input)
        n_tokens = int((tgt_output != model.pad_token).sum())
        
        total_loss += cross_entropy_loss(logits, tgt_output, ignore_index=model.pad_token) * n_tokens
        total_correct += compute_accuracy(logits, tgt_output, ignore_index=model.pad_token) * n_tokens
        total_tokens += n_tokens
    
    return {
        'loss': float(total_loss / total_tokens),
        'accuracy': float(total_correct / total_tokens)
    }


# [PATCH 2026-10-18] Exact-sequence accuracy needs free-running decoding.
def evaluate_generation(model: SimpleTransformer, src_data: np.ndarray, tgt_data: np.ndarray,
                        start_token: int = 1) -> Dict[str, float]:
    """
    Greedy (free-running) decoding of every source sequence, compared with the target.

    Unlike teacher forcing, the model consumes its own predictions, so one early
    mistake can derail the rest. Returns the fraction of sequences reproduced
    exactly and the fraction of target tokens predicted correctly.
    """
    seq_len = tgt_data.shape[1] - 1                      # targets are [start, y_1 .. y_L]
    pred = greedy_decode(model, src_data, max_len=seq_len + 1, start_token=start_token)
    pred = pred[:, 1:seq_len + 1]                        # drop the start token
    if pred.shape[1] < seq_len:                          # stopped early on an end token: pad
        pred = np.pad(pred, ((0, 0), (0, seq_len - pred.shape[1])), constant_values=-1)
    correct = pred == tgt_data[:, 1:]                    # (N, L) bool
    return {
        'exact_match': float(correct.all(axis=1).mean()),   # .all(axis=1): every token of a row right
        'token_accuracy': float(correct.mean())
    }


# =============================================================================
# ATTENTION VISUALIZATION
# =============================================================================

def visualize_attention(model: SimpleTransformer, src: np.ndarray, tgt: np.ndarray):
    """
    Print attention patterns (text-based).

    # [PATCH 2026-10-18] Also shows the cross-attention of the FIRST decoder layer
    # (averaged over heads). In our runs it becomes the alignment: target step t
    # puts ~99% of its weight on source position t (a diagonal; on the reverse
    # task, the anti-diagonal). The second layer's cross-attention ends up diffuse:
    # once layer 0 has fetched the token, nothing more needs reading.
    """
    model.eval()
    
    # Single sample
    src = src[0:1]
    tgt = tgt[0:1, :-1]  # Remove last token
    
    _ = model.forward(src, tgt)
    
    # Get decoder self-attention weights
    decoder_attn = model.model.decoder_layers[0].self_attention.get_attention_weights()
    
    if decoder_attn is not None:
        print("\nDecoder Self-Attention (layer 0, head 0; rows = target positions):")
        print("-" * 40)
        
        attn = decoder_attn[0, 0]  # First sample, first head
        seq_len = min(attn.shape[0], 8)
        
        # Print header
        header = "    " + " ".join(f"{i:5d}" for i in range(seq_len))
        print(header)
        
        for i in range(seq_len):
            row = f"{i:3d} " + " ".join(f"{attn[i, j]:5.2f}" for j in range(seq_len))
            print(row)

    cross_attn = model.model.decoder_layers[0].cross_attention.get_attention_weights()
    if cross_attn is not None:
        print("\nDecoder Cross-Attention (layer 0, mean over heads; rows = target steps,")
        print("columns = source positions):")
        print("-" * 40)
        attn = cross_attn[0].mean(axis=0)       # (heads, T, S) -> (T, S): average the heads
        n_rows, n_cols = min(attn.shape[0], 8), min(attn.shape[1], 8)
        print("    " + " ".join(f"{j:5d}" for j in range(n_cols)))
        for i in range(n_rows):
            print(f"{i:3d} " + " ".join(f"{attn[i, j]:5.2f}" for j in range(n_cols)))


# =============================================================================
# GREEDY DECODING
# =============================================================================

def greedy_decode(
    model: SimpleTransformer,
    src: np.ndarray,
    max_len: int = 20,
    start_token: int = 1,
    end_token: int = 2
) -> np.ndarray:
    """
    Greedy decoding for sequence generation.
    
    At each step, pick the most likely next token.
    """
    model.eval()
    
    batch_size = src.shape[0]
    
    # Start with just the start token
    tgt = np.full((batch_size, 1), start_token, dtype=np.int32)
    
    for _ in range(max_len - 1):
        logits = model.forward(src, tgt)
        
        # Get prediction for last position
        next_token = logits[:, -1, :].argmax(axis=-1, keepdims=True)
        
        # Append to target
        tgt = np.concatenate([tgt, next_token], axis=1)
        
        # Check for end token
        if (next_token == end_token).all():
            break
    
    return tgt


# =============================================================================
# MAIN
# =============================================================================

def report(name: str, model: SimpleTransformer, src: np.ndarray, tgt: np.ndarray,
           max_generate: int = 500) -> Dict[str, float]:
    """
    [PATCH 2026-10-18] Teacher-forced loss/accuracy on all sequences plus greedy-decoding
    accuracy on the first `max_generate` of them (decoding is the slow part), on one line.
    """
    tf = evaluate(model, src, tgt, batch_size=256)
    n_gen = min(max_generate, len(src))
    gen = evaluate_generation(model, src[:n_gen], tgt[:n_gen])
    print(f"  {name:5s}: loss {tf['loss']:.4f} | token acc {tf['accuracy']:.4f} (teacher forcing), "
          f"{gen['token_accuracy']:.4f} (greedy) | exact sequences {gen['exact_match']:.4f}"
          f"{'' if n_gen == len(src) else f' (greedy on first {n_gen})'}")
    return {'loss': tf['loss'], 'token_acc': tf['accuracy'],
            'greedy_token_acc': gen['token_accuracy'], 'seq_acc': gen['exact_match']}


def main():
    parser = argparse.ArgumentParser(description="Train a NumPy Transformer on a synthetic task")
    parser.add_argument("--task", type=str, default="copy", choices=["copy", "reverse"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=12)
    # [PATCH 2026-10-18] training settings (Sec. 5.3-5.4); --lr used to be ignored
    parser.add_argument("--n_train", type=int, default=2000, help="training sequences")
    parser.add_argument("--n_test", type=int, default=200, help="held-out test sequences")
    parser.add_argument("--dropout", type=float, default=0.1, help="P_drop (paper: 0.1)")
    parser.add_argument("--warmup", type=int, default=800,
                        help="warm-up steps of Eq. 3 (paper: 4000 for a 100k-step run)")
    parser.add_argument("--lr", type=float, default=None,
                        help="constant learning rate instead of the Eq. 3 schedule")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="eps_ls (paper: 0.1; 0 here because the toy targets are unambiguous)")
    parser.add_argument("--share_embeddings", action="store_true",
                        help="tie source/target embeddings and the output projection (Sec. 3.4)")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"\nTask: {args.task}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Model: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}"
          f"{', tied embeddings' if args.share_embeddings else ''}")
    
    # Generate data
    print("\nGenerating data...")
    
    make_data = generate_copy_task_data if args.task == "copy" else generate_reverse_task_data
    train_src, train_tgt = make_data(n_samples=args.n_train, seq_len=args.seq_len, vocab_size=args.vocab_size)
    test_src, test_tgt = make_data(n_samples=args.n_test, seq_len=args.seq_len, vocab_size=args.vocab_size)
    
    print(f"Train samples: {len(train_src)}")
    print(f"Test samples: {len(test_src)}")
    print(f"Sample input:  {train_src[0]}")
    print(f"Sample target: {train_tgt[0]}")
    
    # Create model
    print("\nCreating model...")
    model = SimpleTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout_p=args.dropout,
        share_embeddings=args.share_embeddings
    )
    n_params = sum(p.size for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # [PATCH 2026-10-18] Adam + Eq. 3 schedule (or a constant --lr)
    optimizer = Adam(model.parameters(), d_model=args.d_model, warmup=args.warmup, lr=args.lr)
    steps_per_epoch = args.n_train // args.batch_size
    if args.lr is None:
        peak = get_lr(args.warmup, args.d_model, args.warmup)
        print(f"Optimizer: Adam(0.9, 0.98, 1e-9), Eq. 3 schedule: warm-up {args.warmup} steps, "
              f"peak lr {peak:.2e}")
    else:
        print(f"Optimizer: Adam(0.9, 0.98, 1e-9), constant lr {args.lr:g}")
    print(f"Steps: {steps_per_epoch} per epoch x {args.epochs} epochs = {steps_per_epoch * args.epochs}")
    if args.label_smoothing > 0:
        print(f"Label smoothing eps_ls = {args.label_smoothing}: the training objective below "
              "includes it; the evaluation loss does not.")

    print("\nBefore training:")
    before = {'train': report("train", model, train_src, train_tgt),
              'test': report("test", model, test_src, test_tgt)}
    
    # Training loop
    print("\nTraining...")
    print("-" * 40)
    t0 = time.time()
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, train_src, train_tgt, optimizer,
            batch_size=args.batch_size,
            label_smoothing=args.label_smoothing
        )
        
        test_metrics = evaluate(model, test_src, test_tgt, batch_size=256)
        
        print(f"Epoch {epoch + 1:3d}/{args.epochs}: train objective {train_metrics['loss']:.4f}, "
              f"acc {train_metrics['accuracy']:.4f} | test loss {test_metrics['loss']:.4f}, "
              f"acc {test_metrics['accuracy']:.4f} | lr {train_metrics['lr']:.2e} "
              f"| {time.time() - t0:5.1f} s")
    train_time = time.time() - t0
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    print("After training:")
    after = {'train': report("train", model, train_src, train_tgt),
             'test': report("test", model, test_src, test_tgt)}
    print(f"Training time: {train_time:.1f} s ({optimizer.t} Adam steps)")
    
    # Show some predictions
    print("\nSample Predictions:")
    print("-" * 40)
    
    for i in range(min(3, len(test_src))):
        src = test_src[i:i+1]
        tgt = test_tgt[i:i+1]
        
        pred = greedy_decode(model, src, max_len=args.seq_len + 1)
        
        print(f"  Input:    {src[0]}")
        print(f"  Expected: {tgt[0]}")
        print(f"  Predicted: {pred[0]}")
        print()
    
    # Show attention pattern
    visualize_attention(model, test_src[:1], test_tgt[:1])
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    return {'before': before, 'after': after, 'train_time_s': train_time, 'steps': optimizer.t}


if __name__ == "__main__":
    main()
