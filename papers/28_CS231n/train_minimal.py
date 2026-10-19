"""
train_minimal.py - Train a Simple CNN on CIFAR-10

Trains the SimpleCNN from implementation.py on a CIFAR-10 subset with
mini-batch SGD + momentum, using backpropagation: the analytic gradients of
SimpleCNN.backward (conv backward via im2col/col2im, max-pool gradient
routing, ReLU, FC, softmax cross-entropy).
Downloads CIFAR-10 automatically if not already present (into data/ next to
this file; --data points elsewhere).

[PATCH 2026-10-18] This script used to train with central-difference
numerical gradients: 2 x 67,642 forward passes per SGD step, about 15 min per
step at batch size 1 on one busy CPU core (re-measured: 9 min wall, 7.3 min of
CPU), so no run could finish an epoch. Now a step costs one forward and one
backward pass (about 85 ms of CPU at batch 64). numerical_gradient() is kept
for what CS231n uses it for: checking the analytic gradient (--gradcheck).

Usage (times are CPU time on one core; a busy machine takes longer):
    python train_minimal.py                # 10,000 train / 1,000 test images, 6 epochs: ~1.5 min,
                                           #   test accuracy 10% -> 51%
    python train_minimal.py --long         # all 50,000 / 10,000 images, 5 epochs: ~6.5 min,
                                           #   test accuracy 10% -> 63% (needs ~1.5 GB of RAM)
    python train_minimal.py --gradcheck    # backprop vs numerical gradients on 4 images, then exit
    python train_minimal.py --epochs 10 --lr 0.005 --batch-size 32

Reference: CS231n Course Notes - https://cs231n.github.io/convolutional-networks/
           (training: https://cs231n.github.io/neural-networks-3/)
"""

import argparse
import os
import sys
import time
import pickle
import gzip
import urllib.request
import tarfile
import numpy as np

# Import our CNN implementation
from implementation import SimpleCNN, softmax, cross_entropy_loss, softmax_cross_entropy_backward


# ---------------------------------------------------------------------------
# CIFAR-10 Data Loading
# ---------------------------------------------------------------------------

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_cifar10():
    """Download CIFAR-10 if not already present."""
    tar_path = os.path.join(CIFAR10_DIR, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(CIFAR10_DIR, "cifar-10-batches-py")

    if os.path.exists(extract_dir):
        return extract_dir

    os.makedirs(CIFAR10_DIR, exist_ok=True)
    print(f"Downloading CIFAR-10 from {CIFAR10_URL}...")
    urllib.request.urlretrieve(CIFAR10_URL, tar_path)
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(CIFAR10_DIR)
    os.remove(tar_path)
    print("CIFAR-10 ready.")

    return extract_dir


def load_cifar10_batch(filepath):
    """Load a single CIFAR-10 batch file."""
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(batch[b'labels'])
    return images, labels


def load_cifar10(data_dir=None, max_train=5000, max_test=1000):
    """
    Load CIFAR-10 dataset.

    Uses a subset by default to keep training fast with our NumPy CNN.
    [PATCH 2026-10-18] With backprop, one epoch over all 50,000 training images
    takes about 75 s of CPU time on one core (see train_minimal.py --long).

    Args:
        data_dir: Path to cifar-10-batches-py directory
        max_train: Maximum number of training samples
        max_test: Maximum number of test samples

    Returns:
        X_train, y_train, X_test, y_test
    """
    if data_dir is None:
        data_dir = download_cifar10()

    # Load training batches
    train_images, train_labels = [], []
    for i in range(1, 6):
        if sum(len(lbls) for lbls in train_labels) >= max_train:
            break  # [PATCH 2026-10-18] each file holds 10,000 images: read only the files max_train needs
        filepath = os.path.join(data_dir, f"data_batch_{i}")
        if os.path.exists(filepath):
            imgs, lbls = load_cifar10_batch(filepath)
            train_images.append(imgs)
            train_labels.append(lbls)

    X_train = np.concatenate(train_images)[:max_train]
    y_train = np.concatenate(train_labels)[:max_train]
    del train_images  # [PATCH 2026-10-18] free the per-file copies (614 MB for all five files)

    # Load test batch
    test_path = os.path.join(data_dir, "test_batch")
    if os.path.exists(test_path):
        X_test, y_test = load_cifar10_batch(test_path)
        X_test = X_test[:max_test]
        y_test = y_test[:max_test]
    else:
        # [PATCH 2026-10-18] this fallback was silent: test accuracy would be measured on random noise
        print(f"WARNING: {test_path} not found -> test set is SYNTHETIC random data; "
              "test accuracy below is meaningless.", file=sys.stderr, flush=True)
        X_test = np.random.randn(max_test, 3, 32, 32).astype(np.float32)
        y_test = np.random.randint(0, 10, max_test)

    # Normalize: zero mean, unit variance per channel
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    # [PATCH 2026-10-18] in place (-=, /=) instead of (X - mean) / std: same values, no extra full-size copies
    X_train -= mean
    X_train /= std
    X_test -= mean
    X_test /= std

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def numerical_gradient(model, x, y, param_name, h=1e-5, indices=None):
    """
    Compute gradient numerically using finite differences.

    [PATCH 2026-10-18] A gradient-CHECKING tool, not a trainer. It uses the
    centred difference (f(w+h) - f(w-h)) / 2h, as CS231n recommends
    (neural-networks-3, "Gradient Checks"), which costs two forward passes per
    coordinate: 2 x 67,642 forward passes for all of SimpleCNN, which is why
    training with it took ~15 min per step. Compare it with the backprop
    gradient on a few coordinates (see gradient_check below).

    Args:
        model: SimpleCNN
        x, y: a (small) batch and its labels
        param_name: key of model.params, e.g. 'W1'
        h: step size (CS231n: "approximately 1e-5 or so")
        indices: [PATCH 2026-10-18] optional list of index tuples; only these
            coordinates are computed (the rest of the result stays 0)
    """
    param = model.params[param_name]
    grad = np.zeros_like(param)

    # np.ndindex(shape) yields every index tuple of an array of that shape: (0,0,0,0), (0,0,0,1), ...
    for idx in (np.ndindex(param.shape) if indices is None else indices):
        old_val = param[idx]

        # f(x + h)
        param[idx] = old_val + h
        scores_plus = model.forward(x)
        probs_plus = softmax(scores_plus)
        loss_plus = cross_entropy_loss(probs_plus, y)

        # f(x - h)
        param[idx] = old_val - h
        scores_minus = model.forward(x)
        probs_minus = softmax(scores_minus)
        loss_minus = cross_entropy_loss(probs_minus, y)

        # Gradient
        grad[idx] = (loss_plus - loss_minus) / (2 * h)

        # Restore
        param[idx] = old_val

    return grad


def gradient_check(model, x, y, num_checks=5, h=1e-5, seed=0):
    """
    [PATCH 2026-10-18] CS231n-style gradient check of SimpleCNN.backward.

    For every parameter tensor, compares the backprop gradient a with the
    centred-difference gradient n at `num_checks` random coordinates
    (CS231n neural-networks-3, "Gradient Checks": check a few dimensions of
    EVERY parameter, use few data points and double precision). The relative
    error is |a - n| / max(1e-8, |a| + |n|), the rel_error of the CS231n
    assignments: the 1e-8 floor keeps gradient entries that are ~0 (dead ReLUs)
    from turning round-off into a large ratio. CS231n's rule of thumb: < 1e-7
    is great; with kinks (ReLU, max-pool) up to ~1e-4 can still be fine,
    because a finite difference can step across a kink. The full check (every
    parameter and the input vs torch.autograd) is
    validation/day28_backprop_check.py.

    Returns:
        worst relative error over all checked coordinates
    """
    probs = softmax(model.forward(x))
    grads = model.backward(softmax_cross_entropy_backward(probs, y))
    rng = np.random.default_rng(seed)
    worst = 0.0
    print(f"Gradient check on {len(x)} images: backprop vs centred differences (h={h:g}), "
          f"{num_checks} random coordinates per parameter")
    for name, param in model.params.items():
        # num_checks random index tuples into this parameter, e.g. (k, c, fh, fw) for a conv weight
        idx = [tuple(int(rng.integers(n)) for n in param.shape) for _ in range(num_checks)]
        numeric = numerical_gradient(model, x, y, name, h=h, indices=idx)
        rel = [abs(grads[name][i] - numeric[i]) / max(1e-8, abs(grads[name][i]) + abs(numeric[i])) for i in idx]
        worst = max(worst, max(rel))
        print(f"  {name:3s} {str(param.shape):16s} max relative error {max(rel):.1e}")
    print(f"  worst: {worst:.1e}  (CS231n: < 1e-7 great; < 1e-4 usually fine with ReLU/max-pool kinks)")
    return worst


def evaluate(model, X, y, batch_size=500):
    """
    [PATCH 2026-10-18] Mean cross-entropy loss and accuracy of the model on (X, y).

    Runs in chunks of batch_size so the activations that forward() keeps for
    backprop stay small (about 0.5 MB per image), and does not change the weights.
    """
    total_loss, correct = 0.0, 0
    for start in range(0, len(X), batch_size):
        X_chunk, y_chunk = X[start:start + batch_size], y[start:start + batch_size]
        scores = model.forward(X_chunk)
        total_loss += cross_entropy_loss(softmax(scores), y_chunk) * len(y_chunk)  # mean -> sum over the chunk
        correct += np.sum(np.argmax(scores, axis=1) == y_chunk)
    model.cache = None  # drop the last chunk's activations
    return total_loss / len(X), correct / len(X)


def train(model, X_train, y_train, X_test, y_test,
          epochs=5, lr=0.01, batch_size=16, momentum=0.9, weight_decay=0.0,
          lr_decay=1.0, eval_train_size=1000, verbose=True):
    """
    Train the CNN with mini-batch SGD + momentum on backprop gradients.

    [PATCH 2026-10-18] Was: SGD on numerical gradients (2 forward passes per
    parameter per step). Each step is now CS231n's loop: forward pass ->
    softmax cross-entropy loss -> gradient on the scores -> backward pass ->
    parameter update (SimpleCNN.update: momentum, optional L2 weight decay).
    After every epoch the learning rate is multiplied by lr_decay (a per-epoch
    form of CS231n's "step decay", neural-networks-3).

    Reported per epoch (epoch 0 = before training):
      - batch loss / acc: mean over the epoch's mini-batches, measured on each
        batch just before its update (the weights change during the epoch);
      - train loss / acc: weights frozen at the end of the epoch, on the first
        eval_train_size training images;
      - test loss / acc: weights frozen, on all of X_test.
    Loss is the mean cross-entropy (chance level: ln 10 = 2.303, accuracy 10%).
    Before training it is well above 2.303 (3.6 to 7 for seeds 0-2): with He
    initialization the initial scores have a standard deviation of 2-3, so
    some classes dominate. CS231n's "look for correct loss at chance
    performance" check (2.302 for CIFAR-10) assumes small initial weights.

    Args:
        model: SimpleCNN instance
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Mini-batch size (the last partial batch of an epoch is skipped)
        momentum: SGD momentum mu (0 = plain SGD)
        weight_decay: L2 regularization strength on the weights (0 = none)
        lr_decay: multiply lr by this after every epoch (1 = constant)
        eval_train_size: how many training images to evaluate per epoch
        verbose: print the progress table

    Returns:
        history: list of dicts, one per epoch (epoch 0 = before training), with
            keys epoch, lr, batch_loss, batch_acc, train_loss, train_acc,
            test_loss, test_acc, seconds (cumulative training + evaluation time)
    """
    n_batches = max(1, len(X_train) // batch_size)
    X_eval, y_eval = X_train[:eval_train_size], y_train[:eval_train_size]

    if verbose:
        print("=" * 60)
        print("Training SimpleCNN on CIFAR-10 (backprop + SGD with momentum)")
        print("=" * 60)
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples:  {len(X_test)}")
        print(f"  Parameters:    {model.count_parameters():,}")
        print(f"  Epochs:        {epochs}")
        print(f"  Learning rate: {lr} (x{lr_decay} per epoch)")
        print(f"  Momentum:      {momentum}")
        print(f"  Weight decay:  {weight_decay}")
        print(f"  Batch size:    {batch_size} ({n_batches} steps per epoch)")
        print()
        print(f"  {'epoch':>5}  {'lr':>7} | {'batch loss':>10} {'acc':>6} | {'train loss':>10} {'acc':>6} | "
              f"{'test loss':>9} {'acc':>6} | {'time':>6}")

    history = []

    def log(record):
        history.append(record)
        if verbose:
            batch = (f"{record['batch_loss']:10.4f} {record['batch_acc']:6.1%}" if record['epoch'] > 0
                     else f"{'-':>10} {'-':>6}")
            print(f"  {record['epoch']:5d}  {record['lr']:7.4f} | {batch} | {record['train_loss']:10.4f} "
                  f"{record['train_acc']:6.1%} | {record['test_loss']:9.4f} {record['test_acc']:6.1%} | "
                  f"{record['seconds']:5.0f}s", flush=True)

    t0 = time.time()
    train_loss, train_acc = evaluate(model, X_eval, y_eval)
    test_loss, test_acc = evaluate(model, X_test, y_test)
    log(dict(epoch=0, lr=lr, batch_loss=None, batch_acc=None, train_loss=train_loss, train_acc=train_acc,
             test_loss=test_loss, test_acc=test_acc, seconds=time.time() - t0))

    for epoch in range(1, epochs + 1):
        # Shuffle training data (a fresh order every epoch)
        perm = np.random.permutation(len(X_train))

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(X_train))
            idx = perm[start:end]                       # indices of this mini-batch
            X_batch, y_batch = X_train[idx], y_train[idx]  # fancy indexing copies just these rows

            # Forward pass (SimpleCNN.forward keeps the intermediates backward needs)
            scores = model.forward(X_batch)
            probs = softmax(scores)
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss
            epoch_correct += np.sum(np.argmax(scores, axis=1) == y_batch)
            epoch_seen += len(y_batch)

            # Backward pass: dL/dscores = (p - onehot) / N, then backprop through FC, ReLU, pool and conv
            dscores = softmax_cross_entropy_backward(probs, y_batch)
            grads = model.backward(dscores)

            # Parameter update
            model.update(grads, lr, momentum=momentum, weight_decay=weight_decay)

        train_loss, train_acc = evaluate(model, X_eval, y_eval)
        test_loss, test_acc = evaluate(model, X_test, y_test)
        log(dict(epoch=epoch, lr=lr, batch_loss=epoch_loss / n_batches, batch_acc=epoch_correct / epoch_seen,
                 train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc,
                 seconds=time.time() - t0))
        lr *= lr_decay

    if verbose:
        first, last = history[0], history[-1]
        print(f"\nTraining complete in {last['seconds']:.0f} s.")
        print(f"  Test accuracy: {first['test_acc']:.1%} before training -> {last['test_acc']:.1%} after "
              f"{epochs} epochs (chance: 10%); test loss {first['test_loss']:.3f} -> {last['test_loss']:.3f}.")
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# [PATCH 2026-10-18] Two presets. The default fits in a few minutes on one CPU core; --long uses all of
# CIFAR-10 (still far less than a GPU run: no augmentation, a 67k-parameter net, a few epochs).
PRESETS = {
    'default': dict(epochs=6, max_train=10000, max_test=1000),
    'long': dict(epochs=5, max_train=50000, max_test=10000),
}


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple CNN on CIFAR-10 with backprop (Day 28)"
    )
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 6; 5 with --long)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, 0 = plain SGD (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='L2 weight decay on the weights (default: 5e-4)')
    parser.add_argument('--lr-decay', type=float, default=0.8,
                        help='Multiply the learning rate by this after every epoch (default: 0.8)')
    parser.add_argument('--max-train', type=int, default=None,
                        help='Max training samples (default: 10,000; 50,000 with --long)')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Max test samples (default: 1,000; 10,000 with --long)')
    parser.add_argument('--long', action='store_true',
                        help='Use all of CIFAR-10: 50,000 train / 10,000 test images, 5 epochs')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for the weight init and the batch order (default: 0)')
    parser.add_argument('--gradcheck', action='store_true',
                        help='Compare backprop with numerical gradients on 4 training images, then exit')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to CIFAR-10 data directory')

    args = parser.parse_args()
    preset = PRESETS['long' if args.long else 'default']
    for key, value in preset.items():          # flags given on the command line win over the preset
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Load data
    try:
        X_train, y_train, X_test, y_test = load_cifar10(
            data_dir=args.data,
            max_train=args.max_train,
            max_test=args.max_test
        )
    except Exception as e:
        print(f"Could not load CIFAR-10: {e}")
        print("Generating synthetic data for demonstration...")
        # [PATCH 2026-10-18] make the fallback impossible to miss (random images + random labels, chance = 10%)
        print("\n" + "!" * 72 + "\nWARNING: CIFAR-10 NOT AVAILABLE -> TRAINING ON SYNTHETIC RANDOM DATA.\n"
              "Loss/accuracy below say nothing about the CNN.\n" + "!" * 72 + "\n", file=sys.stderr, flush=True)
        X_train = np.random.randn(args.max_train, 3, 32, 32).astype(np.float32)
        y_train = np.random.randint(0, 10, args.max_train)
        X_test = np.random.randn(args.max_test, 3, 32, 32).astype(np.float32)
        y_test = np.random.randint(0, 10, args.max_test)

    # Create and train model
    np.random.seed(args.seed)  # [PATCH 2026-10-18] reproducible init and batch order
    model = SimpleCNN(num_classes=10)
    if args.gradcheck:
        gradient_check(model, X_train[:4], y_train[:4])
        return
    train(model, X_train, y_train, X_test, y_test,
          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, momentum=args.momentum,
          weight_decay=args.weight_decay, lr_decay=args.lr_decay)


if __name__ == "__main__":
    main()
