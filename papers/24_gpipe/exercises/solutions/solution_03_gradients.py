"""
Solution 03: Gradient Accumulation

Accumulating gradients manually across micro-batches.
"""

import torch
import torch.nn as nn

def compute_accumulated_gradients(model, micro_batches, targets, criterion):
    # [PATCH 2026-10-18] unscaled per-micro-batch mean losses summed to M x the full-batch
    # gradient (an M-times larger SGD step). Weight each micro-batch by its share n_m / N.
    n_total = sum(x.shape[0] for x in micro_batches)
    for x_small, y_small in zip(micro_batches, targets):
        out = model(x_small)
        loss = criterion(out, y_small)
        # MSELoss('mean') averages over the micro-batch; weighting by n_m / N makes the
        # accumulated gradient equal the full mini-batch gradient exactly.
        (loss * (x_small.shape[0] / n_total)).backward()

def test_accum():
    model = nn.Linear(5, 1)
    micro_batches = [torch.randn(4, 5), torch.randn(4, 5)]
    targets = [torch.randn(4, 1), torch.randn(4, 1)]
    criterion = nn.MSELoss()
    
    compute_accumulated_gradients(model, micro_batches, targets, criterion)
    
    for p in model.parameters():
        assert p.grad is not None
    print("[OK] Solution 03 verified.")

if __name__ == "__main__":
    test_accum()
