"""
Solution 4: Convex Hull with Pointer Networks
==============================================

Implementation of a geometry-aware Pointer Network.

[PATCH 2026-10-18] Targets are the hull vertices in one canonical order (leftmost point first,
then counter-clockwise), then a pointer to the STOP slot, then IGNORE_INDEX padding. The old
labels repeated the first vertex as padding; the decoder had already masked it, so the loss
was infinite. The convex hull is a Pointer Networks experiment (Day 18), not in Order Matters.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial import ConvexHull as ScipyConvexHull
import os
import sys
# [PATCH 2026-10-18] was sys.path.append('..'), which resolves against the *current
# directory*: run from the day folder (or from solutions/) the import of
# implementation.py failed with ModuleNotFoundError. Resolve relative to this file.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from implementation import ReadProcessWrite, IGNORE_INDEX, compute_loss


class ConvexHullDataset(Dataset):
    def __init__(self, num_samples, num_points):
        self.num_samples = num_samples
        self.num_points = num_points
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        points = torch.rand(self.num_points, 2)
        p = points.numpy()
        vertices = ScipyConvexHull(p).vertices      # (k,) hull indices, counter-clockwise (2-D)
        start = np.argmin(p[vertices, 0])           # where the leftmost vertex sits in that cycle
        vertices = np.roll(vertices, -start)        # rotate the cycle to start there
        target = torch.full((self.num_points + 1,), IGNORE_INDEX, dtype=torch.long)
        target[:len(vertices)] = torch.from_numpy(vertices)
        target[len(vertices)] = self.num_points     # STOP slot
        return points, target


def train_convex_hull(num_points=10, num_epochs=8, num_samples=1000, hidden_dim=128, seed=0):
    """Train on fresh random hulls; return the mean teacher-forced loss of each epoch and the
    greedy exact-hull accuracy on 500 held-out sets."""
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # [PATCH 2026-10-18] stop_token=True adds the STOP slot (index num_points); was use_set_encoder=True
    model = ReadProcessWrite(input_dim=2, hidden_dim=hidden_dim, process_steps=5, glimpses=1,
                             stop_token=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = num_points + 1                           # at most num_points vertices, then STOP
    
    history = []
    for epoch in range(num_epochs):
        model.train()
        loader = DataLoader(ConvexHullDataset(num_samples, num_points), batch_size=32, shuffle=True)
        total = 0.0
        for points, targets in loader:
            points, targets = points.to(device), targets.to(device)
            lengths = torch.full((points.size(0),), num_points, device=device)
            _, log_probs, _ = model(points, lengths, steps, teacher_forcing=targets)
            
            # [PATCH 2026-10-18] mean NLL over the steps up to and including STOP
            loss = compute_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += loss.item() / len(loader)
        history.append(total)
        print(f"Epoch {epoch + 1:2d} | mean teacher-forced loss {total:.4f}")

    model.eval()
    points, targets = next(iter(DataLoader(ConvexHullDataset(500, num_points), batch_size=500)))
    with torch.no_grad():
        pointers, _, _ = model(points.to(device), torch.full((500,), num_points, device=device), steps)
    # exact hull: every step up to and including STOP matches (IGNORE steps don't count)
    exact = ((pointers.cpu() == targets) | (targets == IGNORE_INDEX)).all(dim=1).float().mean().item()
    print(f"Greedy exact hulls on 500 held-out sets of {num_points} points: {exact:.1%}")
    return history, exact

if __name__ == "__main__":
    train_convex_hull()
