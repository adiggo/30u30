"""
Training script for Pointer Networks.
=====================================

Tasks supported:
- sorting: numerical sorting
- convex_hull: boundary point identification
- tsp: traveling salesman problem approximation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import ConvexHull as ScipyConvexHull

from implementation import ReadProcessWrite, compute_loss, accuracy


class SortingDataset(Dataset):
    def __init__(self, num_samples: int, set_size: int, min_val: float = 0.0, max_val: float = 10.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.min_val = min_val
        self.max_val = max_val
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        values = torch.rand(self.set_size) * (self.max_val - self.min_val) + self.min_val
        _, sorted_indices = torch.sort(values)
        return values.unsqueeze(1), sorted_indices


class ConvexHullDataset(Dataset):
    def __init__(self, num_samples: int, set_size: int):
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        points = torch.rand(self.set_size, 2)
        try:
            hull = ScipyConvexHull(points.numpy())
            hull_indices = hull.vertices
            if len(hull_indices) < self.set_size:
                padding = [hull_indices[0]] * (self.set_size - len(hull_indices))
                hull_indices = np.concatenate([hull_indices, padding])
            return points, torch.from_numpy(hull_indices).long()
        except Exception:
            return points, torch.argsort(points[:, 0])


class TSPDataset(Dataset):
    def __init__(self, num_samples: int, num_cities: int):
        self.num_samples = num_samples
        self.num_cities = num_cities
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        cities = torch.rand(self.num_cities, 2)
        return cities, self._greedy_tour(cities)
    
    def _greedy_tour(self, cities):
        n = len(cities)
        unvisited = set(range(n))
        current = 0
        tour = [current]
        unvisited.remove(current)
        while unvisited:
            nearest = min(unvisited, key=lambda c: torch.norm(cities[current] - cities[c]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return torch.tensor(tour, dtype=torch.long)


def train_epoch(model, dataloader, optimizer, device, use_teacher_forcing=True):
    model.train()
    total_loss, total_acc, num_batches = 0, 0, 0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, set_size = inputs.size(0), inputs.size(1)
        lengths = torch.full((batch_size,), set_size, device=device)
        
        pointers, log_probs, attn = model(
            inputs, lengths, max_steps=set_size,
            teacher_forcing=targets if use_teacher_forcing else None
        )

        loss = compute_loss(log_probs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        total_loss += loss.item()
        # [PATCH 2026-10-18] Under teacher forcing `pointers` ARE the targets, so
        # accuracy(pointers, targets) was always 1.0. Score the model's own
        # per-step argmax instead (teacher-forced next-pointer accuracy).
        predicted = torch.stack([a.argmax(dim=-1) for a in attn], dim=1)
        total_acc += accuracy(predicted, targets)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_acc, num_batches = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            set_size = inputs.size(1)
            lengths = torch.full((inputs.size(0),), set_size, device=device)
            pointers, _, _ = model(inputs, lengths, max_steps=set_size)
            # [PATCH 2026-10-18] The greedy pass returns log p(argmax pointer), so
            # the old loss measured the model's confidence in its own choices, not
            # the NLL of the targets. Score the targets with a teacher-forced pass.
            _, log_probs, _ = model(inputs, lengths, max_steps=set_size, teacher_forcing=targets)
            total_loss += compute_loss(log_probs, targets).item()
            total_acc += accuracy(pointers, targets)
            num_batches += 1
    return total_loss / num_batches, total_acc / num_batches


def main():
    # [PATCH 2026-10-18] train.py had no main(): `python train.py --task sort ...`
    # (README, CHEATSHEET) and the `train-pointer=train:main` entry point in
    # setup.py just printed "Training script loaded." and exited 0 without
    # training anything. Minimal CLI wiring of the functions above.
    parser = argparse.ArgumentParser(description="Day 16: train a pointer network on a set task")
    parser.add_argument("--task", choices=["sort", "tsp", "hull"], default="sort")
    parser.add_argument("--set-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.task == "hull":
        # ConvexHullDataset pads each hull with repeats of its first index, which the
        # decoder has already masked out: teacher-forced log-prob -inf, loss inf.
        raise SystemExit("--task hull is not trainable: padded hull labels repeat an already-"
                         "selected index (loss = inf); the decoder needs an end-of-hull token.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "sort":
        train_set, val_set, input_dim = SortingDataset(args.num_samples, args.set_size), SortingDataset(1000, args.set_size), 1
    else:
        train_set, val_set, input_dim = TSPDataset(args.num_samples, args.set_size), TSPDataset(500, args.set_size), 2
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=100)
    model = ReadProcessWrite(input_dim, args.hidden_dim, use_set_encoder=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print(f"task={args.task} set_size={args.set_size} params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | train NLL {tr_loss:.4f} | teacher-forced acc {tr_acc:.3f} | "
              f"val NLL {va_loss:.4f} | val greedy exact-match {va_acc:.3f}")


if __name__ == "__main__":
    main()
