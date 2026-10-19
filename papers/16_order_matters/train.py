"""
Training script for Read-Process-Write and the Pointer Network baseline.
========================================================================

Tasks supported:
- sort: sort N random numbers (the paper's experiment, Sec. 4.4 / Table 1)
- hull: convex hull of N random points (a Pointer Networks experiment, Vinyals et al. 2015a;
        not in Order Matters) -- variable-length output ended by a STOP pointer
- tsp: nearest-neighbour tours of N random cities (also a Pointer Networks task)

Models (--model):
- rpw:    ReadProcessWrite, the paper's model (Sec. 4.3): per-element MLP read, T process
          steps (Eqs. 3-7, --process-steps), pointer-network write with 0/1 glimpses (--glimpses)
- ptrnet: PointerNetwork, the order-sensitive LSTM-encoder baseline ("Ptr-Net" in Table 1)
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import ConvexHull as ScipyConvexHull

from implementation import ReadProcessWrite, PointerNetwork, compute_loss, accuracy, IGNORE_INDEX


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
    """Points uniform in the unit square -> their convex hull as a sequence of pointers.

    [PATCH 2026-10-18] The old labels padded short hulls by repeating the first vertex, an
    index the decoder had already masked (log-prob -inf, infinite loss), and started the
    cycle at whichever vertex scipy listed first. Now each target is
        [hull vertices in canonical order..., STOP, IGNORE_INDEX, ...]   (length N + 1)
    - canonical order: start at the leftmost point (smallest x) and go counter-clockwise
      (Order Matters Sec. 5.1.3: restrict the output equivalence class to one member).
      Pointer Networks started tours at the lowest *index*; an order-invariant model never
      sees indices, so its start vertex must be defined by content.
    - STOP = N, the extra learned memory slot of ReadProcessWrite(stop_token=True)
      (Pointer Networks' end-of-sequence symbol); IGNORE_INDEX steps are skipped by the loss.
    """

    def __init__(self, num_samples: int, set_size: int):
        if set_size < 3:
            raise ValueError("a convex hull needs at least 3 points")
        self.num_samples = num_samples
        self.set_size = set_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        points = torch.rand(self.set_size, 2)
        p = points.numpy()
        vertices = ScipyConvexHull(p).vertices     # (k,) indices; for 2-D hulls scipy lists them counter-clockwise
        start = np.argmin(p[vertices, 0])          # position of the leftmost vertex within that cycle
        vertices = np.roll(vertices, -start)       # rotate the cycle to start there (still counter-clockwise)
        target = torch.full((self.set_size + 1,), IGNORE_INDEX, dtype=torch.long)
        target[:len(vertices)] = torch.from_numpy(vertices)
        target[len(vertices)] = self.set_size      # STOP slot
        return points, target


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
        # [PATCH 2026-10-18] was `current = 0`: ReadProcessWrite cannot tell which city has
        # index 0 (it is order-invariant), so the first step of such a target is unlearnable.
        # Start at the leftmost city instead, a choice defined by content.
        current = int(torch.argmin(cities[:, 0]))
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
        
        # [PATCH 2026-10-18] max_steps = target length (N + 1 for hulls: vertices + STOP)
        pointers, log_probs, attn = model(
            inputs, lengths, max_steps=targets.size(1),
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
            set_size, steps = inputs.size(1), targets.size(1)
            lengths = torch.full((inputs.size(0),), set_size, device=device)
            pointers, _, _ = model(inputs, lengths, max_steps=steps)
            # [PATCH 2026-10-18] The greedy pass returns log p(argmax pointer), so
            # the old loss measured the model's confidence in its own choices, not
            # the NLL of the targets. Score the targets with a teacher-forced pass.
            _, log_probs, _ = model(inputs, lengths, max_steps=steps, teacher_forcing=targets)
            total_loss += compute_loss(log_probs, targets).item()
            total_acc += accuracy(pointers, targets)
            num_batches += 1
    return total_loss / num_batches, total_acc / num_batches


def build_model(model_name: str, task: str, input_dim: int, hidden_dim: int,
                process_steps: int = 5, glimpses: int = 1, mask_chosen: bool = True) -> nn.Module:
    """[PATCH 2026-10-18] rpw = the paper's Read-Process-Write; ptrnet = LSTM-encoder baseline."""
    if model_name == "rpw":
        return ReadProcessWrite(input_dim, hidden_dim, process_steps=process_steps,
                                glimpses=glimpses, stop_token=(task == "hull"),
                                mask_chosen=mask_chosen)
    if task == "hull":
        raise SystemExit("--task hull needs an end-of-output pointer; only --model rpw has a STOP slot")
    return PointerNetwork(input_dim, hidden_dim, dropout=0.0, mask_chosen=mask_chosen)


def main():
    # [PATCH 2026-10-18] train.py had no main(): `python train.py --task sort ...`
    # (README, CHEATSHEET) and the `train-pointer=train:main` entry point in
    # setup.py just printed "Training script loaded." and exited 0 without
    # training anything. Minimal CLI wiring of the functions above.
    # [PATCH 2026-10-18] Default model is now the paper's Read-Process-Write; --task hull trains
    # (STOP pointer + canonical hull order) instead of refusing.
    parser = argparse.ArgumentParser(description="Day 16: train Read-Process-Write (or a Ptr-Net) on a set task")
    parser.add_argument("--task", choices=["sort", "tsp", "hull"], default="sort")
    parser.add_argument("--model", choices=["rpw", "ptrnet"], default="rpw",
                        help="rpw: Read-Process-Write (Sec. 4.3); ptrnet: LSTM-encoder Pointer Network baseline")
    parser.add_argument("--process-steps", type=int, default=5, help="T, process steps of Eqs. 3-7 (rpw only)")
    parser.add_argument("--glimpses", type=int, choices=[0, 1], default=1, help="glimpses before each pointer (rpw only)")
    parser.add_argument("--no-mask", action="store_true",
                        help="do not mask elements already pointed at (the model must learn not to repeat them)")
    parser.add_argument("--set-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "sort":
        train_set, val_set, input_dim = SortingDataset(args.num_samples, args.set_size), SortingDataset(1000, args.set_size), 1
    elif args.task == "hull":
        train_set, val_set, input_dim = ConvexHullDataset(args.num_samples, args.set_size), ConvexHullDataset(1000, args.set_size), 2
    else:
        train_set, val_set, input_dim = TSPDataset(args.num_samples, args.set_size), TSPDataset(500, args.set_size), 2
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=100)
    model = build_model(args.model, args.task, input_dim, args.hidden_dim,
                        args.process_steps, args.glimpses, mask_chosen=not args.no_mask).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    rpw_cfg = f" process_steps={args.process_steps} glimpses={args.glimpses}" if args.model == "rpw" else ""
    print(f"task={args.task} set_size={args.set_size} model={args.model}{rpw_cfg} "
          f"mask_chosen={not args.no_mask} params={sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    # [PATCH 2026-10-18] Score the untrained model too. fork_rng restores the random state
    # afterwards, so this extra pass does not change the training data that follows.
    with torch.random.fork_rng():
        va_loss, va_acc = evaluate(model, val_loader, device)
    print(f"Epoch  0 | before training{' ' * 40}| val NLL {va_loss:.4f} | val greedy exact-match {va_acc:.3f}")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:2d} | train NLL {tr_loss:.4f} | teacher-forced acc {tr_acc:.3f} | "
              f"val NLL {va_loss:.4f} | val greedy exact-match {va_acc:.3f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
