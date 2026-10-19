"""
Solution 5: Traveling Salesman Problem (TSP)
============================================

Deep learning solution for approximately solving the TSP.
(TSP is a Pointer Networks experiment, Day 18; it is not in Order Matters.)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
# [PATCH 2026-10-18] was sys.path.append('..'), which resolves against the *current
# directory*: run from the day folder (or from solutions/) the import of
# implementation.py failed with ModuleNotFoundError. Resolve relative to this file.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from implementation import ReadProcessWrite


class TSPDataset(Dataset):
    def __init__(self, num_samples, num_cities):
        self.num_samples = num_samples
        self.num_cities = num_cities
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        cities = torch.rand(self.num_cities, 2)
        unvisited = set(range(self.num_cities))
        # [PATCH 2026-10-18] was `current = 0`: an order-invariant model cannot tell which city
        # has index 0. Start at the leftmost city (defined by content) instead.
        current = int(torch.argmin(cities[:, 0]))
        tour = [current]
        unvisited.remove(current)
        while unvisited:
            nearest = min(unvisited, key=lambda c: torch.norm(cities[current] - cities[c]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        return cities, torch.tensor(tour, dtype=torch.long)


def train_tsp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # [PATCH 2026-10-18] new Read-Process-Write constructor (was use_set_encoder=True, num_layers=3)
    model = ReadProcessWrite(input_dim=2, hidden_dim=128, process_steps=5, glimpses=1).to(device)
    dataset = TSPDataset(2000, 10)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for cities, targets in loader:
        cities, targets = cities.to(device), targets.to(device)
        lengths = torch.full((cities.size(0),), 10, device=device)
        _, log_probs, _ = model(cities, lengths, 10, teacher_forcing=targets)
        
        # [PATCH 2026-10-18] log_probs already holds log p(target_t) per step (teacher
        # forcing); gather(1, targets) re-indexed the step axis by pointer values.
        loss = -log_probs.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    print("TSP solution loaded.")
