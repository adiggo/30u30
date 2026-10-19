"""
Solution 3: Training Pointer Networks for Sorting
==================================================

Implementation of the training loop for numerical sorting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
# [PATCH 2026-10-18] was sys.path.append('..'), which resolves against the *current
# directory*: run from the day folder (or from solutions/) the import of
# implementation.py failed with ModuleNotFoundError. Resolve relative to this file.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from implementation import PointerNetwork


class SortingDataset(Dataset):
    def __init__(self, num_samples, set_size, min_val=0.0, max_val=10.0):
        self.num_samples = num_samples
        self.set_size = set_size
        self.min_val = min_val
        self.max_val = max_val
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        values = torch.rand(self.set_size) * (self.max_val - self.min_val) + self.min_val
        _, target = torch.sort(values)
        inputs = values.unsqueeze(1)
        return inputs, target


def compute_loss(log_probs, targets):
    # [PATCH 2026-10-18] PointerNetwork returns log p(target_t) per step under teacher
    # forcing, shape (batch, steps); gather(1, targets) re-indexed the step axis by
    # pointer values (equal to the NLL only for permutation targets). Use the mean.
    return -log_probs.mean()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, set_size = inputs.size(0), inputs.size(1)
        
        lengths = torch.full((batch_size,), set_size, device=device)
        pointers, log_probs, attn = model(inputs, lengths, set_size, teacher_forcing=targets)
        
        loss = compute_loss(log_probs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        
        total_loss += loss.item()
        # [PATCH 2026-10-18] with teacher forcing `pointers` are the targets themselves,
        # so this accuracy was always 100%. Use the model's per-step argmax instead.
        predicted = torch.stack([a.argmax(dim=-1) for a in attn], dim=1)
        correct += (predicted == targets).all(dim=1).sum().item()
        total += batch_size
    
    return total_loss / len(dataloader), correct / total

if __name__ == "__main__":
    print("Sorting Training Solution loaded.")
