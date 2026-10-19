"""
Exercise 4: Convex Hull with Pointer Networks
==============================================

Solve geometric problems using neural networks.

Task: Given a set of 2D points, find the subset of boundary points 
      (convex hull) in counter-clockwise order, starting at the leftmost point,
      and then point at STOP to end the output.

Concept: The convex hull is the smallest convex polygon that contains 
         all the points in a set.

Note: the convex hull is a Pointer Networks experiment (Vinyals et al. 2015a, Day 18),
      not one of Order Matters'. Here it practises a variable-length output: the decoder
      ends it by pointing at an extra learned "STOP" memory slot.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial import ConvexHull as ScipyConvexHull
import matplotlib.pyplot as plt
import os
import sys
# [PATCH 2026-10-18] was sys.path.append('..'), which resolves against the *current
# directory*: run from the day folder (or from solutions/) the import of
# implementation.py failed with ModuleNotFoundError. Resolve relative to this file.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from implementation import ReadProcessWrite, IGNORE_INDEX


class ConvexHullDataset(Dataset):
    """
    Generate random point sets and compute ground truth convex hulls.
    """
    
    def __init__(self, num_samples, num_points):
        self.num_samples = num_samples
        self.num_points = num_points
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # TODO: Generate random 2D points in [0, 1]^2
        points = None 
        
        # [PATCH 2026-10-18] The old scaffold padded short hulls by repeating the first
        # vertex: an index the decoder has already masked, so the loss was infinite.
        # TODO: Compute the hull vertices with ScipyConvexHull(points.numpy()).vertices
        #       (for 2-D, scipy lists them counter-clockwise, starting anywhere).
        # TODO: Rotate that cycle to start at the leftmost point (smallest x), so every hull
        #       has one canonical order (Order Matters Sec. 5.1.3). Hint: np.argmin, np.roll.
        #       Why not "lowest index first" (Pointer Networks' choice)? An order-invariant
        #       model never sees indices; the start must be defined by content.
        # TODO: Target of length num_points + 1: the vertices, then STOP = num_points (the
        #       extra slot of ReadProcessWrite(stop_token=True)), then IGNORE_INDEX (-1).
        hull_indices = None
        
        return points, hull_indices


def compute_hull_accuracy(predicted, true, stop_index=None):
    """
    Evaluate predicted hull indices against ground truth.

    [PATCH 2026-10-18] With stop_index, each sequence is cut at its first STOP (the
    target's IGNORE_INDEX padding is dropped with it) before comparing.
    """
    def vertices(seq):
        seq = [i for i in seq.tolist() if i != IGNORE_INDEX]
        return seq[:seq.index(stop_index)] if stop_index in seq else seq

    predicted, true = vertices(predicted), vertices(true)
    exact_match = predicted == true
    
    # Intersection over Union (IoU) on the set of indices
    pred_set = set(predicted)
    true_set = set(true)
    overlap = len(pred_set.intersection(true_set)) / len(true_set)
    
    return exact_match, overlap


def demo_convex_hull():
    print("Training Pointer Network for Convex Hull")
    print("-" * 30)
    
    num_points = 10
    hidden_dim = 128
    num_epochs = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TODO: Initialize dataset, model, and training components
    # [PATCH 2026-10-18] the model now needs the STOP slot (was use_set_encoder=True)
    model = ReadProcessWrite(
        input_dim=2, 
        hidden_dim=hidden_dim,
        process_steps=5,
        glimpses=1,
        stop_token=True
    ).to(device)
    
    print("Convex Hull model initialized.")


if __name__ == "__main__":
    demo_convex_hull()
    
    print("Exercise 4 Summary")
    print("-" * 30)
    print("""
Key concepts covered:
1. Geometric Learning: Applying Pointer Networks to spatial configurations.
2. Invariant Encoding: Why treating points as a set is necessary for geometry.
3. End-of-sequence pointer: a STOP slot ends variable-length outputs (hull sizes); steps
   after it are padded with IGNORE_INDEX and skipped by the loss.
    """)
