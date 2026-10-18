import torch
import os, sys  # [PATCH 2026-10-18] put the day directory on sys.path: run as a script, sys.path[0] is
# exercises/ (or exercises/solutions/), so `from implementation import ...` raised ModuleNotFoundError from any cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from implementation import RelationNetwork

def verify_permutation_invariance(model: RelationNetwork, objects: torch.Tensor) -> bool:
    """
    Solution 3: Proving permutation invariance.
    """
    model.eval()
    
    # Original Pass
    with torch.no_grad():
        out_orig = model(objects)
        
    # Shuffle the objects along the N dimension
    indices = torch.randperm(objects.size(1))
    shuffled_objects = objects[:, indices, :]
    
    # Shuffled Pass
    with torch.no_grad():
        out_shuffled = model(shuffled_objects)
        
    diff = torch.abs(out_orig - out_shuffled).max().item()
    return diff < 1e-6
