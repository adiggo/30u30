import torch
import os, sys  # [PATCH 2026-10-18] put the day directory on sys.path: run as a script, sys.path[0] is
# exercises/ (or exercises/solutions/), so `from implementation import ...` raised ModuleNotFoundError from any cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from implementation import RelationNetwork

def compare_aggregators():
    """
    TODO: Prove that 'sum' is better for counting than 'mean'.
    Section 2.1: "The choice of summation [...] allows the model to count."
    
    1. Create a baseline set of 2 'ones' vectors.
    2. Create a larger set of 8 'ones' vectors.
    3. Compare output magnitudes using a RelationNetwork with aggregator='mean'.
    4. Compare output magnitudes using a RelationNetwork with aggregator='sum'.
    5. Show that 'mean' is size-invariant while 'sum' preserves the count.
    """
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    print("-" * 50)
    print("Exercise 7: Cardinality Bias (Sum vs Mean)")
    print("-" * 50)
    compare_aggregators()
    print("Exercise 7: [COMPLETE]")
