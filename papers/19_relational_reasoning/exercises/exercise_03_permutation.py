import torch
import os, sys  # [PATCH 2026-10-18] put the day directory on sys.path: run as a script, sys.path[0] is
# exercises/ (or exercises/solutions/), so `from implementation import ...` raised ModuleNotFoundError from any cwd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from implementation import RelationNetwork

def verify_invariance():
    """
    TODO: Programmatically prove that the RN is permutation invariant.
    Section 2.1 states: "RNs are invariant to the order of objects in the set."
    
    1. Create a set of random objects (B=1, N=5, D=8).
    2. Create a RelationNetwork.
    3. Forward pass objects.
    4. Shuffle the objects along the N dimension.
    5. Forward pass shuffled objects.
    6. Assert that the outputs are identical.
    """
    # YOUR CODE HERE
    pass

if __name__ == "__main__":
    print("-" * 50)
    print("Exercise 3: Order Invariance Proof")
    print("-" * 50)
    verify_invariance()
    print("Exercise 3: [COMPLETE]")
