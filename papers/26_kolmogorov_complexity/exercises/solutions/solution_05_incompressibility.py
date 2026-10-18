""" Day 26: The Incompressibility Challenge | Solution 05 | Part of 30u30 """
"""
Solution for Exercise 5
=======================
"""

import random
import string
import os, sys  # [PATCH 2026-10-18] make `implementation` importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from implementation import HuffmanCoder

def create_incompressible_string(length=100):
    # Random strings are the best way to achieve high Kolmogorov complexity
    # relative to a specific encoder.
    # [PATCH 2026-10-18] with 62 symbols a 100-char sample has empirical entropy ~5.5 bits/char,
    # so Huffman lands at ~0.69 of raw and the >= 0.70 check passed only 11% of runs.
    # 94 symbols (log2 94 = 6.55) pass 99.5% of runs.
    return "".join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))

if __name__ == "__main__":
    h_coder = HuffmanCoder()
    s = create_incompressible_string(100)
    bits = h_coder.get_complexity(s)
    raw_bits = len(s) * 8
    
    print(f"String: {s[:20]}...")
    print(f"Raw bits: {raw_bits}")
    print(f"Compressed: {bits}")
    print(f"Reduction: {(1 - bits/raw_bits)*100:.2f}%")
    
    # ASCII chars have approx 5.9 bits of entropy, so 5.9/8 = 0.73
    if bits >= raw_bits * 0.70:
        print("SUCCESS: String is incompressible!")
    else:
        print("FAILURE: String was too easy to compress.")
