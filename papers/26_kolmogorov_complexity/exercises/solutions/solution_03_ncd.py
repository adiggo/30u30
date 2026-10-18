""" Day 26: Similarity Check (NCD) | Solution 03 | Part of 30u30 """
"""
Solution for Exercise 3
=======================
"""

import os, sys  # [PATCH 2026-10-18] make `implementation` importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import zlib
from implementation import HuffmanCoder, ComplexityMetrics
import random
import string

def test_ncd():
    h_coder = HuffmanCoder()
    # [PATCH 2026-10-18] Huffman is memoryless: C(xx) = 2C(x), so NCD(x, x) came out exactly 1.
    # NCD needs a "normal" compressor with C(xx) ~ C(x) (Cilibrasi & Vitanyi, Def. 3.2).
    compressor = lambda x: 8 * len(zlib.compress(x.encode("utf-8"), 9))
    
    s_base = "Machine learning is the study of computer algorithms that improve through experience."
    s_same = s_base[:]
    s_diff = "I like green eggs and ham, I do not like them Sam I am."
    s_rand = "".join(random.choices(string.printable, k=len(s_base)))
    
    # Calculate NCD for the combinations
    d_identity = ComplexityMetrics.ncd(s_base, s_same, compressor)
    d_different = ComplexityMetrics.ncd(s_base, s_diff, compressor)
    d_random = ComplexityMetrics.ncd(s_base, s_rand, compressor)
    
    print(f"Distance (Identity):  {d_identity:.4f}")
    print(f"Distance (Different): {d_different:.4f}")
    print(f"Distance (Random):    {d_random:.4f}")

if __name__ == "__main__":
    test_ncd()
