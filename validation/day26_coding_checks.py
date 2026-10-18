"""Day 26 reference checks (Shen-Uspensky-Vereshchagin, Ch. 7; Cilibrasi & Vitanyi 2005).

1. HuffmanCoder: prefix-free; Kraft sum <= 1 (Kraft lemma, book p. 214); average length in
   [H, H+1) (Theorem 138); total length equals an independent Huffman cost (sum of merged
   weights); lossless round trip.
2. Arithmetic coding: an EXACT rational-arithmetic coder (fractions.Fraction) round-trips
   and its code length is <= ceil(-log2 P(x)) + 1.  The repo's float coder tracks P(x) only
   while P(x) > ~1e-16 relative to `low`; beyond ~50 bits the interval collapses.  Also
   checks train_minimal's 'a_bits_estimated' heuristic.
3. NCD (Eq. 3.1): repo formula == independent formula; Huffman-as-compressor gives
   NCD(x,x) = 1 exactly (memoryless => C(xx) = 2C(x)); zlib gives ~0.
4. Order-0 blindness: 'ABC'*1666 costs 5/3 bits/char under Huffman (ratio 0.208) but zlib
   (which sees repetition) needs far fewer bits -- Huffman length is ~N*h(freqs), the
   Theorem-146 bound, not an estimate of C(x).
"""
import heapq
import math
import os
import random
import string
import sys
import zlib
from collections import Counter
from fractions import Fraction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
sys.path.insert(0, os.path.join(REPO, "papers", "26_kolmogorov_complexity"))
from implementation import HuffmanCoder, ArithmeticCoder, ComplexityMetrics

ok = True
random.seed(0)
texts = {"periodic": "ABC" * 333, "english": "Complexity is the amount of information required to describe an object. " * 7,
         "random62": "".join(random.choices(string.ascii_letters + string.digits, k=1000)),
         "skewed": "a" * 900 + "b" * 90 + "c" * 10}

# ---- 1. Huffman --------------------------------------------------------------------------
for name, t in texts.items():
    hc = HuffmanCoder(); bits = hc.encode(t); codes = hc.codes
    words = list(codes.values())
    prefix_free = not any(a != b and b.startswith(a) for a in words for b in words)
    kraft = sum(2.0 ** -len(w) for w in words)
    N = len(t); freqs = Counter(t)
    H = -sum(c / N * math.log2(c / N) for c in freqs.values())       # empirical entropy, bits/char
    avg = len(bits) / N
    heap = list(freqs.values()); heapq.heapify(heap); cost = 0          # independent Huffman cost
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap); cost += a + b; heapq.heappush(heap, a + b)
    inv = {w: ch for ch, w in codes.items()}; out, cur = [], ""
    for bit in bits:                                                    # greedy prefix decoding
        cur += bit
        if cur in inv: out.append(inv[cur]); cur = ""
    rt = "".join(out) == t
    good = prefix_free and kraft <= 1 + 1e-12 and H - 1e-12 <= avg < H + 1 and cost == len(bits) and rt
    ok &= good
    print(f"1. {name:9s} prefix_free={prefix_free} kraft={kraft:.3f} H={H:.4f} avg={avg:.4f} "
          f"in[H,H+1)={H <= avg < H + 1} optimal(cost={cost})={cost == len(bits)} roundtrip={rt}")

# ---- 2. Arithmetic coding ------------------------------------------------------------------
def exact_encode(t):
    N = len(t); f = Counter(t); syms = sorted(f)
    cum, acc = {}, Fraction(0)
    for s in syms: cum[s] = (acc, acc + Fraction(f[s], N)); acc += Fraction(f[s], N)
    lo, hi = Fraction(0), Fraction(1)
    for ch in t:
        w = hi - lo; a, b = cum[ch]; lo, hi = lo + w * a, lo + w * b
    # shortest binary fraction k/2^L inside [lo, hi)
    L = 0
    while True:
        k = math.ceil(lo * 2 ** L)
        if Fraction(k, 2 ** L) < hi: return k, L, lo, hi, cum
        L += 1

def exact_decode(k, L, cum, N):
    lam, lo, hi, out = Fraction(k, 2 ** L), Fraction(0), Fraction(1), []
    for _ in range(N):
        w = hi - lo
        for s, (a, b) in cum.items():
            if lo + w * a <= lam < lo + w * b: out.append(s); lo, hi = lo + w * a, lo + w * b; break
    return "".join(out)

for name in ("periodic", "skewed", "random62"):
    t = texts[name][:300]
    k, L, lo, hi, cum = exact_encode(t)
    P = hi - lo; nlog = -math.log2(P.numerator) + math.log2(P.denominator)   # -log2 P(x) exactly
    rt = exact_decode(k, L, cum, len(t)) == t
    ok &= rt and L <= math.ceil(nlog) + 1
    print(f"2. exact coder {name:9s} n={len(t)}: -log2 P = {nlog:8.2f} bits, code = {L} bits "
          f"(<= ceil+1: {L <= math.ceil(nlog) + 1}), round trip = {rt}")

ac = ArithmeticCoder()
for n in (10, 30, 60, 300):
    t = texts["random62"][:n]
    p = ac._get_probabilities(t); lo, hi = 0.0, 1.0
    for ch in t:
        w = hi - lo; hi = lo + w * p[ch][1]; lo = lo + w * p[ch][0]
    _, _, elo, ehi, _ = exact_encode(t)
    exact_bits = -math.log2((ehi - elo).numerator) + math.log2((ehi - elo).denominator)
    float_bits = -math.log2(hi - lo) if hi > lo else float("inf")
    print(f"2. repo float coder n={n:3d}: exact -log2P={exact_bits:8.2f}  float width gives {float_bits:8.2f}"
          f"  {'(collapsed: high == low)' if hi <= lo else ''}")
print(f"2. train_minimal heuristic calculate_min_bits(v, v+1e-15) = {ArithmeticCoder.calculate_min_bits(0.3, 0.3 + 1e-15)} bits for ANY input")

# ---- 3. NCD ---------------------------------------------------------------------------------
def ncd_ref(x, y, C):
    cx, cy, cxy = C(x), C(y), C(x + y)
    return (cxy - min(cx, cy)) / max(cx, cy)                            # Eq. (3.1)
hc = HuffmanCoder(); huff = lambda s: hc.get_complexity(s)
z = lambda s: 8 * len(zlib.compress(s.encode(), 9))
base = "Machine learning is the study of computer algorithms that improve through experience."
diff = "I like green eggs and ham, I do not like them Sam I am."
rnd = "".join(random.choices(string.printable, k=len(base)))
for cname, C in (("huffman", huff), ("zlib", z)):
    vals = [ComplexityMetrics.ncd(base, y, C) for y in (base, diff, rnd)]
    d = max(abs(v - ncd_ref(base, y, C)) for v, y in zip(vals, (base, diff, rnd)))
    ok &= d == 0
    print(f"3. NCD with {cname:7s}: identical={vals[0]:.3f} different={vals[1]:.3f} random={vals[2]:.3f}  |repo - Eq.3.1| = {d}")

# ---- 4. order-0 blindness -------------------------------------------------------------------
t = "ABC" * 1666
print(f"4. 'ABC'*1666: Huffman {HuffmanCoder().get_complexity(t)} bits (ratio {HuffmanCoder().get_complexity(t)/(8*len(t)):.3f}),"
      f" zlib {z(t)} bits (ratio {z(t)/(8*len(t)):.4f})")
print("RESULT:", "PASS" if ok else "FAIL")
