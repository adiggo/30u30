"""Day 22 reference checks for implementation.py (Deep Speech 2).

1. CTC: independent log-space alpha recursion (Graves et al. 2006) vs F.ctc_loss / nn.CTCLoss, incl.
   repeated labels at the minimum feasible length, an infeasible case, empty targets, a padded batch with
   concatenated targets exactly as collate_batch builds them, reduction='mean' as in train_step, and
   gradients w.r.t. the pre-log_softmax logits.
2. Greedy decoding (CharEncoder.ctc_decode / greedy_decode) vs "collapse repeats, then drop blanks".
3. beam_search_decode vs brute-force exact argmax_y p(y|x) on tiny problems (pre-fix code from git vs patched).
4. SequenceBatchNorm vs nn.BatchNorm1d over (batch*time), and with zero padding (pre-fix vs patched).
5. Conv output time/frequency length formulas vs the actual conv stack.

Run:  OMP_NUM_THREADS=1 python validation/day22_ctc_decode_bn_check.py
"""
import itertools
import os
import subprocess
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '22_deep_speech_2')
# Pre-fix code for the original-vs-patched comparison: the commit the fixes on this branch were
# made on (read with `git show`, no checkout; needs the repo's git history).
ORIG_REF = '195dcf2'
sys.path.insert(0, DAY)


def module_from_source(name, src):
    mod = types.ModuleType(name)
    sys.modules[name] = mod                      # dataclasses need the module registered
    exec(compile(src, name, 'exec'), mod.__dict__)
    return mod


new = module_from_source('ds2_new', open(DAY + '/implementation.py').read())
head = module_from_source('ds2_head', subprocess.check_output(
    ['git', '-C', REPO, 'show', ORIG_REF + ':papers/22_deep_speech_2/implementation.py'], text=True))
NEG = -1e30   # finite stand-in for log(0): keeps autograd free of (-inf) - (-inf) = nan


def ctc_nll(lp, target, blank=0):
    """-log p(target | x) by the CTC forward (alpha) recursion. lp: (T, C) log-probabilities."""
    ext = [blank]
    for c in target:
        ext += [c, blank]                       # l' = blank, l1, blank, l2, ..., blank  (length 2L+1)
    S, T = len(ext), lp.shape[0]
    alpha = [lp[0, ext[0]]] + ([lp[0, ext[1]]] if S > 1 else []) + [torch.tensor(NEG)] * max(S - 2, 0)
    for t in range(1, T):
        new_alpha = []
        for s in range(S):
            terms = [alpha[s]] + ([alpha[s - 1]] if s >= 1 else [])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(alpha[s - 2])     # skip the blank only between DIFFERENT labels
            new_alpha.append(torch.logsumexp(torch.stack(terms), 0) + lp[t, ext[s]])
        alpha = new_alpha
    last = alpha[-1] if S == 1 else torch.logsumexp(torch.stack([alpha[-1], alpha[-2]]), 0)
    return -last


C = 5
cases = [(1, []), (4, []), (1, [2]), (3, [1, 1]), (2, [1, 1]), (5, [2, 2, 2]), (6, [1, 2, 1, 2, 3]),
         (8, [3]), (10, [1, 2, 2, 3]), (12, [4, 4, 1, 1, 2]), (20, [1, 2, 3, 4, 1, 2, 3])]
worst_l = worst_g = 0.0
print('1. CTC single utterances (C=5, blank=0):')
for T, tgt in cases:
    logits = torch.randn(T, C, requires_grad=True)
    lp = F.log_softmax(logits, -1)
    mine = ctc_nll(lp, tgt)
    ref = F.ctc_loss(lp[:, None, :], torch.tensor(tgt, dtype=torch.long), torch.tensor([T]),
                     torch.tensor([len(tgt)]), blank=0, reduction='none', zero_infinity=False)[0]
    feasible = T >= len(tgt) + sum(a == b for a, b in zip(tgt, tgt[1:]))
    if not feasible:
        print('   T=%2d target=%s infeasible: mine=%.1e  torch=%s' % (T, tgt, mine.item(), ref.item()))
        continue
    g_mine, = torch.autograd.grad(mine, logits, retain_graph=True)
    g_ref, = torch.autograd.grad(ref, logits)
    worst_l = max(worst_l, abs(mine.item() - ref.item()))
    worst_g = max(worst_g, (g_mine - g_ref).abs().max().item())
print('   loss max|d| = %.2e, grad wrt logits max|d| = %.2e  %s'
      % (worst_l, worst_g, 'PASS' if max(worst_l, worst_g) < 1e-10 else 'FAIL'))

# padded batch, concatenated targets, reduction='mean' as in train_step
Ts, tg = [9, 14, 6, 11], [[1, 2], [3, 3, 4, 1], [2], [4, 1, 1]]
x = torch.randn(max(Ts), len(Ts), C)
x[6:, 2] = 50.0                               # garbage in padded frames must not matter
lp = F.log_softmax(x, -1)
flat = torch.tensor(sum(tg, []))
lens = torch.tensor([len(t) for t in tg])
ref_mean = F.ctc_loss(lp, flat, torch.tensor(Ts), lens, blank=0, reduction='mean', zero_infinity=True)
mod_mean = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)(lp, flat, torch.tensor(Ts), lens)
mine_mean = torch.stack([ctc_nll(lp[:Ts[i], i], tg[i]) / len(tg[i]) for i in range(len(Ts))]).mean()
d = max(abs(ref_mean - mine_mean).item(), abs(mod_mean - mine_mean).item())
print('   padded batch, reduction=mean (sum_i nll_i/L_i / B): max|d| vs F.ctc_loss & nn.CTCLoss = %.2e  %s'
      % (d, 'PASS' if d < 1e-10 else 'FAIL'))

# ---- 2. greedy decoding ------------------------------------------------------------------------
enc = new.CharEncoder('english')


def ref_collapse(seq, blank=0):
    return [k for k, _ in itertools.groupby(seq) if k != blank]


bad = 0
for _ in range(2000):
    seq = list(np.random.choice([0, 0, 0, 1, 1, 2, 5], size=np.random.randint(1, 30)))
    bad += enc.ctc_decode(seq) != enc.decode(ref_collapse(seq))
lp = torch.randn(25, 3, 29)
L = torch.tensor([25, 17, 4])
gd = new.greedy_decode(lp, enc, L)
bad += sum(gd[i] != enc.decode(ref_collapse(lp[:L[i], i].argmax(-1).tolist())) for i in range(3))
print('2. greedy decode vs collapse-then-drop-blanks reference: %d mismatches in 2003 cases  %s'
      % (bad, 'PASS' if bad == 0 else 'FAIL'))

# ---- 3. beam search vs brute-force exact decoding ----------------------------------------------
def exact_best(lp3):
    """argmax over labelings y of p(y|x) = sum over alignments, enumerating all 3^T paths."""
    T = lp3.shape[0]
    score = {}
    for path in itertools.product(range(3), repeat=T):
        key = tuple(ref_collapse(path))
        score[key] = np.logaddexp(score.get(key, -np.inf), sum(lp3[t, k] for t, k in enumerate(path)))
    return max(score, key=score.get)


def as_full(lp3):
    full = torch.full((lp3.shape[0], 1, 29), -1e9)      # only blank, 'a'(1), 'b'(2) are possible
    full[:, 0, :3] = torch.from_numpy(lp3)
    return full


demo = np.log(np.array([[0.05, 0.9, 0.05], [0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]))  # path a _ a
print('3. "a _ a" demo: greedy=%r  exact=%r  beam(pre-fix)=%r  beam(patched)=%r'
      % (new.greedy_decode(as_full(demo), enc)[0], enc.decode(list(exact_best(demo))),
         head.beam_search_decode(as_full(demo), head.CharEncoder('english'), beam_width=10)[0],
         new.beam_search_decode(as_full(demo), enc, beam_width=10)[0]))
agree = {'HEAD': 0, 'patched': 0, 'greedy': 0}
N = 150
rs = np.random.RandomState(1)
for _ in range(N):
    T = rs.randint(2, 7)
    lp3 = np.log(rs.dirichlet(np.ones(3) * 0.7, size=T))
    ex = enc.decode(list(exact_best(lp3)))
    agree['HEAD'] += head.beam_search_decode(as_full(lp3), head.CharEncoder('english'), beam_width=64)[0] == ex
    agree['patched'] += new.beam_search_decode(as_full(lp3), enc, beam_width=64)[0] == ex
    agree['greedy'] += new.greedy_decode(as_full(lp3), enc)[0] == ex
print('   %d random tiny problems, agreement with exact argmax_y p(y|x): pre-fix beam %d, patched beam %d, '
      'greedy %d  %s' % (N, agree['HEAD'], agree['patched'], agree['greedy'],
                         'PASS' if agree['patched'] == N else 'FAIL'))

# ---- 4. sequence-wise batch norm ---------------------------------------------------------------
B, T, Cf = 4, 10, 8
x = torch.randn(B, T, Cf) * 2 + 1
bn = new.SequenceBatchNorm(Cf).train()
ref = torch.nn.BatchNorm1d(Cf).train()
d = (bn(x) - ref(x.reshape(-1, Cf)).reshape(B, T, Cf)).abs().max().item()
print('4. SequenceBatchNorm (no padding) vs BatchNorm1d over batch*time: max|d| = %.2e  %s'
      % (d, 'PASS' if d < 1e-12 else 'FAIL'))
lens = torch.tensor([10, 7, 4, 2])
valid = torch.arange(T)[None, :] < lens[:, None]
xp = x * valid[..., None]                                  # pad_packed_sequence zero-fills
v = xp[valid]
ref_out = (xp - v.mean(0)) / torch.sqrt(v.var(0, unbiased=False) + 1e-5)
d_head = (head.SequenceBatchNorm(Cf).train()(xp) - ref_out)[valid].abs().max().item()
d_new = (new.SequenceBatchNorm(Cf).train()(xp, lens) - ref_out)[valid].abs().max().item()
print('   padded batch (%.0f%% padding), valid frames vs masked-statistics reference: pre-fix max|d| = %.2f (FAIL), '
      'patched max|d| = %.1e  %s' % (100 * (1 - valid.float().mean()), d_head, d_new,
                                     'PASS' if d_new < 1e-12 else 'FAIL'))
fr = []
for seed in range(5):                                      # padding share in train_minimal-like batches
    np.random.seed(seed)
    batch = new.collate_batch(new.generate_synthetic_dataset(8, min_words=1, max_words=3), enc, n_fft=256)
    m = new.DeepSpeech2(n_freq=129, n_conv=2, n_rnn=1, rnn_hidden=8)
    ol = m._compute_time_after_conv(batch.feature_lengths)
    fr.append(1 - ol.sum().item() / (len(ol) * ol.max().item()))
print('   padding share of BN statistics in train_minimal-style batches (5 batches): %s'
      % ', '.join('%.0f%%' % (100 * f) for f in fr))
print('   placement: DS2 Eq. 8 normalizes only W^l h^{l-1} inside the recurrence; implementation.py'
      ' normalizes the bidirectional GRU *output* (structural deviation, not changed)')

# ---- 5. conv length formulas -------------------------------------------------------------------
m = new.DeepSpeech2(n_freq=129, n_conv=3, n_rnn=1, rnn_hidden=8).eval()
bad = 0
for Lt in range(1, 80):
    with torch.no_grad():
        out = m.conv(torch.zeros(1, 1, 129, Lt))
    bad += (out.size(3) != m._compute_time_after_conv(torch.tensor([Lt]))[0].item()
            or out.size(2) != m._compute_freq_after_conv(129))
print('5. conv time/freq length formulas vs actual conv stack (L=1..79, 3 conv layers): %d mismatches  %s'
      % (bad, 'PASS' if bad == 0 else 'FAIL'))
