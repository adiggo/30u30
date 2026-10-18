"""Day 25 (Kaplan et al. 2020 scaling laws, + Chinchilla) reference checks.

1. KaplanTransformer.count_parameters('Kaplan') vs Eq. 2.1, N = 12 n_layer d^2: the exact gap must be
   the sub-leading terms Eq. 2.1 omits (biases 5d + LayerNorms 4d per layer, final LayerNorm 2d).
2. ComputeEconomy: C = 6 N D and 1 PF-day = 1e15 * 24 * 3600 = 8.64e19 FLOPs (Sec 1.3).
3. Eq. 1.5 L(N, D) reduces to Eq. 1.1 L(N) as D -> inf and to Eq. 1.2 L(D) as N -> inf (Table 5 constants).
4. Eq. 4.4 (D >~ 5e3 N^0.74) is the data size at which Eq. 4.3's overfitting penalty dL ~ 0.02
   (the seed-to-seed variation the paper quotes), using the Table 2 fit of L(N, D).
5. Sec 1.2: doubling N multiplies L(N) by 2^-0.076 = 0.95; Fig. 14: per 10x compute N grows 10^0.73,
   data 10^0.27; Chinchilla quotes 5.5x / 1.8x.
6. MasterFitter (L_inf + (Xc/X)^alpha) recovers alpha on synthetic data; report how well.
7. Exercise solutions: solution_05 count = 12 d^2 + 4 d exactly; solution_04 (patched) = Eq. 1.1.
8. Chinchilla: a = beta/(alpha+beta), b = alpha/(alpha+beta) from Eq. 10's fit; D ~ 20 N with C = 6ND at
   the Gopher budget (5.76e23 FLOPs) gives ~70B params / ~1.4T tokens (Chinchilla's actual size).

Run:  OMP_NUM_THREADS=1 python validation/day25_scaling_formulas.py
"""
import math
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file is in validation/)
DAY = os.path.join(REPO, 'papers', '25_scaling_laws')
sys.path.insert(0, DAY)
from implementation import KaplanTransformer, ComputeEconomy, MasterFitter  # noqa: E402

# Kaplan et al. 2020, Appendix A, Table 5 (and Eqs. 1.1-1.3)
ALPHA_N, N_C = 0.076, 8.8e13
ALPHA_D, D_C = 0.095, 5.4e13
ALPHA_CMIN, CMIN_C = 0.050, 3.1e8
# Table 2: joint fit of Eq. 1.5
T2 = dict(alpha_N=0.076, alpha_D=0.103, N_c=6.4e13, D_c=1.8e13)


def L_N(N): return (N_C / N) ** ALPHA_N
def L_D(D): return (D_C / D) ** ALPHA_D
def L_ND(N, D, aN=ALPHA_N, aD=ALPHA_D, Nc=N_C, Dc=D_C): return ((Nc / N) ** (aN / aD) + Dc / D) ** aD


def check_params():
    worst = 0
    for d, L in [(64, 2), (128, 2), (256, 4), (512, 12)]:
        m = KaplanTransformer(vocab_size=100, d_model=d, n_heads=4, n_layers=L, max_seq_len=64)
        n = m.count_parameters('Kaplan')
        expected = 12 * L * d * d + L * (5 * d + 4 * d) + 2 * d
        worst = max(worst, abs(n - expected))
        print(f'[1] d={d:3d} L={L:2d}: count {n:,} = 12Ld^2 {12*L*d*d:,} + 9Ld + 2d -> |d| = {abs(n-expected)}; '
              f'relative excess over 12Ld^2 = {(n-12*L*d*d)/(12*L*d*d):.2%}')
    return worst


def check_compute():
    pf_day = 1e15 * 24 * 3600
    c = ComputeEconomy.calculate_c_pfdays(175e9, 300e9)
    print(f'[2] 1 PF-day = {pf_day:.3e} FLOPs (paper: 8.64e19); GPT-3-sized 6ND = {6*175e9*300e9:.3e} FLOPs = {c:,.0f} PF-days')
    assert abs(pf_day - 8.64e19) < 1 and abs(c - 6 * 175e9 * 300e9 / 8.64e19) < 1e-9


def check_limits():
    d1 = max(abs(L_ND(N, 1e40) - L_N(N)) for N in np.logspace(5, 11, 13))
    d2 = max(abs(L_ND(1e60, D) - L_D(D)) for D in np.logspace(7, 12, 11))
    print(f'[3] Eq.1.5 -> Eq.1.1 as D->inf: max |d| = {d1:.1e}; -> Eq.1.2 as N->inf: max |d| = {d2:.1e}')
    return max(d1, d2)


def check_overfit_rule():
    vals = []
    for N in [1e6, 1e7, 1e8, 1e9]:
        D = 5e3 * N ** 0.74
        dL = (1 + (N / T2['N_c']) ** (T2['alpha_N'] / T2['alpha_D']) * T2['D_c'] / D) ** T2['alpha_D'] - 1
        vals.append((N, D, dL))
    print('[4] Eq.4.3 penalty at D = 5e3*N^0.74 (Table 2 constants): ' +
          ', '.join(f'N={N:.0e}: D={D:.2e}, dL={dL:.4f}' for N, D, dL in vals))
    print(f"[4] exponent alpha_N/alpha_D = {T2['alpha_N']/T2['alpha_D']:.3f} (Table 2) and "
          f'{ALPHA_N/ALPHA_D:.3f} (Eqs. 1.1/1.2) vs 0.74 in Eq. 4.4; N=1e9 needs D={vals[-1][1]:.2e} '
          '(WebText2 has 2.29e10 tokens, Sec 2.3)')
    return vals


def check_ratios():
    print(f'[5] 2^-alpha_N = {2**-ALPHA_N:.4f} (paper: 0.95); 10^0.73 = {10**0.73:.2f}, 10^0.27 = {10**0.27:.2f} '
          f'(Fig. 14: ~5x params, ~2x data per 10x compute; Chinchilla quotes 5.5x / 1.8x)')
    print(f'[5] Eq. 1.1 at N = 1.75e11: L = {L_N(1.75e11):.4f} nats/token (WebText2 tokenisation; converged, early-stopped)')


def check_fitter():
    rng = np.random.default_rng(0)
    out = []
    x = np.logspace(5, 8, 10)
    for noise in (0.0, 0.01):
        y = 1.7 + (N_C / x) ** ALPHA_N + rng.normal(0, noise, x.size)
        f = MasterFitter(x, y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f.fit()
        out.append((noise, f.alpha, f.l_inf, f.xc))
        print(f'[6] MasterFitter, noise sd {noise}: alpha = {f.alpha:.4f} (true 0.076), L_inf = {f.l_inf:.3f} '
              f'(true 1.7), Xc = {f.xc:.2e} (true 8.8e13); max fit residual '
              f'{np.max(np.abs(f.predict(x) - y)):.4f}')
    return out


def check_solutions():
    sys.path.insert(0, os.path.join(DAY, 'exercises', 'solutions'))
    import solution_05_transformer as s5
    import solution_04_predictor as s4
    import solution_01_fitter as s1
    b = s5.ScalingBlock(128)
    n = b.count_params()
    print(f'[7] solution_05: {n} = 12*128^2 + 4*128 = {12*128*128 + 4*128}: {n == 12*128*128 + 4*128}')
    print(f'[7] solution_04 (patched): L(1.75e11) = {s4.predict_loss(1.75e11):.4f}; Eq. 1.1 = {L_N(1.75e11):.4f}')
    a, A = s1.fit_power_law(np.array([1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8]),
                            np.array([3.20, 2.85, 2.70, 2.40, 2.28, 2.05, 1.95]))
    print(f'[7] solution_01 fitted alpha = {a:.4f} (exercise README success criterion: 0.076 +/- 0.001 -> '
          f'{"met" if abs(a-0.076) <= 0.001 else "NOT met"})')


def check_chinchilla():
    alpha, beta, E, A, B = 0.34, 0.28, 1.69, 406.4, 410.7  # Hoffmann et al. 2022, Eq. 10
    a, b = beta / (alpha + beta), alpha / (alpha + beta)
    C = 5.76e23  # Gopher's training compute (Hoffmann et al. Fig. 2 caption)
    N = math.sqrt(C / (6 * 20)); D = 20 * N
    print(f'[8] Chinchilla Eq. 4 exponents from the Eq. 10 fit: a = {a:.3f}, b = {b:.3f} (Table 2, approach 3: 0.46/0.54)')
    print(f'[8] D = 20N and C = 6ND at C = {C:.2e}: N = {N/1e9:.1f}B params, D = {D/1e12:.2f}T tokens '
          f'(Chinchilla: 70B, 1.4T); Table 3 ratios: 20.2B/1B = 20.2, 205.1B/10B = 20.5, 1.5T/67B = 22.4')
    print(f'[8] Eq. 10 (MassiveText fit) at Chinchilla vs Gopher size/tokens (paper: same compute budget): '
          f'L(70B, 1.4T) = {E + A/70e9**alpha + B/1.4e12**beta:.3f}, '
          f'L(280B, 300B) = {E + A/280e9**alpha + B/300e9**beta:.3f}')


if __name__ == '__main__':
    p = check_params()
    check_compute()
    lim = check_limits()
    check_overfit_rule()
    check_ratios()
    check_fitter()
    check_solutions()
    check_chinchilla()
    print(f'SUMMARY day25: param-count residual {p}, Eq.1.5 limits {lim:.1e}')
