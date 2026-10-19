"""
Exercise 4: NML (Normalized Maximum Likelihood) Complexity
==========================================================

[PATCH 2026-10-18] Exercise redesigned. The old version asked for a Monte Carlo average of the
maximised likelihood over y drawn from a box, without the box volume, for a model whose NML
complexity is infinite. It produced numbers around -380 bits that depended on the box.

Refined MDL (tutorial Secs. 2.5.3 and 2.6) codes the data with the NML distribution

    P_nml(y) = f(y | theta_hat(y)) / C,     C = integral of f(z | theta_hat(z)) over ALL datasets z

and charges each model its parametric complexity COMP = log2 C (eq. 2.17): the log of how many
datasets the model can fit well. By eq. 2.21 it is also the log of the number of distinguishable
distributions in the model (Sec. 2.6.2).

The catch (Example 2.20, Sec. 2.7.2; Sec. 2.8 for regression): for Gaussian models C is INFINITE.
A model that fits data centred at 0 fits data centred at 10^6 just as well, so the integral over
all datasets never converges. If sigma is also fitted, it diverges a second time: a curve through
the data with sigma_hat -> 0 has unbounded likelihood. Plain NML is undefined here.

The tutorial's remedy (Sec. 2.7.2) is to restrict where the fit may lie. You will:
  1. compute COMP of the polynomial model with KNOWN noise sigma and the fitted curve
     restricted to a mean square of at most R:  (1/n) sum_i (X beta)_i^2 <= R  (closed form);
  2. check it with a Monte Carlo estimate of the same integral that keeps the volume term;
  3. pick the degree with Rissanen's renormalised NML (RNML) for UNKNOWN sigma, which chooses
     the restriction from the data and renormalises (eq. 2.36), then compare it with the crude
     two-part code of Exercise 1.

Restaurant analogy, made precise: COMP is log2 of the number of dishes the kitchen can tell
apart, and that number is only finite if the menu (the allowed region R) is finite.
"""

import numpy as np
from math import comb, lgamma  # noqa: F401  (binomials and log-Gamma: you will need them)
from typing import Tuple, Dict


def nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    R: float,
    sigma: float = 1.0
) -> float:
    """
    COMP (bits) of the restricted Gaussian polynomial model with known noise sigma:

        M_R = { y ~ N(X beta, sigma^2 I) : (1/n) ||X beta||^2 <= R },   X = [1, x, ..., x^degree]

        COMP = log2  integral over all y of  max_{beta allowed} f(y | beta, sigma)  dy

    The allowed fitted-value vectors mu = X beta form a ball of radius sqrt(n R) inside the
    k-dimensional column space (k = degree + 1). Writing s = sqrt(n R) / sigma (that radius in
    noise units), the integral has the closed form

        C = (2 pi)^(-k/2) * [ V_k s^k + k V_k * sum_{j=0}^{k-1} binom(k-1, j) s^(k-1-j) m_j ]

        V_k = pi^(k/2) / Gamma(k/2 + 1)          (volume of the unit ball in R^k)
        m_j = 2^((j-1)/2) * Gamma((j+1)/2)        (= integral_0^inf t^j exp(-t^2/2) dt)

    and COMP = log2 C. The first term alone, (k/2) log2(n/2pi) + log2[V_k (R/sigma^2)^(k/2)], is
    eq. 2.21 for this model: log2 of the number of fitted curves the region holds at resolution
    ~1/sqrt(n). The sum is a boundary term (datasets whose fit lies just outside the ball are
    still fit well by its surface) that fades as n grows.

    Args:
        x: Design points (n,).
        degree: Polynomial degree.
        R: Bound on the mean squared fitted value (>= 0).
        sigma: Known noise standard deviation.

    Returns:
        COMP in bits (always >= 0, non-decreasing in degree, grows like (k/2) log2 R).

    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> nml_complexity_polynomial(x, 0, R=100.0)   # about 5.8 bits
    """
    # TODO: Implement the closed form.
    # Steps:
    #   1. n = len(x), k = degree + 1, s = sqrt(n R) / sigma
    #   2. Work in logs: ln V_k = (k/2) ln(pi) - lgamma(k/2 + 1); ln m_j = ((j-1)/2) ln 2 + lgamma((j+1)/2)
    #   3. Collect the k + 1 log-terms of the bracket, combine them with logsumexp (helper below)
    #   4. Subtract (k/2) ln(2 pi) and convert nats to bits (divide by ln 2)
    #   Careful with R = 0: s^0 = 1 but s^j = 0 for j > 0 (COMP is then exactly 0).

    pass  # Your code here


def estimate_nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    R: float,
    sigma: float = 1.0,
    n_samples: int = 200_000,
    seed: int = 0
) -> Tuple[float, float]:
    """
    Monte Carlo estimate (bits) of the same integral, for a TINY design (n <= 5 or so).

    Importance sampling: integral g(y) dy = E_q[ g(y) / q(y) ] for any density q that covers g.
    With q uniform on the box [-L, L]^n, g / q = (2L)^n * g(y): the factor (2L)^n is the box
    VOLUME. Forgetting it (as the old code did) gives a number n log2(2L) bits too small that
    changes when you change L.

    Algorithm:
      1. L = sqrt(n R) + 8 sigma  (any allowed mean has |mu_i| <= sqrt(n R); the integrand is
         negligible beyond 8 sigma from the allowed means)
      2. Draw n_samples datasets Y uniformly from [-L, L]^n          -> shape (n_samples, n)
      3. Least-squares fit of each dataset: project onto the column space. With an orthonormal
         basis Q (n, k) of [1, x, ..., x^degree] (np.linalg.qr), the fit's coordinates are
         Z = Y @ Q (shape (n_samples, k)).
      4. Restricted maximum likelihood: if ||Z|| > sqrt(n R), shrink Z onto the sphere of that
         radius (the closest allowed fit); fitted values Y_fit = Z_fit @ Q.T.
      5. g(y) = (2 pi sigma^2)^(-n/2) exp(-||y - y_fit||^2 / (2 sigma^2)); weight w = (2L)^n g(y).
      6. Estimate = log2(mean(w)), standard error = std(w) / (sqrt(n_samples) mean(w) ln 2).
         Work with log-weights and logsumexp to avoid overflow.

    Returns:
        (estimate_bits, standard_error_bits)
    """
    # TODO: Implement the Monte Carlo estimate (with the volume term!)

    pass  # Your code here


def compute_nml_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> Tuple[float, float, float]:
    """
    Refined-MDL code length (bits) with UNKNOWN sigma: Rissanen's renormalised NML (RNML).

    Rissanen (2000) restricts the data to sigma_hat^2 >= tau0 and R_hat <= R, which gives the
    finite integral
        C(tau0, R) = (4/k^2) (R/tau0)^(k/2) (n/(2e))^(n/2) / (Gamma(k/2) Gamma((n-k)/2)),
    then sets the restriction to the data's own values (tau0 = sigma_hat^2, R = R_hat) and
    renormalises over them (tutorial eq. 2.36). The constant of that second normalisation is
    the same for every degree, so it can be dropped. The result is

        total = [ (n-k)/2 ln sigma_hat^2 + k/2 ln R_hat - ln Gamma(k/2) - ln Gamma((n-k)/2)
                  + n/2 ln(n pi) ] / ln 2

    with k = degree + 1, sigma_hat^2 = RSS / n (ML noise variance) and R_hat = mean(y_hat^2)
    (the fitted curve's mean square). Split it as
        fit        = -log2 f(y | beta_hat, sigma_hat) = (n/2) log2(2 pi e sigma_hat^2)
        complexity = total - fit   (grows like (k/2) log2(R_hat / sigma_hat^2))

    Args:
        x: Input values
        y: Target values
        degree: Polynomial degree

    Returns:
        Tuple of (total_bits, fit_bits, complexity_bits)

    Example:
        >>> x = np.linspace(0, 10, 50)
        >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.0
        >>> total, fit, comp = compute_nml_mdl(x, y, 2)
        >>> print(f"NML = {fit:.1f} (fit) + {comp:.1f} (complexity) = {total:.1f}")
    """
    # TODO: Implement RNML
    # Steps:
    #   1. Fit the polynomial (np.polyfit / np.polyval), get sigma_hat^2 and R_hat
    #   2. total from the formula above (math.lgamma for ln Gamma), fit from the Gaussian NLL
    #   3. Return (total, fit, total - fit), all in bits

    pass  # Your code here


def compare_nml_vs_two_part(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict:
    """
    Select the degree with NML (compute_nml_mdl) and with the crude two-part code
    (two_part_mdl below, 33 bits per coefficient as in Exercise 1).

    The two-part code's model cost depends on an arbitrary choice (bits per coefficient);
    NML replaces it with a complexity derived from the model itself. They often agree when the
    signal is strong; the crude code is far more conservative (33 bits per coefficient against
    NML's few bits), so it under-fits when a coefficient is small.

    Returns:
        Dictionary with 'nml_best', 'two_part_best' (degrees) and 'nml_scores',
        'two_part_scores' (dicts degree -> bits)
    """
    # TODO: Implement comparison

    pass  # Your code here


def visualize_complexity_vs_degree(
    x: np.ndarray,
    max_degree: int = 10,
    R_values: Tuple[float, ...] = (1.0, 100.0, 10000.0),
    sigma: float = 1.0
) -> Dict[float, list]:
    """
    Print COMP (nml_complexity_polynomial) for every degree and several bounds R.

    What to look for: COMP grows with the degree (more coefficients, more distinguishable
    curves) and, for a fixed degree, grows by about (k/2) log2(100) = 3.3 k bits each time R
    grows 100x: NML's price depends on how you bound the parameters.

    Returns:
        Dictionary R -> list of COMP values (bits) for degrees 0..max_degree
    """
    # TODO: Compute and print the table (plotting is optional)

    pass  # Your code here


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def logsumexp(log_values: np.ndarray) -> float:
    """
    Compute log(sum(exp(values))) in a numerically stable way.

    This is essential for NML computation to avoid overflow.

    Args:
        log_values: Array of log values (may contain -inf, meaning a zero term)

    Returns:
        log(sum(exp(log_values)))
    """
    # TODO: Implement numerically stable logsumexp
    # Hint: log(sum exp(xi)) = max(xi) + log(sum exp(xi - max))

    pass  # Your code here


def gaussian_log_likelihood(
    y: np.ndarray,
    predictions: np.ndarray,
    sigma: float = None
) -> float:
    """
    Compute Gaussian log-likelihood.

    Args:
        y: Actual values
        predictions: Model predictions
        sigma: Standard deviation (estimated from residuals if None)

    Returns:
        Log-likelihood (not in bits, in nats)
    """
    # TODO: Implement Gaussian log-likelihood

    pass  # Your code here


def two_part_mdl(x: np.ndarray, y: np.ndarray, degree: int, precision_bits: int = 32) -> float:
    """Crude two-part code of Exercise 1 (provided): L(H) + L(D|H) in bits."""
    n = len(x)
    residuals = y - np.polyval(np.polyfit(x, y, degree), x)
    sigma = max(np.std(residuals), 1e-10)
    # Rissanen's universal code for the integer degree + 1 (log-star), as in Exercise 1
    length, current = np.log2(2.865064), float(degree + 1)
    while current > 1:
        length += np.log2(current)
        current = np.log2(current)
    model_cost = length + (degree + 1) * (precision_bits + 1)
    nll = 0.5 * n * np.log2(2 * np.pi * sigma**2) + np.sum(residuals**2) / (2 * sigma**2 * np.log(2))
    return model_cost + nll + precision_bits + 10   # + simplified sigma code, as in Exercise 1


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_implementation():
    """Test NML complexity implementation."""

    print("Testing NML (Normalized Maximum Likelihood) Implementation")
    print("=" * 55)

    np.random.seed(42)

    x = np.linspace(0, 10, 50)

    # Test 1: properties of the closed form
    print("\n1. Testing nml_complexity_polynomial() properties...")
    try:
        ok = True
        for R in (0.01, 1.0, 100.0, 1e4):
            comps = [nml_complexity_polynomial(x, d, R) for d in range(6)]
            print(f"   R = {R:>7}: " + "  ".join(f"{c:6.2f}" for c in comps))
            ok &= all(c >= 0 for c in comps) and all(b >= a for a, b in zip(comps, comps[1:]))
        growth = nml_complexity_polynomial(x, 2, 1e6) - nml_complexity_polynomial(x, 2, 1e4)
        print(f"   degree 2, R x100: +{growth:.3f} bits (expected (k/2) log2(100) = {1.5 * np.log2(100):.3f})")
        ok &= abs(growth - 1.5 * np.log2(100)) < 0.05
        assert ok, "COMP must be >= 0, non-decreasing in degree, and grow by (k/2) log2 of the R ratio"
        print("   [ok] PASSED - finite, >= 0, monotone in degree, grows like (k/2) log2 R")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")

    # Test 2: logsumexp
    print("\n2. Testing logsumexp()...")
    try:
        values = np.array([1000, 1001, 1002])  # Would overflow with naive exp
        result = logsumexp(values)

        # Should be approximately 1002 + log(1 + e^-1 + e^-2) ≈ 1002.41
        assert 1002 < result < 1003, f"Expected ~1002.4, got {result}"
        print(f"   logsumexp([1000, 1001, 1002]) = {result:.2f}")
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")

    # Test 3: Monte Carlo (with the volume term) vs closed form on tiny designs
    print("\n3. Testing estimate_nml_complexity_polynomial() against the closed form...")
    try:
        ok = True
        for n_tiny, deg, R in ((3, 0, 4.0), (4, 1, 2.0)):
            x_tiny = np.linspace(0, 10, n_tiny)
            est, se = estimate_nml_complexity_polynomial(x_tiny, deg, R, n_samples=400_000)
            exact = nml_complexity_polynomial(x_tiny, deg, R)
            print(f"   n={n_tiny}, degree {deg}, R={R}: Monte Carlo {est:.3f} +- {se:.3f}, closed form {exact:.3f}")
            ok &= abs(est - exact) < 4 * se + 0.02
        assert ok, "Monte Carlo and closed form disagree (did you include the box volume?)"
        print("   [ok] PASSED - Monte Carlo agrees with the closed form")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")

    # Test 4: NML MDL score
    print("\n4. Testing compute_nml_mdl()...")
    try:
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5  # True degree 2

        nml_1 = compute_nml_mdl(x, y, 1)
        nml_2 = compute_nml_mdl(x, y, 2)
        nml_5 = compute_nml_mdl(x, y, 5)

        print(f"   NML(degree=1): {nml_1[0]:.1f} = {nml_1[1]:.1f} + {nml_1[2]:.1f}")
        print(f"   NML(degree=2): {nml_2[0]:.1f} = {nml_2[1]:.1f} + {nml_2[2]:.1f}")
        print(f"   NML(degree=5): {nml_5[0]:.1f} = {nml_5[1]:.1f} + {nml_5[2]:.1f}")

        if nml_2[0] < nml_1[0] and nml_2[0] < nml_5[0]:
            print("   [ok] PASSED - NML correctly prefers degree 2!")
        else:
            print("   [NOTE] NML did not prefer degree 2 on this dataset")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")

    # Test 5: Compare NML vs Two-Part
    print("\n5. Testing compare_nml_vs_two_part()...")
    try:
        y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5

        result = compare_nml_vs_two_part(x, y, max_degree=6)

        print(f"   NML selects: degree {result['nml_best']}")
        print(f"   Two-Part selects: degree {result['two_part_best']}")

        if result['nml_best'] == result['two_part_best']:
            print("   [ok] PASSED - Methods agree!")
        else:
            print("   [NOTE] Methods disagree (both are valid codes)")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")

    # Test 6: Complexity table
    print("\n6. Testing visualize_complexity_vs_degree()...")
    try:
        table = visualize_complexity_vs_degree(x, max_degree=5)
        assert len(table) == 3 and all(len(v) == 6 for v in table.values())
        print("   [ok] PASSED")
    except Exception as e:
        print(f"   [FAIL] FAILED: {e}")

    print("\n" + "=" * 55)
    print("Testing complete!")
    print("\nKey insight: NML complexity grows with model flexibility, and it is only finite")
    print("once the parameters are bounded; how you bound them changes the price.")


if __name__ == "__main__":
    test_implementation()
