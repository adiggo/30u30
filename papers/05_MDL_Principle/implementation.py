"""
Day 5: MDL (Minimum Description Length) Principle - Implementation
=====================================================================

A complete implementation of MDL-based model selection, covering:
- Two-Part Codes (crude MDL)
- Prequential (Predictive) MDL
- Normalized Maximum Likelihood (NML)
- Practical model selection for polynomials and linear models

Paper: "A Tutorial Introduction to the Minimum Description Length Principle"
        by Peter Grünwald (2004)

Think of this as: A library for finding the SHORTEST description of your data.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from math import comb, lgamma  # [PATCH 2026-10-18] binomials and log-Gamma for the NML closed forms (Section 4)
import warnings


# =============================================================================
# SECTION 1: CORE MDL COMPUTATIONS
# =============================================================================

def log2(x: float) -> float:
    """Safe log base 2, returns 0 for x <= 0."""
    if x <= 0:
        return 0.0
    return np.log2(x)


def universal_code_integers(n: int) -> float:
    """
    Universal prior code length for positive integers.
    
    This is the "fair" way to encode integers when you don't know
    their range in advance. Used for encoding model parameters.
    
    L(n) = log*(n) + log(c₀)
    where log*(n) = log(n) + log(log(n)) + ... (only positive terms)
    
    Coding Analogy:
    ================
    Imagine telling a receiver a number without knowing the range.
    You can't just use 4 bits (that assumes max 15).
    
    Universal code says: "First tell me how many digits, then the number."
    This works for ANY number, big or small.
    
    Args:
        n: Positive integer to encode
        
    Returns:
        Code length in bits
    """
    if n <= 0:
        return float('inf')
    
    # Log-star: sum of iterated logs until negative
    c0 = 2.865064  # Normalization constant for valid prefix code
    length = log2(c0)
    
    current = float(n)
    while current > 1:
        length += log2(current)
        current = log2(current)
        if current <= 0:
            break
    
    return max(length, 0)


def precision_code_length(precision_bits: int = 32) -> float:
    """
    Code length for specifying precision of real numbers.
    
    When encoding real numbers, we must specify the precision.
    Higher precision = more bits needed.
    
    Args:
        precision_bits: Bits per floating-point number
        
    Returns:
        Length of precision specification in bits
    """
    return universal_code_integers(precision_bits)


def real_number_code_length(value: float, precision_bits: int = 32) -> float:
    """
    Code length for encoding a single real number.
    
    We need to encode:
    1. The sign (1 bit)
    2. The exponent (encoded with universal integer code)
    3. The mantissa (precision_bits)
    
    Args:
        value: Real number to encode
        precision_bits: Bits for mantissa
        
    Returns:
        Total code length in bits
    """
    if value == 0:
        return 1  # Just encode "zero" flag
    
    sign_bit = 1
    
    # Exponent encoding
    if abs(value) >= 1:
        exponent = int(np.floor(np.log2(abs(value)))) + 1
    else:
        exponent = -int(np.ceil(-np.log2(abs(value))))
    
    exponent_length = universal_code_integers(abs(exponent) + 1) + 1  # +1 for sign
    
    return sign_bit + exponent_length + precision_bits


# =============================================================================
# SECTION 2: TWO-PART CODE MDL
# =============================================================================

def two_part_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> Tuple[float, float, float]:
    """
    Compute Two-Part MDL score for polynomial regression.
    
    Two-Part Code = L(Model) + L(Data|Model)
    
    Coding Analogy (crude two-part MDL, tutorial Secs. 1.3 and 2.4; [PATCH 2026-10-18] section cite fixed):
    ==========================================
    You're sending temperature readings to a receiver.
    
    L(Model): Cost of sending the polynomial formula
              "Use: T = 23.5 + 0.02*hour - 0.001*hour²"
              
    L(Data|Model): Cost of sending the deviations
                   "Actual readings differ from formula by: +0.1, -0.3, ..."
    
    The best polynomial minimizes the TOTAL message length.
    
    Args:
        x: Input values (shape: [n])
        y: Target values (shape: [n])
        degree: Polynomial degree (0 = constant, 1 = line, etc.)
        precision_bits: Bits per floating-point coefficient
        
    Returns:
        Tuple of (total_mdl, model_cost, data_cost)
    """
    n = len(x)
    
    # Fit polynomial
    try:
        coefficients = np.polyfit(x, y, degree)
        poly = np.poly1d(coefficients)
        predictions = poly(x)
        residuals = y - predictions
    except np.linalg.LinAlgError:
        return float('inf'), float('inf'), 0.0
    
    # =========================================================================
    # PART 1: Model Description Length L(H)
    # =========================================================================
    
    # Cost to encode degree (which polynomial?)
    degree_cost = universal_code_integers(degree + 1)
    
    # Cost to encode coefficients
    num_params = degree + 1
    param_cost = num_params * (precision_bits + 1)  # +1 for sign
    
    # Cost to encode precision
    precision_cost = precision_code_length(precision_bits)
    
    model_cost = degree_cost + param_cost + precision_cost
    
    # =========================================================================
    # PART 2: Data Description Length L(D|H)
    # =========================================================================
    
    # We encode residuals using a Gaussian model
    # Under optimal Gaussian coding, each residual costs:
    # log(σ√(2πe)) + (residual²)/(2σ²ln2) bits
    
    sigma = np.std(residuals) if np.std(residuals) > 0 else 1e-10
    
    # Simplified: Use log-likelihood conversion
    # bits = -log₂(P(data|model))
    
    if sigma > 0:
        # Gaussian log-likelihood
        nll = 0.5 * n * np.log2(2 * np.pi * sigma**2) + \
              np.sum(residuals**2) / (2 * sigma**2 * np.log(2))
        
        # Add cost to encode sigma
        sigma_cost = real_number_code_length(sigma, precision_bits)
        data_cost = nll + sigma_cost
    else:
        data_cost = 0  # Perfect fit (rare)
    
    total_mdl = model_cost + data_cost
    
    return total_mdl, model_cost, data_cost


def select_polynomial_degree(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    precision_bits: int = 32
) -> Dict[str, Any]:
    """
    Select optimal polynomial degree using Two-Part MDL.
    
    This is the practical "just use this" function for polynomial model selection.
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = 3 + 2*x - 0.5*x**2 + np.random.randn(100) * 0.5  # True: degree 2
    >>> result = select_polynomial_degree(x, y)
    >>> print(f"Best degree: {result['best_degree']}")  # Should be 2
    
    Args:
        x: Input values
        y: Target values
        max_degree: Maximum degree to consider
        precision_bits: Bits per coefficient
        
    Returns:
        Dictionary with:
        - best_degree: Optimal polynomial degree
        - scores: Dict mapping degree -> MDL score
        - coefficients: Fitted coefficients for best model
        - model_cost: L(H) for best model
        - data_cost: L(D|H) for best model
    """
    scores = {}
    details = {}
    
    for degree in range(max_degree + 1):
        total, model, data = two_part_mdl_polynomial(x, y, degree, precision_bits)
        scores[degree] = total
        details[degree] = {'model_cost': model, 'data_cost': data}
    
    best_degree = min(scores, key=scores.get)
    
    # Get coefficients for best model
    coefficients = np.polyfit(x, y, best_degree)
    
    return {
        'best_degree': best_degree,
        'scores': scores,
        'details': details,
        'coefficients': coefficients,
        'model_cost': details[best_degree]['model_cost'],
        'data_cost': details[best_degree]['data_cost'],
        'total_mdl': scores[best_degree]
    }


# =============================================================================
# SECTION 3: PREQUENTIAL (PREDICTIVE) MDL
# =============================================================================

def prequential_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    start_index: int = None
) -> float:
    """
    Compute Prequential MDL score for polynomial regression.
    
    Prequential = "Predict Each, Then Learn From It"
    
    The Weather Forecaster Analogy:
    ================================
    Imagine a weather forecaster who:
    1. Predicts tomorrow's temperature (BEFORE seeing it)
    2. Records how surprised they are by the actual value
    3. Updates their model based on the new observation
    4. Repeat for each day
    
    Total "surprise" across all days = Prequential MDL score
    
    This is MORE ROBUST than two-part code because:
    - No need to specify parameter precision
    - Naturally handles model complexity
    - Better for time series and sequential data
    
    Mathematical Form:
    ===================
    L_prequential = -Σ log₂ P(yᵢ | y₁, ..., yᵢ₋₁, model)
    
    Args:
        x: Input values (shape: [n])
        y: Target values (shape: [n])
        degree: Polynomial degree
        start_index: Where to start predictions (default: degree + 2)
        
    Returns:
        Prequential MDL score in bits
    """
    n = len(x)
    
    # Need at least degree+1 points to fit, plus one to predict
    if start_index is None:
        start_index = degree + 2
    
    if start_index >= n:
        return float('inf')
    
    total_code_length = 0.0
    
    for i in range(start_index, n):
        # Fit model on points 0..i-1
        x_train = x[:i]
        y_train = y[:i]
        
        try:
            coefficients = np.polyfit(x_train, y_train, degree)
            poly = np.poly1d(coefficients)
            
            # Predict point i
            y_pred = poly(x[i])
            
            # Estimate variance from training residuals
            train_pred = poly(x_train)
            residuals = y_train - train_pred
            sigma = np.std(residuals) if np.std(residuals) > 1e-10 else 1e-10
            
            # Code length for this prediction (Gaussian model)
            error = y[i] - y_pred
            code_length = 0.5 * np.log2(2 * np.pi * sigma**2) + \
                          (error**2) / (2 * sigma**2 * np.log(2))
            
            total_code_length += code_length
            
        # [PATCH 2026-10-18] np.RankWarning was removed in NumPy 2.0; evaluating this except
        # clause raised AttributeError whenever polyfit failed. It lives in np.exceptions.
        except (np.linalg.LinAlgError, getattr(np, "exceptions", np).RankWarning):  # works on NumPy 1.x and 2.x
            total_code_length += 100  # Penalty for numerical issues
    
    return total_code_length


def prequential_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Any]:
    """
    Select best polynomial using Prequential MDL.
    
    Example:
    --------
    >>> x = np.linspace(0, 5, 50)
    >>> y = np.sin(x) + 0.1 * np.random.randn(50)
    >>> result = prequential_model_selection(x, y)
    >>> print(f"Best degree: {result['best_degree']}")
    
    Args:
        x: Input values
        y: Target values
        max_degree: Maximum degree to try
        
    Returns:
        Dictionary with best_degree, scores, etc.
    """
    scores = {}

    # [PATCH 2026-10-18] With the default start_index=degree+2 each model coded a different
    # number of points (48 for degree 0 ... 38 for degree 10), so totals were not comparable,
    # and low degrees paid for early near-zero-sigma predictions the high degrees skipped.
    # A prequential code length covers the same data for every model: use one start index.
    start = max_degree + 2
    for degree in range(max_degree + 1):
        scores[degree] = prequential_mdl_polynomial(x, y, degree, start_index=start)

    best_degree = min(scores, key=scores.get)
    coefficients = np.polyfit(x, y, best_degree)

    return {
        'best_degree': best_degree,
        'scores': scores,
        'coefficients': coefficients,
        'total_mdl': scores[best_degree]
    }


# =============================================================================
# SECTION 4: NORMALIZED MAXIMUM LIKELIHOOD (NML) -- REFINED MDL
# =============================================================================
#
# [PATCH 2026-10-18] Section redesigned. The old nml_complexity_polynomial averaged the
# maximised likelihood of y drawn uniformly from a box and never multiplied by the box volume:
# it returned about -380 bits (COMP is always >= 0), and widening the box 10x moved it by
# -n*log2(10). Worse, its target is infinite. For Gaussian models the Shtarkov integral
# COMP = log int f(y | theta_hat(y)) dy (eq. 2.17) diverges: a model that fits data centred at 0
# fits data centred at 10^6 just as well, so the integral over all datasets never converges
# (Example 2.20, Sec. 2.7.2). With sigma also fitted it diverges a second time, as the residual
# variance goes to 0. The tutorial says the same for regression (Sec. 2.8): "Ideally, it should be
# an NML distribution, but just as in Example 2.20, this NML distribution is not well-defined."
#
# What is computed now, following the tutorial's remedies (Sec. 2.7.2):
#   * nml_complexity_polynomial   COMP of the polynomial model with KNOWN noise level sigma and
#                                 the fitted curve restricted to a ball of "signal power":
#                                     (1/n) * sum_i (X beta)_i^2 <= R
#                                 Exact closed form. Finite, >= 0, grows with the degree and like
#                                 (k/2) log2 R with the restriction: NML's price depends on how
#                                 you bound the parameters, which is the tutorial's point.
#   * nml_complexity_asymptotic   the same quantity from eq. 2.21 (Fisher-information volume).
#   * nml_complexity_monte_carlo  Monte Carlo estimate of the same integral that keeps the
#                                 proposal density (the "volume term"), to check the closed form.
#   * nml_mdl_polynomial          sigma UNKNOWN: Rissanen's (2000) NML for linear regression,
#                                 restricted to sigma_hat^2 >= tau0 and R_hat <= R and then
#                                 renormalised (RNML, eq. 2.36), so no bound has to be picked by
#                                 hand. This is the code length used to select the degree.
#
# Notation used throughout: n data points, k = degree + 1 coefficients, X = [1, x, ..., x^degree]
# the (n, k) design matrix, y_hat = X beta_hat the least-squares fit, and "bits" = log base 2.


def _polynomial_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Orthonormal basis Q (shape (n, k)) of the column space of [1, x, ..., x^degree].

    Rescaling x changes neither the column space nor the fitted values, only the conditioning,
    so we divide by max|x| before building the Vandermonde matrix.
    """
    scale = np.max(np.abs(x))
    scale = scale if scale > 0 else 1.0
    V = np.vander(x / scale, degree + 1, increasing=True)  # (n, k): column j holds (x/scale)**j
    Q, _ = np.linalg.qr(V)                                  # Q: (n, k) with orthonormal columns, Q.T @ Q = I
    return Q


def _num_coefficients(x: np.ndarray, degree: int) -> int:
    """k = degree + 1, capped at the number of distinct x values (the rank of the design)."""
    return min(degree + 1, len(np.unique(x)))  # np.unique: sorted distinct values


def nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    R: float,
    sigma: float = 1.0,
    restrict: str = 'parameters'
) -> float:
    """
    Parametric complexity COMP (bits) of the restricted Gaussian polynomial model, sigma known.

    The model (tutorial eqs. 2.38-2.39 with sigma fixed):
        M_R = { y ~ N(X beta, sigma^2 I) : (1/n) ||X beta||^2 <= R }
    R bounds the mean squared value of the fitted curve (its "signal power", in units of y^2).
    In the space of fitted-value vectors mu = X beta this is a k-dimensional ball of radius
    r = sqrt(n R); in noise units its radius is s = r / sigma.

    Two ways to restrict (both in the tutorial's Sec. 2.7.2):
      restrict='parameters' (default; Foster & Stine's option): the model only contains the
          means inside the ball, so the maximum likelihood is taken over the ball:
              COMP = log2 integral over ALL y of max_{mu in ball} N(y; mu, sigma^2 I) dy
          This is the minimax regret (eqs. 2.16-2.19) of the restricted model. It is >= 0 (the
          integrand is >= any single density, which integrates to 1) and non-decreasing in the
          degree (a degree-d ball sits inside the degree-(d+1) ball), for every R >= 0.
      restrict='data' (Example 2.20 / eq. 2.33's "conditional" version): integrate the
          unrestricted maximised likelihood only over datasets whose fit lies in the ball.
          This equals eq. 2.21 exactly. It can be negative when R < sigma^2 (the allowed region
          holds less than one distinguishable curve); it is >= 0 and non-decreasing in the
          degree when R >= sigma^2 and degree + 3 <= n.

    Closed forms (k = number of coefficients, V_k = pi^(k/2) / Gamma(k/2 + 1) = volume of the
    unit ball in R^k):
        'data':        2^COMP = V_k * s^k / (2 pi)^(k/2)
                       i.e. COMP = (k/2) log2(n / (2 pi)) + log2[ V_k (R / sigma^2)^(k/2) ]
                       For degree 0 (k = 1, the Gaussian location model, mean in [-K, K] with
                       K = sqrt(R)) this is log2 K + 1/2 log2(n/2pi) - log2 sigma + 1,
                       exactly Example 2.20's formula.
        'parameters':  2^COMP = (2 pi)^(-k/2) * [ V_k s^k
                                 + k V_k * sum_{j=0}^{k-1} C(k-1, j) s^(k-1-j) m_j ],
                       m_j = int_0^inf t^j exp(-t^2/2) dt = 2^((j-1)/2) Gamma((j+1)/2).
                       The first term is the 'data' version; the sum is a boundary term:
                       datasets whose least-squares fit lies just outside the ball are still fit
                       well by the nearest point on its surface. It adds a relative
                       ~ k sqrt(pi/2) / s, so the two versions agree as n (hence s) grows.
    Derivation: write y = (projection onto the column space) + (orthogonal part). The orthogonal
    part's Gaussian factor integrates to 1; what remains is a k-dimensional integral of
    exp(-dist(z, ball)^2 / 2) over z in R^k, done in polar coordinates.

    Intuition for each piece of the 'data' formula, 2^COMP = (n/2pi)^(k/2) * V_k (R/sigma^2)^(k/2):
      V_k (R / sigma^2)^(k/2): the allowed region's volume measured with the Fisher information
          I(beta) = X^T X / (n sigma^2), i.e. int sqrt(det I) d beta over the region (eq. 2.21).
          It is a k-ball of radius sqrt(R)/sigma: the signal-to-noise ratio. The design matrix
          cancels, so with this restriction COMP depends on k, n and R/sigma^2 only.
      (n / 2pi)^(k/2): with n observations, curves closer than about sqrt(2 pi / n) (in those
          units) along each of the k directions cannot be told apart.
      Product: the number of distinguishable fitted curves in the region at resolution
          ~ 1/sqrt(n). COMP is log2 of that count (Sec. 2.6.2, counting interpretation).
    Restriction-size dependence: COMP grows by (k/2) * log2(10) ~ 1.66 k bits for every 10x
    increase of R. There is no "natural" R: this is why the tutorial pays for R with a
    meta-two-part code (eq. 2.34) or renormalises (eq. 2.36), and warns against cutting the
    range at an arbitrary value (Sec. 2.10.2). nml_mdl_polynomial below does the latter.

    Args:
        x: Design points (shape (n,)).
        degree: Polynomial degree (k = degree + 1 coefficients).
        R: Bound on the mean squared fitted value, (1/n) sum_i (X beta)_i^2 <= R. Must be >= 0.
        sigma: Known noise standard deviation.
        restrict: 'parameters' (default) or 'data' (see above).

    Returns:
        COMP in bits.
    """
    if R < 0 or sigma <= 0:
        raise ValueError("need R >= 0 and sigma > 0")
    n = len(x)
    k = _num_coefficients(x, degree)
    s = np.sqrt(n * R) / sigma          # radius of the allowed ball, in units of the noise sigma
    log_vk = 0.5 * k * np.log(np.pi) - lgamma(0.5 * k + 1)   # ln V_k (unit-ball volume in R^k)

    if restrict == 'data':
        if s == 0:
            return float('-inf')        # an empty region holds no datasets
        ln_c = log_vk + k * np.log(s) - 0.5 * k * np.log(2 * np.pi)
        return float(ln_c / np.log(2))
    if restrict != 'parameters':
        raise ValueError("restrict must be 'parameters' or 'data'")

    # ln of each term of the bracket [V_k s^k + k V_k sum_j C(k-1,j) s^(k-1-j) m_j];
    # -inf marks a term that is exactly 0 (positive powers of s when R = 0).
    with np.errstate(divide='ignore'):
        log_s = np.log(s)               # -inf when s == 0
    log_terms = [log_vk + k * log_s if s > 0 else -np.inf]
    for j in range(k):
        log_m_j = 0.5 * (j - 1) * np.log(2) + lgamma(0.5 * (j + 1))
        power = k - 1 - j
        log_s_power = power * log_s if power > 0 else 0.0   # s^0 = 1 even when s = 0
        if np.isneginf(log_s_power):
            log_terms.append(-np.inf)
            continue
        log_terms.append(np.log(k) + log_vk + np.log(comb(k - 1, j)) + log_s_power + log_m_j)
    log_terms = np.array(log_terms)                          # shape (k + 1,)
    top = np.max(log_terms)
    # log-sum-exp: ln sum_i e^{a_i} = top + ln sum_i e^{a_i - top}; avoids overflow for large s
    ln_bracket = top + np.log(np.sum(np.exp(log_terms - top)))
    ln_c = ln_bracket - 0.5 * k * np.log(2 * np.pi)
    return float(max(ln_c, 0.0) / np.log(2))   # max() only removes -1e-16 round-off at R = 0 (exact value 0)


def nml_complexity_asymptotic(
    x: np.ndarray,
    degree: int,
    R: float,
    sigma: float = 1.0
) -> float:
    """
    Eq. 2.21 for the restricted model of nml_complexity_polynomial (bits):

        COMP ~ (k/2) log2(n / (2 pi)) + log2 int_{region} sqrt(det I(beta)) d beta

    computed "the long way", without using the cancellation:
      I(beta) = X^T X / (n sigma^2)  -- the tutorial's per-observation Fisher information for a
          non-i.i.d. model, (1/n) E[-d^2/dbeta^2 log f(y | beta)]; constant in beta here;
      region  = ellipsoid {beta : beta^T X^T X beta <= n R} with volume
                V_k (n R)^(k/2) / sqrt(det X^T X).
    The determinants cancel, so this equals nml_complexity_polynomial(..., restrict='data')
    exactly; the default 'parameters' version approaches it as n grows (eq. 2.21's o(1)).
    Eq. 2.21 needs a finite Fisher volume, which is why the region must be bounded.
    """
    n = len(x)
    k = degree + 1
    if k > len(np.unique(x)):
        raise ValueError("eq. 2.21 needs a full-rank design: degree + 1 <= number of distinct x")
    X = np.vander(x, k, increasing=True)          # (n, k): columns 1, x, x^2, ..., x^degree
    # log det(X^T X) from a QR factorisation, X = Q Rq: det(X^T X) = prod(diag(Rq))^2.
    # Forming X^T X explicitly would square the (large) condition number of the Vandermonde matrix.
    Rq = np.linalg.qr(X, mode='r')                # (k, k) upper-triangular factor
    log_det_xtx = 2 * np.sum(np.log(np.abs(np.diag(Rq))))
    log_vk = 0.5 * k * np.log(np.pi) - lgamma(0.5 * k + 1)
    log_sqrt_det_fisher = 0.5 * (log_det_xtx - k * np.log(n * sigma**2))   # ln sqrt(det I)
    log_volume = log_vk + 0.5 * k * np.log(n * R) - 0.5 * log_det_xtx       # ln vol(ellipsoid)
    ln_comp = 0.5 * k * np.log(n / (2 * np.pi)) + log_sqrt_det_fisher + log_volume
    return float(ln_comp / np.log(2))


def nml_complexity_monte_carlo(
    x: np.ndarray,
    degree: int,
    R: float,
    sigma: float = 1.0,
    n_samples: int = 200_000,
    restrict: str = 'parameters',
    proposal: str = 'box',
    box_halfwidth: Optional[float] = None,
    seed: int = 0,
    batch_size: int = 50_000
) -> Tuple[float, float]:
    """
    Monte Carlo estimate (bits) of the Shtarkov integral of nml_complexity_polynomial, with its
    standard error. Only practical for small n: the integrand lives in a thin slab around the
    k-dimensional column space, which a proposal over R^n rarely hits when n is large.

    Importance sampling: integral of g(y) dy = E_q[ g(y) / q(y) ] for any density q that is
    positive wherever g is. With q uniform on the box [-L, L]^n, 1/q(y) = (2L)^n is the box
    VOLUME -- the factor the old estimator dropped, which made it about -n log2(2L) bits too
    small and dependent on L. The default L = sqrt(n R) + 8 sigma covers all but a negligible
    part of the integrand (any allowed mean has |mu_i| <= sqrt(n R)).

    Args:
        x, degree, R, sigma, restrict: as in nml_complexity_polynomial.
        n_samples: number of simulated datasets.
        proposal: 'box' (uniform on [-L, L]^n) or 'gaussian' (N(0, tau^2 I), tau = sqrt(R) + 2 sigma).
        box_halfwidth: L for the 'box' proposal (default sqrt(n R) + 8 sigma).
        seed: random seed.
        batch_size: datasets simulated per batch (bounds memory).

    Returns:
        (estimate_bits, standard_error_bits)
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    Q = _polynomial_basis(x, degree)              # (n, k)
    r = np.sqrt(n * R)                            # radius of the allowed ball of fitted-value vectors
    if proposal == 'box':
        L = box_halfwidth if box_halfwidth is not None else r + 8 * sigma
    elif proposal == 'gaussian':
        tau = np.sqrt(R) + 2 * sigma
    else:
        raise ValueError("proposal must be 'box' or 'gaussian'")

    log_w_batches = []
    remaining = n_samples
    while remaining > 0:
        m = min(batch_size, remaining)
        remaining -= m
        if proposal == 'box':
            Y = rng.uniform(-L, L, size=(m, n))   # (m, n): m simulated datasets
            log_q = np.full(m, -n * np.log(2 * L))   # log density of the uniform box; -log q = log volume
        else:
            Y = tau * rng.standard_normal((m, n))
            log_q = -0.5 * n * np.log(2 * np.pi * tau**2) - np.sum(Y**2, axis=1) / (2 * tau**2)
        Z = Y @ Q                                 # (m, n) @ (n, k) -> (m, k): coordinates of each fit
        rho = np.linalg.norm(Z, axis=1)           # (m,): length of each least-squares fit = sqrt(n R_hat)
        if restrict == 'parameters':
            # Maximum likelihood over the ball: a fit outside it is pulled back onto its surface.
            shrink = np.minimum(1.0, r / np.maximum(rho, 1e-300))   # (m,)
            Z_fit = Z * shrink[:, None]           # (m, k) * (m, 1): broadcasting rescales each row
            inside = np.ones(m, dtype=bool)
        elif restrict == 'data':
            Z_fit = Z                             # unrestricted fit, but only datasets with fit in the ball count
            inside = rho <= r
        else:
            raise ValueError("restrict must be 'parameters' or 'data'")
        Y_fit = Z_fit @ Q.T                       # (m, k) @ (k, n) -> (m, n): fitted values
        resid2 = np.sum((Y - Y_fit) ** 2, axis=1) # (m,): squared residual norm per dataset
        log_f = -0.5 * n * np.log(2 * np.pi * sigma**2) - resid2 / (2 * sigma**2)  # maximised log-likelihood
        # importance weight g/q, with g = 0 outside the data region ('data' restriction)
        log_w_batches.append(np.where(inside, log_f - log_q, -np.inf))
    log_w = np.concatenate(log_w_batches)         # (n_samples,)
    if not np.any(np.isfinite(log_w)):
        return float('-inf'), float('inf')
    top = np.max(log_w)
    w = np.exp(log_w - top)                       # weights rescaled into (0, 1] to avoid overflow
    mean_w = np.mean(w)
    se_w = np.std(w) / np.sqrt(len(w))
    estimate_bits = (top + np.log(mean_w)) / np.log(2)
    se_bits = se_w / mean_w / np.log(2)           # delta method: d(log m) = dm / m
    return float(estimate_bits), float(se_bits)


def nml_complexity_unknown_sigma(n: int, k: int, tau0: float, R: float) -> float:
    """
    Restricted complexity (bits) when sigma is ALSO fitted (Rissanen 2000, "MDL denoising").

    log2 of the integral of the maximised likelihood f(y | beta_hat(y), sigma_hat(y)) over the
    datasets with  sigma_hat^2 >= tau0  and  R_hat = (1/n) ||y_hat||^2 <= R:

        C(tau0, R) = (4 / k^2) * (R / tau0)^(k/2) * (n / (2e))^(n/2) / (Gamma(k/2) Gamma((n-k)/2))

    Without the lower bound tau0 the integral diverges as sigma_hat -> 0 (a curve through
    every point has infinite density); without R it diverges like Example 2.20. The bound pair
    enters only through the ratio R / tau0, a signal-to-noise ratio, as (k/2) log2(R/tau0).
    """
    if k >= n:
        return float('inf')
    ln_c = (np.log(4.0 / k**2) + 0.5 * k * np.log(R / tau0) + 0.5 * n * np.log(n / (2 * np.e))
            - lgamma(0.5 * k) - lgamma(0.5 * (n - k)))
    return float(ln_c / np.log(2))


def nml_mdl_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> Tuple[float, float, float]:
    """
    Refined-MDL code length (bits) of y under the degree-`degree` polynomial model with unknown
    noise level: Rissanen's renormalised NML (RNML) for linear regression (Rissanen 2000; the
    tutorial's eq. 2.36 and Sec. 2.8). Smaller is better; select the degree that minimises it.

    Construction:
      1. Restricted NML (nml_complexity_unknown_sigma): normalise the maximised likelihood over
         the datasets with sigma_hat^2 >= tau0 and R_hat <= R. Code length = fit + log2 C(tau0, R).
      2. Use the tightest restriction that still contains y: tau0 = sigma_hat^2(y), R = R_hat(y).
         Then renormalise this over the datasets whose (sigma_hat^2, R_hat) lie in fixed ranges
         [tau1, tau2] x [R1, R2] (eq. 2.36). The second normaliser is exactly
         (k^2/4) * ln(R2/R1) * ln(tau2/tau1); its k^2/4 cancels the 4/k^2 of step 1, and the
         ln-ln factor is the same for every degree, so it is dropped (it cannot change which
         degree wins). The ranges only need to contain the fits of every candidate degree.
    Result (bits), with sigma_hat^2 = RSS/n and R_hat = ||y_hat||^2 / n:
        L(y | k) = [ (n-k)/2 ln sigma_hat^2 + k/2 ln R_hat - ln Gamma(k/2) - ln Gamma((n-k)/2)
                     + n/2 ln(n pi) ] / ln 2
    The model index itself is coded uniformly over the candidate degrees (constant, dropped;
    Sec. 2.7.1). R_hat is measured from y = 0, so the code length depends on the origin of y
    (the tutorial's "region indifference" caveat after eq. 2.36).

    Args:
        x: Input values (n,).
        y: Target values (n,).
        degree: Polynomial degree.

    Returns:
        (total_bits, fit_bits, complexity_bits), with
          fit_bits        = -log2 f(y | beta_hat, sigma_hat) = (n/2) log2(2 pi e sigma_hat^2)
          complexity_bits = total - fit = log2 C(sigma_hat^2, R_hat) + log2(k^2/4)
                          = (k/2) log2(R_hat / sigma_hat^2) + (n/2) log2(n/(2e))
                            - log2 Gamma(k/2) - log2 Gamma((n-k)/2): it grows like
                            (k/2) log2(signal-to-noise ratio of the fit), the data-driven version
                            of nml_complexity_polynomial's (k/2) log2(R / sigma^2).
    """
    n = len(x)
    k = _num_coefficients(x, degree)
    if k >= n:
        return float('inf'), float('inf'), float('inf')    # no residual degrees of freedom left
    coefficients = np.polyfit(x, y, degree)
    y_hat = np.polyval(coefficients, x)                      # (n,): least-squares fit
    sigma2_hat = np.sum((y - y_hat) ** 2) / n                # ML noise variance RSS / n
    R_hat = np.sum(y_hat ** 2) / n                           # mean squared fitted value ("signal power")
    if not (sigma2_hat > 0 and R_hat > 0):
        return float('inf'), float('inf'), float('inf')    # exact fit or all-zero fit: region is empty
    fit_bits = float(0.5 * n * np.log2(2 * np.pi * np.e * sigma2_hat))
    complexity_bits = (nml_complexity_unknown_sigma(n, k, sigma2_hat, R_hat)   # step 1
                       + float(np.log2(k**2 / 4.0)))                           # step 2 (renormalisation)
    return fit_bits + complexity_bits, fit_bits, complexity_bits


def nml_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Any]:
    """
    Select the polynomial degree with the RNML code length (nml_mdl_polynomial).

    Returns:
        Dictionary with best_degree, scores (total bits), fit and complexity per degree,
        coefficients of the selected model and total_mdl.
    """
    scores, fits, complexities = {}, {}, {}
    for degree in range(max_degree + 1):
        scores[degree], fits[degree], complexities[degree] = nml_mdl_polynomial(x, y, degree)
    best_degree = min(scores, key=scores.get)
    return {
        'best_degree': best_degree,
        'scores': scores,
        'fit': fits,
        'complexity': complexities,
        'coefficients': np.polyfit(x, y, best_degree),
        'total_mdl': scores[best_degree]
    }


# =============================================================================
# SECTION 5: MODEL CLASSES FOR GENERAL MDL
# =============================================================================

@dataclass
class MDLModel(ABC):
    """
    Abstract base class for MDL-compatible models.
    
    To create your own MDL model:
    1. Inherit from MDLModel
    2. Implement fit(), predict(), model_code_length(), and data_code_length()
    
    Example:
    --------
    class MyCustomModel(MDLModel):
        def fit(self, x, y): ...
        def predict(self, x): ...
        def model_code_length(self): ...
        def data_code_length(self, x, y): ...
    """
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit model to data."""
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def model_code_length(self) -> float:
        """Return L(H) - bits to describe the model."""
        pass
    
    @abstractmethod
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return L(D|H) - bits to describe data given model."""
        pass
    
    def total_mdl(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return total MDL = L(H) + L(D|H)."""
        return self.model_code_length() + self.data_code_length(x, y)


class PolynomialMDL(MDLModel):
    """
    Polynomial model with MDL scoring.
    
    Example:
    --------
    >>> model = PolynomialMDL(degree=2)
    >>> model.fit(x, y)
    >>> score = model.total_mdl(x, y)
    >>> predictions = model.predict(x_test)
    """
    
    def __init__(self, degree: int, precision_bits: int = 32):
        self.degree = degree
        self.precision_bits = precision_bits
        self.coefficients = None
        self.sigma = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit polynomial to data."""
        self.coefficients = np.polyfit(x, y, self.degree)
        self.poly = np.poly1d(self.coefficients)
        residuals = y - self.poly(x)
        self.sigma = np.std(residuals) if np.std(residuals) > 0 else 1e-10
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y values."""
        if self.coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.poly(x)
    
    def model_code_length(self) -> float:
        """Bits to describe polynomial model."""
        if self.coefficients is None:
            return float('inf')
        
        # Degree specification
        degree_cost = universal_code_integers(self.degree + 1)
        
        # Coefficients
        num_params = len(self.coefficients)
        param_cost = num_params * (self.precision_bits + 1)
        
        return degree_cost + param_cost
    
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Bits to describe residuals."""
        if self.coefficients is None:
            self.fit(x, y)
        
        predictions = self.predict(x)
        residuals = y - predictions
        n = len(y)
        
        # Gaussian coding
        nll = 0.5 * n * np.log2(2 * np.pi * self.sigma**2) + \
              np.sum(residuals**2) / (2 * self.sigma**2 * np.log(2))
        
        # Add sigma encoding cost
        sigma_cost = real_number_code_length(self.sigma, self.precision_bits)
        
        return nll + sigma_cost


class GaussianMDL(MDLModel):
    """
    Simple Gaussian model with MDL scoring.
    
    Models data as y ~ N(μ, σ²).
    
    Useful as a baseline - if a more complex model can't beat this,
    there's no significant structure in the data.
    """
    
    def __init__(self, precision_bits: int = 32):
        self.precision_bits = precision_bits
        self.mean = None
        self.std = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit Gaussian to y values (ignores x)."""
        self.mean = np.mean(y)
        self.std = np.std(y) if np.std(y) > 0 else 1e-10
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict mean everywhere."""
        return np.full(len(x), self.mean)
    
    def model_code_length(self) -> float:
        """Bits to describe μ and σ."""
        if self.mean is None:
            return float('inf')
        
        return (real_number_code_length(self.mean, self.precision_bits) +
                real_number_code_length(self.std, self.precision_bits))
    
    def data_code_length(self, x: np.ndarray, y: np.ndarray) -> float:
        """Bits to describe deviations from mean."""
        if self.mean is None:
            self.fit(x, y)
        
        n = len(y)
        nll = 0.5 * n * np.log2(2 * np.pi * self.std**2) + \
              np.sum((y - self.mean)**2) / (2 * self.std**2 * np.log(2))
        
        return nll


# =============================================================================
# SECTION 6: COMPARISON WITH AIC/BIC
# =============================================================================

def aic_score(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> float:
    """
    Compute AIC (Akaike Information Criterion) score.
    
    AIC = 2k - 2ln(L)
    
    where k = number of parameters, L = maximum likelihood.
    
    Lower is better.
    """
    n = len(x)
    k = degree + 2  # Polynomial coeffs + variance
    
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        rss = np.sum(residuals**2)
        
        # Log-likelihood (Gaussian)
        sigma_ml = np.sqrt(rss / n)
        if sigma_ml <= 0:
            sigma_ml = 1e-10
        
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
        
        return 2 * k - 2 * log_likelihood
        
    except:
        return float('inf')


def bic_score(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> float:
    """
    Compute BIC (Bayesian Information Criterion) score.
    
    BIC = k*ln(n) - 2ln(L)
    
    where k = number of parameters, n = sample size, L = maximum likelihood.
    
    Lower is better. BIC has a stronger penalty than AIC for large n.
    """
    n = len(x)
    k = degree + 2  # Polynomial coeffs + variance
    
    try:
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        residuals = y - poly(x)
        rss = np.sum(residuals**2)
        
        # Log-likelihood (Gaussian)
        sigma_ml = np.sqrt(rss / n)
        if sigma_ml <= 0:
            sigma_ml = 1e-10
        
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
        
        return k * np.log(n) - 2 * log_likelihood
        
    except:
        return float('inf')


def compare_mdl_aic_bic(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Any]:
    """
    Compare MDL, AIC, and BIC for polynomial model selection.
    
    This is the "showdown" function to see how the criteria differ.
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2 + 3*x - 0.5*x**2 + np.random.randn(50) * 2  # True degree: 2
    >>> results = compare_mdl_aic_bic(x, y)
    >>> print(f"MDL picks: {results['mdl_best']}")
    >>> print(f"AIC picks: {results['aic_best']}")
    >>> print(f"BIC picks: {results['bic_best']}")
    
    Returns:
        Dictionary with scores and best choices for each criterion.
    """
    mdl_scores = {}
    aic_scores = {}
    bic_scores = {}
    
    for degree in range(max_degree + 1):
        mdl_scores[degree] = two_part_mdl_polynomial(x, y, degree)[0]
        aic_scores[degree] = aic_score(x, y, degree)
        bic_scores[degree] = bic_score(x, y, degree)
    
    return {
        'mdl_scores': mdl_scores,
        'aic_scores': aic_scores,
        'bic_scores': bic_scores,
        'mdl_best': min(mdl_scores, key=mdl_scores.get),
        'aic_best': min(aic_scores, key=aic_scores.get),
        'bic_best': min(bic_scores, key=bic_scores.get)
    }


# =============================================================================
# SECTION 7: UTILITY FUNCTIONS
# =============================================================================

def compression_ratio(
    data: np.ndarray,
    model_mdl: float
) -> float:
    """
    Compute compression ratio achieved by MDL model.
    
    Ratio > 1: Model compresses data (good!)
    Ratio < 1: Model expands data (bad!)
    Ratio = 1: No compression (data might be random)
    
    Args:
        data: Original data array
        model_mdl: MDL score of the model
        
    Returns:
        Compression ratio
    """
    # Raw data cost: float32 per element
    raw_bits = len(data) * 32
    
    return raw_bits / model_mdl if model_mdl > 0 else 0


def mdl_model_probability(
    mdl_scores: Dict[int, float]
) -> Dict[int, float]:
    """
    Convert MDL scores to approximate posterior probabilities.
    
    Using: P(model) ∝ 2^(-MDL score)
    
    This gives an intuitive sense of how much better one model is.
    
    Args:
        mdl_scores: Dict mapping model ID -> MDL score
        
    Returns:
        Dict mapping model ID -> probability
    """
    # Normalize to prevent overflow
    min_score = min(mdl_scores.values())
    normalized = {k: -(v - min_score) for k, v in mdl_scores.items()}
    
    # Convert to probabilities
    powers = {k: 2**v for k, v in normalized.items()}
    total = sum(powers.values())
    
    return {k: v / total for k, v in powers.items()}


def generate_test_data(
    true_degree: int = 2,
    n_points: int = 50,
    noise_std: float = 1.0,
    x_range: Tuple[float, float] = (0, 10),
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic polynomial data for testing.
    
    Args:
        true_degree: True polynomial degree
        n_points: Number of data points
        noise_std: Standard deviation of Gaussian noise
        x_range: Range of x values
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (x, y_noisy, true_coefficients)
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Generate random coefficients
    true_coeffs = np.random.randn(true_degree + 1) * 2
    poly = np.poly1d(true_coeffs)
    
    y_true = poly(x)
    y_noisy = y_true + np.random.randn(n_points) * noise_std
    
    return x, y_noisy, true_coeffs


# =============================================================================
# SECTION 8: MAIN DEMONSTRATION
# =============================================================================

def main():
    """
    Demonstrate MDL model selection on synthetic data.
    """
    print("=" * 70)
    print("MDL (Minimum Description Length) Principle - Implementation Demo")
    print("=" * 70)
    print()
    
    # Generate test data
    np.random.seed(42)
    true_degree = 2
    x, y, true_coeffs = generate_test_data(
        true_degree=true_degree,
        n_points=50,
        noise_std=1.5,
        seed=42
    )
    
    print(f"Generated data with TRUE degree = {true_degree}")
    print(f"True coefficients: {true_coeffs}")
    print(f"Number of points: {len(x)}")
    print()
    
    # =========================================================================
    # Two-Part Code MDL
    # =========================================================================
    print("-" * 50)
    print("METHOD 1: Two-Part Code MDL")
    print("-" * 50)
    
    result = select_polynomial_degree(x, y, max_degree=10)
    
    print(f"\nMDL Scores by Degree:")
    for deg in range(6):
        score = result['scores'][deg]
        marker = " <-- BEST" if deg == result['best_degree'] else ""
        print(f"  Degree {deg}: {score:.2f} bits{marker}")
    
    print(f"\nTwo-Part MDL selects: Degree {result['best_degree']}")
    print(f"  Model cost: {result['model_cost']:.2f} bits")
    print(f"  Data cost: {result['data_cost']:.2f} bits")
    print(f"  Total: {result['total_mdl']:.2f} bits")
    
    # =========================================================================
    # Prequential MDL
    # =========================================================================
    print()
    print("-" * 50)
    print("METHOD 2: Prequential MDL")
    print("-" * 50)
    
    preq_result = prequential_model_selection(x, y, max_degree=10)
    
    print(f"\nPrequential MDL Scores:")
    for deg in range(6):
        score = preq_result['scores'][deg]
        marker = " <-- BEST" if deg == preq_result['best_degree'] else ""
        print(f"  Degree {deg}: {score:.2f} bits{marker}")
    
    print(f"\nPrequential MDL selects: Degree {preq_result['best_degree']}")

    # =========================================================================
    # [PATCH 2026-10-18] Refined MDL: NML (restricted complexity + Rissanen's RNML)
    # =========================================================================
    print()
    print("-" * 50)
    print("METHOD 3: Refined MDL (NML)")
    print("-" * 50)

    nml_result = nml_model_selection(x, y, max_degree=10)
    print("\nRNML code length = fit + complexity (bits), sigma unknown:")
    for deg in range(6):
        marker = " <-- BEST" if deg == nml_result['best_degree'] else ""
        print(f"  Degree {deg}: {nml_result['fit'][deg]:8.2f} + {nml_result['complexity'][deg]:6.2f}"
              f" = {nml_result['scores'][deg]:8.2f} bits{marker}")
    print(f"\nNML selects: Degree {nml_result['best_degree']}")

    # Parametric complexity of the restricted model (sigma known): finite once the fitted curve's
    # mean square is bounded by R; it grows with the degree and like (k/2) log2 R with the bound.
    sigma_est = np.sqrt(np.mean((y - np.polyval(nml_result['coefficients'], x))**2))
    R_data = np.mean(np.polyval(np.polyfit(x, y, 10), x)**2)   # signal power of the largest fit
    print(f"\nCOMP of the restricted model, sigma = {sigma_est:.2f} known, mean squared fit <= R:")
    for R in (R_data, 100 * R_data):
        comps = [nml_complexity_polynomial(x, d, R, sigma_est) for d in range(6)]
        print(f"  R = {R:9.1f}: " + "  ".join(f"d{d} {c:5.1f}" for d, c in enumerate(comps)))
    print("  (x100 in R adds (k/2) log2(100) = 3.3 bits per coefficient: NML's price depends on the bound)")

    # =========================================================================
    # Comparison with AIC and BIC
    # =========================================================================
    print()
    print("-" * 50)
    print("COMPARISON: MDL vs AIC vs BIC")
    print("-" * 50)
    
    comparison = compare_mdl_aic_bic(x, y, max_degree=10)
    
    print(f"\nBest degree selected by each criterion:")
    print(f"  Two-part MDL: Degree {comparison['mdl_best']}")
    print(f"  NML (RNML): Degree {nml_result['best_degree']}")      # [PATCH 2026-10-18] refined MDL
    print(f"  Prequential: Degree {preq_result['best_degree']}")
    print(f"  AIC: Degree {comparison['aic_best']}")
    print(f"  BIC: Degree {comparison['bic_best']}")
    print(f"  True: Degree {true_degree}")
    
    # =========================================================================
    # Model Probabilities
    # =========================================================================
    print()
    print("-" * 50)
    print("MDL MODEL PROBABILITIES")
    print("-" * 50)
    
    probs = mdl_model_probability(result['scores'])
    
    print("\nPosterior probability of each degree:")
    for deg in range(6):
        bar = "█" * int(probs[deg] * 50)
        print(f"  Degree {deg}: {probs[deg]:.4f} {bar}")
    
    # =========================================================================
    # Compression Analysis
    # =========================================================================
    print()
    print("-" * 50)
    print("COMPRESSION ANALYSIS")
    print("-" * 50)
    
    raw_bits = len(y) * 32
    best_mdl = result['total_mdl']
    ratio = compression_ratio(y, best_mdl)
    
    print(f"\nRaw data size: {raw_bits} bits ({len(y)} × 32-bit floats)")
    print(f"MDL compressed: {best_mdl:.0f} bits")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"\n{'[ok] Found compressible structure!' if ratio > 1 else '[FAIL] Data appears random.'}")
    
    print()
    print("=" * 70)
    print("Demo complete! Run visualization.py for plots.")
    print("=" * 70)


if __name__ == "__main__":
    main()
