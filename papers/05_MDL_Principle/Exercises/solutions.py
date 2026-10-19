"""
Solutions for Day 5: MDL (Minimum Description Length) Exercises
================================================================

WARNING: Try the exercises yourself first!
These solutions are for reference after you've attempted each problem.

Each solution includes:
1. Complete working code
2. Explanation of the approach
3. Key insights
"""

import numpy as np
from math import comb, lgamma  # [PATCH 2026-10-18] for the exercise-4 closed forms
from typing import Tuple, Dict, List


# =============================================================================
# EXERCISE 1 SOLUTIONS: Two-Part Code
# =============================================================================

def solution_universal_code_length(n: int) -> float:
    """
    Universal code for positive integers using log-star.
    
    L(n) = log₂(c₀) + log₂(n) + log₂(log₂(n)) + ...
    
    Only include positive terms in the sum.
    """
    if n <= 0:
        return float('inf')
    
    c0 = 2.865064  # Normalization constant
    length = np.log2(c0)
    
    current = float(n)
    while current > 1:
        length += np.log2(current)
        current = np.log2(current)
        if current <= 0:
            break
    
    return max(length, 0)


def solution_model_description_length(degree: int, precision_bits: int = 32) -> float:
    """
    Model cost = degree encoding + coefficient encoding.
    """
    # Encode degree
    degree_cost = solution_universal_code_length(degree + 1)
    
    # Encode coefficients (degree+1 of them)
    num_coefficients = degree + 1
    coefficient_cost = num_coefficients * (precision_bits + 1)  # +1 for sign
    
    return degree_cost + coefficient_cost


def solution_data_description_length(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> float:
    """
    Data cost = encode residuals using Gaussian model.
    """
    n = len(x)
    
    # Fit polynomial
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    predictions = poly(x)
    residuals = y - predictions
    
    # Estimate sigma
    sigma = np.std(residuals)
    sigma = max(sigma, 1e-10)  # Avoid division by zero
    
    # Gaussian negative log-likelihood (in bits)
    nll = 0.5 * n * np.log2(2 * np.pi * sigma**2) + \
          np.sum(residuals**2) / (2 * sigma**2 * np.log(2))
    
    # Add cost to encode sigma
    sigma_cost = precision_bits + 10  # Simplified
    
    return nll + sigma_cost


def solution_two_part_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    precision_bits: int = 32
) -> Tuple[float, float, float]:
    """
    Complete two-part MDL: L(H) + L(D|H).
    """
    model_cost = solution_model_description_length(degree, precision_bits)
    data_cost = solution_data_description_length(x, y, degree, precision_bits)
    total = model_cost + data_cost
    
    return total, model_cost, data_cost


# =============================================================================
# EXERCISE 2 SOLUTIONS: Prequential MDL
# =============================================================================

def solution_predict_next_point(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_new: float,
    degree: int
) -> Tuple[float, float]:
    """
    Fit model on training data and predict the next point.
    """
    # Fit polynomial
    coeffs = np.polyfit(x_train, y_train, degree)
    poly = np.poly1d(coeffs)
    
    # Estimate sigma from training residuals
    train_pred = poly(x_train)
    residuals = y_train - train_pred
    sigma = np.std(residuals) if np.std(residuals) > 1e-10 else 1e-10
    
    # Predict new point
    prediction = poly(x_new)
    
    return prediction, sigma


def solution_code_length_for_point(
    y_actual: float,
    y_predicted: float,
    sigma: float
) -> float:
    """
    Gaussian code length for one observation.
    """
    sigma = max(sigma, 1e-10)
    error = y_actual - y_predicted
    
    code_length = 0.5 * np.log2(2 * np.pi * sigma**2) + \
                  (error**2) / (2 * sigma**2 * np.log(2))
    
    return code_length


def solution_prequential_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    start_index: int = None
) -> float:
    """
    Prequential MDL: predict each point, sum the surprises.
    """
    n = len(x)
    
    # Need at least degree+1 points to fit, plus 1 to predict
    if start_index is None:
        start_index = degree + 2
    
    if start_index >= n:
        return float('inf')
    
    total_code_length = 0.0
    
    for i in range(start_index, n):
        # Training data: everything before point i
        x_train = x[:i]
        y_train = y[:i]
        
        # Predict point i
        prediction, sigma = solution_predict_next_point(
            x_train, y_train, x[i], degree
        )
        
        # Code length for this prediction
        code_length = solution_code_length_for_point(y[i], prediction, sigma)
        total_code_length += code_length
    
    return total_code_length


def solution_prequential_model_selection(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Tuple[int, dict]:
    """
    Select best degree using prequential MDL.
    """
    scores = {}

    # [PATCH 2026-10-18] Code the same points under every model (common start index);
    # the per-degree default start (degree+2) made the totals incomparable.
    for degree in range(max_degree + 1):
        scores[degree] = solution_prequential_mdl(x, y, degree, start_index=max_degree + 2)
    
    best_degree = min(scores, key=scores.get)
    
    return best_degree, scores


# =============================================================================
# EXERCISE 3 SOLUTIONS: Model Selection
# =============================================================================

class SolutionMDLModelSelector:
    """Complete implementation of MDL model selector."""
    
    def __init__(self, max_degree: int = 10, precision_bits: int = 32):
        self.max_degree = max_degree
        self.precision_bits = precision_bits
        self.scores = {}
        self.model_costs = {}
        self.data_costs = {}
        self.best_degree = None
        self.best_coefficients = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SolutionMDLModelSelector':
        """Fit model selector to data."""
        for deg in range(self.max_degree + 1):
            total, model, data = solution_two_part_mdl(
                x, y, deg, self.precision_bits
            )
            self.scores[deg] = total
            self.model_costs[deg] = model
            self.data_costs[deg] = data
        
        self.best_degree = min(self.scores, key=self.scores.get)
        self.best_coefficients = np.polyfit(x, y, self.best_degree)
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions using selected model."""
        if self.best_coefficients is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.poly1d(self.best_coefficients)(x)
    
    def get_compression_ratio(self, x: np.ndarray, y: np.ndarray, precision: float = 0.01) -> float:
        """Fixed-length code / model code, both for y rounded to `precision` (tutorial Sec. 2.2.2).

        [PATCH 2026-10-18] was len(y) * 32 / MDL score, which compared float32 precision with
        the unit precision of a density code length.
        """
        n_grid = np.floor((np.max(y) - np.min(y)) / precision) + 1   # grid points over the observed range
        fixed_bits = len(y) * np.log2(n_grid)
        model_bits = self.scores[self.best_degree] - len(y) * np.log2(precision)
        return fixed_bits / model_bits if model_bits > 0 else 0
    
    def get_model_probability(self, degree: int) -> float:
        """Get approximate posterior probability."""
        min_score = min(self.scores.values())
        normalized = {k: -(v - min_score) for k, v in self.scores.items()}
        powers = {k: 2**v for k, v in normalized.items()}
        total = sum(powers.values())
        return powers[degree] / total if total > 0 else 0


def solution_compare_with_cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10,
    n_folds: int = 5
) -> Dict[str, int]:
    """Compare MDL with k-fold cross-validation."""
    # MDL selection
    selector = SolutionMDLModelSelector(max_degree=max_degree)
    selector.fit(x, y)
    mdl_best = selector.best_degree
    
    # K-fold CV
    n = len(x)
    fold_size = n // n_folds
    
    cv_errors = {}
    
    for deg in range(max_degree + 1):
        fold_errors = []
        
        for fold in range(n_folds):
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size
            
            val_idx = np.arange(val_start, val_end)
            train_idx = np.concatenate([np.arange(0, val_start), 
                                         np.arange(val_end, n)])
            
            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]
            
            # Fit and evaluate
            coeffs = np.polyfit(x_train, y_train, deg)
            predictions = np.poly1d(coeffs)(x_val)
            mse = np.mean((y_val - predictions)**2)
            fold_errors.append(mse)
        
        cv_errors[deg] = np.mean(fold_errors)
    
    cv_best = min(cv_errors, key=cv_errors.get)
    
    return {'mdl_best': mdl_best, 'cv_best': cv_best}


# =============================================================================
# EXERCISE 4 SOLUTIONS: NML Complexity
# =============================================================================
# [PATCH 2026-10-18] Rewritten with the redesigned exercise. The old solution estimated an
# infinite quantity (Gaussian NML with sigma fitted, Example 2.20 / Sec. 2.8) and dropped the
# Monte Carlo box volume, returning about -380 bits. Now: closed-form COMP of the restricted
# model (sigma known), a Monte Carlo check that keeps the volume term, and Rissanen's RNML for
# selection (sigma unknown). implementation.py Section 4 has the same formulas with more comments.

def solution_logsumexp(log_values: np.ndarray) -> float:
    """Numerically stable log-sum-exp; -inf entries (zero terms) are allowed."""
    log_values = np.asarray(log_values, dtype=float)
    max_val = np.max(log_values)
    if np.isneginf(max_val):
        return float('-inf')
    return float(max_val + np.log(np.sum(np.exp(log_values - max_val))))


def solution_gaussian_log_likelihood(
    y: np.ndarray,
    predictions: np.ndarray,
    sigma: float = None
) -> float:
    """Gaussian log-likelihood (in nats, not bits)."""
    residuals = y - predictions
    n = len(y)

    if sigma is None:
        sigma = np.sqrt(np.mean(residuals**2))
    sigma = max(sigma, 1e-10)

    return -0.5 * n * np.log(2 * np.pi * sigma**2) - \
           np.sum(residuals**2) / (2 * sigma**2)


def solution_nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    R: float,
    sigma: float = 1.0
) -> float:
    """COMP (bits) of {N(X beta, sigma^2 I): mean((X beta)^2) <= R}, sigma known (closed form)."""
    n, k = len(x), degree + 1
    s = np.sqrt(n * R) / sigma                       # allowed radius in noise units
    log_vk = 0.5 * k * np.log(np.pi) - lgamma(0.5 * k + 1)   # ln(unit-ball volume in R^k)
    # ln of the k + 1 terms of [V_k s^k + k V_k sum_j C(k-1,j) s^(k-1-j) m_j]; -inf = zero term
    log_terms = [log_vk + k * np.log(s) if s > 0 else -np.inf]
    for j in range(k):
        power = k - 1 - j
        if power > 0 and s == 0:
            log_terms.append(-np.inf)                # s^power = 0
            continue
        log_m_j = 0.5 * (j - 1) * np.log(2) + lgamma(0.5 * (j + 1))
        log_terms.append(np.log(k) + log_vk + np.log(comb(k - 1, j))
                         + (power * np.log(s) if power > 0 else 0.0) + log_m_j)
    ln_c = solution_logsumexp(np.array(log_terms)) - 0.5 * k * np.log(2 * np.pi)
    return max(ln_c, 0.0) / np.log(2)                # max() removes round-off at R = 0


def solution_estimate_nml_complexity_polynomial(
    x: np.ndarray,
    degree: int,
    R: float,
    sigma: float = 1.0,
    n_samples: int = 200_000,
    seed: int = 0
) -> Tuple[float, float]:
    """Monte Carlo (uniform box, WITH its volume (2L)^n) estimate of the same integral, in bits."""
    n = len(x)
    rng = np.random.default_rng(seed)
    r = np.sqrt(n * R)
    L = r + 8 * sigma
    V = np.vander(x / max(np.max(np.abs(x)), 1e-300), degree + 1, increasing=True)  # (n, k) design
    Q, _ = np.linalg.qr(V)                           # (n, k) orthonormal basis of the column space
    log_w = []
    for start in range(0, n_samples, 50_000):        # batches bound the memory use
        m = min(50_000, n_samples - start)
        Y = rng.uniform(-L, L, size=(m, n))          # (m, n) datasets drawn uniformly from the box
        Z = Y @ Q                                    # (m, k): least-squares fit coordinates
        rho = np.linalg.norm(Z, axis=1)              # (m,): length of each fit
        Z_fit = Z * np.minimum(1.0, r / np.maximum(rho, 1e-300))[:, None]  # restricted ML: shrink onto the ball
        resid2 = np.sum((Y - Z_fit @ Q.T) ** 2, axis=1)                    # (m,) squared residual norms
        log_g = -0.5 * n * np.log(2 * np.pi * sigma**2) - resid2 / (2 * sigma**2)
        log_w.append(log_g + n * np.log(2 * L))      # + log volume: weight = g / q = (2L)^n g
    log_w = np.concatenate(log_w)
    log_mean = solution_logsumexp(log_w) - np.log(len(log_w))
    w = np.exp(log_w - np.max(log_w))
    se_rel = np.std(w) / np.sqrt(len(w)) / np.mean(w)            # relative standard error of the mean
    return log_mean / np.log(2), se_rel / np.log(2)


def solution_compute_nml_mdl(
    x: np.ndarray,
    y: np.ndarray,
    degree: int
) -> Tuple[float, float, float]:
    """Rissanen's RNML code length (bits) for unknown sigma: (total, fit, complexity)."""
    n, k = len(x), degree + 1
    if k >= n:
        return float('inf'), float('inf'), float('inf')
    y_hat = np.polyval(np.polyfit(x, y, degree), x)
    sigma2_hat = np.mean((y - y_hat) ** 2)            # ML noise variance RSS / n
    R_hat = np.mean(y_hat ** 2)                       # mean squared fitted value
    total = ((n - k) / 2 * np.log(sigma2_hat) + k / 2 * np.log(R_hat)
             - lgamma(k / 2) - lgamma((n - k) / 2) + n / 2 * np.log(n * np.pi)) / np.log(2)
    fit = 0.5 * n * np.log2(2 * np.pi * np.e * sigma2_hat)
    return total, fit, total - fit


def solution_compare_nml_vs_two_part(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict:
    """Degree chosen by RNML and by the crude two-part code of Exercise 1."""
    nml_scores = {d: solution_compute_nml_mdl(x, y, d)[0] for d in range(max_degree + 1)}
    two_part_scores = {d: solution_two_part_mdl(x, y, d)[0] for d in range(max_degree + 1)}
    return {
        'nml_best': min(nml_scores, key=nml_scores.get),
        'two_part_best': min(two_part_scores, key=two_part_scores.get),
        'nml_scores': nml_scores,
        'two_part_scores': two_part_scores,
    }


def solution_visualize_complexity_vs_degree(
    x: np.ndarray,
    max_degree: int = 10,
    R_values: Tuple[float, ...] = (1.0, 100.0, 10000.0),
    sigma: float = 1.0
) -> Dict[float, list]:
    """Print COMP for each degree and bound R; returns {R: [COMP(degree) for degree]}."""
    table = {}
    print("   degree: " + "".join(f"{d:>7d}" for d in range(max_degree + 1)))
    for R in R_values:
        table[R] = [solution_nml_complexity_polynomial(x, d, R, sigma) for d in range(max_degree + 1)]
        print(f"   R={R:<7g}" + "".join(f"{c:7.2f}" for c in table[R]))
    return table


# =============================================================================
# EXERCISE 5 SOLUTIONS: MDL vs AIC vs BIC
# =============================================================================

def solution_compute_aic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Compute AIC score."""
    n = len(x)
    k = degree + 2  # Coefficients + variance
    
    coeffs = np.polyfit(x, y, degree)
    residuals = y - np.poly1d(coeffs)(x)
    sigma_ml = np.sqrt(np.sum(residuals**2) / n)
    sigma_ml = max(sigma_ml, 1e-10)
    
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
    
    return 2 * k - 2 * log_likelihood


def solution_compute_bic(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """Compute BIC score."""
    n = len(x)
    k = degree + 2
    
    coeffs = np.polyfit(x, y, degree)
    residuals = y - np.poly1d(coeffs)(x)
    sigma_ml = np.sqrt(np.sum(residuals**2) / n)
    sigma_ml = max(sigma_ml, 1e-10)
    
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma_ml**2) + 1)
    
    return k * np.log(n) - 2 * log_likelihood


def solution_select_with_all_criteria(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict[str, Dict]:
    """Run all three criteria."""
    results = {
        'mdl': {'scores': {}, 'best': None},
        'aic': {'scores': {}, 'best': None},
        'bic': {'scores': {}, 'best': None}
    }
    
    for deg in range(max_degree + 1):
        results['mdl']['scores'][deg] = solution_two_part_mdl(x, y, deg)[0]
        results['aic']['scores'][deg] = solution_compute_aic(x, y, deg)
        results['bic']['scores'][deg] = solution_compute_bic(x, y, deg)
    
    for method in ['mdl', 'aic', 'bic']:
        results[method]['best'] = min(
            results[method]['scores'],
            key=results[method]['scores'].get
        )
    
    return results


def solution_monte_carlo_comparison(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_std: float = 1.5,
    n_trials: int = 100
) -> Dict[str, float]:
    """Compare accuracy of all criteria."""
    x = np.linspace(0, 10, n_samples)
    
    correct = {'mdl': 0, 'aic': 0, 'bic': 0}
    
    for trial in range(n_trials):
        np.random.seed(trial)
        
        # Generate polynomial
        true_coeffs = np.random.randn(true_degree + 1) * 2
        y_true = np.poly1d(true_coeffs)(x)
        y = y_true + np.random.randn(n_samples) * noise_std
        
        # Run all criteria
        results = solution_select_with_all_criteria(x, y, max_degree=10)
        
        for method in ['mdl', 'aic', 'bic']:
            if results[method]['best'] == true_degree:
                correct[method] += 1
    
    return {
        'mdl_accuracy': correct['mdl'] / n_trials,
        'aic_accuracy': correct['aic'] / n_trials,
        'bic_accuracy': correct['bic'] / n_trials
    }


# [PATCH 2026-10-18] Reference solutions for the three exercise-5 analyses that had none.
def solution_analyze_sample_size_effect(
    true_degree: int = 2,
    sample_sizes: Tuple[int, ...] = (20, 50, 100, 200, 500),
    noise_std: float = 1.5,
    n_trials: int = 50
) -> Dict[str, List[float]]:
    """Accuracy of each criterion at each sample size (same seeds 0..n_trials-1 at every size)."""
    out = {'mdl': [], 'aic': [], 'bic': []}
    for n in sample_sizes:
        acc = solution_monte_carlo_comparison(true_degree, n, noise_std, n_trials)
        for method in out:
            out[method].append(acc[f'{method}_accuracy'])
    return out


def solution_analyze_noise_level_effect(
    true_degree: int = 2,
    n_samples: int = 50,
    noise_levels: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    n_trials: int = 50
) -> Dict[str, List[float]]:
    """Accuracy of each criterion at each noise level (same seeds at every level)."""
    out = {'mdl': [], 'aic': [], 'bic': []}
    for noise_std in noise_levels:
        acc = solution_monte_carlo_comparison(true_degree, n_samples, noise_std, n_trials)
        for method in out:
            out[method].append(acc[f'{method}_accuracy'])
    return out


def solution_when_do_they_disagree(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 10
) -> Dict:
    """Degree picked by each criterion and why they differ.

    All three trade the same fit term, -2 ln L at the ML fit, against a price per coefficient:
    AIC 2, BIC ln n, and the crude two-part code 33 bits = 33 * 2 ln 2 = 45.7 on the -2 ln L
    scale (plus a small log* term for the degree). Between two degrees lo < hi, a criterion
    prefers hi exactly when the drop in -2 ln L exceeds its price times (hi - lo).
    """
    n = len(x)
    results = solution_select_with_all_criteria(x, y, max_degree)
    picks = {m: results[m]['best'] for m in ('mdl', 'aic', 'bic')}
    lo, hi = min(picks.values()), max(picks.values())
    if lo == hi:
        return {'picks': picks, 'methods_agree': True,
                'explanation': f"all three pick degree {lo}"}
    # -2 ln L(d) = AIC(d) - 2k with k = d + 2 (coefficients + variance)
    minus_2ll = {d: results['aic']['scores'][d] - 2 * (d + 2) for d in (lo, hi)}
    drop, extra = minus_2ll[lo] - minus_2ll[hi], hi - lo
    explanation = (
        f"AIC picks {picks['aic']}, BIC {picks['bic']}, two-part MDL {picks['mdl']}. Degree {hi} lowers "
        f"-2 ln L by {drop:.1f} relative to degree {lo} with {extra} more coefficient(s). A criterion "
        f"prefers {hi} if that exceeds its price x {extra}: AIC 2, BIC ln n = {np.log(n):.1f}, "
        f"two-part MDL {33 * 2 * np.log(2):.1f} per coefficient.")
    return {'picks': picks, 'methods_agree': False, 'fit_gain': drop, 'explanation': explanation}


# =============================================================================
# MAIN: Verify solutions work
# =============================================================================

def verify_solutions():
    """Quick verification that solutions work."""
    print("Verifying Exercise Solutions")
    print("=" * 50)
    
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 3 + 2*x - 0.5*x**2 + np.random.randn(50) * 1.5
    
    print("\n[ok] Exercise 1: Two-Part MDL")
    total, model, data = solution_two_part_mdl(x, y, 2)
    print(f"  MDL(degree=2) = {total:.1f} bits")
    
    print("\n[ok] Exercise 2: Prequential MDL")
    preq = solution_prequential_mdl(x, y, 2)
    print(f"  Prequential(degree=2) = {preq:.1f} bits")
    
    print("\n[ok] Exercise 3: Model Selection")
    selector = SolutionMDLModelSelector(max_degree=8)
    selector.fit(x, y)
    print(f"  Selected degree: {selector.best_degree}")
    
    print("\n[ok] Exercise 4: NML Complexity")
    # [PATCH 2026-10-18] restricted COMP (sigma known), its Monte Carlo check, RNML selection
    complexity = solution_nml_complexity_polynomial(x, 2, R=100.0)
    print(f"  COMP(degree=2, R=100, sigma=1) = {complexity:.2f} bits")
    x_tiny = np.linspace(0, 10, 3)
    est, se = solution_estimate_nml_complexity_polynomial(x_tiny, 0, R=4.0)
    print(f"  tiny case n=3: Monte Carlo {est:.3f} +- {se:.3f} vs closed form "
          f"{solution_nml_complexity_polynomial(x_tiny, 0, R=4.0):.3f} bits")
    cmp = solution_compare_nml_vs_two_part(x, y, max_degree=8)
    print(f"  NML (RNML) selects degree {cmp['nml_best']}, two-part selects {cmp['two_part_best']}")
    
    print("\n[ok] Exercise 5: MDL vs AIC vs BIC")
    mc = solution_monte_carlo_comparison(n_trials=50)
    print(f"  MDL: {mc['mdl_accuracy']*100:.0f}%, "
          f"AIC: {mc['aic_accuracy']*100:.0f}%, "
          f"BIC: {mc['bic_accuracy']*100:.0f}%")
    
    print("\n" + "=" * 50)
    print("All solutions verified!")


if __name__ == "__main__":
    verify_solutions()
