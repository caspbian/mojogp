"""Statistical metrics for system benchmarks.

All metric functions accept numpy arrays and return float values.
These are the PRIMARY metrics for evaluating GP model quality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats

# Import properscoring for CRPS
try:
    import properscoring as ps

    HAS_PROPERSCORING = True
except ImportError:
    HAS_PROPERSCORING = False
    print(
        "Warning: properscoring not installed. CRPS will use fallback implementation."
    )


# =============================================================================
# Core Accuracy Metrics
# =============================================================================


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def max_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum Absolute Error."""
    return float(np.max(np.abs(y_true - y_pred)))


# =============================================================================
# Probabilistic Metrics (require mean + std)
# =============================================================================


def crps_gaussian(
    y_true: np.ndarray, pred_mean: np.ndarray, pred_std: np.ndarray
) -> float:
    """Continuous Ranked Probability Score for Gaussian predictive distribution.

    CRPS = E[|Y - X|] - 0.5 * E[|X - X'|] where X, X' ~ N(mu, sigma^2).
    Lower is better. Units match y.

    Uses properscoring library if available, otherwise falls back to analytical formula.
    """
    # Ensure positive std
    pred_std = np.maximum(pred_std, 1e-10)

    if HAS_PROPERSCORING:
        # properscoring expects (observations, mu, sig)
        scores = ps.crps_gaussian(y_true, mu=pred_mean, sig=pred_std)
        return float(np.mean(scores))
    else:
        # Analytical formula for Gaussian CRPS
        # CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
        # where z = (y - mu) / sigma, Phi = CDF, phi = PDF
        z = (y_true - pred_mean) / pred_std
        crps_values = pred_std * (
            z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
        )
        return float(np.mean(crps_values))


def mean_standardized_log_loss(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    y_train_mean: Optional[float] = None,
    y_train_std: Optional[float] = None,
) -> float:
    """Mean Standardized Log Loss (MSLL).

    MSLL = mean( 0.5*log(2*pi*var) + (y-mu)^2 / (2*var) ) - baseline
    Baseline is the trivial predictor N(mean(y_train), var(y_train)).
    Negative is better than trivial.
    """
    pred_std = np.maximum(pred_std, 1e-10)
    pred_var = pred_std**2

    # Log loss for predictions
    log_loss = 0.5 * np.log(2 * np.pi * pred_var) + (y_true - pred_mean) ** 2 / (
        2 * pred_var
    )
    mean_log_loss = np.mean(log_loss)

    # Baseline: trivial predictor using training data statistics
    if y_train_mean is None:
        y_train_mean = np.mean(y_true)
    if y_train_std is None:
        y_train_std = np.std(y_true)

    baseline_var = max(y_train_std**2, 1e-10)
    baseline_log_loss = 0.5 * np.log(2 * np.pi * baseline_var) + (
        y_true - y_train_mean
    ) ** 2 / (2 * baseline_var)
    mean_baseline = np.mean(baseline_log_loss)

    return float(mean_log_loss - mean_baseline)


def negative_log_predictive_density(
    y_true: np.ndarray, pred_mean: np.ndarray, pred_std: np.ndarray
) -> float:
    """Negative Log Predictive Density (NLPD).

    NLPD = -mean(log p(y | mu, sigma))
    Lower is better.
    """
    pred_std = np.maximum(pred_std, 1e-10)
    log_probs = stats.norm.logpdf(y_true, loc=pred_mean, scale=pred_std)
    return float(-np.mean(log_probs))


# =============================================================================
# Calibration Metrics
# =============================================================================


def calibration_coverage(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    levels: List[float] = [0.5, 0.9, 0.95, 0.99],
) -> Dict[float, float]:
    """Compute empirical coverage at specified confidence levels.

    For each confidence level, compute fraction of y_true within the interval.
    Returns {level: empirical_coverage}.
    Well-calibrated: empirical_coverage ~ level.
    """
    pred_std = np.maximum(pred_std, 1e-10)
    result = {}

    for level in levels:
        # Compute z-score for this confidence level
        z = stats.norm.ppf((1 + level) / 2)
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std

        # Count how many true values fall within the interval
        within = np.logical_and(y_true >= lower, y_true <= upper)
        result[level] = float(np.mean(within))

    return result


def calibration_error(
    y_true: np.ndarray, pred_mean: np.ndarray, pred_std: np.ndarray, num_bins: int = 20
) -> float:
    """Mean absolute calibration error across quantile bins.

    Lower is better. 0 = perfectly calibrated.
    """
    pred_std = np.maximum(pred_std, 1e-10)

    # Compute standardized residuals
    z = (y_true - pred_mean) / pred_std

    # Expected quantiles
    expected_quantiles = np.linspace(0, 1, num_bins + 1)[1:]

    # Observed quantiles (fraction of z <= z_expected)
    errors = []
    for q in expected_quantiles:
        z_threshold = stats.norm.ppf(q)
        observed = np.mean(z <= z_threshold)
        errors.append(abs(observed - q))

    return float(np.mean(errors))


def sharpness(pred_std: np.ndarray) -> float:
    """Mean predictive standard deviation.

    Lower is better (given good calibration).
    Measures how "confident" the predictions are.
    """
    return float(np.mean(pred_std))


def interval_width(
    pred_mean: np.ndarray, pred_std: np.ndarray, level: float = 0.95
) -> float:
    """Mean width of prediction intervals at given confidence level."""
    z = stats.norm.ppf((1 + level) / 2)
    widths = 2 * z * pred_std
    return float(np.mean(widths))


# =============================================================================
# Hyperparameter Recovery Metrics
# =============================================================================


def param_relative_error(learned: float, true: float) -> float:
    """Relative error: abs(learned - true) / abs(true)."""
    if abs(true) < 1e-10:
        return abs(learned)
    return abs(learned - true) / abs(true)


def param_log_ratio(learned: float, true: float) -> float:
    """Log ratio: log(learned / true). Symmetric in log space."""
    if true <= 0 or learned <= 0:
        return float("inf")
    return np.log(learned / true)


# =============================================================================
# Aggregate Metrics
# =============================================================================


def compute_all_accuracy_metrics(
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    y_train_mean: Optional[float] = None,
    y_train_std: Optional[float] = None,
) -> Dict[str, float]:
    """Compute all accuracy metrics in one call.

    Returns a dictionary with all metrics.
    """
    coverage = calibration_coverage(y_true, pred_mean, pred_std)

    return {
        "rmse": rmse(y_true, pred_mean),
        "mae": mae(y_true, pred_mean),
        "r_squared": r_squared(y_true, pred_mean),
        "max_error": max_absolute_error(y_true, pred_mean),
        "crps": crps_gaussian(y_true, pred_mean, pred_std),
        "msll": mean_standardized_log_loss(
            y_true, pred_mean, pred_std, y_train_mean, y_train_std
        ),
        "nlpd": negative_log_predictive_density(y_true, pred_mean, pred_std),
        "calibration_50": coverage[0.5],
        "calibration_90": coverage[0.9],
        "calibration_95": coverage[0.95],
        "calibration_99": coverage[0.99],
        "calibration_error": calibration_error(y_true, pred_mean, pred_std),
        "sharpness": sharpness(pred_std),
        "interval_width_95": interval_width(pred_mean, pred_std, 0.95),
    }
