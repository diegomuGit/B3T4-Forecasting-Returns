"""
frac_diff_utils.py — Fractional differentiation: weights, forward transform, and inversion.

Self-contained implementation of FFD (Fixed-width window Fractional Differentiation,
López de Prado 2018). Does NOT depend on dataset_utils so the backtest notebook can
run independently.

Key functions
-------------
ffd_weights(d, threshold)   → 1-D array of FFD weights ω_0 … ω_{L-1}
ffd_apply(log_prices, d, threshold)  → frac-diff series (same shape, first L-1 rows NaN)
ffd_invert_expected_return(
        pred_avg_fracdiff,   → (N_ASSETS,) model output
        log_prices_history,  → (T, N_ASSETS) known log-prices up to rebalancing date
        weights,             → FFD weights
        horizon              → V_OUT (90)
) → (N_ASSETS,) vector of expected simple returns over the horizon
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# 1. FFD weight computation
# ─────────────────────────────────────────────────────────────────────

def ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute the FFD weights ω_k for order *d*.

    The recursion is:
        ω_0 = 1
        ω_k = -ω_{k-1} · (d - k + 1) / k     for k ≥ 1

    Weights are retained while |ω_k| > threshold.

    Returns
    -------
    w : np.ndarray, shape (L,)
        The L retained weights, with w[0] = 1.0.
    """
    w = [1.0]
    k = 1
    while True:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < threshold:
            break
        w.append(w_next)
        k += 1
    return np.array(w, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# 2. Forward frac-diff (for verification / live inference)
# ─────────────────────────────────────────────────────────────────────

def ffd_apply(log_prices: np.ndarray, d: float,
              threshold: float = 1e-5) -> np.ndarray:
    """
    Apply FFD(d) to each column of *log_prices*.

    Parameters
    ----------
    log_prices : (T, N_ASSETS) or (T,) array of log(price).
    d          : fractional order.
    threshold  : FFD weight truncation.

    Returns
    -------
    result : same shape as input; first L-1 rows are NaN.
    """
    w = ffd_weights(d, threshold)
    L = len(w)

    if log_prices.ndim == 1:
        log_prices = log_prices[:, None]
        squeeze = True
    else:
        squeeze = False

    T, N = log_prices.shape
    out = np.full_like(log_prices, np.nan)

    for t in range(L - 1, T):
        # dot product of w with log_prices[t], log_prices[t-1], ..., log_prices[t-L+1]
        window = log_prices[t - L + 1: t + 1][::-1]  # shape (L, N), reversed
        out[t] = w @ window  # (L,) @ (L, N) → (N,)

    if squeeze:
        out = out.squeeze(-1)
    return out


# ─────────────────────────────────────────────────────────────────────
# 3. FFD inversion — predicted frac-diff → expected returns
# ─────────────────────────────────────────────────────────────────────

def ffd_invert_expected_return(
    pred_avg_fracdiff: np.ndarray,
    log_prices_history: np.ndarray,
    weights: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Invert the FFD operator to convert predicted average daily frac-diff
    into expected simple returns over *horizon* days.

    Methodology
    -----------
    The frac-diff at time t is:
        y_t = Σ_{k=0}^{L-1} w_k · x_{t-k}

    Since w_0 = 1, we can solve for the log-price:
        x_t = y_t - Σ_{k=1}^{L-1} w_k · x_{t-k}

    We assume each future day's frac-diff equals the model's predicted
    average (constant-daily assumption).  Starting from the known
    historical log-prices, we iterate forward *horizon* steps to
    reconstruct predicted log-prices, then compute:
        expected_return = exp(x_{t+horizon} - x_t) - 1

    Parameters
    ----------
    pred_avg_fracdiff : (N_ASSETS,)
        Model output: predicted average daily frac-diff over the horizon.
    log_prices_history : (T, N_ASSETS)
        Known historical log-prices ending at the rebalancing date (inclusive).
        Must contain at least L rows (L = len(weights)).
    weights : (L,)
        FFD weights from ffd_weights().
    horizon : int
        Number of days to project forward (= V_OUT).

    Returns
    -------
    expected_returns : (N_ASSETS,)
        Expected simple return for each asset over *horizon* trading days.
    """
    L = len(weights)
    T, N = log_prices_history.shape
    assert T >= L, (
        f"Need at least L={L} rows of history, got {T}."
    )
    assert pred_avg_fracdiff.shape == (N,), (
        f"pred_avg_fracdiff shape {pred_avg_fracdiff.shape} != ({N},)"
    )

    # Build an extended log-price array: [history | future forecast]
    # We only need the last L-1 rows of history + horizon new rows.
    buffer_len = L - 1 + horizon
    x = np.empty((buffer_len, N), dtype=np.float64)

    # Fill the known tail of history
    x[:L - 1] = log_prices_history[-(L - 1):]

    # The log-price at the rebalancing date (last known)
    x_rebal = log_prices_history[-1].copy()

    # Iterate forward: for each future day, reconstruct the log-price
    for step in range(horizon):
        idx = (L - 1) + step          # position in the buffer
        y_t = pred_avg_fracdiff        # constant-daily assumption

        # Σ_{k=1}^{L-1} w_k · x_{idx - k}
        correction = np.zeros(N, dtype=np.float64)
        for k in range(1, L):
            correction += weights[k] * x[idx - k]

        x[idx] = y_t - correction

    # Predicted log-price at the end of the horizon
    x_end = x[-1]

    # Expected simple return
    expected_log_return = x_end - x_rebal
    expected_return = np.expm1(expected_log_return)  # exp(r) - 1

    return expected_return
