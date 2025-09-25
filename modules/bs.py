import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm

def _d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0: return 0.0
    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * sqrt(T)

def bs_price_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * exp(-r * T) - K * exp(-r * T), 0.0)
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(d1, sigma, T)
    return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)

def bs_price_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * exp(-r * T) - S * exp(-r * T), 0.0)
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(d1, sigma, T)
    return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def mc_prob_option_gain(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T_total: float,
    horizon_years: float,
    current_option_price: float,
    target_gain_frac: float,
    is_call: bool,
    n_paths: int = 2000,
    seed: int | None = 123
) -> float:
    """
    Probability that option price at horizon ≥ current_option_price * (1 + target_gain_frac).
    Uses GBM for underlying over horizon, then Black-Scholes with time-to-expiry reduced by horizon.
    """
    if S0 <= 0 or K <= 0 or current_option_price <= 0 or horizon_years <= 0:
        return 0.0
    T_rem = max(T_total - horizon_years, 1e-6)
    if sigma <= 0:
        # No volatility → deterministic path
        ST = S0 * exp((r - 0.5 * sigma * sigma) * horizon_years)
        price = bs_price_call(ST, K, r, sigma, T_rem) if is_call else bs_price_put(ST, K, r, sigma, T_rem)
        return 1.0 if price >= current_option_price * (1.0 + target_gain_frac) else 0.0
    rs = np.random.RandomState(seed)
    mu = (r - 0.5 * sigma * sigma) * horizon_years
    ST = S0 * np.exp(mu + sigma * np.sqrt(horizon_years) * rs.randn(n_paths))
    if is_call:
        prices = np.vectorize(bs_price_call)(ST, K, r, sigma, T_rem)
    else:
        prices = np.vectorize(bs_price_put)(ST, K, r, sigma, T_rem)
    thresh = current_option_price * (1.0 + target_gain_frac)
    return float(np.mean(prices >= thresh))


