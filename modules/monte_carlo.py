
import numpy as np
from math import log, sqrt
from scipy.stats import norm

def _z_ln_threshold(S0, B, r, sigma, T):
    if sigma <= 0 or T <= 0: return float("inf") if B>S0 else float("-inf")
    return (log(B/S0) - (r - 0.5*sigma**2)*T) / (sigma*sqrt(T))

def pop_bull_put_closed_form(S0, K_short, credit, r, sigma, T):
    B = max(K_short - credit, 1e-9); z = _z_ln_threshold(S0, B, r, sigma, T); return float(1.0 - norm.cdf(z))

def pop_bear_call_closed_form(S0, K_short, credit, r, sigma, T):
    B = max(K_short + credit, 1e-9); z = _z_ln_threshold(S0, B, r, sigma, T); return float(norm.cdf(z))

def pop_band_log_normal(S0, L, R, r, sigma, T):
    if L <= 0: L = 1e-9
    zL = _z_ln_threshold(S0, L, r, sigma, T); zR = _z_ln_threshold(S0, R, r, sigma, T); return float(max(0.0, min(1.0, norm.cdf(zR)-norm.cdf(zL))))

def expected_pnl_mc_vertical(S0, K_short, K_long, credit, is_put_credit, r, sigma, T, n_paths=50000, seed=42):
    rs = np.random.RandomState(seed); dt = T; mu = (r - 0.5*sigma**2)*dt
    ST = S0 * np.exp(mu + sigma*np.sqrt(dt)*rs.randn(n_paths)); width = abs(K_short-K_long)
    if is_put_credit:
        pnl = np.where(ST >= K_short, credit, np.where(ST <= K_long, credit - width, credit + (ST - K_short)))
    else:
        pnl = np.where(ST <= K_short, credit, np.where(ST >= K_long, credit - width, credit - (ST - K_short)))
    return float(pnl.mean()), float((pnl>=0).mean())
