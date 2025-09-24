
import os, numpy as np, pandas as pd, yfinance as yf

def _days_to_expiry(expiry_str: str, now_ts: pd.Timestamp | None = None) -> int:
    now_ts = now_ts or pd.Timestamp.utcnow().normalize()
    exp = pd.Timestamp(expiry_str); return max((exp - now_ts).days, 0)

def _hv_annualized(prices: pd.Series, window: int = 20) -> float:
    if len(prices) < window + 1: return np.nan
    rets = np.log(prices).diff().dropna(); hv = rets.rolling(window).std().iloc[-1]*np.sqrt(252)
    return float(hv) if pd.notna(hv) else np.nan

def _atm_iv_from_chain(chain_df: pd.DataFrame, spot: float) -> float:
    if chain_df.empty: return np.nan
    df = chain_df.copy(); df["strike_dist"] = (df["strike"] - spot).abs()
    near = df.sort_values("strike_dist").head(3); iv = near["impliedVolatility"].replace([np.inf,-np.inf], np.nan).dropna().mean()
    return float(iv) if pd.notna(iv) else np.nan

def fetch_underlying_and_hv(ticker: str, hv_window: int=20) -> tuple[float,float]:
    t = yf.Ticker(ticker); hist = t.history(period="6mo", interval="1d")["Close"]
    if hist.empty: raise RuntimeError(f"No price history for {ticker}")
    return float(hist.iloc[-1]), _hv_annualized(hist, hv_window)

def fetch_option_expiries(ticker: str) -> list[str]:
    t = yf.Ticker(ticker); return t.options or []

def fetch_option_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = yf.Ticker(ticker); opt = t.option_chain(expiry); calls, puts = opt.calls.copy(), opt.puts.copy()
    for df in (calls, puts):
        for col in ("impliedVolatility","bid","ask","lastPrice","volume","openInterest","strike","contractSymbol"):
            if col not in df.columns: df[col] = np.nan
        df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0))/2.0
        df["spread_pct"] = np.where(df["mid"]>0, (df["ask"]-df["bid"])/df["mid"], np.nan)
    return calls, puts

def filter_liquidity(df: pd.DataFrame, max_spread_pct: float, min_oi: int) -> pd.DataFrame:
    return df[(df["spread_pct"] <= max_spread_pct) & (df["openInterest"] >= min_oi)].copy()

def build_bull_put_spreads(puts: pd.DataFrame, spot: float) -> list[tuple[float,float,float]]:
    ps = puts.sort_values("strike"); out = []
    for _, s in ps.iterrows():
        if s["strike"] <= spot*0.98:
            lower = ps[ps["strike"] < s["strike"]].tail(1)
            if not lower.empty:
                credit = max(s["bid"],0) - max(float(lower["ask"].iloc[0]),0)
                if credit > 0: out.append((float(s["strike"]), float(lower["strike"].iloc[0]), float(credit)))
    return out[:20]

def build_bear_call_spreads(calls: pd.DataFrame, spot: float) -> list[tuple[float,float,float]]:
    cs = calls.sort_values("strike"); out = []
    for _, s in cs.iterrows():
        if s["strike"] >= spot*1.02:
            higher = cs[cs["strike"] > s["strike"]].head(1)
            if not higher.empty:
                credit = max(s["bid"],0) - max(float(higher["ask"].iloc[0]),0)
                if credit > 0: out.append((float(s["strike"]), float(higher["strike"].iloc[0]), float(credit)))
    return out[:20]
