
import os, numpy as np, matplotlib.pyplot as plt
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
def payoff_bull_put_chart(spot, short_strike, long_strike, credit, path):
    xs = np.linspace(max(0, long_strike*0.8), short_strike*1.2, 200)
    width = short_strike - long_strike
    pnl = np.where(xs >= short_strike, credit, np.where(xs <= long_strike, credit - width, credit - (short_strike - xs)))
    plt.figure(); plt.axhline(0); plt.plot(xs, pnl)
    plt.xlabel("Underlying Price at Expiration"); plt.ylabel("P&L per spread ($)")
    plt.title(f"Bull Put {long_strike}/{short_strike} credit {credit:.2f}")
    os.makedirs(DATA_DIR, exist_ok=True); plt.savefig(path, bbox_inches="tight"); plt.close()
