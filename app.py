
import os, json, math
from datetime import datetime
import streamlit as st, pandas as pd, numpy as np, yfinance as yf

from modules.scanner import (
    fetch_underlying_and_hv, fetch_option_expiries, fetch_option_chain,
    filter_liquidity, build_bull_put_spreads, build_bear_call_spreads, _atm_iv_from_chain, _days_to_expiry
)
from modules.monte_carlo import pop_bull_put_closed_form, pop_bear_call_closed_form, pop_band_log_normal, expected_pnl_mc_vertical
from modules.utils import payoff_bull_put_chart
from modules.discord import send_discord_webhook
from modules.storage import archive_scan_dataframe, archive_top20_csv, archive_gappers_csv

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

st.set_page_config(page_title="Ultimate Options Bot (Offline)", layout="wide")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH,"r") as f: return json.load(f)
    return {"watchlist":[],"scan":{}}

cfg = load_config()

tabs = st.tabs(["🔎 Auto-Scanner", "⏰ Premarket Gappers", "📊 Single Ticker", "⚙️ Settings", "🧾 Logs"])

# ---- Settings ----
with tabs[3]:
    st.subheader("Scanning Filters")
    colA, colB, colC = st.columns(3)
    with colA:
        dte_min, dte_max = st.slider("DTE Range", 7, 180, (cfg["scan"]["dte_min"], cfg["scan"]["dte_max"]))
        rf = st.number_input("Risk-Free Rate", 0.0, 0.2, float(cfg["scan"]["risk_free_rate"]), 0.001)
    with colB:
        max_spread = st.slider("Max Spread %", 0.5, 20.0, float(cfg["scan"]["max_spread_pct"]*100), 0.5)/100.0
        min_oi = st.number_input("Min OI", 0, 5000, int(cfg["scan"]["min_oi"]), 50)
    with colC:
        rank_mode = st.selectbox("Rank by", ["Composite Score","POP %","Expected Return %"])
    st.markdown("---")
    st.subheader("Composite Score Weights")
    colw1, colw2, colw3, colw4, colw5 = st.columns(5)
    with colw1: w_pop = st.slider("POP w", 0.0, 1.0, 0.40, 0.05)
    with colw2: w_roi = st.slider("ROI w", 0.0, 1.0, 0.20, 0.05)
    with colw3: w_liq = st.slider("Liq w", 0.0, 1.0, 0.20, 0.05)
    with colw4: w_ive = st.slider("IV Edge w", 0.0, 1.0, 0.10, 0.05)
    with colw5: w_uvx = st.slider("Unusual Vol w", 0.0, 1.0, 0.10, 0.05)
    w_sum = w_pop + w_roi + w_liq + w_ive + w_uvx
    if w_sum == 0: w_pop, w_roi, w_liq, w_ive, w_uvx, w_sum = 0.4,0.2,0.2,0.1,0.1,1.0
    w_pop, w_roi, w_liq, w_ive, w_uvx = [w/w_sum for w in (w_pop, w_roi, w_liq, w_ive, w_uvx)]
    st.markdown("---")
    st.subheader("Discord Alerts")
    discord_enabled = st.toggle("Enable Discord alerts", value=False)
    discord_url = st.text_input("Discord Webhook URL", value="", type="default")

# ---- Single Ticker ----
with tabs[2]:
    st.subheader("Single Ticker")
    tk = st.text_input("Ticker", "AAPL").upper().strip()
    if st.button("Analyze", key="single"):
        try:
            spot, hv = fetch_underlying_and_hv(tk, 20)
            exps = fetch_option_expiries(tk)
            rows = []
            for exp in exps:
                dte = _days_to_expiry(exp)
                if dte < dte_min or dte > dte_max: continue
                calls, puts = fetch_option_chain(tk, exp)
                atm_iv = _atm_iv_from_chain(pd.concat([calls, puts], ignore_index=True), spot)
                puts_f = filter_liquidity(puts, max_spread, min_oi)
                calls_f = filter_liquidity(calls, max_spread, min_oi)
                for _, r in puts_f.iterrows():
                    rows.append({"type":"put","expiry":exp,"dte":dte,"strike":r["strike"],"mid":r["mid"],"oi":r["openInterest"],"vol":r["volume"],"iv":r["impliedVolatility"]})
                for _, r in calls_f.iterrows():
                    rows.append({"type":"call","expiry":exp,"dte":dte,"strike":r["strike"],"mid":r["mid"],"oi":r["openInterest"],"vol":r["volume"],"iv":r["impliedVolatility"]})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.error(str(e))

# ---- Premarket Gappers ----
with tabs[1]:
    st.subheader("Premarket Gappers")
    wl_text_g = st.text_input("Tickers (comma-separated) or CSV upload below (`ticker` column).", ",".join(cfg["watchlist"]), key="gap_wl")
    upg = st.file_uploader("Optional CSV upload", type=["csv"], key="gap_csv")
    tickers_g = [t.strip().upper() for t in wl_text_g.split(",") if t.strip()]
    if upg is not None:
        try:
            df_upg = pd.read_csv(upg)
            if "ticker" in df_upg.columns: tickers_g = [str(t).upper().strip() for t in df_upg["ticker"].dropna().tolist()]
        except Exception as e:
            st.warning(f"CSV read error: {e}")
    def premarket_change(tk: str) -> dict:
        try:
            t = yf.Ticker(tk)
            df = t.history(period="1d", interval="1m", prepost=True)
            if df.empty: return {"ticker":tk,"premkt_pct":np.nan,"premkt_vol":0}
            if df.index.tz is not None: df = df.tz_convert("America/New_York")
            premkt = df.between_time("04:00","09:30")
            pm_vol = float(premkt["Volume"].sum()) if not premkt.empty else 0.0
            hist2 = t.history(period="2d", interval="1d")
            prev_close = float(hist2["Close"].iloc[-2]) if len(hist2)>=2 else np.nan
            pm_last = float(premkt["Close"].iloc[-1]) if not premkt.empty else np.nan
            if not np.isnan(pm_last) and not np.isnan(prev_close) and prev_close>0:
                chg = 100.0*(pm_last - prev_close)/prev_close
            else:
                chg = np.nan
            return {"ticker":tk,"premkt_pct":chg,"premkt_vol":pm_vol}
        except Exception:
            return {"ticker":tk,"premkt_pct":np.nan,"premkt_vol":0}
    if st.button("Scan Gappers", key="scan_gappers", type="primary"):
        rows = [premarket_change(t) for t in tickers_g]
        dfg = pd.DataFrame(rows).dropna(subset=["premkt_pct"], how="all")
        if not dfg.empty:
            dfg = dfg.sort_values(["premkt_pct","premkt_vol"], ascending=[False,False]).reset_index(drop=True)
            st.dataframe(dfg.head(50), use_container_width=True)
            gcsv = os.path.join(DATA_DIR, "premarket_gappers.csv"); dfg.to_csv(gcsv, index=False)
            try: archive_gappers_csv(gcsv, "gappers")
            except Exception as e: st.caption(f"Gappers archive warning: {e}")
            st.caption("Saved premarket_gappers.csv and archived to SQLite/Parquet.")
        else:
            st.info("No premarket movers found.")

# ---- Auto-Scanner ----
with tabs[0]:
    st.subheader("Watchlist")
    wl_text = st.text_input("Tickers (comma-separated) or CSV upload below (`ticker` column).", ",".join(cfg["watchlist"]))
    up = st.file_uploader("Optional CSV upload", type=["csv"], key="scan_csv")
    tickers = [t.strip().upper() for t in wl_text.split(",") if t.strip()]
    if up is not None:
        try:
            df_up = pd.read_csv(up)
            if "ticker" in df_up.columns: tickers = [str(t).upper().strip() for t in df_up["ticker"].dropna().tolist()]
        except Exception as e:
            st.warning(f"CSV read error: {e}")
    use_bull_put = st.checkbox("Bull Put (credit)", value=True)
    use_bear_call = st.checkbox("Bear Call (credit)", value=True)
    send_discord_on_alert = st.checkbox("Send Discord alert for top candidate", value=False)

    if st.button("Run Scan", type="primary"):
        results = []
        for tk in tickers:
            try:
                spot, hv = fetch_underlying_and_hv(tk, 20)
                exps = fetch_option_expiries(tk)
                for exp in exps:
                    dte = _days_to_expiry(exp)
                    if dte < dte_min or dte > dte_max: continue
                    calls, puts = fetch_option_chain(tk, exp)
                    atm_iv = _atm_iv_from_chain(pd.concat([calls, puts], ignore_index=True), spot)
                    sigma = atm_iv if not np.isnan(atm_iv) else (hv if not np.isnan(hv) else 0.3)
                    puts_f = filter_liquidity(puts, max_spread, min_oi)
                    calls_f = filter_liquidity(calls, max_spread, min_oi)

                    if use_bull_put and not puts_f.empty:
                        for short_k, long_k, credit in build_bull_put_spreads(puts_f, spot):
                            row = puts_f[puts_f["strike"]==short_k].head(1).iloc[0].to_dict()
                            liq = float(0.6*(1 - min(max(row.get("spread_pct",0.0),0.0),0.2)/0.2) + 0.4*min((row.get("openInterest",0) or 0)/1000.0,1.0))
                            pop = pop_bull_put_closed_form(spot, short_k, credit, rf, sigma, dte/252.0)
                            width = short_k - long_k; max_loss = max(width - credit, 0.01); roi = credit / max_loss
                            iv_edge = (sigma - hv) if (not np.isnan(sigma) and not np.isnan(hv)) else 0.0
                            uvx = (row.get("volume",0) or 0)/max((row.get("openInterest",1) or 1),1)
                            score = w_pop*pop + w_roi*min(roi,1.0) + w_liq*liq + w_ive*min(max(iv_edge,0.0),1.0) + w_uvx*min(uvx/10.0,1.0)
                            results.append({"strategy":"bull_put","ticker":tk,"spot":round(spot,2),"expiry":exp,"dte":dte,"short":short_k,"long":long_k,"credit":round(credit,2),"pop":round(pop,4),"roi":round(roi,3),"liq":round(liq,3),"iv_edge":round(iv_edge,3),"uvx":round(uvx,3),"score":round(score,3)})

                    if use_bear_call and not calls_f.empty:
                        for short_k, long_k, credit in build_bear_call_spreads(calls_f, spot):
                            row = calls_f[calls_f["strike"]==short_k].head(1).iloc[0].to_dict()
                            liq = float(0.6*(1 - min(max(row.get("spread_pct",0.0),0.0),0.2)/0.2) + 0.4*min((row.get("openInterest",0) or 0)/1000.0,1.0))
                            pop = pop_bear_call_closed_form(spot, short_k, credit, rf, sigma, dte/252.0)
                            width = long_k - short_k; max_loss = max(width - credit, 0.01); roi = credit / max_loss
                            iv_edge = (sigma - hv) if (not np.isnan(sigma) and not np.isnan(hv)) else 0.0
                            uvx = (row.get("volume",0) or 0)/max((row.get("openInterest",1) or 1),1)
                            score = w_pop*pop + w_roi*min(roi,1.0) + w_liq*liq + w_ive*min(max(iv_edge,0.0),1.0) + w_uvx*min(uvx/10.0,1.0)
                            results.append({"strategy":"bear_call","ticker":tk,"spot":round(spot,2),"expiry":exp,"dte":dte,"short":short_k,"long":long_k,"credit":round(credit,2),"pop":round(pop,4),"roi":round(roi,3),"liq":round(liq,3),"iv_edge":round(iv_edge,3),"uvx":round(uvx,3),"score":round(score,3)})
            except Exception as e:
                st.warning(f"{tk}: {e}")
        if results:
            df = pd.DataFrame(results)
            df["pop_pct"] = (df["pop"]*100.0).round(2)
            def exp_ret_pct(row):
                w = max(abs(row["short"]-row["long"]), 0.01); return 100.0*(row["credit"]/w)
            df["exp_ret_pct"] = df.apply(exp_ret_pct, axis=1).round(2)
            if rank_mode=="POP %": df = df.sort_values(["pop_pct","score"], ascending=[False,False])
            elif rank_mode=="Expected Return %": df = df.sort_values(["exp_ret_pct","score"], ascending=[False,False])
            else: df = df.sort_values("score", ascending=False)
            df = df.reset_index(drop=True)
            st.dataframe(df.head(100), use_container_width=True)

            # Archive full scan
            try: archive_scan_dataframe(df, "scans")
            except Exception as e: st.caption(f"Archive warning: {e}")

            # Save Top 20 CSV
            top20 = df.head(20).copy()
            csv_path = os.path.join(DATA_DIR, "top20.csv"); top20.to_csv(csv_path, index=False)
            try: archive_top20_csv(csv_path, "top20")
            except Exception as e: st.caption(f"Top20 archive warning: {e}")

            # Payoff chart & details
            idx = st.number_input("Row index for payoff/chart", 0, min(9, len(df)-1), 0)
            sel = df.iloc[int(idx)]
            chart_path = os.path.join(DATA_DIR, "payoff.png")
            from modules.utils import payoff_bull_put_chart
            if sel["strategy"]=="bull_put":
                payoff_bull_put_chart(sel["spot"], sel["short"], sel["long"], sel["credit"], chart_path)
            else:
                payoff_bull_put_chart(sel["spot"], sel["long"], sel["short"], sel["credit"], chart_path)
            st.image(chart_path, caption=f"{sel['strategy']} payoff")

            # Expected value high-precision
            st.subheader("Expected Profit (Hi-Precision)")
            is_put = (sel["strategy"]=="bull_put")
            ev, pop_mc = expected_pnl_mc_vertical(sel["spot"], float(sel["short"]), float(sel["long"]), float(sel["credit"]), is_put, rf, max(sel.get("iv_edge",0)+0.2,0.15), sel["dte"]/252.0, 30000, 777)
            st.write(f"EV (per spread) ≈ ${ev:.2f} | POP_MC ≈ {pop_mc*100:.2f}% | POP_CF ≈ {sel['pop']*100:.2f}%")

            # Order Ticket Helper
            st.subheader("Order Ticket Helper")
            if sel["strategy"]=="bull_put":
                be = float(sel["short"]) - float(sel["credit"])
                txt = f"""LEG1: SELL TO OPEN {sel['ticker']} PUT {sel['expiry']} {sel['short']:.2f}
LEG2: BUY  TO OPEN {sel['ticker']} PUT {sel['expiry']} {sel['long']:.2f}
TYPE: LIMIT CREDIT (${sel['credit']:.2f}) | QTY: 1+ | TIF: DAY
Breakeven ≈ {be:.2f}"""
            else:
                be = float(sel["short"]) + float(sel["credit"])
                txt = f"""LEG1: SELL TO OPEN {sel['ticker']} CALL {sel['expiry']} {sel['short']:.2f}
LEG2: BUY  TO OPEN {sel['ticker']} CALL {sel['expiry']} {sel['long']:.2f}
TYPE: LIMIT CREDIT (${sel['credit']:.2f}) | QTY: 1+ | TIF: DAY
Breakeven ≈ {be:.2f}"""
            st.code(txt, language="text")

            # Discord alert (optional)
            if send_discord_on_alert and discord_enabled and discord_url:
                try:
                    fields=[{"name":"POP %","value":str(sel['pop_pct']),"inline":True},
                            {"name":"ExpRet %","value":str(sel['exp_ret_pct']),"inline":True},
                            {"name":"Credit","value":str(sel['credit']),"inline":True},
                            {"name":"Expiry","value":sel['expiry'],"inline":True}]
                    strikes=f"short:{sel.get('short','')}, long:{sel.get('long','')}"
                    fields.append({"name":"Strikes","value":strikes,"inline":False})
                    embed=[{"title": f"{sel['ticker']} — {sel['strategy']}", "description": f"Score {sel['score']}", "fields": fields}]
                    files=[("file", ("payoff.png", open(chart_path,"rb"), "image/png"))]
                    # attach top20.csv
                    csvf = os.path.join(DATA_DIR, "top20.csv")
                    if os.path.exists(csvf): files.append(("file", ("top20.csv", open(csvf,"rb"), "text/csv")))
                    send_discord_webhook(discord_url, content="Options Bot Alert", embeds=embed, files=files)
                    st.success("Sent Discord alert (with CSV).")
                except Exception as e:
                    st.warning(f"Discord failed: {e}")
        else:
            st.info("No candidates found. Try adjusting filters.")

# ---- Logs ----
with tabs[4]:
    st.subheader("Where to find your archives")
    st.markdown(f"- SQLite DB: `data/options_bot.db`  \n- Parquet snapshots: `data/scans.parquet`, `data/top20.parquet`, `data/gappers.parquet`  \n- CSVs: `data/top20.csv`, `data/premarket_gappers.csv`")
