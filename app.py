
import os, json, streamlit as st, pandas as pd, numpy as np, yfinance as yf

from modules.scanner import fetch_underlying_and_hv, fetch_option_expiries, fetch_option_chain, filter_liquidity, build_bull_put_spreads, build_bear_call_spreads, _atm_iv_from_chain, _days_to_expiry
from modules.monte_carlo import pop_bull_put_closed_form, pop_bear_call_closed_form, expected_pnl_mc_vertical
from modules.options_math import bs_delta, pop_single_leg
from modules.utils import payoff_bull_put_chart
from modules.discord import send_discord_webhook
from modules.storage import archive_scan_dataframe, archive_top20_csv, archive_gappers_csv

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
st.set_page_config(page_title="Ultimate Options Bot (Auto-Daily)", layout="wide")

def load_config():
    return json.load(open(CONFIG_PATH)) if os.path.exists(CONFIG_PATH) else {"watchlist":[],"scan":{}}
cfg = load_config()

tabs = st.tabs(["🔎 Auto-Scanner", "⏰ Premarket Gappers", "📊 Single Ticker", "⚙️ Settings", "🧾 Logs"])

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
    flow_min_chg = st.slider("Min price change % (today)", 0.0, 10.0, 2.0, 0.1)
    flow_min_vol_mult = st.slider("Min volume multiple vs 30D avg", 1.0, 10.0, 2.0, 0.1)
    flow_w = st.slider("Flow w", 0.0, 1.0, 0.10, 0.05)
    st.markdown("---")
    st.subheader("Discord Alerts")
    discord_enabled = st.toggle("Enable Discord alerts", value=False)
    discord_url = st.text_input("Discord Webhook URL", value="", type="default")

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
    def _flow_metrics(ticker: str):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="1d", interval="1m", prepost=True)
            if df.empty: return {"ticker":ticker,"premkt_pct":np.nan,"premkt_vol":0}
            if df.index.tz is not None: df = df.tz_convert("America/New_York")
            premkt = df.between_time("04:00","09:30")
            pm_vol = float(premkt["Volume"].sum()) if not premkt.empty else 0.0
            hist2 = t.history(period="2d", interval="1d")
            prev_close = float(hist2["Close"].iloc[-2]) if len(hist2)>=2 else np.nan
            pm_last = float(premkt["Close"].iloc[-1]) if not premkt.empty else np.nan
            chg = 100.0*(pm_last - prev_close)/prev_close if (not np.isnan(pm_last) and not np.isnan(prev_close) and prev_close>0) else np.nan
            return {"ticker":ticker,"premkt_pct":chg,"premkt_vol":pm_vol}
        except Exception:
            return {"ticker":ticker,"premkt_pct":np.nan,"premkt_vol":0}
    if st.button("Scan Gappers", key="scan_gappers", type="primary"):
        rows = [_flow_metrics(t) for t in tickers_g]
        dfg = pd.DataFrame(rows).dropna(subset=["premkt_pct"], how="all")
        if not dfg.empty:
            dfg = dfg.sort_values(["premkt_pct","premkt_vol"], ascending=[False,False]).reset_index(drop=True)
            st.dataframe(dfg.head(50), use_container_width=True)
            gcsv = os.path.join(DATA_DIR, "premarket_gappers.csv"); dfg.to_csv(gcsv, index=False)
            try: archive_gappers_csv(gcsv, "gappers")
            except Exception as e: st.caption(f"Gappers archive warning: {e}")
            st.caption("Saved premarket_gappers.csv and archived.")
        else:
            st.info("No premarket movers found.")

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
    use_calls = st.checkbox("Single-Leg Calls (debit)", value=True)
    use_puts = st.checkbox("Single-Leg Puts (debit)", value=True)
    send_discord_on_alert = st.checkbox("Send Discord alert for top candidate", value=False)

    def _flow_intraday(ticker: str):
        try:
            t = yf.Ticker(ticker)
            day = t.history(period="2d", interval="1d")
            if len(day) >= 2:
                prev_close = float(day["Close"].iloc[-2]); today_close = float(day["Close"].iloc[-1])
                chg = 100.0*(today_close - prev_close)/prev_close if prev_close>0 else 0.0
            else:
                chg = 0.0
            hist = t.history(period="2mo", interval="1d")
            vol30 = float(hist["Volume"].tail(30).mean()) if len(hist)>=30 else float(hist["Volume"].mean())
            today_vol = float(day["Volume"].iloc[-1]) if len(day) else 0.0
            mult = today_vol/vol30 if vol30>0 else 1.0
            return chg, mult
        except Exception:
            return 0.0, 1.0

    if st.button("Run Scan", type="primary"):
        results = []
        for tk in tickers:
            try:
                flow_chg, flow_mult = _flow_intraday(tk)
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
                            uvx = (row.get("volume",0) or 0)/max((row.get("openInterest",1) or 1),1)
                            flow_score = (min(abs(flow_chg)/10.0,1.0) + min(flow_mult/5.0,1.0))/2.0
                            score = w_pop*pop + w_roi*min(roi,1.0) + w_liq*liq + w_ive*0.5 + w_uvx*min(uvx/10.0,1.0) + flow_w*flow_score
                            results.append({"strategy":"bull_put","ticker":tk,"spot":round(spot,2),"expiry":exp,"dte":dte,"short":short_k,"long":long_k,"credit":round(credit,2),"pop":round(pop,4),"roi":round(roi,3),"liq":round(liq,3),"score":round(score,3),"flow_chg":round(flow_chg,2),"flow_mult":round(flow_mult,2)})

                    if use_bear_call and not calls_f.empty:
                        for short_k, long_k, credit in build_bear_call_spreads(calls_f, spot):
                            row = calls_f[calls_f["strike"]==short_k].head(1).iloc[0].to_dict()
                            liq = float(0.6*(1 - min(max(row.get("spread_pct",0.0),0.0),0.2)/0.2) + 0.4*min((row.get("openInterest",0) or 0)/1000.0,1.0))
                            pop = pop_bear_call_closed_form(spot, short_k, credit, rf, sigma, dte/252.0)
                            width = long_k - short_k; max_loss = max(width - credit, 0.01); roi = credit / max_loss
                            uvx = (row.get("volume",0) or 0)/max((row.get("openInterest",1) or 1),1)
                            flow_score = (min(abs(flow_chg)/10.0,1.0) + min(flow_mult/5.0,1.0))/2.0
                            score = w_pop*pop + w_roi*min(roi,1.0) + w_liq*liq + w_ive*0.5 + w_uvx*min(uvx/10.0,1.0) + flow_w*flow_score
                            results.append({"strategy":"bear_call","ticker":tk,"spot":round(spot,2),"expiry":exp,"dte":dte,"short":short_k,"long":long_k,"credit":round(credit,2),"pop":round(pop,4),"roi":round(roi,3),"liq":round(liq,3),"score":round(score,3),"flow_chg":round(flow_chg,2),"flow_mult":round(flow_mult,2)})

                    if (use_calls or use_puts) and (not calls_f.empty or not puts_f.empty) and (3 <= dte <= 10):
                        if use_calls and not calls_f.empty:
                            cf = calls_f.copy()
                            cf["delta_bs"] = cf.apply(lambda r: bs_delta(spot, float(r["strike"]), rf, float(r.get("impliedVolatility", sigma) or sigma), dte/252.0, "call"), axis=1)
                            pick = cf.iloc[(cf["delta_bs"] - 0.35).abs().argsort()].head(1)
                            if not pick.empty:
                                r = pick.iloc[0].to_dict()
                                premium = float(r.get("mid", (r.get("ask",0)+r.get("bid",0))/2.0) or 0.0)
                                pop = pop_single_leg(spot, float(r["strike"]), premium, rf, sigma, dte/252.0, "call")
                                liq = float(0.6*(1 - min(max(r.get("spread_pct",0.0),0.0),0.2)/0.2) + 0.4*min((r.get("openInterest",0) or 0)/1000.0,1.0))
                                uvx = (r.get("volume",0) or 0)/max((r.get("openInterest",1) or 1),1)
                                flow_score = (min(abs(flow_chg)/10.0,1.0) + min(flow_mult/5.0,1.0))/2.0
                                roi = premium/max(spot*0.01,1e-6)
                                results.append({"strategy":"long_call","ticker": tk, "expiry": exp, "dte": dte, "strike": float(r["strike"]), "debit": round(premium,2), "pop": round(pop,4), "roi": round(roi,3), "liq": round(liq,3), "score": round(w_pop*pop + w_roi*min(roi,1.0) + w_liq*liq + w_ive*0.5 + w_uvx*min(uvx/10.0,1.0) + flow_w*flow_score,3), "spot": round(spot,2), "flow_chg": round(flow_chg,2), "flow_mult": round(flow_mult,2)})
                        if use_puts and not puts_f.empty:
                            pf = puts_f.copy()
                            pf["delta_bs"] = pf.apply(lambda r: bs_delta(spot, float(r["strike"]), rf, float(r.get("impliedVolatility", sigma) or sigma), dte/252.0, "put"), axis=1)
                            pick = pf.iloc[(pf["delta_bs"] + 0.35).abs().argsort()].head(1)
                            if not pick.empty:
                                r = pick.iloc[0].to_dict()
                                premium = float(r.get("mid", (r.get("ask",0)+r.get("bid",0))/2.0) or 0.0)
                                pop = pop_single_leg(spot, float(r["strike"]), premium, rf, sigma, dte/252.0, "put")
                                liq = float(0.6*(1 - min(max(r.get("spread_pct",0.0),0.0),0.2)/0.2) + 0.4*min((r.get("openInterest",0) or 0)/1000.0,1.0))
                                uvx = (r.get("volume",0) or 0)/max((r.get("openInterest",1) or 1),1)
                                flow_score = (min(abs(flow_chg)/10.0,1.0) + min(flow_mult/5.0,1.0))/2.0
                                roi = premium/max(spot*0.01,1e-6)
                                results.append({"strategy":"long_put","ticker": tk, "expiry": exp, "dte": dte, "strike": float(r["strike"]), "debit": round(premium,2), "pop": round(pop,4), "roi": round(roi,3), "liq": round(liq,3), "score": round(w_pop*pop + w_roi*min(roi,1.0) + w_liq*liq + w_ive*0.5 + w_uvx*min(uvx/10.0,1.0) + flow_w*flow_score,3), "spot": round(spot,2), "flow_chg": round(flow_chg,2), "flow_mult": round(flow_mult,2)})

            except Exception as e:
                st.warning(f"{tk}: {e}")
        if results:
            df = pd.DataFrame(results)
            df["pop_pct"] = (df["pop"]*100.0).round(2)
            def exp_ret_pct(row):
                if row["strategy"] in ("bull_put","bear_call"):
                    w = max(abs(row["short"]-row["long"]), 0.01); return 100.0*(row["credit"]/w)
                if row["strategy"] in ("long_call","long_put"):
                    return 100.0 * (row.get("debit",0) / max(row.get("spot",1),1e-6))
                return 0.0
            df["exp_ret_pct"] = df.apply(exp_ret_pct, axis=1).round(2)
            if rank_mode=="POP %": df = df.sort_values(["pop_pct","score"], ascending=[False,False])
            elif rank_mode=="Expected Return %": df = df.sort_values(["exp_ret_pct","score"], ascending=[False,False])
            else: df = df.sort_values("score", ascending=False)
            df = df.reset_index(drop=True)
            st.dataframe(df.head(100), use_container_width=True)

            # Best Trade for Tomorrow
            st.subheader("Best Trade for Tomorrow")
            prefs = st.radio("Pick method", ["Composite Score","POP %"], horizontal=True)
            dfr = df.copy()
            if prefs=="POP %": dfr = dfr.sort_values(["pop_pct","score"], ascending=[False, False])
            else: dfr = dfr.sort_values("score", ascending=False)
            top = dfr.head(1).to_dict(orient="records")[0]
            st.write(f"**{top['ticker']}** — **{top['strategy']}** — Exp {top.get('expiry','')} — Strike(s) {top.get('strike', top.get('short',''))}{('/'+str(top.get('long',''))) if 'long' in top else ''}")
            if top["strategy"] in ("long_call","long_put"):
                side = "CALL" if top["strategy"]=="long_call" else "PUT"
                be = (top["strike"] + top["debit"]) if side=="CALL" else (top["strike"] - top["debit"])
                st.code(f"ENTRY (Limit Debit): ${top['debit']:.2f}\nTP: +30–50%\nSL: -25%\nBreakeven ≈ {be:.2f}", language="text")
            else:
                be = (top.get("short",0) - top.get("credit",0)) if top["strategy"]=="bull_put" else (top.get("short",0) + top.get("credit",0))
                st.code(f"ENTRY (Limit Credit): ${top.get('credit',0):.2f}\nTP: +50–60% credit\nSL: Breakeven {'<=' if top['strategy']=='bull_put' else '>='} {be:.2f}", language="text")

            # Archive full scan
            try: archive_scan_dataframe(df, "scans")
            except Exception as e: st.caption(f"Archive warning: {e}")

            # Save Top 20 CSV
            top20 = df.head(20).copy()
            csv_path = os.path.join(DATA_DIR, "top20.csv"); top20.to_csv(csv_path, index=False)
            try: archive_top20_csv(csv_path, "top20")
            except Exception as e: st.caption(f"Top20 archive warning: {e}")

            # Payoff chart (use spread chart placeholder for singles too)
            idx = st.number_input("Row index for payoff/chart", 0, min(9, len(df)-1), 0)
            sel = df.iloc[int(idx)]
            chart_path = os.path.join(DATA_DIR, "payoff.png")
            if sel["strategy"]=="bull_put":
                payoff_bull_put_chart(sel["spot"], sel["short"], sel["long"], sel["credit"], chart_path)
            elif sel["strategy"]=="bear_call":
                payoff_bull_put_chart(sel["spot"], sel["long"], sel["short"], sel["credit"], chart_path)
            else:
                payoff_bull_put_chart(sel["spot"], sel.get("strike", sel.get("short",0)), sel.get("strike", sel.get("long",0))*0.98, sel.get("debit", sel.get("credit",0)), chart_path)
            st.image(chart_path, caption=f"{sel['strategy']} payoff (schematic)")

            if send_discord_on_alert and discord_enabled and discord_url:
                try:
                    fields=[{"name":"POP %","value":str(sel['pop_pct']),"inline":True},
                            {"name":"ExpRet %","value":str(sel['exp_ret_pct']),"inline":True},
                            {"name":"Expiry","value":sel.get('expiry',''),"inline":True}]
                    if sel["strategy"] in ("bull_put","bear_call"):
                        strikes=f"short:{sel.get('short','')}, long:{sel.get('long','')}, credit:{sel.get('credit','')}"
                    else:
                        strikes=f"strike:{sel.get('strike','')}, debit:{sel.get('debit','')}"
                    fields.append({"name":"Contract","value":strikes,"inline":False})
                    embed=[{"title": f"{sel['ticker']} — {sel['strategy']}", "description": f"Score {sel['score']}", "fields": fields}]
                    files=[("file", ("payoff.png", open(chart_path,"rb"), "image/png"))]
                    csvf = os.path.join(DATA_DIR, "top20.csv")
                    if os.path.exists(csvf): files.append(("file", ("top20.csv", open(csvf,"rb"), "text/csv")))
                    send_discord_webhook(discord_url, content="Options Bot Alert", embeds=embed, files=files)
                    st.success("Sent Discord alert (with CSV).")
                except Exception as e:
                    st.warning(f"Discord failed: {e}")
        else:
            st.info("No candidates found. Try adjusting filters.")

with tabs[4]:
    st.subheader("Where to find your archives")
    st.markdown(f"- SQLite DB: `data/options_bot.db`  \n- Parquet: `data/scans.parquet`, `data/top20.parquet`, `data/gappers.parquet`  \n- CSVs: `data/top20.csv`, `data/premarket_gappers.csv`")
