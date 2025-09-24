
import os, sqlite3, pandas as pd
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "options_bot.db")
def _ensure_db():
    os.makedirs(DATA_DIR, exist_ok=True); return sqlite3.connect(DB_PATH)
def archive_scan_dataframe(df: pd.DataFrame, table_name: str="scans"):
    if df is None or df.empty: return
    conn = _ensure_db(); df2 = df.copy()
    df2.to_sql(table_name, conn, if_exists="append", index=False); conn.close()
    pq = os.path.join(DATA_DIR, f"{table_name}.parquet")
    try:
        if os.path.exists(pq):
            old = pd.read_parquet(pq); pd.concat([old, df2], ignore_index=True).to_parquet(pq, index=False)
        else:
            df2.to_parquet(pq, index=False)
    except Exception: pass
def archive_top20_csv(path: str, table_name: str="top20"):
    if os.path.exists(path): archive_scan_dataframe(pd.read_csv(path), table_name)
def archive_gappers_csv(path: str, table_name: str="gappers"):
    if os.path.exists(path): archive_scan_dataframe(pd.read_csv(path), table_name)
