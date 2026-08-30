"""
snapshot.py — Single self-contained NSE option-chain snapshot script.
=====================================================================

Triggered by the GitHub Actions "Run workflow" button (workflow_dispatch).
Everything happens inside THIS one file:

    1. Get the NIFTY50 spot price
    2. Get the list of expiries
    3. Fetch the full option chain from the NSE API (every available expiry)
    4. Pull IV from the API and compute Greeks (via py_vollib)
    5. Tag every row with the exact run timestamp (date + time, IST)
    6. Write a NEW parquet file for THIS run, named by its date + time:
           data/option_chain_YYYYMMDD_HHMM.parquet

Every trigger produces its own file, e.g.
    run on 31 Aug 2026 @ 10:15  →  data/option_chain_20260831_1015.parquet
    run on 31 Aug 2026 @ 11:30  →  data/option_chain_20260831_1130.parquet

Run locally:
    python snapshot.py

Inspect files later:
    python snapshot.py --list                 # list all snapshot timestamps
    python snapshot.py --at 2026-08-31_10:15   # print that snapshot's data
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd
import pytz
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Config — constants that control what we fetch and where we write it
# ──────────────────────────────────────────────────────────────────────────────

IST          = pytz.timezone("Asia/Kolkata")   # all timestamps are in IST
SYMBOL       = "NIFTY50"                        # our internal symbol label
NSE_SYMBOL   = "NIFTY"                          # symbol as the NSE API expects it
NSE_OC_TYPE  = "Indices"                        # NIFTY is an index option chain
RISK_FREE    = 0.065                            # ~6.5% risk-free rate used for Greeks/IV
DATA_DIR     = "data"                           # each run writes its own dated file here

# NSE option-chain-v3 endpoint — one request per expiry.
# {typ}=Indices, {sym}=NIFTY, {expiry}=DD-Mon-YYYY
_NSE_OC_V3_URL = (
    "https://www.nseindia.com/api/option-chain-v3"
    "?type={typ}&symbol={sym}&expiry={expiry}"
)
# Referer/origin page NSE requires so the API accepts the request
_NSE_OC_ORIGIN = "https://www.nseindia.com/option-chain"

# Canonical column order for the parquet file. Every saved file has exactly
# these columns in this order, so all snapshots stay schema-compatible.
_OC_COLS = [
    "timestamp", "datetime_ist", "symbol", "expiry", "strike", "option_type",
    "spot", "ltp", "volume", "oi", "oi_chg", "iv",
    "delta", "gamma", "theta", "vega", "rho",
]

# Log to stdout so the GitHub Actions run log shows progress live.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("snapshot")


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_float(val) -> Optional[float]:
    """Safely parse any value to float. Returns None for missing/invalid/NaN."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    try:
        f = float(val)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def _to_float_nonneg(val) -> Optional[float]:
    """Like _to_float but rejects negatives (used for OI, volume, LTP)."""
    f = _to_float(val)
    return f if (f is not None and f >= 0) else None


def _next_thursday(ref: date) -> date:
    """
    Return the next Thursday on/after `ref` (NIFTY weekly expiry day).
    Used only by the fallback expiry generator when the NSE API is unreachable.
    """
    # weekday(): Mon=0 ... Thu=3. Distance to the coming Thursday.
    days = (3 - ref.weekday()) % 7
    return ref + timedelta(days=max(days, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Greeks + IV  (py_vollib — no custom formulas)
# ──────────────────────────────────────────────────────────────────────────────

def compute_greeks(spot, strike, tte, r, iv, option_type) -> dict:
    """
    Compute the five option Greeks (delta, gamma, theta, vega, rho) using the
    py_vollib Black-Scholes library — no hand-rolled formulas.

    Params:
        spot        : underlying price (S)
        strike      : option strike (K)
        tte         : time to expiry in YEARS (e.g. 7/365)
        r           : risk-free rate as a decimal (e.g. 0.065)
        iv          : implied volatility as a decimal (e.g. 0.15)
        option_type : "CE" (call) or "PE" (put)

    Returns a dict of the 5 Greeks, or all-None if inputs are invalid/fail.
    """
    null = {k: None for k in ("delta", "gamma", "theta", "vega", "rho")}
    # Guard: every input must be present and positive, else Black-Scholes is undefined.
    if not (spot and strike and tte and iv and spot > 0 and strike > 0 and tte > 0 and iv > 0):
        return null
    # py_vollib uses 'c'/'p' flags rather than CE/PE.
    flag = "c" if option_type == "CE" else "p"
    try:
        # Imported lazily so the module loads even if py_vollib isn't installed.
        from py_vollib.black_scholes.greeks.analytical import (
            delta as _d, gamma as _g, theta as _t, vega as _v, rho as _r,
        )
        # py_vollib already scales theta per-day and vega/rho per 1% move.
        return {
            "delta": round(float(_d(flag, spot, strike, tte, r, iv)), 4),
            "gamma": round(float(_g(flag, spot, strike, tte, r, iv)), 6),
            "theta": round(float(_t(flag, spot, strike, tte, r, iv)), 4),
            "vega":  round(float(_v(flag, spot, strike, tte, r, iv)), 4),
            "rho":   round(float(_r(flag, spot, strike, tte, r, iv)), 4),
        }
    except Exception:
        # Never let a single bad strike crash the whole run — return None Greeks.
        logger.debug("greeks failed strike=%s type=%s", strike, option_type, exc_info=True)
        return null


def iv_from_price(flag, S, K, t, price) -> Optional[float]:
    """
    Back out implied volatility from an option's market price via py_vollib.

    This is a FALLBACK only — used when the NSE API does not return IV for a
    contract. flag is 'c'/'p'. Returns IV as a decimal (e.g. 0.18) or None.
    """
    # All inputs must be positive for the solver to converge.
    if not (S > 0 and K > 0 and t > 0 and price > 0):
        return None
    try:
        from py_vollib.black_scholes.implied_volatility import implied_volatility
        iv = implied_volatility(price, S, K, t, RISK_FREE, flag)
        # Sanity-bound the result: reject absurd IVs (<0.1% or >1000%).
        return round(float(iv), 4) if (iv and 0.001 < iv < 10) else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# NSE data fetch
# ──────────────────────────────────────────────────────────────────────────────

def get_spot() -> Optional[float]:
    """
    Get the current NIFTY50 spot price.

    Tries two sources in order (whichever answers first wins):
      1. yfinance (^NSEI) — reliable from GitHub Actions runners
      2. nselib capital_market index data — fallback
    Returns the price as a float, or None if both sources fail.
    """
    # ── Source 1: yfinance ────────────────────────────────────────────────────
    try:
        import yfinance as yf
        ticker = yf.Ticker("^NSEI")
        price = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("regularMarketPrice")
        if price and float(price) > 0:
            logger.info("spot (yfinance): %.2f", float(price))
            return float(price)
    except Exception as e:
        logger.warning("yfinance spot failed: %s", e)

    # ── Source 2: nselib ──────────────────────────────────────────────────────
    try:
        from nselib import capital_market
        data = capital_market.index_data()
        if data is not None and not data.empty and "indexSymbol" in data.columns:
            row = data[data["indexSymbol"] == "NIFTY 50"]
            if not row.empty:
                return float(row.iloc[0]["last"])
    except Exception as e:
        logger.warning("nselib spot failed: %s", e)

    return None


def get_expiry_dates() -> List[str]:
    """
    Get every available NIFTY option expiry (weekly + monthly) as
    'DD-Mon-YYYY' strings.

    Primary source is nselib. If that fails, we fall back to generating the
    next ~3 months of weekly Thursdays so the run can still proceed.
    """
    # ── Primary: real expiry list from NSE via nselib ─────────────────────────
    try:
        from nselib import derivatives
        data = derivatives.expiry_dates_option_index()
        expiries = data.get(NSE_SYMBOL, [])
        if expiries:
            logger.info("expiries (%d): %s", len(expiries), expiries)
            return expiries
    except Exception:
        logger.warning("nselib expiry fetch failed", exc_info=True)

    # ── Fallback (API failed): synthesise ~3 months of weekly Thursdays ───────
    result, cursor = [], _next_thursday(date.today())
    for _ in range(14):
        result.append(cursor.strftime("%d-%b-%Y"))
        cursor += timedelta(weeks=1)
    return result


def fetch_option_chain(spot: float) -> pd.DataFrame:
    """
    Fetch the FULL option chain for EVERY available expiry from the NSE v3 API
    and return it as a flat DataFrame (one row per strike per CE/PE).

    For each contract it collects OI, volume, LTP, IV, and the 5 Greeks.
    `spot` is the fallback underlying price if the API doesn't report one.
    """
    from nselib.libutil import nse_urlfetch   # nselib handles NSE cookies/headers

    expiries = get_expiry_dates()   # all expiries available on NSE
    if not expiries:
        logger.error("no expiries available")
        return pd.DataFrame()
    logger.info("fetching %d expiries", len(expiries))

    rows: list = []
    # One API call per expiry — NSE's v3 endpoint returns a single expiry each.
    for expiry in expiries:
        url = _NSE_OC_V3_URL.format(typ=NSE_OC_TYPE, sym=NSE_SYMBOL, expiry=expiry)
        try:
            resp     = nse_urlfetch(url, origin_url=_NSE_OC_ORIGIN)
            records  = resp.json().get("records", {})
            # Prefer the API's own underlying value; fall back to our spot arg.
            api_spot = _to_float(records.get("underlyingValue"))
            use_spot = api_spot if (api_spot and api_spot > 0) else spot
            raw      = records.get("data", [])   # list of strike rows
            logger.info("expiry=%s rows=%d spot=%.2f", expiry, len(raw), use_spot)
        except Exception as e:
            # A single failed expiry is skipped, not fatal to the whole run.
            logger.warning("API failed expiry=%s: %s", expiry, e)
            continue

        # Time-to-expiry in YEARS (floor at half a day so 0-DTE stays positive).
        try:
            tte = max((datetime.strptime(expiry, "%d-%b-%Y").date() - date.today()).days, 0.5) / 365.0
        except Exception:
            tte = None

        # Each `item` holds both the CE and PE leg for one strike.
        for item in raw:
            strike = _to_float(item.get("strikePrice"))
            if strike is None:
                continue
            # Process the call (CE) and put (PE) legs separately.
            for otype in ("CE", "PE"):
                d = item.get(otype, {})
                if not d:
                    continue   # this strike has no data for this side

                # ── Market fields straight from the API ───────────────────────
                ltp    = _to_float_nonneg(d.get("lastPrice"))
                oi     = _to_float_nonneg(d.get("openInterest"))
                chg_oi = _to_float(d.get("changeinOpenInterest"))
                vol    = _to_float_nonneg(d.get("totalTradedVolume"))

                # ── IV: use the API's value (already a %); else solve from LTP ──
                iv_api = _to_float(d.get("impliedVolatility"))
                iv_pct = iv_api if (iv_api and iv_api > 0) else None
                if iv_pct is None and ltp and ltp > 0 and tte:
                    flag = "c" if otype == "CE" else "p"
                    iv_dec_fb = iv_from_price(flag, use_spot, strike, tte, ltp)
                    iv_pct = round(iv_dec_fb * 100, 2) if iv_dec_fb else None
                iv_dec = iv_pct / 100 if iv_pct else None   # decimal form for Greeks

                # ── Greeks computed from that IV (py_vollib) ──────────────────
                greeks = compute_greeks(use_spot, strike, tte, RISK_FREE, iv_dec, otype)

                # ── Drop rows with no real price/spot (illiquid/blank strikes) ─
                if strike <= 0 or use_spot <= 0 or not ltp or ltp <= 0:
                    continue

                # One flat record; **greeks unpacks delta/gamma/theta/vega/rho.
                rows.append({
                    "symbol":      SYMBOL,
                    "expiry":      expiry,
                    "strike":      strike,
                    "option_type": otype,
                    "spot":        use_spot,
                    "ltp":         ltp,
                    "volume":      vol,
                    "oi":          oi,
                    "oi_chg":      chg_oi,
                    "iv":          iv_pct,
                    **greeks,
                })

    logger.info("total rows fetched: %d", len(rows))
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Parquet writer — one timestamped file per run
# ──────────────────────────────────────────────────────────────────────────────

def save_snapshot(df: pd.DataFrame, dt: datetime) -> str:
    """
    Write this run's option chain into its OWN new parquet file, named by the
    exact date + time of the trigger:

        data/option_chain_YYYYMMDD_HHMM.parquet

    e.g. a run on 31 Aug 2026 at 10:15 IST →
        data/option_chain_20260831_1015.parquet

    Returns the path of the file written (or "" if there was nothing to save).
    """
    if df.empty:
        logger.warning("nothing to save — empty dataframe")
        return ""

    # Two representations of the same instant:
    #   ts_key  → compact, sortable, used in the filename and `timestamp` column
    #   ts_read → human-readable, stored in the `datetime_ist` column
    ts_key  = dt.strftime("%Y%m%d_%H%M")
    ts_read = dt.strftime("%Y-%m-%d %H:%M:%S IST")

    df = df.copy()
    df["timestamp"]    = ts_key
    df["datetime_ist"] = ts_read

    # Enforce the canonical schema: add any missing columns as None, then
    # reorder to _OC_COLS so every saved file has identical structure.
    for col in _OC_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[_OC_COLS]

    # Guard against duplicate strikes and give the file a stable sort order.
    df = (
        df.drop_duplicates(subset=["expiry", "strike", "option_type"])
          .sort_values(["expiry", "strike", "option_type"])
          .reset_index(drop=True)
    )

    # Write the file (creating data/ if needed).
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"option_chain_{ts_key}.parquet")
    df.to_parquet(path, index=False)
    logger.info("saved %d rows → %s", len(df), path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# README — show the latest snapshot in the repo front page
# ──────────────────────────────────────────────────────────────────────────────

README_FILE = "README.md"


def _fmt(v) -> str:
    """Format a number for a markdown table cell. None/NaN → '-'.
    Large values get thousands separators; small ones get 4 decimals."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    try:
        f = float(v)
        return f"{f:,.2f}" if abs(f) >= 1 else f"{f:.4f}"
    except Exception:
        return str(v)


def write_readme(df: pd.DataFrame, dt: datetime, path: str) -> None:
    """
    Rewrite README.md so the repo front page shows THIS run's snapshot:
    when it was captured, which file holds it, and the full option chain
    rendered as a table per expiry (CE fields | Strike | PE fields).
    """
    # Header/meta values for the "Latest Snapshot" block.
    ts_key  = dt.strftime("%Y%m%d_%H%M")
    ts_read = dt.strftime("%d %b %Y, %H:%M IST")
    fname   = os.path.basename(path) if path else f"option_chain_{ts_key}.parquet"

    # Pull one spot value to display (all rows share the same spot).
    spot = None
    if "spot" in df.columns and not df.empty:
        try:
            spot = float(df["spot"].dropna().iloc[0])
        except Exception:
            spot = None

    # We build the markdown line-by-line, then join and write at the end.
    lines = []
    lines.append("# NIFTY50 Option Chain Snapshots\n")
    lines.append(
        "Each GitHub Actions run fetches the NIFTY50 option chain at that moment "
        "and saves it as a timestamped parquet file in "
        "[`data/`](data). The latest snapshot is shown below.\n"
    )

    # ── Latest snapshot header ────────────────────────────────────────────────
    lines.append("## 📸 Latest Snapshot\n")
    lines.append(f"- **Captured:** {ts_read}")
    lines.append(f"- **File:** `data/{fname}`")
    if spot is not None:
        lines.append(f"- **Spot:** {spot:,.2f}")
    lines.append(f"- **Rows:** {len(df)}\n")

    # ── One option-chain table per expiry, in chronological order ─────────────
    for expiry in sorted(df["expiry"].dropna().unique(),
                         key=lambda e: datetime.strptime(e, "%d-%b-%Y")):
        exp_df = df[df["expiry"] == expiry]

        # Regroup the flat rows into  strike → {"CE": row, "PE": row}
        # so we can print the call and put legs side by side.
        chain: dict = {}
        for _, r in exp_df.iterrows():
            k = float(r["strike"])
            chain.setdefault(k, {})[str(r["option_type"]).upper()] = r

        # Table header: CE fields on the left, Strike in the middle, PE on the right.
        lines.append(f"### Expiry: {expiry}\n")
        header = (
            "| CE OI | CE ChgOI | CE Vol | CE IV | CE LTP | "
            "**Strike** | "
            "PE LTP | PE IV | PE Vol | PE ChgOI | PE OI |"
        )
        sep = "| " + " | ".join(["---"] * 11) + " |"
        lines.append(header)
        lines.append(sep)

        # One row per strike (ascending).
        for k in sorted(chain):
            ce = chain[k].get("CE")   # may be None if that leg is missing
            pe = chain[k].get("PE")

            # Small helper: format one field of one leg, '-' if the leg is absent.
            def g(side, col):
                return _fmt(side.get(col)) if side is not None else "-"

            lines.append(
                f"| {g(ce,'oi')} | {g(ce,'oi_chg')} | {g(ce,'volume')} | "
                f"{g(ce,'iv')} | {g(ce,'ltp')} | "
                f"**{int(k)}** | "
                f"{g(pe,'ltp')} | {g(pe,'iv')} | {g(pe,'volume')} | "
                f"{g(pe,'oi_chg')} | {g(pe,'oi')} |"
            )
        lines.append("")   # blank line after each table

    # ── Static footer docs ────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## How it works\n")
    lines.append(
        "Everything lives in one self-contained script: [`snapshot.py`](snapshot.py). "
        "Each run fetches the spot price, all available expiries, the full option "
        "chain (IV from the NSE API, Greeks via `py_vollib`), then writes "
        "`data/option_chain_YYYYMMDD_HHMM.parquet` and updates this README.\n"
    )
    lines.append("### Run it\n")
    lines.append(
        "- **GitHub:** Actions tab → *Fetch Option Chain Snapshot* → **Run workflow**\n"
        "- **Locally:** `pip install -r requirements.txt` then `python snapshot.py`\n"
    )
    lines.append("### Read a snapshot back\n")
    lines.append("```bash")
    lines.append("python snapshot.py --list                 # list all timestamps")
    lines.append("python snapshot.py --at 2026-08-31_10:15    # print that snapshot")
    lines.append("```\n")

    # Overwrite README.md with the freshly built content.
    with open(README_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("README updated with snapshot %s", ts_key)


# ──────────────────────────────────────────────────────────────────────────────
# Read helpers — list the dated files / read one back
# ──────────────────────────────────────────────────────────────────────────────

def list_timestamps() -> List[str]:
    """Return the 'YYYYMMDD_HHMM' timestamp of every saved snapshot, sorted."""
    if not os.path.exists(DATA_DIR):
        return []
    stamps = []
    for f in os.listdir(DATA_DIR):
        if f.startswith("option_chain_") and f.endswith(".parquet"):
            # Strip the "option_chain_" prefix and ".parquet" suffix.
            stem = f[len("option_chain_"):-len(".parquet")]   # "YYYYMMDD_HHMM"
            stamps.append(stem)
    return sorted(stamps)


def read_snapshot(ts: str) -> pd.DataFrame:
    """
    Load one saved snapshot back into a DataFrame.

    `ts` is flexible — it accepts either the stored key "YYYYMMDD_HHMM" or a
    friendlier "YYYY-MM-DD_HH:MM"; we normalise it to the filename key.
    Returns an empty DataFrame if no matching file exists.
    """
    # Keep only the digits, then rebuild the canonical "YYYYMMDD_HHMM" key.
    digits = "".join(ch for ch in ts if ch.isdigit())   # e.g. "202608311015"
    key = ts
    if len(digits) >= 12:
        key = f"{digits[:8]}_{digits[8:12]}"

    path = os.path.join(DATA_DIR, f"option_chain_{key}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_snapshot() -> None:
    """
    The main end-to-end flow for one capture (what a workflow run executes):

        spot → expiries → option chain + Greeks → save parquet → update README

    Exits non-zero if spot or the option chain can't be fetched, so a failed
    GitHub Actions run is clearly reported.
    """
    # Stamp the whole run with a single "now" so the filename, the parquet
    # `timestamp` column, and the README all refer to the exact same instant.
    now = datetime.now(IST)
    logger.info("=== snapshot run @ %s ===", now.strftime("%Y-%m-%d %H:%M:%S IST"))

    # 1. Underlying spot price (required for Greeks / ATM context).
    spot = get_spot() or 0.0
    if spot <= 0:
        logger.error("could not get spot — aborting")
        sys.exit(1)

    # 2. Full option chain across all expiries, with IV + Greeks attached.
    df = fetch_option_chain(spot)
    if df.empty:
        logger.error("empty option chain — aborting")
        sys.exit(1)

    # 3. Persist this snapshot to its own timestamped parquet file, and
    # 4. refresh README.md so the repo front page shows the latest data.
    path = save_snapshot(df, now)
    write_readme(df, now, path)
    logger.info("=== done → %s ===", path)


def main() -> None:
    """
    CLI entry point. Default action captures a new snapshot; flags let you
    inspect existing ones without hitting the network:

        python snapshot.py                     # capture a new snapshot
        python snapshot.py --list              # list saved timestamps
        python snapshot.py --at 2026-08-31_10:15  # print one snapshot
    """
    parser = argparse.ArgumentParser(description="NSE option chain single-file snapshot")
    parser.add_argument("--list", action="store_true", help="list all captured timestamps")
    parser.add_argument("--at", metavar="TS", help="print the snapshot at a timestamp")
    args = parser.parse_args()

    # --list: just print every saved snapshot's timestamp.
    if args.list:
        for t in list_timestamps():
            print(t)
        return

    # --at TS: print the rows of one specific snapshot.
    if args.at:
        snap = read_snapshot(args.at)
        if snap.empty:
            print(f"No snapshot found for {args.at}")
        else:
            print(snap.to_string(index=False))
        return

    # No flags → run the full fetch-and-save flow.
    run_snapshot()


if __name__ == "__main__":
    # Top-level guard: log any unexpected crash and exit non-zero so CI fails loudly.
    try:
        main()
    except Exception:
        logger.exception("fatal error")
        sys.exit(1)
