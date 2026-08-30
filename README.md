# NIFTY50 Option Chain Snapshots

A single-file tool that fetches the NIFTY50 option chain from the NSE API and
saves each capture as its own timestamped parquet file. Runs on demand from
GitHub Actions or locally.

## How it works

Everything lives in one self-contained script: [`snapshot.py`](snapshot.py).

Each run:

1. Gets the NIFTY50 spot price (yfinance, falling back to nselib)
2. Gets the list of near expiries
3. Fetches the full option chain from the NSE option-chain-v3 API
4. Reads implied volatility from the API and computes Greeks with `py_vollib`
   (delta, gamma, theta, vega, rho) — no hand-rolled formulas
5. Writes a **new** parquet file named by the exact run time (IST):

   ```
   data/option_chain_YYYYMMDD_HHMM.parquet
   ```

Every trigger produces its own file, for example:

| Trigger time (IST) | File |
| --- | --- |
| 31 Aug 2026, 10:15 | `data/option_chain_20260831_1015.parquet` |
| 31 Aug 2026, 11:30 | `data/option_chain_20260831_1130.parquet` |

## Running it

### On GitHub (manual)

Open the **Actions** tab → **Fetch Option Chain Snapshot** → **Run workflow**.
The workflow fetches a snapshot at that moment, commits the new parquet file,
and pushes it. There is also one scheduled run at 09:00 IST on weekdays.

### Locally

```bash
pip install -r requirements.txt
python snapshot.py
```

## Reading a snapshot back

```bash
python snapshot.py --list                  # list every captured timestamp
python snapshot.py --at 2026-08-31_10:15    # print that snapshot's rows
```

Or in Python:

```python
import snapshot
df = snapshot.read_snapshot("20260831_1015")   # DataFrame for that moment
```

## Parquet columns

| Column | Description |
| --- | --- |
| `timestamp` | Capture key, `YYYYMMDD_HHMM` (IST) |
| `datetime_ist` | Human-readable capture time |
| `symbol` | `NIFTY50` |
| `expiry` | Option expiry, e.g. `04-Sep-2026` |
| `strike` | Strike price |
| `option_type` | `CE` or `PE` |
| `spot` | Underlying spot at capture |
| `ltp` | Last traded price |
| `volume` | Total traded volume |
| `oi` | Open interest |
| `oi_chg` | Change in open interest |
| `iv` | Implied volatility (%) from the NSE API |
| `delta`, `gamma`, `theta`, `vega`, `rho` | Greeks via `py_vollib` |

## Requirements

`pandas`, `pytz`, `requests`, `nselib`, `yfinance`, `py_vollib`, `pyarrow`.
See [`requirements.txt`](requirements.txt).
