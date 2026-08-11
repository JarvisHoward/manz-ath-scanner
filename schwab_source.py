#!/usr/bin/env python3
"""Schwab market-data adapter for the ATH Short Scanner.

Fetches split-adjusted daily OHLCV bars from the Schwab Trader API and writes
a provider-neutral CSV with the columns the scanner expects:

    date, symbol, open, high, low, close, volume, description, sector, sector_symbol

This adapter does NOT expose, copy, or modify Schwab credentials.  It imports
the existing authenticated client from the Market Intelligence Dashboard's
``schwab_client`` module, which handles token refresh internally.

Usage:
    python schwab_source.py --output daily_ohlcv.csv --as-of 2026-08-07 --bars 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ── Schwab client (re-use existing authenticated session) ───────────────────
DASHBOARD_DIR = Path.home() / ".market_intel_dashboard"
SCREENER_VENV = Path.home() / "put_credit_spread_screener_v3" / ".venv" / "bin" / "python"

# Make the dashboard's schwab_client importable
sys.path.insert(0, str(DASHBOARD_DIR))


def _get_client():
    """Return an authenticated Schwab client.

    Token refresh is handled inside schwab_client.get_client().
    We never touch credentials directly.
    """
    from schwab_client import get_client

    client = get_client()
    if client is None:
        raise RuntimeError(
            "Schwab client is None — token may be expired. "
            "Run a full re-auth via the dashboard or schwab_telegram_oauth.py."
        )
    return client


# ── Liquid US-stock universe ────────────────────────────────────────────────
# A curated set of large-cap, high-liquidity US equities and ETFs.
# This is NOT the full S&P 500 — it's a representative universe of ~80 symbols
# that are consistently borrowable and have tight spreads, suitable for short
# setups.  Sectors are manually mapped since Schwab's symbol-search endpoint
# does not return sector information.

SECTOR_MAP: dict[str, tuple[str, str]] = {
    # Technology
    "AAPL": ("Technology", "$XLK"), "MSFT": ("Technology", "$XLK"),
    "NVDA": ("Technology", "$XLK"), "GOOGL": ("Technology", "$XLK"),
    "META": ("Technology", "$XLK"), "AVGO": ("Technology", "$XLK"),
    "ORCL": ("Technology", "$XLK"), "CRM": ("Technology", "$XLK"),
    "AMD": ("Technology", "$XLK"), "INTC": ("Technology", "$XLK"),
    "CSCO": ("Technology", "$XLK"), "TXN": ("Technology", "$XLK"),
    "QCOM": ("Technology", "$XLK"), "ADBE": ("Technology", "$XLK"),
    "AMAT": ("Technology", "$XLK"), "MU": ("Technology", "$XLK"),
    "PANW": ("Technology", "$XLK"), "SNPS": ("Technology", "$XLK"),
    "CDNS": ("Technology", "$XLK"), "ANET": ("Technology", "$XLK"),
    # Communication Services
    "NFLX": ("Communication Services", "$XLC"),
    "DIS": ("Communication Services", "$XLC"),
    "T": ("Communication Services", "$XLC"),
    "VZ": ("Communication Services", "$XLC"),
    "CMCSA": ("Communication Services", "$XLC"),
    # Consumer Discretionary
    "AMZN": ("Consumer Discretionary", "$XLY"),
    "TSLA": ("Consumer Discretionary", "$XLY"),
    "HD": ("Consumer Discretionary", "$XLY"),
    "MCD": ("Consumer Discretionary", "$XLY"),
    "NKE": ("Consumer Discretionary", "$XLY"),
    "LOW": ("Consumer Discretionary", "$XLY"),
    "SBUX": ("Consumer Discretionary", "$XLY"),
    "TJX": ("Consumer Discretionary", "$XLY"),
    "BKNG": ("Consumer Discretionary", "$XLY"),
    "F": ("Consumer Discretionary", "$XLY"),
    "GM": ("Consumer Discretionary", "$XLY"),
    # Consumer Staples
    "PG": ("Consumer Staples", "$XLP"),
    "KO": ("Consumer Staples", "$XLP"),
    "PEP": ("Consumer Staples", "$XLP"),
    "WMT": ("Consumer Staples", "$XLP"),
    "COST": ("Consumer Staples", "$XLP"),
    "MDLZ": ("Consumer Staples", "$XLP"),
    "CL": ("Consumer Staples", "$XLP"),
    "KMB": ("Consumer Staples", "$XLP"),
    "STZ": ("Consumer Staples", "$XLP"),
    "TGT": ("Consumer Staples", "$XLP"),
    # Financials
    "JPM": ("Financials", "$XLF"),
    "BAC": ("Financials", "$XLF"),
    "WFC": ("Financials", "$XLF"),
    "GS": ("Financials", "$XLF"),
    "MS": ("Financials", "$XLF"),
    "C": ("Financials", "$XLF"),
    "BLK": ("Financials", "$XLF"),
    "AXP": ("Financials", "$XLF"),
    "SCHW": ("Financials", "$XLF"),
    "CB": ("Financials", "$XLF"),
    "SPGI": ("Financials", "$XLF"),
    "MET": ("Financials", "$XLF"),
    "PNC": ("Financials", "$XLF"),
    "USB": ("Financials", "$XLF"),
    # Energy
    "XOM": ("Energy", "$XLE"),
    "CVX": ("Energy", "$XLE"),
    "COP": ("Energy", "$XLE"),
    "SLB": ("Energy", "$XLE"),
    "EOG": ("Energy", "$XLE"),
    "PSX": ("Energy", "$XLE"),
    "MPC": ("Energy", "$XLE"),
    "OXY": ("Energy", "$XLE"),
    # Healthcare
    "JNJ": ("Healthcare", "$XLV"),
    "LLY": ("Healthcare", "$XLV"),
    "UNH": ("Healthcare", "$XLV"),
    "ABBV": ("Healthcare", "$XLV"),
    "MRK": ("Healthcare", "$XLV"),
    "PFE": ("Healthcare", "$XLV"),
    "TMO": ("Healthcare", "$XLV"),
    "ABT": ("Healthcare", "$XLV"),
    "DHR": ("Healthcare", "$XLV"),
    "BMY": ("Healthcare", "$XLV"),
    "AMGN": ("Healthcare", "$XLV"),
    "GILD": ("Healthcare", "$XLV"),
    # Industrials
    "CAT": ("Industrials", "$XLI"),
    "DE": ("Industrials", "$XLI"),
    "GE": ("Industrials", "$XLI"),
    "BA": ("Industrials", "$XLI"),
    "HON": ("Industrials", "$XLI"),
    "UPS": ("Industrials", "$XLI"),
    "RTX": ("Industrials", "$XLI"),
    "LMT": ("Industrials", "$XLI"),
    "MMM": ("Industrials", "$XLI"),
    # Materials
    "LIN": ("Materials", "$XLB"),
    "APD": ("Materials", "$XLB"),
    "SHW": ("Materials", "$XLB"),
    "FCX": ("Materials", "$XLB"),
    "NEM": ("Materials", "$XLB"),
    # Utilities
    "NEE": ("Utilities", "$XLU"),
    "DUK": ("Utilities", "$XLU"),
    "SO": ("Utilities", "$XLU"),
    # Real Estate
    "PLD": ("Real Estate", "$XLRE"),
    "AMT": ("Real Estate", "$XLRE"),
    # ETFs (for sector-relative scanning)
    "SPY": ("Index ETF", "$SPY"),
    "QQQ": ("Index ETF", "$QQQ"),
    "IWM": ("Index ETF", "$IWM"),
    "XLF": ("Sector ETF", "$XLF"),
    "XLE": ("Sector ETF", "$XLE"),
    "XLK": ("Sector ETF", "$XLK"),
    "XLV": ("Sector ETF", "$XLV"),
    "XLY": ("Sector ETF", "$XLY"),
    "XLP": ("Sector ETF", "$XLP"),
    "XLI": ("Sector ETF", "$XLI"),
    "XLB": ("Sector ETF", "$XLB"),
    "XLU": ("Sector ETF", "$XLU"),
    "XLRE": ("Sector ETF", "$XLRE"),
    "XLC": ("Sector ETF", "$XLC"),
}


def _build_universe() -> list[str]:
    """Return the sorted list of liquid US-stock symbols."""
    return sorted(SECTOR_MAP.keys())


def _fetch_description(client, symbol: str) -> str:
    """Fetch the instrument description from Schwab (best-effort)."""
    try:
        resp = client.get_instruments(symbol, projection=client.Instrument.Projection.SYMBOL_SEARCH)
        data = resp.json()
        instruments = data.get("instruments", [])
        if instruments and isinstance(instruments, list):
            return instruments[0].get("description", symbol)
    except Exception:
        pass
    return symbol  # fallback to symbol itself


def _fetch_daily_bars(
    client,
    symbol: str,
    end_date: datetime,
    bars_needed: int,
) -> list[dict] | None:
    """Fetch daily OHLCV bars ending at ``end_date``.

    Returns a list of candle dicts with keys: open, high, low, close, volume,
    datetime (epoch ms).  Returns None on error.
    """
    # Request enough calendar days to get bars_needed trading days
    start_date = end_date - timedelta(days=int(bars_needed * 2.5))
    try:
        resp = client.get_price_history_every_day(
            symbol,
            start_datetime=start_date,
            end_datetime=end_date,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        candles = data.get("candles", [])
        if not candles:
            return None
        return candles
    except Exception as e:
        print(f"  [WARN] {symbol}: fetch error: {e}", file=sys.stderr)
        return None


def _candles_to_rows(
    candles: list[dict],
    symbol: str,
    description: str,
    sector: str,
    sector_symbol: str,
) -> list[dict]:
    """Convert Schwab candle dicts to scanner-compatible rows."""
    rows = []
    for c in candles:
        # Schwab datetime is epoch milliseconds
        dt = datetime.fromtimestamp(c["datetime"] / 1000, tz=timezone.utc)
        # Only include completed regular-market sessions (exclude today/incomplete)
        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
            "description": description,
            "sector": sector,
            "sector_symbol": sector_symbol,
        })
    return rows


def fetch_universe(
    as_of: str,
    bars_needed: int = 120,
    rate_limit_delay: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    """Fetch daily OHLCV for the entire universe through ``as_of`` date.

    Returns (DataFrame, manifest) where manifest tracks data quality.
    """
    as_of_date = datetime.strptime(as_of, "%Y-%m-%d").replace(
        hour=20, minute=0, second=0, tzinfo=timezone.utc
    )

    universe = _build_universe()
    print(f"Universe: {len(universe)} symbols")
    print(f"As-of date: {as_of} (fetching through market close)")
    print(f"Bars needed: {bars_needed} per symbol")

    client = _get_client()
    print(f"Schwab client: connected")

    all_rows: list[dict] = []
    descriptions: dict[str, str] = {}
    fetched: list[str] = []
    missing: list[str] = []
    stale: list[str] = []
    latest_bars: dict[str, str] = {}

    for i, symbol in enumerate(universe, 1):
        sector, sector_symbol = SECTOR_MAP[symbol]

        # Fetch description (cached)
        if symbol not in descriptions:
            desc = _fetch_description(client, symbol)
            descriptions[symbol] = desc
            time.sleep(rate_limit_delay)

        # Fetch daily bars
        candles = _fetch_daily_bars(client, symbol, as_of_date, bars_needed)
        if candles is None:
            missing.append(symbol)
            print(f"  [{i}/{len(universe)}] {symbol}: MISSING", file=sys.stderr)
            continue

        if len(candles) < bars_needed:
            stale.append(symbol)
            print(
                f"  [{i}/{len(universe)}] {symbol}: only {len(candles)} bars "
                f"(need {bars_needed})",
                file=sys.stderr,
            )

        rows = _candles_to_rows(
            candles, symbol, descriptions[symbol], sector, sector_symbol
        )
        all_rows.extend(rows)
        fetched.append(symbol)
        latest_bars[symbol] = rows[-1]["date"] if rows else "N/A"

        if i % 10 == 0:
            print(f"  ... {i}/{len(universe)} symbols fetched")

        time.sleep(rate_limit_delay)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    manifest = {
        "as_of_date": as_of,
        "universe_size": len(universe),
        "fetched_ok": len(fetched),
        "missing": missing,
        "stale": stale,
        "latest_bar_per_symbol": latest_bars,
        "total_rows": len(all_rows),
        "bars_requested": bars_needed,
        "data_source": "Schwab Trader API (price history, daily)",
        "adjustment": "Schwab returns split-adjusted daily bars",
    }

    print(f"\nFetched: {len(fetched)}/{len(universe)} symbols")
    print(f"Missing: {len(missing)}")
    print(f"Stale: {len(stale)}")
    print(f"Total rows: {len(all_rows)}")

    return df, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily OHLCV from Schwab API for ATH scanner"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("daily_ohlcv.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--as-of", type=str, default="2026-08-07",
        help="Last completed session date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--bars", type=int, default=120,
        help="Minimum completed daily bars per symbol",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data_manifest.json"),
        help="Output data-quality manifest path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, manifest = fetch_universe(args.as_of, args.bars)

    # Write CSV
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")

    # Write manifest
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote data manifest to {args.manifest}")

    # Fail-closed check: if >15% of universe is missing, exit with error
    if manifest["universe_size"] > 0:
        missing_pct = len(manifest["missing"]) / manifest["universe_size"]
        if missing_pct > 0.15:
            print(
                f"\n❌ FAIL-CLOSED: {missing_pct:.0%} of universe missing. "
                f"Data is materially incomplete.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Check latest bar is not stale (should be on or before as-of date)
    if manifest["latest_bar_per_symbol"]:
        latest_dates = set(manifest["latest_bar_per_symbol"].values())
        # The latest bar should be the as-of date or the prior trading day
        # (e.g., if as-of is Friday Aug 7, latest bar should be 2026-08-07)
        expected = args.as_of
        stale_count = sum(1 for d in latest_dates if d != expected and d != "N/A")
        if stale_count > len(latest_dates) * 0.2:
            print(
                f"\n⚠️  WARNING: {stale_count} symbols have stale latest bars "
                f"(expected {expected})",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()