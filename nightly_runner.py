#!/usr/bin/env python3
"""Market-calendar-aware wrapper for a provider-neutral nightly ATH scan."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from ath_scanner import ScanConfig, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ATH scanner on US market sessions")
    parser.add_argument("--input", required=True, type=Path, help="Provider-neutral adjusted daily OHLCV CSV")
    parser.add_argument("--output-root", type=Path, default=Path("nightly_output"))
    parser.add_argument("--as-of", type=date.fromisoformat, help="Session date; defaults to current New York date")
    parser.add_argument("--risk", type=float, default=1000.0)
    parser.add_argument("--min-rr", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Run even when the date is not an XNYS session")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_date = args.as_of or datetime.now(ZoneInfo("America/New_York")).date()
    calendar = xcals.get_calendar("XNYS")
    is_session = calendar.is_session(pd.Timestamp(session_date))
    if not is_session and not args.force:
        print(f"Skipped {session_date}: not an XNYS trading session")
        return

    destination = args.output_root / session_date.isoformat()
    config = ScanConfig(risk_dollars=args.risk, minimum_reward_to_risk=args.min_rr)
    candidates = run(args.input, destination, config)
    print(f"Completed {session_date}: {len(candidates)} candidate(s) in {destination}")


if __name__ == "__main__":
    main()
