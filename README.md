# Around The Horn Short Scanner - Prototype v2

This provider-neutral prototype scans adjusted daily OHLCV data for three Adrian Manz short setups:

- Fast Ball Short: expansion of range and volume after a pullback or consolidation in a downtrend.
- Infield Fly Short: extension reversal after an extended move culminates in a weak gap-up day.
- Switch Hitter Short: ratio pullback after a strong down wave and a 1-5 day, lower-volume Fibonacci correction.

## What changed in v2

- Targets are no longer fixed at 1.5R.
- A setup qualifies only when a technical target offers at least the configured minimum, 1.0R by default; there is no preset R:R ceiling.
- Targets are selected from clustered natural support, next-day floor pivots, and pattern-specific Fibonacci or measured-move levels.
- The actual candidate R:R and the evidence behind the selected target appear in the CSV and PDF.
- Each run writes `hermes_candidates.json`, a broker-neutral machine-readable handoff with order transmission disabled and human approval required.
- `nightly_runner.py` checks the official XNYS session calendar, so weekends and US market holidays are skipped.

The target selector reports any strong support cluster that lies before the selected target as an intervening-support review flag. A setup is rejected when no technical level offers at least the minimum R:R. A single technical target is retained with a separate review flag; confluence is preferred.

## Outputs

- `candidates.csv`: qualifying setups and evidence.
- `trade_plan.pdf`: one formatted plan page per candidate.
- `hermes_candidates.json`: structured, review-gated agent handoff.
- `run_manifest.json`: run status and target policy.

## Input contract

Required CSV columns:

```text
date,symbol,open,high,low,close,volume
```

Optional report columns:

```text
description,sector,sector_symbol
```

Use split-adjusted daily bars consistently and provide at least 60 bars per symbol. The scanner is not tied to IBKR: any data provider or internal export can produce this contract.

## Run the demonstration

```bash
python ath_scanner.py --demo --output-dir output
```

## Scan provider data

```bash
python ath_scanner.py \
  --input daily_ohlcv.csv \
  --output-dir output \
  --risk 1000 \
  --min-rr 1.0
```

## Nightly market-session wrapper

```bash
python nightly_runner.py \
  --input daily_ohlcv.csv \
  --output-root nightly_output \
  --risk 1000 \
  --min-rr 1.0
```

Schedule this after the chosen provider has finalized its adjusted daily bars. The wrapper uses the XNYS calendar and creates a dated output directory. Delivery (email, cloud folder, or agent inbox) should be a separate adapter so scanning remains broker- and transport-neutral.

## Pattern implementation

### Fast Ball Short

- SMA20 below SMA50, falling SMA20, ADX at least 20.
- Pullback or 5-15 day consolidation precursor.
- Downside break on the widest range of the last 10 sessions.
- Volume expansion and a close in the bottom 25% of the setup bar.
- Target inputs include floor pivots, natural support, 127.2%/150% extensions, and an 80% measured continuation.

### Infield Fly Short

- Prior advance of at least 5%, or 10% below $40.
- Gap open above the prior high; bearish lower-half close.
- Trigger $0.10 below the gap-day low.
- Target inputs include the gap-fill/previous high, natural support, floor pivots, and 38.2%/50%/61.8% retracements of the advance.

### Switch Hitter Short

- Initial down wave of at least 5%, or 10% below $40, ending at a two-week low.
- One-to-five-session correction retracing 38.2%-61.8% on contracting volume.
- Bearish reversal bar, preferably closing in the bottom 25%.
- Target inputs include the 38.2% pullback level, initial-wave low, 127.2%/150% extensions, natural support, and floor pivots.

## Important limitations

The protective stop remains provisional at setup-bar resistance plus the configured buffer. Manz's full method also uses intraday five-minute support/resistance and VWAP, which daily OHLCV cannot reconstruct. Before trading, review borrow availability, spread/liquidity, news and earnings, market/sector behavior, intraday obstacles, entry trigger, stop, and target.

The Hermes JSON is deliberately a candidate package, not an executable order ticket. Automatic order placement should remain disabled until the scanner has been replayed against labeled history and a separate execution policy, permissions, and human-approval gate have been tested.
