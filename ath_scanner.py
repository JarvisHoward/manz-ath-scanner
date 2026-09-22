#!/usr/bin/env python3
"""Around The Horn short-setup scanner prototype.

Scans daily OHLCV data for:
  * Fast Ball Short (expansion of range and volume)
  * Infield Fly Short (extension reversal)
  * Switch Hitter Short (ratio pullback)

The implementation deliberately separates candidate detection from next-day
execution. Pattern definitions are based on the supplied Adrian Manz material;
several discretionary concepts are exposed as parameters so they can be
calibrated against the labeled 2024-2026 trade logs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Iterable, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ath_scanner_mpl"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


REQUIRED_COLUMNS = {"date", "symbol", "open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class ScanConfig:
    risk_dollars: float = 1000.0
    minimum_reward_to_risk: float = 1.0
    maximum_reward_to_risk: float | None = None
    entry_buffer: float = 0.10
    stop_buffer: float = 0.10
    support_lookback: int = 120
    support_cluster_atr_fraction: float = 0.15
    fastball_range_lookback: int = 10
    fastball_precursor_days: int = 10
    close_extreme_fraction: float = 0.25
    adx_minimum: float = 20.0
    fastball_consolidation_min_days: int = 3
    fastball_consolidation_max_atr: float = 2.5
    linedrive_gap_buffer: float = 0.10
    linedrive_stop_buffer: float = 0.10
    linedrive_min_rr: float = 2.0
    # Day-trade ATR-based stops/targets (added 2026-08-16)
    atr_stop_mult: float = 0.35       # Stop at 0.35x ATR from entry
    atr_target_mult: float = 0.75     # Target at 0.75x ATR from entry
    use_atr_stops: bool = True        # Use ATR-based stops instead of swing-level
    switch_correction_min: int = 1
    switch_correction_max: int = 5
    switch_retrace_min: float = 0.382
    switch_retrace_max: float = 0.618


@dataclass
class Candidate:
    date: str
    symbol: str
    description: str
    sector: str
    sector_symbol: str
    pattern: str
    position: str
    score: float
    entry: float
    stop: float
    target: float
    target_rr: float
    target_method: str
    target_evidence: str
    halfway: float
    risk_per_share: float
    shares: int
    position_value: float
    pivot: float
    resistance_1: float
    resistance_2: float
    support_1: float
    support_2: float
    reasons: str
    warnings: str


@dataclass(frozen=True)
class TargetLevel:
    price: float
    label: str
    source: str


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator and np.isfinite(denominator) else np.nan


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high_diff = frame["high"].diff()
    low_diff = -frame["low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr = _true_range(frame).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.replace([np.inf, -np.inf], np.nan).ewm(alpha=1 / period, adjust=False).mean()


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("OHLCV columns contain missing or non-numeric values")

    result["range"] = result["high"] - result["low"]
    result["atr14"] = _true_range(result).rolling(14).mean()
    result["sma20"] = result["close"].rolling(20).mean()
    result["sma50"] = result["close"].rolling(50).mean()
    result["adx14"] = _adx(result)
    result["close_pct"] = (result["close"] - result["low"]) / result["range"].replace(0, np.nan)
    return result


def floor_pivots(high: float, low: float, close: float) -> dict[str, float]:
    pivot = (high + low + close) / 3.0
    return {
        "pivot": pivot,
        "resistance_1": 2 * pivot - low,
        "resistance_2": pivot + (high - low),
        "support_1": 2 * pivot - high,
        "support_2": pivot - (high - low),
    }


def _daily_support_levels(frame: pd.DataFrame, index: int, lookback: int) -> list[TargetLevel]:
    """Return objective support observations available before the setup closes."""
    start = max(2, index - lookback)
    levels: list[TargetLevel] = []
    for i in range(start, max(start, index - 1)):
        low = float(frame["low"].iloc[i])
        window = frame["low"].iloc[max(0, i - 2) : min(index, i + 3)]
        if len(window) >= 3 and low <= float(window.min()) + 1e-9:
            stamp = pd.Timestamp(frame["date"].iloc[i]).date().isoformat()
            levels.append(TargetLevel(low, f"daily swing low {stamp}", "natural support"))
    return levels


def _daily_resistance_levels(frame: pd.DataFrame, index: int, lookback: int) -> list[TargetLevel]:
    """Return objective resistance observations available before the setup closes."""
    start = max(2, index - lookback)
    levels: list[TargetLevel] = []
    for i in range(start, max(start, index - 1)):
        high = float(frame["high"].iloc[i])
        window = frame["high"].iloc[max(0, i - 2) : min(index, i + 3)]
        if len(window) >= 3 and high >= float(window.max()) - 1e-9:
            stamp = pd.Timestamp(frame["date"].iloc[i]).date().isoformat()
            levels.append(TargetLevel(high, f"daily swing high {stamp}", "natural resistance"))
    return levels


def _cluster_levels(levels: list[TargetLevel], tolerance: float) -> list[list[TargetLevel]]:
    clusters: list[list[TargetLevel]] = []
    for level in sorted(levels, key=lambda item: item.price, reverse=True):
        match = next(
            (
                cluster
                for cluster in clusters
                if abs(level.price - np.mean([item.price for item in cluster])) <= tolerance
            ),
            None,
        )
        if match is None:
            clusters.append([level])
        else:
            match.append(level)
    return clusters


def select_short_target(
    frame: pd.DataFrame,
    index: int,
    entry: float,
    stop: float,
    pivots: dict[str, float],
    pattern_levels: Iterable[TargetLevel],
    config: ScanConfig,
) -> tuple[float, float, str, str, list[str]] | None:
    """Select a support/Fibonacci target; never manufacture one from a fixed R multiple."""
    risk = stop - entry
    if risk <= 0:
        return None

    levels = _daily_support_levels(frame, index, config.support_lookback)
    levels.extend(pattern_levels)
    for key, label in [("support_1", "next-day floor pivot S1"), ("support_2", "next-day floor pivot S2")]:
        levels.append(TargetLevel(float(pivots[key]), label, "floor pivot"))

    levels = [item for item in levels if np.isfinite(item.price) and item.price < entry]
    atr = float(frame["atr14"].iloc[index]) if pd.notna(frame["atr14"].iloc[index]) else risk
    tolerance = max(0.10, config.support_cluster_atr_fraction * atr)
    clusters = _cluster_levels(levels, tolerance)
    qualified: list[tuple[float, float, int, list[TargetLevel]]] = []
    blockers: list[str] = []
    source_weight = {"natural support": 3, "floor pivot": 2, "fibonacci": 2, "gap support": 3, "measured move": 2}

    for cluster in clusters:
        price = float(np.mean([item.price for item in cluster]))
        rr = (entry - price) / risk
        sources = {item.source for item in cluster}
        strength = sum(source_weight.get(source, 1) for source in sources) + min(3, len(cluster) - 1)
        evidence = ", ".join(dict.fromkeys(item.label for item in cluster))
        if 0 < rr < config.minimum_reward_to_risk and (len(sources) >= 2 or len(cluster) >= 2):
            blockers.append(f"support before 1R near ${price:.2f}: {evidence}")
        within_maximum = config.maximum_reward_to_risk is None or rr <= config.maximum_reward_to_risk
        if config.minimum_reward_to_risk <= rr and within_maximum:
            qualified.append((price, rr, strength, cluster))

    if not qualified:
        return None

    nearest_rr = min(item[1] for item in qualified)
    shortlist = [item for item in qualified if item[1] <= nearest_rr + 0.75]
    price, rr, _, cluster = max(shortlist, key=lambda item: (item[2], -item[1]))
    sources = sorted({item.source for item in cluster})
    method = " + ".join(sources)
    evidence = "; ".join(dict.fromkeys(item.label for item in cluster))
    warnings: list[str] = [f"intervening {item}" for item in blockers]
    if len(sources) == 1 and len(cluster) == 1:
        warnings.append("target has one technical confirmation; review intraday support before trading")
    return round(price, 2), round(rr, 2), method, evidence, warnings


def select_long_target(
    frame: pd.DataFrame,
    index: int,
    entry: float,
    stop: float,
    pivots: dict[str, float],
    pattern_levels: Iterable[TargetLevel],
    config: ScanConfig,
) -> tuple[float, float, str, str, list[str]] | None:
    """Select a resistance/Fibonacci target for long positions; never manufacture one from a fixed R multiple."""
    risk = entry - stop
    if risk <= 0:
        return None

    levels = _daily_resistance_levels(frame, index, config.support_lookback)
    levels.extend(pattern_levels)
    for key, label in [("resistance_1", "next-day floor pivot R1"), ("resistance_2", "next-day floor pivot R2")]:
        levels.append(TargetLevel(float(pivots[key]), label, "floor pivot"))

    levels = [item for item in levels if np.isfinite(item.price) and item.price > entry]
    atr = float(frame["atr14"].iloc[index]) if pd.notna(frame["atr14"].iloc[index]) else risk
    tolerance = max(0.10, config.support_cluster_atr_fraction * atr)
    clusters = _cluster_levels(levels, tolerance)
    qualified: list[tuple[float, float, int, list[TargetLevel]]] = []
    blockers: list[str] = []
    source_weight = {"natural resistance": 3, "floor pivot": 2, "fibonacci": 2, "gap resistance": 3, "measured move": 2}

    for cluster in clusters:
        price = float(np.mean([item.price for item in cluster]))
        rr = (price - entry) / risk
        sources = {item.source for item in cluster}
        strength = sum(source_weight.get(source, 1) for source in sources) + min(3, len(cluster) - 1)
        evidence = ", ".join(dict.fromkeys(item.label for item in cluster))
        if 0 < rr < config.minimum_reward_to_risk and (len(sources) >= 2 or len(cluster) >= 2):
            blockers.append(f"resistance before 1R near ${price:.2f}: {evidence}")
        within_maximum = config.maximum_reward_to_risk is None or rr <= config.maximum_reward_to_risk
        if config.minimum_reward_to_risk <= rr and within_maximum:
            qualified.append((price, rr, strength, cluster))

    if not qualified:
        return None

    nearest_rr = min(item[1] for item in qualified)
    shortlist = [item for item in qualified if item[1] <= nearest_rr + 0.75]
    price, rr, _, cluster = max(shortlist, key=lambda item: (item[2], -item[1]))
    sources = sorted({item.source for item in cluster})
    method = " + ".join(sources)
    evidence = "; ".join(dict.fromkeys(item.label for item in cluster))
    warnings: list[str] = [f"intervening {item}" for item in blockers]
    if len(sources) == 1 and len(cluster) == 1:
        warnings.append("target has one technical confirmation; review intraday resistance before trading")
    return round(price, 2), round(rr, 2), method, evidence, warnings


def build_candidate(
    frame: pd.DataFrame,
    index: int,
    pattern: str,
    score: float,
    reasons: Iterable[str],
    config: ScanConfig,
    target_levels: Iterable[TargetLevel] = (),
    warnings: Iterable[str] = (),
    position: str = "Short",
) -> Candidate | None:
    row = frame.iloc[index]
    if position == "Long":
        entry = round(float(row["high"]) + config.entry_buffer, 2)
        stop = round(float(row["low"]) - config.stop_buffer, 2)
    else:
        entry = round(float(row["low"]) - config.entry_buffer, 2)
        # This is intentionally provisional. Manz often uses chart resistance rather
        # than a universal formula for the protective stop.
        stop = round(float(row["high"]) + config.stop_buffer, 2)
    risk_per_share = round(abs(stop - entry), 2)
    pivots = floor_pivots(float(row["high"]), float(row["low"]), float(row["close"]))

    # ATR-based day-trade stops/targets (2026-08-16)
    if config.use_atr_stops:
        atr_raw = cast(float, row["atr14"]) if "atr14" in row.index else np.nan
        atr = float(atr_raw) if not pd.isna(atr_raw) else 0
        if atr > 0:
            if position == "Long":
                stop = round(entry - config.atr_stop_mult * atr, 2)
                target = round(entry + config.atr_target_mult * atr, 2)
            else:
                stop = round(entry + config.atr_stop_mult * atr, 2)
                target = round(entry - config.atr_target_mult * atr, 2)
            # Override target selection — use ATR target directly
            risk_per_share = round(abs(stop - entry), 2)
            target_rr = round(abs(target - entry) / risk_per_share, 2) if risk_per_share > 0 else 0
            target_method = f"{config.atr_stop_mult}x/{config.atr_target_mult}x ATR day-trade"
            target_evidence = f"ATR14=${atr:.2f}; stop={config.atr_stop_mult}x ATR, target={config.atr_target_mult}x ATR"
            # Skip the select_short_target/select_long_target call entirely
            # Compute halfway and shares from the ATR values
            if position == "Long":
                halfway = round(entry + 0.5 * (target - entry), 2)
            else:
                halfway = round(entry - 0.5 * (entry - target), 2)
            shares = math.floor(config.risk_dollars / risk_per_share) if risk_per_share > 0 else 0
            # Return directly — don't call the target selection functions
            return Candidate(
                date=pd.Timestamp(row["date"]).date().isoformat(),
                symbol=str(row["symbol"]),
                description=str(row.get("description", "")) if pd.notna(row.get("description", "")) else "",
                sector=str(row.get("sector", "")) if pd.notna(row.get("sector", "")) else "",
                sector_symbol=str(row.get("sector_symbol", "")) if pd.notna(row.get("sector_symbol", "")) else "",
                pattern=pattern,
                position=position,
                score=round(score, 1),
                entry=entry,
                stop=stop,
                target=target,
                target_rr=target_rr,
                target_method=target_method,
                target_evidence=target_evidence,
                halfway=halfway,
                risk_per_share=risk_per_share,
                shares=shares,
                position_value=round(shares * entry, 2),
                reasons="; ".join(reasons),
                warnings="; ".join(warnings),
                **{name: round(value, 2) for name, value in pivots.items()},
            )

    if position == "Long":
        selected = select_long_target(frame, index, entry, stop, pivots, target_levels, config)
    else:
        selected = select_short_target(frame, index, entry, stop, pivots, target_levels, config)
    if selected is None:
        return None
    target, target_rr, target_method, target_evidence, target_warnings = selected
    if position == "Long":
        halfway = round(entry + 0.5 * (target - entry), 2)
    else:
        halfway = round(entry - 0.5 * (entry - target), 2)
    shares = math.floor(config.risk_dollars / risk_per_share) if risk_per_share > 0 else 0
    return Candidate(
        date=pd.Timestamp(row["date"]).date().isoformat(),
        symbol=str(row["symbol"]),
        description=str(row.get("description", "")) if pd.notna(row.get("description", "")) else "",
        sector=str(row.get("sector", "")) if pd.notna(row.get("sector", "")) else "",
        sector_symbol=str(row.get("sector_symbol", "")) if pd.notna(row.get("sector_symbol", "")) else "",
        pattern=pattern,
        position=position,
        score=round(score, 1),
        entry=entry,
        stop=stop,
        target=target,
        target_rr=target_rr,
        target_method=target_method,
        target_evidence=target_evidence,
        halfway=halfway,
        risk_per_share=risk_per_share,
        shares=shares,
        position_value=round(shares * entry, 2),
        reasons="; ".join(reasons),
        warnings="; ".join([*warnings, *target_warnings]),
        **{name: round(value, 2) for name, value in pivots.items()},
    )


def _entry_stop_for_row(
    row: pd.Series,
    config: ScanConfig,
    position: str,
    entry_buffer: float | None = None,
    stop_buffer: float | None = None,
) -> tuple[float, float]:
    entry_offset = config.entry_buffer if entry_buffer is None else entry_buffer
    stop_offset = config.stop_buffer if stop_buffer is None else stop_buffer
    if position == "Long":
        entry = round(float(row["high"]) + entry_offset, 2)
        stop = round(float(row["low"]) - stop_offset, 2)
    else:
        entry = round(float(row["low"]) - entry_offset, 2)
        stop = round(float(row["high"]) + stop_offset, 2)
    if config.use_atr_stops:
        atr_raw = cast(float, row["atr14"]) if "atr14" in row.index else np.nan
        atr = float(atr_raw) if not pd.isna(atr_raw) else 0
        if atr > 0:
            if position == "Long":
                stop = round(entry - config.atr_stop_mult * atr, 2)
            else:
                stop = round(entry + config.atr_stop_mult * atr, 2)
    return entry, stop


def _vwap_between_entry_stop(frame: pd.DataFrame, index: int, entry: float, stop: float) -> bool:
    if "vwap" not in frame.columns:
        return False
    value = frame["vwap"].iloc[index]
    if pd.isna(value):
        return False
    vwap = float(value)
    return bool(np.isfinite(vwap) and min(entry, stop) <= vwap <= max(entry, stop))


def _fastball_fibonacci_retracement(frame: pd.DataFrame, index: int, config: ScanConfig, position: str) -> float | None:
    prior = frame.iloc[index - config.fastball_precursor_days : index]
    if len(prior) < config.fastball_precursor_days:
        return None

    if position == "Long":
        pivot_offset = int(np.argmax(prior["high"].to_numpy()))
        trend_high = float(prior["high"].iloc[pivot_offset])
        trend_low = float(prior["low"].iloc[: pivot_offset + 1].min())
        move = trend_high - trend_low
        if move <= 0:
            return None
        pullback_low = float(prior["low"].iloc[pivot_offset:].min())
        retracement = _safe_ratio(trend_high - pullback_low, move)
    else:
        pivot_offset = int(np.argmin(prior["low"].to_numpy()))
        trend_low = float(prior["low"].iloc[pivot_offset])
        trend_high = float(prior["high"].iloc[: pivot_offset + 1].max())
        move = trend_high - trend_low
        if move <= 0:
            return None
        pullback_high = float(prior["high"].iloc[pivot_offset:].max())
        retracement = _safe_ratio(pullback_high - trend_low, move)

    return float(retracement) if np.isfinite(retracement) else None


def _find_fastball_consolidation(
    frame: pd.DataFrame,
    index: int,
    config: ScanConfig,
) -> tuple[int, float, float, float, float, float] | None:
    row = frame.iloc[index]
    atr = float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return None

    best: tuple[int, float, float, float, float, float] | None = None
    max_days = min(config.fastball_precursor_days, index)
    for days in range(config.fastball_consolidation_min_days, max_days + 1):
        consolidation = frame.iloc[index - days : index]
        if len(consolidation) < config.fastball_consolidation_min_days:
            continue
        consolidation_high = float(consolidation["high"].max())
        consolidation_low = float(consolidation["low"].min())
        consolidation_span = consolidation_high - consolidation_low
        maximum_span = config.fastball_consolidation_max_atr * atr
        if consolidation_span > maximum_span:
            continue

        trend_start = max(0, index - days - config.fastball_precursor_days)
        trend_window = frame.iloc[trend_start : index - days]
        if len(trend_window) < config.fastball_consolidation_min_days:
            continue
        trend_return = _safe_ratio(float(trend_window["close"].iloc[-1]), float(trend_window["close"].iloc[0])) - 1
        trend_range = float(trend_window["high"].max() - trend_window["low"].min())
        trend_range_pct = _safe_ratio(trend_range, float(trend_window["close"].iloc[0]))
        if not np.isfinite(trend_return):
            continue
        if abs(trend_return) < 0.03 and (not np.isfinite(trend_range_pct) or trend_range_pct < 0.05):
            continue

        tightness = max(0.0, 1.0 - _safe_ratio(consolidation_span, maximum_span))
        candidate = (days, consolidation_high, consolidation_low, consolidation_span, tightness, float(trend_return))
        if best is None or days > best[0] or (days == best[0] and tightness > best[4]):
            best = candidate

    return best


def detect_fast_ball_short(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    if index < max(55, config.fastball_precursor_days + 10):
        return None
    row = frame.iloc[index]
    prior = frame.iloc[index - config.fastball_precursor_days : index]
    range_prior = frame["range"].iloc[index - config.fastball_range_lookback + 1 : index]

    downtrend = bool(
        row["sma20"] < row["sma50"]
        and frame["sma20"].iloc[index] < frame["sma20"].iloc[index - 5]
        and row["adx14"] >= config.adx_minimum
    )
    widest_ten = bool(row["range"] >= range_prior.max())
    volume_ratio = _safe_ratio(float(row["volume"]), float(prior["volume"].mean()))
    volume_expansion = bool(volume_ratio > 1.0)
    lower_quarter_close = bool(row["close_pct"] <= config.close_extreme_fraction)
    downside_break = bool(row["low"] < prior["low"].tail(5).min())

    five_day_return = _safe_ratio(float(prior["close"].iloc[-1]), float(prior["close"].iloc[-5])) - 1
    pullback = bool(five_day_return > 0)
    consolidation_span = float(prior["high"].max() - prior["low"].min())
    consolidation = bool(consolidation_span <= 2.5 * float(row["atr14"]))
    valid_precursor = pullback or consolidation

    checks = [downtrend, widest_ten, volume_expansion, lower_quarter_close, downside_break, valid_precursor]
    if not all(checks):
        return None

    reasons = [
        f"downtrend ADX {row['adx14']:.1f}",
        "widest range of 10 sessions",
        f"volume {volume_ratio:.2f}x precursor average",
        f"close in bottom {row['close_pct']:.0%} of bar",
        "downside break from precursor",
        "pullback precursor" if pullback else "5-15 day consolidation precursor",
    ]
    score = 65 + min(15, max(0, (volume_ratio - 1) * 15)) + min(10, row["adx14"] / 5) + (10 if pullback else 7)
    retracement = _fastball_fibonacci_retracement(frame, index, config, "Short")
    if retracement is not None and 0.382 <= retracement <= 0.500:
        score += 10
        reasons.append(f"pullback retraced {retracement:.1%} into 38.2%-50% Fibonacci zone")
    entry, stop = _entry_stop_for_row(row, config, "Short")
    if _vwap_between_entry_stop(frame, index, entry, stop):
        score += 10
        reasons.append("VWAP sits between entry and provisional stop")
    volume_20_ratio = _safe_ratio(float(row["volume"]), float(frame["volume"].iloc[index - 20 : index].mean()))
    if volume_20_ratio > 1.5:
        score += 5
        reasons.append(f"daily volume {volume_20_ratio:.2f}x 20-day average (5-min close-volume proxy)")
    score = min(100.0, score)
    precursor_high = float(prior["high"].max())
    precursor_low = float(prior["low"].min())
    precursor_range = precursor_high - precursor_low
    target_levels = [
        TargetLevel(precursor_low - 0.272 * precursor_range, "127.2% precursor-range extension", "fibonacci"),
        TargetLevel(precursor_low - 0.500 * precursor_range, "150% precursor-range extension", "fibonacci"),
        TargetLevel(precursor_low - 0.800 * precursor_range, "80% measured continuation", "measured move"),
        TargetLevel(precursor_low - 1.272 * precursor_range, "227.2% precursor-range extension", "fibonacci"),
    ]
    return build_candidate(frame, index, "Fast Ball Short", score, reasons, config, target_levels)


def detect_infield_fly_short(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """Extension reversal: sharp advance, gap higher, weak lower-half close."""
    if index < 12:
        return None
    row = frame.iloc[index]
    previous = frame.iloc[index - 1]
    lookback = frame.iloc[index - 10 : index]
    prior_low = float(lookback["low"].min())
    extension = _safe_ratio(float(previous["close"]), prior_low) - 1
    minimum_extension = 0.10 if row["close"] < 40 else 0.05

    gap_higher = bool(row["open"] > previous["high"])
    full_gap = bool(row["low"] > previous["high"])
    weak_close = bool(row["close_pct"] <= 0.50)
    upward_extension = bool(extension >= minimum_extension)
    bearish_gap_bar = bool(row["close"] < row["open"])

    if not all([gap_higher, weak_close, upward_extension, bearish_gap_bar]):
        return None

    reasons = [
        f"prior advance {extension:.1%}",
        "gap opened above prior high",
        f"gap-day close at {row['close_pct']:.0%} of range",
        "bearish gap-day bar",
    ]
    if full_gap:
        reasons.append("open profit room below gap-day low")
    warnings = [] if full_gap else ["gap overlaps prior range; source says pattern can still work"]
    score = min(100.0, 68 + min(17, extension * 100) + (10 if full_gap else 3) + max(0, (0.5 - row["close_pct"]) * 20))
    advance_high = float(row["high"])
    advance_range = advance_high - prior_low
    target_levels = [
        TargetLevel(float(previous["high"]), "previous-day high / gap-fill support", "gap support"),
        TargetLevel(advance_high - 0.382 * advance_range, "38.2% advance retracement", "fibonacci"),
        TargetLevel(advance_high - 0.500 * advance_range, "50% advance retracement", "fibonacci"),
        TargetLevel(advance_high - 0.618 * advance_range, "61.8% advance retracement", "fibonacci"),
    ]
    return build_candidate(frame, index, "Infield Fly Short", score, reasons, config, target_levels, warnings)


def detect_switch_hitter_short(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """Ratio pullback with the photographed long rules reversed for a short."""
    if index < 25:
        return None
    row = frame.iloc[index]
    best: tuple[float, list[str], list[TargetLevel]] | None = None

    for correction_days in range(config.switch_correction_min, config.switch_correction_max + 1):
        pivot_index = index - correction_days
        if pivot_index < 12:
            continue
        pivot_low = float(frame["low"].iloc[pivot_index])
        two_week_low = float(frame["low"].iloc[pivot_index - 9 : pivot_index + 1].min())
        if pivot_low > two_week_low + 1e-9:
            continue

        initial_window = frame.iloc[pivot_index - 10 : pivot_index + 1]
        start_relative = int(np.argmax(initial_window["high"].to_numpy()))
        start_index = pivot_index - 10 + start_relative
        start_high = float(frame["high"].iloc[start_index])
        wave_size = start_high - pivot_low
        if wave_size <= 0:
            continue
        minimum_move = 0.10 if row["close"] < 40 else 0.05
        move_fraction = wave_size / start_high
        if move_fraction < minimum_move:
            continue

        correction = frame.iloc[pivot_index + 1 : index + 1]
        correction_high = float(correction["high"].max())
        retracement = (correction_high - pivot_low) / wave_size
        if not (config.switch_retrace_min <= retracement <= config.switch_retrace_max):
            continue

        initial_wave = frame.iloc[start_index : pivot_index + 1]
        volume_contracts = float(correction["volume"].mean()) < float(initial_wave["volume"].mean())
        bearish_reversal = bool(row["close"] < row["open"] and row["close_pct"] <= config.close_extreme_fraction)
        if not (volume_contracts and bearish_reversal):
            continue

        volume_ratio = _safe_ratio(float(correction["volume"].mean()), float(initial_wave["volume"].mean()))
        reasons = [
            f"initial down wave {move_fraction:.1%}",
            "initial wave ended at two-week low",
            f"{correction_days}-day correction",
            f"retracement {retracement:.1%}",
            f"correction volume {volume_ratio:.2f}x initial-wave volume",
            f"bearish close in bottom {row['close_pct']:.0%} of bar",
        ]
        target_levels = [
            TargetLevel(pivot_low + 0.382 * wave_size, "38.2% pullback level", "fibonacci"),
            TargetLevel(pivot_low, "100% return to initial-wave low", "fibonacci"),
            TargetLevel(pivot_low - 0.272 * wave_size, "127.2% initial-wave extension", "fibonacci"),
            TargetLevel(pivot_low - 0.500 * wave_size, "150% initial-wave extension", "fibonacci"),
        ]
        score = min(100.0, 68 + min(12, move_fraction * 100) + max(0, (1 - volume_ratio) * 15) + max(0, (0.25 - row["close_pct"]) * 20))
        if best is None or score > best[0]:
            best = (score, reasons, target_levels)

    return build_candidate(frame, index, "Switch Hitter Short", best[0], best[1], config, best[2]) if best else None


def detect_fast_ball_long(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """Uptrend mirror of Fast Ball Short: expansion of range/volume, close in top 25%, upside break."""
    if index < max(55, config.fastball_precursor_days + 10):
        return None
    row = frame.iloc[index]
    prior = frame.iloc[index - config.fastball_precursor_days : index]
    range_prior = frame["range"].iloc[index - config.fastball_range_lookback + 1 : index]

    uptrend = bool(
        row["sma20"] > row["sma50"]
        and frame["sma20"].iloc[index] > frame["sma20"].iloc[index - 5]
        and row["adx14"] >= config.adx_minimum
    )
    widest_ten = bool(row["range"] >= range_prior.max())
    volume_ratio = _safe_ratio(float(row["volume"]), float(prior["volume"].mean()))
    volume_expansion = bool(volume_ratio > 1.0)
    upper_quarter_close = bool(row["close_pct"] >= 1.0 - config.close_extreme_fraction)
    upside_break = bool(row["high"] > prior["high"].tail(5).max())

    five_day_return = _safe_ratio(float(prior["close"].iloc[-1]), float(prior["close"].iloc[-5])) - 1
    pullback = bool(five_day_return < 0)
    consolidation_span = float(prior["high"].max() - prior["low"].min())
    consolidation = bool(consolidation_span <= 2.5 * float(row["atr14"]))
    valid_precursor = pullback or consolidation

    checks = [uptrend, widest_ten, volume_expansion, upper_quarter_close, upside_break, valid_precursor]
    if not all(checks):
        return None

    reasons = [
        f"uptrend ADX {row['adx14']:.1f}",
        "widest range of 10 sessions",
        f"volume {volume_ratio:.2f}x precursor average",
        f"close in top {row['close_pct']:.0%} of bar",
        "upside break from precursor",
        "pullback precursor" if pullback else "5-15 day consolidation precursor",
    ]
    score = 65 + min(15, max(0, (volume_ratio - 1) * 15)) + min(10, row["adx14"] / 5) + (10 if pullback else 7)
    retracement = _fastball_fibonacci_retracement(frame, index, config, "Long")
    if retracement is not None and 0.382 <= retracement <= 0.500:
        score += 10
        reasons.append(f"pullback retraced {retracement:.1%} into 38.2%-50% Fibonacci zone")
    entry, stop = _entry_stop_for_row(row, config, "Long")
    if _vwap_between_entry_stop(frame, index, entry, stop):
        score += 10
        reasons.append("VWAP sits between entry and provisional stop")
    volume_20_ratio = _safe_ratio(float(row["volume"]), float(frame["volume"].iloc[index - 20 : index].mean()))
    if volume_20_ratio > 1.5:
        score += 5
        reasons.append(f"daily volume {volume_20_ratio:.2f}x 20-day average (5-min close-volume proxy)")
    score = min(100.0, score)
    precursor_high = float(prior["high"].max())
    precursor_low = float(prior["low"].min())
    precursor_range = precursor_high - precursor_low
    target_levels = [
        TargetLevel(precursor_high + 0.272 * precursor_range, "127.2% precursor-range extension", "fibonacci"),
        TargetLevel(precursor_high + 0.500 * precursor_range, "150% precursor-range extension", "fibonacci"),
        TargetLevel(precursor_high + 0.800 * precursor_range, "80% measured continuation", "measured move"),
        TargetLevel(precursor_high + 1.272 * precursor_range, "227.2% precursor-range extension", "fibonacci"),
    ]
    return build_candidate(frame, index, "Fast Ball Long", score, reasons, config, target_levels, position="Long")


def detect_fast_ball_consolidation_short(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    if index < max(55, config.fastball_precursor_days + 10):
        return None
    row = frame.iloc[index]
    consolidation = _find_fastball_consolidation(frame, index, config)
    if consolidation is None:
        return None
    days, consolidation_high, consolidation_low, consolidation_span, tightness, trend_return = consolidation
    consolidation_frame = frame.iloc[index - days : index]
    range_prior = frame["range"].iloc[index - config.fastball_range_lookback + 1 : index]

    widest_ten = bool(row["range"] >= range_prior.max())
    volume_ratio = _safe_ratio(float(row["volume"]), float(consolidation_frame["volume"].mean()))
    lower_quarter_close = bool(row["close_pct"] <= config.close_extreme_fraction)
    downside_break = bool(row["low"] < consolidation_low and row["close"] < consolidation_low)
    adx_ok = bool(row["adx14"] >= config.adx_minimum)

    if not all([widest_ten, volume_ratio > 1.0, lower_quarter_close, downside_break, adx_ok]):
        return None

    continuation = trend_return < 0
    direction_bonus = 5 if continuation else 3
    score = (
        60
        + min(15, max(0, (volume_ratio - 1) * 15))
        + min(10, row["adx14"] / 5)
        + min(10, max(0, tightness * 10))
        + direction_bonus
    )
    entry, stop = _entry_stop_for_row(row, config, "Short")
    reasons = [
        f"{days}-day consolidation span {_safe_ratio(consolidation_span, float(row['atr14'])):.2f} ATR",
        "widest range of 10 sessions",
        f"volume {volume_ratio:.2f}x consolidation average",
        f"close in bottom {row['close_pct']:.0%} of bar",
        "breakout below consolidation low",
        "trend-continuation break" if continuation else "trend-reversal break",
        f"ADX {row['adx14']:.1f}",
    ]
    if _vwap_between_entry_stop(frame, index, entry, stop):
        score += 5
        reasons.append("VWAP sits between entry and provisional stop")
    score = min(100.0, score)

    consolidation_range = consolidation_high - consolidation_low
    target_levels = [
        TargetLevel(consolidation_low - 0.272 * consolidation_range, "127.2% consolidation-range extension", "fibonacci"),
        TargetLevel(consolidation_low - 0.500 * consolidation_range, "150% consolidation-range extension", "fibonacci"),
        TargetLevel(consolidation_low - 0.800 * consolidation_range, "80% measured continuation", "measured move"),
        TargetLevel(consolidation_low - 1.272 * consolidation_range, "227.2% consolidation-range extension", "fibonacci"),
    ]
    return build_candidate(frame, index, "Fast Ball Consolidation Short", score, reasons, config, target_levels)


def detect_fast_ball_consolidation_long(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    if index < max(55, config.fastball_precursor_days + 10):
        return None
    row = frame.iloc[index]
    consolidation = _find_fastball_consolidation(frame, index, config)
    if consolidation is None:
        return None
    days, consolidation_high, consolidation_low, consolidation_span, tightness, trend_return = consolidation
    consolidation_frame = frame.iloc[index - days : index]
    range_prior = frame["range"].iloc[index - config.fastball_range_lookback + 1 : index]

    widest_ten = bool(row["range"] >= range_prior.max())
    volume_ratio = _safe_ratio(float(row["volume"]), float(consolidation_frame["volume"].mean()))
    upper_quarter_close = bool(row["close_pct"] >= 1.0 - config.close_extreme_fraction)
    upside_break = bool(row["high"] > consolidation_high and row["close"] > consolidation_high)
    adx_ok = bool(row["adx14"] >= config.adx_minimum)

    if not all([widest_ten, volume_ratio > 1.0, upper_quarter_close, upside_break, adx_ok]):
        return None

    continuation = trend_return > 0
    direction_bonus = 5 if continuation else 3
    score = (
        60
        + min(15, max(0, (volume_ratio - 1) * 15))
        + min(10, row["adx14"] / 5)
        + min(10, max(0, tightness * 10))
        + direction_bonus
    )
    entry, stop = _entry_stop_for_row(row, config, "Long")
    reasons = [
        f"{days}-day consolidation span {_safe_ratio(consolidation_span, float(row['atr14'])):.2f} ATR",
        "widest range of 10 sessions",
        f"volume {volume_ratio:.2f}x consolidation average",
        f"close in top {row['close_pct']:.0%} of bar",
        "breakout above consolidation high",
        "trend-continuation break" if continuation else "trend-reversal break",
        f"ADX {row['adx14']:.1f}",
    ]
    if _vwap_between_entry_stop(frame, index, entry, stop):
        score += 5
        reasons.append("VWAP sits between entry and provisional stop")
    score = min(100.0, score)

    consolidation_range = consolidation_high - consolidation_low
    target_levels = [
        TargetLevel(consolidation_high + 0.272 * consolidation_range, "127.2% consolidation-range extension", "fibonacci"),
        TargetLevel(consolidation_high + 0.500 * consolidation_range, "150% consolidation-range extension", "fibonacci"),
        TargetLevel(consolidation_high + 0.800 * consolidation_range, "80% measured continuation", "measured move"),
        TargetLevel(consolidation_high + 1.272 * consolidation_range, "227.2% consolidation-range extension", "fibonacci"),
    ]
    return build_candidate(frame, index, "Fast Ball Consolidation Long", score, reasons, config, target_levels, position="Long")


def detect_line_drive_gap_long(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    if index < max(55, config.fastball_precursor_days + 10):
        return None
    row = frame.iloc[index]
    previous = frame.iloc[index - 1]

    breakaway_gap = bool(row["low"] > previous["high"])
    ten_day_high = bool(row["high"] >= frame["high"].iloc[index - 9 : index + 1].max())
    green_bar = bool(row["close"] > row["open"])
    upper_quarter_close = bool(row["close_pct"] >= 1.0 - config.close_extreme_fraction)
    if not all([breakaway_gap, ten_day_high, green_bar, upper_quarter_close]):
        return None

    entry, stop = _entry_stop_for_row(
        row,
        config,
        "Long",
        entry_buffer=config.linedrive_gap_buffer,
        stop_buffer=config.linedrive_stop_buffer,
    )
    risk = entry - stop
    if risk <= 0:
        return None

    atr = float(row["atr14"]) if pd.notna(row["atr14"]) else np.nan
    gap_size = float(row["low"] - previous["high"])
    gap_atr_ratio = _safe_ratio(gap_size, atr)
    gap_bonus = min(10, max(0, gap_atr_ratio * 10)) if np.isfinite(gap_atr_ratio) else 0
    close_bonus = min(10, max(0, (float(row["close_pct"]) - 0.75) / 0.25 * 10))
    volume_20_ratio = _safe_ratio(float(row["volume"]), float(frame["volume"].iloc[index - 20 : index].mean()))
    volume_bonus = min(10, max(0, (volume_20_ratio - 1) * 10)) if np.isfinite(volume_20_ratio) else 0
    twenty_day_high = bool(row["high"] >= frame["high"].iloc[index - 19 : index + 1].max())
    score = min(100.0, 65 + gap_bonus + close_bonus + volume_bonus + (5 if twenty_day_high else 0))

    reasons = [
        f"breakaway gap ${gap_size:.2f} above prior high",
        "made a 10-day high",
        "green breakaway bar",
        f"close in top {row['close_pct']:.0%} of bar",
        f"entry ${entry:.2f} above breakaway high; stop ${stop:.2f} below breakaway low",
        f"volume {volume_20_ratio:.2f}x 20-day average" if np.isfinite(volume_20_ratio) else "20-day volume comparison unavailable",
    ]
    if twenty_day_high:
        reasons.append("also made a 20-day high")

    required_rr = max(config.minimum_reward_to_risk, config.linedrive_min_rr)
    line_drive_config = replace(
        config,
        entry_buffer=config.linedrive_gap_buffer,
        stop_buffer=config.linedrive_stop_buffer,
        minimum_reward_to_risk=required_rr,
    )
    target_levels = [
        TargetLevel(entry + risk * required_rr, f"{required_rr:.1f}:1 measured move from breakaway bar", "measured move"),
    ]
    warnings = ["Line Drive Gap Extension is an intraday setup; scanner does not manage end-of-session exit"]
    return build_candidate(frame, index, "Line Drive Gap Long", score, reasons, line_drive_config, target_levels, warnings, position="Long")


def detect_infield_fly_long(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """Extension reversal: sharp DECLINE, gap LOWER, weak close in TOP half, bullish gap-day bar."""
    if index < 12:
        return None
    row = frame.iloc[index]
    previous = frame.iloc[index - 1]
    lookback = frame.iloc[index - 10 : index]
    prior_high = float(lookback["high"].max())
    decline = _safe_ratio(prior_high, float(previous["close"])) - 1
    minimum_decline = 0.10 if row["close"] < 40 else 0.05

    gap_lower = bool(row["open"] < previous["low"])
    full_gap = bool(row["high"] < previous["low"])
    weak_close = bool(row["close_pct"] >= 0.50)  # close in top half = bullish reversal
    downward_extension = bool(decline >= minimum_decline)
    bullish_gap_bar = bool(row["close"] > row["open"])

    if not all([gap_lower, weak_close, downward_extension, bullish_gap_bar]):
        return None

    reasons = [
        f"prior decline {decline:.1%}",
        "gap opened below prior low",
        f"gap-day close at {row['close_pct']:.0%} of range",
        "bullish gap-day bar",
    ]
    if full_gap:
        reasons.append("open profit room above gap-day high")
    warnings = [] if full_gap else ["gap overlaps prior range; source says pattern can still work"]
    score = min(100.0, 68 + min(17, decline * 100) + (10 if full_gap else 3) + max(0, (row["close_pct"] - 0.5) * 20))
    decline_low = float(row["low"])
    decline_range = prior_high - decline_low
    target_levels = [
        TargetLevel(float(previous["low"]), "previous-day low / gap-fill resistance", "gap resistance"),
        TargetLevel(decline_low + 0.382 * decline_range, "38.2% decline retracement", "fibonacci"),
        TargetLevel(decline_low + 0.500 * decline_range, "50% decline retracement", "fibonacci"),
        TargetLevel(decline_low + 0.618 * decline_range, "61.8% decline retracement", "fibonacci"),
    ]
    return build_candidate(frame, index, "Infield Fly Long", score, reasons, config, target_levels, warnings, position="Long")


def detect_switch_hitter_long(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """Ratio pullback mirror for long: initial UP wave to two-week high, 1-5 day correction, 38.2-61.8% retracement, bullish reversal."""
    if index < 25:
        return None
    row = frame.iloc[index]
    best: tuple[float, list[str], list[TargetLevel]] | None = None

    for correction_days in range(config.switch_correction_min, config.switch_correction_max + 1):
        pivot_index = index - correction_days
        if pivot_index < 12:
            continue
        pivot_high = float(frame["high"].iloc[pivot_index])
        two_week_high = float(frame["high"].iloc[pivot_index - 9 : pivot_index + 1].max())
        if pivot_high < two_week_high - 1e-9:
            continue

        initial_window = frame.iloc[pivot_index - 10 : pivot_index + 1]
        start_relative = int(np.argmin(initial_window["low"].to_numpy()))
        start_index = pivot_index - 10 + start_relative
        start_low = float(frame["low"].iloc[start_index])
        wave_size = pivot_high - start_low
        if wave_size <= 0:
            continue
        minimum_move = 0.10 if row["close"] < 40 else 0.05
        move_fraction = wave_size / start_low
        if move_fraction < minimum_move:
            continue

        correction = frame.iloc[pivot_index + 1 : index + 1]
        correction_low = float(correction["low"].min())
        retracement = (pivot_high - correction_low) / wave_size
        if not (config.switch_retrace_min <= retracement <= config.switch_retrace_max):
            continue

        initial_wave = frame.iloc[start_index : pivot_index + 1]
        volume_contracts = float(correction["volume"].mean()) < float(initial_wave["volume"].mean())
        bullish_reversal = bool(row["close"] > row["open"] and row["close_pct"] >= 1.0 - config.close_extreme_fraction)
        if not (volume_contracts and bullish_reversal):
            continue

        volume_ratio = _safe_ratio(float(correction["volume"].mean()), float(initial_wave["volume"].mean()))
        reasons = [
            f"initial up wave {move_fraction:.1%}",
            "initial wave ended at two-week high",
            f"{correction_days}-day correction",
            f"retracement {retracement:.1%}",
            f"correction volume {volume_ratio:.2f}x initial-wave volume",
            f"bullish close in top {row['close_pct']:.0%} of bar",
        ]
        target_levels = [
            TargetLevel(pivot_high - 0.382 * wave_size, "38.2% pullback level", "fibonacci"),
            TargetLevel(pivot_high, "100% return to initial-wave high", "fibonacci"),
            TargetLevel(pivot_high + 0.272 * wave_size, "127.2% initial-wave extension", "fibonacci"),
            TargetLevel(pivot_high + 0.500 * wave_size, "150% initial-wave extension", "fibonacci"),
        ]
        score = min(100.0, 68 + min(12, move_fraction * 100) + max(0, (1 - volume_ratio) * 15) + max(0, (row["close_pct"] - 0.75) * 20))
        if best is None or score > best[0]:
            best = (score, reasons, target_levels)

    return build_candidate(frame, index, "Switch Hitter Long", best[0], best[1], config, best[2], position="Long") if best else None


def detect_3_2_pitch_short(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """3-2 Pitch Short: downtrend, 3 bullish pullback bars, then 2 bearish resumption bars, widest range, volume expansion."""
    if index < 20:
        return None
    row = frame.iloc[index]
    # Check downtrend
    if row["sma20"] >= row["sma50"] or row["adx14"] < config.adx_minimum:
        return None

    # Look at last 5 bars: bars -4 to 0 (index-4 to index)
    bars = frame.iloc[index - 4 : index + 1]
    if len(bars) != 5:
        return None

    # First 3 bars should be bullish pullback (close > open)
    pullback_bars = bars.iloc[:3]
    if not all(pullback_bars["close"] > pullback_bars["open"]):
        return None

    # Last 2 bars should be bearish resumption (close < open)
    resumption_bars = bars.iloc[3:]
    if not all(resumption_bars["close"] < resumption_bars["open"]):
        return None

    # Setup bar (current bar, index) must be widest range in 10 sessions
    range_prior = frame["range"].iloc[index - 9 : index]
    if row["range"] < range_prior.max():
        return None

    # Volume expansion > 1.0x precursor average (prior 5 bars before the 5-bar pattern)
    prior = frame.iloc[index - 9 : index - 4]
    volume_ratio = _safe_ratio(float(row["volume"]), float(prior["volume"].mean()))
    if volume_ratio <= 1.0:
        return None

    # Close in bottom 25% of bar (bearish close)
    if row["close_pct"] > config.close_extreme_fraction:
        return None

    reasons = [
        f"downtrend ADX {row['adx14']:.1f}",
        "3-bar bullish pullback then 2-bar bearish resumption",
        "widest range of 10 sessions",
        f"volume {volume_ratio:.2f}x precursor average",
        f"close in bottom {row['close_pct']:.0%} of bar",
    ]
    score = min(100.0, 60 + min(15, max(0, (volume_ratio - 1) * 10)) + min(10, row["adx14"] / 5))

    # Targets same as Fast Ball: Fibonacci extensions below precursor low
    prior_all = frame.iloc[index - 10 : index]
    precursor_high = float(prior_all["high"].max())
    precursor_low = float(prior_all["low"].min())
    precursor_range = precursor_high - precursor_low
    target_levels = [
        TargetLevel(precursor_low - 0.272 * precursor_range, "127.2% precursor-range extension", "fibonacci"),
        TargetLevel(precursor_low - 0.500 * precursor_range, "150% precursor-range extension", "fibonacci"),
        TargetLevel(precursor_low - 0.800 * precursor_range, "80% measured continuation", "measured move"),
        TargetLevel(precursor_low - 1.272 * precursor_range, "227.2% precursor-range extension", "fibonacci"),
    ]
    return build_candidate(frame, index, "3-2 Pitch Short", score, reasons, config, target_levels)


def detect_3_2_pitch_long(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """3-2 Pitch Long: uptrend, 3 bearish pullback bars, then 2 bullish resumption bars, widest range, volume expansion."""
    if index < 20:
        return None
    row = frame.iloc[index]
    # Check uptrend
    if row["sma20"] <= row["sma50"] or row["adx14"] < config.adx_minimum:
        return None

    # Look at last 5 bars: bars -4 to 0 (index-4 to index)
    bars = frame.iloc[index - 4 : index + 1]
    if len(bars) != 5:
        return None

    # First 3 bars should be bearish pullback (close < open)
    pullback_bars = bars.iloc[:3]
    if not all(pullback_bars["close"] < pullback_bars["open"]):
        return None

    # Last 2 bars should be bullish resumption (close > open)
    resumption_bars = bars.iloc[3:]
    if not all(resumption_bars["close"] > resumption_bars["open"]):
        return None

    # Setup bar (current bar, index) must be widest range in 10 sessions
    range_prior = frame["range"].iloc[index - 9 : index]
    if row["range"] < range_prior.max():
        return None

    # Volume expansion > 1.0x precursor average
    prior = frame.iloc[index - 9 : index - 4]
    volume_ratio = _safe_ratio(float(row["volume"]), float(prior["volume"].mean()))
    if volume_ratio <= 1.0:
        return None

    # Close in top 25% of bar (bullish close)
    if row["close_pct"] < 1.0 - config.close_extreme_fraction:
        return None

    reasons = [
        f"uptrend ADX {row['adx14']:.1f}",
        "3-bar bearish pullback then 2-bar bullish resumption",
        "widest range of 10 sessions",
        f"volume {volume_ratio:.2f}x precursor average",
        f"close in top {row['close_pct']:.0%} of bar",
    ]
    score = min(100.0, 60 + min(15, max(0, (volume_ratio - 1) * 10)) + min(10, row["adx14"] / 5))

    # Targets: Fibonacci extensions above precursor high
    prior_all = frame.iloc[index - 10 : index]
    precursor_high = float(prior_all["high"].max())
    precursor_low = float(prior_all["low"].min())
    precursor_range = precursor_high - precursor_low
    target_levels = [
        TargetLevel(precursor_high + 0.272 * precursor_range, "127.2% precursor-range extension", "fibonacci"),
        TargetLevel(precursor_high + 0.500 * precursor_range, "150% precursor-range extension", "fibonacci"),
        TargetLevel(precursor_high + 0.800 * precursor_range, "80% measured continuation", "measured move"),
        TargetLevel(precursor_high + 1.272 * precursor_range, "227.2% precursor-range extension", "fibonacci"),
    ]
    return build_candidate(frame, index, "3-2 Pitch Long", score, reasons, config, target_levels, position="Long")


def detect_double_header_short(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """Double Header Short: two consecutive bearish expansion bars after pullback/consolidation."""
    if index < 15:
        return None
    row = frame.iloc[index]
    prev = frame.iloc[index - 1]

    # Check downtrend
    if row["sma20"] >= row["sma50"] or row["adx14"] < config.adx_minimum:
        return None

    # Both bars must be bearish (close < open)
    if not (row["close"] < row["open"] and prev["close"] < prev["open"]):
        return None

    # Second bar range >= first bar range (expansion continues)
    if row["range"] < prev["range"]:
        return None

    # Combined volume > 2x average volume (lookback 10 bars before the two bars)
    prior = frame.iloc[index - 11 : index - 1]
    avg_volume = float(prior["volume"].mean())
    combined_volume = float(row["volume"] + prev["volume"])
    if combined_volume <= 2.0 * avg_volume:
        return None

    # Check for pullback/consolidation before the two bars (5-10 bars prior)
    pullback_region = frame.iloc[index - 10 : index - 2]
    five_day_return = _safe_ratio(float(pullback_region["close"].iloc[-1]), float(pullback_region["close"].iloc[0])) - 1
    consolidation_span = float(pullback_region["high"].max() - pullback_region["low"].min())
    consolidation = bool(consolidation_span <= 2.5 * float(row["atr14"]))
    if not (five_day_return > 0 or consolidation):
        return None

    reasons = [
        f"downtrend ADX {row['adx14']:.1f}",
        "two consecutive bearish expansion bars",
        f"second bar range >= first bar range ({row['range']:.2f} >= {prev['range']:.2f})",
        f"combined volume {combined_volume / avg_volume:.2f}x average",
        "pullback/consolidation precursor",
    ]
    score = min(100.0, 65 + min(15, max(0, (combined_volume / avg_volume - 2) * 5)) + min(10, row["adx14"] / 5))

    # Entry = second bar low - buffer, Stop = first bar high + buffer
    # Targets: Fibonacci extensions from combined range of both bars
    combined_low = min(row["low"], prev["low"])
    combined_high = max(row["high"], prev["high"])
    combined_range = combined_high - combined_low
    target_levels = [
        TargetLevel(combined_low - 0.272 * combined_range, "127.2% combined-range extension", "fibonacci"),
        TargetLevel(combined_low - 0.500 * combined_range, "150% combined-range extension", "fibonacci"),
        TargetLevel(combined_low - 1.000 * combined_range, "200% combined-range extension", "measured move"),
        TargetLevel(combined_low - 1.272 * combined_range, "227.2% combined-range extension", "fibonacci"),
    ]
    return build_candidate(frame, index, "Double Header Short", score, reasons, config, target_levels)


def detect_double_header_long(frame: pd.DataFrame, index: int, config: ScanConfig) -> Candidate | None:
    """Double Header Long: two consecutive bullish expansion bars after pullback/consolidation."""
    if index < 15:
        return None
    row = frame.iloc[index]
    prev = frame.iloc[index - 1]

    # Check uptrend
    if row["sma20"] <= row["sma50"] or row["adx14"] < config.adx_minimum:
        return None

    # Both bars must be bullish (close > open)
    if not (row["close"] > row["open"] and prev["close"] > prev["open"]):
        return None

    # Second bar range >= first bar range (expansion continues)
    if row["range"] < prev["range"]:
        return None

    # Combined volume > 2x average volume
    prior = frame.iloc[index - 11 : index - 1]
    avg_volume = float(prior["volume"].mean())
    combined_volume = float(row["volume"] + prev["volume"])
    if combined_volume <= 2.0 * avg_volume:
        return None

    # Check for pullback/consolidation before the two bars
    pullback_region = frame.iloc[index - 10 : index - 2]
    five_day_return = _safe_ratio(float(pullback_region["close"].iloc[-1]), float(pullback_region["close"].iloc[0])) - 1
    consolidation_span = float(pullback_region["high"].max() - pullback_region["low"].min())
    consolidation = bool(consolidation_span <= 2.5 * float(row["atr14"]))
    if not (five_day_return < 0 or consolidation):
        return None

    reasons = [
        f"uptrend ADX {row['adx14']:.1f}",
        "two consecutive bullish expansion bars",
        f"second bar range >= first bar range ({row['range']:.2f} >= {prev['range']:.2f})",
        f"combined volume {combined_volume / avg_volume:.2f}x average",
        "pullback/consolidation precursor",
    ]
    score = min(100.0, 65 + min(15, max(0, (combined_volume / avg_volume - 2) * 5)) + min(10, row["adx14"] / 5))

    # Entry = second bar high + buffer, Stop = first bar low - buffer
    # Targets: Fibonacci extensions from combined range of both bars
    combined_low = min(row["low"], prev["low"])
    combined_high = max(row["high"], prev["high"])
    combined_range = combined_high - combined_low
    target_levels = [
        TargetLevel(combined_high + 0.272 * combined_range, "127.2% combined-range extension", "fibonacci"),
        TargetLevel(combined_high + 0.500 * combined_range, "150% combined-range extension", "fibonacci"),
        TargetLevel(combined_high + 1.000 * combined_range, "200% combined-range extension", "measured move"),
        TargetLevel(combined_high + 1.272 * combined_range, "227.2% combined-range extension", "fibonacci"),
    ]
    return build_candidate(frame, index, "Double Header Long", score, reasons, config, target_levels, position="Long")


def scan_latest(data: pd.DataFrame, config: ScanConfig) -> tuple[list[Candidate], dict[str, pd.DataFrame]]:
    candidates: list[Candidate] = []
    prepared: dict[str, pd.DataFrame] = {}
    for symbol, raw_frame in data.groupby("symbol", sort=True):
        frame = prepare_frame(raw_frame)
        prepared[str(symbol)] = frame
        index = len(frame) - 1
        for detector in [
            detect_fast_ball_short, detect_fast_ball_long,
            detect_fast_ball_consolidation_short, detect_fast_ball_consolidation_long,
            detect_line_drive_gap_long,
            detect_infield_fly_short, detect_infield_fly_long,
            detect_switch_hitter_short, detect_switch_hitter_long,
            detect_3_2_pitch_short, detect_3_2_pitch_long,
            detect_double_header_short, detect_double_header_long,
        ]:
            candidate = detector(frame, index, config)
            if candidate:
                candidates.append(candidate)
    candidates.sort(key=lambda item: (-item.score, item.symbol, item.pattern))
    return candidates, prepared


def _draw_candles(frame: pd.DataFrame, destination: Path, candidate: Candidate) -> None:
    plot = frame.tail(65).copy()
    dates = mdates.date2num(plot["date"].to_numpy())
    fig = plt.figure(figsize=(8.0, 3.65), dpi=160)
    grid = fig.add_gridspec(4, 1, hspace=0.05)
    ax = fig.add_subplot(grid[:3, 0])
    volume_ax = fig.add_subplot(grid[3, 0], sharex=ax)
    width = 0.55
    for x, (_, row) in zip(dates, plot.iterrows()):
        up = row["close"] >= row["open"]
        color = "#207567" if up else "#B84B4B"
        ax.vlines(x, row["low"], row["high"], color=color, linewidth=0.8)
        body_low = min(row["open"], row["close"])
        body_height = max(abs(row["close"] - row["open"]), 0.01)
        ax.add_patch(plt.Rectangle((x - width / 2, body_low), width, body_height, facecolor=color, edgecolor=color, linewidth=0.6))
        volume_ax.bar(x, row["volume"] / 1_000_000, width=width, color=color, alpha=0.8)

    if plot["sma20"].notna().any():
        ax.plot(dates, plot["sma20"], color="#31688E", linewidth=1.0, label="SMA20")
    if plot["sma50"].notna().any():
        ax.plot(dates, plot["sma50"], color="#9B4F96", linewidth=1.0, label="SMA50")
    ax.axhline(candidate.entry, color="#C5413B", linewidth=0.8, linestyle="--", label="Entry")
    ax.axhline(candidate.stop, color="#5B6670", linewidth=0.7, linestyle=":", label="Stop")
    ax.axhline(candidate.target, color="#2A8B57", linewidth=0.8, linestyle="--", label=f"Technical Target ({candidate.target_rr:.2f}R)")
    ax.set_title(f"{candidate.symbol} - {candidate.pattern}", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="y", color="#E6E8EB", linewidth=0.5)
    ax.legend(loc="best", fontsize=6, ncol=5, frameon=False)
    volume_ax.grid(axis="y", color="#EEF0F2", linewidth=0.4)
    volume_ax.set_ylabel("Volume (M)", fontsize=6)
    volume_ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    volume_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(volume_ax.get_xticklabels(), fontsize=6)
    ax.tick_params(labelsize=6)
    volume_ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.13)
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _paragraph(pdf: canvas.Canvas, text: str, x: float, y: float, width: float, height: float, style: ParagraphStyle) -> None:
    paragraph = Paragraph(text, style)
    paragraph.wrapOn(pdf, width, height)
    paragraph.drawOn(pdf, x, y - paragraph.height)


def generate_trade_plan_pdf(
    candidates: list[Candidate],
    frames: dict[str, pd.DataFrame],
    output_path: Path,
    config: ScanConfig,
) -> None:
    if not candidates:
        raise ValueError("Cannot generate a trade plan without candidates")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix="ath-plan-"))
    pdfmetrics.registerFont(TTFont("ATHSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("ATHSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("ATHSerif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"))
    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    page_width, page_height = letter
    styles = getSampleStyleSheet()
    note_style = ParagraphStyle("note", parent=styles["BodyText"], fontName="ATHSans", fontSize=7.4, leading=9.2, textColor=colors.HexColor("#27313A"), alignment=TA_LEFT)
    footer_style = ParagraphStyle("footer", parent=note_style, fontSize=6.5, leading=8, textColor=colors.HexColor("#626B73"), alignment=TA_CENTER)

    for page_number, candidate in enumerate(candidates, start=1):
        chart_path = temporary_dir / f"{candidate.symbol}-{page_number}.png"
        _draw_candles(frames[candidate.symbol], chart_path, candidate)

        pdf.setFillColor(colors.HexColor("#17222B"))
        pdf.setFont("ATHSerif", 24)
        pdf.drawCentredString(page_width / 2, page_height - 43, "Around The Horn Scanner Plan")
        pdf.setFont("ATHSerif", 14)
        pdf.drawCentredString(page_width / 2, page_height - 64, candidate.date)
        pdf.setStrokeColor(colors.HexColor("#33424D"))
        pdf.setLineWidth(1.2)
        pdf.line(38, page_height - 77, page_width - 38, page_height - 77)

        pdf.drawImage(str(chart_path), 44, 386, width=524, height=270, preserveAspectRatio=True, anchor="c")

        left_x, value_x = 55, 150
        details = [
            ("Symbol", candidate.symbol),
            ("Description", candidate.description or "-"),
            ("Pattern", candidate.pattern),
            ("Sector", candidate.sector or "-"),
            ("Position", candidate.position),
            ("Entry", f"${candidate.entry:.2f}"),
            ("Provisional Stop", f"${candidate.stop:.2f}"),
            ("Technical Target", f"${candidate.target:.2f}"),
            ("Reward / Risk", f"{candidate.target_rr:.2f}:1"),
            ("50% To Target", f"${candidate.halfway:.2f}"),
            ("Shares @ Risk", f"{candidate.shares:,}"),
        ]
        y = 354
        pdf.setFont("ATHSans", 8.2)
        for label, value in details:
            pdf.setFillColor(colors.HexColor("#4C5963"))
            pdf.drawString(left_x, y, label)
            pdf.setFillColor(colors.HexColor("#17222B"))
            pdf.setFont("ATHSans-Bold" if label in {"Pattern", "Position"} else "ATHSans", 8.2)
            pdf.drawString(value_x, y, str(value))
            pdf.setFont("ATHSans", 8.2)
            y -= 15

        table_x, table_y, row_height, table_width = 330, 351, 20, 220
        pivot_rows = [
            ("Resistance 2", candidate.resistance_2, "#F3D648"),
            ("Resistance 1", candidate.resistance_1, "#7EA1C4"),
            ("Pivot", candidate.pivot, "#C95D5D"),
            ("Support 1", candidate.support_1, "#72A66D"),
            ("Support 2", candidate.support_2, "#F3D648"),
        ]
        pdf.setFont("ATHSans-Bold", 8)
        for idx, (label, value, fill) in enumerate(pivot_rows):
            row_y = table_y - idx * row_height
            pdf.setFillColor(colors.HexColor(fill))
            pdf.rect(table_x, row_y - row_height + 4, table_width - 58, row_height, fill=1, stroke=0)
            pdf.setFillColor(colors.HexColor("#20272D"))
            pdf.drawString(table_x + 7, row_y - 10, label)
            pdf.drawRightString(table_x + table_width, row_y - 10, f"${value:.2f}")

        pdf.setFillColor(colors.HexColor("#17222B"))
        pdf.setFont("ATHSans-Bold", 9)
        pdf.drawString(330, 240, f"Candidate Score: {candidate.score:.0f}/100")
        pdf.setFont("ATHSans", 7.5)
        pdf.drawString(330, 225, f"Risk/share: ${candidate.risk_per_share:.2f}   Position value: ${candidate.position_value:,.0f}")
        _paragraph(pdf, f"<b>Target basis:</b> {candidate.target_method}<br/>{candidate.target_evidence}", 330, 210, 220, 45, note_style)

        pdf.setStrokeColor(colors.HexColor("#CBD1D5"))
        pdf.line(45, 174, page_width - 45, 174)
        pdf.setFont("ATHSans-Bold", 8.5)
        pdf.drawString(50, 158, "Why it qualified")
        _paragraph(pdf, candidate.reasons, 50, 149, 512, 48, note_style)
        if candidate.warnings:
            pdf.setFillColor(colors.HexColor("#A34A2C"))
            pdf.setFont("ATHSans-Bold", 7.5)
            pdf.drawString(50, 104, "Review flag")
            _paragraph(pdf, candidate.warnings, 108, 110, 454, 25, note_style)

        disclaimer = (
            "Scanner candidate only - not an order. Target is support/Fibonacci-driven and passed the configured minimum R:R; "
            "the stop remains a provisional chart-level estimate and must be reviewed. "
            "Confirm borrow availability, spread, news, market/sector context, and next-day trigger behavior before trading."
        )
        _paragraph(pdf, disclaimer, 50, 79, 512, 35, footer_style)
        pdf.setFont("ATHSans", 6.5)
        pdf.setFillColor(colors.HexColor("#737C84"))
        pdf.drawRightString(page_width - 42, 28, f"Page {page_number} of {len(candidates)}")
        pdf.showPage()

    # ── Summary page ──
    summary_date = candidates[0].date
    pdf.setFillColor(colors.HexColor("#17222B"))
    pdf.rect(0, page_height - 80, page_width, 80, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#FFFFFF"))
    pdf.setFont("ATHSerif", 22)
    pdf.drawCentredString(page_width / 2, page_height - 42, f"Trade Summary — {summary_date}")
    pdf.setFont("ATHSans", 11)
    pdf.drawCentredString(page_width / 2, page_height - 62, "Scanner candidates only — verify before trading")

    table_columns = [
        ("#", 25),
        ("Symbol", 55),
        ("Pattern", 100),
        ("Dir", 35),
        ("Entry", 50),
        ("Stop", 50),
        ("Target", 50),
        ("RR", 30),
        ("Score", 35),
        ("Shares", 45),
        ("Risk $", 45),
    ]
    table_width = sum(width for _, width in table_columns)
    table_x = (page_width - table_width) / 2
    table_top = page_height - 126
    header_height = 24
    row_height = 24
    row_pad = 5

    header_bg = colors.HexColor("#17222B")
    header_fg = colors.HexColor("#FFFFFF")
    border = colors.HexColor("#D7DCE0")
    odd_bg = colors.HexColor("#FFFFFF")
    even_bg = colors.HexColor("#F5F5F5")
    text_color = colors.HexColor("#17222B")
    muted_text = colors.HexColor("#4C5963")
    long_bg = colors.HexColor("#72A66D")
    short_bg = colors.HexColor("#C95D5D")

    def draw_cell_text(text: str, x: float, y: float, width: float, font: str = "ATHSans", size: float = 7.0,
                       align: str = "left", fill: colors.Color = text_color) -> None:
        value = str(text)
        max_width = max(width - 2 * row_pad, 4)
        while value and pdf.stringWidth(value, font, size) > max_width:
            value = value[:-1]
        if value != str(text):
            value = value[:-1] + "…" if value else "…"
        pdf.setFillColor(fill)
        pdf.setFont(font, size)
        if align == "right":
            pdf.drawRightString(x + width - row_pad, y, value)
        elif align == "center":
            pdf.drawCentredString(x + width / 2, y, value)
        else:
            pdf.drawString(x + row_pad, y, value)

    # Table header
    x = table_x
    pdf.setStrokeColor(border)
    pdf.setLineWidth(0.4)
    for label, width in table_columns:
        pdf.setFillColor(header_bg)
        pdf.rect(x, table_top - header_height, width, header_height, fill=1, stroke=0)
        draw_cell_text(label, x, table_top - 15.5, width, font="ATHSans-Bold", size=7.2, align="center", fill=header_fg)
        x += width

    # Color-coded candidate rows
    y = table_top - header_height
    for rank, item in enumerate(candidates, start=1):
        row_bottom = y - row_height
        row_bg = even_bg if rank % 2 == 0 else odd_bg
        pdf.setFillColor(row_bg)
        pdf.rect(table_x, row_bottom, table_width, row_height, fill=1, stroke=0)

        position = item.position.strip().lower()
        is_long = position.startswith("long")
        direction = "Long" if is_long else "Short"
        risk_dollars = item.shares * item.risk_per_share
        values = [
            str(rank),
            item.symbol,
            item.pattern,
            direction,
            f"${item.entry:.2f}",
            f"${item.stop:.2f}",
            f"${item.target:.2f}",
            f"{item.target_rr:.2f}:1",
            f"{item.score:.1f}",
            f"{item.shares:,}",
            f"${risk_dollars:,.0f}",
        ]

        x = table_x
        for (label, width), value in zip(table_columns, values):
            if label == "Dir":
                pdf.setFillColor(long_bg if is_long else short_bg)
                pdf.rect(x, row_bottom, width, row_height, fill=1, stroke=0)
                draw_cell_text(value, x, row_bottom + 8.0, width, font="ATHSans-Bold", size=6.8,
                               align="center", fill=header_fg)
            else:
                align = "right" if label in {"Entry", "Stop", "Target", "RR", "Score", "Shares", "Risk $"} else "center" if label == "#" else "left"
                font = "ATHSans-Bold" if label == "Symbol" else "ATHSans"
                size = 6.2 if label == "RR" else 6.8
                draw_cell_text(value, x, row_bottom + 8.0, width, font=font, size=size, align=align, fill=text_color)
            x += width

        pdf.setStrokeColor(border)
        pdf.line(table_x, row_bottom, table_x + table_width, row_bottom)
        y = row_bottom

    pdf.setStrokeColor(border)
    pdf.rect(table_x, table_top - header_height - row_height * len(candidates), table_width,
             header_height + row_height * len(candidates), fill=0, stroke=1)

    total_candidates = len(candidates)
    total_capital_at_risk = sum(item.shares * item.risk_per_share for item in candidates)
    average_rr = sum(item.target_rr for item in candidates) / total_candidates
    footer_top = y - 34
    pdf.setFillColor(colors.HexColor("#F5F5F5"))
    pdf.rect(table_x, footer_top - 94, table_width, 94, fill=1, stroke=0)
    pdf.setStrokeColor(border)
    pdf.rect(table_x, footer_top - 94, table_width, 94, fill=0, stroke=1)

    footer_items = [
        ("Total candidates", f"{total_candidates}"),
        ("Total capital at risk", f"${total_capital_at_risk:,.0f}"),
        ("Average R:R", f"{average_rr:.2f}:1"),
        ("Trade date", summary_date),
    ]
    footer_x = table_x + 18
    footer_y = footer_top - 22
    pdf.setFont("ATHSans", 8.3)
    for label, value in footer_items:
        pdf.setFillColor(muted_text)
        pdf.drawString(footer_x, footer_y, f"{label}:")
        pdf.setFillColor(text_color)
        pdf.setFont("ATHSans-Bold", 8.3)
        pdf.drawString(footer_x + 126, footer_y, value)
        pdf.setFont("ATHSans", 8.3)
        footer_y -= 16

    pdf.setFillColor(colors.HexColor("#A34A2C"))
    pdf.setFont("ATHSans-Bold", 8.2)
    pdf.drawCentredString(page_width / 2, footer_top - 86, "Scanner candidates only — not orders. Verify before trading.")
    pdf.setFillColor(colors.HexColor("#737C84"))
    pdf.setFont("ATHSans", 6.5)
    pdf.drawRightString(page_width - 42, 28, f"Page {len(candidates) + 1} of {len(candidates) + 1}")
    pdf.showPage()

    pdf.save()


def generate_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    dates = pd.bdate_range(end=date.today(), periods=85)
    records: list[dict[str, object]] = []

    def add_symbol(symbol: str, closes: np.ndarray, volumes: np.ndarray) -> None:
        previous = closes[0] * 1.002
        for idx, (day, close, volume) in enumerate(zip(dates, closes, volumes)):
            open_price = previous + rng.normal(0, 0.12)
            high = max(open_price, close) + abs(rng.normal(0.35, 0.09))
            low = min(open_price, close) - abs(rng.normal(0.35, 0.09))
            records.append({"date": day, "symbol": symbol, "description": f"{symbol} Demonstration Corp", "sector": "Demonstration", "sector_symbol": "$DEMO", "open": open_price, "high": high, "low": low, "close": close, "volume": int(volume)})
            previous = close

    # Fast Ball: persistent downtrend, short pullback/consolidation, then XRV break.
    fb = np.linspace(76, 55, 85) + rng.normal(0, 0.28, 85)
    fb[-10:-1] = np.array([56.2, 56.5, 56.8, 56.6, 56.9, 57.1, 57.0, 57.2, 56.9])
    fb[-1] = 54.15
    fb_vol = rng.integers(700_000, 1_050_000, 85).astype(float)
    fb_vol[-1] = 2_150_000
    add_symbol("FBDEMO", fb, fb_vol)
    fb_indices = [i for i, record in enumerate(records) if record["symbol"] == "FBDEMO"]
    final_fb = records[fb_indices[-1]]
    final_fb["open"] = 55.30
    final_fb["high"] = 55.75
    final_fb["low"] = 53.85
    final_fb["close"] = 54.15

    # Infield Fly: extended rise followed by a gap higher and weak close.
    inf = np.linspace(39, 58, 85) + rng.normal(0, 0.22, 85)
    inf[-1] = 57.85
    inf_vol = rng.integers(650_000, 1_300_000, 85).astype(float)
    add_symbol("IFDEMO", inf, inf_vol)
    # Replace final bar to guarantee a strict gap and lower-half bearish close.
    mask = pd.Series([record["symbol"] == "IFDEMO" for record in records])
    if_indices = np.flatnonzero(mask.to_numpy())
    previous_record = records[if_indices[-2]]
    final_record = records[if_indices[-1]]
    final_record["open"] = float(previous_record["high"]) + 1.05
    final_record["high"] = float(final_record["open"]) + 0.55
    final_record["low"] = float(previous_record["high"]) + 0.18
    final_record["close"] = float(final_record["low"]) + 0.08
    final_record["volume"] = 1_900_000

    # Switch Hitter: down wave to a two-week low, 3-day 50% retracement, bearish reversal.
    sh = np.linspace(69, 61, 85) + rng.normal(0, 0.18, 85)
    sh[-14:-3] = np.linspace(67.0, 59.0, 11)
    sh[-4] = 59.0
    sh[-3] = 60.25
    sh[-2] = 61.25
    sh[-1] = 60.45
    sh_vol = rng.integers(850_000, 1_250_000, 85).astype(float)
    sh_vol[-14:-3] = 1_500_000
    sh_vol[-3:] = 650_000
    add_symbol("SHDEMO", sh, sh_vol)
    sh_indices = [i for i, record in enumerate(records) if record["symbol"] == "SHDEMO"]
    final_sh = records[sh_indices[-1]]
    final_sh["open"] = 62.20
    final_sh["high"] = 62.50
    final_sh["low"] = 60.35
    final_sh["close"] = 60.42

    return pd.DataFrame.from_records(records)


def write_hermes_handoff(candidates: list[Candidate], output_path: Path, config: ScanConfig) -> None:
    """Write a broker-neutral, review-gated package for an execution agent."""
    payload = {
        "schema": "ath.hermes.candidates.v1",
        "scanner_version": "2.0",
        "generated_on": date.today().isoformat(),
        "order_transmission_enabled": False,
        "requires_human_approval": True,
        "risk_dollars_per_trade": config.risk_dollars,
        "minimum_reward_to_risk": config.minimum_reward_to_risk,
        "candidate_count": len(candidates),
        "instructions": (
            "Candidates are plans, not orders. Revalidate borrow, spread, news, entry trigger, stop resistance, "
            "target support, and position size before any execution."
        ),
        "candidates": [asdict(item) for item in candidates],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(input_path: Path | None, output_dir: Path, config: ScanConfig, demo: bool = False,
        top_n: int | None = None, sort_by: str = "score") -> list[Candidate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if demo:
        data = generate_demo_data()
        data.to_csv(output_dir / "demo_ohlcv.csv", index=False)
    elif input_path:
        data = pd.read_csv(input_path)
    else:
        raise ValueError("Provide --input or use --demo")

    candidates, frames = scan_latest(data, config)

    # Rank and trim to top N
    if sort_by == "rr":
        candidates.sort(key=lambda item: (-item.target_rr, -item.score, item.symbol))
    else:  # "score" — composite regime score (pattern strength + volume + ADX + close)
        candidates.sort(key=lambda item: (-item.score, -item.target_rr, item.symbol))

    if top_n is not None and top_n > 0:
        # Collapse same-symbol candidates BEFORE trimming so top_n means
        # top_n DISTINCT tradable symbols. One symbol can qualify under two
        # patterns on the same bar (e.g. WBD as "Fast Ball Long" and
        # "Line Drive Gap Long"); without this, a single name eats two slots
        # and the deck silently loses a distinct candidate.
        #
        # The stager keys position state and its deterministic OCA group by
        # SYMBOL, so two rows for one symbol would also open a DOUBLE position
        # under one OCA group. Keep the first (highest-ranked) occurrence.
        deduped = []
        seen_symbols = set()
        for item in candidates:
            if item.symbol in seen_symbols:
                continue
            seen_symbols.add(item.symbol)
            deduped.append(item)
        if len(deduped) < len(candidates):
            print(f"Collapsed {len(candidates) - len(deduped)} same-symbol duplicate(s)")
        candidates = deduped

        trimmed = len(candidates) > top_n
        candidates = candidates[:top_n]
        if trimmed:
            print(f"Trimmed to top {top_n} by {sort_by}")

    pd.DataFrame([asdict(item) for item in candidates]).to_csv(output_dir / "candidates.csv", index=False)
    write_hermes_handoff(candidates, output_dir / "hermes_candidates.json", config)
    manifest = {
        "scanner_version": "2.0",
        "candidate_count": len(candidates),
        "input": "generated demo data" if demo else str(input_path),
        "outputs": ["candidates.csv", "hermes_candidates.json"] + (["trade_plan.pdf"] if candidates else []),
        "minimum_reward_to_risk": config.minimum_reward_to_risk,
        "target_policy": "support/resistance and Fibonacci confluence; no fixed R target",
        "top_n": top_n,
        "sort_by": sort_by,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if candidates:
        generate_trade_plan_pdf(candidates, frames, output_dir / "trade_plan.pdf", config)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan daily OHLCV data for ATH short setups")
    parser.add_argument("--input", type=Path, help="CSV containing date,symbol,open,high,low,close,volume")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--risk", type=float, default=1000.0, help="Dollar risk per trade")
    parser.add_argument("--min-rr", type=float, default=1.0, help="Minimum technical-target reward-to-risk")
    parser.add_argument("--top-n", type=int, default=10, help="Keep only top N candidates (0 = all)")
    parser.add_argument("--sort-by", choices=["score", "rr"], default="score",
                        help="Rank candidates by 'score' (composite regime) or 'rr' (reward-to-risk)")
    parser.add_argument("--demo", action="store_true", help="Generate demo data and a sample plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScanConfig(
        risk_dollars=args.risk,
        minimum_reward_to_risk=args.min_rr,
    )
    candidates = run(args.input, args.output_dir, config, args.demo,
                     top_n=args.top_n, sort_by=args.sort_by)
    print(f"Found {len(candidates)} candidate(s)")
    for item in candidates:
        print(
            f"{item.symbol:10} {item.pattern:24} score={item.score:5.1f} entry={item.entry:.2f} "
            f"stop={item.stop:.2f} target={item.target:.2f} rr={item.target_rr:.2f}"
        )


if __name__ == "__main__":
    main()
