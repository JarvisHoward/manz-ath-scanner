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
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

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


def build_candidate(
    frame: pd.DataFrame,
    index: int,
    pattern: str,
    score: float,
    reasons: Iterable[str],
    config: ScanConfig,
    target_levels: Iterable[TargetLevel] = (),
    warnings: Iterable[str] = (),
) -> Candidate | None:
    row = frame.iloc[index]
    entry = round(float(row["low"]) - config.entry_buffer, 2)
    # This is intentionally provisional. Manz often uses chart resistance rather
    # than a universal formula for the protective stop.
    stop = round(float(row["high"]) + config.stop_buffer, 2)
    risk_per_share = round(stop - entry, 2)
    pivots = floor_pivots(float(row["high"]), float(row["low"]), float(row["close"]))
    selected = select_short_target(frame, index, entry, stop, pivots, target_levels, config)
    if selected is None:
        return None
    target, target_rr, target_method, target_evidence, target_warnings = selected
    halfway = round(entry - 0.5 * (entry - target), 2)
    shares = math.floor(config.risk_dollars / risk_per_share) if risk_per_share > 0 else 0
    return Candidate(
        date=pd.Timestamp(row["date"]).date().isoformat(),
        symbol=str(row["symbol"]),
        description=str(row.get("description", "")) if pd.notna(row.get("description", "")) else "",
        sector=str(row.get("sector", "")) if pd.notna(row.get("sector", "")) else "",
        sector_symbol=str(row.get("sector_symbol", "")) if pd.notna(row.get("sector_symbol", "")) else "",
        pattern=pattern,
        position="Short",
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
    score = min(100.0, 65 + min(15, max(0, (volume_ratio - 1) * 15)) + min(10, row["adx14"] / 5) + (10 if pullback else 7))
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


def scan_latest(data: pd.DataFrame, config: ScanConfig) -> tuple[list[Candidate], dict[str, pd.DataFrame]]:
    candidates: list[Candidate] = []
    prepared: dict[str, pd.DataFrame] = {}
    for symbol, raw_frame in data.groupby("symbol", sort=True):
        frame = prepare_frame(raw_frame)
        prepared[str(symbol)] = frame
        index = len(frame) - 1
        for detector in [detect_fast_ball_short, detect_infield_fly_short, detect_switch_hitter_short]:
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


def run(input_path: Path | None, output_dir: Path, config: ScanConfig, demo: bool = False) -> list[Candidate]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if demo:
        data = generate_demo_data()
        data.to_csv(output_dir / "demo_ohlcv.csv", index=False)
    elif input_path:
        data = pd.read_csv(input_path)
    else:
        raise ValueError("Provide --input or use --demo")

    candidates, frames = scan_latest(data, config)
    pd.DataFrame([asdict(item) for item in candidates]).to_csv(output_dir / "candidates.csv", index=False)
    write_hermes_handoff(candidates, output_dir / "hermes_candidates.json", config)
    manifest = {
        "scanner_version": "2.0",
        "candidate_count": len(candidates),
        "input": "generated demo data" if demo else str(input_path),
        "outputs": ["candidates.csv", "hermes_candidates.json"] + (["trade_plan.pdf"] if candidates else []),
        "minimum_reward_to_risk": config.minimum_reward_to_risk,
        "target_policy": "support/resistance and Fibonacci confluence; no fixed R target",
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
    parser.add_argument("--demo", action="store_true", help="Generate demo data and a sample plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScanConfig(
        risk_dollars=args.risk,
        minimum_reward_to_risk=args.min_rr,
    )
    candidates = run(args.input, args.output_dir, config, args.demo)
    print(f"Found {len(candidates)} candidate(s)")
    for item in candidates:
        print(
            f"{item.symbol:10} {item.pattern:24} score={item.score:5.1f} entry={item.entry:.2f} "
            f"stop={item.stop:.2f} target={item.target:.2f} rr={item.target_rr:.2f}"
        )


if __name__ == "__main__":
    main()
