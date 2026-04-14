"""auto_tuner.py — Self-tuning layer for CPR Gold Bot v5.3

Runs after every trade closes AND daily at 08:00 SGT.
Analyses the last N closed trades, detects bad patterns, adjusts
settings.json within safe bounds, and sends a Telegram summary.

Pattern detectors
─────────────────
1. DIRECTION_BIAS     — ≥4 consecutive losses in the same direction
                        → raise sl_direction_cooldown_min, raise consecutive_sl_guard
2. COUNTER_TREND      — ≥3 losses where direction contradicts H4 trend stored in record
                        → this is now blocked by H4 filter; logged for visibility only
3. LOW_WIN_RATE       — rolling 20-trade win rate < 35%
                        → raise signal_threshold by 1 (max 5)
4. HIGH_WIN_RATE      — rolling 20-trade win rate ≥ 60%
                        → lower signal_threshold by 1 (min 4) to get more trades
5. TIGHT_RR           — avg realised RR over last 20 trades < 1.5
                        → raise rr_ratio by 0.25 (max 3.0)
6. OVERSIZED_SL       — avg SL cost > $20 (sl_max_usd breach pattern)
                        → lower atr_sl_multiplier by 0.1 (min 0.8)
7. LOSS_STREAK        — ≥3 consecutive losses any direction
                        → raise loss_streak_cooldown_min

Bounds (hard limits — tuner will never go outside these)
─────────────────────────────────────────────────────────
signal_threshold          : 4 – 5
rr_ratio                  : 1.8 – 3.0
atr_sl_multiplier         : 0.8 – 1.5
loss_streak_cooldown_min  : 30 – 180
sl_direction_cooldown_min : 60 – 360
consecutive_sl_guard      : 2 – 5
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytz

from config_loader import DATA_DIR, load_settings
from telegram_alert import TelegramAlert

log = logging.getLogger(__name__)
SGT = pytz.timezone("Asia/Singapore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SETTINGS_PATH = Path(__file__).parent / "settings.json"
SETTINGS_BACKUP_DIR = Path(DATA_DIR) / "settings_backups"

# ── Bounds ─────────────────────────────────────────────────────────────────────
BOUNDS: dict[str, tuple[float, float]] = {
    "signal_threshold":          (4,   5),
    "rr_ratio":                  (1.8, 3.0),
    "atr_sl_multiplier":         (0.8, 1.5),
    "loss_streak_cooldown_min":  (30,  180),
    "sl_direction_cooldown_min": (60,  360),
    "consecutive_sl_guard":      (2,   5),
}

# Minimum trades required before tuner acts
MIN_TRADES_TO_TUNE = 10
# How many recent trades to analyse
ANALYSIS_WINDOW = 20


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clamp(key: str, value: float) -> float:
    lo, hi = BOUNDS[key]
    return max(lo, min(hi, value))


def _load_history() -> list[dict]:
    """Load history.json from DATA_DIR. Returns [] on any error."""
    path = Path(DATA_DIR) / "history.json"
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _closed_trades(history: list[dict]) -> list[dict]:
    """Return trades that have a realized PnL (i.e. fully closed)."""
    return [
        t for t in history
        if t.get("realized_pnl_usd") is not None
    ]


def _backup_settings(settings: dict) -> None:
    """Save a timestamped backup of current settings before any change."""
    SETTINGS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(SGT).strftime("%Y%m%d_%H%M%S")
    dst = SETTINGS_BACKUP_DIR / f"settings_{ts}.json"
    try:
        with open(dst, "w") as f:
            json.dump(settings, f, indent=2)
        log.info("Auto-tuner: settings backed up to %s", dst)
    except Exception as exc:
        log.warning("Auto-tuner: could not back up settings: %s", exc)


def _save_settings(settings: dict) -> None:
    """Atomic write of settings.json."""
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(settings, f, indent=2)
    tmp.replace(SETTINGS_PATH)
    log.info("Auto-tuner: settings.json updated.")


# ── Core analysis ──────────────────────────────────────────────────────────────

def _count_consecutive_losses(closed: list[dict]) -> int:
    """Count consecutive losses from the most recent trade backwards."""
    count = 0
    for t in reversed(closed):
        if (t.get("realized_pnl_usd") or 0) < 0:
            count += 1
        else:
            break
    return count


def _consecutive_losses_same_direction(closed: list[dict]) -> tuple[int, str]:
    """Return (count, direction) for the current consecutive-loss streak in one direction."""
    if not closed:
        return 0, ""
    last_dir = closed[-1].get("direction", "")
    count = 0
    for t in reversed(closed):
        if t.get("direction") == last_dir and (t.get("realized_pnl_usd") or 0) < 0:
            count += 1
        else:
            break
    return count, last_dir


def _rolling_win_rate(closed: list[dict], window: int = ANALYSIS_WINDOW) -> float | None:
    """Win rate over the last `window` closed trades. None if not enough data."""
    sample = closed[-window:]
    if len(sample) < MIN_TRADES_TO_TUNE:
        return None
    wins = sum(1 for t in sample if (t.get("realized_pnl_usd") or 0) > 0)
    return wins / len(sample)


def _avg_realised_rr(closed: list[dict], window: int = ANALYSIS_WINDOW) -> float | None:
    """Average realised R:R over last window trades. None if not enough data."""
    sample = closed[-window:]
    if len(sample) < MIN_TRADES_TO_TUNE:
        return None
    rrs = []
    for t in sample:
        pnl = t.get("realized_pnl_usd") or 0
        sl_usd = t.get("sl_usd") or 0
        if sl_usd > 0:
            rrs.append(pnl / sl_usd)
    return sum(rrs) / len(rrs) if rrs else None


def _avg_sl_cost(closed: list[dict], window: int = ANALYSIS_WINDOW) -> float | None:
    sample = [t for t in closed[-window:] if (t.get("realized_pnl_usd") or 0) < 0]
    if not sample:
        return None
    costs = [abs(t.get("realized_pnl_usd") or 0) for t in sample]
    return sum(costs) / len(costs)


# ── Pattern detection & adjustment ────────────────────────────────────────────

def _analyse_and_tune(settings: dict, closed: list[dict]) -> tuple[dict, list[str]]:
    """
    Analyse closed trades and return (updated_settings, list_of_change_messages).
    settings is modified in-place and also returned.
    """
    changes: list[str] = []
    patterns: list[str] = []

    if len(closed) < MIN_TRADES_TO_TUNE:
        log.info("Auto-tuner: only %d closed trades — skipping (need %d)", len(closed), MIN_TRADES_TO_TUNE)
        return settings, changes

    # ── Pattern 1: Direction bias (consecutive losses same direction) ──────────
    consec_dir, losing_dir = _consecutive_losses_same_direction(closed)
    if consec_dir >= 4:
        patterns.append(f"DIRECTION_BIAS ({consec_dir}× {losing_dir} losses)")

        old_cd = float(settings.get("sl_direction_cooldown_min", 60))
        new_cd = _clamp("sl_direction_cooldown_min", old_cd + 60)
        if new_cd != old_cd:
            settings["sl_direction_cooldown_min"] = int(new_cd)
            changes.append(f"sl_direction_cooldown_min: {int(old_cd)} → {int(new_cd)} min")

        old_guard = int(settings.get("consecutive_sl_guard", 2))
        new_guard = _clamp("consecutive_sl_guard", old_guard + 1)
        if new_guard != old_guard:
            settings["consecutive_sl_guard"] = int(new_guard)
            changes.append(f"consecutive_sl_guard: {old_guard} → {int(new_guard)}")

    # ── Pattern 2: Counter-trend losses (logged only — H4 filter should block) ─
    counter_trend_losses = sum(
        1 for t in closed[-10:]
        if (t.get("realized_pnl_usd") or 0) < 0
        and t.get("levels", {}).get("h4_trend_bullish") is not None
        and (
            (t.get("direction") == "SELL" and t["levels"]["h4_trend_bullish"])
            or (t.get("direction") == "BUY" and not t["levels"]["h4_trend_bullish"])
        )
    )
    if counter_trend_losses >= 3:
        patterns.append(f"COUNTER_TREND ({counter_trend_losses} losses vs H4 — H4 filter may need review)")

    # ── Pattern 3: Low rolling win rate ────────────────────────────────────────
    wr = _rolling_win_rate(closed)
    if wr is not None:
        if wr < 0.35:
            patterns.append(f"LOW_WIN_RATE ({wr*100:.0f}% over last {ANALYSIS_WINDOW})")
            old_thresh = int(settings.get("signal_threshold", 4))
            new_thresh = _clamp("signal_threshold", old_thresh + 1)
            if new_thresh != old_thresh:
                settings["signal_threshold"] = int(new_thresh)
                changes.append(f"signal_threshold: {old_thresh} → {int(new_thresh)}")

        elif wr >= 0.60:
            patterns.append(f"HIGH_WIN_RATE ({wr*100:.0f}% over last {ANALYSIS_WINDOW} — relaxing threshold)")
            old_thresh = int(settings.get("signal_threshold", 4))
            new_thresh = _clamp("signal_threshold", old_thresh - 1)
            if new_thresh != old_thresh:
                settings["signal_threshold"] = int(new_thresh)
                changes.append(f"signal_threshold: {old_thresh} → {int(new_thresh)}")

    # ── Pattern 4: Tight R:R (avg realised RR too low) ─────────────────────────
    avg_rr = _avg_realised_rr(closed)
    if avg_rr is not None and avg_rr < 1.5:
        patterns.append(f"TIGHT_RR (avg realised RR={avg_rr:.2f})")
        old_rr = float(settings.get("rr_ratio", 2.0))
        new_rr = _clamp("rr_ratio", round(old_rr + 0.25, 2))
        if new_rr != old_rr:
            settings["rr_ratio"] = new_rr
            changes.append(f"rr_ratio: {old_rr} → {new_rr}")

    # ── Pattern 5: Oversized SL cost ───────────────────────────────────────────
    avg_sl = _avg_sl_cost(closed)
    if avg_sl is not None and avg_sl > 20:
        patterns.append(f"OVERSIZED_SL (avg loss cost ${avg_sl:.2f})")
        old_mult = float(settings.get("atr_sl_multiplier", 1.0))
        new_mult = _clamp("atr_sl_multiplier", round(old_mult - 0.1, 2))
        if new_mult != old_mult:
            settings["atr_sl_multiplier"] = new_mult
            changes.append(f"atr_sl_multiplier: {old_mult} → {new_mult}")

    # ── Pattern 6: Loss streak (any direction) ─────────────────────────────────
    consec_any = _count_consecutive_losses(closed)
    if consec_any >= 3:
        patterns.append(f"LOSS_STREAK ({consec_any} consecutive losses)")
        old_cd = float(settings.get("loss_streak_cooldown_min", 30))
        new_cd = _clamp("loss_streak_cooldown_min", old_cd + 30)
        if new_cd != old_cd:
            settings["loss_streak_cooldown_min"] = int(new_cd)
            changes.append(f"loss_streak_cooldown_min: {int(old_cd)} → {int(new_cd)} min")

    if patterns:
        log.info("Auto-tuner patterns detected: %s", ", ".join(patterns))
    else:
        log.info("Auto-tuner: no bad patterns detected in last %d trades.", len(closed[-ANALYSIS_WINDOW:]))

    return settings, changes


# ── Telegram message ───────────────────────────────────────────────────────────

def _build_telegram_message(
    changes: list[str],
    closed: list[dict],
    wr: float | None,
    avg_rr: float | None,
) -> str:
    now_str = datetime.now(SGT).strftime("%Y-%m-%d %H:%M SGT")
    lines = [f"🤖 *Auto-Tuner Report* — {now_str}"]

    n = min(len(closed), ANALYSIS_WINDOW)
    wins = sum(1 for t in closed[-n:] if (t.get("realized_pnl_usd") or 0) > 0)
    losses = n - wins
    lines.append(f"📊 Last {n} trades: {wins}W / {losses}L"
                 + (f" | Win rate: {wr*100:.0f}%" if wr is not None else ""))
    if avg_rr is not None:
        lines.append(f"📐 Avg realised R:R: {avg_rr:.2f}")

    if changes:
        lines.append("")
        lines.append("⚙️ *Settings adjusted:*")
        for c in changes:
            lines.append(f"  • {c}")
    else:
        lines.append("✅ No changes needed — settings look healthy.")

    return "\n".join(lines)


# ── Public entry points ────────────────────────────────────────────────────────

def run_auto_tune(trigger: str = "manual") -> None:
    """
    Main entry point. Call this:
      - after every trade closes (trigger="trade_close")
      - from the daily scheduler job (trigger="daily")
      - manually (trigger="manual")

    Loads history, analyses patterns, adjusts settings if needed,
    sends Telegram alert if anything changed.
    """
    log.info("Auto-tuner triggered: %s", trigger)

    history = _load_history()
    closed  = _closed_trades(history)

    if len(closed) < MIN_TRADES_TO_TUNE:
        log.info(
            "Auto-tuner: %d closed trades, need %d — skipping.",
            len(closed), MIN_TRADES_TO_TUNE
        )
        return

    settings = load_settings()
    _backup_settings(settings)

    settings, changes = _analyse_and_tune(settings, closed)

    wr     = _rolling_win_rate(closed)
    avg_rr = _avg_realised_rr(closed)

    # Only save & alert if something actually changed
    if changes:
        _save_settings(settings)
        log.info("Auto-tuner applied %d change(s): %s", len(changes), "; ".join(changes))
        try:
            msg = _build_telegram_message(changes, closed, wr, avg_rr)
            TelegramAlert().send(msg)
        except Exception as exc:
            log.warning("Auto-tuner: could not send Telegram alert: %s", exc)
    else:
        # Send a brief daily health check even when no changes
        if trigger == "daily":
            try:
                msg = _build_telegram_message([], closed, wr, avg_rr)
                TelegramAlert().send(msg)
            except Exception as exc:
                log.warning("Auto-tuner daily report: could not send Telegram: %s", exc)


def run_auto_tune_after_trade_close() -> None:
    """Thin wrapper called from bot.py after each trade closes."""
    run_auto_tune(trigger="trade_close")


def run_auto_tune_daily() -> None:
    """Thin wrapper called from scheduler.py at 08:00 SGT."""
    run_auto_tune(trigger="daily")
