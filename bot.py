"""Main orchestrator for the CPR Gold Bot — v5.1

Runs the N-minute trading cycle for XAU/USD (default 5 min), applies
session and risk controls, places orders through OANDA, manages break-even,
and persists runtime state.

Position sizing (v4.2) — values are read from settings.json:
  score 5–6  →  position_full_usd    (default $100)
  score 4    →  position_partial_usd (default $66)
  score < 4  →  no trade (below signal_threshold=4)

Active trading windows (SGT) — v4.2:
  London:    16:00–20:59  (London full session, 08:00–13:00 GMT)
  US:        21:00–00:59  (London/NY overlap + NY morning, 13:00–17:00 EDT)
  Dead Zone: 01:00–15:59  (no new entries — existing trades managed only)
  Asian:     08:00–15:59 SGT — enabled v4.4 for 2-week evaluation

v4.7 (current):
  - ATR-based SL replaces fixed 0.25% percentage SL
  - TP always derived from SL via rr_ratio (no more fallback_tp_multiplier hack)
  - S2/R2 Extended setups hard-blocked when exhaustion stretch > threshold
  - Breakeven re-enabled at $15 trigger (was disabled in v4.2)
  - Same-setup cooldown microsecond precision bug fixed
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytz

from calendar_fetcher import run_fetch as refresh_calendar
from config_loader import DATA_DIR, get_bool_env, load_settings
from database import Database
from logging_utils import configure_logging, get_logger
from news_filter import NewsFilter
from oanda_trader import OandaTrader
from signals import SignalEngine, score_to_position_usd
from startup_checks import run_startup_checks
from state_utils import (
    RUNTIME_STATE_FILE, SCORE_CACHE_FILE, OPS_STATE_FILE, TRADE_HISTORY_FILE,
    update_runtime_state, load_json, save_json, parse_sgt_timestamp,
)
from telegram_alert import TelegramAlert
from telegram_templates import (
    msg_signal_update, msg_trade_opened, msg_breakeven, msg_trade_closed,
    msg_news_block, msg_news_penalty, msg_cooldown_started, msg_daily_cap,
    msg_session_cap,
    msg_spread_skip, msg_order_failed, msg_error, msg_friday_cutoff,
    msg_margin_adjustment, msg_pyramid_opened,
)
from reconcile_state import reconcile_runtime_state, startup_oanda_reconcile
from ai_reasoning import ai_should_trade

configure_logging()
log = get_logger(__name__)

SGT          = pytz.timezone("Asia/Singapore")
INSTRUMENT   = "XAU_USD"  # v4.2: overridden at runtime by settings["instrument"]

# v4.2 — startup reconcile runs exactly once per process (not every 5-min cycle)
_startup_reconcile_done: bool = False
ASSET        = "XAUUSD"  # v4.2: overridden at runtime by settings["instrument_display"]
HISTORY_FILE = TRADE_HISTORY_FILE
HISTORY_DAYS = 90
# Removed: ARCHIVE_FILE — archival removed; 90-day rolling window stored in trade_history.json

# v4.2 — 9-hour trading window: London open (16:00 SGT) → NY morning close (00:59 SGT)
# v4.4 — All 3 sessions enabled: Asian (08–15), London (16–20), US (21–00).
# 2-week evaluation to determine Asian session viability for CPR breakouts.
# Each tuple: (window_name, macro_session, start_hour, end_hour, fallback_threshold)
SESSIONS = [
    ("Asian Window",  "Asian",   8, 15, 3),   # 08:00–15:59 SGT (00:00–07:59 GMT)
    ("London Window", "London", 16, 20, 3),   # 16:00–20:59 SGT (08:00–13:00 GMT)
    ("US Window",     "US",     21, 23, 3),   # 21:00–23:59 SGT (13:00–16:00 EDT)
    ("US Window",     "US",      0,  0, 3),   # 00:00–00:59 SGT (16:00–17:00 EDT)
]

# v4.4 — Three sessions: Asian, London, US
SESSION_BANNERS = {
    "Asian":  "🌏 ASIAN",
    "London": "🇬🇧 LONDON",
    "US":     "🗽 US",
}


def _clean_reason(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "No reason available"
    for part in reversed([p.strip() for p in text.split("|") if p.strip()]):
        plain = re.sub(r"^[^A-Za-z0-9]+", "", part).strip()
        if plain:
            return plain[:120]
    return text[:120]


def _build_signal_checks(score: int, direction: str, rr_ratio: float | None = None, tp_pct: float | None = None, settings: dict | None = None,
                         spread_pips: int | None = None, spread_limit: int | None = None, session_ok: bool = True,
                         news_ok: bool = True, open_trade_ok: bool = True, margin_ok: bool | None = None,
                         cooldown_ok: bool = True):
    mandatory_checks = [
        (f"Score >= {(settings or {}).get('signal_threshold', 4)}", score >= int((settings or {}).get('signal_threshold', 4)) and direction != "NONE", f"{score}/6"),
        (f"RR >= {settings.get('rr_ratio', 2.65):.2f}", None if rr_ratio is None else rr_ratio >= float(settings.get('rr_ratio', 2.65)), "n/a" if rr_ratio is None else f"{rr_ratio:.2f}"),
    ]
    quality_checks = [
        ("TP >= 0.5%", None if tp_pct is None else tp_pct >= 0.5, "n/a" if tp_pct is None else f"{tp_pct:.2f}%"),
    ]
    execution_checks = [
        ("Session active", session_ok, "active" if session_ok else "inactive"),
        ("News clear", news_ok, "clear" if news_ok else "blocked"),
        ("Cooldown clear", cooldown_ok, "clear" if cooldown_ok else "active"),
        ("No open trade", open_trade_ok, "ready" if open_trade_ok else "existing position"),
        ("Spread OK", None if spread_pips is None or spread_limit is None else spread_pips <= spread_limit, "n/a" if spread_pips is None or spread_limit is None else f"{spread_pips}/{spread_limit} pips"),
        ("Margin OK", margin_ok, "n/a" if margin_ok is None else ("pass" if margin_ok else "insufficient")),
    ]
    return mandatory_checks, quality_checks, execution_checks




def _signal_payload(settings: dict | None = None, **kwargs):
    mandatory_checks, quality_checks, execution_checks = _build_signal_checks(**kwargs, settings=settings)
    return {
        "mandatory_checks": mandatory_checks,
        "quality_checks": quality_checks,
        "execution_checks": execution_checks,
    }
# ── Settings ───────────────────────────────────────────────────────────────────

def validate_settings(settings: dict) -> dict:
    # ── Inject defaults for ALL keys before any validation ────────────────────
    # Required keys get safe production defaults first so stale Railway
    # persistent-volume settings.json files from older versions never crash
    # the bot on startup. The old hard-fail on missing required keys is replaced
    # by setdefault — a missing key is treated the same as any other default.
    # v4.4 — Three sessions active
    settings.setdefault("spread_limits",             {"Asian": 150, "London": 140, "US": 140})
    settings.setdefault("max_trades_day",            999)   # v4.0-uncapped
    settings.setdefault("max_losing_trades_day",     999)   # v4.0-uncapped
    settings.setdefault("sl_mode",                   "atr_based")   # v4.0
    settings.setdefault("tp_mode",                   "rr_multiple")
    settings.setdefault("rr_ratio",                  2.65)          # v4.2 — read from settings.json
    settings.setdefault("max_rr_ratio",              3.0)           # v4.6 — hard TP ceiling as multiple of SL
    settings.setdefault("sl_min_atr_mult",           0.8)           # v5.1 — adaptive SL floor as fraction of ATR
    settings.setdefault("h1_trend_filter_enabled",   True)          # v5.1 — H1 EMA trend filter
    settings.setdefault("h1_ema_period",             21)            # v5.1 — H1 EMA period for trend
    settings.setdefault("require_candle_close",      True)          # v5.1 — wait for M15 candle close
    settings.setdefault("sl_direction_cooldown_min", 60)            # v5.1 — cooldown after direction guard fires
    settings.setdefault("post_win_candle_block",     True)          # v5.2 — block entries until M15 candle after TP close + next candle confirmed
    settings.setdefault("ai_reasoning",              True)          # v5.2 — Claude AI filter before order placement
    settings.setdefault("signal_threshold",          4)
    settings.setdefault("position_full_usd",         100)
    settings.setdefault("position_partial_usd",      66)
    settings.setdefault("account_balance_override",  0)
    settings.setdefault("enabled",                   True)
    settings.setdefault("atr_sl_multiplier",         1.0)           # v4.0 — raised from 0.5
    settings.setdefault("sl_min_usd",                15.0)          # v5.2 — SL floor = 1500 pips ($15)
    settings.setdefault("sl_max_usd",                17.0)          # v5.2 — SL ceiling = 1700 pips ($17)
    settings.setdefault("fixed_sl_usd",              20.0)          # v4.0 — raised from 5.0
    settings.setdefault("breakeven_trigger_usd",     15.0)          # v4.0 — raised from 3.0
    settings.setdefault("sl_pct",                   0.0025)
    settings.setdefault("tp_pct",                   0.0075)
    settings.setdefault("margin_safety_factor",      0.6)
    settings.setdefault("margin_retry_safety_factor", 0.4)
    settings.setdefault("xau_margin_rate_override",   0.05)
    settings.setdefault("auto_scale_on_margin_reject", True)
    settings.setdefault("telegram_show_margin",      True)
    settings.setdefault("friday_cutoff_hour_sgt",    23)
    settings.setdefault("friday_cutoff_minute_sgt",  0)     # Friday cutoff kept at 23:00 SGT
    settings.setdefault("news_lookahead_min",         120)
    settings.setdefault("news_medium_penalty_score",  -1)
    settings.setdefault("fixed_tp_usd",              None)
    settings.setdefault("loss_streak_cooldown_min",   0)    # v4.0-uncapped — disabled
    settings.setdefault("max_concurrent_trades",      1)
    # v4.0-uncapped — per-session caps removed
    settings.setdefault("max_trades_london",          999)
    settings.setdefault("max_trades_us",              999)
    settings.setdefault("max_spread_pips",            150)
    settings.setdefault("session_only",               True)
    settings.setdefault("session_thresholds",         {"Asian": 4, "London": 4, "US": 4})
    settings.setdefault("news_filter_enabled",        True)
    settings.setdefault("news_block_before_min",      30)
    settings.setdefault("news_block_after_min",       30)
    # ── Pyramid (Trade 2 add-on) settings ─────────────────────────────────────
    settings.setdefault("pyramid_enabled",            False)
    settings.setdefault("pyramid_min_score",          5)
    settings.setdefault("pyramid_sl_usd",             1.50)
    settings.setdefault("pyramid_max_risk_usd",       50)
    # v4.2 — trading day boundary and per-session loss sub-cap
    settings.setdefault("trading_day_start_hour_sgt", 8)
    settings.setdefault("max_losing_trades_session",  999)  # v4.0-uncapped
    settings.setdefault("midnight_guard_min",         0)
    # v4.2 — trading window explicit boundary
    settings.setdefault("session_start_hour_sgt",     16)
    settings.setdefault("session_end_hour_sgt",        1)
    # v4.2 — same-setup re-entry cooldown (microsecond bug fixed in v4.0)
    settings.setdefault("same_setup_cooldown_min",     15)
    # NOTE: fallback_tp_multiplier removed in v4.0 — ATR-based SL makes it redundant

    cooldown_min = int(settings.get("loss_streak_cooldown_min", 30))
    if cooldown_min < 0:
        raise ValueError("loss_streak_cooldown_min must be >= 0 (set to 0 to disable)")

    return settings


def is_friday_cutoff(now_sgt: datetime, settings: dict) -> bool:
    if now_sgt.weekday() != 4:
        return False
    cutoff_hour   = int(settings.get("friday_cutoff_hour_sgt", 23))
    cutoff_minute = int(settings.get("friday_cutoff_minute_sgt", 0))
    return now_sgt.hour > cutoff_hour or (
        now_sgt.hour == cutoff_hour and now_sgt.minute >= cutoff_minute
    )


# ── Trade history helpers ──────────────────────────────────────────────────────

def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history: list):
    atomic_json_write(HISTORY_FILE, history)


def atomic_json_write(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def prune_old_trades(history: list) -> list:
    """Drop trades older than HISTORY_DAYS from the active history.

    No archive file is written. The 90-day rolling window in
    trade_history.json is sufficient for all daily/weekly/monthly reports.
    Trades simply expire after 90 days.
    """
    cutoff = datetime.now(SGT) - timedelta(days=HISTORY_DAYS)
    active = []
    pruned = 0
    for trade in history:
        ts = trade.get("timestamp_sgt", "")
        try:
            dt = SGT.localize(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            if dt < cutoff:
                pruned += 1
            else:
                active.append(trade)
        except Exception:
            active.append(trade)
    if pruned:
        log.info("Pruned %d trade(s) older than %d days | Active: %d", pruned, HISTORY_DAYS, len(active))
    return active


# ── Session helpers ────────────────────────────────────────────────────────────

def get_session(now: datetime, settings: dict = None):
    h = now.hour
    session_thresholds = (settings or {}).get("session_thresholds", {})
    # v4.4 — per-session enable/disable flags (Asian added)
    _enabled = {
        "Asian":  bool((settings or {}).get("asian_session_enabled",  True)),
        "London": bool((settings or {}).get("london_session_enabled", True)),
        "US":     bool((settings or {}).get("us_session_enabled",     True)),
    }
    for name, macro, start, end, fallback_thr in SESSIONS:
        if start <= h <= end:
            if not _enabled.get(macro, True):
                return None, None, None  # session disabled
            thr = int(session_thresholds.get(macro, fallback_thr))
            return name, macro, thr
    return None, None, None


def is_dead_zone_time(now_sgt: datetime, settings: dict | None = None) -> bool:
    # v4.4 — dead zone is any hour not covered by an enabled session in SESSIONS tuple.
    h = now_sgt.hour
    _enabled = {
        "Asian":  bool((settings or {}).get("asian_session_enabled",  True)),
        "London": bool((settings or {}).get("london_session_enabled", True)),
        "US":     bool((settings or {}).get("us_session_enabled",     True)),
    }
    for _, macro, start, end, _ in SESSIONS:
        if _enabled.get(macro, True) and start <= h <= end:
            return False
    return True


def get_window_key(session_name: str | None) -> str | None:
    # v4.4 — window keys map to macro session names
    if session_name == "Asian Window":
        return "Asian"
    if session_name == "London Window":
        return "London"
    if session_name == "US Window":
        return "US"
    return None


def get_window_trade_cap(window_key: str | None, settings: dict) -> int | None:
    # v4.4 — separate caps per session including Asian
    if window_key == "Asian":
        return int(settings.get("max_trades_asian", 5))
    if window_key == "London":
        return int(settings.get("max_trades_london", 10))
    if window_key == "US":
        return int(settings.get("max_trades_us", 10))
    return None


def window_trade_count(history: list, today_str: str, window_key: str) -> int:
    # v4.4 — Three sessions tracked independently
    aliases = {
        "Asian":  {"Asian", "Asian Window"},
        "London": {"London", "London Window"},
        "US":     {"US", "US Window"},
    }
    valid = aliases.get(window_key, {window_key})
    count = 0
    for t in history:
        if not t.get("timestamp_sgt", "").startswith(today_str):
            continue
        if t.get("status") != "FILLED":
            continue
        trade_window = t.get("window") or t.get("session") or t.get("macro_session")
        if trade_window in valid:
            count += 1
    return count


# ── Risk / daily cap helpers ───────────────────────────────────────────────────

def daily_totals(history: list, today_str: str, trader=None, instrument: str = INSTRUMENT):
    pnl, count, losses = 0.0, 0, 0
    for t in history:
        if t.get("timestamp_sgt", "").startswith(today_str) and t.get("status") == "FILLED":
            count += 1
            p = t.get("realized_pnl_usd")
            if isinstance(p, (int, float)):
                pnl += p
                if p < 0:
                    losses += 1
    if trader is not None:
        try:
            position = trader.get_position(instrument)
            if position:
                unrealized = trader.check_pnl(position)
                pnl += unrealized
                # v4.2 (from v4.2): count an open losing position as a loss so the cap
                # fires before the position closes, preventing the 4/3 overshoot
                # where backfill_pnl records the loss one cycle too late.
                if unrealized < 0:
                    losses += 1
        except Exception as e:
            log.warning("Could not fetch unrealized P&L for daily cap: %s", e)
    return pnl, count, losses


def get_trading_day(now_sgt: "datetime", day_start_hour: int = 8) -> str:
    """Return the trading-day date string (YYYY-MM-DD) for a given SGT datetime.

    The trading day starts at day_start_hour SGT (default 08:00) and runs
    until day_start_hour SGT the following calendar day.  Before 08:00 SGT
    the current calendar date still belongs to the *previous* trading day,
    so losses from e.g. 00:30 SGT count against yesterday's cap — not today's.

    This aligns the loss cap with the Asian → London → US session block
    (08:00–23:00 SGT) and prevents mid-night reconcile artefacts from
    poisoning a fresh session's counter (v4.2).
    """
    if now_sgt.hour < day_start_hour:
        return (now_sgt - timedelta(days=1)).strftime("%Y-%m-%d")
    return now_sgt.strftime("%Y-%m-%d")


def session_losses(history: list, session_name: str, trading_day: str) -> int:
    """Count losses recorded during a specific session on a given trading day.

    Used for the per-session loss sub-cap (v4.2): 2 losses inside one session
    stops entries for that session; the next session gets a clean counter.
    session_name should match the macro-session stored on each trade record
    ('Asian' | 'London' | 'US').
    """
    count = 0
    for t in history:
        if t.get("timestamp_sgt", "").startswith(trading_day) and t.get("status") == "FILLED":
            pnl = t.get("realized_pnl_usd")
            if isinstance(pnl, (int, float)) and pnl < 0:
                if t.get("macro_session") == session_name or t.get("session") == session_name:
                    count += 1
    return count


def get_closed_trade_records_today(history: list, today_str: str) -> list:
    closed = []
    for t in history:
        if not t.get("timestamp_sgt", "").startswith(today_str):
            continue
        if t.get("status") != "FILLED":
            continue
        if isinstance(t.get("realized_pnl_usd"), (int, float)):
            closed.append(t)
    closed.sort(key=lambda t: t.get("closed_at_sgt") or t.get("timestamp_sgt") or "")
    return closed


def consecutive_loss_streak_today(history: list, today_str: str) -> int:
    streak = 0
    for t in reversed(get_closed_trade_records_today(history, today_str)):
        pnl = t.get("realized_pnl_usd")
        if not isinstance(pnl, (int, float)):
            continue
        if pnl < 0:
            streak += 1
        else:
            break
    return streak


# _parse_sgt_timestamp — canonical implementation lives in state_utils.parse_sgt_timestamp.
# Alias kept so call sites within this file need no change.
_parse_sgt_timestamp = parse_sgt_timestamp


def maybe_start_loss_cooldown(history: list, today_str: str, now_sgt: datetime, settings: dict):
    cooldown_min = int(settings.get("loss_streak_cooldown_min", 30))
    if cooldown_min <= 0:
        return None, None, 0
    streak = consecutive_loss_streak_today(history, today_str)
    if streak < 2:
        return None, None, streak
    closed = get_closed_trade_records_today(history, today_str)
    if len(closed) < 2:
        return None, None, streak
    trigger_trade  = closed[-1]
    trigger_marker = (
        trigger_trade.get("trade_id")
        or trigger_trade.get("closed_at_sgt")
        or trigger_trade.get("timestamp_sgt")
    )
    runtime_state = load_json(RUNTIME_STATE_FILE, {})
    if runtime_state.get("loss_cooldown_trigger") == trigger_marker:
        cooldown_until = _parse_sgt_timestamp(runtime_state.get("cooldown_until_sgt"))
        return cooldown_until, trigger_marker, streak
    cooldown_until = now_sgt + timedelta(minutes=cooldown_min)
    save_json(
        RUNTIME_STATE_FILE,
        {
            **runtime_state,
            "loss_cooldown_trigger": trigger_marker,
            "cooldown_until_sgt":   cooldown_until.strftime("%Y-%m-%d %H:%M:%S"),
            "cooldown_reason":      f"{streak} consecutive losses",
            "updated_at_sgt":       now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return cooldown_until, trigger_marker, streak


def active_cooldown_until(now_sgt: datetime):
    runtime_state  = load_json(RUNTIME_STATE_FILE, {})
    cooldown_until = _parse_sgt_timestamp(runtime_state.get("cooldown_until_sgt"))
    if cooldown_until and now_sgt < cooldown_until:
        return cooldown_until
    return None


# ── Position sizing ────────────────────────────────────────────────────────────

def compute_sl_usd(levels: dict, settings: dict) -> float:
    """Derive SL distance in USD (= price distance for XAU_USD at 1 unit = 1 oz).

    v4.0 — ATR-based SL is now the default and takes priority.
    The old signal-engine percentage recommendation is no longer used as an
    override; sl_mode in settings.json governs the calculation.

    Modes:
      atr_based  : SL = ATR(14) × atr_sl_multiplier, clamped to [sl_min_usd, sl_max_usd]
      pct_based  : SL = entry_price × sl_pct
      fixed_usd  : SL = fixed_sl_usd
    """
    sl_mode = str(settings.get("sl_mode", "atr_based")).lower()

    if sl_mode == "atr_based":
        atr = levels.get("atr")
        if atr and atr > 0:
            mult         = float(settings.get("atr_sl_multiplier", 1.0))
            sl_min_fixed = float(settings.get("sl_min_usd", 25.0))
            sl_max       = float(settings.get("sl_max_usd", 60.0))
            # v5.1 — adaptive floor: sl_min = max(fixed_floor, ATR × sl_min_atr_mult)
            # On quiet days (ATR $20) floor adapts down to $16 instead of locking at $35.
            # On volatile days (ATR $50) floor stays at $40, respecting sl_min_usd.
            atr_floor    = atr * float(settings.get("sl_min_atr_mult", 0.8))
            sl_min       = max(sl_min_fixed, atr_floor)
            raw_sl       = atr * mult
            sl_usd       = max(sl_min, min(sl_max, raw_sl))
            log.debug(
                "ATR SL: ATR=%.2f × %.2f = %.2f → adaptive_floor=$%.2f → clamped $%.2f",
                atr, mult, raw_sl, sl_min, sl_usd,
            )
            return round(sl_usd, 2)
        # ATR unavailable — fall through to pct_based
        log.warning("atr_based SL: ATR not available — falling back to pct_based")

    if sl_mode == "fixed_usd":
        return float(settings.get("fixed_sl_usd", 20.0))

    # pct_based (default fallback)
    entry  = levels.get("entry") or levels.get("current_price", 0)
    sl_pct = float(settings.get("sl_pct", 0.0025))
    if entry and entry > 0 and sl_pct > 0:
        sl_usd = round(entry * sl_pct, 2)
        log.debug("Pct SL: %.2f × %.4f%% = $%.2f", entry, sl_pct * 100, sl_usd)
        return sl_usd
    fallback = float(settings.get("fixed_sl_usd", 20.0))
    log.warning("pct_based SL: no valid entry price — fallback $%.2f", fallback)
    return fallback


def compute_tp_usd(levels: dict, sl_usd: float, settings: dict) -> float:
    """Derive TP distance in USD.

    v4.6 priority order:
      1. Structural TP from signal engine (tp_usd_rec) if it satisfies min RR
         AND does not exceed max_rr_ratio cap.
         This prevents the 1:5 TP bug where structural levels place TP far
         beyond the intended RR range.
      2. fixed_usd override when tp_mode == "fixed_usd".
      3. Fallback: sl_usd x rr_ratio (min RR multiple).
    All results capped at sl_usd x max_rr_ratio.
    """
    min_rr  = float(settings.get("rr_ratio", 2.65))
    max_rr  = float(settings.get("max_rr_ratio", 3.0))
    tp_ceil = round(sl_usd * max_rr, 2)   # hard ceiling regardless of source

    # 1. Structural TP from signals.py (R1/S1 level)
    structural_tp = levels.get("tp_usd_rec")
    if structural_tp is not None:
        try:
            stp = float(structural_tp)
            if stp > 0 and stp >= sl_usd * min_rr:
                return round(min(stp, tp_ceil), 2)
        except (TypeError, ValueError):
            pass

    # 2. Fixed USD override
    tp_mode = str(settings.get("tp_mode", "rr_multiple")).lower()
    if tp_mode == "fixed_usd":
        fixed = settings.get("fixed_tp_usd")
        if fixed is not None:
            try:
                v = float(fixed)
                if v > 0:
                    return round(min(v, tp_ceil), 2)
            except (TypeError, ValueError):
                pass

    # 3. RR multiple fallback (already within cap since min_rr <= max_rr)
    return round(sl_usd * min_rr, 2)


def derive_rr_ratio(levels: dict, sl_usd: float, tp_usd: float, settings: dict) -> float:
    try:
        rr = float(levels.get("rr_ratio"))
        if rr > 0:
            return rr
    except (TypeError, ValueError):
        pass
    if sl_usd > 0 and tp_usd > 0:
        return round(tp_usd / sl_usd, 2)
    return float(settings.get("rr_ratio", 2.65))


# Note: compute_atr_sl_usd alias removed — no external callers exist in this codebase

def calculate_units_from_position(position_usd: int, sl_usd: float) -> float:
    """Convert score-based position risk to OANDA units.

    units = position_usd / sl_usd
    e.g. $66 risk at $6 SL = 11 units of XAU_USD
    """
    if sl_usd <= 0 or position_usd <= 0:
        return 0.0
    return round(position_usd / sl_usd, 2)


def apply_margin_guard(
    trader,
    instrument: str,
    requested_units: float,
    entry_price: float,
    free_margin: float,
    settings: dict,
) -> tuple[float, dict]:
    """Floor requested units against available margin before order placement."""
    margin_safety = float(settings.get("margin_safety_factor", 0.75))
    margin_retry_safety = float(settings.get("margin_retry_safety_factor", 0.4))
    specs = trader.get_instrument_specs(instrument)
    configured_floor = float(settings.get("xau_margin_rate_override", 0.05) or 0.05) if instrument == "XAU_USD" else 0.0
    margin_rate = max(float(specs.get("marginRate", 0.05) or 0.05), configured_floor)
    normalized_requested = trader.normalize_units(instrument, requested_units)
    required_margin_requested = trader.estimate_required_margin(instrument, normalized_requested, entry_price)

    if free_margin <= 0 or entry_price <= 0 or margin_rate <= 0:
        return 0.0, {
            "status": "SKIP",
            "reason": "invalid_margin_context",
            "free_margin": float(free_margin or 0),
            "required_margin": required_margin_requested,
            "requested_units": normalized_requested,
            "final_units": 0.0,
        }

    max_units_by_margin = (free_margin * margin_safety) / (entry_price * margin_rate)
    normalized_capped = trader.normalize_units(instrument, min(normalized_requested, max_units_by_margin))
    required_margin_final = trader.estimate_required_margin(instrument, normalized_capped, entry_price)
    status = "NORMAL" if abs(normalized_capped - normalized_requested) < 1e-9 else "ADJUSTED"
    reason = "margin_guard" if status == "ADJUSTED" else "ok"

    if normalized_capped <= 0:
        retry_units = trader.normalize_units(
            instrument,
            (free_margin * margin_retry_safety) / (entry_price * margin_rate),
        )
        required_retry = trader.estimate_required_margin(instrument, retry_units, entry_price)
        if retry_units > 0:
            return retry_units, {
                "status": "ADJUSTED",
                "reason": "margin_retry_floor",
                "free_margin": float(free_margin),
                "required_margin": required_retry,
                "requested_units": normalized_requested,
                "final_units": retry_units,
            }
        return 0.0, {
            "status": "SKIP",
            "reason": "insufficient_margin",
            "free_margin": float(free_margin),
            "required_margin": required_margin_requested,
            "requested_units": normalized_requested,
            "final_units": 0.0,
        }

    return normalized_capped, {
        "status": status,
        "reason": reason,
        "free_margin": float(free_margin),
        "required_margin": required_margin_final,
        "requested_units": normalized_requested,
        "final_units": normalized_capped,
    }


def compute_sl_tp_pips(sl_usd: float, tp_usd: float):
    pip = 0.01
    return round(sl_usd / pip), round(tp_usd / pip)


def compute_sl_tp_prices(entry: float, direction: str, sl_usd: float, tp_usd: float):
    """Return (sl_price, tp_price) based on direction and dollar distances."""
    if direction == "BUY":
        return round(entry - sl_usd, 2), round(entry + tp_usd, 2)
    return round(entry + sl_usd, 2), round(entry - tp_usd, 2)


def get_effective_balance(balance: float | None, settings: dict) -> float:
    override = settings.get("account_balance_override")
    if override is not None:
        try:
            v = float(override)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return float(balance or 0)


# ── Score / cache helpers ─────────────────────────────────────────────────────

def load_signal_cache() -> dict:
    """Load signal dedup cache (score, direction, last_signal_msg)."""
    if not SCORE_CACHE_FILE.exists():
        return {}
    try:
        with open(SCORE_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_signal_cache(cache: dict):
    atomic_json_write(SCORE_CACHE_FILE, cache)


def load_ops_state() -> dict:
    """Load ops state cache (ops_state, last_session)."""
    if not OPS_STATE_FILE.exists():
        return {}
    try:
        with open(OPS_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_ops_state(state: dict):
    atomic_json_write(OPS_STATE_FILE, state)


# Keep backward-compat aliases so nothing outside bot.py needs touching
load_score_cache = load_signal_cache
save_score_cache = save_signal_cache


def send_once_per_state(alert, cache: dict, key: str, value: str, message: str):
    if cache.get(key) != value:
        alert.send(message)
        cache[key] = value
        save_ops_state(cache)


# ── Break-even management ──────────────────────────────────────────────────────

def check_breakeven(history: list, trader, alert, settings: dict):
    """Tiered exit management — v4.1.

    Stage 1 (at 1x SL profit):
      - Partial-close 50% of the position to lock realized profit.
      - Move SL to breakeven so the runner is risk-free.

    Stage 2:
      - The server-side trailing stop (set at order placement) handles
        the runner automatically — no further polling needed.

    The ``breakeven_moved`` flag gates both stages so they fire at most
    once per trade.
    """
    demo    = settings.get("demo_mode", True)
    sl_min  = float(settings.get("sl_min_usd", 20.0))
    changed = False

    for trade in history:
        if trade.get("status") != "FILLED":
            continue
        if trade.get("breakeven_moved"):
            continue

        trade_id   = trade.get("trade_id")
        entry      = trade.get("entry")
        direction  = trade.get("direction", "")
        sl_usd     = trade.get("sl_usd") or sl_min
        units_open = trade.get("size")

        if not trade_id or not entry or direction not in ("BUY", "SELL"):
            continue

        open_trade = trader.get_open_trade(str(trade_id))
        if open_trade is None:
            continue

        try:
            unrealized_pnl = float(open_trade.get("unrealizedPL", 0))
        except (TypeError, ValueError):
            continue

        # Gate: trigger only when unrealized profit >= 1x SL risk
        if unrealized_pnl < sl_usd:
            continue

        trigger_price = (
            entry + sl_usd if direction == "BUY" else entry - sl_usd
        )

        # Stage 1a: partial close 50%
        partial_ok = False
        if units_open and units_open > 0:
            half_units   = round(units_open * 0.5, 1)
            close_result = trader.close_partial(str(trade_id), half_units)
            partial_ok   = close_result.get("success", False)
            if partial_ok:
                realized = close_result.get("realized_pnl", 0)
                log.info(
                    "Partial close %.1f units | trade %s | unrealized=+$%.2f | realized=+$%.2f",
                    half_units, trade_id, unrealized_pnl, realized,
                )
            else:
                log.warning(
                    "Partial close failed for trade %s: %s",
                    trade_id, close_result.get("error"),
                )

        # Stage 1b: move SL to breakeven on remaining position
        sl_result = trader.modify_sl(str(trade_id), float(entry))
        if sl_result.get("success"):
            trade["breakeven_moved"] = True
            trade["partial_closed"]  = partial_ok
            changed = True
            log.info(
                "Breakeven set | trade %s | entry=%.2f | unrealized=+$%.2f | partial=%s",
                trade_id, entry, unrealized_pnl, partial_ok,
            )
            alert.send(msg_breakeven(
                trade_id=trade_id,
                direction=direction,
                entry=entry,
                trigger_price=trigger_price,
                trigger_usd=sl_usd,
                current_price=trigger_price,
                unrealized_pnl=unrealized_pnl,
                demo=demo,
            ))
        else:
            log.warning(
                "Breakeven SL move failed for trade %s: %s",
                trade_id, sl_result.get("error"),
            )

    if changed:
        save_history(history)


# v4.1: Consecutive-direction loss guard helper

def _count_consecutive_sl(history: list, direction: str) -> int:
    """Count consecutive SL-hit trades in the same direction.
    Resets on a direction change or a profitable close."""
    count = 0
    for trade in reversed(history):
        if trade.get("status") != "FILLED":
            continue
        pnl = trade.get("realized_pnl_usd")
        if pnl is None:
            continue                          # still open — skip
        if trade.get("direction") != direction:
            break                             # direction flipped — streak over
        if pnl < 0:
            count += 1
        else:
            break                             # TP hit — streak over
    return count


# ── PnL backfill ───────────────────────────────────────────────────────────────

def backfill_pnl(history: list, trader, alert, settings: dict) -> list:
    changed = False
    demo = settings.get("demo_mode", True)
    for trade in history:
        if trade.get("status") == "FILLED" and trade.get("realized_pnl_usd") is None:
            trade_id = trade.get("trade_id")
            if trade_id:
                pnl = trader.get_trade_pnl(str(trade_id))
                if pnl is not None:
                    trade["realized_pnl_usd"] = pnl
                    trade["closed_at_sgt"] = datetime.now(SGT).strftime("%Y-%m-%d %H:%M:%S")
                    changed = True
                    log.info("Back-filled P&L trade %s: $%.2f", trade_id, pnl)
                    if not trade.get("closed_alert_sent"):
                        try:
                            _cp  = trade.get("tp_price") if pnl > 0 else trade.get("sl_price")
                            _dur = ""
                            _t1s = trade.get("timestamp_sgt", "")
                            _t2s = trade.get("closed_at_sgt", "")
                            if _t1s and _t2s:
                                _d = int(
                                    (datetime.strptime(_t2s, "%Y-%m-%d %H:%M:%S") -
                                     datetime.strptime(_t1s, "%Y-%m-%d %H:%M:%S")).total_seconds() // 60
                                )
                                _dur = f"{_d // 60}h {_d % 60}m" if _d >= 60 else f"{_d}m"
                            alert.send(msg_trade_closed(
                                trade_id=trade_id,
                                direction=trade.get("direction", ""),
                                setup=trade.get("setup", ""),
                                entry=float(trade.get("entry", 0)),
                                close_price=float(_cp or 0),
                                pnl=float(pnl),
                                session=trade.get("session", ""),
                                demo=demo,
                                duration_str=_dur,
                            ))
                            trade["closed_alert_sent"] = True
                        except Exception as _e:
                            log.warning("Could not send trade_closed alert: %s", _e)
    if changed:
        save_history(history)
    return history


# ── Logging helper ─────────────────────────────────────────────────────────────

def log_event(code: str, message: str, level: str = "info", **extra):
    logger_fn = getattr(log, level, log.info)
    payload   = {"event": code}
    payload.update(extra)
    logger_fn(f"[{code}] {message}", extra=payload)


# ── Main cycle ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Cycle phases
#
# run_bot_cycle() is the thin public entry point called by the scheduler.
# It delegates to three private helpers, each with a single responsibility:
#
#   _guard_phase()      — all pre-trade checks: calendar, login, caps, session,
#                         news, cooldowns, spread.  Returns a populated ctx dict
#                         on success, or None to abort the cycle.
#   _signal_phase()     — CPR signal evaluation, position sizing, margin guard.
#                         Returns ctx with execution-ready parameters, or None.
#   _execution_phase()  — places the order and persists the trade record.
# ─────────────────────────────────────────────────────────────────────────────



def _should_pyramid(ctx: dict, open_trades: list, history: list, settings: dict) -> tuple[bool, str]:
    """Check all pyramid conditions. Returns (allowed, reason_string).

    Conditions (ALL must pass):
      1. pyramid_enabled is True in settings
      2. Exactly 1 open trade on the instrument
      3. Signal score >= pyramid_min_score (default 5)
      4. News penalty is zero (clean signal only)
      5. Open trade direction matches current signal direction
      6. Open trade unrealized P&L > 0 (adding to a winner only)
    """
    if not settings.get("pyramid_enabled", False):
        return False, "pyramid_disabled"

    if len(open_trades) != 1:
        return False, f"open_trades={len(open_trades)} (need exactly 1)"

    score     = ctx.get("score", 0)
    direction = ctx.get("direction", "NONE")
    min_score = int(settings.get("pyramid_min_score", 5))

    if score < min_score:
        return False, f"score {score}/6 < pyramid_min_score {min_score}"

    if ctx.get("news_penalty", 0) != 0:
        return False, f"news_penalty active ({ctx['news_penalty']})"

    if direction == "NONE":
        return False, "no directional signal"

    # Direction match — check local history first, fall back to broker units
    broker_trade  = open_trades[0]
    broker_id     = str(broker_trade.get("id", ""))
    open_local    = next(
        (t for t in history
         if t.get("status") == "FILLED" and str(t.get("trade_id", "")) == broker_id),
        None,
    )
    if open_local:
        trade_direction = open_local.get("direction", "")
    else:
        units_val       = float(broker_trade.get("currentUnits", 0))
        trade_direction = "BUY" if units_val > 0 else "SELL"

    if trade_direction != direction:
        return False, f"direction mismatch (T1={trade_direction}, signal={direction})"

    # Trade 1 must be in profit
    unrealized = float(broker_trade.get("unrealizedPL", 0))
    if unrealized <= 0:
        return False, f"T1 not yet profitable (P&L=${unrealized:.2f})"

    return True, "all_pass"


def _pyramid_phase(db, run_id, settings, alert, trader, history, now_sgt, today, demo, ctx) -> dict | None:
    """v4.2 — Evaluate and configure pyramid (Trade 2 add-on) parameters.

    Re-sizes ctx with tight SL ($1.50 fixed) and same S1/R1 TP as Trade 1.
    Returns modified ctx if all conditions pass, or None to abort.
    """
    # v4.2 fix: re-check the loss cap here using live broker state.
    # _guard_phase() checked at cycle start, but a trade may have closed
    # as a loss between then and now (e.g. fast stop-out during signal eval).
    # If the cap is now met, block the pyramid add immediately — no new position.
    _pyr_losses_pnl, _pyr_losses_count, _pyr_losses = daily_totals(history, today, trader=trader)
    _pyr_max_losses = int(settings.get("max_losing_trades_day", 3))
    if _pyr_losses >= _pyr_max_losses:
        reason = f"loss_cap_reached ({_pyr_losses}/{_pyr_max_losses}) — pyramid blocked (v4.2)"
        log.warning("Pyramid blocked by loss cap re-check: %s", reason, extra={"run_id": run_id})
        update_runtime_state(
            last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
            status="SKIPPED_LOSS_CAP",
        )
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "pyramid_loss_cap", "reason": reason})
        return None

    open_trades = trader.get_open_trades(INSTRUMENT)
    allowed, reason = _should_pyramid(ctx, open_trades, history, settings)

    if not allowed:
        log.info("Pyramid skipped: %s", reason, extra={"run_id": run_id})
        update_runtime_state(
            last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
            status="SKIPPED_PYRAMID_CONDITIONS",
            reason=reason,
        )
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "pyramid_guard", "reason": reason})
        return None

    broker_trade    = open_trades[0]
    broker_id       = str(broker_trade.get("id", ""))
    unrealized_pnl  = float(broker_trade.get("unrealizedPL", 0))

    pyramid_sl_usd  = float(settings.get("pyramid_sl_usd", 1.50))
    pyramid_max_risk = float(settings.get("pyramid_max_risk_usd", 50))

    # Units from tight SL
    pyramid_units = calculate_units_from_position(pyramid_max_risk, pyramid_sl_usd)

    # TP stays same S1/R1 level from signal phase
    tp_usd = ctx["tp_usd"]

    # Re-run margin guard with pyramid sizing
    pyramid_units, margin_info = apply_margin_guard(
        trader=trader, instrument=INSTRUMENT,
        requested_units=pyramid_units, entry_price=ctx["entry"],
        free_margin=ctx["margin_available"], settings=settings,
    )

    if pyramid_units <= 0:
        reason = f"insufficient margin for pyramid add (free=${ctx['margin_available']:.2f})"
        log.info("Pyramid skipped: %s", reason, extra={"run_id": run_id})
        update_runtime_state(
            last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
            status="SKIPPED_PYRAMID_MARGIN",
        )
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "pyramid_margin"})
        return None

    stop_pips, tp_pips = compute_sl_tp_pips(pyramid_sl_usd, tp_usd)
    reward_usd = round(pyramid_units * tp_usd, 2)
    rr_ratio   = round(tp_usd / pyramid_sl_usd, 2) if pyramid_sl_usd > 0 else 0

    log.info(
        "Pyramid ADD approved | score=%d/6 | dir=%s | T1_pnl=+$%.2f | units=%.2f | sl=$%.2f | tp=$%.2f | R:R 1:%.1f",
        ctx["score"], ctx["direction"], unrealized_pnl,
        pyramid_units, pyramid_sl_usd, tp_usd, rr_ratio,
        extra={"run_id": run_id},
    )

    ctx.update({
        "is_pyramid":              True,
        "pyramid_trade_id":        broker_id,
        "pyramid_unrealized_pnl":  unrealized_pnl,
        "sl_usd":                  pyramid_sl_usd,
        "position_usd":            int(pyramid_max_risk),
        "units":                   pyramid_units,
        "stop_pips":               stop_pips,
        "tp_pips":                 tp_pips,
        "reward_usd":              reward_usd,
        "rr_ratio":                rr_ratio,
        "margin_info":             margin_info,
    })
    return ctx


def _guard_phase(db, run_id, settings, alert, trader, history, now_sgt, today, demo) -> dict | None:
    """All pre-trade guards.  Returns a populated context dict or None (cycle aborted)."""

    # ops_state cache: deduplicates operational Telegram alerts (session changes,
    # news blocks, cooldowns, caps). Stored in ops_state.json — separate from
    # signal_cache.json which tracks score/direction dedup.
    ops = load_ops_state()

    warnings = run_startup_checks()
    for warning in warnings:
        log.warning(warning, extra={"run_id": run_id})

    log.info(
        "=== %s | %s SGT ===",
        settings.get("bot_name", "CPR Gold Bot"),
        now_sgt.strftime("%Y-%m-%d %H:%M"),
        extra={"run_id": run_id},
    )
    update_runtime_state(
        last_cycle_started=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
        last_run_id=run_id,
        status="RUNNING",
    )
    db.upsert_state("last_cycle_started", {
        "run_id": run_id,
        "started_at_sgt": now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
    })

    if not settings.get("enabled", True) or get_bool_env("TRADING_DISABLED", False):
        log.warning("Trading disabled.", extra={"run_id": run_id})
        send_once_per_state(alert, ops, "ops_state", "disabled", "⏸️ Trading disabled by configuration.")
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_DISABLED")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "enabled_check", "reason": "disabled"})
        return None

    history[:] = prune_old_trades(history)
    save_history(history)

    weekday = now_sgt.weekday()
    if weekday == 5:
        log.info("Saturday — market closed.", extra={"run_id": run_id})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_MARKET_CLOSED")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "market_guard", "reason": "Saturday"})
        return None
    if weekday == 6:
        log.info("Sunday — waiting for Monday open.", extra={"run_id": run_id})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_MARKET_CLOSED")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "market_guard", "reason": "Sunday"})
        return None
    if weekday == 0 and now_sgt.hour < 8:
        log.info("Monday pre-open (before 08:00 SGT) — skipping.", extra={"run_id": run_id})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_MARKET_CLOSED")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "market_guard", "reason": "Monday pre-open"})
        return None

    if settings.get("news_filter_enabled", True):
        try:
            refresh_calendar()
        except Exception as e:
            log.warning("Calendar refresh failed (using cached): %s", e, extra={"run_id": run_id})

    history[:] = backfill_pnl(history, trader, alert, settings)
    # v4.1 — gated by breakeven_enabled (default True).
    # Set breakeven_enabled: false to disable tiered exit entirely.
    if settings.get("breakeven_enabled", True):
        check_breakeven(history, trader, alert, settings)

    # ── Early daily loss-cap check ─────────────────────────────────────────────
    # Must run BEFORE cooldown_started notification so we never show a misleading
    # "Resumes HH:MM" timestamp when the daily cap is already exhausted for the day.
    # trader=None is intentional here — we only need the loss *count* from history.
    _early_pnl, _early_trades, _early_losses = daily_totals(history, today)
    _max_losses_early = int(settings.get("max_losing_trades_day", 3))
    if _early_losses >= _max_losses_early:
        _day_start_h = int(settings.get("trading_day_start_hour_sgt", 8))
        _day_reset = (now_sgt + timedelta(days=1)).replace(hour=_day_start_h, minute=0, second=0, microsecond=0)
        msg = msg_daily_cap(
            "losing_trades", _early_losses, _max_losses_early,
            day_start_sgt=f"{_day_start_h:02d}:00", day_end_sgt="01:00",  # v4.2 — US session closes 00:59 SGT
            day_reset_sgt=_day_reset.strftime("%Y-%m-%d %H:%M"),
        )
        log_event("COOLDOWN_ACTIVE", msg, run_id=run_id)
        send_once_per_state(alert, ops, "loss_cap_state", f"loss_cap:{today}", msg)
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_LOSS_CAP")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "daily_caps", "reason": "loss_cap"})
        return None

    cooldown_started_until, _, cooldown_streak = maybe_start_loss_cooldown(history, today, now_sgt, settings)
    if cooldown_started_until and now_sgt < cooldown_started_until:
        send_once_per_state(
            alert, ops, "cooldown_started_state",
            f"cooldown_started:{cooldown_started_until.strftime('%Y-%m-%d %H:%M:%S')}",
            msg_cooldown_started(streak=cooldown_streak, cooldown_until_sgt=cooldown_started_until.strftime("%H:%M")),
        )
        log_event("COOLDOWN_STARTED", f"Cooldown until {cooldown_started_until.strftime('%Y-%m-%d %H:%M:%S')} SGT.", run_id=run_id)

    session, macro, threshold = get_session(now_sgt, settings)

    if is_friday_cutoff(now_sgt, settings):
        log_event("FRIDAY_CUTOFF", "Friday cutoff active.", run_id=run_id)
        send_once_per_state(alert, ops, "ops_state",
            f"friday_cutoff:{now_sgt.strftime('%Y-%m-%d')}",
            msg_friday_cutoff(int(settings.get("friday_cutoff_hour_sgt", 23))),
        )
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_FRIDAY_CUTOFF")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "friday_cutoff"})
        return None

    # v4.2 — The midnight guard (blocking entries at 00:00 SGT) is superseded
    # by the 08:00 SGT trading-day boundary. Trades between 00:00 and 08:00 SGT
    # are inside the dead zone (no active sessions) AND counted against the
    # previous trading day's cap, so double protection exists without a separate guard.

    if settings.get("session_only", True):
        if session is None:
            if is_dead_zone_time(now_sgt, settings):
                log_event("DEAD_ZONE_SKIP", "Dead zone — entry blocked, management active.", run_id=run_id)
            else:
                log.info("Outside all sessions — skipping.", extra={"run_id": run_id})
            if ops.get("last_session") is not None:
                send_once_per_state(alert, ops, "ops_state", "outside_session", "⏸️ Outside active session — no trade.")
                ops["last_session"] = None
                save_ops_state(ops)
            update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_OUTSIDE_SESSION")
            db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "session_check", "reason": "outside_session"})
            return None
    else:
        if session is None:
            session, macro = "All Hours", "London"
        threshold = int(settings.get("signal_threshold", 4))

    threshold = threshold or int(settings.get("signal_threshold", 4))
    banner    = SESSION_BANNERS.get(macro, "📊")
    log.info("Session: %s (%s)", session, macro, extra={"run_id": run_id})

    if ops.get("last_session") != session:
        ops["last_session"] = session
        ops.pop("ops_state", None)
        save_ops_state(ops)

    # ── News filter ────────────────────────────────────────────────────────────
    news_penalty = 0
    news_status  = {}
    if settings.get("news_filter_enabled", True):
        nf = NewsFilter(
            before_minutes=int(settings.get("news_block_before_min", 30)),
            after_minutes=int(settings.get("news_block_after_min", 30)),
            lookahead_minutes=int(settings.get("news_lookahead_min", 120)),
            medium_penalty=int(settings.get("news_medium_penalty_score", -1)),
        )
        news_status  = nf.get_status_now()
        blocked      = bool(news_status.get("blocked"))
        reason       = str(news_status.get("reason", "No blocking news"))
        news_penalty = int(news_status.get("penalty", 0))
        lookahead    = news_status.get("lookahead", [])
        if lookahead:
            la_summary = " | ".join(
                f"{e['name']} in {e['mins_away']}min ({e['severity']})"
                for e in lookahead[:3]
            )
            log.info("Upcoming news: %s", la_summary, extra={"run_id": run_id})
        if blocked:
            _evt       = news_status.get("event", {})
            _block_msg = msg_news_block(
                event_name=_evt.get("name", reason),
                event_time_sgt=_evt.get("time_sgt", ""),
                before_min=int(settings.get("news_block_before_min", 30)),
                after_min=int(settings.get("news_block_after_min", 30)),
            )
            send_once_per_state(alert, ops, "ops_state", f"news:{reason}", _block_msg)
            db.upsert_state("last_news_block", {"blocked": True, "reason": reason, "checked_at_sgt": now_sgt.strftime("%Y-%m-%d %H:%M:%S")})
            update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_NEWS_BLOCK", reason=reason)
            db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "news_filter", "reason": reason})
            return None
        db.upsert_state("last_news_block", {
            "blocked": False, "reason": reason if news_penalty else None,
            "penalty": news_penalty, "checked_at_sgt": now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # ── OANDA login (single call — balance + margin in one request) ───────────
    account_summary = trader.login_with_summary()
    if account_summary is None:
        alert.send(msg_error("OANDA login failed", "Check OANDA_API_KEY and OANDA_ACCOUNT_ID"))
        db.finish_cycle(run_id, status="FAILED", summary={"stage": "oanda_login", "reason": "login_failed"})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="FAILED_LOGIN")
        return None
    balance = account_summary["balance"]
    if balance <= 0:
        alert.send(msg_error("Cannot fetch balance", "OANDA account returned $0 or invalid"))
        db.finish_cycle(run_id, status="FAILED", summary={"stage": "oanda_login", "reason": "invalid_balance"})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="FAILED_LOGIN")
        return None

    reconcile = reconcile_runtime_state(trader, history, INSTRUMENT, now_sgt, alert=alert)
    if reconcile.get("recovered_trade_ids") or reconcile.get("backfilled_trade_ids"):
        save_history(history)
    db.upsert_state("last_reconciliation", {**reconcile, "checked_at_sgt": now_sgt.strftime("%Y-%m-%d %H:%M:%S")})

    # ── Daily caps ─────────────────────────────────────────────────────────────
    daily_pnl, daily_trades, daily_losses = daily_totals(history, today, trader=trader)
    max_losses = int(settings.get("max_losing_trades_day", 3))
    if daily_losses >= max_losses:
        day_start_h = int(settings.get("trading_day_start_hour_sgt", 8))
        day_end_h   = 23  # US session hard cutoff
        day_reset_sgt = (now_sgt + timedelta(days=1)).replace(hour=day_start_h, minute=0, second=0, microsecond=0)
        msg = msg_daily_cap(
            "losing_trades", daily_losses, max_losses,
            day_start_sgt=f"{day_start_h:02d}:00", day_end_sgt="01:00",  # v4.2 — US session closes 00:59 SGT
            day_reset_sgt=day_reset_sgt.strftime("%Y-%m-%d %H:%M"),
        )
        log_event("COOLDOWN_ACTIVE", msg, run_id=run_id)
        send_once_per_state(alert, ops, "loss_cap_state", f"loss_cap:{today}", msg)
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_LOSS_CAP")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "daily_caps", "reason": "loss_cap"})
        return None

    # v4.2 — Per-session loss sub-cap.
    # 2 losses in a single session (Asian/London/US) pause entries for that
    # session only. The next session starts with a clean counter; the overall
    # 3-loss daily hard stop still applies across all sessions.
    if session is not None:
        max_session_losses = int(settings.get("max_losing_trades_session", 2))
        sess_losses = session_losses(history, session, today)
        if sess_losses >= max_session_losses:
            # v4.2 — Asian disabled; only London and US
            # v4.2 — derive next-session display strings from settings, not hardcoded
            _us_start    = int(settings.get("session_end_hour_sgt", 1))   # US ends at 01:00
            _lon_start   = int(settings.get("session_start_hour_sgt", 16))
            _us_show     = 21  # US Window starts 21:00 SGT (within session block)
            next_sessions = {
                "Asian":  f"London ({_lon_start:02d}:00 SGT)",
                "London": f"US ({_us_show:02d}:00 SGT)",
                "US":     f"London ({_lon_start:02d}:00 SGT next day)",
            }
            msg = msg_session_cap(
                session=session, count=sess_losses, limit=max_session_losses,
                next_session=next_sessions.get(session, "next session"),
                day_losses=daily_losses, day_limit=max_losses,
            )
            log_event("COOLDOWN_ACTIVE", msg, run_id=run_id)
            send_once_per_state(alert, ops, "session_cap_state", f"session_cap:{today}:{session}", msg)
            update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_SESSION_CAP")
            db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "session_cap", "reason": f"{session}_loss_cap"})
            return None

    if daily_trades >= int(settings.get("max_trades_day", 8)):
        msg = msg_daily_cap("total_trades", daily_trades, int(settings.get("max_trades_day", 8)))
        send_once_per_state(alert, ops, "trade_cap_state", f"trade_cap:{today}", msg)
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_TRADE_CAP")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "daily_caps", "reason": "trade_cap"})
        return None

    cooldown_until = active_cooldown_until(now_sgt)
    if cooldown_until:
        remaining_min = max(1, int((cooldown_until - now_sgt).total_seconds() // 60))
        msg = f"🧊 Cooldown active — new entries paused for {remaining_min} more minute(s)."
        send_once_per_state(alert, ops, "cooldown_guard_state", f"cooldown:{cooldown_until.strftime('%Y-%m-%d %H:%M:%S')}", msg)
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_COOLDOWN")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "cooldown_guard"})
        return None

    # ── Post-win M15 candle block (v5.2) ──────────────────────────────────────
    # After a TP close, block new entries until:
    #   1. The M15 candle the win closed ON has fully passed, AND
    #   2. A brand-new M15 candle has opened and confirmed.
    # This is NOT a cooldown timer — it is purely candle-boundary based.
    if settings.get("post_win_candle_block", True):
        _last_win = next(
            (t for t in reversed(history)
             if t.get("status") == "FILLED" and (t.get("realized_pnl_usd") or 0) > 0),
            None,
        )
        if _last_win:
            _closed_at_str = _last_win.get("closed_at_sgt", "")
            if _closed_at_str:
                try:
                    _closed_at = datetime.strptime(_closed_at_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=SGT)
                    # Which M15 candle did the win close on?
                    _win_candle_start = _closed_at.replace(
                        minute=(_closed_at.minute // 15) * 15,
                        second=0, microsecond=0,
                    )
                    _win_candle_end = _win_candle_start + timedelta(minutes=15)
                    # The very next candle after that also needs to fully open —
                    # so the block lifts when a new candle starts AFTER win_candle_end.
                    # i.e. current_candle_start must be >= win_candle_end
                    _current_candle_start = now_sgt.replace(
                        minute=(now_sgt.minute // 15) * 15,
                        second=0, microsecond=0,
                    )
                    if _current_candle_start < _win_candle_end:
                        # Still on the same candle the win closed on
                        _secs_left = int((_win_candle_end - now_sgt).total_seconds())
                        _mins_left = max(1, _secs_left // 60)
                        _msg = (
                            f"🕯️ Post-win candle block — win closed on M15 candle "
                            f"{_win_candle_start.strftime('%H:%M')}–{_win_candle_end.strftime('%H:%M')} SGT "
                            f"(block lifts when next candle opens, ≈{_mins_left}m)"
                        )
                        log.info(_msg, extra={"run_id": run_id})
                        send_once_per_state(
                            alert, ops, "post_win_candle_state",
                            f"post_win_same_candle:{_win_candle_start.strftime('%Y-%m-%d %H:%M')}",
                            _msg,
                        )
                        update_runtime_state(
                            last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
                            status="SKIPPED_POST_WIN_CANDLE",
                        )
                        db.finish_cycle(run_id, status="SKIPPED",
                                        summary={"stage": "post_win_candle_block",
                                                 "win_candle": _win_candle_start.strftime("%H:%M"),
                                                 "reason": "still_on_win_candle"})
                        return None
                    elif _current_candle_start == _win_candle_end:
                        # On the very next candle right after the win — wait for it to close too
                        _next_candle_end = _win_candle_end + timedelta(minutes=15)
                        _secs_left = int((_next_candle_end - now_sgt).total_seconds())
                        _mins_left = max(1, _secs_left // 60)
                        _msg = (
                            f"🕯️ Post-win candle block — confirming next M15 candle "
                            f"{_current_candle_start.strftime('%H:%M')}–{_next_candle_end.strftime('%H:%M')} SGT "
                            f"(block lifts when candle closes, ≈{_mins_left}m)"
                        )
                        log.info(_msg, extra={"run_id": run_id})
                        send_once_per_state(
                            alert, ops, "post_win_candle_state",
                            f"post_win_confirm_candle:{_current_candle_start.strftime('%Y-%m-%d %H:%M')}",
                            _msg,
                        )
                        update_runtime_state(
                            last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
                            status="SKIPPED_POST_WIN_CANDLE_CONFIRM",
                        )
                        db.finish_cycle(run_id, status="SKIPPED",
                                        summary={"stage": "post_win_candle_block",
                                                 "win_candle": _win_candle_start.strftime("%H:%M"),
                                                 "reason": "waiting_for_confirm_candle"})
                        return None
                    # else: current candle is ≥ 2 candles after win candle → block lifted, proceed
                except Exception as _pwe:
                    log.warning("Post-win candle block check error: %s", _pwe, extra={"run_id": run_id})

    window_key = get_window_key(session)
    window_cap = get_window_trade_cap(window_key, settings)
    if window_key and window_cap is not None:
        trades_in_window = window_trade_count(history, today, window_key)
        if trades_in_window >= window_cap:
            msg = msg_daily_cap("window", trades_in_window, window_cap, window=window_key)
            send_once_per_state(alert, ops, "window_cap_state", f"window_cap:{today}:{window_key}", msg)
            update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_WINDOW_CAP")
            db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "window_guard", "window": window_key})
            return None

    open_count     = trader.get_open_trades_count(INSTRUMENT)
    max_concurrent = int(settings.get("max_concurrent_trades", 1))
    pyramid_enabled = bool(settings.get("pyramid_enabled", False))

    # Pyramid passthrough: if exactly 1 trade open, pyramid is enabled, and
    # max_concurrent allows a second — let the cycle continue to signal
    # evaluation so _pyramid_phase can check the full conditions.
    pyramid_possible = (
        pyramid_enabled
        and open_count == 1
        and max_concurrent >= 2
    )

    if open_count >= max_concurrent and not pyramid_possible:
        msg = f"⏸️ Max concurrent trades reached ({open_count}/{max_concurrent}) — waiting."
        send_once_per_state(alert, ops, "open_cap_state", f"open_cap:{open_count}:{max_concurrent}", msg)
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_OPEN_TRADE_CAP")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "open_trade_guard"})
        return None

    return {
        "balance": balance, "account_summary": account_summary,
        "session": session, "macro": macro, "threshold": threshold,
        "banner": banner, "ops": ops,
        "news_penalty": news_penalty, "news_status": news_status,
        "effective_balance": get_effective_balance(balance, settings),
        "pyramid_possible": pyramid_possible,
    }


def _signal_phase(db, run_id, settings, alert, trader, history, now_sgt, today, demo, ctx) -> dict | None:
    """CPR signal evaluation, sizing, and margin guard.
    Returns ctx extended with execution parameters, or None (cycle aborted)."""

    session      = ctx["session"]
    macro        = ctx["macro"]
    banner       = ctx["banner"]
    ops          = ctx["ops"]
    sig_cache    = load_signal_cache()
    news_penalty = ctx["news_penalty"]
    news_status  = ctx["news_status"]
    balance      = ctx["balance"]
    account_summary = ctx["account_summary"]

    # ── Signal ────────────────────────────────────────────────────────────────
    engine = SignalEngine(demo=demo)
    score, direction, details, levels, position_usd = engine.analyze(asset=ASSET, settings=settings)

    raw_score        = score
    raw_position_usd = position_usd

    if news_penalty:
        score        = max(score + news_penalty, 0)
        position_usd = score_to_position_usd(score, settings)
        details      = details + f" | ⚠️ News penalty applied ({news_penalty:+d})"
        _nev = news_status.get("events", [])
        if not _nev and news_status.get("event"):
            _nev = [news_status["event"]]
        send_once_per_state(
            alert, ops, "ops_state", f"news_penalty:{news_penalty}:{today}",
            msg_news_penalty(
                event_names=[e.get("name", "") for e in _nev],
                penalty=news_penalty,
                score_after=score,
                score_before=raw_score,
                position_after=position_usd,
                position_before=raw_position_usd,
            ),
        )

    db.record_signal(
        {"pair": INSTRUMENT, "timeframe": settings.get("timeframe", "M15"), "side": direction,
         "score": score, "raw_score": raw_score,
         "news_penalty": news_penalty, "details": details, "levels": levels},
        timeframe=settings.get("timeframe", "M15"), run_id=run_id,
    )

    cpr_w = levels.get("cpr_width_pct", 0)

    def _send_signal_update(decision, reason, extra_payload=None):
        payload = _signal_payload(settings=settings, score=score, direction=direction, **(extra_payload or {}))
        msg = msg_signal_update(
            banner=banner, session=session, direction=direction,
            score=score, position_usd=position_usd, cpr_width_pct=cpr_w,
            detail_lines=details.split(" | "), news_penalty=news_penalty,
            raw_score=raw_score, decision=decision, reason=reason,
            cycle_minutes=int(settings.get("cycle_minutes", 5)),
            **payload,
        )
        if msg != sig_cache.get("last_signal_msg", ""):
            alert.send(msg)
            sig_cache.update({"score": score, "direction": direction, "last_signal_msg": msg})
            save_signal_cache(sig_cache)

    # ── No setup or below threshold ───────────────────────────────────────────
    # v4.2 fix: signal_threshold from settings is now enforced here.
    # Previously threshold was computed and stored in ctx but never compared
    # against score — meaning signal_threshold=4 had no effect.
    # Now score must meet the configured threshold (default 4) to proceed.
    threshold = ctx.get("threshold") or int(settings.get("signal_threshold", 4))

    if direction == "NONE" or position_usd <= 0:
        _send_signal_update("WATCHING", _clean_reason(details),
                            {"session_ok": True, "news_ok": True, "open_trade_ok": True})
        log.info("No trade. Score=%s dir=%s position=$%s", score, direction, position_usd, extra={"run_id": run_id})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="COMPLETED_NO_SIGNAL", score=score, direction=direction)
        db.finish_cycle(run_id, status="COMPLETED", summary={"signals": 1, "trades_placed": 0, "score": score, "direction": direction})
        return None

    if score < threshold:
        reason = f"Score {score}/6 below threshold {threshold} — no entry"
        _send_signal_update("WATCHING", reason,
                            {"session_ok": True, "news_ok": True, "open_trade_ok": True})
        log.info("Score below threshold: %d < %d — skipping", score, threshold, extra={"run_id": run_id})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_BELOW_THRESHOLD", score=score, direction=direction)
        db.finish_cycle(run_id, status="SKIPPED", summary={"signals": 1, "trades_placed": 0, "score": score, "direction": direction, "reason": f"below_threshold_{threshold}"})
        return None

    if not settings.get("trade_gold", True):
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_TRADE_GOLD_DISABLED")
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "trade_switch"})
        return None

    # ── Same-setup re-entry cooldown guard ────────────────────────────────────
    # v4.2: Prevents a duplicate trade when a CPR cache invalidation re-fetches
    # and re-fires the same signal within minutes (e.g. two S2 Extended Breakdown
    # SELLs at 16:00 and 16:05 as observed on 2026-03-19).
    _same_setup_cooldown = int(settings.get("min_reentry_wait_min") or settings.get("same_setup_cooldown_min", 15))
    if _same_setup_cooldown > 0 and history:
        _current_setup = levels.get("setup", "")
        _cutoff_dt     = now_sgt.replace(microsecond=0) - timedelta(minutes=_same_setup_cooldown)
        for _past in reversed(history[-20:]):
            _past_ts = _parse_sgt_timestamp(_past.get("timestamp_sgt"))
            if (
                _past_ts and _past_ts >= _cutoff_dt
                and _past.get("setup") == _current_setup
                and _past.get("status") == "FILLED"
                and not _past.get("is_pyramid")
            ):
                _mins_ago = int((now_sgt - _past_ts).total_seconds() / 60)
                _reason = (
                    f"Same-setup cooldown active: '{_current_setup}' last entered "
                    f"{_mins_ago}min ago (cooldown={_same_setup_cooldown}min)"
                )
                _send_signal_update("BLOCKED", _reason,
                                    {"session_ok": True, "news_ok": True, "open_trade_ok": True})
                log.info("Same-setup cooldown blocking entry: %s", _reason, extra={"run_id": run_id})
                update_runtime_state(
                    last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
                    status="SKIPPED_SAME_SETUP_COOLDOWN",
                )
                db.finish_cycle(run_id, status="SKIPPED", summary={
                    "stage": "same_setup_cooldown", "setup": _current_setup,
                    "mins_ago": _mins_ago, "reason": _reason,
                })
                return None

    # ── Position sizing ───────────────────────────────────────────────────────
    entry = levels.get("entry", 0)
    if entry <= 0:
        _, _, ask = trader.get_price(INSTRUMENT)
        entry = ask or 0

    sl_usd   = compute_sl_usd(levels, settings)
    tp_usd   = compute_tp_usd(levels, sl_usd, settings)
    rr_ratio = derive_rr_ratio(levels, sl_usd, tp_usd, settings)
    units    = calculate_units_from_position(position_usd, sl_usd)
    tp_pct   = (tp_usd / entry * 100) if entry > 0 else None

    if units <= 0:
        alert.send(msg_error("Position size = 0", f"position_usd=${position_usd} sl=${sl_usd:.2f}"))
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "position_sizing", "reason": "zero_units"})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_ZERO_UNITS")
        return None

    # v4.1: RR gate using the ACTUAL executed SL (not the signal-engine estimate).
    # signals.py validates RR against its own 0.25% fixed SL (~$11-12).
    # bot.py uses ATR-based SL ($15-40) which can be 3x larger, breaking the RR.
    _min_rr = float(settings.get("rr_ratio", 2.65))
    if rr_ratio < _min_rr:
        _rr_reason = (
            f"Actual R:R {rr_ratio:.2f} < minimum {_min_rr:.1f} "
            f"(sl=${sl_usd:.2f} tp=${tp_usd:.2f}) — trade skipped"
        )
        _send_signal_update("BLOCKED", _rr_reason,
                            {"rr_ratio": rr_ratio, "tp_pct": tp_pct,
                             "session_ok": True, "news_ok": True, "open_trade_ok": True})
        log.info("RR gate blocked entry: %s", _rr_reason, extra={"run_id": run_id})
        update_runtime_state(
            last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
            status="SKIPPED_RR_GATE",
            reason=_rr_reason,
        )
        db.finish_cycle(run_id, status="SKIPPED",
                        summary={"stage": "rr_gate", "rr_ratio": rr_ratio,
                                 "min_rr": _min_rr, "reason": _rr_reason})
        return None

    # v4.1: Consecutive-direction loss guard.
    # After N consecutive SL hits in the same direction, require an elevated
    # signal score before re-entering. Prevents cascading losses in trending
    # markets where the bot repeatedly fades the move.
    _guard_n    = int(settings.get("consecutive_sl_guard", 2))
    _sl_streak  = _count_consecutive_sl(history, direction)

    # v5.1 — time-based direction cooldown: after direction guard fires, block
    # same-direction re-entry for sl_direction_cooldown_min minutes regardless of score.
    _dir_cooldown_min = int(settings.get("sl_direction_cooldown_min", 60))
    if _dir_cooldown_min > 0:
        _rt = load_json(RUNTIME_STATE_FILE, {})
        _dir_block_key = f"direction_block_{direction.lower()}"
        _dir_block_until = _parse_sgt_timestamp(_rt.get(_dir_block_key))
        if _dir_block_until and now_sgt < _dir_block_until:
            _remaining = int((_dir_block_until - now_sgt).total_seconds() / 60)
            _cooldown_reason = (
                f"Direction cooldown active — {direction} blocked for {_remaining}min more "
                f"(after {_sl_streak} consecutive SL streak)"
            )
            _send_signal_update("BLOCKED", _cooldown_reason,
                                {"rr_ratio": rr_ratio, "tp_pct": tp_pct,
                                 "session_ok": True, "news_ok": True, "open_trade_ok": True})
            log.info("Direction time-block active: %s", _cooldown_reason, extra={"run_id": run_id})
            update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
                                 status="SKIPPED_DIRECTION_COOLDOWN", reason=_cooldown_reason)
            db.finish_cycle(run_id, status="SKIPPED",
                            summary={"stage": "direction_cooldown", "direction": direction,
                                     "remaining_min": _remaining})
            return None

    if _sl_streak >= _guard_n:
        _required  = int(settings.get("signal_threshold", 4)) + 1
        if score < _required:
            _guard_reason = (
                f"{_sl_streak} consecutive SL hits in {direction} direction — "
                f"score {score}/6 < elevated threshold {_required} required"
            )
            _send_signal_update("BLOCKED", _guard_reason,
                                {"rr_ratio": rr_ratio, "tp_pct": tp_pct,
                                 "session_ok": True, "news_ok": True, "open_trade_ok": True})
            log.info("Direction guard blocked entry: %s", _guard_reason, extra={"run_id": run_id})
            # v5.1 — set time-based cooldown when direction guard fires
            if _dir_cooldown_min > 0:
                _block_until = now_sgt + timedelta(minutes=_dir_cooldown_min)
                _dir_block_key = f"direction_block_{direction.lower()}"
                save_json(RUNTIME_STATE_FILE, {
                    **load_json(RUNTIME_STATE_FILE, {}),
                    _dir_block_key: _block_until.strftime("%Y-%m-%d %H:%M:%S"),
                })
                log.info("Direction cooldown set: %s blocked until %s SGT",
                         direction, _block_until.strftime("%H:%M"), extra={"run_id": run_id})
            update_runtime_state(
                last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
                status="SKIPPED_DIRECTION_GUARD",
                reason=_guard_reason,
            )
            db.finish_cycle(run_id, status="SKIPPED",
                            summary={"stage": "direction_guard", "streak": _sl_streak,
                                     "score": score, "required": _required, "reason": _guard_reason})
            return None
        log.info(
            "Direction guard active (%d streak) — score %d meets elevated threshold %d, proceeding",
            _sl_streak, score, _required, extra={"run_id": run_id},
        )

    signal_blockers = list(levels.get("signal_blockers") or [])
    if signal_blockers:
        _send_signal_update("BLOCKED", signal_blockers[0],
                            {"rr_ratio": rr_ratio, "tp_pct": tp_pct, "session_ok": True, "news_ok": True, "open_trade_ok": True, "margin_ok": None})
        log.info("Signal blocked before execution: %s", signal_blockers[0], extra={"run_id": run_id})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_SIGNAL_BLOCKED", reason=signal_blockers[0])
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "signal_validation", "reason": signal_blockers[0]})
        return None

    # ── Margin guard ──────────────────────────────────────────────────────────
    # account_summary already fetched at login — no second OANDA call needed
    margin_available  = float(account_summary.get("margin_available", balance or 0) or 0)
    price_for_margin  = entry if entry > 0 else float(levels.get("current_price", entry) or 0)
    units, margin_info = apply_margin_guard(
        trader=trader, instrument=INSTRUMENT,
        requested_units=units, entry_price=price_for_margin,
        free_margin=margin_available, settings=settings,
    )
    if margin_info.get("status") == "ADJUSTED":
        log.warning(
            "Margin protection adjusted %.2f → %.2f units | free_margin=%.2f required=%.2f",
            float(margin_info.get("requested_units", 0)), float(margin_info.get("final_units", 0)),
            float(margin_info.get("free_margin", 0)), float(margin_info.get("required_margin", 0)),
        )
        alert.send(msg_margin_adjustment(
            instrument=INSTRUMENT,
            requested_units=float(margin_info.get("requested_units", 0)),
            adjusted_units=float(margin_info.get("final_units", 0)),
            free_margin=float(margin_info.get("free_margin", 0)),
            required_margin=float(margin_info.get("required_margin", 0)),
            reason=str(margin_info.get("reason", "margin_guard")),
        ))
    if units <= 0:
        _send_signal_update("BLOCKED", "Insufficient margin after safety checks",
                            {"rr_ratio": rr_ratio, "tp_pct": tp_pct, "session_ok": True, "news_ok": True, "open_trade_ok": True, "margin_ok": False})
        alert.send(msg_error(
            "Insufficient margin — trade skipped",
            f"free_margin=${margin_available:.2f} required=${float(margin_info.get('required_margin', 0)):.2f}",
        ))
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "margin_cap", "reason": "insufficient_margin"})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_MARGIN")
        return None

    stop_pips, tp_pips = compute_sl_tp_pips(sl_usd, tp_usd)
    reward_usd = round(units * tp_usd, 2)

    # ── Spread guard ──────────────────────────────────────────────────────────
    mid, bid, ask = trader.get_price(INSTRUMENT)
    if mid is None:
        alert.send(msg_error("Cannot fetch price", "OANDA pricing endpoint returned None"))
        db.finish_cycle(run_id, status="FAILED", summary={"stage": "pricing"})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="FAILED_PRICING")
        return None

    spread_pips  = round(abs(ask - bid) / 0.01)
    spread_limit = int(settings.get("spread_limits", {}).get(macro, settings.get("max_spread_pips", 150)))
    if spread_pips > spread_limit:
        _send_signal_update("BLOCKED", f"Spread too high ({spread_pips} > {spread_limit} pips)",
                            {"rr_ratio": rr_ratio, "tp_pct": tp_pct, "spread_pips": spread_pips,
                             "spread_limit": spread_limit, "session_ok": True, "news_ok": True, "open_trade_ok": True, "margin_ok": True})
        send_once_per_state(alert, ops, "spread_state", f"spread:{macro}:{spread_pips}",
                            msg_spread_skip(banner, session, spread_pips, spread_limit))
        db.finish_cycle(run_id, status="SKIPPED", summary={"stage": "spread_guard"})
        update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="SKIPPED_SPREAD_GUARD")
        return None

    _send_signal_update("READY", "All must-pass checks satisfied",
                        {"rr_ratio": rr_ratio, "tp_pct": tp_pct, "spread_pips": spread_pips,
                         "spread_limit": spread_limit, "session_ok": True, "news_ok": True, "open_trade_ok": True, "margin_ok": True})

    # ── AI Reasoning Filter (v5.2) ─────────────────────────────────────────────
    # Claude evaluates every qualifying signal before order placement.
    # HIGH confidence → lot_multiplier 2–3x | LOW confidence → trade blocked.
    lot_multiplier = 1
    ai_confidence  = "DISABLED"
    ai_reason      = "AI reasoning disabled"

    if settings.get("ai_reasoning", True):
        try:
            log.info("AI reasoning layer — evaluating trade...", extra={"run_id": run_id})
            _wins_today   = sum(1 for t in history
                                if t.get("status") == "FILLED"
                                and (t.get("realized_pnl_usd") or 0) > 0
                                and (t.get("timestamp_sgt") or "")[:10] == today)
            _losses_today = sum(1 for t in history
                                if t.get("status") == "FILLED"
                                and (t.get("realized_pnl_usd") or 0) < 0
                                and (t.get("timestamp_sgt") or "")[:10] == today)
            _last_loss = next(
                (t for t in reversed(history)
                 if t.get("status") == "FILLED" and (t.get("realized_pnl_usd") or 0) < 0),
                {},
            )
            _last_win = next(
                (t for t in reversed(history)
                 if t.get("status") == "FILLED" and (t.get("realized_pnl_usd") or 0) > 0),
                {},
            )
            # Build a concise H4 trend string from levels if available
            _h4_trend = "UNKNOWN"
            if levels.get("h4_trend"):
                _h4_trend = str(levels["h4_trend"])
            elif levels.get("h1_ema_trend"):
                _h4_trend = str(levels["h1_ema_trend"])
            elif levels.get("h1_trend_bullish") is not None:
                # Fall back to H1 EMA trend as a proxy for macro direction
                _h4_trend = "BULLISH" if levels["h1_trend_bullish"] else "BEARISH"

            ai_result = ai_should_trade(
                direction       = direction,
                score           = score,
                price           = entry,
                signal_details  = details[:400],
                wins_today      = _wins_today,
                losses_today    = _losses_today,
                last_loss_entry = float(_last_loss.get("entry", 0) or 0),
                last_loss_exit  = float(_last_loss.get("sl_price", 0) or 0),
                last_loss_dir   = _last_loss.get("direction", ""),
                last_win_exit   = float(_last_win.get("tp_price", 0) or 0),
                recent_candles  = [],   # optional: wire up H1 closes for extra context
                session         = session or "",
                h4_trend        = _h4_trend,
                is_asian        = (macro == "Asian"),
            )

            if not ai_result["allow"]:
                ai_block_msg = (
                    f"🤖 AI blocked — {ai_result['reason']} "
                    f"(confidence={ai_result['confidence']})"
                )
                _send_signal_update("BLOCKED", ai_block_msg,
                                    {"rr_ratio": rr_ratio, "tp_pct": tp_pct,
                                     "spread_pips": spread_pips, "spread_limit": spread_limit,
                                     "session_ok": True, "news_ok": True,
                                     "open_trade_ok": True, "margin_ok": True})
                log.info("AI BLOCKED entry: %s", ai_result["reason"], extra={"run_id": run_id})
                alert.send(ai_block_msg)
                update_runtime_state(
                    last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
                    status="SKIPPED_AI_BLOCK",
                    reason=ai_result["reason"],
                )
                db.finish_cycle(run_id, status="SKIPPED",
                                summary={"stage": "ai_reasoning",
                                         "reason": ai_result["reason"],
                                         "confidence": ai_result["confidence"]})
                return None

            lot_multiplier = int(ai_result.get("lot_multiplier", 1))
            lot_multiplier = max(1, min(3, lot_multiplier))
            ai_confidence  = ai_result.get("confidence", "MEDIUM")
            ai_reason      = ai_result.get("reason", "")
            log.info(
                "AI APPROVED | confidence=%s | lot_multiplier=%dx | reason=%s",
                ai_confidence, lot_multiplier, ai_reason,
                extra={"run_id": run_id},
            )
            # Scale units by lot_multiplier (AI can increase size on HIGH-confidence setups)
            if lot_multiplier > 1:
                units = round(units * lot_multiplier, 2)
                log.info("Units scaled by AI lot_multiplier: %dx → %.2f units",
                         lot_multiplier, units, extra={"run_id": run_id})

        except Exception as _ai_exc:
            log.warning("AI reasoning error — proceeding with normal size: %s",
                        _ai_exc, extra={"run_id": run_id})
            lot_multiplier = 1
            ai_confidence  = "ERROR"
            ai_reason      = str(_ai_exc)[:80]

    ctx.update({
        "score": score, "raw_score": raw_score, "direction": direction,
        "details": details, "levels": levels, "position_usd": position_usd,
        "entry": entry, "sl_usd": sl_usd, "tp_usd": tp_usd,
        "rr_ratio": rr_ratio, "units": units, "stop_pips": stop_pips,
        "tp_pips": tp_pips, "reward_usd": reward_usd, "cpr_w": cpr_w,
        "spread_pips": spread_pips, "bid": bid, "ask": ask,
        "margin_available": margin_available, "price_for_margin": price_for_margin,
        "margin_info": margin_info,
        "lot_multiplier": lot_multiplier,
        "ai_confidence": ai_confidence,
        "ai_reason": ai_reason,
    })
    return ctx


def _execution_phase(db, run_id, settings, alert, trader, history, now_sgt, today, demo, ctx):
    """Places the order and persists the trade record."""

    session     = ctx["session"]
    macro       = ctx["macro"]
    banner      = ctx["banner"]
    score       = ctx["score"]
    raw_score   = ctx["raw_score"]
    direction   = ctx["direction"]
    details     = ctx["details"]
    levels      = ctx["levels"]
    position_usd = ctx["position_usd"]
    entry       = ctx["entry"]
    sl_usd      = ctx["sl_usd"]
    tp_usd      = ctx["tp_usd"]
    rr_ratio    = ctx["rr_ratio"]
    units       = ctx["units"]
    stop_pips   = ctx["stop_pips"]
    tp_pips     = ctx["tp_pips"]
    reward_usd  = ctx["reward_usd"]
    cpr_w       = ctx["cpr_w"]
    spread_pips = ctx["spread_pips"]
    bid         = ctx["bid"]
    ask         = ctx["ask"]
    margin_available  = ctx["margin_available"]
    price_for_margin  = ctx["price_for_margin"]
    margin_info       = ctx["margin_info"]
    effective_balance = ctx["effective_balance"]
    news_penalty      = ctx["news_penalty"]

    sl_price, tp_price = compute_sl_tp_prices(entry, direction, sl_usd, tp_usd)

    record = {
        "timestamp_sgt":        now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
        "mode":                 "DEMO" if demo else "LIVE",
        "instrument":           INSTRUMENT,
        "direction":            direction,
        "setup":                levels.get("setup", ""),
        "session":              session,
        "window":               get_window_key(session),
        "macro_session":        macro,
        "score":                score,
        "raw_score":            raw_score,
        "news_penalty":         news_penalty,
        "position_usd":         position_usd,
        "entry":                round(entry, 2),
        "sl_price":             sl_price,
        "tp_price":             tp_price,
        "size":                 units,
        "cpr_width_pct":        cpr_w,
        "sl_usd":               round(sl_usd, 2),
        "tp_usd":               round(tp_usd, 2),
        "estimated_risk_usd":   round(position_usd, 2),
        "estimated_reward_usd": round(reward_usd, 2),
        "spread_pips":          spread_pips,
        "stop_pips":            stop_pips,
        "tp_pips":              tp_pips,
        "levels":               levels,
        "details":              details,
        "trade_id":             None,
        "status":               "FAILED",
        "realized_pnl_usd":     None,
    }

    # ── Place order ───────────────────────────────────────────────────────────
    # v4.1: trailing stop at 0.5x SL pips — server-enforced by OANDA, no polling needed
    _trail_mult = float(settings.get("trailing_stop_atr_mult", 0.5))
    _trail_pips = round(stop_pips * _trail_mult) if _trail_mult > 0 else None

    result = trader.place_order(
        instrument=INSTRUMENT, direction=direction,
        size=units, stop_distance=stop_pips, limit_distance=tp_pips,
        bid=bid, ask=ask,
        trailing_distance_pips=_trail_pips,
    )

    if not result.get("success"):
        err = result.get("error", "Unknown")
        retry_attempted = False
        if settings.get("auto_scale_on_margin_reject", True) and "MARGIN" in str(err).upper():
            retry_attempted = True
            retry_safety     = float(settings.get("margin_retry_safety_factor", 0.4))
            retry_specs      = trader.get_instrument_specs(INSTRUMENT)
            retry_margin_rate = max(
                float(retry_specs.get("marginRate", 0.05) or 0.05),
                float(settings.get("xau_margin_rate_override", 0.05) or 0.05) if INSTRUMENT == "XAU_USD" else 0.0,
            )
            retry_units = trader.normalize_units(
                INSTRUMENT,
                (margin_available * retry_safety) / max(price_for_margin * retry_margin_rate, 1e-9),
            )
            if 0 < retry_units < units:
                alert.send(msg_margin_adjustment(
                    instrument=INSTRUMENT,
                    requested_units=units,
                    adjusted_units=retry_units,
                    free_margin=margin_available,
                    required_margin=trader.estimate_required_margin(INSTRUMENT, retry_units, price_for_margin),
                    reason="broker_margin_reject_retry",
                ))
                retry_result = trader.place_order(
                    instrument=INSTRUMENT, direction=direction,
                    size=retry_units, stop_distance=stop_pips, limit_distance=tp_pips,
                    bid=bid, ask=ask,
                    trailing_distance_pips=_trail_pips,
                )
                if retry_result.get("success"):
                    result = retry_result
                    units  = retry_units
                    record["size"] = units
                    record["estimated_reward_usd"] = round(units * tp_usd, 2)

        if not result.get("success"):
            err = result.get("error", "Unknown")
            alert.send(msg_order_failed(
                direction, INSTRUMENT, units, err,
                free_margin=margin_available,
                required_margin=trader.estimate_required_margin(INSTRUMENT, units, price_for_margin),
                retry_attempted=retry_attempted,
            ))
            log.error("Order failed: %s", err, extra={"run_id": run_id})

    if result.get("success"):
        record["trade_id"] = result.get("trade_id")
        record["status"]   = "FILLED"
        fill_price = result.get("fill_price")
        if fill_price and fill_price > 0:
            actual_entry           = fill_price
            record["entry"]        = round(actual_entry, 2)
            record["signal_entry"] = round(entry, 2)
            record["sl_price"]     = round(actual_entry - sl_usd if direction == "BUY" else actual_entry + sl_usd, 2)
            record["tp_price"]     = round(actual_entry + tp_usd if direction == "BUY" else actual_entry - tp_usd, 2)
        else:
            actual_entry = entry

        if ctx.get("is_pyramid"):
            alert.send(msg_pyramid_opened(
                banner=banner, direction=direction,
                session=session, fill_price=record["entry"], signal_price=entry,
                sl_price=record["sl_price"], tp_price=record["tp_price"],
                sl_usd=sl_usd, tp_usd=tp_usd, units=units,
                rr_ratio=rr_ratio, spread_pips=spread_pips, score=score,
                t1_trade_id=ctx.get("pyramid_trade_id", ""),
                t1_unrealized_pnl=ctx.get("pyramid_unrealized_pnl", 0.0),
                pyramid_max_risk=ctx.get("position_usd", 50),
                demo=demo,
            ))
        else:
            alert.send(msg_trade_opened(
                banner=banner, direction=direction, setup=levels.get("setup", ""),
                session=session, fill_price=record["entry"], signal_price=entry,
                sl_price=record["sl_price"], tp_price=record["tp_price"],
                sl_usd=sl_usd, tp_usd=tp_usd, units=units, position_usd=position_usd,
                rr_ratio=rr_ratio, cpr_width_pct=cpr_w, spread_pips=spread_pips,
                score=score, balance=effective_balance, demo=demo,
                news_penalty=news_penalty, raw_score=raw_score,
                free_margin=margin_info.get("free_margin"),
                required_margin=trader.estimate_required_margin(INSTRUMENT, units, price_for_margin),
                margin_mode=("RETRIED" if record["size"] != float(margin_info.get("final_units", record["size"])) else margin_info.get("status", "NORMAL")),
                margin_usage_pct=(
                    (trader.estimate_required_margin(INSTRUMENT, units, price_for_margin) / float(margin_info.get("free_margin", 0)) * 100)
                    if float(margin_info.get("free_margin", 0)) > 0 else None
                ),
            ))
        log.info("Trade placed: %s", record, extra={"run_id": run_id})

    history.append(record)
    save_history(history)
    db.record_trade_attempt(
        {"pair": INSTRUMENT, "timeframe": settings.get("timeframe", "M15"), "side": direction, "score": score, **record},
        ok=bool(result.get("success")), note=result.get("error", "trade placed"),
        broker_trade_id=record.get("trade_id"), run_id=run_id,
    )
    db.upsert_state("last_trade_attempt", {
        "run_id": run_id, "success": bool(result.get("success")),
        "trade_id": record.get("trade_id"), "timestamp_sgt": record["timestamp_sgt"],
    })
    update_runtime_state(
        last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"),
        status="COMPLETED", score=score, direction=direction,
        trade_status=record["status"],
    )
    db.finish_cycle(run_id, status="COMPLETED", summary={
        "signals": 1, "trades_placed": int(bool(result.get("success"))),
        "score": score, "direction": direction, "trade_status": record["status"],
    })


def run_bot_cycle():
    """Thin orchestrator — sets up shared objects and delegates to the three phases."""
    global _startup_reconcile_done

    settings = validate_settings(load_settings())
    # v4.2 — instrument driven by settings, not module constant
    global INSTRUMENT, ASSET
    INSTRUMENT = settings.get("instrument", INSTRUMENT)
    ASSET = settings.get("instrument_display", ASSET).replace("/", "")
    db       = Database()
    demo     = settings.get("demo_mode", True)
    alert    = TelegramAlert()
    trader   = OandaTrader(demo=demo)
    history  = load_history()
    now_sgt  = datetime.now(SGT)
    # v4.2 — use 08:00 SGT as the trading-day boundary instead of calendar
    # midnight. The bot's active window is 08:00–23:00 SGT; before 08:00 SGT
    # any trade belongs to the previous trading day's cap counter.
    day_start_hour = int(validate_settings(load_settings()).get("trading_day_start_hour_sgt", 8))
    today    = get_trading_day(now_sgt, day_start_hour)

    # v4.2 — startup OANDA reconcile: runs once per process start to re-sync
    # today's closed trades from the broker before the first cycle.
    # Ensures the loss cap sees the correct count even after a mid-day redeploy
    # where history.json may be missing trades closed since the last save.
    if not _startup_reconcile_done:
        try:
            recon = startup_oanda_reconcile(trader, history, INSTRUMENT, today, now_sgt)
            if recon["injected"] or recon["backfilled"]:
                save_history(history)
                log.info(
                    "Startup reconcile: injected=%s backfilled=%s — history saved",
                    recon["injected"], recon["backfilled"],
                )
                if recon["injected"]:
                    alert.send(
                        f"♻️ Startup reconcile injected {len(recon['injected'])} missing "
                        f"closed trade(s) into history before first cycle.\n"
                        f"Trade IDs: {', '.join(recon['injected'])}"
                    )
        except Exception as _recon_exc:
            log.warning("Startup reconcile failed (non-fatal): %s", _recon_exc)
        finally:
            _startup_reconcile_done = True

    with db.cycle() as run_id:
        try:
            ctx = _guard_phase(db, run_id, settings, alert, trader, history, now_sgt, today, demo)
            if ctx is None:
                return

            ctx = _signal_phase(db, run_id, settings, alert, trader, history, now_sgt, today, demo, ctx)
            if ctx is None:
                return

            # v4.2 — Pyramid phase: only when guard flagged a possible add
            if ctx.get("pyramid_possible"):
                ctx = _pyramid_phase(db, run_id, settings, alert, trader, history, now_sgt, today, demo, ctx)
                if ctx is None:
                    return

            _execution_phase(db, run_id, settings, alert, trader, history, now_sgt, today, demo, ctx)

        except Exception as exc:
            update_runtime_state(last_cycle_finished=now_sgt.strftime("%Y-%m-%d %H:%M:%S"), status="FAILED", error=str(exc))
            raise


def main():
    return run_bot_cycle()


if __name__ == "__main__":
    main()
