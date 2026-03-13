"""
OANDA Trading Bot - Demo Account 2
====================================
Strategy: Mean Reversion
- Bollinger Bands (20, 2) entry signals
- RSI extremes confirmation (< 35 BUY / > 65 SELL)
- ATR ranging filter (skip if trending)
- Dynamic TP = Middle Bollinger Band
- Same risk controls as Demo 1
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
import pytz

from oanda_trader import OandaTrader
from signals import SignalEngine
from telegram_alert import TelegramAlert
from calendar_filter import EconomicCalendar

# Safe logging - never expose API keys
class SafeFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        key = os.environ.get("OANDA_API_KEY", "")
        if key and key in msg:
            msg = msg.replace(key, "***")
        return msg

handler      = logging.StreamHandler()
handler.setFormatter(SafeFormatter("%(asctime)s | %(levelname)s | %(message)s"))
file_handler = logging.FileHandler("performance_log.txt")
file_handler.setFormatter(SafeFormatter("%(asctime)s | %(levelname)s | %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[handler, file_handler])
log = logging.getLogger(__name__)

# ── ASSET CONFIGURATION ──────────────────────────────────────────────────────
# MEAN REVERSION BEST PAIRS (5-year analysis):
# AUD/USD  → ranges for weeks, China-sensitive, slow grinding = perfect MR
# EUR/GBP  → extremely tight range 0.84-0.92, always reverts = perfect MR
# USD/CHF  → SNB keeps it rangy, mirrors EUR/USD inversely = perfect MR
# XAG/USD  → follows gold but spikier, reversals common = perfect MR
# MAX TRADE DURATION = 4 hours for day trading
# Force close if trade runs longer than max_hours
# ── FAST TRADE CONFIG ────────────────────────────────────────────────────────
# Target: trades complete within 1 hour max
# Logic:  Small TP/SL = price hits target quickly
# M15 signals used = faster entry/exit timing
#
# Per trade P&L at 0.10 lots ($1/pip):
# AUD/USD: SL=8pips=$8   TP=12pips=$12  R:R=1:1.5
# EUR/GBP: SL=6pips=$6   TP=9pips=$9    R:R=1:1.5
# USD/CHF: SL=8pips=$8   TP=12pips=$12  R:R=1:1.5
# XAG/USD: SL=30pips=$3  TP=45pips=$4.5 R:R=1:1.5  (100oz, pip=0.01)

MAX_TRADE_HOURS = 1   # Hard limit: 1 hour max per trade

ASSETS = {
    "AUD_USD": {
        "instrument":  "AUD_USD",
        "asset":       "AUDUSD",
        "emoji":       "🦘",
        "setting":     "trade_audusd",
        "stop_pips":   8,     # $8 max loss
        "tp_pips":     12,    # $12 max profit → R:R 1:1.5
        "pip":         0.0001,
        "precision":   5,
        "min_atr":     0.0003,
        "max_hours":   1,
        "session_hours": [(6, 11), (14, 17)],
    },
    "EUR_GBP": {
        "instrument":  "EUR_GBP",
        "asset":       "EURGBP",
        "emoji":       "🇪🇺",
        "setting":     "trade_eurgbp",
        "stop_pips":   6,     # $6 max loss  (tight range pair = small moves)
        "tp_pips":     9,     # $9 max profit → R:R 1:1.5
        "pip":         0.0001,
        "precision":   5,
        "min_atr":     0.0002,
        "max_hours":   1,
        "session_hours": [(14, 19)],
    },
    "USD_CHF": {
        "instrument":  "USD_CHF",
        "asset":       "USDCHF",
        "emoji":       "🇨🇭",
        "setting":     "trade_usdchf",
        "stop_pips":   8,     # $8 max loss
        "tp_pips":     12,    # $12 max profit → R:R 1:1.5
        "pip":         0.0001,
        "precision":   5,
        "min_atr":     0.0003,
        "max_hours":   1,
        "session_hours": [(14, 23), (0, 1)],
    },
    "XAG_USD": {
        "instrument":  "XAG_USD",
        "asset":       "XAGUSD",
        "emoji":       "🥈",
        "setting":     "trade_silver",
        "stop_pips":   30,    # 30 × 0.01 × 100oz = $3 max loss
        "tp_pips":     45,    # 45 × 0.01 × 100oz = $4.5 max profit → R:R 1:1.5
        "pip":         0.01,
        "precision":   2,
        "lot_size":    100,
        "min_atr":     0.05,
        "max_hours":   1,
        "session_hours": [(14, 23), (0, 1)],
    },
    "EUR_USD": {
        "instrument":  "EUR_USD",
        "asset":       "EURUSD",
        "emoji":       "🇪🇺💵",
        "setting":     "trade_eurusd",
        "stop_pips":   8,     # $8 max loss — tight SL for fast exit
        "tp_pips":     12,    # $12 max profit → R:R 1:1.5
        "pip":         0.0001,
        "precision":   5,
        "min_atr":     0.0003,
        "max_hours":   1,     # 1hr hard limit
        "session_hours": [(14, 22)],  # London + NY only (best EUR/USD hours)
    },
}

def load_settings():
    default = {
        "max_trades_day":    4,       # Target 4 trades/day for 60% win rate
        "max_daily_loss":    40.0,
        "signal_threshold":  4,
        "demo_mode":         True,
        "trade_audusd":      True,
        "trade_eurgbp":      True,
        "trade_usdchf":      True,
        "trade_silver":      True,
        "trade_eurusd":      True,   # EUR/USD - most liquid pair added back!
        "fixed_lot_size":     0.10,
        "max_consec_losses": 2,
        "max_spread_pips":   2,
        "strategy":          "mean_reversion",
    }
    try:
        with open("settings.json") as f:
            saved = json.load(f)
            default.update(saved)
    except FileNotFoundError:
        with open("settings.json", "w") as f:
            json.dump(default, f, indent=2)
    return default

def calc_position_size(pip_value, config=None):
    """
    Fixed lot sizes:
    Forex   = 10,000 units (0.10 lots) → $1 per pip
    Silver  = 100 oz                   → $1 per pip (pip=0.01)
    """
    # Use lot_size from config if provided
    if config and "lot_size" in config:
        return config["lot_size"]
    if pip_value <= 0.0001:
        return 10000   # 0.10 lots forex → $1/pip
    elif pip_value == 0.01:
        return 100     # 100 oz silver  → $1/pip
    else:
        return 10000

def get_bb_tp_pips(trader, instrument, direction, pip, precision):
    """
    Dynamic TP = Middle Bollinger Band (20 EMA on H1)
    This is the CORE of mean reversion - we target the mean!
    """
    try:
        import math
        url    = trader.base_url + "/v3/instruments/" + instrument + "/candles"
        params = {"count": "50", "granularity": "H1", "price": "M"}
        r      = requests.get(url, headers=trader.headers, params=params, timeout=10)
        if r.status_code != 200:
            return None

        candles = r.json()["candles"]
        closes  = [float(c["mid"]["c"]) for c in candles if c["complete"]]
        if len(closes) < 20:
            return None

        # Calculate BB middle (20 SMA)
        recent = closes[-20:]
        middle = sum(recent) / 20

        # Current price
        price, _, _ = trader.get_price(instrument)
        if not price:
            return None

        # Distance from current price to middle band in pips
        distance_pips = abs(price - middle) / pip
        log.info(instrument + " BB middle=" + str(round(middle, precision)) +
                 " price=" + str(round(price, precision)) +
                 " TP_pips=" + str(round(distance_pips, 1)))

        # Minimum TP = 10 pips (not worth trading if too close to mean)
        if distance_pips < 10:
            return None

        return round(distance_pips)

    except Exception as e:
        log.warning("BB TP calc error: " + str(e))
        return None

def check_spread(trader, instrument, max_spread_pips, pip):
    """Skip trade if spread too wide"""
    try:
        bid, ask, _ = trader.get_price(instrument)
        if bid is None:
            return True, 0
        bid_val, ask_val = ask, bid  # get_price returns mid, bid, ask
        # Re-fetch properly
        mid, bid_val, ask_val = trader.get_price(instrument)
        spread_pips = (ask_val - bid_val) / pip
        log.info(instrument + " spread=" + str(round(spread_pips, 1)) + " pips")
        if spread_pips > max_spread_pips:
            log.warning(instrument + " spread too wide: " + str(round(spread_pips, 1)))
            return False, spread_pips
        return True, spread_pips
    except Exception as e:
        log.warning("Spread check error: " + str(e))
        return True, 0

def is_in_cooldown(today, instrument):
    cooldowns = today.get("cooldowns", {})
    if instrument not in cooldowns:
        return False
    last_loss  = datetime.fromisoformat(cooldowns[instrument])
    wait_until = last_loss + timedelta(minutes=30)
    now_utc    = datetime.utcnow()
    if now_utc < wait_until:
        mins = int((wait_until - now_utc).seconds / 60)
        log.info(instrument + " cooldown " + str(mins) + " mins left")
        return True
    return False

def set_cooldown(today, instrument):
    if "cooldowns" not in today:
        today["cooldowns"] = {}
    today["cooldowns"][instrument] = datetime.utcnow().isoformat()

def run_bot(state=None):
    log.info("OANDA Bot Demo 2 - Mean Reversion starting!")
    settings = load_settings()
    sg_tz    = pytz.timezone("Asia/Singapore")
    now      = datetime.now(sg_tz)
    alert    = TelegramAlert()
    hour     = now.hour

    # Session detection - expanded to cover ALL pair sessions
    # Tokyo: 6am-12pm SGT (AUD/USD, USD/JPY best)
    # London: 2pm-8pm SGT (EUR, GBP pairs best)
    # NY: 8pm-11pm SGT (USD pairs best)
    tokyo_session  = (6 <= hour <= 11)
    london_session = (14 <= hour <= 19)
    ny_session     = (20 <= hour <= 23)
    late_ny        = (0 <= hour <= 1)
    good_session   = tokyo_session or london_session or ny_session or late_ny

    if 14 <= hour <= 17:
        session = "London Open 🇬🇧"
    elif 20 <= hour <= 23:
        session = "London+NY Overlap 🔥"
    elif 18 <= hour <= 19:
        session = "London Session 🇬🇧"
    elif 0 <= hour <= 1:
        session = "NY Late Session 🇺🇸"
    elif 6 <= hour <= 11:
        session = "Tokyo Session 🇯🇵"
    else:
        session = "Off-hours (SKIP)"

    # Weekend check
    if now.weekday() == 5:
        alert.send("Saturday - markets closed! Bot 2 resumes Monday 5am SGT")
        return
    if now.weekday() == 6 and hour < 5:
        alert.send("Sunday early - markets open at 5am SGT")
        return

    # Login
    trader = OandaTrader(demo=settings["demo_mode"])
    if not trader.login():
        alert.send("DEMO 2 Login FAILED! Check OANDA_API_KEY and OANDA_ACCOUNT_ID secrets")
        return

    current_balance = trader.get_balance()
    mode            = "DEMO2" if settings["demo_mode"] else "LIVE"

    # ── STATE MANAGEMENT (in-memory for Railway, file for GitHub Actions) ──
    trade_log = "trades_" + now.strftime("%Y%m%d") + ".json"
    if state is not None:
        # Railway mode: use in-memory state passed from main.py
        today = state
        if "start_balance" not in today:
            today["start_balance"] = current_balance
            log.info("New day! Start balance: $" + str(round(current_balance, 2)))
    else:
        # GitHub Actions mode: use file-based state
        try:
            with open(trade_log) as f:
                today = json.load(f)
        except FileNotFoundError:
            today = {
                "trades":        0,
                "start_balance": current_balance,
                "daily_pnl":     0.0,
                "stopped":       False,
                "wins":          0,
                "losses":        0,
                "consec_losses": 0,
                "cooldowns":     {}
            }
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)
            log.info("New day! Start balance: $" + str(round(current_balance, 2)))

    # PnL tracking
    start_balance = today.get("start_balance", current_balance)
    open_pnl      = sum(
        trader.check_pnl(trader.get_position(n))
        for n in ASSETS if trader.get_position(n)
    )
    realized_pnl = current_balance - start_balance
    total_pnl    = realized_pnl + open_pnl
    pl_sgd       = realized_pnl * 1.35
    pnl_emoji    = "✅" if realized_pnl >= 0 else "❌"

    today["daily_pnl"] = realized_pnl
    with open(trade_log, "w") as f:
        json.dump(today, f, indent=2)

    # Daily loss protection
    if today.get("stopped"):
        alert.send(
            "🔄 DEMO 2 Bot stopped for today\n"
            "Daily limit hit!\n"
            "Realized: $" + str(round(realized_pnl, 2)) + " USD\n"
            "Resumes tomorrow!"
        )
        return

    if realized_pnl <= -settings["max_daily_loss"]:
        today["stopped"] = True
        if state is None:
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)
        alert.send(
            "🔴 DEMO 2 DAILY LOSS LIMIT HIT!\n"
            "Loss: $" + str(abs(round(realized_pnl, 2))) + " USD\n"
            "Limit: $" + str(settings["max_daily_loss"]) + " USD\n"
            "Bot stopped! Resumes tomorrow."
        )
        return

    # Consecutive loss protection
    consec = today.get("consec_losses", 0)
    if consec >= settings.get("max_consec_losses", 2):
        today["stopped"] = True
        if state is None:
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)
        alert.send(
            "⛔ DEMO 2: 2 CONSECUTIVE LOSSES!\n"
            "Protecting capital!\n"
            "Stopped for today.\n"
            "Realized: $" + str(round(realized_pnl, 2)) + " USD\n"
            "Resumes tomorrow!"
        )
        return

    # Max trades
    if today["trades"] >= settings["max_trades_day"]:
        alert.send(
            "✅ DEMO 2 Max trades reached!\n"
            "Trades: " + str(today["trades"]) + "/" + str(settings["max_trades_day"]) + "\n"
            "Realized: $" + str(round(realized_pnl, 2)) + " USD " + pnl_emoji + "\n"
            "= $" + str(round(pl_sgd, 2)) + " SGD\n"
            "New trades resume tomorrow!"
        )
        return

    # Off hours - just monitor
    if not good_session:
        open_positions = []
        for name, config in ASSETS.items():
            pos = trader.get_position(name)
            if pos:
                pnl       = trader.check_pnl(pos)
                direction = "BUY" if int(float(pos["long"]["units"])) > 0 else "SELL"
                open_positions.append(config["emoji"] + " " + name + ": " + direction + " $" + str(round(pnl, 2)))

        positions_str = "\n".join(open_positions) if open_positions else "No open trades"
        alert.send(
            "📊 DEMO 2 Off-hours\n"
            "Strategy: Mean Reversion\n"
            "Time: " + now.strftime("%H:%M SGT") + "\n"
            "Balance: $" + str(round(current_balance, 2)) + "\n"
            "Realized: $" + str(round(realized_pnl, 2)) + " USD " + pnl_emoji + "\n"
            "Trading starts: 2pm SGT\n"
            "---\n" + positions_str
        )
        return

    # ── END OF DAY HARD CLOSE at 10:55pm SGT ────────────────────────────────
    # Force close ALL positions before midnight - no overnight trades!
    if hour == 22 and now.minute >= 55:
        for name, config in ASSETS.items():
            pos = trader.get_position(name)
            if pos:
                pnl    = trader.check_pnl(pos)
                result = trader.close_position(name)
                if result["success"]:
                    alert.send(
                        "🌙 DEMO 2 END-OF-DAY CLOSE\n"
                        + config["emoji"] + " " + name + "\n"
                        + "Reason: 10:55pm SGT - no overnight trades!\n"
                        + "PnL: $" + str(round(pnl, 2)) + " USD\n"
                        + "= $" + str(round(pnl * 1.35, 2)) + " SGD"
                    )
                    log.info("EOD close " + name + " PnL=$" + str(round(pnl, 2)))
        return

    # Active session - scan for mean reversion setups!
    signals      = SignalEngine()
    calendar     = EconomicCalendar()
    scan_results = []

    # News warning
    news_summary = calendar.get_today_summary()
    if "No high" not in news_summary:
        alert.send("⚠️ DEMO 2 NEWS ALERT!\n" + news_summary)

    for name, config in ASSETS.items():
        if not settings.get(config["setting"], True):
            continue
        if today["trades"] >= settings["max_trades_day"]:
            break

        # Check open position
        position = trader.get_position(name)
        if position:
            pnl       = trader.check_pnl(position)
            direction = "BUY" if int(float(position["long"]["units"])) > 0 else "SELL"
            emoji     = "📈" if pnl > 0 else "📉"

            # ── MAX TRADE DURATION CHECK (uses OANDA open time) ──
            max_hours  = config.get("max_hours", 1)
            hours_open = 0
            try:
                # Fetch trade open time directly from OANDA - works across days!
                trade_id   = position.get("id") or position.get("tradeID")
                if trade_id:
                    t_url  = trader.base_url + "/v3/accounts/" + trader.account_id + "/trades/" + str(trade_id)
                    t_resp = requests.get(t_url, headers=trader.headers, timeout=10)
                    if t_resp.status_code == 200:
                        open_str   = t_resp.json()["trade"]["openTime"]
                        open_utc   = datetime.fromisoformat(open_str.replace("Z", "+00:00"))
                        now_utc    = datetime.now(pytz.utc)
                        hours_open = (now_utc - open_utc).total_seconds() / 3600
                        log.info(name + " open for " + str(round(hours_open, 2)) + "h / max " + str(max_hours) + "h")
            except Exception as e:
                log.warning("Trade age fetch error: " + str(e))
                # Fallback: use open_times from daily log
                open_times = today.get("open_times", {})
                if name in open_times:
                    try:
                        open_time  = datetime.fromisoformat(open_times[name])
                        open_time  = open_time.replace(tzinfo=sg_tz) if open_time.tzinfo is None else open_time
                        hours_open = (now - open_time).total_seconds() / 3600
                    except:
                        hours_open = 0

            if hours_open >= max_hours:
                close_result = trader.close_position(name)
                if close_result["success"]:
                    log.info(name + " FORCE CLOSED after " + str(round(hours_open, 2)) + "h. PnL=$" + str(round(pnl, 2)))
                    if pnl > 0:
                        today["wins"] = today.get("wins", 0) + 1
                    else:
                        today["losses"]        = today.get("losses", 0) + 1
                        today["consec_losses"] = today.get("consec_losses", 0) + 1
                    if "open_times" in today and name in today["open_times"]:
                        today["open_times"].pop(name, None)
                    with open(trade_log, "w") as f:
                        json.dump(today, f, indent=2)
                    alert.send(
                        "⏰ DEMO 2 FORCE CLOSE (1hr limit)\n"
                        + config["emoji"] + " " + name + "\n"
                        + "Open: " + str(round(hours_open, 2)) + "h (max " + str(max_hours) + "h)\n"
                        + "Direction: " + direction + "\n"
                        + "PnL: $" + str(round(pnl, 2)) + " USD\n"
                        + "= $" + str(round(pnl * 1.35, 2)) + " SGD\n"
                        + "Closed ✅"
                    )
                    scan_results.append(config["emoji"] + " " + name + ": 1HR LIMIT closed " + emoji + " $" + str(round(pnl, 2)))
                    continue

            scan_results.append(config["emoji"] + " " + name + ": " + direction + " open " + emoji + " $" + str(round(pnl, 2)) + " (" + str(round(hours_open, 2)) + "h/" + str(max_hours) + "h)")
            continue

        # Per-pair session filter - uses session_hours from ASSETS config
        session_hours = config.get("session_hours", [(14, 23)])
        pair_ok = any(start <= hour <= end for (start, end) in session_hours)
        if not pair_ok:
            hours_str = " & ".join(str(s) + "am-" + str(e) + "pm SGT" for (s, e) in session_hours)
            scan_results.append(config["emoji"] + " " + name + ": off-session (best " + hours_str + ")")
            continue

        # Cooldown check
        if is_in_cooldown(today, name):
            scan_results.append(config["emoji"] + " " + name + ": cooldown (30min after SL)")
            continue

        # Spread check
        spread_ok, spread_val = check_spread(trader, name, settings.get("max_spread_pips", 2), config["pip"])
        if not spread_ok:
            scan_results.append(config["emoji"] + " " + name + ": spread too wide - skip")
            continue

        # News blackout
        news_active, news_reason = calendar.is_news_time(name)
        if news_active:
            scan_results.append(config["emoji"] + " " + name + ": PAUSED - " + news_reason)
            continue

        # Mean reversion signal
        score, direction, details = signals.analyze(asset=config["asset"])
        log.info(name + ": score=" + str(score) + " dir=" + direction + " | " + details)

        if score < settings["signal_threshold"] or direction == "NONE":
            scan_results.append(config["emoji"] + " " + name + ": " + str(score) + "/5 no mean reversion setup")
            continue

        # Fixed TP/SL only - dynamic BB TP was too far (20-50 pips = hrs to reach)
        # Small fixed TP = price hits target within 1 hour
        tp_pips   = config["tp_pips"]
        stop_pips = config["stop_pips"]

        # Position sizing
        # Fixed 0.10 lots - no dynamic sizing
        size       = calc_position_size(config["pip"], config)
        max_loss   = round(size * stop_pips * config["pip"], 2)
        max_profit = round(size * tp_pips * config["pip"], 2)

        log.info(name + " size=" + str(size) + " stop=" + str(stop_pips) +
                 " tp=" + str(tp_pips) + " (dynamic=" + str(dynamic_tp is not None) + ")")

        # Place order
        result = trader.place_order(
            instrument     = name,
            direction      = direction,
            size           = size,
            stop_distance  = stop_pips,
            limit_distance = tp_pips
        )

        if result["success"]:
            today["trades"]       += 1
            today["consec_losses"] = 0
            # Track trade open time for max duration check
            if "open_times" not in today:
                today["open_times"] = {}
            today["open_times"][name] = now.isoformat()
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)

            price, _, _ = trader.get_price(name)
            tp_type     = "Dynamic BB" if dynamic_tp else "Fixed"
            alert.send(
                "🔄 DEMO 2 NEW TRADE! " + mode + "\n"
                + config["emoji"] + " " + name + "\n"
                "Strategy:  Mean Reversion\n"
                "Direction: " + direction + "\n"
                "Score:     " + str(score) + "/5\n"
                "Entry:     " + str(round(price, config["precision"])) + "\n"
                "Size:      " + str(size) + " units\n"
                "Stop Loss: " + str(stop_pips) + " pips = $" + str(max_loss) + "\n"
                "Take Prof: " + str(tp_pips) + " pips = $" + str(max_profit) + " (" + tp_type + ")\n"
                "Spread:    " + str(round(spread_val, 1)) + " pips\n"
                "Trade #" + str(today["trades"]) + "/" + str(settings["max_trades_day"]) + "\n"
                "Session: " + session + "\n"
                "Signals: " + details
            )
            scan_results.append(config["emoji"] + " " + name + ": " + direction + " PLACED! " + str(score) + "/5")
        else:
            set_cooldown(today, name)
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)
            scan_results.append(config["emoji"] + " " + name + ": order failed")

    # Summary
    target_hit = realized_pnl >= 22
    if target_hit:
        target_msg = "🎯 TARGET HIT! $" + str(round(pl_sgd, 0)) + " SGD today!"
    elif realized_pnl > 0:
        target_msg = "Profit $" + str(round(pl_sgd, 0)) + " SGD (target $30 SGD)"
    elif realized_pnl < 0:
        target_msg = "Loss $" + str(abs(round(pl_sgd, 0))) + " SGD today"
    else:
        target_msg = "Waiting for closed trades..."

    summary = "\n".join(scan_results) if scan_results else "No setups this scan"
    wins    = today.get("wins", 0)
    losses  = today.get("losses", 0)
    consec  = today.get("consec_losses", 0)

    alert.send(
        "🔄 DEMO 2 Scan Complete! " + mode + "\n"
        "Strategy: Mean Reversion\n"
        "Time: " + now.strftime("%H:%M SGT") + "\n"
        "Session: " + session + "\n"
        "Balance: $" + str(round(current_balance, 2)) + "\n"
        "Start:   $" + str(round(start_balance, 2)) + "\n"
        "Realized: $" + str(round(realized_pnl, 2)) + " USD " + pnl_emoji + "\n"
        "= $" + str(round(pl_sgd, 2)) + " SGD\n"
        "Open PnL: $" + str(round(open_pnl, 2)) + " USD\n"
        "Total:    $" + str(round(total_pnl, 2)) + " USD\n"
        + target_msg + "\n"
        "Trades: " + str(today["trades"]) + "/" + str(settings["max_trades_day"]) + "\n"
        "W/L: " + str(wins) + "/" + str(losses) + " | Consec loss: " + str(consec) + "\n"
        "---\n"
        + summary
    )

# ── MAIN LOOP — runs every 5 minutes on Railway ──────────────────────────────
if __name__ == "__main__":
    import time
    log.info("🔄 FOREX V2 BOT starting — scanning every 5 minutes via Railway...")
    while True:
        try:
            run_bot()
        except Exception as e:
            log.error("Bot error: " + str(e))
        log.info("Sleeping 5 minutes until next scan...")
        time.sleep(300)  # 300 seconds = 5 minutes
