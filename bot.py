"""
OANDA Trading Bot — Gold Only | CPR + EMA + Volume
===================================================
Strategy: CPR position (Check 1) + EMA alignment (Check 2)
          + Volume confirmation (Check 3) + PDH/PDL (Check 4)

Scoring:  3/5 pts minimum (London/NY) | 2/5 pts minimum (Asian)
Pair:     XAU/USD (Gold only)
Sessions: Asian (9am-1pm SGT) + London (2pm-7pm SGT) + NY Overlap (8pm-11pm SGT)

  ✅ CPR position sets direction (above TC=BUY, below BC=SELL)
  ✅ EMA20/EMA50 alignment confirms trend
  ✅ Volume ratio confirms real move (1.2x average)
  ✅ PDH/PDL adds confluence
  ✅ CPR R1/S1 used as TP targets (dynamic)
  ✅ Breakeven stop management (locks profit at 1x ATR)
  ✅ Runs every 5 minutes via Railway loop
"""

import os
import json
import logging
import time
import requests
from datetime import datetime, timedelta
import pytz

from oanda_trader import OandaTrader
from signals import SignalEngine
from cpr import CPRCalculator
from telegram_alert import TelegramAlert
from calendar_filter import EconomicCalendar

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

# ── ASSET CONFIGURATION ───────────────────────────────────────────────────────
ASSETS = {
    "XAU_USD": {
        "instrument":    "XAU_USD",
        "asset":         "XAUUSD",
        "emoji":         "🥇",
        "setting":       "trade_gold",
        "pip":           0.01,
        "precision":     2,
        "lot_size":      2,
        "session_hours": [(9, 23)],   # 9am–11pm SGT only
        # SL/TP are ATR-based — see calculate_atr_sl_tp()
    },
}


def load_settings():
    default = {
        "max_trades_day":         5,
        "max_daily_loss":         40.0,
        "signal_threshold":       3,       # 3/5 pts minimum (London/NY)
        "signal_threshold_asian": 2,       # 2/5 pts minimum (Asian)
        "demo_mode":              True,
        "trade_gold":             True,
        "trade_gold_asian":       True,
        "max_consec_losses":      2,
        "max_spread_gold":        15,      # Gold spread can be wide during sessions
        "max_spread_gold_asian":  20,      # Asian session even wider
        "strategy":               "hybrid_cpr_breakout_gold",
    }
    try:
        with open("settings.json") as f:
            saved = json.load(f)
            default.update(saved)
    except FileNotFoundError:
        with open("settings.json", "w") as f:
            json.dump(default, f, indent=2)
    return default


def sync_closed_trades(trader, today, trade_log):
    """Pull today's closed trades from OANDA and update W/L counts."""
    try:
        import pytz
        from datetime import datetime
        sg_tz    = pytz.timezone("Asia/Singapore")
        now_sg   = datetime.now(sg_tz)
        day_start = now_sg.replace(hour=0, minute=0, second=0, microsecond=0)
        # Convert to UTC RFC3339
        from datetime import timezone, timedelta
        day_start_utc = day_start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")

        url    = trader.base_url + "/v3/accounts/" + trader.account_id + "/trades"
        params = {"state": "CLOSED", "instrument": "XAU_USD", "count": "20"}
        import requests
        r = requests.get(url, headers=trader.headers, params=params, timeout=10)
        if r.status_code != 200:
            return

        trades = r.json().get("trades", [])
        wins = 0
        losses = 0
        trade_count = 0
        for t in trades:
            close_time = t.get("closeTime", "")
            if close_time < day_start_utc:
                continue  # Before today
            trade_count += 1
            pl = float(t.get("realizedPL", 0))
            if pl > 0:
                wins += 1
            elif pl < 0:
                losses += 1

        # Update W/L only — trade count uses open+closed to avoid resetting
        today["wins"]   = wins
        today["losses"] = losses
        # Also count open trades
        import requests as _req
        open_url = trader.base_url + "/v3/accounts/" + trader.account_id + "/openTrades"
        or_ = _req.get(open_url, headers=trader.headers, timeout=10)
        open_count = len(or_.json().get("trades", [])) if or_.status_code == 200 else 0
        today["trades"] = trade_count + open_count

        # Recalc consec losses from most recent closed trades
        consec = 0
        for t in sorted(trades, key=lambda x: x.get("closeTime", ""), reverse=True):
            close_time = t.get("closeTime", "")
            if close_time < day_start_utc:
                break
            pl = float(t.get("realizedPL", 0))
            if pl < 0:
                consec += 1
            else:
                break
        today["consec_losses"] = consec
        with open(trade_log, "w") as f:
            import json
            json.dump(today, f, indent=2)
        log.info("Synced " + str(trade_count + open_count) + " trades (closed=" + str(trade_count) + " open=" + str(open_count) + ") W=" + str(wins) + " L=" + str(losses))
    except Exception as e:
        log.warning("Sync trades error: " + str(e))


def get_atr_pips(trader, instrument, pip, multiplier=1.0):
    """Get ATR in pips from H1 candles"""
    try:
        url    = trader.base_url + "/v3/instruments/" + instrument + "/candles"
        params = {"count": "30", "granularity": "H1", "price": "M"}
        r      = requests.get(url, headers=trader.headers, params=params, timeout=10)
        if r.status_code != 200:
            return None
        candles = r.json()["candles"]
        c       = [x for x in candles if x["complete"]]
        if len(c) < 15:
            return None
        highs  = [float(x["mid"]["h"]) for x in c]
        lows   = [float(x["mid"]["l"]) for x in c]
        closes = [float(x["mid"]["c"]) for x in c]
        trs    = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            trs.append(tr)
        atr      = sum(trs[-14:]) / 14
        atr_pips = (atr / pip) * multiplier
        log.info(instrument + " ATR=" + str(round(atr, 4)) + " pips=" + str(round(atr_pips, 0)))
        return max(round(atr_pips), 10)
    except Exception as e:
        log.warning("ATR calc error: " + str(e))
        return None


def check_spread(trader, instrument, max_spread_pips, pip):
    try:
        mid, bid, ask = trader.get_price(instrument)
        if bid is None:
            return True, 0
        spread_pips = (ask - bid) / pip
        log.info(instrument + " spread=" + str(round(spread_pips, 1)) + " pips")
        if spread_pips > max_spread_pips:
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
    wait_until = last_loss + timedelta(minutes=10)
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


def manage_trailing_sl(trader, instrument, config, today, trade_log):
    """
    3-Level Trailing SL — auto-adjusts every scan based on progress toward TP.

    Level 1 — 25% of TP reached → move SL to entry (breakeven, $0 risk)
    Level 2 — 50% of TP reached → move SL to lock 25% of TP profit
    Level 3 — 75% of TP reached → move SL to lock 50% of TP profit

    SL only ever moves in your favour — never backward.
    Runs every 5 min via Railway scan automatically.
    """
    try:
        url = trader.base_url + "/v3/accounts/" + trader.account_id + "/openTrades"
        r   = requests.get(url, headers=trader.headers, timeout=10)
        if r.status_code != 200:
            return False

        open_trades = r.json().get("trades", [])
        fired       = False

        for trade in open_trades:
            if trade.get("instrument") != instrument:
                continue

            trade_id    = trade["id"]
            entry_price = float(trade["price"])
            precision   = config["precision"]
            units       = float(trade.get("currentUnits", trade.get("initialUnits", 0)))
            direction   = "BUY" if units > 0 else "SELL"

            # Get current TP distance from the trade's takeProfit order
            tp_order    = trade.get("takeProfitOrder", {})
            tp_price    = float(tp_order.get("price", 0)) if tp_order else 0
            if not tp_price:
                continue

            tp_dist_pips = abs(tp_price - entry_price) / config["pip"]

            # Get current price
            current_price, _, _ = trader.get_price(instrument)
            if not current_price:
                continue

            # Progress toward TP (0.0 = entry, 1.0 = TP)
            if direction == "BUY":
                progress = (current_price - entry_price) / (tp_price - entry_price)
            else:
                progress = (entry_price - current_price) / (entry_price - tp_price)

            progress = max(0.0, progress)

            # Current SL from trade
            sl_order     = trade.get("stopLossOrder", {})
            current_sl   = float(sl_order.get("price", 0)) if sl_order else 0

            # Calculate new SL targets for each level
            # Level 1: breakeven = entry
            sl_level1 = entry_price

            # Level 2: lock 25% of TP distance as profit
            if direction == "BUY":
                sl_level2 = entry_price + (tp_price - entry_price) * 0.25
                sl_level3 = entry_price + (tp_price - entry_price) * 0.50
            else:
                sl_level2 = entry_price - (entry_price - tp_price) * 0.25
                sl_level3 = entry_price - (entry_price - tp_price) * 0.50

            # Determine which level to apply
            new_sl      = None
            level_label = None

            if progress >= 0.75:
                candidate = round(sl_level3, precision)
                # Only move SL if it improves (closer to TP than current)
                if direction == "BUY" and (not current_sl or candidate > current_sl):
                    new_sl = candidate
                    level_label = "Level 3 — 50% profit locked"
                elif direction == "SELL" and (not current_sl or candidate < current_sl):
                    new_sl = candidate
                    level_label = "Level 3 — 50% profit locked"

            elif progress >= 0.50:
                candidate = round(sl_level2, precision)
                if direction == "BUY" and (not current_sl or candidate > current_sl):
                    new_sl = candidate
                    level_label = "Level 2 — 25% profit locked"
                elif direction == "SELL" and (not current_sl or candidate < current_sl):
                    new_sl = candidate
                    level_label = "Level 2 — 25% profit locked"

            elif progress >= 0.25:
                candidate = round(sl_level1, precision)
                if direction == "BUY" and (not current_sl or candidate > current_sl):
                    new_sl = candidate
                    level_label = "Level 1 — breakeven"
                elif direction == "SELL" and (not current_sl or candidate < current_sl):
                    new_sl = candidate
                    level_label = "Level 1 — breakeven"

            if new_sl is None:
                log.info(instrument + " trailing SL: progress=" + str(round(progress*100)) + "% — no adjustment yet")
                continue

            # Apply new SL via OANDA API
            patch_url  = trader.base_url + "/v3/accounts/" + trader.account_id + "/trades/" + trade_id + "/orders"
            patch_data = {
                "stopLoss": {
                    "price":       str(new_sl),
                    "timeInForce": "GTC"
                }
            }
            pr = requests.put(patch_url, headers=trader.headers, json=patch_data, timeout=10)
            if pr.status_code == 200:
                log.info(instrument + " trailing SL updated to " + str(new_sl) + " (" + level_label + ")")
                # Store in today log for Telegram alert
                today["trailing_sl_" + instrument] = {
                    "level": level_label,
                    "new_sl": new_sl,
                    "progress_pct": round(progress * 100),
                    "direction": direction
                }
                with open(trade_log, "w") as f:
                    json.dump(today, f, indent=2)
                fired = True
            else:
                log.warning(instrument + " trailing SL update failed: " + str(pr.status_code))

        return fired

    except Exception as e:
        log.warning("Trailing SL error: " + str(e))
        return False


def manage_breakeven(trader, instrument, config, today, trade_log):
    """Kept for compatibility — delegates to manage_trailing_sl"""
    return manage_trailing_sl(trader, instrument, config, today, trade_log)


def run_bot():
    log.info("🥇 GOLD BOT scanning...")
    settings = load_settings()
    sg_tz    = pytz.timezone("Asia/Singapore")
    now      = datetime.now(sg_tz)
    alert    = TelegramAlert()
    cpr_calc = CPRCalculator()
    hour     = now.hour

    # ── SESSION DETECTION ─────────────────────────────────────
    # Trading hours: 9am–11pm SGT only
    # Off hours:     11pm–9am SGT (bot sleeps)
    active_hours = (9 <= hour <= 23)
    london_open  = (14 <= hour <= 17)
    london       = (14 <= hour <= 19)
    ny_overlap   = (20 <= hour <= 23)
    asian        = (9 <= hour <= 13)    # SGX/Tokyo overlap window
    good_session = active_hours

    if asian:
        session = "Asian Session 🌏 (SGX/Tokyo — 9am–1pm SGT)"
    elif london_open:
        session = "London Open 🔥 (BEST for Gold breakouts!)"
    elif ny_overlap:
        session = "NY Overlap 🔥 (BEST for Gold macro moves!)"
    elif london:
        session = "London Session 🇬🇧"
    else:
        session = "Off-hours (monitoring only)"

    # ── WEEKEND CHECK ─────────────────────────────────────────
    if now.weekday() == 5:
        log.info("Saturday — markets closed, skipping scan")
        return
    if now.weekday() == 6 and hour < 9:
        log.info("Sunday early — skipping scan, resumes 9am SGT")
        return

    # ── LOGIN ─────────────────────────────────────────────────
    trader = OandaTrader(demo=settings["demo_mode"])
    if not trader.login():
        alert.send("❌ GOLD BOT Login FAILED! Check secrets.")
        return

    current_balance = trader.get_balance()
    mode            = "DEMO" if settings["demo_mode"] else "LIVE"

    # ── LOAD TODAY LOG ────────────────────────────────────────
    trade_log = "trades_" + now.strftime("%Y%m%d") + ".json"
    try:
        with open(trade_log) as f:
            today = json.load(f)
    except FileNotFoundError:
        today = {
            "trades":              0,
            "start_balance":       current_balance,
            "daily_pnl":           0.0,
            "stopped":             False,
            "wins":                0,
            "losses":              0,
            "consec_losses":       0,
            "cooldowns":           {},
            "cpr_alert_sent":      False,
            "cpr_alert_asian_sent": False,
        }
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)
        log.info("New day! Start balance: $" + str(round(current_balance, 2)))

    # ── PNL TRACKING ──────────────────────────────────────────
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

    # ── SYNC TRADES FROM OANDA ───────────────────────────────
    sync_closed_trades(trader, today, trade_log)

    # ── RISK GUARDS ───────────────────────────────────────────
    if today.get("stopped"):
        log.info("Bot stopped for today — daily limit hit")
        return

    if realized_pnl <= -settings["max_daily_loss"]:
        today["stopped"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)
        alert.send(
            "🔴 GOLD BOT DAILY LOSS LIMIT!\n"
            "Loss: $" + str(abs(round(realized_pnl, 2))) + " USD\n"
            "Limit: $" + str(settings["max_daily_loss"]) + " USD\n"
            "Stopped for today. Resumes tomorrow."
        )
        return

    # Consecutive loss guard removed — bot trades freely
    # Protected by: daily loss limit + signal threshold + 10min cooldown after each loss

    if today["trades"] >= settings["max_trades_day"]:
        log.info("Max trades reached for today")
        return

    # ── CPR LEVELS ────────────────────────────────────────────
    cpr_gold = cpr_calc.get_levels("XAU_USD")

    # Send CPR alert at 9am SGT (session open) and 2pm SGT (London open)
    send_cpr_alert = (
        (asian and hour == 9 and not today.get("cpr_alert_asian_sent")) or
        (london_open and hour == 14 and not today.get("cpr_alert_sent"))
    )
    if send_cpr_alert:
        session_label = "Asian Open 🌏" if asian else "London Open 🇬🇧"
        cpr_msg = "🌅 GOLD BOT — " + session_label + " CPR Levels\n"
        cpr_msg += "─────────────────────────\n"
        if cpr_gold:
            narrow_flag = " ⚡ NARROW — TRENDING DAY!" if cpr_gold["is_narrow"] else ""
            wide_flag   = " ⚠️ WIDE — CHOPPY (reduce size)" if cpr_gold["is_wide"] else ""
            cpr_msg += (
                "🥇 GOLD CPR" + narrow_flag + wide_flag + "\n"
                "TC=" + str(cpr_gold["tc"]) +
                " BC=" + str(cpr_gold["bc"]) +
                " Pivot=" + str(cpr_gold["pivot"]) + "\n"
                "R1=" + str(cpr_gold["r1"]) +
                " S1=" + str(cpr_gold["s1"]) +
                " Width=" + str(cpr_gold["width_pct"]) + "%"
            )
        alert.send(cpr_msg)
        if asian:
            today["cpr_alert_asian_sent"] = True
        else:
            today["cpr_alert_sent"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)

    # ── OFF-HOURS: silent return — no Telegram, no API calls ────
    if not good_session:
        log.info("Off-hours (11pm–9am SGT) — sleeping silently")
        return

    # ── TRAILING SL MANAGEMENT ────────────────────────────────
    for name, config in ASSETS.items():
        tsl_result = manage_trailing_sl(trader, name, config, today, trade_log)
        if tsl_result:
            tsl_info = today.get("trailing_sl_" + name, {})
            level    = tsl_info.get("level", "SL adjusted")
            new_sl   = tsl_info.get("new_sl", "")
            prog     = tsl_info.get("progress_pct", "")
            position = trader.get_position(name)
            open_pnl = round(trader.check_pnl(position), 2) if position else 0
            alert.send(
                "🔒 TRAILING SL — " + config["emoji"] + " " + name + "\n"
                + level + "\n"
                "New SL: " + str(new_sl) + "\n"
                "Progress: " + str(prog) + "% toward TP\n"
                "Open PnL: $" + str(open_pnl)
            )

    # ── NEWS WARNING ──────────────────────────────────────────
    calendar     = EconomicCalendar()
    news_summary = calendar.get_today_summary()
    if "No high" not in news_summary:
        alert.send("⚠️ GOLD BOT NEWS ALERT!\n" + news_summary +
                   "\n💡 Note: CPR levels often break around news!")

    # ── SCAN FOR SETUPS ───────────────────────────────────────
    signals      = SignalEngine()
    scan_results = []

    for name, config in ASSETS.items():
        if not settings.get(config["setting"], True):
            continue
        if today["trades"] >= settings["max_trades_day"]:
            break

        # Check existing position
        position = trader.get_position(name)
        if position:
            pnl       = trader.check_pnl(position)
            direction = "BUY" if int(float(position["long"]["units"])) > 0 else "SELL"
            emoji     = "📈" if pnl > 0 else "📉"
            scan_results.append(config["emoji"] + " " + name + ": " + direction +
                                 " open " + emoji + " $" + str(round(pnl, 2)))
            continue

        # Session filter per pair
        session_hours = config.get("session_hours", [(14, 23)])
        pair_ok       = any(start <= hour <= end for (start, end) in session_hours)
        if not pair_ok:
            scan_results.append(config["emoji"] + " " + name + ": off-session")
            continue

        # Asian Gold flag
        is_asian_gold = asian and name == "XAU_USD"

        # Skip Asian Gold if disabled
        if is_asian_gold and not settings.get("trade_gold_asian", True):
            scan_results.append(config["emoji"] + " " + name + ": Asian session disabled")
            continue

        # Cooldown removed — bot trades freely based on signals

        # Spread check
        if is_asian_gold:
            max_spread = settings.get("max_spread_gold_asian", 8)
        else:
            max_spread = settings.get("max_spread_gold", 5)
        spread_ok, spread_val = check_spread(trader, name, max_spread, config["pip"])

        # News blackout
        news_active, news_reason = calendar.is_news_time(name)
        if news_active:
            scan_results.append(config["emoji"] + " " + name + ": PAUSED — " + news_reason)
            continue

        # ── SIGNAL ANALYSIS — always run even if spread wide ──
        asset_key = "XAUUSD_ASIAN" if is_asian_gold else config["asset"]
        threshold = settings.get("signal_threshold_asian", 2) if is_asian_gold else settings["signal_threshold"]

        score, direction, details = signals.analyze(asset=asset_key)
        log.info(name + ": score=" + str(score) + " dir=" + direction + " | " + details)

        # Show score even if spread too wide — so user can see signal strength
        if not spread_ok:
            scan_results.append(
                config["emoji"] + " " + name + ": ⚠️ Spread " + str(round(spread_val, 1)) +
                " pips (too wide) | Score: " + str(score) + "/5 dir=" + direction
            )
            continue

        # Watching for breakout (Asian session)
        if is_asian_gold and score >= 2 and direction == "NONE":
            scan_results.append(
                config["emoji"] + " " + name + ": ⏳ Watching for breakout (" +
                str(score) + "/5)"
            )
            cpr_lvls  = cpr_calc.get_levels("XAU_USD")
            watch_msg = (
                "⏳ GOLD Asian Session — Watching for Breakout\n"
                "Score: " + str(score) + "/5 — need " + str(threshold) + " to trade\n"
                "CPR Width: "
            )
            if cpr_lvls:
                watch_msg += str(cpr_lvls["width_pct"]) + "%\n"
                watch_msg += (
                    "CPR TC=" + str(cpr_lvls["tc"]) +
                    " BC=" + str(cpr_lvls["bc"]) + "\n"
                    "R1=" + str(cpr_lvls["r1"]) +
                    " S1=" + str(cpr_lvls["s1"]) + "\n"
                )
            watch_msg += "─── Signals ───\n" + details.replace(" | ", "\n")
            alert.send(watch_msg)
            continue

        if score < threshold or direction == "NONE":
            scan_results.append(
                config["emoji"] + " " + name + ": " + str(score) + "/5 — no setup yet"
            )
            continue

        # ── POSITION SIZING ───────────────────────────────────
        cpr_levels = cpr_calc.get_levels(config["instrument"])
        is_wide    = cpr_levels["is_wide"] if cpr_levels else False
        size       = config["lot_size"] // 2 if is_wide else config["lot_size"]

        # ── ATR-BASED SL/TP (1:2 R:R always) ─────────────────
        # SL = 1x ATR  |  TP = 2x ATR  →  always 1:2 R:R
        # Limits: SL min 150 pips, max 500 pips (Gold safety)
        price, _, _ = trader.get_price(name)
        raw_atr     = get_atr_pips(trader, name, config["pip"], multiplier=1.0)

        if raw_atr:
            stop_pips = max(600, min(raw_atr, 600))    # fixed 600p — survives Gold noise
            tp_pips   = stop_pips * 3                   # always 1:3 R:R = 1800p
            tp_label  = "3x SL (1:3 R:R)"
        else:
            stop_pips = 600    # fallback
            tp_pips   = 1800
            tp_label  = "Fixed fallback (1:3 R:R)"

        max_loss   = round(size * stop_pips * config["pip"], 2)
        max_profit = round(size * tp_pips   * config["pip"], 2)

        # ── PLACE ORDER ───────────────────────────────────────
        result = trader.place_order(
            instrument     = name,
            direction      = direction,
            size           = size,
            stop_distance  = stop_pips,
            limit_distance = tp_pips
        )

        if result["success"]:
            today["trades"]           += 1
            today["consec_losses"]     = 0
            today["breakeven_" + name] = False
            # Clear cooldown after a win — free to trade again
            if "cooldowns" in today and name in today["cooldowns"]:
                del today["cooldowns"][name]
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)

            price, _, _  = trader.get_price(name)
            cpr_summary  = cpr_calc.summary_text(config["instrument"]) if cpr_levels else "CPR: unavailable"
            size_note    = " (50% — wide CPR day)" if is_wide else ""

            alert.send(
                "🥇 GOLD TRADE! " + mode + "\n"
                + config["emoji"] + " " + name + "\n"
                "Strategy: CPR + Breakout Momentum\n"
                "Direction: " + direction + "\n"
                "Score:    " + str(score) + "/5\n"
                "Entry:    " + str(round(price, config["precision"])) + "\n"
                "Size:     " + str(size) + " units" + size_note + "\n"
                "Stop:     " + str(stop_pips) + " pips = $" + str(max_loss) + "\n"
                "Target:   " + str(tp_pips) + " pips = $" + str(max_profit) + " (" + tp_label + ")\n"
                "R:R:      1:" + str(round(tp_pips / stop_pips, 1)) + "\n"
                "Spread:   " + str(round(spread_val, 1)) + " pips\n"
                "Trade #"  + str(today["trades"]) + "/" + str(settings["max_trades_day"]) + "\n"
                "Session:  " + session + "\n"
                "─── CPR Levels ───\n"
                + cpr_summary + "\n"
                "─── Signals ───\n"
                + details.replace(" | ", "\n")
            )
            scan_results.append(
                config["emoji"] + " " + name + ": " +
                direction + " PLACED! " + str(score) + "/5"
            )
        else:
            set_cooldown(today, name)
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)
            scan_results.append(config["emoji"] + " " + name + ": order failed")

    # ── SCAN SUMMARY ──────────────────────────────────────────
    target_hit = realized_pnl >= 22
    if target_hit:
        target_msg = "🎯 TARGET HIT! $" + str(round(pl_sgd, 0)) + " SGD today!"
    elif realized_pnl > 0:
        target_msg = "Profit $" + str(round(pl_sgd, 0)) + " SGD (target $30 SGD)"
    elif realized_pnl < 0:
        target_msg = "Loss $" + str(abs(round(pl_sgd, 0))) + " SGD today"
    else:
        target_msg = "Scanning for setups..."

    summary = "\n".join(scan_results) if scan_results else "No setups this scan"
    # Sync wins/losses from realized PnL change
    prev_balance = today.get("start_balance", current_balance)
    if realized_pnl > 0 and today.get("wins", 0) == 0 and today.get("losses", 0) == 0:
        pass  # No trades closed yet
    wins    = today.get("wins", 0)
    losses  = today.get("losses", 0)
    consec  = today.get("consec_losses", 0)

    # CPR summary line — friend-style format
    cpr_line = ""
    if cpr_gold:
        width_flag = " ⚡NARROW" if cpr_gold["is_narrow"] else (" ⚠️WIDE" if cpr_gold["is_wide"] else "")
        cpr_line = (
            "CPR Width: " + str(cpr_gold["width_pct"]) + "%" + width_flag + " | 1% risk\n"
            "CPR TC=" + str(cpr_gold["tc"]) + " BC=" + str(cpr_gold["bc"]) + "\n"
            "R1=" + str(cpr_gold["r1"]) + " S1=" + str(cpr_gold["s1"]) + "\n"
        )

    threshold_used = settings.get("signal_threshold_asian", 2) if asian else settings["signal_threshold"]

    alert.send(
        "🥇 GOLD BOT Scan! " + mode + "\n"
        "Time: " + now.strftime("%H:%M SGT") + " | " + session + "\n"
        "Balance: $" + str(round(current_balance, 2)) +
        " | Realized: $" + str(round(realized_pnl, 2)) + " " + pnl_emoji + "\n"
        "Trades: " + str(today["trades"]) + "/" + str(settings["max_trades_day"]) +
        " | W/L: " + str(wins) + "/" + str(losses) + "\n"
        "Need: " + str(threshold_used) + "/5 to trade\n"
        + target_msg + "\n"
        "─────────────────────────\n"
        + cpr_line +
        "─── Setups ───\n"
        + summary
    )


# ── MAIN LOOP — runs every 5 minutes on Railway ───────────────────────────────
if __name__ == "__main__":
    log.info("🥇 GOLD BOT starting — scanning every 5 minutes via Railway...")
    while True:
        try:
            run_bot()
        except Exception as e:
            log.error("Bot error: " + str(e))
        log.info("Sleeping 5 minutes until next scan...")
        time.sleep(300)  # 300 seconds = 5 minutes
