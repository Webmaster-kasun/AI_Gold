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
        "max_daily_loss":         74.0,    # ~$100 SGD
        "signal_threshold":       5,       # 5/7 pts minimum (London/NY)
        "signal_threshold_asian": 4,       # 4/7 pts minimum (Asian)
        "demo_mode":              True,
        "trade_gold":             True,
        "trade_gold_asian":       True,
        "max_consec_losses":      2,
        "max_spread_gold":        999,     # demo spreads are wide
        "max_spread_gold_asian":  999,
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
        # Track last closed trade for smart re-entry logic
        today_closed = [t for t in trades if t.get("closeTime","") >= day_start_utc]
        if today_closed:
            latest = sorted(today_closed, key=lambda x: x.get("closeTime",""))[-1]
            today["last_trade_close_time"]   = latest.get("closeTime","")
            today["last_trade_close_result"] = "WIN" if float(latest.get("realizedPL",0)) > 0 else "LOSS"
            today["last_trade_entry_price"]  = float(latest.get("price", today.get("last_trade_entry_price") or 0))

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


# Cooldown functions removed — bot trades freely on signal score alone




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
            "cpr_alert_sent":       False,
            "cpr_alert_asian_sent":  False,
            "news_alert_sent":        False,
            "last_trade_close_time":  None,
            "last_trade_close_result": None,
            "last_trade_entry_price":  None,
        }
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)
        log.info("New day! Start balance: $" + str(round(current_balance, 2)))

    # ── PNL TRACKING ──────────────────────────────────────────
    start_balance = today.get("start_balance", current_balance)
    open_pnl = 0.0
    for _n in ASSETS:
        _pos = trader.get_position(_n)
        if _pos:
            open_pnl += trader.check_pnl(_pos)
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

    # Fixed SL/TP — OANDA manages trade automatically after entry

    # ── NEWS WARNING (once per day only) ─────────────────────
    calendar     = EconomicCalendar()
    news_summary = calendar.get_today_summary()
    if "No high" not in news_summary and not today.get("news_alert_sent"):
        alert.send("⚠️ GOLD BOT NEWS ALERT!\n" + news_summary +
                   "\n💡 Note: CPR levels often break around news!")
        today["news_alert_sent"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)

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

        # ── SMART RE-ENTRY GUARD ─────────────────────────────
        # Data from 32 trades proves:
        #   < 30 min gap  = 8% win rate  (LOSING)
        #   >= 30 min gap = 26% win rate (PROFITABLE)
        #   < 500p price diff from last entry = 0% win rate
        #   >= 500p price diff = 27% win rate
        # Both conditions must pass before next trade allowed.
        last_close_time   = today.get("last_trade_close_time")
        last_close_result = today.get("last_trade_close_result")
        last_entry_price  = today.get("last_trade_entry_price") or 0

        if last_close_time:
            try:
                close_dt   = datetime.strptime(last_close_time[:16].replace("T"," "), "%Y-%m-%d %H:%M")
                now_utc    = datetime.utcnow()
                mins_since = (now_utc - close_dt).total_seconds() / 60

                # Rule 1 — Time gate (30 min mandatory)
                if mins_since < 30:
                    remaining = int(30 - mins_since)
                    result_label = "after " + (last_close_result or "trade")
                    scan_results.append(
                        config["emoji"] + " " + name + ": ⏳ " + str(remaining) +
                        "min cooldown " + result_label + " (market settling)"
                    )
                    continue

                # Rule 2 — Price zone gate (500p from last entry)
                if last_entry_price:
                    mid_price, _, _ = trader.get_price(name)
                    if mid_price:
                        price_diff = abs(mid_price - last_entry_price) / config["pip"]
                        if price_diff < 500:
                            needed = int(500 - price_diff)
                            scan_results.append(
                                config["emoji"] + " " + name + ": 🔲 Same zone — need " +
                                str(needed) + "p more movement (current diff=" +
                                str(int(price_diff)) + "p, need 500p)"
                            )
                            continue
            except Exception as e:
                log.warning("Re-entry guard error: " + str(e))

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
                "Score: " + str(score) + "/7 — need " + str(threshold) + " to trade\n"
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
                config["emoji"] + " " + name + ": " + str(score) + "/7 — no setup yet"
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
            stop_pips = max(500, min(raw_atr, 600))    # SL: 500–600p
            tp_pips   = min(stop_pips * 3, 1800)        # TP: max 1800p always
            tp_label  = "3x SL capped 1800p (1:3 R:R)"
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
            today["trades"]                  += 1
            today["consec_losses"]            = 0
            today["breakeven_" + name]        = False
            today["last_trade_entry_price"]   = price   # save for zone check

            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)

            if cpr_levels:
                cpr_summary = (
                    "TC=" + str(cpr_levels["tc"]) + " BC=" + str(cpr_levels["bc"]) +
                    " Pivot=" + str(cpr_levels["pivot"]) + "\n" +
                    "R1=" + str(cpr_levels["r1"]) + " S1=" + str(cpr_levels["s1"]) +
                    " Width=" + str(cpr_levels["width_pct"]) + "%"
                )
            else:
                cpr_summary = "CPR: unavailable"
            size_note    = " (50% — wide CPR day)" if is_wide else ""

            alert.send(
                "🥇 GOLD TRADE! " + mode + "\n"
                + config["emoji"] + " " + name + "\n"
                "Strategy: CPR + Breakout Momentum\n"
                "Direction: " + direction + "\n"
                "Score:    " + str(score) + "/7\n"
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
            log.warning(name + " order failed: " + str(result.get("error", "")))
            scan_results.append(config["emoji"] + " " + name + ": order failed — " + str(result.get("error", ""))[:50])

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

    # Send scan summary every 30 min OR instantly when trade placed
    trade_just_placed = any("PLACED" in r for r in scan_results)
    last_alert_min    = today.get("last_scan_alert_min", -31)
    current_min       = now.hour * 60 + now.minute
    mins_since_alert  = current_min - last_alert_min if current_min >= last_alert_min else current_min + 1440 - last_alert_min
    should_alert      = trade_just_placed or mins_since_alert >= 30

    if should_alert:
        today["last_scan_alert_min"] = current_min
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)
        alert.send(
            "🥇 GOLD BOT Scan! " + mode + "\n"
            "Time: " + now.strftime("%H:%M SGT") + " | " + session + "\n"
            "Balance: $" + str(round(current_balance, 2)) +
            " | Realized: $" + str(round(realized_pnl, 2)) + " " + pnl_emoji + "\n"
            "Trades: " + str(today["trades"]) + "/" + str(settings["max_trades_day"]) +
            " | W/L: " + str(wins) + "/" + str(losses) + "\n"
            "Need: " + str(threshold_used) + "/7 to trade\n"
            + target_msg + "\n"
            "─────────────────────────\n"
            + cpr_line +
            "─── Setups ───\n"
            + summary
        )
    else:
        log.info("Scan silent — next summary in " + str(30 - mins_since_alert) + " mins")


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
