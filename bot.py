"""
OANDA Trading Bot — Hybrid CPR + Breakout Momentum
====================================================
Strategy: CPR bias (Layer 1) + H4 breakout (Layer 2) + Macro (Layer 3)
          + EMA trend (Layer 4) + RSI/MACD momentum (Layer 5)

Scoring:  4/8 points minimum to trade
Pairs:    GBP/USD + XAU/USD
Sessions: London (2pm–7pm SGT) + NY Overlap (8pm–11pm SGT)

New vs Demo 3:
  ✅ CPR daily levels as primary bias filter
  ✅ CPR R1/S1 used as TP targets (dynamic, smarter than fixed pips)
  ✅ Breakeven stop management (locks profit at 1x ATR)
  ✅ Narrow CPR alert at session open
  ✅ Wide CPR = reduced position size (50% on choppy days)
  ✅ 5-layer score shown in every Telegram message
"""

import os
import json
import logging
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
    "GBP_USD": {
        "instrument":    "GBP_USD",
        "asset":         "GBPUSD",
        "emoji":         "💷",
        "setting":       "trade_gbpusd",
        "stop_pips":     30,
        "tp_pips":       60,
        "pip":           0.0001,
        "precision":     5,
        "min_atr":       0.0015,
        "lot_size":      20000,
        "session_hours": [(14, 19)],   # London only — GBP best 2pm–7pm SGT
    },
    "XAU_USD": {
        "instrument":    "XAU_USD",
        "asset":         "XAUUSD",
        "emoji":         "🥇",
        "setting":       "trade_gold",
        "stop_pips":     800,
        "tp_pips":       1600,
        "pip":           0.01,
        "precision":     2,
        "min_atr":       8.0,
        "lot_size":      2,
        "session_hours": [(14, 23), (0, 1)],   # London + NY — Gold active both
    },
}


def load_settings():
    default = {
        "max_trades_day":    5,         # Up to 3 Gold + 2 GBP per day
        "max_daily_loss":    40.0,
        "signal_threshold":  4,         # 4/8 points minimum
        "demo_mode":         True,
        "trade_gbpusd":      True,
        "trade_gold":        True,
        "max_consec_losses": 2,
        "max_spread_pips":   2,
        "max_spread_gold":   5,
        "strategy":          "hybrid_cpr_breakout",
    }
    try:
        with open("settings.json") as f:
            saved = json.load(f)
            default.update(saved)
    except FileNotFoundError:
        with open("settings.json", "w") as f:
            json.dump(default, f, indent=2)
    return default


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
    wait_until = last_loss + timedelta(minutes=45)
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


def manage_breakeven(trader, instrument, config, today, trade_log):
    """
    Breakeven stop management:
    If open PnL >= 1x ATR value → move SL to entry price.
    Only triggers once per trade (tracked in today log).
    """
    be_key = "breakeven_" + instrument
    if today.get(be_key):
        return  # Already set breakeven for this trade

    position = trader.get_position(instrument)
    if not position:
        return

    pnl = trader.check_pnl(position)

    # Get 1x ATR value in USD
    atr_pips = get_atr_pips(trader, instrument, config["pip"], multiplier=1.0)
    if not atr_pips:
        return

    # For Gold: 1 pip = $0.01 per unit. For Forex: varies
    atr_usd = atr_pips * config["pip"] * config["lot_size"]

    if pnl >= atr_usd:
        # Try to move SL to entry (breakeven)
        try:
            long_units  = int(float(position["long"]["units"]))
            short_units = int(float(position["short"]["units"]))

            # Get current open trade IDs from OANDA
            url = trader.base_url + "/v3/accounts/" + trader.account_id + "/openTrades"
            r   = requests.get(url, headers=trader.headers, timeout=10)
            if r.status_code != 200:
                return

            trades = r.json().get("trades", [])
            for trade in trades:
                if trade.get("instrument") != instrument:
                    continue

                trade_id    = trade["id"]
                entry_price = float(trade["price"])
                precision   = config["precision"]

                # Patch SL to entry price
                patch_url  = trader.base_url + "/v3/accounts/" + trader.account_id + "/trades/" + trade_id + "/orders"
                patch_data = {
                    "stopLoss": {
                        "price":       str(round(entry_price, precision)),
                        "timeInForce": "GTC"
                    }
                }
                pr = requests.put(patch_url, headers=trader.headers, json=patch_data, timeout=10)
                if pr.status_code == 200:
                    today[be_key] = True
                    with open(trade_log, "w") as f:
                        json.dump(today, f, indent=2)
                    log.info(instrument + " BREAKEVEN set at " + str(round(entry_price, precision)))
                    return True

        except Exception as e:
            log.warning("Breakeven error: " + str(e))

    return False


def run_bot():
    log.info("OANDA Hybrid CPR + Breakout Bot starting!")
    settings = load_settings()
    sg_tz    = pytz.timezone("Asia/Singapore")
    now      = datetime.now(sg_tz)
    alert    = TelegramAlert()
    cpr_calc = CPRCalculator()
    hour     = now.hour

    # ── SESSION DETECTION ─────────────────────────────────────
    london_open = (14 <= hour <= 17)
    london      = (14 <= hour <= 19)
    ny_overlap  = (20 <= hour <= 23)
    late_ny     = (0 <= hour <= 1)
    good_session = london or ny_overlap or late_ny

    if london_open:
        session = "London Open 🔥 (BEST for GBP & Gold breakouts!)"
    elif ny_overlap:
        session = "NY Overlap 🔥 (BEST for Gold macro moves!)"
    elif london:
        session = "London Session 🇬🇧"
    elif late_ny:
        session = "NY Late Session 🇺🇸"
    else:
        session = "Off-hours (monitoring only)"

    # ── WEEKEND CHECK ─────────────────────────────────────────
    if now.weekday() == 5:
        alert.send("Saturday — markets closed! Hybrid bot resumes Monday 5am SGT")
        return
    if now.weekday() == 6 and hour < 5:
        alert.send("Sunday early — Hybrid bot resumes at 5am SGT")
        return

    # ── LOGIN ─────────────────────────────────────────────────
    trader = OandaTrader(demo=settings["demo_mode"])
    if not trader.login():
        alert.send("❌ HYBRID BOT Login FAILED! Check secrets.")
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
            "trades":        0,
            "start_balance": current_balance,
            "daily_pnl":     0.0,
            "stopped":       False,
            "wins":          0,
            "losses":        0,
            "consec_losses": 0,
            "cooldowns":     {},
            "cpr_alert_sent": False,
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

    # ── RISK GUARDS ───────────────────────────────────────────
    if today.get("stopped"):
        alert.send(
            "💥 HYBRID BOT stopped for today\n"
            "Daily limit hit!\n"
            "Realized: $" + str(round(realized_pnl, 2)) + " USD\n"
            "Resumes tomorrow!"
        )
        return

    if realized_pnl <= -settings["max_daily_loss"]:
        today["stopped"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)
        alert.send(
            "🔴 HYBRID BOT DAILY LOSS LIMIT!\n"
            "Loss: $" + str(abs(round(realized_pnl, 2))) + " USD\n"
            "Limit: $" + str(settings["max_daily_loss"]) + " USD\n"
            "Stopped for today. Resumes tomorrow."
        )
        return

    consec = today.get("consec_losses", 0)
    if consec >= settings.get("max_consec_losses", 2):
        today["stopped"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)
        alert.send(
            "⛔ HYBRID BOT: 2 CONSECUTIVE LOSSES!\n"
            "Capital protection activated!\n"
            "Realized: $" + str(round(realized_pnl, 2)) + " USD\n"
            "Resumes tomorrow!"
        )
        return

    if today["trades"] >= settings["max_trades_day"]:
        alert.send(
            "✅ HYBRID BOT Max trades reached!\n"
            "Trades: " + str(today["trades"]) + "/" + str(settings["max_trades_day"]) + "\n"
            "Realized: $" + str(round(realized_pnl, 2)) + " USD " + pnl_emoji + "\n"
            "= $" + str(round(pl_sgd, 2)) + " SGD\n"
            "Resumes tomorrow!"
        )
        return

    # ── CPR LEVELS (fetch once per day, send alert at London open) ──
    cpr_gold = cpr_calc.get_levels("XAU_USD")
    cpr_gbp  = cpr_calc.get_levels("GBP_USD")

    if london_open and not today.get("cpr_alert_sent"):
        cpr_msg = "🌅 HYBRID BOT — London Open CPR Levels\n"
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
                " Width=" + str(cpr_gold["width_pct"]) + "%\n\n"
            )
        if cpr_gbp:
            narrow_flag = " ⚡ NARROW" if cpr_gbp["is_narrow"] else ""
            cpr_msg += (
                "💷 GBP/USD CPR" + narrow_flag + "\n"
                "TC=" + str(cpr_gbp["tc"]) +
                " BC=" + str(cpr_gbp["bc"]) +
                " Pivot=" + str(cpr_gbp["pivot"]) + "\n"
                "R1=" + str(cpr_gbp["r1"]) +
                " S1=" + str(cpr_gbp["s1"]) +
                " Width=" + str(cpr_gbp["width_pct"]) + "%"
            )
        alert.send(cpr_msg)
        today["cpr_alert_sent"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)

    # ── OFF-HOURS MONITORING ──────────────────────────────────
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
            "💥 HYBRID BOT Off-hours\n"
            "Time: " + now.strftime("%H:%M SGT") + "\n"
            "Balance: $" + str(round(current_balance, 2)) + "\n"
            "Realized: $" + str(round(realized_pnl, 2)) + " USD " + pnl_emoji + "\n"
            "Trading starts: 2pm SGT\n"
            "---\n" + positions_str
        )
        return

    # ── BREAKEVEN MANAGEMENT (check all open positions) ───────
    for name, config in ASSETS.items():
        be_result = manage_breakeven(trader, name, config, today, trade_log)
        if be_result:
            price, _, _ = trader.get_price(name)
            alert.send(
                "🔒 BREAKEVEN SET — " + config["emoji"] + " " + name + "\n"
                "SL moved to entry price!\n"
                "Trade is now risk-free 🎯\n"
                "Open PnL: $" + str(round(trader.check_pnl(trader.get_position(name)), 2))
            )

    # ── NEWS WARNING ──────────────────────────────────────────
    calendar     = EconomicCalendar()
    news_summary = calendar.get_today_summary()
    if "No high" not in news_summary:
        alert.send("⚠️ HYBRID BOT NEWS ALERT!\n" + news_summary +
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

        # Cooldown check
        if is_in_cooldown(today, name):
            scan_results.append(config["emoji"] + " " + name + ": cooldown 45min")
            continue

        # Spread check
        max_spread = settings.get("max_spread_gold", 5) if name == "XAU_USD" else settings.get("max_spread_pips", 2)
        spread_ok, spread_val = check_spread(trader, name, max_spread, config["pip"])
        if not spread_ok:
            scan_results.append(config["emoji"] + " " + name + ": spread too wide — skip")
            continue

        # News blackout
        news_active, news_reason = calendar.is_news_time(name)
        if news_active:
            scan_results.append(config["emoji"] + " " + name + ": PAUSED — " + news_reason)
            continue

        # ── HYBRID SIGNAL ANALYSIS ────────────────────────────
        score, direction, details = signals.analyze(asset=config["asset"])
        log.info(name + ": score=" + str(score) + " dir=" + direction + " | " + details)

        if score < settings["signal_threshold"] or direction == "NONE":
            scan_results.append(
                config["emoji"] + " " + name + ": " +
                str(score) + "/8 — no setup yet"
            )
            continue

        # ── POSITION SIZING — Reduce 50% on wide CPR days ────
        cpr_levels = cpr_calc.get_levels(config["instrument"])
        is_wide    = cpr_levels["is_wide"] if cpr_levels else False
        size       = config["lot_size"] // 2 if is_wide else config["lot_size"]
        if is_wide:
            log.info(name + " Wide CPR day — position size halved to " + str(size))

        # ── TAKE PROFIT — CPR R1/S1 or 2x ATR ────────────────
        price, _, _ = trader.get_price(name)
        cpr_tp_pips, cpr_tp_target = None, None
        if cpr_levels and price:
            cpr_tp_pips, cpr_tp_target = cpr_calc.get_cpr_tp(config["instrument"], direction, price)

        atr_tp_pips = get_atr_pips(trader, name, config["pip"], multiplier=2.0)

        # Use CPR TP if valid and closer than 2x ATR — ensures realistic daily targets
        if cpr_tp_pips and atr_tp_pips:
            tp_pips  = min(cpr_tp_pips, atr_tp_pips)
            tp_label = "CPR R1/S1" if cpr_tp_pips < atr_tp_pips else "2x ATR"
        elif cpr_tp_pips:
            tp_pips  = cpr_tp_pips
            tp_label = "CPR R1/S1"
        elif atr_tp_pips:
            tp_pips  = atr_tp_pips
            tp_label = "2x ATR"
        else:
            tp_pips  = config["tp_pips"]
            tp_label = "Fixed"

        stop_pips  = config["stop_pips"]
        max_loss   = round(size * stop_pips * config["pip"], 2)
        max_profit = round(size * tp_pips   * config["pip"], 2)

        log.info(
            name + " size=" + str(size) +
            " stop=" + str(stop_pips) +
            " tp=" + str(tp_pips) + " (" + tp_label + ")" +
            " wide_cpr=" + str(is_wide)
        )

        # ── PLACE ORDER ───────────────────────────────────────
        result = trader.place_order(
            instrument     = name,
            direction      = direction,
            size           = size,
            stop_distance  = stop_pips,
            limit_distance = tp_pips
        )

        if result["success"]:
            today["trades"]        += 1
            today["consec_losses"]  = 0
            # Reset breakeven tracker for new trade
            today["breakeven_" + name] = False
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)

            price, _, _ = trader.get_price(name)
            cpr_summary = cpr_calc.summary_text(config["instrument"]) if cpr_levels else "CPR: unavailable"
            size_note   = " (50% — wide CPR day)" if is_wide else ""

            alert.send(
                "💥 HYBRID TRADE! " + mode + "\n"
                + config["emoji"] + " " + name + "\n"
                "Strategy: CPR + Breakout Momentum\n"
                "Direction: " + direction + "\n"
                "Score:    " + str(score) + "/8\n"
                "Entry:    " + str(round(price, config["precision"])) + "\n"
                "Size:     " + str(size) + " units" + size_note + "\n"
                "Stop:     " + str(stop_pips) + " pips = $" + str(max_loss) + "\n"
                "Target:   " + str(tp_pips) + " pips = $" + str(max_profit) + " (" + tp_label + ")\n"
                "R:R:      1:" + str(round(tp_pips / stop_pips, 1)) + "\n"
                "Spread:   " + str(round(spread_val, 1)) + " pips\n"
                "Trade #" + str(today["trades"]) + "/" + str(settings["max_trades_day"]) + "\n"
                "Session:  " + session + "\n"
                "─── CPR Levels ───\n"
                + cpr_summary + "\n"
                "─── Signals ───\n"
                + details.replace(" | ", "\n")
            )
            scan_results.append(
                config["emoji"] + " " + name + ": " +
                direction + " PLACED! " + str(score) + "/8"
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
        target_msg = "Scanning for CPR + breakout setups..."

    summary = "\n".join(scan_results) if scan_results else "No setups this scan"
    wins    = today.get("wins", 0)
    losses  = today.get("losses", 0)
    consec  = today.get("consec_losses", 0)

    # Build CPR summary line for status message
    cpr_status = ""
    if cpr_gold:
        cpr_status += "🥇 CPR Width=" + str(cpr_gold["width_pct"]) + "% " + ("⚡ NARROW" if cpr_gold["is_narrow"] else ("⚠️ WIDE" if cpr_gold["is_wide"] else "NORMAL")) + "\n"
    if cpr_gbp:
        cpr_status += "💷 CPR Width=" + str(cpr_gbp["width_pct"]) + "% " + ("⚡ NARROW" if cpr_gbp["is_narrow"] else "NORMAL") + "\n"

    alert.send(
        "💥 HYBRID Scan! " + mode + "\n"
        "Strategy: CPR + Breakout Momentum\n"
        "Time:     " + now.strftime("%H:%M SGT") + "\n"
        "Session:  " + session + "\n"
        "Balance:  $" + str(round(current_balance, 2)) + "\n"
        "Start:    $" + str(round(start_balance, 2)) + "\n"
        "Realized: $" + str(round(realized_pnl, 2)) + " USD " + pnl_emoji + "\n"
        "= $" + str(round(pl_sgd, 2)) + " SGD\n"
        "Open PnL: $" + str(round(open_pnl, 2)) + " USD\n"
        "Total:    $" + str(round(total_pnl, 2)) + " USD\n"
        + target_msg + "\n"
        "Trades: " + str(today["trades"]) + "/" + str(settings["max_trades_day"]) + "\n"
        "W/L: " + str(wins) + "/" + str(losses) + " | Consec loss: " + str(consec) + "\n"
        "─── CPR ───\n"
        + cpr_status +
        "─── Setups ───\n"
        + summary
    )


if __name__ == "__main__":
    run_bot()
