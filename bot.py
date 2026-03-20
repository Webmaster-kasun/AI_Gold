"""
OANDA Trading Bot — Gold Only | CPR + EMA + Volume
===================================================
Pair:     XAU/USD (Gold only)
Sessions: Asian (9am-1pm SGT) | London (2pm-7pm SGT) | NY (8pm-11pm SGT)

CHANGELOG:
  FIX 1  - Removed trade_journal import (next phase)
  FIX 2  - Fixed open_count NameError crash in sync
  FIX 3  - Hard 10-min duplicate lock (stops 4-6 orders/min bug)
  FIX 4  - sync no longer overwrites today["trades"] counter
  FIX 5  - sync no longer overwrites last_trade_entry_price
  FIX 6  - Smart re-entry guard (4 rules)
  FIX 7  - Daily summary at 11pm SGT
  FIX 8  - Breakeven logic: move SL to entry after 1:1 R:R hit
  FIX 9  - SL range widened to 1000–2400 pips (gold's real swings)
  FIX 10 - Wilder RSI in signals.py (matches TradingView values)
  FIX 11 - Real limits in settings: max_trades_day=10, max_consec_losses=10
           spread checks: gold=150p, asian=200p, session caps: asian=3, main=7
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

ASSETS = {
    "XAU_USD": {
        "instrument":    "XAU_USD",
        "asset":         "XAUUSD",
        "emoji":         "🥇",
        "setting":       "trade_gold",
        "pip":           0.01,
        "precision":     2,
        "session_hours": [(9, 23)],
    },
}

RISK_PCT_PER_TRADE = 0.01
RISK_USD_MAX       = 15.0
RISK_USD_MIN       = 1.0


def calc_position_size(balance, stop_pips, pip, score, price):
    try:
        risk_dollars  = min(balance * RISK_PCT_PER_TRADE, RISK_USD_MAX)
        risk_dollars  = max(risk_dollars, RISK_USD_MIN)
        risk_per_unit = stop_pips * pip
        if risk_per_unit <= 0:
            return 1
        scale = 1.0 if score >= 6 else 0.75
        units = max(1, int((risk_dollars / risk_per_unit) * scale))
        log.info(f"Size: bal=${balance:.2f} risk=${risk_dollars:.2f} stop={stop_pips}p units={units} score={score}/7")
        return units
    except Exception as e:
        log.warning(f"Position size error: {e}")
        return 1


def load_settings():
    default = {
        "max_trades_day":         10,
        "signal_threshold":       5,
        "signal_threshold_asian": 4,
        "demo_mode":              True,
        "trade_gold":             True,
        "trade_gold_asian":       True,
        "max_consec_losses":      10,
        "max_spread_gold":        150,
        "max_spread_gold_asian":  200,
        "strategy":               "hybrid_cpr_breakout_gold",
        "max_trades_asian":       3,
        "max_trades_main":        7,
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
    """Sync W/L from OANDA. Does NOT touch trade counter or entry price."""
    try:
        from datetime import timezone
        sg_tz         = pytz.timezone("Asia/Singapore")
        now_sg        = datetime.now(sg_tz)
        day_start     = now_sg.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_utc = day_start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")

        url    = trader.base_url + "/v3/accounts/" + trader.account_id + "/trades"
        params = {"state": "CLOSED", "instrument": "XAU_USD", "count": "20"}
        r      = requests.get(url, headers=trader.headers, params=params, timeout=10)
        if r.status_code != 200:
            return

        trades = r.json().get("trades", [])
        wins = losses = 0
        for t in trades:
            if t.get("closeTime", "") < day_start_utc:
                continue
            pl = float(t.get("realizedPL", 0))
            if pl > 0:   wins   += 1
            elif pl < 0: losses += 1

        today["wins"]   = wins
        today["losses"] = losses
        # FIX 4: Never overwrite today["trades"] — local counter is source of truth

        consec = 0
        for t in sorted(trades, key=lambda x: x.get("closeTime", ""), reverse=True):
            if t.get("closeTime", "") < day_start_utc:
                break
            if float(t.get("realizedPL", 0)) < 0:
                consec += 1
            else:
                break
        today["consec_losses"] = consec

        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)

        today_closed = [t for t in trades if t.get("closeTime", "") >= day_start_utc]
        if today_closed:
            latest = sorted(today_closed, key=lambda x: x.get("closeTime", ""))[-1]
            today["last_trade_close_time"]   = latest.get("closeTime", "")
            today["last_trade_close_result"] = "WIN" if float(latest.get("realizedPL", 0)) > 0 else "LOSS"
            # FIX 5: Do NOT overwrite last_trade_entry_price here

        log.info("Synced W=" + str(wins) + " L=" + str(losses) + " consec=" + str(consec))

    except Exception as e:
        log.warning("Sync trades error: " + str(e))


def get_atr_pips(trader, instrument, pip, multiplier=1.0):
    try:
        url    = trader.base_url + "/v3/instruments/" + instrument + "/candles"
        params = {"count": "30", "granularity": "H1", "price": "M"}
        r      = requests.get(url, headers=trader.headers, params=params, timeout=10)
        if r.status_code != 200:
            return None
        c      = [x for x in r.json()["candles"] if x["complete"]]
        if len(c) < 15:
            return None
        highs  = [float(x["mid"]["h"]) for x in c]
        lows   = [float(x["mid"]["l"]) for x in c]
        closes = [float(x["mid"]["c"]) for x in c]
        trs    = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
                  for i in range(1, len(closes))]
        atr      = sum(trs[-14:]) / 14
        atr_pips = (atr / pip) * multiplier
        log.info(instrument + " ATR=" + str(round(atr, 4)) + " pips=" + str(round(atr_pips, 0)))
        return max(round(atr_pips), 10)
    except Exception as e:
        log.warning("ATR error: " + str(e))
        return None


def check_spread(trader, instrument, max_spread_pips, pip):
    try:
        mid, bid, ask = trader.get_price(instrument)
        if bid is None:
            return True, 0
        spread_pips = (ask - bid) / pip
        log.info(instrument + " spread=" + str(round(spread_pips, 1)) + " pips")
        return (spread_pips <= max_spread_pips), spread_pips
    except Exception as e:
        log.warning("Spread error: " + str(e))
        return True, 0


def check_and_move_breakeven(trader, today, trade_log, instrument, pip, precision):
    """
    FIX 8: Breakeven logic.
    After a trade is open, check if price has moved 1:1 in our favour.
    If yes — move SL to entry price (breakeven) so we can't lose on this trade.
    Runs every scan cycle for any open position.

    State tracked in today["breakeven_XAU_USD"]:
      False  = not yet moved to breakeven
      True   = already moved, do nothing more
    """
    be_key = "breakeven_" + instrument.replace("_", "")  # "breakeven_XAUUSD"
    if today.get(be_key, False):
        log.info(instrument + " breakeven already set — skipping")
        return

    entry_price = today.get("last_trade_entry_price")
    entry_dir   = today.get("last_trade_entry_direction", "")
    stop_pips   = today.get("last_trade_stop_pips", 0)

    if not entry_price or not entry_dir or stop_pips <= 0:
        return

    position = trader.get_position(instrument)
    if not position:
        return  # no open position

    price, _, _ = trader.get_price(instrument)
    if price is None:
        return

    # How many pips has price moved in our direction?
    if entry_dir == "BUY":
        pips_profit = (price - entry_price) / pip
    else:
        pips_profit = (entry_price - price) / pip

    log.info(instrument + " breakeven check | dir=" + entry_dir +
             " entry=" + str(entry_price) + " price=" + str(price) +
             " profit_pips=" + str(round(pips_profit)) +
             " need=" + str(stop_pips) + "p for 1:1")

    if pips_profit < stop_pips:
        return  # not at 1:1 yet

    # ── Move SL to breakeven ──────────────────────────────────
    try:
        # Get the open trade ID from OANDA
        url    = trader.base_url + "/v3/accounts/" + trader.account_id + "/trades"
        params = {"state": "OPEN", "instrument": instrument}
        r      = requests.get(url, headers=trader.headers, params=params, timeout=10)
        if r.status_code != 200:
            log.warning("Breakeven: could not fetch open trades")
            return

        open_trades = r.json().get("trades", [])
        if not open_trades:
            return

        trade_id = open_trades[0]["id"]
        be_price = round(entry_price, precision)

        # Patch SL to entry price
        patch_url = trader.base_url + "/v3/accounts/" + trader.account_id + "/trades/" + trade_id + "/orders"
        payload   = {
            "stopLoss": {
                "price":       str(be_price),
                "timeInForce": "GTC"
            }
        }
        time.sleep(1.0)
        patch_r = requests.put(patch_url, headers=trader.headers, json=payload, timeout=15)

        if patch_r.status_code in [200, 201]:
            today[be_key] = True
            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)
            log.info(instrument + " ✅ BREAKEVEN SET @ " + str(be_price) +
                     " (profit was " + str(round(pips_profit)) + "p)")
            from telegram_alert import TelegramAlert
            TelegramAlert().send(
                "🛡️ BREAKEVEN SET — " + instrument + "\n"
                "SL moved to entry @ " + str(be_price) + "\n"
                "Profit locked: " + str(round(pips_profit)) + "p in favour\n"
                "Trade is now risk-free ✅"
            )
        else:
            log.warning("Breakeven patch failed: " + str(patch_r.status_code) + " " + patch_r.text[:200])

    except Exception as e:
        log.warning("Breakeven error: " + str(e))


def send_daily_summary(alert, today, cpr_gold, mode):
    """Send P&L summary at 11pm SGT."""
    try:
        wins         = today.get("wins", 0)
        losses       = today.get("losses", 0)
        total        = wins + losses
        win_rate     = round((wins / total * 100)) if total > 0 else 0
        realized     = today.get("daily_pnl", 0.0)
        realized_sgd = round(realized * 1.35, 2)
        pnl_emoji    = "UP" if realized >= 0 else "DOWN"
        wr_emoji     = "GREEN" if win_rate >= 60 else ("YELLOW" if win_rate >= 40 else "RED")

        cpr_line = ""
        if cpr_gold:
            w     = cpr_gold.get("width_pct", 0)
            w_lbl = "NARROW-trending" if cpr_gold["is_narrow"] else ("WIDE-choppy" if cpr_gold["is_wide"] else "NORMAL")
            cpr_line = (
                "\n--- Tomorrow CPR ---\n"
                "TC=" + str(cpr_gold["tc"]) + " BC=" + str(cpr_gold["bc"]) + "\n"
                "R1=" + str(cpr_gold["r1"]) + " S1=" + str(cpr_gold["s1"]) + "\n"
                "Width=" + str(w) + "% " + w_lbl + "\n"
            )

        msg = (
            "📊 GOLD BOT Daily Summary\n"
            "-------------------------\n"
            "Mode:     " + mode + "\n"
            "-------------------------\n"
            "Trades:   " + str(total) + "\n"
            "W / L:    " + str(wins) + " / " + str(losses) + "\n"
            "Win Rate: " + wr_emoji + " " + str(win_rate) + "%\n"
            "-------------------------\n"
            "P&L: " + pnl_emoji + " $" + str(round(realized, 2)) + " USD\n"
            "     " + pnl_emoji + " $" + str(realized_sgd) + " SGD"
            + cpr_line +
            "-------------------------\n"
            "Bot resumes 9am SGT tomorrow"
        )
        alert.send(msg)
        log.info("Daily summary sent")
    except Exception as e:
        log.warning("Daily summary error: " + str(e))


def run_bot():
    log.info("🥇 GOLD BOT scanning...")
    settings = load_settings()
    sg_tz    = pytz.timezone("Asia/Singapore")
    now      = datetime.now(sg_tz)
    alert    = TelegramAlert()
    cpr_calc = CPRCalculator(demo=settings["demo_mode"])
    hour     = now.hour

    active_hours = (9 <= hour <= 23)
    london_open  = (14 <= hour <= 17)
    london       = (14 <= hour <= 19)
    ny_overlap   = (20 <= hour <= 23)
    asian        = (9 <= hour <= 13)
    good_session = active_hours

    if asian:
        session = "Asian Session (SGX/Tokyo 9am-1pm SGT)"
    elif london_open:
        session = "London Open (BEST for Gold breakouts!)"
    elif ny_overlap:
        session = "NY Overlap (BEST for Gold macro moves!)"
    elif london:
        session = "London Session"
    else:
        session = "Off-hours (monitoring only)"

    if now.weekday() == 5:
        log.info("Saturday — markets closed")
        return
    if now.weekday() == 6 and hour < 9:
        log.info("Sunday early — skipping")
        return

    trader = OandaTrader(demo=settings["demo_mode"])
    if not trader.login():
        alert.send(
            "❌ OANDA Login Failed\n"
            "Check OANDA_API_KEY and OANDA_ACCOUNT_ID\n"
            "demo_mode=true  -> practice account\n"
            "demo_mode=false -> live account"
        )
        return

    current_balance = trader.last_balance
    mode            = "DEMO" if settings["demo_mode"] else "LIVE"

    trade_log = "trades_" + now.strftime("%Y%m%d") + ".json"
    try:
        with open(trade_log) as f:
            today = json.load(f)
    except FileNotFoundError:
        today = {
            "trades":                     0,
            "start_balance":              current_balance,
            "daily_pnl":                  0.0,
            "stopped":                    False,
            "wins":                       0,
            "losses":                     0,
            "consec_losses":              0,
            "cooldowns":                  {},
            "cpr_alert_sent":             False,
            "cpr_alert_asian_sent":       False,
            "news_alert_sent":            False,
            "daily_summary_sent":         False,
            "last_trade_close_time":      None,
            "last_trade_close_result":    None,
            "last_trade_entry_price":     None,
            "last_trade_entry_time":      None,
            "last_trade_entry_score":     0,
            "last_trade_entry_direction": "",
            "last_trade_stop_pips":       0,
            "asian_trades_today":         0,
            "main_trades_today":          0,
            # FIX 8: breakeven flags per instrument
            "breakeven_XAUUSD":           False,
        }
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)
        log.info("New day! Start balance: $" + str(round(current_balance, 2)))

    start_balance = today.get("start_balance", current_balance)
    open_pnl      = 0.0
    for _n in ASSETS:
        _pos = trader.get_position(_n)
        if _pos:
            open_pnl += trader.check_pnl(_pos)
    realized_pnl = current_balance - start_balance
    pl_sgd       = realized_pnl * 1.35
    pnl_emoji    = "✅" if realized_pnl >= 0 else "❌"

    today["daily_pnl"] = realized_pnl
    with open(trade_log, "w") as f:
        json.dump(today, f, indent=2)

    sync_closed_trades(trader, today, trade_log)

    # ── FIX 8: Check breakeven on any open position ──────────
    for name, config in ASSETS.items():
        position = trader.get_position(name)
        if position:
            check_and_move_breakeven(
                trader, today, trade_log,
                name, config["pip"], config["precision"]
            )

    # Daily summary at 11pm SGT
    if hour == 23 and not today.get("daily_summary_sent", False):
        cpr_for_summary = cpr_calc.get_levels("XAU_USD")
        send_daily_summary(alert, today, cpr_for_summary, mode)
        today["daily_summary_sent"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)

    if today["trades"] >= settings["max_trades_day"]:
        log.info("Max trades reached")
        return

    cpr_gold = cpr_calc.get_levels("XAU_USD")

    send_cpr_alert = (
        (asian and hour == 9 and not today.get("cpr_alert_asian_sent")) or
        (london_open and hour == 14 and not today.get("cpr_alert_sent"))
    )
    if send_cpr_alert:
        session_label = "Asian Open" if asian else "London Open"
        cpr_msg = "🌅 GOLD BOT — " + session_label + " CPR Levels\n"
        if cpr_gold:
            narrow_flag = " NARROW — TRENDING DAY!" if cpr_gold["is_narrow"] else ""
            wide_flag   = " WIDE — CHOPPY" if cpr_gold["is_wide"] else ""
            cpr_msg += (
                "🥇 GOLD CPR" + narrow_flag + wide_flag + "\n"
                "TC=" + str(cpr_gold["tc"]) + " BC=" + str(cpr_gold["bc"]) +
                " Pivot=" + str(cpr_gold["pivot"]) + "\n"
                "R1=" + str(cpr_gold["r1"]) + " S1=" + str(cpr_gold["s1"]) +
                " Width=" + str(cpr_gold["width_pct"]) + "%"
            )
        alert.send(cpr_msg)
        if asian:
            today["cpr_alert_asian_sent"] = True
        else:
            today["cpr_alert_sent"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)

    if not good_session:
        log.info("Off-hours — sleeping silently")
        return

    calendar     = EconomicCalendar()
    news_summary = calendar.get_today_summary()
    if "No high" not in news_summary and not today.get("news_alert_sent"):
        alert.send("⚠️ NEWS ALERT!\n" + news_summary + "\nCPR levels often break around news!")
        today["news_alert_sent"] = True
        with open(trade_log, "w") as f:
            json.dump(today, f, indent=2)

    signals      = SignalEngine(demo=settings["demo_mode"])
    scan_results = []
    score        = -1
    direction    = ""

    for name, config in ASSETS.items():
        if not settings.get(config["setting"], True):
            continue
        if today["trades"] >= settings["max_trades_day"]:
            break

        position = trader.get_position(name)
        if position:
            pnl     = trader.check_pnl(position)
            pos_dir = "BUY" if int(float(position["long"]["units"])) > 0 else "SELL"
            emoji   = "📈" if pnl > 0 else "📉"
            scan_results.append(config["emoji"] + " " + name + ": " + pos_dir +
                                 " open " + emoji + " $" + str(round(pnl, 2)))
            continue

        session_hours = config.get("session_hours", [(14, 23)])
        pair_ok       = any(s <= hour <= e for (s, e) in session_hours)
        if not pair_ok:
            scan_results.append(config["emoji"] + " " + name + ": off-session")
            continue

        is_asian_gold = asian and name == "XAU_USD"

        if is_asian_gold and not settings.get("trade_gold_asian", True):
            scan_results.append(config["emoji"] + " " + name + ": Asian disabled")
            continue

        if is_asian_gold:
            cap          = settings.get("max_trades_asian", 3)
            asian_trades = today.get("asian_trades_today", 0)
            if asian_trades >= cap:
                scan_results.append(config["emoji"] + " " + name + ": Asian cap reached")
                continue
        else:
            cap         = settings.get("max_trades_main", 7)
            main_trades = today.get("main_trades_today", 0)
            if main_trades >= cap:
                scan_results.append(config["emoji"] + " " + name + ": Main cap reached")
                continue

        # ══════════════════════════════════════════════════════
        # RE-ENTRY GUARD — FIX 3 + FIX 6
        # ══════════════════════════════════════════════════════
        last_entry_time      = today.get("last_trade_entry_time")
        last_entry_score     = today.get("last_trade_entry_score", 0)
        last_entry_direction = today.get("last_trade_entry_direction", "")
        last_entry_price     = today.get("last_trade_entry_price") or 0
        now_utc              = datetime.utcnow()

        # FIX 3: Hard 10-min duplicate lock
        if last_entry_time:
            try:
                entry_dt   = datetime.strptime(last_entry_time[:16].replace("T", " "), "%Y-%m-%d %H:%M")
                mins_since = (now_utc - entry_dt).total_seconds() / 60
                if mins_since < 10:
                    remaining = int(10 - mins_since)
                    scan_results.append(config["emoji"] + " " + name +
                        ": 🔒 Duplicate lock — " + str(remaining) + " min remaining")
                    log.info(name + " duplicate lock — " + str(round(mins_since, 1)) + " min since last order")
                    continue
            except Exception as e:
                log.warning("Duplicate lock error: " + str(e))

        # FIX 6: Smart re-entry rules
        if last_entry_time and last_entry_score > 0 and last_entry_direction:
            try:
                ak                      = "XAUUSD_ASIAN" if is_asian_gold else config["asset"]
                peek_score, peek_dir, _ = signals.analyze(asset=ak)
                same_dir                = (peek_dir == last_entry_direction)
                price_now, _, _         = trader.get_price(name)
                price_moved             = (abs((price_now or 0) - last_entry_price) / config["pip"]) >= 500 if last_entry_price else False

                log.info(name + " re-entry | last=" + last_entry_direction + "@" + str(last_entry_score) +
                         " now=" + peek_dir + "@" + str(peek_score) +
                         " same=" + str(same_dir) + " moved=" + str(price_moved))

                if same_dir and peek_score <= last_entry_score and not price_moved:
                    scan_results.append(config["emoji"] + " " + name +
                        ": 🚫 Chasing — same " + last_entry_direction +
                        " score " + str(peek_score) + " <= " + str(last_entry_score))
                    continue
                elif same_dir and peek_score >= 6:
                    log.info(name + " ALLOWED — stronger score " + str(peek_score))
                    today["last_trade_entry_score"]     = 0
                    today["last_trade_entry_direction"] = ""
                elif not same_dir and peek_score >= 5:
                    log.info(name + " ALLOWED — direction flip to " + peek_dir)
                    today["last_trade_entry_score"]     = 0
                    today["last_trade_entry_direction"] = ""
                elif price_moved and peek_score >= 5:
                    log.info(name + " ALLOWED — new zone 500p+")
                    today["last_trade_entry_score"]     = 0
                    today["last_trade_entry_direction"] = ""
                else:
                    reason = ("same dir " + str(peek_score) + "/7" if same_dir
                              else peek_dir + " score=" + str(peek_score) + "/7 < 5")
                    scan_results.append(config["emoji"] + " " + name +
                        ": ⏳ Re-entry blocked — " + reason)
                    continue

                with open(trade_log, "w") as f:
                    json.dump(today, f, indent=2)

            except Exception as e:
                log.warning("Re-entry guard error: " + str(e))

        max_spread            = settings.get("max_spread_gold_asian", 200) if is_asian_gold else settings.get("max_spread_gold", 150)
        spread_ok, spread_val = check_spread(trader, name, max_spread, config["pip"])

        news_active, news_reason = calendar.is_news_time(name)
        if news_active:
            scan_results.append(config["emoji"] + " " + name + ": PAUSED — " + news_reason)
            continue

        asset_key = "XAUUSD_ASIAN" if is_asian_gold else config["asset"]
        threshold = settings.get("signal_threshold_asian", 4) if is_asian_gold else settings["signal_threshold"]

        score, direction, details = signals.analyze(asset=asset_key)
        log.info(name + ": score=" + str(score) + " dir=" + direction + " | " + details)

        if not spread_ok:
            scan_results.append(config["emoji"] + " " + name +
                ": Spread " + str(round(spread_val, 1)) + " pips | Score: " + str(score) + "/7")
            continue

        if is_asian_gold and score >= 2 and direction == "NONE":
            scan_results.append(config["emoji"] + " " + name + ": Watching for breakout (" + str(score) + "/7)")
            cpr_lvls  = cpr_calc.get_levels("XAU_USD")
            watch_msg = "Watching for breakout\nScore: " + str(score) + "/7 need " + str(threshold) + "\n"
            if cpr_lvls:
                watch_msg += "TC=" + str(cpr_lvls["tc"]) + " BC=" + str(cpr_lvls["bc"]) + "\n"
            watch_msg += details.replace(" | ", "\n")
            alert.send(watch_msg)
            continue

        if score < threshold or direction == "NONE":
            scan_results.append(config["emoji"] + " " + name + ": " + str(score) + "/7 — no setup yet")
            continue

        cpr_levels = cpr_calc.get_levels(config["instrument"])
        is_wide    = cpr_levels.get("is_wide", False) if cpr_levels else False

        price, _, _ = trader.get_price(name)
        raw_atr     = get_atr_pips(trader, name, config["pip"], multiplier=1.0)
        pip         = config["pip"]

        # FIX 10: SL range 900–1100 pips — tighter control matching user expectation.
        # 900p  = ~$9  per unit minimum stop.
        # 1100p = ~$11 per unit maximum stop.
        # ATR-based: clamp to 900–1100. Default 1000p when ATR unavailable.
        if raw_atr:
            stop_pips = max(900, min(raw_atr, 1100))
        else:
            stop_pips = 1000  # safe default when ATR unavailable

        size = calc_position_size(current_balance, stop_pips, pip, score, price)

        # FIX 10: TP capped at 2000–2500 pips to target ~$20–25 USD (~27–33 SGD) per unit.
        # Old code used stop_pips * 3 which produced 3000–7200p TP — way too wide.
        # With stop=900–1100p, a 2:1 R:R gives 1800–2200p TP (~$18–22 per unit).
        # We cap at 2500p to avoid runaway TP targets while still allowing CPR-dynamic targets.
        tp_pips  = min(stop_pips * 2, 2500)   # 2:1 R:R, hard cap 2500 pips
        tp_label = "2:1 R:R capped 2500p (default)"

        if cpr_levels and price:
            r1           = cpr_levels.get("r1", 0)
            s1           = cpr_levels.get("s1", 0)
            target_level = r1 if direction == "BUY" else s1
            if target_level:
                dist = abs(target_level - price) / pip
                if stop_pips * 2 <= dist <= stop_pips * 4:
                    tp_pips  = min(int(dist), 2500)   # FIX 10: hard cap at 2500p
                    tp_label = ("R1=" + str(r1) if direction == "BUY" else "S1=" + str(s1)) + " (dynamic, capped)"

        rr = tp_pips / stop_pips
        if rr < 2.0:
            scan_results.append(config["emoji"] + " " + name + ": R:R=" + str(round(rr, 1)) + " < 1:2 skip")
            continue

        max_loss   = round(size * stop_pips * pip, 2)
        max_profit = round(size * tp_pips   * pip, 2)

        try:
            mr = requests.get(trader.base_url + "/v3/accounts/" + trader.account_id,
                              headers=trader.headers, timeout=10)
            if mr.status_code == 200:
                acct      = mr.json().get("account", {})
                margin_av = float(acct.get("marginAvailable", current_balance))
                max_units = int((margin_av * 0.8) / (price * 0.05)) if price else size
                if max_units < 1:
                    scan_results.append(config["emoji"] + " " + name + ": Insufficient margin")
                    continue
                if size > max_units:
                    size = max_units
        except Exception as _me:
            log.warning("Margin check error: " + str(_me))

        result = trader.place_order(
            instrument     = name,
            direction      = direction,
            size           = size,
            stop_distance  = stop_pips,
            limit_distance = tp_pips
        )

        if result["success"]:
            today["trades"]                    += 1
            today["consec_losses"]              = 0
            today["last_trade_entry_price"]     = price
            today["last_trade_entry_time"]      = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
            today["last_trade_entry_score"]     = score
            today["last_trade_entry_direction"] = direction
            today["last_trade_stop_pips"]       = stop_pips   # FIX 8: store for breakeven calc
            # FIX 8: reset breakeven flag for new trade
            today["breakeven_XAUUSD"]           = False

            if is_asian_gold:
                today["asian_trades_today"] = today.get("asian_trades_today", 0) + 1
            else:
                today["main_trades_today"]  = today.get("main_trades_today", 0) + 1

            with open(trade_log, "w") as f:
                json.dump(today, f, indent=2)

            cpr_summary = (
                "TC=" + str(cpr_levels["tc"]) + " BC=" + str(cpr_levels["bc"]) +
                " Pivot=" + str(cpr_levels["pivot"]) + "\n" +
                "R1=" + str(cpr_levels["r1"]) + " S1=" + str(cpr_levels["s1"]) +
                " Width=" + str(cpr_levels["width_pct"]) + "%"
            ) if cpr_levels else "CPR: unavailable"

            size_note = " (wide CPR)" if is_wide else ""
            alert.send(
                "🥇 GOLD TRADE! " + mode + "\n"
                + config["emoji"] + " " + name + "\n"
                "Direction: " + direction + "\n"
                "Score:    " + str(score) + "/7\n"
                "Entry:    " + str(round(price, config["precision"])) + "\n"
                "Size:     " + str(size) + " units" + size_note + "\n"
                "Stop:     " + str(stop_pips) + "p = $" + str(max_loss) + "\n"
                "Target:   " + str(tp_pips) + "p = $" + str(max_profit) + " (" + tp_label + ")\n"
                "R:R:      1:" + str(round(tp_pips / stop_pips, 1)) + "\n"
                "Spread:   " + str(round(spread_val, 1)) + "p\n"
                "Trade #"   + str(today["trades"]) + "/" + str(settings["max_trades_day"]) + "\n"
                "Session:  " + session + "\n"
                "🛡️ Breakeven: moves to entry after 1:1 hit\n"
                "--- CPR ---\n" + cpr_summary + "\n"
                "--- Signals ---\n" + details.replace(" | ", "\n")
            )
            scan_results.append(config["emoji"] + " " + name + ": " + direction + " PLACED! " + str(score) + "/7")
        else:
            log.warning(name + " order failed: " + str(result.get("error", "")))
            scan_results.append(config["emoji"] + " " + name + ": order failed — " + str(result.get("error", ""))[:50])

    target_hit = realized_pnl >= 22
    if target_hit:
        target_msg = "TARGET HIT! $" + str(round(pl_sgd, 0)) + " SGD today!"
    elif realized_pnl > 0:
        target_msg = "Profit $" + str(round(pl_sgd, 0)) + " SGD"
    elif realized_pnl < 0:
        target_msg = "Loss $" + str(abs(round(pl_sgd, 0))) + " SGD"
    else:
        target_msg = "Scanning for setups..."

    summary  = "\n".join(scan_results) if scan_results else "No setups this scan"
    wins     = today.get("wins", 0)
    losses   = today.get("losses", 0)
    cpr_line = ""
    if cpr_gold:
        w_flag   = " NARROW" if cpr_gold["is_narrow"] else (" WIDE" if cpr_gold["is_wide"] else "")
        cpr_line = (
            "CPR Width: " + str(cpr_gold["width_pct"]) + "%" + w_flag + "\n"
            "TC=" + str(cpr_gold["tc"]) + " BC=" + str(cpr_gold["bc"]) + "\n"
            "R1=" + str(cpr_gold["r1"]) + " S1=" + str(cpr_gold["s1"]) + "\n"
        )

    threshold_used    = settings.get("signal_threshold_asian", 4) if asian else settings["signal_threshold"]
    trade_just_placed = any("PLACED" in r for r in scan_results)
    last_alert_min    = today.get("last_scan_alert_min", -61)
    last_alert_score  = today.get("last_alert_score", -1)
    last_alert_dir    = today.get("last_alert_direction", "")
    current_min       = now.hour * 60 + now.minute
    mins_since_alert  = current_min - last_alert_min if current_min >= last_alert_min else current_min + 1440 - last_alert_min
    score_changed     = (score != last_alert_score or direction != last_alert_dir)
    should_alert      = trade_just_placed or score_changed or mins_since_alert >= 60

    if should_alert:
        today["last_scan_alert_min"]  = current_min
        today["last_alert_score"]     = score
        today["last_alert_direction"] = direction
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
            "-------------------------\n"
            + cpr_line +
            "--- Setups ---\n"
            + summary
        )
    else:
        log.info("Scan silent — next alert in " + str(60 - mins_since_alert) + " mins")


if __name__ == "__main__":
    log.info("🥇 GOLD BOT starting — scanning every 5 minutes via Railway...")
    while True:
        try:
            run_bot()
        except Exception as e:
            log.error("Bot error: " + str(e))
        log.info("Sleeping 5 minutes...")
        time.sleep(300)
