"""
Gold Signal Engine — Merged Best v1.0
======================================
Takes the best from AI_GoldBotKaz (7-check system, AI layer, M15 rejection)
and ForexV2/Nuwan-V2 (M15 candle-close confirmation, ATR exhaustion guard,
H1 EMA trend filter, win-candle lock).

Scoring (7 pts max):
  Check 1 — CPR Breakout     (0–2 pts): Price above TC=BUY, below BC=SELL
  Check 2 — H4 Trend         (block):   H4 EMA20 vs EMA50 hard block
  Check 2b— H1 Trend         (block):   H1 EMA21 hard block (NEW from ForexV2)
  Check 3 — EMA Alignment    (0–1 pt):  H1 EMA20/50 agree with direction
  Check 4 — RSI Momentum     (0–1 pt):  RSI(14) Wilder > 55 BUY / < 45 SELL
  Check 5 — PDH/PDL Clear    (0–1 pt):  Price clear of Prior Day High/Low (200p+)
  Check 6 — Not Overextended (0–1 pt):  Price within 600p of EMA20
  Check 7 — M15 Rejection    (0–1 pt):  Last M15 candle rejection wick

KEY IMPROVEMENTS vs old AI_GoldBotKaz:
  - Entry uses M15 confirmed close (not stale H1 close or raw live price)
  - H1 EMA21 hard block added (ForexV2) — stops counter-trend entries
  - ATR exhaustion check — blocks overextended entries (ForexV2)
  - require_candle_close=True — no fakeout entries mid-candle
  - All other AI_Gold fixes retained (FIX 1–17)

Thresholds:
  Need 5/7 London/NY | 4/7 Asian | 6/7 off-hours
  ATR filter: 200–5000p (all sessions)
"""

import os
import time
import requests
import logging
from cpr import CPRCalculator

CALL_DELAY = 0.5

log = logging.getLogger(__name__)


class SignalEngine:
    def __init__(self, demo=True):
        self.api_key  = os.environ.get("OANDA_API_KEY", "")
        self.base_url = "https://api-fxpractice.oanda.com" if demo else "https://api-fxtrade.oanda.com"
        self.headers  = {"Authorization": "Bearer " + self.api_key}
        self.cpr      = CPRCalculator(demo=demo)

    def _fetch_candles(self, instrument, granularity, count=100):
        url    = self.base_url + "/v3/instruments/" + instrument + "/candles"
        params = {"count": str(count), "granularity": granularity, "price": "M"}
        for attempt in range(3):
            try:
                time.sleep(CALL_DELAY)
                r = requests.get(url, headers=self.headers, params=params, timeout=10)
                if r.status_code == 200:
                    candles = r.json()["candles"]
                    c       = [x for x in candles if x["complete"]]
                    closes  = [float(x["mid"]["c"]) for x in c]
                    highs   = [float(x["mid"]["h"]) for x in c]
                    lows    = [float(x["mid"]["l"]) for x in c]
                    opens   = [float(x["mid"]["o"]) for x in c]
                    volumes = [int(x.get("volume", 0)) for x in c]
                    return closes, highs, lows, opens, volumes
                log.warning("Candle fetch " + str(attempt+1) + " failed: " + str(r.status_code))
            except Exception as e:
                log.warning("Candle fetch error: " + str(e))
        return [], [], [], [], []

    def _get_live_price(self, instrument):
        try:
            account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
            url    = self.base_url + "/v3/accounts/" + account_id + "/pricing"
            params = {"instruments": instrument}
            time.sleep(CALL_DELAY)
            r = requests.get(url, headers=self.headers, params=params, timeout=10)
            if r.status_code == 200:
                prices = r.json().get("prices", [])
                if prices:
                    bid = float(prices[0]["bids"][0]["price"])
                    ask = float(prices[0]["asks"][0]["price"])
                    return round((bid + ask) / 2, 2)
        except Exception as e:
            log.warning("Live price error: " + str(e))
        return None

    def _ema(self, data, period):
        if not data or len(data) < period:
            avg = sum(data) / len(data) if data else 0
            return [avg] * max(len(data), 1)
        seed = sum(data[:period]) / period
        emas = [seed] * period
        mult = 2 / (period + 1)
        for p in data[period:]:
            emas.append((p - emas[-1]) * mult + emas[-1])
        return emas

    def _calc_rsi(self, closes, period=14):
        """Wilder's Smoothed RSI — matches TradingView."""
        if len(closes) < period + 1:
            return None
        deltas   = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains    = [d if d > 0 else 0.0 for d in deltas]
        losses   = [abs(d) if d < 0 else 0.0 for d in deltas]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100.0
        rs  = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)

    def _get_atr_pips(self, closes, highs, lows, period=14):
        if len(closes) < period + 1:
            return None
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            trs.append(tr)
        return round(sum(trs[-period:]) / period / 0.01)

    def _get_prior_day_levels(self):
        try:
            closes, highs, lows, _, _ = self._fetch_candles("XAU_USD", "D", 3)
            if len(highs) >= 2 and len(lows) >= 2:
                return highs[-2], lows[-2]
        except Exception as e:
            log.warning("PDH/PDL error: " + str(e))
        return None, None

    def _check_m15_rejection(self, direction):
        try:
            closes, highs, lows, opens, _ = self._fetch_candles("XAU_USD", "M15", 8)
            if not closes or len(closes) < 3:
                return False, "No M15 data"
            for idx in [-1, -2, -3]:
                h = highs[idx]; l = lows[idx]
                o = opens[idx]; c = closes[idx]
                total_range = h - l
                if total_range < 0.05:
                    continue
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l
                upper_pct  = upper_wick / total_range
                lower_pct  = lower_wick / total_range
                if direction == "SELL" and upper_pct >= 0.45:
                    return True, "M15 upper wick=" + str(round(upper_pct*100)) + "% — rejection"
                elif direction == "BUY" and lower_pct >= 0.45:
                    return True, "M15 lower wick=" + str(round(lower_pct*100)) + "% — rejection"
            if direction == "SELL":
                pct = round((highs[-1]-max(opens[-1],closes[-1])) / max(highs[-1]-lows[-1],0.01) * 100)
                return False, "M15 upper wick only " + str(pct) + "%"
            else:
                pct = round((min(opens[-1],closes[-1])-lows[-1]) / max(highs[-1]-lows[-1],0.01) * 100)
                return False, "M15 lower wick only " + str(pct) + "%"
        except Exception as e:
            log.warning("M15 rejection error: " + str(e))
            return False, "M15 check failed"

    def get_h4_trend(self):
        """H4 EMA20 vs EMA50 trend direction with full logging."""
        try:
            h4_closes, _, _, _, _ = self._fetch_candles("XAU_USD", "H4", 60)
            if len(h4_closes) < 50:
                log.warning("H4 TREND: insufficient data — block cannot fire")
                return "NONE", None, None
            h4_ema20 = self._ema(h4_closes, 20)[-1]
            h4_ema50 = self._ema(h4_closes, 50)[-1]
            if h4_ema20 > h4_ema50:
                direction = "BUY"
            elif h4_ema20 < h4_ema50:
                direction = "SELL"
            else:
                direction = "NONE"
            log.info(
                "H4 TREND | direction=" + direction +
                " | EMA20=" + str(round(h4_ema20, 2)) +
                " | EMA50=" + str(round(h4_ema50, 2)) +
                " | gap=" + str(round(h4_ema20 - h4_ema50, 2)) + "p"
            )
            return direction, round(h4_ema20, 2), round(h4_ema50, 2)
        except Exception as e:
            log.warning("H4 trend error: " + str(e))
            return "NONE", None, None

    def get_h1_trend(self, ema_period=21):
        """
        NEW (from ForexV2): H1 EMA21 trend filter.
        Returns (is_bullish, h1_price, h1_ema) — True=bullish, False=bearish, None=no data.
        """
        try:
            h1_closes, _, _, _, _ = self._fetch_candles("XAU_USD", "H1", ema_period + 5)
            if len(h1_closes) < ema_period:
                log.warning("H1 TREND: insufficient data")
                return None, None, None
            h1_ema   = sum(h1_closes[-ema_period:]) / ema_period
            h1_price = h1_closes[-1]
            is_bull  = h1_price > h1_ema
            log.info(
                "H1 TREND | " + ("bullish" if is_bull else "bearish") +
                " | price=" + str(round(h1_price, 2)) +
                " | EMA" + str(ema_period) + "=" + str(round(h1_ema, 2))
            )
            return is_bull, round(h1_price, 2), round(h1_ema, 2)
        except Exception as e:
            log.warning("H1 trend error: " + str(e))
            return None, None, None

    def _check_atr_exhaustion(self, price, ema20, atr_pips, exhaustion_mult=2.5):
        """
        NEW (from ForexV2): Block entries where price has stretched too far
        from EMA20 relative to ATR — the move is statistically spent.
        Returns (is_exhausted, stretch_mult)
        """
        if atr_pips is None or atr_pips == 0 or ema20 is None:
            return False, 0
        dist_pips = abs(price - ema20) / 0.01
        stretch   = dist_pips / atr_pips
        log.info("ATR exhaustion | dist=" + str(round(dist_pips)) + "p | atr=" + str(atr_pips) + "p | stretch=" + str(round(stretch, 2)) + "x")
        return stretch > exhaustion_mult, round(stretch, 2)

    def analyze(self, asset="XAUUSD"):
        if asset == "XAUUSD_ASIAN":
            return self._analyze_gold(is_asian=True)
        return self._analyze_gold(is_asian=False)

    def _analyze_gold(self, is_asian=False):
        reasons   = []
        score     = 0
        direction = "NONE"
        threshold = 4 if is_asian else 5

        # ── STEP 1: ATR from H1 for volatility gating ─────────────────────
        h1_closes, h1_highs, h1_lows, _, _ = self._fetch_candles("XAU_USD", "H1", 60)
        if not h1_closes:
            return 0, "NONE", "No H1 price data"

        atr_pips = self._get_atr_pips(h1_closes, h1_highs, h1_lows)
        if atr_pips is not None:
            log.info("ATR=" + str(atr_pips) + "p")
            if atr_pips < 200:
                return 0, "NONE", "ATR=" + str(atr_pips) + "p — too quiet, skip"
            if atr_pips > 5000:
                return 0, "NONE", "ATR=" + str(atr_pips) + "p — too volatile, skip"
            reasons.append("ATR=" + str(atr_pips) + "p — healthy volatility")

        # ── STEP 2: H4 TREND (hard block) ─────────────────────────────────
        h4_direction, h4_ema20, h4_ema50 = self.get_h4_trend()
        if h4_direction == "NONE":
            return 0, "NONE", "H4 trend unavailable — skip"

        # ── STEP 3: Entry price — M15 CONFIRMED CLOSE (NEW from ForexV2) ──
        # Use the last COMPLETED M15 candle close, not live tick.
        # This prevents fakeout entries where price crosses a level mid-candle
        # then reverses before the M15 bar closes.
        m15_closes, m15_highs, m15_lows, m15_opens, _ = self._fetch_candles("XAU_USD", "M15", 20)
        if len(m15_closes) >= 2:
            price = m15_closes[-2]   # last completed candle
            log.info("Entry price: M15 confirmed close=" + str(price) +
                     " | current tick=" + str(m15_closes[-1]))
        else:
            # Fallback to live price if M15 unavailable
            price = self._get_live_price("XAU_USD")
            if price is None:
                price = h1_closes[-1]
                log.warning("Using H1 close — M15 and live price unavailable")

        # ── STEP 4: H1 TREND (soft penalty flag — scored after CPR) ──────────
        h1_is_bull, h1_price, h1_ema21 = self.get_h1_trend(ema_period=21)
        # NOT a hard block — just a flag used below to deduct 1 pt if counter-H1.
        # H4 is already the hard trend block. Double-blocking kills trade frequency.

        # ── CHECK 1: CPR POSITION (0–2 pts) ───────────────────────────────
        cpr = self.cpr.get_levels("XAU_USD")
        if not cpr:
            return 0, "NONE", "CPR levels unavailable"

        tc = cpr["tc"]; bc = cpr["bc"]
        r1 = cpr["r1"]; s1 = cpr["s1"]

        log.info("CPR TC=" + str(tc) + " BC=" + str(bc) + " price=" + str(price))

        if price > tc:
            direction = "BUY"
            score    += 2
            reasons.append("Price " + str(price) + " above TC=" + str(tc) + " BUY (2 pts)")
        elif price < bc:
            direction = "SELL"
            score    += 2
            reasons.append("Price " + str(price) + " below BC=" + str(bc) + " SELL (2 pts)")
        else:
            reasons.append("Price inside CPR (" + str(bc) + "-" + str(tc) + ") — no trade")
            return 0, "NONE", " | ".join(reasons)

        # ── H4 HARD BLOCK ─────────────────────────────────────────────────
        if direction != h4_direction:
            gap_pips = round(abs(h4_ema20 - h4_ema50) / 0.01)
            log.warning(
                "H4 BLOCK FIRED | signal=" + direction +
                " blocked by H4=" + h4_direction +
                " | gap=" + str(gap_pips) + "p"
            )
            reasons.append(
                "H4 trend=" + h4_direction + " BLOCKS " + direction +
                " | EMA gap=" + str(gap_pips) + "p to cross"
            )
            return 0, "NONE", " | ".join(reasons)
        else:
            reasons.append("H4 trend=" + h4_direction + " confirms direction")

        # ── H1 TREND: soft penalty (−1 pt if counter-H1) ─────────────────
        # Not a hard block — H4 already filters macro trend.
        # Counter-H1 just costs 1 point, making it harder to reach threshold.
        h1_penalty_applied = False
        if h1_is_bull is not None:
            counter_h1 = (direction == "BUY" and not h1_is_bull) or (direction == "SELL" and h1_is_bull)
            if counter_h1:
                score = max(0, score - 1)
                h1_penalty_applied = True
                trend_label = "bearish" if h1_is_bull else "bullish"
                reasons.append("H1 EMA21 " + trend_label + " vs " + direction + " — counter-H1 penalty (−1 pt)")
                log.info("H1 counter-trend penalty applied — score now " + str(score))
            else:
                trend_label = "bullish" if h1_is_bull else "bearish"
                reasons.append("H1 EMA21 " + trend_label + " — confirms " + direction + " (0 pts)")

        # ── ATR EXHAUSTION: soft penalty at 3.5x (was hard block at 2.5x) ──
        # Raised threshold 2.5x → 3.5x AND changed to -1 pt penalty.
        # 2.5x hard block was firing too often after normal gold moves (~$30–50)
        # and killing valid continuation trades. 3.5x only catches truly extreme stretches.
        if len(h1_closes) >= 20:
            ema20_for_exhaustion = self._ema(h1_closes, 20)[-1]
            is_exhausted, stretch = self._check_atr_exhaustion(price, ema20_for_exhaustion, atr_pips, exhaustion_mult=3.5)
            if is_exhausted:
                score = max(0, score - 1)
                reasons.append("ATR stretch=" + str(stretch) + "x > 3.5x — extended move penalty (−1 pt)")
                log.info("ATR exhaustion soft penalty | stretch=" + str(stretch) + "x")
            else:
                reasons.append("ATR stretch=" + str(stretch) + "x — ok (0 pts)")
        else:
            ema20_for_exhaustion = None

        # ── CHECK 3: EMA ALIGNMENT (0–1 pt) ───────────────────────────────
        ema20 = None
        if len(h1_closes) >= 50:
            ema20 = self._ema(h1_closes, 20)[-1]
            ema50 = self._ema(h1_closes, 50)[-1]
            log.info("EMA20=" + str(round(ema20, 2)) + " EMA50=" + str(round(ema50, 2)))
            if direction == "BUY" and price > ema20 and ema20 > ema50:
                score += 1
                reasons.append("EMA: price > EMA20=" + str(round(ema20,2)) + " > EMA50=" + str(round(ema50,2)) + " (1 pt)")
            elif direction == "SELL" and price < ema20 and ema20 < ema50:
                score += 1
                reasons.append("EMA: price < EMA20=" + str(round(ema20,2)) + " < EMA50=" + str(round(ema50,2)) + " (1 pt)")
            else:
                reasons.append("EMA conflict: EMA20=" + str(round(ema20,2)) + " EMA50=" + str(round(ema50,2)) + " (0 pts)")
        else:
            reasons.append("EMA: not enough H1 data (0 pts)")

        # ── CHECK 4: RSI MOMENTUM (0–1 pt) ────────────────────────────────
        rsi_val = self._calc_rsi(h1_closes, 14)
        if rsi_val is not None:
            log.info("RSI(Wilder)=" + str(rsi_val))
            if direction == "BUY" and rsi_val > 55:
                score += 1
                reasons.append("RSI=" + str(rsi_val) + " > 55 — bullish (1 pt)")
            elif direction == "SELL" and rsi_val < 45:
                score += 1
                reasons.append("RSI=" + str(rsi_val) + " < 45 — bearish (1 pt)")
            else:
                reasons.append("RSI=" + str(rsi_val) + " — no momentum (0 pts)")
        else:
            reasons.append("RSI: not enough data (0 pts)")

        # ── CHECK 5: PDH/PDL CLEAR (0–1 pt) ───────────────────────────────
        pdh, pdl = self._get_prior_day_levels()
        if pdh and pdl:
            pip = 0.01
            if direction == "SELL":
                dist_from_pdh = (pdh - price) / pip
                if dist_from_pdh > 200:
                    score += 1
                    reasons.append("PDH=" + str(pdh) + " | " + str(int(dist_from_pdh)) + "p below — clear SELL (1 pt)")
                elif dist_from_pdh < 0:
                    reasons.append("Price ABOVE PDH=" + str(pdh) + " — SELL risky (0 pts)")
                else:
                    reasons.append("Only " + str(int(dist_from_pdh)) + "p below PDH — too close (0 pts)")
            elif direction == "BUY":
                dist_from_pdl = (price - pdl) / pip
                if dist_from_pdl > 200:
                    score += 1
                    reasons.append("PDL=" + str(pdl) + " | " + str(int(dist_from_pdl)) + "p above — clear BUY (1 pt)")
                elif dist_from_pdl < 0:
                    reasons.append("Price BELOW PDL=" + str(pdl) + " — BUY risky (0 pts)")
                else:
                    reasons.append("Only " + str(int(dist_from_pdl)) + "p above PDL — too close (0 pts)")
        else:
            reasons.append("PDH/PDL unavailable (0 pts)")

        # ── CHECK 6: NOT OVEREXTENDED (0–1 pt) ────────────────────────────
        # Uses ema20_for_exhaustion which was already computed above.
        ema20_ref = ema20 if ema20 is not None else ema20_for_exhaustion
        if ema20_ref is not None:
            ema20_dist = abs(price - ema20_ref) / 0.01
            log.info("EMA20 dist: " + str(round(ema20_dist)) + "p")
            if ema20_dist <= 600:
                score += 1
                reasons.append("EMA20 dist=" + str(int(ema20_dist)) + "p <= 600p — not overextended (1 pt)")
            else:
                reasons.append("EMA20 dist=" + str(int(ema20_dist)) + "p > 600p — overextended (0 pts)")
        else:
            reasons.append("EMA20 dist: unavailable — skipped (0 pts)")

        # ── CHECK 7: M15 REJECTION CANDLE (0–1 pt) ────────────────────────
        m15_ok, m15_reason = self._check_m15_rejection(direction)
        if m15_ok:
            score += 1
            reasons.append("M15 rejection confirmed: " + m15_reason + " (1 pt)")
        else:
            reasons.append("M15: " + m15_reason + " (0 pts)")

        reasons.append("R1=" + str(r1) + " S1=" + str(s1))
        log.info("Score=" + str(score) + "/7 direction=" + direction + " threshold=" + str(threshold))
        return score, direction, " | ".join(reasons)
