"""
Gold Signal Engine — 7-Check Professional Entry System
=======================================================
Scoring (7 pts max):
  Check 1 — CPR Breakout    (0–2 pts): Price above TC=BUY, below BC=SELL
  Check 2 — H4 Trend        (block):   H4 EMA20 vs EMA50 — hard block if against trend
  Check 3 — EMA Alignment   (0–1 pt):  H1 EMA20/50 agree with direction
  Check 4 — RSI Momentum    (0–1 pt):  RSI > 55 BUY / RSI < 45 SELL
  Check 5 — PDH/PDL Clear   (0–1 pt):  Price clear of Prior Day High/Low (200p+)
  Check 6 — Not Overextended(0–1 pt):  Price within 800p of EMA20 (not chasing)
  Check 7 — M15 Rejection   (0–1 pt):  Last M15 candle shows rejection at level

  Need 5/7 to trade (London/NY) | 4/7 Asian session
  ATR filter: 500–2500p range (healthy volatility)
"""

import os
import requests
import logging
from cpr import CPRCalculator

log = logging.getLogger(__name__)


class SignalEngine:
    def __init__(self):
        self.api_key  = os.environ.get("OANDA_API_KEY", "")
        self.base_url = "https://api-fxpractice.oanda.com"
        self.headers  = {"Authorization": "Bearer " + self.api_key}
        self.cpr      = CPRCalculator()

    def _fetch_candles(self, instrument, granularity, count=100):
        url    = self.base_url + "/v3/instruments/" + instrument + "/candles"
        params = {"count": str(count), "granularity": granularity, "price": "M"}
        for attempt in range(3):
            try:
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
        """Real-time mid price"""
        try:
            account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
            url    = self.base_url + "/v3/accounts/" + account_id + "/pricing"
            params = {"instruments": instrument}
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
        if len(closes) < period + 1:
            return None
        deltas   = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains    = [d if d > 0 else 0 for d in deltas[-period:]]
        losses   = [-d if d < 0 else 0 for d in deltas[-period:]]
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            return 100.0
        return round(100 - (100 / (1 + avg_gain / avg_loss)), 1)

    def _get_atr_pips(self, closes, highs, lows, period=14):
        if len(closes) < period + 1:
            return None
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            trs.append(tr)
        return round(sum(trs[-period:]) / period / 0.01)

    def _get_prior_day_levels(self):
        """Get yesterday's High and Low from D1 candles"""
        try:
            closes, highs, lows, _, _ = self._fetch_candles("XAU_USD", "D", 3)
            if len(highs) >= 2 and len(lows) >= 2:
                pdh = highs[-2]   # yesterday's high
                pdl = lows[-2]    # yesterday's low
                log.info("PDH=" + str(pdh) + " PDL=" + str(pdl))
                return pdh, pdl
        except Exception as e:
            log.warning("PDH/PDL error: " + str(e))
        return None, None

    def _check_m15_rejection(self, direction):
        """
        Check if last M15 candle shows a rejection wick
        SELL: upper wick > 40% of candle range = rejection at top
        BUY:  lower wick > 40% of candle range = rejection at bottom
        """
        try:
            closes, highs, lows, opens, _ = self._fetch_candles("XAU_USD", "M15", 5)
            if not closes or len(closes) < 2:
                return False, "No M15 data"

            # Use last complete candle
            h = highs[-1]
            l = lows[-1]
            o = opens[-1]
            c = closes[-1]
            total_range = h - l
            if total_range < 0.01:
                return False, "M15 candle too small"

            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            upper_pct  = upper_wick / total_range
            lower_pct  = lower_wick / total_range

            if direction == "SELL" and upper_pct >= 0.40:
                return True, "M15 upper wick=" + str(round(upper_pct*100)) + "% — rejection at top ✅"
            elif direction == "BUY" and lower_pct >= 0.40:
                return True, "M15 lower wick=" + str(round(lower_pct*100)) + "% — rejection at bottom ✅"
            else:
                if direction == "SELL":
                    return False, "M15 upper wick only " + str(round(upper_pct*100)) + "% — no rejection"
                else:
                    return False, "M15 lower wick only " + str(round(lower_pct*100)) + "% — no rejection"
        except Exception as e:
            log.warning("M15 rejection error: " + str(e))
            return False, "M15 check failed"

    def analyze(self, asset="XAUUSD"):
        if asset == "XAUUSD_ASIAN":
            return self._analyze_gold(is_asian=True)
        return self._analyze_gold(is_asian=False)

    def _analyze_gold(self, is_asian=False):
        reasons   = []
        score     = 0
        direction = "NONE"
        threshold = 4 if is_asian else 5

        # Fetch all timeframes
        h4_closes, _, _, _, _               = self._fetch_candles("XAU_USD", "H4", 60)
        h1_closes, h1_highs, h1_lows, _, _  = self._fetch_candles("XAU_USD", "H1", 60)

        if not h1_closes:
            return 0, "NONE", "No price data"

        # Live price
        price = self._get_live_price("XAU_USD")
        if price is None:
            price = h1_closes[-1]
            log.warning("Using H1 close — live price unavailable")

        # ── ATR FILTER ────────────────────────────────────────
        atr_pips = self._get_atr_pips(h1_closes, h1_highs, h1_lows)
        if atr_pips is not None:
            log.info("ATR=" + str(atr_pips) + "p")
            min_atr = 300 if is_asian else 500
            if atr_pips < min_atr:
                return 0, "NONE", "ATR=" + str(atr_pips) + "p — too quiet, skip"
            if atr_pips > 2500:
                return 0, "NONE", "ATR=" + str(atr_pips) + "p — too volatile, skip"
            reasons.append("✅ ATR=" + str(atr_pips) + "p — healthy volatility")

        # ── H4 TREND FILTER (hard block) ─────────────────────
        h4_direction = "NONE"
        if len(h4_closes) >= 50:
            h4_ema20 = self._ema(h4_closes, 20)[-1]
            h4_ema50 = self._ema(h4_closes, 50)[-1]
            if h4_ema20 > h4_ema50:
                h4_direction = "BUY"
            elif h4_ema20 < h4_ema50:
                h4_direction = "SELL"
            log.info("H4 trend=" + h4_direction + " EMA20=" + str(round(h4_ema20,2)) + " EMA50=" + str(round(h4_ema50,2)))
        else:
            return 0, "NONE", "H4 data insufficient — skipping unfiltered trade"

        # ── CHECK 1: CPR POSITION (0–2 pts) ──────────────────
        cpr = self.cpr.get_levels("XAU_USD")
        if not cpr:
            return 0, "NONE", "CPR levels unavailable"

        tc = cpr["tc"]
        bc = cpr["bc"]
        r1 = cpr["r1"]
        s1 = cpr["s1"]

        log.info("CPR TC=" + str(tc) + " BC=" + str(bc) + " price=" + str(price))

        if price > tc:
            direction = "BUY"
            score    += 2
            reasons.append("✅ Price " + str(price) + " above TC=" + str(tc) + " → BUY (2 pts)")
        elif price < bc:
            direction = "SELL"
            score    += 2
            reasons.append("✅ Price " + str(price) + " below BC=" + str(bc) + " → SELL (2 pts)")
        else:
            reasons.append("❌ Price inside CPR (" + str(bc) + "–" + str(tc) + ") — no trade")
            return 0, "NONE", " | ".join(reasons)

        # ── H4 HARD BLOCK ────────────────────────────────────
        if h4_direction != "NONE" and direction != h4_direction:
            reasons.append("🚫 H4 trend=" + h4_direction + " blocks " + direction + " signal")
            return score, "NONE", " | ".join(reasons)
        elif h4_direction != "NONE":
            reasons.append("✅ H4 trend=" + h4_direction + " confirms direction")

        # ── CHECK 3: EMA ALIGNMENT (0–1 pt) ──────────────────
        if len(h1_closes) >= 50:
            ema20 = self._ema(h1_closes, 20)[-1]
            ema50 = self._ema(h1_closes, 50)[-1]
            log.info("EMA20=" + str(round(ema20,2)) + " EMA50=" + str(round(ema50,2)))
            if direction == "BUY" and price > ema20 and ema20 > ema50:
                score += 1
                reasons.append("✅ EMA: price > EMA20=" + str(round(ema20,2)) + " > EMA50=" + str(round(ema50,2)) + " (1 pt)")
            elif direction == "SELL" and price < ema20 and ema20 < ema50:
                score += 1
                reasons.append("✅ EMA: price < EMA20=" + str(round(ema20,2)) + " < EMA50=" + str(round(ema50,2)) + " (1 pt)")
            else:
                reasons.append("❌ EMA conflict: EMA20=" + str(round(ema20,2)) + " EMA50=" + str(round(ema50,2)) + " (0 pts)")
        else:
            ema20 = price  # fallback
            reasons.append("❌ EMA: not enough H1 data (0 pts)")

        # ── CHECK 4: RSI MOMENTUM (0–1 pt) ───────────────────
        rsi_val = self._calc_rsi(h1_closes, 14)
        if rsi_val is not None:
            log.info("RSI=" + str(rsi_val))
            rsi_buy  = 52 if is_asian else 55
            rsi_sell = 48 if is_asian else 45
            if direction == "BUY" and rsi_val > rsi_buy:
                score += 1
                reasons.append("✅ RSI=" + str(rsi_val) + " > " + str(rsi_buy) + " — bullish (1 pt)")
            elif direction == "SELL" and rsi_val < rsi_sell:
                score += 1
                reasons.append("✅ RSI=" + str(rsi_val) + " < " + str(rsi_sell) + " — bearish (1 pt)")
            else:
                reasons.append("❌ RSI=" + str(rsi_val) + " — no momentum (0 pts)")
        else:
            reasons.append("❌ RSI: not enough data (0 pts)")

        # ── CHECK 5: PDH/PDL CLEAR (0–1 pt) ──────────────────
        # Price must be 200p+ AWAY from Prior Day High (SELL) or Low (BUY)
        # Entering near PDH when selling = selling into resistance = bad
        pdh, pdl = self._get_prior_day_levels()
        if pdh and pdl:
            pip = 0.01
            if direction == "SELL":
                dist_from_pdh = (pdh - price) / pip  # positive = price below PDH (good for sell)
                if dist_from_pdh > 200:
                    score += 1
                    reasons.append("✅ PDH=" + str(pdh) + " | price " + str(int(dist_from_pdh)) + "p below — clear for SELL (1 pt)")
                elif dist_from_pdh < 0:
                    reasons.append("❌ Price ABOVE PDH=" + str(pdh) + " — SELL too risky near resistance (0 pts)")
                else:
                    reasons.append("❌ Price only " + str(int(dist_from_pdh)) + "p below PDH=" + str(pdh) + " — too close (0 pts)")
            elif direction == "BUY":
                dist_from_pdl = (price - pdl) / pip  # positive = price above PDL (good for buy)
                if dist_from_pdl > 200:
                    score += 1
                    reasons.append("✅ PDL=" + str(pdl) + " | price " + str(int(dist_from_pdl)) + "p above — clear for BUY (1 pt)")
                elif dist_from_pdl < 0:
                    reasons.append("❌ Price BELOW PDL=" + str(pdl) + " — BUY too risky near support (0 pts)")
                else:
                    reasons.append("❌ Price only " + str(int(dist_from_pdl)) + "p above PDL=" + str(pdl) + " — too close (0 pts)")
        else:
            reasons.append("⚠️ PDH/PDL unavailable — skipping check (0 pts)")

        # ── CHECK 6: NOT OVEREXTENDED (0–1 pt) ───────────────
        # Price must be within 800p of EMA20
        # More than 800p away = chasing = likely reversal
        ema20_dist = abs(price - ema20) / 0.01
        log.info("Distance from EMA20: " + str(round(ema20_dist)) + "p")
        if ema20_dist <= 800:
            score += 1
            reasons.append("✅ EMA20 dist=" + str(int(ema20_dist)) + "p ≤ 800p — not overextended (1 pt)")
        else:
            reasons.append("❌ EMA20 dist=" + str(int(ema20_dist)) + "p > 800p — overextended, likely reversal (0 pts)")

        # ── CHECK 7: M15 REJECTION CANDLE (0–1 pt) ───────────
        # Wait for M15 to confirm rejection at key level
        # This is the entry timing filter — prevents entering mid-air
        m15_ok, m15_reason = self._check_m15_rejection(direction)
        if m15_ok:
            score += 1
            reasons.append("✅ M15 rejection confirmed: " + m15_reason + " (1 pt)")
        else:
            reasons.append("❌ M15: " + m15_reason + " (0 pts) — no confirmation yet")

        reasons.append("R1=" + str(r1) + " S1=" + str(s1))
        log.info("Score=" + str(score) + "/7 direction=" + direction + " threshold=" + str(threshold))
        return score, direction, " | ".join(reasons)
