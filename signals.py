"""
Gold Signal Engine — CPR + EMA + Volume
========================================
Simplified logic matching proven working bot style.

Scoring (need 3/5 to trade):
  Check 1 — CPR Position  (0-2 pts): Price broke above TC (bull) or below BC (bear)
  Check 2 — EMA Alignment (0-1 pt):  Price & EMA20/EMA50 agree on direction
  Check 3 — Volume        (0-1 pt):  Current volume > 1.2x average
  Check 4 — PDH/PDL       (0-1 pt):  Price near prior day high/low

Direction set purely by CPR: above TC = BUY, below BC = SELL
Asian session needs 2/5 to trade.
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
                log.warning("Candle fetch attempt " + str(attempt + 1) + " failed: " + str(r.status_code))
            except Exception as e:
                log.warning("Candle fetch error: " + str(e))
        return [], [], [], [], []

    def analyze(self, asset="XAUUSD"):
        if asset == "XAUUSD_ASIAN":
            return self._analyze_gold_asian()
        return self._analyze_gold()

    # ══════════════════════════════════════════════════════════
    # MAIN GOLD ANALYSIS — London / NY sessions
    # ══════════════════════════════════════════════════════════
    def _analyze_gold(self):
        reasons   = []
        score     = 0
        direction = "NONE"

        h1_closes, h1_highs, h1_lows, _, h1_vols = self._fetch_candles("XAU_USD", "H1", 60)
        m15_closes, _, _, _, m15_vols             = self._fetch_candles("XAU_USD", "M15", 30)

        if not h1_closes:
            return 0, "NONE", "No price data"

        price = h1_closes[-1]

        # ── CHECK 1: CPR POSITION (0–2 pts) ──────────────────
        cpr = self.cpr.get_levels("XAU_USD")
        if not cpr:
            return 0, "NONE", "CPR levels unavailable"

        tc = cpr["tc"]
        bc = cpr["bc"]
        r1 = cpr["r1"]
        s1 = cpr["s1"]

        log.info("Gold CPR TC=" + str(tc) + " BC=" + str(bc) + " price=" + str(round(price, 2)))

        if price > tc:
            direction = "BUY"
            score    += 2
            reasons.append("✅ Price " + str(round(price, 2)) + " broke above TC=" + str(tc) + " (2 pts)")
        elif price < bc:
            direction = "SELL"
            score    += 2
            reasons.append("✅ Price " + str(round(price, 2)) + " broke below BC=" + str(bc) + " (2 pts)")
        else:
            reasons.append("❌ Price inside CPR (" + str(bc) + "–" + str(tc) + ") — no trade")
            return 0, "NONE", " | ".join(reasons)

        # ── CHECK 2: EMA ALIGNMENT (0–1 pt) ──────────────────
        if len(h1_closes) >= 50:
            ema20 = self._ema(h1_closes, 20)[-1]
            ema50 = self._ema(h1_closes, 50)[-1]

            log.info("Gold EMA20=" + str(round(ema20, 2)) + " EMA50=" + str(round(ema50, 2)))

            if direction == "BUY" and price > ema20 and ema20 > ema50:
                score += 1
                reasons.append(
                    "✅ EMA OK: price " + str(round(price, 2)) +
                    " > EMA20 " + str(round(ema20, 2)) +
                    " & EMA50 " + str(round(ema50, 2)) + " (1 pt)"
                )
            elif direction == "SELL" and price < ema20 and ema20 < ema50:
                score += 1
                reasons.append(
                    "✅ EMA OK: price " + str(round(price, 2)) +
                    " < EMA20 " + str(round(ema20, 2)) +
                    " & EMA50 " + str(round(ema50, 2)) + " (1 pt)"
                )
            else:
                reasons.append(
                    "❌ EMA conflict: EMA20=" + str(round(ema20, 2)) +
                    " EMA50=" + str(round(ema50, 2)) + " (0 pts)"
                )
        else:
            reasons.append("❌ EMA: not enough H1 data")

        # ── CHECK 3: VOLUME CONFIRMATION (0–1 pt) ─────────────
        vol_checked = False
        if len(h1_vols) >= 20:
            avg_vol = sum(h1_vols[-20:-1]) / 19
            cur_vol = h1_vols[-1]
            ratio   = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 0
            log.info("Gold H1 volume ratio=" + str(ratio))
            if ratio >= 1.2:
                score += 1
                reasons.append("✅ Volume confirmed (ratio:" + str(ratio) + "x) (1 pt)")
            else:
                reasons.append("❌ Volume low (ratio:" + str(ratio) + "x) need 1.2x (0 pts)")
            vol_checked = True

        if not vol_checked and len(m15_vols) >= 20:
            avg_vol = sum(m15_vols[-20:-1]) / 19
            cur_vol = m15_vols[-1]
            ratio   = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 0
            if ratio >= 1.2:
                score += 1
                reasons.append("✅ Volume confirmed M15 (ratio:" + str(ratio) + "x) (1 pt)")
            else:
                reasons.append("❌ Volume low M15 (ratio:" + str(ratio) + "x) (0 pts)")

        # ── CHECK 4: PDH/PDL (0–1 pt) ─────────────────────────
        d1_closes, d1_highs, d1_lows, _, _ = self._fetch_candles("XAU_USD", "D", 3)
        if len(d1_highs) >= 2:
            pdh = d1_highs[-2]
            pdl = d1_lows[-2]
            log.info("Gold PDH=" + str(round(pdh, 2)) + " PDL=" + str(round(pdl, 2)))

            if direction == "BUY" and price >= pdh * 0.9985:
                score += 1
                reasons.append("✅ Price near/above PDH=" + str(round(pdh, 2)) + " (1 pt)")
            elif direction == "SELL" and price <= pdl * 1.0015:
                score += 1
                reasons.append("✅ Price near/below PDL=" + str(round(pdl, 2)) + " (1 pt)")
            else:
                reasons.append("❌ PDH/PDL: inside prior range (0 pts)")

            reasons.append(
                "PDH=" + str(round(pdh, 2)) + " PDL=" + str(round(pdl, 2)) +
                " R1=" + str(r1) + " S1=" + str(s1)
            )

        log.info("Gold final score=" + str(score) + " direction=" + direction)
        reason_str = " | ".join(reasons)
        # Return raw score — bot.py applies threshold, not signals.py
        return score, direction, reason_str

    # ══════════════════════════════════════════════════════════
    # ASIAN SESSION ANALYSIS — same checks, lower threshold
    # ══════════════════════════════════════════════════════════
    def _analyze_gold_asian(self):
        reasons   = []
        score     = 0
        direction = "NONE"

        h1_closes, h1_highs, h1_lows, _, h1_vols = self._fetch_candles("XAU_USD", "H1", 60)

        if not h1_closes:
            return 0, "NONE", "No price data"

        price = h1_closes[-1]

        # CHECK 1: CPR
        cpr = self.cpr.get_levels("XAU_USD")
        if not cpr:
            return 0, "NONE", "CPR unavailable"

        tc = cpr["tc"]
        bc = cpr["bc"]

        if price > tc:
            direction = "BUY"
            score    += 2
            reasons.append("✅ Price " + str(round(price, 2)) + " above TC=" + str(tc) + " (2 pts)")
        elif price < bc:
            direction = "SELL"
            score    += 2
            reasons.append("✅ Price " + str(round(price, 2)) + " below BC=" + str(bc) + " (2 pts)")
        else:
            reasons.append("❌ Price inside CPR — no direction")
            return 0, "NONE", " | ".join(reasons)

        # CHECK 2: EMA
        if len(h1_closes) >= 50:
            ema20 = self._ema(h1_closes, 20)[-1]
            ema50 = self._ema(h1_closes, 50)[-1]
            between = min(ema20, ema50) < price < max(ema20, ema50)

            if between:
                reasons.append("❌ EMA conflict: price between EMA20/EMA50 (0 pts)")
            elif direction == "BUY" and ema20 > ema50:
                score += 1
                reasons.append("✅ EMA OK: uptrend (1 pt)")
            elif direction == "SELL" and ema20 < ema50:
                score += 1
                reasons.append("✅ EMA OK: downtrend (1 pt)")
            else:
                reasons.append("❌ EMA vs CPR mismatch (0 pts)")

        # CHECK 3: Volume
        if len(h1_vols) >= 20:
            avg_vol = sum(h1_vols[-20:-1]) / 19
            cur_vol = h1_vols[-1]
            ratio   = round(cur_vol / avg_vol, 2) if avg_vol > 0 else 0
            if ratio >= 1.1:
                score += 1
                reasons.append("✅ Volume confirmed (ratio:" + str(ratio) + "x) (1 pt)")
            else:
                reasons.append("❌ Volume low (ratio:" + str(ratio) + "x) (0 pts)")

        # CHECK 4: PDH/PDL
        d1_closes, d1_highs, d1_lows, _, _ = self._fetch_candles("XAU_USD", "D", 3)
        if len(d1_highs) >= 2:
            pdh = d1_highs[-2]
            pdl = d1_lows[-2]
            if direction == "BUY" and price >= pdh * 0.999:
                score += 1
                reasons.append("✅ Near/above PDH=" + str(round(pdh, 2)) + " (1 pt)")
            elif direction == "SELL" and price <= pdl * 1.001:
                score += 1
                reasons.append("✅ Near/below PDL=" + str(round(pdl, 2)) + " (1 pt)")
            else:
                reasons.append("❌ Inside PDH/PDL range (0 pts)")

            reasons.append(
                "PDH=" + str(round(pdh, 2)) + " PDL=" + str(round(pdl, 2)) +
                " R1=" + str(cpr["r1"]) + " S1=" + str(cpr["s1"])
            )

        log.info("Gold Asian score=" + str(score) + " direction=" + direction)
        reason_str = " | ".join(reasons)
        # Return raw score — bot.py applies threshold, not signals.py
        return score, direction, reason_str

    # ══════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════
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
