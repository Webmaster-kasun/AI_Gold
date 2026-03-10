"""
CPR (Central Pivot Range) Calculator
=====================================
Calculates daily pivot levels from yesterday's OANDA D1 candle.
Used as Layer 1 bias filter in the Hybrid strategy.

Levels calculated:
  Pivot (P)  = (H + L + C) / 3
  BC         = (H + L) / 2
  TC         = (P - BC) + P
  R1         = (2 * P) - L
  S1         = (2 * P) - H
  R2         = P + (H - L)
  S2         = P - (H - L)

CPR Width %:
  < 0.3% = Narrow  → trending day expected (BEST for breakouts)
  0.3–0.6% = Normal
  > 0.6% = Wide   → choppy/range day (reduce size, be selective)
"""

import logging
import requests
import os

log = logging.getLogger(__name__)


class CPRCalculator:
    def __init__(self):
        self.api_key    = os.environ.get("OANDA_API_KEY", "")
        self.account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        self.base_url   = "https://api-fxpractice.oanda.com"
        self.headers    = {"Authorization": "Bearer " + self.api_key}
        self._cache     = {}  # instrument -> CPR levels (cached per day)

    def _fetch_yesterday_candle(self, instrument):
        """Fetch the last completed D1 candle from OANDA"""
        url    = self.base_url + "/v3/instruments/" + instrument + "/candles"
        params = {"count": "3", "granularity": "D", "price": "M"}
        for attempt in range(3):
            try:
                r = requests.get(url, headers=self.headers, params=params, timeout=10)
                if r.status_code == 200:
                    candles = [c for c in r.json()["candles"] if c["complete"]]
                    if candles:
                        c = candles[-1]  # Most recent completed daily candle
                        return {
                            "high":  float(c["mid"]["h"]),
                            "low":   float(c["mid"]["l"]),
                            "close": float(c["mid"]["c"]),
                            "time":  c["time"][:10]
                        }
                log.warning("D1 candle fetch attempt " + str(attempt + 1) + " failed: " + str(r.status_code))
            except Exception as e:
                log.warning("D1 candle error attempt " + str(attempt + 1) + ": " + str(e))
        return None

    def get_levels(self, instrument):
        """
        Calculate and return CPR levels for instrument.
        Returns dict with all levels + width classification.
        Cached after first call — call once per session.
        """
        if instrument in self._cache:
            log.info("CPR " + instrument + " (cached): " + str(self._cache[instrument]))
            return self._cache[instrument]

        candle = self._fetch_yesterday_candle(instrument)
        if not candle:
            log.warning("CPR: Could not fetch D1 candle for " + instrument)
            return None

        H = candle["high"]
        L = candle["low"]
        C = candle["close"]

        pivot = (H + L + C) / 3
        bc    = (H + L) / 2
        tc    = (pivot - bc) + pivot
        r1    = (2 * pivot) - L
        s1    = (2 * pivot) - H
        r2    = pivot + (H - L)
        s2    = pivot - (H - L)

        width_pct = abs(tc - bc) / pivot * 100

        if width_pct < 0.3:
            width_label = "NARROW 🔥 (trending day!)"
        elif width_pct < 0.6:
            width_label = "NORMAL"
        else:
            width_label = "WIDE ⚠️ (choppy day)"

        levels = {
            "pivot":       round(pivot, 2),
            "tc":          round(tc, 2),
            "bc":          round(bc, 2),
            "r1":          round(r1, 2),
            "s1":          round(s1, 2),
            "r2":          round(r2, 2),
            "s2":          round(s2, 2),
            "width_pct":   round(width_pct, 3),
            "width_label": width_label,
            "is_narrow":   width_pct < 0.3,
            "is_wide":     width_pct > 0.6,
            "date":        candle["time"]
        }

        self._cache[instrument] = levels

        log.info(
            "CPR " + instrument + " | "
            "Pivot=" + str(levels["pivot"]) + " "
            "TC=" + str(levels["tc"]) + " "
            "BC=" + str(levels["bc"]) + " "
            "R1=" + str(levels["r1"]) + " "
            "S1=" + str(levels["s1"]) + " "
            "Width=" + str(levels["width_pct"]) + "% " + width_label
        )

        return levels

    def get_bias(self, instrument, current_price):
        """
        Check CPR bias based on current price vs TC/BC.
        Returns: ('BULL', score, reason) | ('BEAR', score, reason) | ('NEUTRAL', 0, reason)
        Score: 2 if strong (price broke above TC or below BC), 1 if inside CPR
        """
        levels = self.get_levels(instrument)
        if not levels:
            return "NEUTRAL", 0, "CPR levels unavailable"

        tc = levels["tc"]
        bc = levels["bc"]

        if current_price > tc:
            return "BULL", 2, (
                "Price " + str(round(current_price, 2)) +
                " > TC " + str(tc) + " → CPR bullish bias"
            )
        elif current_price < bc:
            return "BEAR", 2, (
                "Price " + str(round(current_price, 2)) +
                " < BC " + str(bc) + " → CPR bearish bias"
            )
        else:
            return "NEUTRAL", 0, (
                "Price " + str(round(current_price, 2)) +
                " inside CPR (" + str(bc) + "–" + str(tc) + ") → No bias"
            )

    def get_cpr_tp(self, instrument, direction, current_price):
        """
        Return CPR-based TP target (R1 for BUY, S1 for SELL).
        Converts to pips for Gold and Forex.
        """
        levels = self.get_levels(instrument)
        if not levels:
            return None, None

        pip = 0.01 if instrument == "XAU_USD" else 0.0001

        if direction == "BUY":
            target = levels["r1"]
            if target > current_price:
                pips = round((target - current_price) / pip)
                return pips, target
        else:
            target = levels["s1"]
            if target < current_price:
                pips = round((current_price - target) / pip)
                return pips, target

        return None, None

    def summary_text(self, instrument):
        """Returns a formatted CPR summary for Telegram alerts"""
        levels = self.get_levels(instrument)
        if not levels:
            return "CPR: unavailable"

        return (
            "CPR " + instrument + " | " + levels["width_label"] + "\n"
            "TC=" + str(levels["tc"]) +
            " BC=" + str(levels["bc"]) +
            " Pivot=" + str(levels["pivot"]) + "\n"
            "R1=" + str(levels["r1"]) +
            " S1=" + str(levels["s1"]) +
            " Width=" + str(levels["width_pct"]) + "%"
        )
