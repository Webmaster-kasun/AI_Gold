"""
Hybrid Signal Engine — CPR + Breakout Momentum
===============================================
Pairs:    GBP/USD + XAU/USD
Strategy: 5-Layer scoring system (need 4/6 points to trade)

Layer 1 — CPR Bias       (0–2 pts): Price above TC (bull) or below BC (bear)
Layer 2 — H4 Structure   (0–2 pts): Breakout above H4 resistance / below support
Layer 3 — Macro Filter   (0–1 pt):  DXY direction + VIX + bond yields (Gold)
Layer 4 — EMA Trend      (0–1 pt):  EMA20 > EMA50 on H1
Layer 5 — M15 Momentum   (0–2 pts): RSI momentum + MACD histogram expanding

Total possible: 8 pts | Minimum to trade: 4 pts | Direction must match across layers
"""

import os
import requests
import logging
from cpr import CPRCalculator

log = logging.getLogger(__name__)


class SignalEngine:
    def __init__(self):
        self.api_key    = os.environ.get("OANDA_API_KEY", "")
        self.account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        self.base_url   = "https://api-fxpractice.oanda.com"
        self.headers    = {"Authorization": "Bearer " + self.api_key}
        self.cpr        = CPRCalculator()

    OANDA_MAP = {
        "GBPUSD": "GBP_USD",
        "XAUUSD": "XAU_USD",
    }

    def _fetch_candles(self, instrument, granularity, count=100):
        """Fetch OANDA candles with 3 retries"""
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
                    return closes, highs, lows, opens
                log.warning("Candle fetch attempt " + str(attempt + 1) + " failed: " + str(r.status_code))
            except Exception as e:
                log.warning("Candle fetch error attempt " + str(attempt + 1) + ": " + str(e))
        return [], [], [], []

    def _fetch_yahoo(self, ticker, interval="1d", range_="5d"):
        url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker + "?interval=" + interval + "&range=" + range_
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    closes = [c for c in r.json()["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c]
                    return closes
            except Exception as e:
                log.warning("Yahoo attempt " + str(attempt + 1) + " error: " + str(e))
        return []

    def analyze(self, asset="GBPUSD"):
        self.asset = asset
        log.info("Hybrid analyzing " + asset + "...")
        if asset == "XAUUSD":
            return self._analyze_gold()
        return self._analyze_gbpusd()

    # ══════════════════════════════════════════════════════════
    # GOLD HYBRID ANALYSIS
    # ══════════════════════════════════════════════════════════
    def _analyze_gold(self):
        reasons = []
        bull    = 0
        bear    = 0

        # ── LAYER 1: CPR BIAS (0–2 pts) ──────────────────────
        h1_closes, h1_highs, h1_lows, _ = self._fetch_candles("XAU_USD", "H1", 5)
        current_price = h1_closes[-1] if h1_closes else None

        if current_price:
            cpr_dir, cpr_pts, cpr_reason = self.cpr.get_bias("XAU_USD", current_price)
            if cpr_dir == "BULL":
                bull += cpr_pts
                reasons.append("L1 ✅ CPR: " + cpr_reason)
            elif cpr_dir == "BEAR":
                bear += cpr_pts
                reasons.append("L1 ✅ CPR: " + cpr_reason)
            else:
                reasons.append("L1 ❌ CPR: " + cpr_reason + " (0 pts)")
        else:
            reasons.append("L1 ❌ CPR: No price data")

        # ── LAYER 3: MACRO FILTER (0–1 pt) ───────────────────
        # DXY — inverse relationship with Gold
        dxy = self._fetch_yahoo("DX-Y.NYB", "1h", "2d")
        if len(dxy) >= 3:
            chg = ((dxy[-1] - dxy[-3]) / dxy[-3]) * 100
            log.info("Gold DXY 2h chg=" + str(round(chg, 3)) + "%")
            if chg < -0.3:
                bull += 1
                reasons.append("L3 ✅ Macro: DXY falling " + str(round(chg, 2)) + "% → Gold BUY")
            elif chg > 0.3:
                bear += 1
                reasons.append("L3 ✅ Macro: DXY rising " + str(round(chg, 2)) + "% → Gold SELL")
            else:
                reasons.append("L3 ❌ Macro: DXY neutral (" + str(round(chg, 2)) + "%) (0 pts)")

        # VIX — high fear = Gold up
        vix = self._fetch_yahoo("%5EVIX", "1d", "5d")
        if vix:
            v = vix[-1]
            log.info("Gold VIX=" + str(round(v, 1)))
            if v > 25:
                bull += 1
                reasons.append("L3 ✅ Macro: VIX=" + str(round(v, 0)) + " high fear → Gold BUY bonus")
            elif v > 18:
                reasons.append("L3 ℹ️ Macro: VIX=" + str(round(v, 0)) + " elevated (no bonus)")
            elif v < 13:
                bear += 1
                reasons.append("L3 ✅ Macro: VIX=" + str(round(v, 0)) + " low fear → Gold SELL")

        # Bond yields — inverse with Gold
        yields = self._fetch_yahoo("%5ETNX", "1d", "5d")
        if len(yields) >= 2:
            chg = yields[-1] - yields[-2]
            log.info("Gold yields chg=" + str(round(chg, 3)))
            if chg < -0.06:
                bull += 1
                reasons.append("L3 ✅ Macro: Yields falling sharply → Gold BUY bonus")
            elif chg > 0.06:
                bear += 1
                reasons.append("L3 ✅ Macro: Yields rising sharply → Gold SELL bonus")

        # ── LAYER 2: H4 STRUCTURE (0–2 pts) ──────────────────
        h4_closes, h4_highs, h4_lows, _ = self._fetch_candles("XAU_USD", "H4", 60)
        if len(h4_closes) < 20:
            return 0, "NONE", "Not enough H4 Gold data"

        resistance = max(h4_highs[-23:-3])
        support    = min(h4_lows[-23:-3])
        current_h4 = h4_closes[-1]
        prev_h4    = h4_closes[-2]
        atr_h4     = self._atr(h4_highs, h4_lows, h4_closes, 14)

        log.info(
            "Gold H4 resistance=" + str(round(resistance, 2)) +
            " support=" + str(round(support, 2)) +
            " current=" + str(round(current_h4, 2)) +
            " ATR=" + str(round(atr_h4, 2))
        )

        broke_resistance = current_h4 > resistance and prev_h4 <= resistance
        broke_support    = current_h4 < support    and prev_h4 >= support
        near_resistance  = abs(current_h4 - resistance) < atr_h4 * 0.3 and current_h4 > resistance
        near_support     = abs(current_h4 - support)    < atr_h4 * 0.3 and current_h4 < support

        # Extra bonus if CPR TC AND H4 resistance both broken (highest conviction setup)
        cpr_levels = self.cpr.get_levels("XAU_USD")
        cpr_tc = cpr_levels["tc"] if cpr_levels else None

        if broke_resistance:
            bonus = 1 if (cpr_tc and current_h4 > cpr_tc) else 0
            bull += 2 + bonus
            reasons.append(
                "L2 ✅ H4: BREAKOUT above $" + str(round(resistance, 0)) +
                ("! + CPR TC ALIGNED 🔥🔥" if bonus else "!")
            )
        elif near_resistance:
            bull += 1
            reasons.append("L2 ✅ H4: Near resistance $" + str(round(resistance, 0)) + " (1 pt)")

        if broke_support:
            bonus = 1 if (cpr_tc and current_h4 < (cpr_levels["bc"] if cpr_levels else 0)) else 0
            bear += 2 + bonus
            reasons.append(
                "L2 ✅ H4: BREAKOUT below $" + str(round(support, 0)) +
                ("! + CPR BC ALIGNED 🔥🔥" if bonus else "!")
            )
        elif near_support:
            bear += 1
            reasons.append("L2 ✅ H4: Near support $" + str(round(support, 0)) + " (1 pt)")

        if not (broke_resistance or broke_support or near_resistance or near_support):
            reasons.append("L2 ❌ H4: No breakout setup (0 pts)")

        # ── LAYER 4: EMA TREND on H1 (0–1 pt) ───────────────
        h1_closes, h1_highs, h1_lows, _ = self._fetch_candles("XAU_USD", "H1", 60)
        if len(h1_closes) >= 50:
            ema20 = self._ema(h1_closes, 20)
            ema50 = self._ema(h1_closes, 50)
            atr_h1   = self._atr(h1_highs, h1_lows, h1_closes, 14)
            atr_prev = self._atr(h1_highs[:-5], h1_lows[:-5], h1_closes[:-5], 14)
            atr_expand = atr_h1 > atr_prev * 1.15

            log.info(
                "Gold H1 EMA20=" + str(round(ema20[-1], 2)) +
                " EMA50=" + str(round(ema50[-1], 2)) +
                " ATR_expand=" + str(atr_expand)
            )

            if ema20[-1] > ema50[-1] and bull > bear:
                bull += 1
                reasons.append("L4 ✅ EMA: H1 uptrend EMA20 > EMA50 (1 pt)")
            elif ema20[-1] < ema50[-1] and bear > bull:
                bear += 1
                reasons.append("L4 ✅ EMA: H1 downtrend EMA20 < EMA50 (1 pt)")
            else:
                reasons.append("L4 ❌ EMA: EMA trend conflicts with bias (0 pts)")

            # ATR expansion sub-check (informational)
            if atr_expand:
                reasons.append("L4 ℹ️ H1 ATR expanding — breakout has real momentum")
        else:
            reasons.append("L4 ❌ EMA: Not enough H1 data")

        # ── LAYER 5: M15 MOMENTUM (0–2 pts) ──────────────────
        m15_closes, m15_highs, m15_lows, _ = self._fetch_candles("XAU_USD", "M15", 100)
        if len(m15_closes) >= 30:
            rsi_m15              = self._rsi(m15_closes, 14)
            macd_val, macd_sig, macd_hist, prev_hist = self._macd(m15_closes)
            hist_expanding       = abs(macd_hist) > abs(prev_hist)

            log.info(
                "Gold M15 RSI=" + str(round(rsi_m15, 1)) +
                " MACD_hist=" + str(round(macd_hist, 2)) +
                " expanding=" + str(hist_expanding)
            )

            # RSI (1 pt)
            if rsi_m15 > 55 and bull > bear:
                bull += 1
                reasons.append("L5 ✅ RSI=" + str(round(rsi_m15, 0)) + " bullish momentum (1 pt)")
            elif rsi_m15 < 45 and bear > bull:
                bear += 1
                reasons.append("L5 ✅ RSI=" + str(round(rsi_m15, 0)) + " bearish momentum (1 pt)")
            else:
                reasons.append("L5 ❌ RSI=" + str(round(rsi_m15, 0)) + " weak momentum (0 pts)")

            # MACD (1 pt)
            if macd_hist > 0 and hist_expanding and bull > bear:
                bull += 1
                reasons.append("L5 ✅ MACD histogram positive & expanding (1 pt)")
            elif macd_hist < 0 and hist_expanding and bear > bull:
                bear += 1
                reasons.append("L5 ✅ MACD histogram negative & expanding (1 pt)")
            else:
                reasons.append("L5 ❌ MACD not confirming (0 pts)")
        else:
            reasons.append("L5 ❌ M15: Not enough data")

        log.info("Gold Hybrid bull=" + str(bull) + " bear=" + str(bear))
        reason_str = " | ".join(reasons) if reasons else "No setup"

        # Need 4 points minimum, clear directional edge
        if bull >= 4 and bull > bear:
            return min(bull, 8), "BUY", reason_str
        elif bear >= 4 and bear > bull:
            return min(bear, 8), "SELL", reason_str
        return max(bull, bear), "NONE", reason_str

    # ══════════════════════════════════════════════════════════
    # GBP/USD HYBRID ANALYSIS
    # ══════════════════════════════════════════════════════════
    def _analyze_gbpusd(self):
        reasons = []
        bull    = 0
        bear    = 0

        # ── LAYER 1: CPR BIAS — GBP (0–1 pt, less weight than Gold) ──
        h1_closes, _, _, _ = self._fetch_candles("GBP_USD", "H1", 5)
        current_price = h1_closes[-1] if h1_closes else None

        if current_price:
            cpr_dir, cpr_pts, cpr_reason = self.cpr.get_bias("GBP_USD", current_price)
            # CPR is less reliable for event-driven GBP, cap at 1 pt
            cpr_pts = min(cpr_pts, 1)
            if cpr_dir == "BULL":
                bull += cpr_pts
                reasons.append("L1 ✅ CPR: " + cpr_reason + " (1 pt)")
            elif cpr_dir == "BEAR":
                bear += cpr_pts
                reasons.append("L1 ✅ CPR: " + cpr_reason + " (1 pt)")
            else:
                reasons.append("L1 ❌ CPR: " + cpr_reason + " (0 pts)")

        # ── LAYER 3: MACRO — DXY for GBP/USD (0–1 pt) ────────
        dxy = self._fetch_yahoo("DX-Y.NYB", "1h", "2d")
        if len(dxy) >= 3:
            chg = ((dxy[-1] - dxy[-3]) / dxy[-3]) * 100
            log.info("GBP/USD DXY 2h chg=" + str(round(chg, 3)) + "%")
            if chg < -0.2:
                bull += 1
                reasons.append("L3 ✅ Macro: DXY falling → GBP/USD BUY (1 pt)")
            elif chg > 0.2:
                bear += 1
                reasons.append("L3 ✅ Macro: DXY rising → GBP/USD SELL (1 pt)")
            else:
                reasons.append("L3 ❌ Macro: DXY neutral (0 pts)")

        # ── LAYER 2: H4 STRUCTURE (0–2 pts) ───────────────────
        h4_closes, h4_highs, h4_lows, _ = self._fetch_candles("GBP_USD", "H4", 60)
        if len(h4_closes) < 20:
            return 0, "NONE", "Not enough H4 GBP data"

        resistance = max(h4_highs[-23:-3])
        support    = min(h4_lows[-23:-3])
        current_h4 = h4_closes[-1]
        prev_h4    = h4_closes[-2]
        atr_h4     = self._atr(h4_highs, h4_lows, h4_closes, 14)

        log.info(
            "GBP/USD H4 resistance=" + str(round(resistance, 5)) +
            " support=" + str(round(support, 5)) +
            " current=" + str(round(current_h4, 5)) +
            " ATR=" + str(round(atr_h4, 5))
        )

        broke_resistance = current_h4 > resistance and prev_h4 <= resistance
        broke_support    = current_h4 < support    and prev_h4 >= support
        near_resistance  = abs(current_h4 - resistance) < atr_h4 * 0.2 and current_h4 > resistance
        near_support     = abs(current_h4 - support)    < atr_h4 * 0.2 and current_h4 < support

        if broke_resistance:
            bull += 2
            reasons.append("L2 ✅ H4: BREAKOUT above " + str(round(resistance, 5)) + " (2 pts)")
        elif near_resistance:
            bull += 1
            reasons.append("L2 ✅ H4: Near resistance break (1 pt)")

        if broke_support:
            bear += 2
            reasons.append("L2 ✅ H4: BREAKOUT below " + str(round(support, 5)) + " (2 pts)")
        elif near_support:
            bear += 1
            reasons.append("L2 ✅ H4: Near support break (1 pt)")

        if not (broke_resistance or broke_support or near_resistance or near_support):
            reasons.append("L2 ❌ H4: No breakout setup (0 pts)")

        # ── LAYER 4: EMA TREND on H1 (0–1 pt) ────────────────
        h1_closes, h1_highs, h1_lows, _ = self._fetch_candles("GBP_USD", "H1", 60)
        if len(h1_closes) >= 50:
            ema20 = self._ema(h1_closes, 20)
            ema50 = self._ema(h1_closes, 50)
            atr_h1   = self._atr(h1_highs, h1_lows, h1_closes, 14)
            atr_prev = self._atr(h1_highs[:-5], h1_lows[:-5], h1_closes[:-5], 14)

            log.info(
                "GBP H1 EMA20=" + str(round(ema20[-1], 5)) +
                " EMA50=" + str(round(ema50[-1], 5)) +
                " ATR_expand=" + str(atr_h1 > atr_prev * 1.2)
            )

            if ema20[-1] > ema50[-1] and bull > bear:
                bull += 1
                reasons.append("L4 ✅ EMA: H1 uptrend EMA20 > EMA50 (1 pt)")
            elif ema20[-1] < ema50[-1] and bear > bull:
                bear += 1
                reasons.append("L4 ✅ EMA: H1 downtrend EMA20 < EMA50 (1 pt)")
            else:
                reasons.append("L4 ❌ EMA: Trend conflicts with bias (0 pts)")
        else:
            reasons.append("L4 ❌ EMA: Not enough H1 data")

        # ── LAYER 5: M15 MOMENTUM (0–2 pts) ──────────────────
        m15_closes, _, _, _ = self._fetch_candles("GBP_USD", "M15", 100)
        if len(m15_closes) >= 30:
            rsi_m15                          = self._rsi(m15_closes, 14)
            macd_val, macd_sig, macd_hist, prev_hist = self._macd(m15_closes)
            hist_expanding                   = abs(macd_hist) > abs(prev_hist)

            log.info(
                "GBP M15 RSI=" + str(round(rsi_m15, 1)) +
                " MACD_hist=" + str(round(macd_hist, 6)) +
                " expanding=" + str(hist_expanding)
            )

            # RSI (1 pt)
            if rsi_m15 > 55 and bull > bear:
                bull += 1
                reasons.append("L5 ✅ RSI=" + str(round(rsi_m15, 0)) + " bullish (1 pt)")
            elif rsi_m15 < 45 and bear > bull:
                bear += 1
                reasons.append("L5 ✅ RSI=" + str(round(rsi_m15, 0)) + " bearish (1 pt)")
            else:
                reasons.append("L5 ❌ RSI=" + str(round(rsi_m15, 0)) + " weak (0 pts)")

            # MACD (1 pt)
            if macd_hist > 0 and hist_expanding and bull > bear:
                bull += 1
                reasons.append("L5 ✅ MACD expanding BUY (1 pt)")
            elif macd_hist < 0 and hist_expanding and bear > bull:
                bear += 1
                reasons.append("L5 ✅ MACD expanding SELL (1 pt)")
            else:
                reasons.append("L5 ❌ MACD not confirming (0 pts)")
        else:
            reasons.append("L5 ❌ M15: Not enough data")

        log.info("GBP/USD Hybrid bull=" + str(bull) + " bear=" + str(bear))
        reason_str = " | ".join(reasons) if reasons else "No setup"

        if bull >= 4 and bull > bear:
            return min(bull, 8), "BUY", reason_str
        elif bear >= 4 and bear > bull:
            return min(bear, 8), "SELL", reason_str
        return max(bull, bear), "NONE", reason_str

    # ══════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════
    def _rsi(self, closes, period=14):
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0))
            losses.append(max(-d, 0))
        if len(gains) < period:
            return 50
        ag = sum(gains[-period:]) / period
        al = sum(losses[-period:]) / period
        if al == 0:
            return 100
        return 100 - (100 / (1 + ag / al))

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

    def _macd(self, closes, fast=12, slow=26, signal=9):
        if len(closes) < slow + signal:
            return 0, 0, 0, 0
        ema_fast  = self._ema(closes, fast)
        ema_slow  = self._ema(closes, slow)
        macd_line = [a - b for a, b in zip(ema_fast[-len(ema_slow):], ema_slow)]
        sig_line  = self._ema(macd_line, signal)
        hist      = macd_line[-1] - sig_line[-1]
        prev_hist = macd_line[-2] - sig_line[-2] if len(macd_line) >= 2 else 0
        return macd_line[-1], sig_line[-1], hist, prev_hist

    def _atr(self, highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return 0.001
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            trs.append(tr)
        return sum(trs[-period:]) / period
