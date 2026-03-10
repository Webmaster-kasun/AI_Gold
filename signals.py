"""
Gold Signal Engine — CPR + Breakout Momentum
=============================================
Pair:     XAU/USD (Gold only)
Strategy: 5-Layer scoring system

London/NY (need 4/8 pts):
  Layer 1 — CPR Bias       (0-2 pts): Price above TC (bull) or below BC (bear)
  Layer 2 — H4 Structure   (0-2 pts): Breakout above H4 resistance / below support
  Layer 3 — Macro Filter   (0-1 pt):  DXY + VIX + bond yields
  Layer 4 — EMA Trend      (0-1 pt):  EMA20 vs EMA50 on H1
  Layer 5 — M15 Momentum   (0-2 pts): RSI + MACD histogram

Asian Session (need 3/8 pts):
  Layer 1 — CPR Bias       (0-2 pts): Price vs TC/BC
  Layer 2 — EMA Conflict   (0-2 pts): Blocks trade if price between EMA20/EMA50
  Layer 3 — PDH/PDL        (0-2 pts): Prior day high/low breakout
  Layer 4 — M15 RSI        (0-1 pt):  Momentum confirmation
  Layer 5 — ATR Filter     (0-1 pt):  Min $8 volatility check
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

    def analyze(self, asset="XAUUSD"):
        self.asset = asset
        log.info("Gold Signal Engine analyzing " + asset + "...")
        if asset == "XAUUSD_ASIAN":
            return self._analyze_gold_asian()
        return self._analyze_gold()

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
    # GOLD ASIAN SESSION ANALYSIS
    # Strategy mirrors CPR FX Bot Telegram style:
    #   - CPR bias (TC/BC position)          → 0–2 pts
    #   - EMA20/EMA50 H1 alignment check     → 0–2 pts (conflict = ❌ block)
    #   - PDH/PDL breakout (prior day range) → 0–2 pts
    #   - M15 RSI momentum                   → 0–1 pt
    #   - ATR volatility (Asian range check) → 0–1 pt
    # Minimum to trade: 3/8 (lower — Asian session is quieter)
    # ══════════════════════════════════════════════════════════
    def _analyze_gold_asian(self):
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
            return 0, "NONE", "No price data for Asian Gold"

        # ── LAYER 2: EMA20 / EMA50 CONFLICT CHECK (0–2 pts) ──
        # Matches Telegram bot: if price between EMA20 & EMA50 → conflict ❌
        h1_c, h1_h, h1_l, _ = self._fetch_candles("XAU_USD", "H1", 60)
        ema_conflict = True
        if len(h1_c) >= 50:
            ema20 = self._ema(h1_c, 20)
            ema50 = self._ema(h1_c, 50)
            e20   = ema20[-1]
            e50   = ema50[-1]
            price = h1_c[-1]

            # EMA conflict: price is between EMA20 and EMA50
            between_emas = min(e20, e50) < price < max(e20, e50)

            log.info(
                "Gold Asian EMA20=" + str(round(e20, 2)) +
                " EMA50=" + str(round(e50, 2)) +
                " price=" + str(round(price, 2)) +
                " conflict=" + str(between_emas)
            )

            if between_emas:
                reasons.append(
                    "L2 ❌ EMA conflict: price " + str(round(price, 2)) +
                    " EMA20=" + str(round(e20, 2)) +
                    " EMA50=" + str(round(e50, 2))
                )
                ema_conflict = True
                # EMA conflict in Asian session = no score, watch for breakout
            else:
                ema_conflict = False
                if e20 > e50 and bull > 0:
                    bull += 2
                    reasons.append(
                        "L2 ✅ EMA: Uptrend EMA20=" + str(round(e20, 2)) +
                        " > EMA50=" + str(round(e50, 2)) + " (2 pts)"
                    )
                elif e20 < e50 and bear > 0:
                    bear += 2
                    reasons.append(
                        "L2 ✅ EMA: Downtrend EMA20=" + str(round(e20, 2)) +
                        " < EMA50=" + str(round(e50, 2)) + " (2 pts)"
                    )
                else:
                    reasons.append("L2 ❌ EMA: Trend vs CPR mismatch (0 pts)")
        else:
            reasons.append("L2 ❌ EMA: Not enough H1 data")

        # ── LAYER 3: PDH / PDL BREAKOUT (0–2 pts) ────────────
        # Prior Day High/Low: breakout = strong Asian momentum signal
        d1_closes, d1_highs, d1_lows, _ = self._fetch_candles("XAU_USD", "D", 3)
        if len(d1_highs) >= 2 and current_price:
            pdh = d1_highs[-2]   # Prior day high
            pdl = d1_lows[-2]    # Prior day low
            log.info(
                "Gold Asian PDH=" + str(round(pdh, 2)) +
                " PDL=" + str(round(pdl, 2)) +
                " price=" + str(round(current_price, 2))
            )
            if current_price > pdh:
                bull += 2
                reasons.append(
                    "L3 ✅ PDH breakout: price " + str(round(current_price, 2)) +
                    " > PDH " + str(round(pdh, 2)) + " (2 pts)"
                )
            elif current_price < pdl:
                bear += 2
                reasons.append(
                    "L3 ✅ PDL breakout: price " + str(round(current_price, 2)) +
                    " < PDL " + str(round(pdl, 2)) + " (2 pts)"
                )
            elif current_price > pdh * 0.9985:
                bull += 1
                reasons.append(
                    "L3 ✅ Near PDH: price approaching " + str(round(pdh, 2)) + " (1 pt)"
                )
            elif current_price < pdl * 1.0015:
                bear += 1
                reasons.append(
                    "L3 ✅ Near PDL: price approaching " + str(round(pdl, 2)) + " (1 pt)"
                )
            else:
                reasons.append(
                    "L3 ❌ PDH/PDL: price inside prior day range " +
                    str(round(pdl, 2)) + "–" + str(round(pdh, 2)) + " (0 pts)"
                )
        else:
            reasons.append("L3 ❌ PDH/PDL: not enough D1 data")

        # ── LAYER 4: M15 RSI MOMENTUM (0–1 pt) ───────────────
        m15_c, _, _, _ = self._fetch_candles("XAU_USD", "M15", 50)
        if len(m15_c) >= 20:
            rsi = self._rsi(m15_c, 14)
            log.info("Gold Asian M15 RSI=" + str(round(rsi, 1)))
            if rsi > 55 and bull > bear:
                bull += 1
                reasons.append("L4 ✅ RSI=" + str(round(rsi, 0)) + " bullish momentum (1 pt)")
            elif rsi < 45 and bear > bull:
                bear += 1
                reasons.append("L4 ✅ RSI=" + str(round(rsi, 0)) + " bearish momentum (1 pt)")
            else:
                reasons.append("L4 ❌ RSI=" + str(round(rsi, 0)) + " weak/neutral (0 pts)")
        else:
            reasons.append("L4 ❌ RSI: Not enough M15 data")

        # ── LAYER 5: ASIAN ATR VOLATILITY CHECK (0–1 pt) ─────
        # Asian session Gold needs at least moderate ATR to trade
        if len(h1_h) >= 14 and len(h1_l) >= 14 and len(h1_c) >= 14:
            atr = self._atr(h1_h, h1_l, h1_c, 14)
            log.info("Gold Asian H1 ATR=" + str(round(atr, 2)))
            if atr >= 8.0:   # Min $8 ATR for Asian Gold (vs $10+ London)
                if bull > bear:
                    bull += 1
                elif bear > bull:
                    bear += 1
                reasons.append("L5 ✅ ATR=" + str(round(atr, 2)) + " sufficient volatility (1 pt)")
            else:
                reasons.append("L5 ❌ ATR=" + str(round(atr, 2)) + " too low — thin Asian market (0 pts)")
        else:
            reasons.append("L5 ❌ ATR: Not enough data")

        log.info("Gold Asian bull=" + str(bull) + " bear=" + str(bear) + " ema_conflict=" + str(ema_conflict))
        reason_str = " | ".join(reasons) if reasons else "No setup"

        # EMA conflict in Asian session blocks all entries (watch mode only)
        if ema_conflict:
            return max(bull, bear), "NONE", reason_str

        if bull >= 3 and bull > bear:
            return min(bull, 8), "BUY", reason_str
        elif bear >= 3 and bear > bull:
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
