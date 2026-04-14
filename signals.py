"""Signal engine for CPR breakout detection on XAU/USD — v5.1

Scoring (Bull):
  Main condition  — price above CPR/PDH/R1: +2 | above R2 (extended): +1
  SMA alignment   — both SMA20 & SMA50 below price: +2 | one below: +1
  CPR width       — < 0.5% (narrow): +2 | 0.5%–1.0% (moderate): +1

Scoring (Bear):
  Main condition  — price below CPR/PDL/S1: +2 | below S2 (extended): +1
  SMA alignment   — both SMA20 & SMA50 above price: +2 | one above: +1
  CPR width       — same as Bull
  Trend exhaustion — price > exhaustion_atr_mult × ATR from SMA20: −1
                     (S2/R2 Extended setups are HARD BLOCKED when exhausted — v4.0)

Position size by score:
  score 5–6  →  $100 (full)
  score 4    →  $66  (partial — minimum entry)
  score < 4  →  no trade (below threshold)

SL calculation (v4.0 — ATR-based):
  SL = ATR(14) × atr_sl_multiplier, clamped to [sl_min_usd, sl_max_usd]
  Replaces the old fixed 0.25% percentage SL which was too tight for gold volatility.

TP calculation (v4.0):
  TP = SL × rr_ratio (default 2.0)
  Always derived from SL, not calculated independently.

Non-negotiable rule: R:R must be ≥ 1:2 (TP ≥ 2× SL). Trade is skipped if not met.
"""

import time
import logging
import requests
from config_loader import load_secrets, load_settings, DATA_DIR
from oanda_trader import make_oanda_session

log = logging.getLogger(__name__)


# Minimum score required to trade (scores below this are discarded)
MIN_TRADE_SCORE = 4  # v4.2: this is a fallback only — settings["signal_threshold"] is the live value


def score_to_position_usd(score: int, settings: dict | None = None) -> int:
    """Return the risk-dollar position size for a given score.

    Reads position_full_usd and position_partial_usd from settings when
    provided; falls back to the hardcoded defaults ($100 / $66) otherwise.
    Returns 0 (no trade) for any score below MIN_TRADE_SCORE (4). v4.2
    """
    full = int((settings or {}).get("position_full_usd", 100))
    partial = int((settings or {}).get("position_partial_usd", 66))
    size_tiers = [
        (4, full),    # score >= 5 → full
        (2, partial), # score >= 4 → partial (threshold gates this)
    ]
    for threshold, size in size_tiers:
        if score > threshold:
            return size
    return 0


class SignalEngine:
    def __init__(self, demo: bool = True):
        secrets = load_secrets()
        self.api_key = secrets.get("OANDA_API_KEY", "")
        self.account_id = secrets.get("OANDA_ACCOUNT_ID", "")
        self.base_url = (
            "https://api-fxpractice.oanda.com" if demo else "https://api-fxtrade.oanda.com"
        )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.session = make_oanda_session(allowed_methods=["GET"])

    def analyze(self, asset: str = "XAUUSD", settings: dict | None = None):
        """Run the CPR scoring engine.

        Parameters
        ----------
        asset : str
            Instrument identifier (only XAUUSD supported).
        settings : dict | None
            Bot settings dict; when provided, position sizes are read from
            ``position_full_usd`` and ``position_partial_usd`` keys.

        Returns
        -------
        (score, direction, details, levels, position_usd)
        """
        if settings is None:
            settings = load_settings()
        _instrument_key = (settings or {}).get("instrument_display", "XAU/USD").replace("/", "")
        if asset not in ("XAUUSD", _instrument_key):
            return 0, "NONE", f"Only {_instrument_key} supported in this version", {}, 0

        instrument = (settings or {}).get("instrument", "XAU_USD")

        # ── Daily candles → CPR levels (v4.2: no caching — always fetch fresh) ─
        daily_closes, daily_highs, daily_lows = self._fetch_candles(instrument, "D", 3)
        if len(daily_closes) < 2:
            return 0, "NONE", "Not enough daily data for CPR", {}, 0

        prev_high  = daily_highs[-2]
        prev_low   = daily_lows[-2]
        prev_close = daily_closes[-2]

        pivot      = (prev_high + prev_low + prev_close) / 3
        bc         = (prev_high + prev_low) / 2
        tc         = (pivot - bc) + pivot
        # Ensure TC is always the upper CPR bound (swap if pivot < midpoint of range)
        if tc < bc:
            tc, bc = bc, tc
        daily_range = prev_high - prev_low
        r1          = (2 * pivot) - prev_low
        r2          = pivot + daily_range
        s1          = (2 * pivot) - prev_high
        s2          = pivot - daily_range
        pdh         = prev_high
        pdl         = prev_low
        cpr_width_pct = abs(tc - bc) / pivot * 100

        levels = {
            "pivot":         round(pivot, 2),
            "tc":            round(tc, 2),
            "bc":            round(bc, 2),
            "r1":            round(r1, 2),
            "r2":            round(r2, 2),
            "s1":            round(s1, 2),
            "s2":            round(s2, 2),
            "pdh":           round(pdh, 2),
            "pdl":           round(pdl, 2),
            "cpr_width_pct": round(cpr_width_pct, 3),
        }
        log.info(
            "CPR levels fetched | pivot=%.2f TC=%.2f BC=%.2f R1=%.2f S1=%.2f "
            "R2=%.2f S2=%.2f PDH=%.2f PDL=%.2f width=%.3f%%",
            pivot, tc, bc, r1, s1, r2, s2, pdh, pdl, cpr_width_pct,
        )

        # ── M15 candles → price, SMA, ATR ─────────────────────────────────
        tf = (settings or {}).get("timeframe", "M15")
        m15_closes, m15_highs, m15_lows = self._fetch_candles(instrument, tf, 65)
        if len(m15_closes) < 52:
            return 0, "NONE", "Not enough M15 data (need 52 candles for SMA50)", levels, 0

        # v5.0 — candle-close confirmation:
        # When require_candle_close=True, use the last COMPLETED candle [-2] for
        # signal entry. This prevents fakeouts where price crosses a level
        # intracandle then reverses before the M15 bar closes.
        # When False, use current tick price [-1] (old behaviour).
        _require_close = bool((settings or {}).get("require_candle_close", True))
        current_close  = m15_closes[-2] if _require_close else m15_closes[-1]

        # v5.1 — debug log: shows exactly what candle the bot acted on every cycle.
        # Match close= against your OANDA chart (completed M15 candle) to verify timing.
        log.info(
            "Signal candle | close=%.2f (candle [-2]) | current_tick=%.2f (candle [-1]) | ATR=%.2f",
            m15_closes[-2], m15_closes[-1], self._atr(m15_highs, m15_lows, m15_closes, 14) or 0,
        )

        # SMA20 and SMA50 use the last 20/50 completed candles (exclude current)
        sma20 = sum(m15_closes[-21:-1]) / 20
        sma50 = sum(m15_closes[-51:-1]) / 50

        # ── H1 trend filter (v5.0) ─────────────────────────────────────────
        # Fetch H1 candles to determine macro bias.
        # Only used when h1_trend_filter_enabled = True in settings.
        h1_trend_bullish = None   # None = filter disabled or insufficient data
        _h1_filter = bool((settings or {}).get("h1_trend_filter_enabled", True))
        if _h1_filter:
            _h1_period = int((settings or {}).get("h1_ema_period", 21))
            h1_closes, _, _ = self._fetch_candles(instrument, "H1", _h1_period + 5)
            if len(h1_closes) >= _h1_period:
                _h1_ema = sum(h1_closes[-_h1_period:]) / _h1_period
                _h1_price = h1_closes[-1]
                h1_trend_bullish = _h1_price > _h1_ema
        levels["h1_trend_bullish"] = h1_trend_bullish

        # ── H4 trend filter (v5.3) ─────────────────────────────────────────
        # Hard block on macro H4 direction — prevents shorting a bull trend
        # or buying a bear trend even when M15/CPR gives a counter signal.
        h4_trend_bullish = None   # None = filter disabled or insufficient data
        _h4_filter = bool((settings or {}).get("h4_trend_filter_enabled", True))
        if _h4_filter:
            _h4_period = int((settings or {}).get("h4_ema_period", 21))
            h4_closes, _, _ = self._fetch_candles(instrument, "H4", _h4_period + 5)
            if len(h4_closes) >= _h4_period:
                _h4_ema = sum(h4_closes[-_h4_period:]) / _h4_period
                _h4_price = h4_closes[-1]
                h4_trend_bullish = _h4_price > _h4_ema
        levels["h4_trend_bullish"] = h4_trend_bullish

        # ATR(14) — used by bot.py for SL sizing, not for scoring
        atr_val = self._atr(m15_highs, m15_lows, m15_closes, 14)
        levels["atr"]          = round(atr_val, 2) if atr_val else None
        levels["current_price"] = round(current_close, 2)
        levels["sma20"]         = round(sma20, 2)
        levels["sma50"]         = round(sma50, 2)

        # ── Scoring ────────────────────────────────────────────────────────
        score     = 0
        direction = "NONE"
        reasons   = []

        reasons.append(
            f"CPR TC={tc:.2f} BC={bc:.2f} width={cpr_width_pct:.2f}% | "
            f"R1={r1:.2f} R2={r2:.2f} S1={s1:.2f} S2={s2:.2f} | "
            f"PDH={pdh:.2f} PDL={pdl:.2f}"
        )

        # ── 1. Main condition ──────────────────────────────────────────────
        if current_close > tc:
            direction = "BUY"
            if current_close > r2:
                score += 1
                setup = "R2 Extended Breakout"
                reasons.append(
                    f"⚠️ Price {current_close:.2f} > R2={r2:.2f} — extended entry (+1, main condition)"
                )
            else:
                score += 2
                if current_close > r1:
                    setup = "R1 Breakout"
                elif current_close > pdh:
                    setup = "PDH Breakout"
                else:
                    setup = "CPR Bull Breakout"
                reasons.append(
                    f"✅ Price {current_close:.2f} above CPR/PDH/R1 zone [{setup}] (+2, main condition)"
                )
        elif current_close < bc:
            direction = "SELL"
            if current_close < s2:
                score += 1
                setup = "S2 Extended Breakdown"
                reasons.append(
                    f"⚠️ Price {current_close:.2f} < S2={s2:.2f} — extended entry (+1, main condition)"
                )
            else:
                score += 2
                if current_close < s1:
                    setup = "S1 Breakdown"
                elif current_close < pdl:
                    setup = "PDL Breakdown"
                else:
                    setup = "CPR Bear Breakdown"
                reasons.append(
                    f"✅ Price {current_close:.2f} below CPR/PDL/S1 zone [{setup}] (+2, main condition)"
                )
        else:
            reasons.append(
                f"❌ Price {current_close:.2f} inside CPR (TC={tc:.2f} BC={bc:.2f}) — no signal"
            )
            return 0, "NONE", " | ".join(reasons), levels, 0

        # ── 1b. H1 trend filter (v5.0) ────────────────────────────────────
        # Block trades that go against the H1 EMA trend.
        # BUY blocked if H1 price < H1 EMA21 (bearish trend).
        # SELL blocked if H1 price > H1 EMA21 (bullish trend).
        if h1_trend_bullish is not None:
            if direction == "BUY" and not h1_trend_bullish:
                reasons.append("❌ H1 trend bearish — BUY blocked by trend filter")
                log.info("H1 trend filter blocked BUY — H1 price below EMA%d", _h1_period)
                return 0, "NONE", " | ".join(reasons), levels, 0
            elif direction == "SELL" and h1_trend_bullish:
                reasons.append("❌ H1 trend bullish — SELL blocked by trend filter")
                log.info("H1 trend filter blocked SELL — H1 price above EMA%d", _h1_period)
                return 0, "NONE", " | ".join(reasons), levels, 0
            else:
                trend_label = "bullish" if h1_trend_bullish else "bearish"
                reasons.append(f"✅ H1 trend {trend_label} — aligns with {direction}")

        # ── 1c. H4 trend filter (v5.3) ────────────────────────────────────
        # Macro hard block — H4 EMA21 must agree with trade direction.
        # This catches multi-day bull/bear runs that H1 alone can miss
        # during intraday pullbacks (e.g. Apr 12–13 gold surge).
        if h4_trend_bullish is not None:
            if direction == "BUY" and not h4_trend_bullish:
                reasons.append("❌ H4 trend bearish — BUY blocked by H4 macro filter")
                log.info("H4 trend filter blocked BUY — H4 price below EMA%d", _h4_period)
                return 0, "NONE", " | ".join(reasons), levels, 0
            elif direction == "SELL" and h4_trend_bullish:
                reasons.append("❌ H4 trend bullish — SELL blocked by H4 macro filter")
                log.info("H4 trend filter blocked SELL — H4 price above EMA%d", _h4_period)
                return 0, "NONE", " | ".join(reasons), levels, 0
            else:
                h4_label = "bullish" if h4_trend_bullish else "bearish"
                reasons.append(f"✅ H4 trend {h4_label} — aligns with {direction}")

        # ── 2. SMA alignment ───────────────────────────────────────────────
        if direction == "BUY":
            both_below = sma20 < current_close and sma50 < current_close
            one_below  = (sma20 < current_close) != (sma50 < current_close)
            if both_below:
                score += 2
                reasons.append(
                    f"✅ Both SMAs below price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+2)"
                )
            elif one_below:
                score += 1
                which = "SMA20" if sma20 < current_close else "SMA50"
                reasons.append(
                    f"⚠️ Only {which} below price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+1)"
                )
            else:
                reasons.append(
                    f"❌ Both SMAs above price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+0)"
                )
        else:  # SELL
            both_above = sma20 > current_close and sma50 > current_close
            one_above  = (sma20 > current_close) != (sma50 > current_close)
            if both_above:
                score += 2
                reasons.append(
                    f"✅ Both SMAs above price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+2)"
                )
            elif one_above:
                score += 1
                which = "SMA20" if sma20 > current_close else "SMA50"
                reasons.append(
                    f"⚠️ Only {which} above price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+1)"
                )
            else:
                reasons.append(
                    f"❌ Both SMAs below price — SMA20={sma20:.2f} SMA50={sma50:.2f} (+0)"
                )

        # ── 3. CPR width ───────────────────────────────────────────────────
        if cpr_width_pct < 0.5:
            score += 2
            reasons.append(f"✅ Narrow CPR ({cpr_width_pct:.2f}% < 0.5%) (+2)")
        elif cpr_width_pct <= 1.0:
            score += 1
            reasons.append(f"⚠️ Moderate CPR ({cpr_width_pct:.2f}% in 0.5–1.0%) (+1)")
        else:
            reasons.append(f"❌ Wide CPR ({cpr_width_pct:.2f}% > 1.0%) (+0)")

        # ── 4. Trend exhaustion check ──────────────────────────────────────
        # Penalises entries at the end of a move. When price has drifted more
        # than `exhaustion_atr_mult` × ATR(14) away from SMA20, the trend is
        # statistically overextended and likely to reverse. Deduct 1 point so
        # marginal signals (score 3 → 2) are blocked and strong signals are
        # downsized, without completely disabling entries in genuine breakouts.
        exhaustion_atr_mult = float(settings.get("exhaustion_atr_mult", 2.0)) if settings else 2.0
        if atr_val and atr_val > 0:
            stretch = abs(current_close - sma20) / atr_val
            if stretch > exhaustion_atr_mult:
                # v4.0 — hard block for extended setups (S2/R2) when exhausted.
                # These entries are already far past the CPR; combining with an
                # overextended stretch means the move is statistically spent.
                # Deducting 1 point is insufficient — block entirely.
                if setup in ("S2 Extended Breakdown", "R2 Extended Breakout"):
                    reasons.append(
                        f"🚫 Extended entry blocked — exhaustion {stretch:.1f}× ATR "
                        f"(>{exhaustion_atr_mult:.1f}× threshold) on {setup} — no trade"
                    )
                    log.info(
                        "CPR signal BLOCKED (extended+exhaustion) | setup=%s | dir=%s | stretch=%.1f×",
                        setup, direction, stretch,
                    )
                    levels["score"] = 0
                    levels["signal_blockers"] = [f"Extended entry blocked — exhaustion {stretch:.1f}× ATR"]
                    return 0, "NONE", " | ".join(reasons), levels, 0
                score = max(0, score - 1)
                reasons.append(
                    f"⚠️ Trend stretch {stretch:.1f}× ATR from SMA20 "
                    f"(>{exhaustion_atr_mult:.1f}× threshold) — exhaustion penalty (−1)"
                )
            else:
                reasons.append(
                    f"✅ Trend stretch {stretch:.1f}× ATR from SMA20 "
                    f"(≤{exhaustion_atr_mult:.1f}× threshold) — ok (+0)"
                )
        else:
            reasons.append("⚠️ ATR unavailable — exhaustion check skipped")

        # ── Position size ──────────────────────────────────────────────────
        position_usd = score_to_position_usd(score, settings)

        # ── SL recommendation (priority order) ────────────────────────────
        # 1. Use CPR structural level if it is within 0.25% of entry
        # 2. Fall back to fixed 0.25% percentage SL
        entry = current_close
        if direction == "BUY":
            cpr_sl_candidate = bc          # below the bottom CPR for longs
            cpr_dist_pct = (entry - cpr_sl_candidate) / entry * 100
        else:
            cpr_sl_candidate = tc          # above the top CPR for shorts
            cpr_dist_pct = (cpr_sl_candidate - entry) / entry * 100

        fixed_sl_pct  = 0.25
        if cpr_dist_pct <= fixed_sl_pct:
            sl_pct_used  = round(cpr_dist_pct, 4)
            sl_source    = "below_cpr" if direction == "BUY" else "above_cpr"
        else:
            sl_pct_used  = fixed_sl_pct
            sl_source    = "fixed_pct"
        sl_usd_rec = round(entry * sl_pct_used / 100, 2)

        # ── TP recommendation (priority order) ────────────────────────────
        # 1. Use R1/S1 if it falls in 0.50%–0.75% from entry
        # 2. Use fixed 0.75% if R1/S1 is too far (> 0.75%)
        # 3. v4.2 fix: R1/S1 too close (< 0.50%) — fall back to fixed 0.75% TP
        #    instead of blocking. The fixed TP is always a valid target and
        #    preserves the intended 1:3 R:R regardless of structural levels.
        target_level = r1 if direction == "BUY" else s1
        if direction == "BUY":
            level_dist_pct = (target_level - entry) / entry * 100
        else:
            level_dist_pct = (entry - target_level) / entry * 100

        tp_skip = False
        if 0.50 <= level_dist_pct <= 0.75:
            tp_pct_used = round(level_dist_pct, 4)
            tp_source   = "r1_level" if direction == "BUY" else "s1_level"
        elif level_dist_pct > 0.75:
            tp_pct_used = 0.75
            tp_source   = "fixed_pct"
        else:
            # R1/S1 too close — fall back to fixed 0.75% TP (v4.2 fix)
            tp_pct_used = 0.75
            tp_source   = "fixed_pct_fallback"
            tp_skip     = False
        tp_usd_rec = round(entry * tp_pct_used / 100, 2)

        # ── Mandatory / quality guards — report blockers but preserve signal ──
        rr_ratio = (tp_usd_rec / sl_usd_rec) if sl_usd_rec > 0 else 0
        _min_rr_sig = float((settings or {}).get("rr_ratio", 2.0))
        rr_skip  = rr_ratio < _min_rr_sig

        blocker_reasons = []
        if rr_skip:
            blocker_reasons.append(f"R:R {rr_ratio:.2f} < 1:{_min_rr_sig:.2f}")

        levels["score"]        = score
        levels["position_usd"] = position_usd
        levels["entry"]        = round(entry, 2)
        levels["setup"]        = setup
        levels["sl_usd_rec"]   = sl_usd_rec
        levels["sl_source"]    = sl_source
        levels["sl_pct_used"]  = sl_pct_used
        levels["tp_usd_rec"]   = tp_usd_rec
        levels["tp_source"]    = tp_source
        levels["tp_pct_used"]  = tp_pct_used
        levels["rr_ratio"]     = round(rr_ratio, 2)
        levels["mandatory_checks"] = {
            "score_ok": score >= int((settings or {}).get("signal_threshold", MIN_TRADE_SCORE)),
            "rr_ok": not rr_skip,
        }
        levels["quality_checks"] = {
            "tp_ok": not tp_skip,
        }
        levels["signal_blockers"] = blocker_reasons

        _display_rr = float((settings or {}).get("rr_ratio", rr_ratio))
        reasons.append(
            f"📐 SL={sl_usd_rec} ({sl_source} {sl_pct_used:.3f}%) | "
            f"TP={tp_usd_rec} ({tp_source} {tp_pct_used:.3f}%) | R:R 1:{_display_rr:.2f}"
        )
        if blocker_reasons:
            reasons.append("🚫 " + " | ".join(blocker_reasons))

        details = " | ".join(reasons)
        if blocker_reasons:
            log.info(
                "CPR signal BLOCKED | setup=%s | dir=%s | score=%s/6 | blockers=%s",
                setup, direction, score, "; ".join(blocker_reasons),
            )
        else:
            log.info(
                "CPR signal | setup=%s | dir=%s | score=%s/6 | position=$%s",
                setup, direction, score, position_usd,
            )
        return score, direction, details, levels, position_usd

    # ── Data helpers ───────────────────────────────────────────────────────────

    def _fetch_candles(self, instrument: str, granularity: str, count: int = 60):
        url    = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {"count": str(count), "granularity": granularity, "price": "M"}
        for attempt in range(3):
            try:
                r = self.session.get(url, headers=self.headers, params=params, timeout=15)
                if r.status_code == 200:
                    candles  = r.json().get("candles", [])
                    complete = [c for c in candles if c.get("complete")]
                    closes   = [float(c["mid"]["c"]) for c in complete]
                    highs    = [float(c["mid"]["h"]) for c in complete]
                    lows     = [float(c["mid"]["l"]) for c in complete]
                    return closes, highs, lows
                log.warning("Fetch candles %s %s: HTTP %s", instrument, granularity, r.status_code)
            except Exception as e:
                log.warning(
                    "Fetch candles error (%s %s) attempt %s: %s",
                    instrument, granularity, attempt + 1, e,
                )
            time.sleep(1)
        return [], [], []

    def _atr(self, highs: list, lows: list, closes: list, period: int = 14) -> float | None:
        """Return the most recent ATR value, or None if insufficient data."""
        n = len(closes)
        if n < period + 2 or len(highs) < n or len(lows) < n:
            return None
        trs = [
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            for i in range(1, n)
        ]
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr
