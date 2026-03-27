"""
AI Reasoning Layer — Claude-powered trade filter
=================================================
Sits between signal scoring and order placement.
Called only when score >= threshold (signal already passed 7-check system).

What it does:
  - Reads recent candle direction, momentum, losses today, price zone history
  - Reasons like a senior trader: "does this trade make sense RIGHT NOW?"
  - Returns: decision (YES/NO/REDUCE), confidence (LOW/MEDIUM/HIGH), reason, lot_multiplier
  - On HIGH confidence: increases lot size (up to 3x)
  - On LOW confidence: blocks the trade entirely
  - Knows you are a day trader expecting 5-8 trades per day — will not over-block

Lot sizing tiers:
  HIGH   confidence + score 7/7 = 3x units
  HIGH   confidence + score 6/7 = 2x units
  MEDIUM confidence             = 1x units (normal)
  LOW    confidence             = BLOCK trade
"""

import os
import json
import logging
import requests
import time

log = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"


def _call_claude(prompt: str) -> str:
    """Call Claude API and return the text response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — AI reasoning skipped, trade allowed")
        return '{"decision":"YES","confidence":"MEDIUM","reason":"API key not configured","lot_multiplier":1}'

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }

    body = {
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    for attempt in range(3):
        try:
            time.sleep(0.5)
            r = requests.post(ANTHROPIC_API_URL, headers=headers, json=body, timeout=20)
            if r.status_code == 200:
                data    = r.json()
                content = data.get("content", [])
                if content and content[0].get("type") == "text":
                    return content[0]["text"].strip()
            log.warning("Claude API attempt " + str(attempt+1) + " failed: " + str(r.status_code))
        except Exception as e:
            log.warning("Claude API error attempt " + str(attempt+1) + ": " + str(e))
        time.sleep(2)

    log.warning("Claude API failed after 3 attempts — trade allowed with normal size")
    return '{"decision":"YES","confidence":"MEDIUM","reason":"API unavailable after retries","lot_multiplier":1}'


def _build_prompt(
    direction:           str,
    score:               int,
    price:               float,
    signal_details:      str,
    wins_today:          int,
    losses_today:        int,
    last_loss_entry:     float,
    last_loss_exit:      float,
    last_loss_dir:       str,
    last_win_exit:       float,
    recent_candles:      list,
    session:             str,
    h4_trend:            str,
    is_asian:            bool,
) -> str:
    """Build the reasoning prompt for Claude."""

    candle_summary = ""
    if recent_candles:
        directions = []
        for i in range(1, min(len(recent_candles), 6)):
            move = recent_candles[i] - recent_candles[i-1]
            directions.append("UP" if move > 0 else "DOWN")
        candle_summary = " -> ".join(directions)

    # --- FIX 3: Use actual SL exit price for loss zone, not entry ---
    last_loss_info = "None today"
    if last_loss_exit and last_loss_dir:
        dist_from_exit  = abs(price - last_loss_exit) / 0.01
        dist_from_entry = abs(price - last_loss_entry) / 0.01 if last_loss_entry else 0
        last_loss_info = (
            "Last loss was " + last_loss_dir +
            " | entry=$" + str(last_loss_entry) +
            " | SL hit at=$" + str(last_loss_exit) +
            " | current price is " + str(round(dist_from_exit)) + "p from SL zone" +
            " and " + str(round(dist_from_entry)) + "p from loss entry"
        )

    # --- FIX 1: Chase detection — warn if new entry is far above last win exit ---
    chase_warning = ""
    if last_win_exit and last_win_exit > 0:
        chase_pips = (price - last_win_exit) / 0.01
        if direction == "BUY" and chase_pips > 200:
            chase_warning = (
                "\n⚠️ CHASE RISK: Entry price $" + str(price) +
                " is " + str(round(chase_pips)) + "p above last WIN exit $" + str(last_win_exit) +
                ". Price may be extended — do NOT approve just because the signal scored."
            )
        elif direction == "SELL" and chase_pips < -200:
            chase_warning = (
                "\n⚠️ CHASE RISK: Entry price $" + str(price) +
                " is " + str(round(abs(chase_pips))) + "p below last WIN exit $" + str(last_win_exit) +
                ". Price may be extended — do NOT approve just because the signal scored."
            )

    prompt = """You are a senior gold (XAU/USD) risk manager with a 65% win rate target.
You must respond ONLY with a single valid JSON object, no explanation, no markdown.

STRATEGY CONTEXT:
- CPR breakout with H4 trend filter, H1 EMA, RSI, ATR stops
- Only trade when H4 trend, H1 EMA, and CPR ALL agree on direction
- Best sessions: London Open (14-17 SGT) and NY Overlap (20-22 SGT)
- Asian session: lower conviction, require cleaner setups
- Risk per trade: ~$10-15 USD | Target: 65% win rate

CURRENT SIGNAL:
- Direction: """ + direction + """
- Score: """ + str(score) + """/7
- Entry price: $""" + str(price) + """
- Session: """ + session + """
- H4 trend: """ + h4_trend + """
- Is Asian session: """ + str(is_asian) + """
- Signal details: """ + signal_details[:400] + """

TODAY SO FAR:
- Wins: """ + str(wins_today) + """ | Losses: """ + str(losses_today) + """
- Last loss info: """ + last_loss_info + """
- Recent H1 candles (oldest→newest): """ + (candle_summary if candle_summary else "unavailable") + """
""" + chase_warning + """

DECISION RULES — apply in order, first match wins:
1. Chase block: entry is >300p away from last exit in same direction → NO
2. Zone trap: same direction as last loss AND price within 150p of SL exit → NO, always
3. Loss filter: losses today >= 2 → require score 7/7 + H4 aligned → otherwise NO
4. Asian filter: is_asian=True AND score < 6 → NO (Asian needs cleaner setups)
5. H4 conflict: H4 trend opposes direction → NO (H4 is the macro filter, non-negotiable)
6. H1 momentum: recent candles show 3+ consecutive moves AGAINST signal direction → LOW confidence
7. Session quality: London Open or NY Overlap → allow MEDIUM+ | Other hours → require HIGH
8. Strong setup: H4 aligned + score 6-7 + H1 candles agree + good session → HIGH confidence

CONFIDENCE → LOT SIZE:
- HIGH + score 7 → lot_multiplier 3
- HIGH + score 6 → lot_multiplier 2
- MEDIUM → lot_multiplier 1
- LOW → decision must be NO

Respond with ONLY this JSON (no other text):
{
  "decision": "YES" or "NO",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "reason": "one sentence max 20 words",
  "lot_multiplier": 1 or 2 or 3
}"""

    return prompt


def ai_should_trade(
    direction:       str,
    score:           int,
    price:           float,
    signal_details:  str,
    wins_today:      int,
    losses_today:    int,
    last_loss_entry: float,
    last_loss_exit:  float,
    last_loss_dir:   str,
    last_win_exit:   float,
    recent_candles:  list,
    session:         str,
    h4_trend:        str,
    is_asian:        bool = False,
) -> dict:
    """
    Main entry point. Returns dict:
    {
        "allow":          True/False,
        "confidence":     "HIGH"/"MEDIUM"/"LOW",
        "reason":         "explanation string",
        "lot_multiplier": 1/2/3
    }
    """
    try:
        prompt = _build_prompt(
            direction       = direction,
            score           = score,
            price           = price,
            signal_details  = signal_details,
            wins_today      = wins_today,
            losses_today    = losses_today,
            last_loss_entry = last_loss_entry,
            last_loss_exit  = last_loss_exit,
            last_loss_dir   = last_loss_dir,
            last_win_exit   = last_win_exit,
            recent_candles  = recent_candles,
            session         = session,
            h4_trend        = h4_trend,
            is_asian        = is_asian,
        )

        raw = _call_claude(prompt)
        log.info("AI raw response: " + raw[:200])

        # Strip markdown fences if model added them
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        result = json.loads(clean)

        decision       = result.get("decision", "YES").upper()
        confidence     = result.get("confidence", "MEDIUM").upper()
        reason         = result.get("reason", "No reason provided")
        lot_multiplier = int(result.get("lot_multiplier", 1))

        # Safety clamp
        lot_multiplier = max(1, min(3, lot_multiplier))

        # LOW confidence always blocks
        if confidence == "LOW":
            decision = "NO"

        allow = (decision == "YES")

        log.info(
            "AI DECISION: " + decision +
            " | confidence=" + confidence +
            " | lot_multiplier=" + str(lot_multiplier) +
            " | reason=" + reason
        )

        return {
            "allow":          allow,
            "confidence":     confidence,
            "reason":         reason,
            "lot_multiplier": lot_multiplier if allow else 1,
        }

    except Exception as e:
        log.warning("AI reasoning error: " + str(e) + " — trade allowed with normal size")
        return {
            "allow":          True,
            "confidence":     "MEDIUM",
            "reason":         "AI error — defaulting to allow",
            "lot_multiplier": 1,
        }
