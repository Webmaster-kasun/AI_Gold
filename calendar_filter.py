"""
Economic Calendar Filter - Full Auto Version
=============================================
Uses ForexFactory live feed - updates every week automatically!
No manual updates needed - works forever!

What happens on news days:
- 30 mins BEFORE news = bot pauses trading
- During news          = bot pauses trading  
- 30 mins AFTER news   = bot pauses trading
- After 30 mins        = bot resumes normally!
"""

import requests
import logging
from datetime import datetime, timedelta
import pytz

log = logging.getLogger(__name__)

class EconomicCalendar:
    def __init__(self):
        self.sg_tz   = pytz.timezone("Asia/Singapore")
        self.utc_tz  = pytz.UTC
        self._cache  = None
        self._cached_date = None

    def _fetch_events(self):
        """
        Fetch this week events from ForexFactory
        Free JSON feed - auto updates every week!
        Cached per day to avoid too many requests
        """
        now_sg    = datetime.now(self.sg_tz)
        today_str = now_sg.strftime("%Y-%m-%d")

        # Return cache if same day
        if self._cached_date == today_str and self._cache is not None:
            return self._cache

        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            r   = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            if r.status_code != 200:
                log.warning("Calendar API returned: " + str(r.status_code))
                return []

            all_events   = r.json()
            high_impacts = []

            for event in all_events:
                try:
                    impact   = event.get("impact", "").lower()
                    currency = event.get("currency", "")
                    title    = event.get("title", "")
                    date_str = event.get("date", "")

                    # Only HIGH impact events for USD, GBP, EUR
                    if impact != "high":
                        continue
                    if currency not in ["USD", "GBP", "EUR"]:
                        continue

                    high_impacts.append({
                        "date":     date_str,
                        "currency": currency,
                        "title":    title,
                        "impact":   "HIGH"
                    })

                except Exception as e:
                    log.warning("Event parse error: " + str(e))
                    continue

            # Cache result
            self._cache       = high_impacts
            self._cached_date = today_str

            log.info("Calendar loaded! " + str(len(high_impacts)) + " high impact events this week")
            for e in high_impacts:
                log.info("  " + e["currency"] + " " + e["title"] + " @ " + e["date"])

            return high_impacts

        except Exception as e:
            log.warning("Calendar fetch failed: " + str(e))
            return []

    def _get_affected_currencies(self, instrument):
        """Which currencies affect this instrument"""
        affected = ["USD"]  # USD affects everything!
        if "EUR" in instrument:
            affected.append("EUR")
        if "GBP" in instrument:
            affected.append("GBP")
        if "XAU" in instrument:
            # Gold affected by ALL major currencies!
            affected.extend(["EUR", "GBP"])
        return affected

    def is_news_time(self, instrument="EUR_USD"):
        """
        Check if current time is within news blackout window

        Returns: (is_blackout, reason)
        
        Timeline:
        T-30 mins → PAUSED (preparing for news)
        T+00 mins → NEWS RELEASED (very volatile!)
        T+30 mins → PAUSED (market digesting news)
        T+31 mins → RESUMED (safe to trade again!)
        """
        now_utc  = datetime.utcnow().replace(tzinfo=self.utc_tz)
        affected = self._get_affected_currencies(instrument)
        events   = self._fetch_events()

        if not events:
            # API failed - allow trading but log warning
            log.warning("Calendar unavailable - trading without news filter!")
            return False, ""

        for event in events:
            if event["currency"] not in affected:
                continue

            try:
                date_str = event.get("date", "")
                if not date_str:
                    continue

                # Parse ForexFactory date format
                # Format: "2026-03-07T13:30:00-0500"
                try:
                    if "T" in date_str:
                        # Remove timezone offset and parse
                        clean = date_str[:19]
                        offset_str = date_str[19:]
                        event_dt = datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")

                        # Apply timezone offset
                        if "+" in offset_str or (offset_str.startswith("-") and len(offset_str) > 1):
                            sign = 1 if "+" in offset_str else -1
                            offset_str = offset_str.replace("+", "").replace("-", "")
                            if ":" in offset_str:
                                h, m = offset_str.split(":")
                            else:
                                h = offset_str[:2]
                                m = offset_str[2:] if len(offset_str) > 2 else "00"
                            offset = timedelta(hours=int(h), minutes=int(m)) * sign
                            event_dt = event_dt - offset  # Convert to UTC

                        event_utc = event_dt.replace(tzinfo=self.utc_tz)
                    else:
                        # Date only - use noon UTC as estimate
                        event_dt  = datetime.strptime(date_str[:10], "%Y-%m-%d")
                        event_utc = event_dt.replace(hour=12, tzinfo=self.utc_tz)

                except Exception as parse_err:
                    log.warning("Date parse error: " + str(parse_err) + " for " + date_str)
                    continue

                # Check 30 min window
                window_start = event_utc - timedelta(minutes=30)
                window_end   = event_utc + timedelta(minutes=30)

                if window_start <= now_utc <= window_end:
                    mins_to = int((event_utc - now_utc).total_seconds() / 60)

                    if mins_to > 0:
                        reason = (event["currency"] + " " + event["title"] +
                                 " in " + str(mins_to) + " mins!")
                    elif mins_to == 0:
                        reason = (event["currency"] + " " + event["title"] +
                                 " releasing NOW!")
                    else:
                        reason = (event["currency"] + " " + event["title"] +
                                 " released " + str(abs(mins_to)) + " mins ago")

                    log.warning("NEWS BLACKOUT: " + reason)
                    return True, reason

            except Exception as e:
                log.warning("News check error: " + str(e))
                continue

        return False, ""

    def get_today_summary(self):
        """
        Get today high impact events for Telegram morning alert
        Bot sends this at start of each session
        """
        now_sg    = datetime.now(self.sg_tz)
        today_str = now_sg.strftime("%Y-%m-%d")
        events    = self._fetch_events()

        today_events = []
        for event in events:
            try:
                event_date = event.get("date", "")[:10]
                if event_date == today_str:
                    today_events.append(event)
            except:
                continue

        if not today_events:
            return "No high impact news today - safe to trade!"

        # Convert times to SGT for display
        lines = ["High impact news TODAY:"]
        for e in today_events:
            try:
                date_str = e.get("date", "")
                if "T" in date_str:
                    clean    = date_str[:19]
                    event_dt = datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
                    # Add SGT offset (+8)
                    sgt_dt   = event_dt + timedelta(hours=8)
                    time_str = sgt_dt.strftime("%H:%M SGT")
                else:
                    time_str = "time TBC"

                lines.append(e["currency"] + " " + e["title"] + " @ " + time_str)
            except:
                lines.append(e["currency"] + " " + e["title"])

        lines.append("Bot pauses 30 mins before/after!")
        return "\n".join(lines)

    def get_week_summary(self):
        """Get full week events - useful for Monday morning alert"""
        events = self._fetch_events()
        if not events:
            return "Calendar unavailable this week"

        lines = ["High impact events this week:"]
        for e in events:
            try:
                date_str = e.get("date", "")[:10]
                lines.append(date_str + " " + e["currency"] + ": " + e["title"])
            except:
                continue

        return "\n".join(lines) if len(lines) > 1 else "No high impact events this week"
