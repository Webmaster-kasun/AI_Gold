"""
OANDA Trade Executor - Minimal API calls version
- Single account fetch gives balance + positions + margin
- 1 second delay between calls
- Retry on 429
"""

import os
import time
import requests
import logging

log = logging.getLogger(__name__)


class OandaTrader:
    def __init__(self, demo=True):
        self.api_key      = os.environ.get("OANDA_API_KEY", "")
        self.account_id   = os.environ.get("OANDA_ACCOUNT_ID", "")
        self.demo         = demo
        self.base_url     = "https://api-fxpractice.oanda.com" if demo else "https://api-fxtrade.oanda.com"
        self.headers      = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }
        self.last_balance          = 0.0
        self.last_margin_available = 0.0
        self._account_cache        = None   # full account snapshot — reused within same scan
        log.info(f"OANDA | Mode: {'DEMO/PRACTICE' if demo else 'LIVE'}")
        log.info(f"OANDA | URL:  {self.base_url}")
        log.info(f"Account: {self.account_id}")
        if self.api_key:
            log.info(f"API Key: {self.api_key[:8]}****")
        else:
            log.error("API Key EMPTY — check OANDA_API_KEY")

    def _get(self, url, params=None, retries=3):
        time.sleep(1.0)   # 1 second between every call — conservative
        for attempt in range(retries):
            try:
                r = requests.get(url, headers=self.headers, params=params, timeout=20)
                if r.status_code == 429:
                    wait = 30 * (attempt + 1)
                    log.warning(f"Rate limited — waiting {wait}s")
                    time.sleep(wait)
                    continue
                return r
            except Exception as e:
                log.warning(f"Request error attempt {attempt+1}: {e}")
                time.sleep(5)
        return None

    def login(self):
        """
        Single call — fetches account summary which gives us:
        balance, margin, open trade count — cached for whole scan.
        """
        if not self.api_key or not self.account_id:
            log.error("Login failed: missing API key or account ID")
            return False
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
            r   = self._get(url)
            if r is None:
                log.error("Login failed: no response")
                return False
            if r.status_code == 200:
                acct                       = r.json()["account"]
                self.last_balance          = float(acct["balance"])
                self.last_margin_available = float(acct.get("marginAvailable", self.last_balance))
                self._account_cache        = acct
                log.info(f"Login OK | Balance: ${self.last_balance:.2f} | Margin: ${self.last_margin_available:.2f}")
                return True
            log.error(f"Login failed: {r.status_code} {r.text[:200]}")
            return False
        except Exception as e:
            log.error(f"Login error: {e}")
            return False

    def get_balance(self):
        return self.last_balance   # cached — no API call

    def get_open_trade_count(self):
        """From cache — no API call."""
        if self._account_cache:
            return int(self._account_cache.get("openTradeCount", 0))
        return 0

    def get_price(self, instrument):
        try:
            r = self._get(
                f"{self.base_url}/v3/accounts/{self.account_id}/pricing",
                params={"instruments": instrument}
            )
            if r and r.status_code == 200:
                p   = r.json()["prices"][0]
                bid = float(p["bids"][0]["price"])
                ask = float(p["asks"][0]["price"])
                return (bid + ask) / 2, bid, ask
        except Exception as e:
            log.error(f"get_price error: {e}")
        return None, None, None

    def get_position(self, instrument):
        """Always check live position — never skip, stale cache causes opposite trades."""
        try:
            r = self._get(f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}")
            if r and r.status_code == 200:
                pos         = r.json()["position"]
                long_units  = int(float(pos["long"]["units"]))
                short_units = int(float(pos["short"]["units"]))
                if long_units != 0 or short_units != 0:
                    return pos
            return None
        except Exception as e:
            log.error(f"get_position error: {e}")
            return None

    def check_pnl(self, position):
        try:
            return float(position["long"].get("unrealizedPL", 0)) + \
                   float(position["short"].get("unrealizedPL", 0))
        except:
            return 0

    def get_margin_available(self):
        return self.last_margin_available  # cached — no API call

    def place_order(self, instrument, direction, size, stop_distance, limit_distance):
        try:
            units       = size if direction == "BUY" else -size
            price, bid, ask = self.get_price(instrument)
            if price is None:
                return {"success": False, "error": "Cannot get price"}

            pip       = 0.01 if instrument == "XAU_USD" else (0.01 if "JPY" in instrument else 0.0001)
            precision = 2    if instrument == "XAU_USD" else (3    if "JPY" in instrument else 5)
            entry     = ask if direction == "BUY" else bid

            sl_price = round(entry - stop_distance * pip, precision) if direction == "BUY" \
                       else round(entry + stop_distance * pip, precision)
            tp_price = round(entry + limit_distance * pip, precision) if direction == "BUY" \
                       else round(entry - limit_distance * pip, precision)

            log.info(f"Order: {direction} {instrument} x{units} | entry={entry} SL={sl_price} TP={tp_price}")

            time.sleep(1.0)
            r = requests.post(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders",
                headers=self.headers,
                json={"order": {
                    "type": "MARKET", "instrument": instrument,
                    "units": str(units), "timeInForce": "FOK",
                    "stopLossOnFill":   {"price": str(sl_price), "timeInForce": "GTC"},
                    "takeProfitOnFill": {"price": str(tp_price), "timeInForce": "GTC"}
                }},
                timeout=15
            )
            data = r.json()
            log.info(f"Order response: {r.status_code} {str(data)[:200]}")
            if r.status_code in [200, 201]:
                if "orderFillTransaction" in data:
                    return {"success": True, "trade_id": data["orderFillTransaction"].get("id")}
                elif "orderCancelTransaction" in data:
                    return {"success": False, "error": "Cancelled: " + data["orderCancelTransaction"].get("reason", "?")}
                return {"success": True}
            return {"success": False, "error": data.get("errorMessage", str(data))}
        except Exception as e:
            log.error(f"place_order error: {e}")
            return {"success": False, "error": str(e)}

    def close_position(self, instrument):
        try:
            time.sleep(1.0)
            r = requests.put(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}/close",
                headers=self.headers,
                json={"longUnits": "ALL", "shortUnits": "ALL"},
                timeout=15
            )
            return {"success": r.status_code == 200}
        except Exception as e:
            log.error(f"close_position error: {e}")
            return {"success": False}
