"""
OANDA Trade Executor
Stop Loss + Take Profit set automatically on every order!
OANDA closes trades automatically when SL or TP is hit!
"""

import os
import requests
import logging

log = logging.getLogger(__name__)

class OandaTrader:
    def __init__(self, demo=True):
        self.api_key    = os.environ.get("OANDA_API_KEY", "")
        self.account_id = os.environ.get("OANDA_ACCOUNT_ID", "")
        self.demo       = demo
        self.base_url   = "https://api-fxpractice.oanda.com" if demo else "https://api-trade.oanda.com"
        self.headers    = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }
        log.info(f"OANDA | Mode: {'DEMO' if demo else 'LIVE'}")
        log.info(f"Account: {self.account_id}")
        log.info(f"API Key: {self.api_key[:8]}****")

    def login(self):
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}"
            r   = requests.get(url, headers=self.headers, timeout=15)
            log.info(f"Login status: {r.status_code}")
            log.info(f"Login response: {r.text[:300]}")
            if r.status_code == 200:
                bal = float(r.json()["account"]["balance"])
                log.info(f"Login success! Balance: ${bal:.2f}")
                return True
            log.error(f"Login failed: {r.status_code} {r.text}")
            return False
        except Exception as e:
            log.error(f"Login error: {e}")
            return False

    def get_balance(self):
        try:
            r   = requests.get(f"{self.base_url}/v3/accounts/{self.account_id}", headers=self.headers, timeout=10)
            bal = float(r.json()["account"]["balance"])
            log.info(f"Balance: ${bal:.2f}")
            return bal
        except Exception as e:
            log.error(f"get_balance error: {e}")
            return 0

    def get_price(self, instrument):
        try:
            r     = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/pricing",
                headers=self.headers,
                params={"instruments": instrument},
                timeout=10
            )
            price = r.json()["prices"][0]
            bid   = float(price["bids"][0]["price"])
            ask   = float(price["asks"][0]["price"])
            mid   = (bid + ask) / 2
            return mid, bid, ask
        except Exception as e:
            log.error(f"get_price error: {e}")
            return None, None, None

    def get_position(self, instrument):
        try:
            r = requests.get(
                f"{self.base_url}/v3/accounts/{self.account_id}/positions/{instrument}",
                headers=self.headers,
                timeout=10
            )
            if r.status_code == 200:
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
            long_pnl  = float(position["long"].get("unrealizedPL", 0))
            short_pnl = float(position["short"].get("unrealizedPL", 0))
            return long_pnl + short_pnl
        except:
            return 0

    def place_order(self, instrument, direction, size, stop_distance, limit_distance, currency="USD"):
        try:
            units = size if direction == "BUY" else -size

            # Get current price for SL/TP calculation
            price, bid, ask = self.get_price(instrument)
            if price is None:
                return {"success": False, "error": "Cannot get price"}

            # Pip size per instrument
            if instrument == "XAU_USD":
                pip = 0.01        # Gold: 1 pip = $0.01
            elif "JPY" in instrument:
                pip = 0.01        # JPY pairs
            else:
                pip = 0.0001      # Forex: 1 pip = 0.0001

            # Decimal precision per instrument
            if instrument == "XAU_USD":
                precision = 2     # Gold: 2 decimals e.g. 5140.25
            elif "JPY" in instrument:
                precision = 3     # JPY: 3 decimals e.g. 149.500
            else:
                precision = 5     # Forex: 5 decimals e.g. 1.16080

            # Entry price
            entry = ask if direction == "BUY" else bid

            # SL and TP prices with correct precision
            if direction == "BUY":
                sl_price = round(entry - (stop_distance  * pip), precision)
                tp_price = round(entry + (limit_distance * pip), precision)
            else:
                sl_price = round(entry + (stop_distance  * pip), precision)
                tp_price = round(entry - (limit_distance * pip), precision)

            log.info(f"Placing {direction} {instrument} | units={units} | entry={entry} | SL={sl_price} | TP={tp_price}")

            payload = {
                "order": {
                    "type":        "MARKET",
                    "instrument":  instrument,
                    "units":       str(units),
                    "timeInForce": "FOK",
                    "stopLossOnFill": {
                        "price":       str(sl_price),
                        "timeInForce": "GTC"
                    },
                    "takeProfitOnFill": {
                        "price":       str(tp_price),
                        "timeInForce": "GTC"
                    }
                }
            }

            r    = requests.post(
                f"{self.base_url}/v3/accounts/{self.account_id}/orders",
                headers=self.headers,
                json=payload,
                timeout=15
            )
            data = r.json()
            log.info(f"Order response: {r.status_code} {str(data)[:300]}")

            if r.status_code in [200, 201]:
                if "orderFillTransaction" in data:
                    trade_id = data["orderFillTransaction"].get("id", "N/A")
                    log.info(f"Trade placed! ID: {trade_id}")
                    return {"success": True, "trade_id": trade_id}
                elif "orderCancelTransaction" in data:
                    reason = data["orderCancelTransaction"].get("reason", "Unknown")
                    return {"success": False, "error": f"Order cancelled: {reason}"}
                return {"success": True}
            else:
                error = data.get("errorMessage", str(data))
                return {"success": False, "error": error}

        except Exception as e:
            log.error(f"place_order error: {e}")
            return {"success": False, "error": str(e)}

    def close_position(self, instrument):
        try:
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
