"""
Telegram Alert System - Clean Messages
"""
import os
import requests
import logging

log = logging.getLogger(__name__)

class TelegramAlert:
    def __init__(self):
        self.token   = os.environ.get("TELEGRAM_TOKEN", "")
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    def send(self, message: str):
        if not self.token or not self.chat_id:
            log.warning("Telegram not configured")
            return False
        try:
            url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
            text = f"🤖 OANDA Bot\n{'─'*22}\n{message}"
            data = {"chat_id": self.chat_id, "text": text}
            r    = requests.post(url, data=data, timeout=10)
            if r.status_code == 200:
                log.info("Telegram sent!")
                return True
            log.warning(f"Telegram error: {r.text}")
            return False
        except Exception as e:
            log.error(f"Telegram error: {e}")
            return False
