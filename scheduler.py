from __future__ import annotations

import signal
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from bot import run_bot_cycle
from auto_tuner import run_auto_tune_daily
from oanda_trader import OandaTrader
from reporting import (send_daily_report, send_weekly_report, send_monthly_report,
                       send_asian_session_report, send_london_session_report, send_us_session_report)
from telegram_alert import TelegramAlert
from telegram_templates import msg_startup
from config_loader import DATA_DIR, load_settings
from database import Database
from logging_utils import configure_logging, get_logger
from startup_checks import run_startup_checks

configure_logging()
logger = get_logger(__name__)
SG_TZ = pytz.timezone('Asia/Singapore')

# ── Health-check HTTP server ───────────────────────────────────────────────────
# Railway (and other PaaS platforms) can poll GET /health to confirm the process
# is alive. Runs on PORT env-var (default 8080) in a daemon thread so it never
# blocks the scheduler. Returns 200 OK while the scheduler is running; the
# process exit itself signals failure to the platform.

_scheduler_ref: BlockingScheduler | None = None


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b"OK"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):  # silence access logs
        pass


def _start_health_server(port: int = 8080) -> None:
    """Start the health-check HTTP server in a background daemon thread."""
    import os
    port = int(os.environ.get("PORT", port))
    try:
        server = HTTPServer(("0.0.0.0", port), _HealthHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True, name="health-server")
        t.start()
        logger.info("Health-check server listening on port %d — GET /health", port)
    except Exception as exc:
        logger.warning("Could not start health-check server on port %d: %s", port, exc)


def run_db_retention_cleanup():
    settings = load_settings()
    retention_days = int(settings.get('db_retention_days', 90))
    vacuum_weekly = bool(settings.get('db_vacuum_weekly', True))
    is_weekly_vacuum_day = datetime.now(SG_TZ).weekday() == 6

    logger.info('Starting DB retention cleanup | retention_days=%s | weekly_vacuum=%s', retention_days, vacuum_weekly)
    try:
        db = Database()
        summary = db.purge_old_data(retention_days=retention_days, vacuum=bool(vacuum_weekly and is_weekly_vacuum_day))
        logger.info('DB retention cleanup complete: %s', summary)
    except Exception as exc:
        logger.exception('DB retention cleanup failed: %s', exc)


def main():
    settings = load_settings()
    cycle_minutes = int(settings.get('cycle_minutes', 5))
    cleanup_hour = int(settings.get('db_cleanup_hour_sgt', 0))
    cleanup_minute = int(settings.get('db_cleanup_minute_sgt', 15))
    retention_days = int(settings.get('db_retention_days', 90))

    _start_health_server()

    logger.info('%s — Scheduler starting', settings.get('bot_name', 'CPR Gold Bot'))
    logger.info('DATA_DIR : %s', DATA_DIR)
    logger.info('Python   : %s', sys.version.split()[0])
    for warning in run_startup_checks():
        logger.warning(warning)

    scheduler = BlockingScheduler(timezone=SG_TZ)
    scheduler.add_job(
        run_bot_cycle,
        IntervalTrigger(minutes=cycle_minutes),
        id='trade_cycle',
        name=f'{cycle_minutes}-min trade cycle',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=max(cycle_minutes * 60, 60),
    )

    scheduler.add_job(
        run_db_retention_cleanup,
        CronTrigger(hour=cleanup_hour, minute=cleanup_minute, timezone=SG_TZ),
        id='db_retention_cleanup',
        name=f'DB retention cleanup ({retention_days}-day rolling)',
        max_instances=1,
        coalesce=True,
    )

    # ── Telegram performance reports ───────────────────────────────────────────
    # Monthly: first Monday of each month at 08:00 SGT
    # The first-Monday guard is enforced inside send_monthly_report() itself,
    # so this job fires every Monday but only sends on the first Monday.
    scheduler.add_job(
        send_monthly_report,
        CronTrigger(day_of_week='mon', hour=8, minute=0, timezone=SG_TZ),
        id='monthly_report',
        name='Monthly performance report (first Monday)',
        max_instances=1,
        coalesce=True,
    )

    # Weekly: every Monday at 08:15 SGT (covers prior Mon–Fri)
    scheduler.add_job(
        send_weekly_report,
        CronTrigger(day_of_week='mon', hour=8, minute=15, timezone=SG_TZ),
        id='weekly_report',
        name='Weekly performance report',
        max_instances=1,
        coalesce=True,
    )

    # Daily: Mon–Fri at 15:30 SGT (30 min before London session open at 16:00)
    scheduler.add_job(
        send_daily_report,
        CronTrigger(day_of_week='mon-fri', hour=15, minute=30, timezone=SG_TZ),
        id='daily_report',
        name='Daily performance report',
        max_instances=1,
        coalesce=True,
    )

    # Auto-tuner: daily at 08:00 SGT — analyses last N trades, adjusts settings
    scheduler.add_job(
        run_auto_tune_daily,
        CronTrigger(hour=8, minute=0, timezone=SG_TZ),
        id='auto_tune_daily',
        name='Daily auto-tune (pattern analysis + settings adjustment)',
        max_instances=1,
        coalesce=True,
    )
    # v4.4 — per-session summary reports (~5 min after each session closes)
    _asian_rpt_h  = int(settings.get('asian_report_hour_sgt',   16))
    _asian_rpt_m  = int(settings.get('asian_report_minute_sgt',  5))
    _lon_rpt_h    = int(settings.get('london_report_hour_sgt',  21))
    _lon_rpt_m    = int(settings.get('london_report_minute_sgt',  5))
    _us_rpt_h     = int(settings.get('us_report_hour_sgt',       1))
    _us_rpt_m     = int(settings.get('us_report_minute_sgt',     5))
    scheduler.add_job(
        send_asian_session_report,
        CronTrigger(day_of_week='mon-fri', hour=_asian_rpt_h, minute=_asian_rpt_m, timezone=SG_TZ),
        id='asian_session_report',
        name='Asian session performance report',
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        send_london_session_report,
        CronTrigger(day_of_week='mon-fri', hour=_lon_rpt_h, minute=_lon_rpt_m, timezone=SG_TZ),
        id='london_session_report',
        name='London session performance report',
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        send_us_session_report,
        CronTrigger(day_of_week='mon-fri', hour=_us_rpt_h, minute=_us_rpt_m, timezone=SG_TZ),
        id='us_session_report',
        name='US session performance report',
        max_instances=1, coalesce=True,
    )

    def _graceful_shutdown(signum, frame):
        logger.info('Received signal %s — waiting for active cycle to finish (max 120 s)...', signum)
        # wait=True lets any running trade cycle complete before exit,
        # preventing a half-placed order that is never recorded locally.
        # The thread + join(timeout) provides a hard 120 s safety cap.
        t = threading.Thread(
            target=lambda: scheduler.shutdown(wait=True),
            daemon=True,
            name="scheduler-shutdown",
        )
        t.start()
        t.join(timeout=120)
        if t.is_alive():
            logger.warning('Shutdown timeout reached (120 s) — forcing exit.')
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    logger.info('Jobs scheduled:')
    logger.info('  Trade cycle    — every %s minutes', cycle_minutes)
    logger.info('  DB cleanup     — daily at %02d:%02d Asia/Singapore', cleanup_hour, cleanup_minute)
    logger.info('  DB retention   — rolling %s days', retention_days)
    logger.info('  Monthly report — first Monday of month at 08:00 SGT')
    logger.info('  Weekly report  — every Monday at 08:15 SGT')
    logger.info('  Daily report   — Mon–Fri at 15:30 SGT (30 min before London open)')

    logger.info('Running startup cycle...')
    try:
        from version import __version__, BOT_NAME
        _trader  = OandaTrader(demo=bool(settings.get('demo_mode', True)))
        _summary = _trader.login_with_summary()
        _balance = _summary["balance"] if _summary else 0.0
        _threshold = int(settings.get('signal_threshold', 4))
        _mode    = 'DEMO' if settings.get('demo_mode', True) else 'LIVE'
        _version = f"{BOT_NAME} v{__version__}"
        TelegramAlert().send(msg_startup(
            _version, _mode, _balance, _threshold,
            cycle_minutes=int(settings.get('cycle_minutes', 5)),
            max_trades_london=int(settings.get('max_trades_london', 10)),
            max_trades_us=int(settings.get('max_trades_us', 10)),
            max_trades_tokyo=int(settings.get('max_trades_asian', 5)),
            max_losing_day=int(settings.get('max_losing_trades_day', 8)),
            trading_day_start_hour=int(settings.get('trading_day_start_hour_sgt', 8)),
            tokyo_start=8, tokyo_end=15,
            london_start=16, london_end=20,
            us_start=21, us_end=23, us_early_end=1,
            position_full_usd=int(settings.get('position_full_usd', 100)),
            position_partial_usd=int(settings.get('position_partial_usd', 66)),
            session_thresholds=settings.get('session_thresholds', {}),
            h1_filter_enabled=bool(settings.get('h1_trend_filter_enabled', True)),
            h1_filter_mode=str(settings.get('h1_filter_mode', 'hard')),
        ))
    except Exception as _e:
        logger.warning('Could not send startup Telegram alert: %s', _e)
    run_bot_cycle()
    scheduler.start()


if __name__ == '__main__':
    main()
