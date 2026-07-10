"""
scheduler.py
------------
Milestone 4 – MLOps: Automation Scheduler

Automates:
  • Daily monitoring checks
  • Weekly retraining (if needed)
  • Alert notifications
"""

import json
import schedule
import time
import threading
from pathlib import Path
from datetime import datetime

BASE_DIR  = Path(__file__).parents[2]
MLOPS_DIR = BASE_DIR / "mlops"
MLOPS_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULE_LOG = MLOPS_DIR / "schedule_log.json"


def log_event(event: str, status: str, details: dict = None):
    logs = []
    if SCHEDULE_LOG.exists():
        with open(SCHEDULE_LOG) as f:
            logs = json.load(f)
    logs.append({
        "timestamp": datetime.utcnow().isoformat(),
        "event":     event,
        "status":    status,
        "details":   details or {},
    })
    with open(SCHEDULE_LOG, "w") as f:
        json.dump(logs[-500:], f, indent=2)
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {event} → {status}")


def run_monitoring_job():
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from src.monitoring.monitor import run_monitoring
        report = run_monitoring()
        log_event("MONITORING", "SUCCESS", {
            "retraining_needed": report["retraining_recommended"],
            "drifted_features":  [k for k, v in report["data_drift"].items() if v["drift_detected"]],
        })
        # Trigger retraining if needed
        if report["retraining_recommended"]:
            print("  → Drift detected! Triggering retraining ...")
            run_retraining_job()
    except Exception as e:
        log_event("MONITORING", "FAILED", {"error": str(e)})


def run_retraining_job():
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from src.retraining.retrain import run_retraining
        result = run_retraining()
        log_event("RETRAINING", result["status"], {
            "version":     result.get("version"),
            "lgb_mape":    result.get("lgb_metrics", {}).get("MAPE"),
        })
    except Exception as e:
        log_event("RETRAINING", "FAILED", {"error": str(e)})


def start_scheduler(run_once: bool = False):
    print("=" * 50)
    print("   MLOps Scheduler Started")
    print("=" * 50)
    print("  • Monitoring  → Every day at 06:00 AM")
    print("  • Retraining  → Every Monday at 02:00 AM")
    print("  • Log file    → mlops/schedule_log.json")
    print("=" * 50)

    if run_once:
        print("\n[DEMO] Running jobs once for demonstration ...")
        run_monitoring_job()
        return

    schedule.every().day.at("06:00").do(run_monitoring_job)
    schedule.every().monday.at("02:00").do(run_retraining_job)

    log_event("SCHEDULER", "STARTED", {"jobs": ["daily_monitoring", "weekly_retraining"]})

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    # Run once for demo; set run_once=False for production
    start_scheduler(run_once=True)
