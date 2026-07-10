"""
monitor.py
----------
Milestone 4 – MLOps: Model Monitoring & Drift Detection

Features:
  • Data drift detection (KS test, PSI)
  • Model performance degradation alerts
  • Prediction logging
  • Automated alert system
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parents[2]
MONITOR_DIR = BASE_DIR / "mlops" / "monitoring"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE    = MONITOR_DIR / "prediction_logs.json"
ALERT_FILE  = MONITOR_DIR / "alerts.json"
REPORT_FILE = MONITOR_DIR / "monitoring_report.json"

# ── Thresholds ────────────────────────────────────────────────
DRIFT_THRESHOLD     = 0.05   # p-value below this = drift detected
PSI_THRESHOLD       = 0.2    # PSI above this = significant shift
MAPE_THRESHOLD      = 10.0   # MAPE above this = model degraded
MAE_INCREASE_PCT    = 20.0   # MAE increase % = alert

BASELINE_STATS = {
    "units_sold": {"mean": 20.85, "std": 10.98, "min": 0.0,  "max": 65.0},
    "price":      {"mean": 89.42, "std": 66.14, "min": 8.42, "max": 292.4},
    "stock_level":{"mean": 0.74,  "std": 10.26, "min": 0.0,  "max": 285.0},
}

BASELINE_METRICS = {"MAE": 82.64, "RMSE": 106.05, "MAPE": 3.15}


# ── PSI Calculation ───────────────────────────────────────────
def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins=10) -> float:
    expected = np.clip(expected, np.percentile(expected, 1), np.percentile(expected, 99))
    actual   = np.clip(actual,   np.percentile(actual,   1), np.percentile(actual,   99))
    breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 0.0001, exp_pct)
    act_pct = np.where(act_pct == 0, 0.0001, act_pct)
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi), 4)


# ── KS Drift Test ─────────────────────────────────────────────
def detect_drift(baseline_data: np.ndarray, new_data: np.ndarray, feature_name: str) -> dict:
    ks_stat, p_value = stats.ks_2samp(baseline_data, new_data)
    psi              = calculate_psi(baseline_data, new_data)
    drift_detected   = (p_value < DRIFT_THRESHOLD) or (psi > PSI_THRESHOLD)
    return {
        "feature":        feature_name,
        "ks_statistic":   round(float(ks_stat), 4),
        "p_value":        round(float(p_value), 4),
        "psi":            psi,
        "drift_detected": bool(drift_detected),
        "severity":       "HIGH" if psi > 0.25 else "MEDIUM" if psi > 0.1 else "LOW",
    }


# ── Log Predictions ───────────────────────────────────────────
def log_prediction(prediction: dict) -> None:
    logs = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            logs = json.load(f)
    prediction["logged_at"] = datetime.utcnow().isoformat()
    logs.append(prediction)
    with open(LOG_FILE, "w") as f:
        json.dump(logs[-1000:], f, indent=2)   # keep last 1000


# ── Performance Monitor ───────────────────────────────────────
def check_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    mae_increase  = ((mae  - BASELINE_METRICS["MAE"])  / BASELINE_METRICS["MAE"])  * 100
    mape_increase = ((mape - BASELINE_METRICS["MAPE"]) / BASELINE_METRICS["MAPE"]) * 100

    alerts = []
    if mape > MAPE_THRESHOLD:
        alerts.append(f"⚠️  MAPE={mape:.2f}% exceeds threshold {MAPE_THRESHOLD}%")
    if mae_increase > MAE_INCREASE_PCT:
        alerts.append(f"⚠️  MAE increased by {mae_increase:.1f}% vs baseline")

    return {
        "current_metrics":  {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mape,2)},
        "baseline_metrics": BASELINE_METRICS,
        "mae_change_pct":   round(mae_increase, 2),
        "mape_change_pct":  round(mape_increase, 2),
        "degraded":         len(alerts) > 0,
        "alerts":           alerts,
    }


# ── Full Monitoring Run ───────────────────────────────────────
def run_monitoring(new_data: pd.DataFrame = None) -> dict:
    print("=" * 55)
    print("   MLOps – Model Monitoring Report")
    print("=" * 55)

    # Simulate new incoming data if not provided
    if new_data is None:
        np.random.seed(99)
        n = 500
        new_data = pd.DataFrame({
            "units_sold":  np.random.poisson(22, n),      # slightly shifted mean
            "price":       np.random.uniform(10, 300, n),
            "stock_level": np.random.exponential(2, n),
        })

    # Generate simulated predictions for performance check
    y_true = new_data["units_sold"].values
    y_pred = y_true * np.random.uniform(0.88, 1.12, len(y_true))   # ±12% noise

    # 1. Data Drift Detection
    print("\n[DRIFT] Checking data drift ...")
    drift_results = {}
    for feat in ["units_sold", "price", "stock_level"]:
        baseline_arr = np.random.normal(
            BASELINE_STATS[feat]["mean"],
            BASELINE_STATS[feat]["std"], 1000
        )
        result = detect_drift(baseline_arr, new_data[feat].values, feat)
        drift_results[feat] = result
        icon = "🔴" if result["drift_detected"] else "✅"
        print(f"  {icon} {feat}: PSI={result['psi']} | p={result['p_value']} | drift={result['drift_detected']}")

    # 2. Performance Check
    print("\n[PERFORMANCE] Checking model performance ...")
    perf = check_performance(y_true, y_pred)
    print(f"  Current  → MAE={perf['current_metrics']['MAE']} | RMSE={perf['current_metrics']['RMSE']} | MAPE={perf['current_metrics']['MAPE']}%")
    print(f"  Baseline → MAE={BASELINE_METRICS['MAE']} | RMSE={BASELINE_METRICS['RMSE']} | MAPE={BASELINE_METRICS['MAPE']}%")
    if perf["alerts"]:
        for alert in perf["alerts"]:
            print(f"  {alert}")
    else:
        print("  ✅ Model performance within acceptable range")

    # 3. Retraining recommendation
    any_drift    = any(v["drift_detected"] for v in drift_results.values())
    needs_retrain = any_drift or perf["degraded"]
    print(f"\n[DECISION] Retraining needed: {'🔴 YES' if needs_retrain else '✅ NO'}")

    # 4. Save report
    report = {
        "timestamp":             datetime.utcnow().isoformat(),
        "data_drift":            drift_results,
        "performance":           perf,
        "retraining_recommended": bool(needs_retrain),
        "new_data_samples":      len(new_data),
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    if perf["alerts"]:
        with open(ALERT_FILE, "w") as f:
            json.dump({"timestamp": datetime.utcnow().isoformat(), "alerts": perf["alerts"]}, f, indent=2)

    print(f"\n✅ Report saved → {REPORT_FILE}")
    return report


if __name__ == "__main__":
    run_monitoring()
