"""
retrain.py
----------
Milestone 4 – MLOps: Automated Retraining Pipeline

Triggered when:
  • Data drift detected
  • Model MAPE > threshold
  • Scheduled (weekly)
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR      = Path(__file__).parents[2]
PROCESSED_PATH= BASE_DIR / "data" / "processed" / "retail_sales_processed.csv"
MODEL_DIR     = BASE_DIR / "models"
MLOPS_DIR     = BASE_DIR / "mlops"
RETRAIN_DIR   = MLOPS_DIR / "retraining"
RETRAIN_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "month", "week", "day_of_week", "quarter", "is_weekend",
    "is_promotion", "is_holiday", "price", "stock_level",
    "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28",
    "store_type_enc", "category_enc",
    "price_discount", "stockout_flag", "year",
]
TARGET = "units_sold"


def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_latest_data() -> tuple:
    print("[LOAD] Loading latest processed data ...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])

    agg = df.groupby(["date", "category", "category_enc"]).agg(
        units_sold        =("units_sold",      "sum"),
        price             =("price",           "mean"),
        stock_level       =("stock_level",     "sum"),
        is_promotion      =("is_promotion",    "max"),
        is_holiday        =("is_holiday",      "max"),
        is_weekend        =("is_weekend",      "max"),
        month             =("month",           "first"),
        week              =("week",            "first"),
        day_of_week       =("day_of_week",     "first"),
        quarter           =("quarter",         "first"),
        year              =("year",            "first"),
        month_sin         =("month_sin",       "first"),
        month_cos         =("month_cos",       "first"),
        dayofweek_sin     =("dayofweek_sin",   "first"),
        dayofweek_cos     =("dayofweek_cos",   "first"),
        lag_1             =("lag_1",           "sum"),
        lag_7             =("lag_7",           "sum"),
        lag_14            =("lag_14",          "sum"),
        lag_28            =("lag_28",          "sum"),
        rolling_mean_7    =("rolling_mean_7",  "mean"),
        rolling_mean_14   =("rolling_mean_14", "mean"),
        rolling_mean_28   =("rolling_mean_28", "mean"),
        rolling_std_7     =("rolling_std_7",   "mean"),
        rolling_std_28    =("rolling_std_28",  "mean"),
        price_discount    =("price_discount",  "max"),
        stockout_flag     =("stockout_flag",   "max"),
        store_type_enc    =("store_type_enc",  "first"),
    ).reset_index().sort_values("date").dropna()

    available = [f for f in FEATURES if f in agg.columns]
    X = agg[available]
    y = agg[TARGET]
    dates = agg["date"]
    print(f"  Samples: {len(X)} | Features: {len(available)}")
    return X, y, dates, available


def retrain_lightgbm(X_train, y_train, X_val, y_val, features) -> tuple:
    print("\n[RETRAIN] Training LightGBM ...")
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04,
        max_depth=6, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    preds = np.maximum(model.predict(X_val), 0)
    metrics = {
        "MAE":  round(mean_absolute_error(y_val, preds), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_val, preds)), 2),
        "MAPE": round(mape(y_val.values, preds), 2),
    }
    print(f"  MAE={metrics['MAE']} | RMSE={metrics['RMSE']} | MAPE={metrics['MAPE']}%")
    return model, metrics


def retrain_xgboost(X_train, y_train, X_val, y_val) -> tuple:
    print("\n[RETRAIN] Training XGBoost ...")
    model = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.04,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)
    preds = np.maximum(model.predict(X_val), 0)
    metrics = {
        "MAE":  round(mean_absolute_error(y_val, preds), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_val, preds)), 2),
        "MAPE": round(mape(y_val.values, preds), 2),
    }
    print(f"  MAE={metrics['MAE']} | RMSE={metrics['RMSE']} | MAPE={metrics['MAPE']}%")
    return model, metrics


def save_models(lgb_model, xgb_model, lgb_metrics, xgb_metrics, features):
    version    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_dir = RETRAIN_DIR / f"version_{version}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Backup old models
    for fname in ["lightgbm_model.pkl", "xgboost_model.pkl"]:
        src = MODEL_DIR / fname
        if src.exists():
            joblib.dump(joblib.load(src), backup_dir / fname)

    # Save new models
    joblib.dump(lgb_model, MODEL_DIR / "lightgbm_model.pkl")
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.pkl")

    # Update evaluation summary
    summary = {
        "milestone":      "Milestone 4 – Retrained Models",
        "retrain_date":   datetime.utcnow().isoformat(),
        "version":        version,
        "models": {
            "LightGBM": {"model": "LightGBM", "metrics": lgb_metrics},
            "XGBoost":  {"model": "XGBoost",  "metrics": xgb_metrics},
        },
        "best_model":     "LightGBM" if lgb_metrics["MAPE"] <= xgb_metrics["MAPE"] else "XGBoost",
        "recommendation": "LightGBM for production",
        "features_used":  features,
    }
    with open(MODEL_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save retrain log
    log_path = RETRAIN_DIR / "retrain_history.json"
    history  = []
    if log_path.exists():
        with open(log_path) as f:
            history = json.load(f)
    history.append({
        "version":    version,
        "timestamp":  datetime.utcnow().isoformat(),
        "lgb_metrics":lgb_metrics,
        "xgb_metrics":xgb_metrics,
        "backup_dir": str(backup_dir),
    })
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  ✅ Models saved (version: {version})")
    print(f"  ✅ Backup → {backup_dir}")
    return version


def run_retraining(force: bool = False) -> dict:
    print("=" * 55)
    print("   MLOps – Automated Retraining Pipeline")
    print("=" * 55)

    # Check if retraining is needed
    monitor_report = MLOPS_DIR / "monitoring" / "monitoring_report.json"
    if not force and monitor_report.exists():
        with open(monitor_report) as f:
            report = json.load(f)
        if not report.get("retraining_recommended", True):
            print("\n✅ No retraining needed — model performance is stable.")
            return {"status": "skipped", "reason": "performance_stable"}

    print("\n🔄 Starting retraining pipeline ...")

    # Load data
    X, y, dates, features = load_latest_data()

    # Time-based split
    cutoff    = dates.max() - pd.Timedelta(days=90)
    train_mask= dates <= cutoff
    X_train, X_val = X[train_mask],  X[~train_mask]
    y_train, y_val = y[train_mask],  y[~train_mask]
    print(f"  Train: {train_mask.sum()} | Val: {(~train_mask).sum()}")

    # Retrain
    lgb_model, lgb_metrics = retrain_lightgbm(X_train, y_train, X_val, y_val, features)
    xgb_model, xgb_metrics = retrain_xgboost(X_train, y_train, X_val, y_val)

    # Save
    version = save_models(lgb_model, xgb_model, lgb_metrics, xgb_metrics, features)

    result = {
        "status":      "completed",
        "version":     version,
        "lgb_metrics": lgb_metrics,
        "xgb_metrics": xgb_metrics,
        "timestamp":   datetime.utcnow().isoformat(),
    }

    print("\n" + "=" * 55)
    print(f"✅ Retraining Complete! Version: {version}")
    print(f"   LightGBM MAPE: {lgb_metrics['MAPE']}%")
    print(f"   XGBoost  MAPE: {xgb_metrics['MAPE']}%")
    print("=" * 55)
    return result


if __name__ == "__main__":
    run_retraining(force=True)
