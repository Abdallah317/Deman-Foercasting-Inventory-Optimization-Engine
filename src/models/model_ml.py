"""
model_ml.py
-----------
Milestone 2 – XGBoost & LightGBM Demand Forecasting Models
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
warnings.filterwarnings("ignore")

PROCESSED_PATH = Path(__file__).parents[2] / "data" / "processed" / "retail_sales_processed.csv"
REPORT_DIR     = Path(__file__).parents[2] / "reports"
MODEL_DIR      = Path(__file__).parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BG      = "#F8FAFC"
PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def load_features():
    print("[LOAD] Loading processed data ...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])

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

    # Aggregate by date + category for focused forecasting
    agg = df.groupby(["date", "category", "category_enc"]).agg(
        units_sold        = ("units_sold",        "sum"),
        price             = ("price",             "mean"),
        stock_level       = ("stock_level",        "sum"),
        is_promotion      = ("is_promotion",       "max"),
        is_holiday        = ("is_holiday",         "max"),
        is_weekend        = ("is_weekend",         "max"),
        month             = ("month",              "first"),
        week              = ("week",               "first"),
        day_of_week       = ("day_of_week",        "first"),
        quarter           = ("quarter",            "first"),
        year              = ("year",               "first"),
        month_sin         = ("month_sin",          "first"),
        month_cos         = ("month_cos",          "first"),
        dayofweek_sin     = ("dayofweek_sin",      "first"),
        dayofweek_cos     = ("dayofweek_cos",      "first"),
        lag_1             = ("lag_1",              "sum"),
        lag_7             = ("lag_7",              "sum"),
        lag_14            = ("lag_14",             "sum"),
        lag_28            = ("lag_28",             "sum"),
        rolling_mean_7    = ("rolling_mean_7",     "mean"),
        rolling_mean_14   = ("rolling_mean_14",    "mean"),
        rolling_mean_28   = ("rolling_mean_28",    "mean"),
        rolling_std_7     = ("rolling_std_7",      "mean"),
        rolling_std_28    = ("rolling_std_28",     "mean"),
        price_discount    = ("price_discount",     "max"),
        stockout_flag     = ("stockout_flag",      "max"),
        store_type_enc    = ("store_type_enc",     "first"),
    ).reset_index()

    agg = agg.sort_values("date").dropna()

    FEATURES = [f for f in FEATURES if f in agg.columns]
    X = agg[FEATURES]
    y = agg[TARGET]
    dates = agg["date"]

    print(f"  Features: {len(FEATURES)} | Samples: {len(X)}")
    return X, y, dates, FEATURES

def train_test_split_ts(X, y, dates, test_days=90):
    cutoff = dates.max() - pd.Timedelta(days=test_days)
    mask   = dates <= cutoff
    return X[mask], X[~mask], y[mask], y[~mask], dates[~mask]

def train_xgboost(X_train, X_test, y_train, y_test, dates_test):
    print("\n[XGBoost] Training ...")
    model = xgb.XGBRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 6,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = np.maximum(model.predict(X_test), 0)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mp    = mape(y_test.values, preds)

    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mp:.2f}%")

    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")
    return model, preds, {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mp,2)}

def train_lightgbm(X_train, X_test, y_train, y_test, dates_test):
    print("\n[LightGBM] Training ...")
    model = lgb.LGBMRegressor(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        num_leaves       = 63,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        n_jobs           = -1,
        verbose          = -1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])

    preds = np.maximum(model.predict(X_test), 0)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mp    = mape(y_test.values, preds)

    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mp:.2f}%")

    joblib.dump(model, MODEL_DIR / "lightgbm_model.pkl")
    return model, preds, {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mp,2)}

def plot_predictions(y_test, dates_test, xgb_preds, lgb_preds, xgb_m, lgb_m):
    # Aggregate by date for cleaner plot
    results = pd.DataFrame({
        "date":    dates_test.values,
        "actual":  y_test.values,
        "xgboost": xgb_preds,
        "lightgbm":lgb_preds,
    }).groupby("date").sum().reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # XGBoost
    axes[0].plot(results["date"], results["actual"],  color=PALETTE[2], linewidth=2, label="Actual")
    axes[0].plot(results["date"], results["xgboost"], color=PALETTE[0], linewidth=2, linestyle="--", label="XGBoost")
    axes[0].set_title("XGBoost – Demand Forecast vs Actual", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Units Sold")
    axes[0].legend()
    axes[0].text(0.02, 0.92,
                 f"MAE={xgb_m['MAE']} | RMSE={xgb_m['RMSE']} | MAPE={xgb_m['MAPE']}%",
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # LightGBM
    axes[1].plot(results["date"], results["actual"],   color=PALETTE[2], linewidth=2, label="Actual")
    axes[1].plot(results["date"], results["lightgbm"], color=PALETTE[1], linewidth=2, linestyle="--", label="LightGBM")
    axes[1].set_title("LightGBM – Demand Forecast vs Actual", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Units Sold")
    axes[1].legend()
    axes[1].text(0.02, 0.92,
                 f"MAE={lgb_m['MAE']} | RMSE={lgb_m['RMSE']} | MAPE={lgb_m['MAPE']}%",
                 transform=axes[1].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "ml_forecast.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ ml_forecast.png saved")

def plot_feature_importance(xgb_model, lgb_model, features):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # XGBoost importance
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=True).tail(15)
    axes[0].barh(xgb_imp.index, xgb_imp.values, color=PALETTE[0], edgecolor="white")
    axes[0].set_title("XGBoost – Top 15 Feature Importances", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Importance Score")

    # LightGBM importance
    lgb_imp = pd.Series(lgb_model.feature_importances_, index=features).sort_values(ascending=True).tail(15)
    axes[1].barh(lgb_imp.index, lgb_imp.values, color=PALETTE[1], edgecolor="white")
    axes[1].set_title("LightGBM – Top 15 Feature Importances", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Importance Score")

    plt.suptitle("Feature Importance Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ feature_importance.png saved")

def main():
    X, y, dates, features = load_features()
    X_train, X_test, y_train, y_test, dates_test = train_test_split_ts(X, y, dates)

    xgb_model, xgb_preds, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test, dates_test)
    lgb_model, lgb_preds, lgb_metrics = train_lightgbm(X_train, X_test, y_train, y_test, dates_test)

    plot_predictions(y_test, dates_test, xgb_preds, lgb_preds, xgb_metrics, lgb_metrics)
    plot_feature_importance(xgb_model, lgb_model, features)

    results = {
        "XGBoost":  {"model": "XGBoost",  "metrics": xgb_metrics},
        "LightGBM": {"model": "LightGBM", "metrics": lgb_metrics},
    }
    with open(MODEL_DIR / "ml_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ ML Models done! Saved to {MODEL_DIR}")
    return xgb_metrics, lgb_metrics

if __name__ == "__main__":
    main()
