"""
model_sarima.py
---------------
Milestone 2 – SARIMA Time Series Forecasting Model
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

PROCESSED_PATH = Path(__file__).parents[2] / "data" / "processed" / "retail_sales_processed.csv"
REPORT_DIR     = Path(__file__).parents[2] / "reports"
MODEL_DIR      = Path(__file__).parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BG       = "#F8FAFC"
PALETTE  = ["#2563EB", "#DC2626", "#16A34A"]

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def check_stationarity(series, name="series"):
    result = adfuller(series.dropna())
    print(f"  ADF Statistic ({name}): {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Stationary: {'✅ Yes' if result[1] < 0.05 else '❌ No (needs differencing)'}")
    return result[1] < 0.05

def load_and_aggregate():
    print("[LOAD] Loading processed data ...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    # Aggregate to daily total units sold (all stores, all products)
    daily = (df.groupby("date")["units_sold"]
               .sum()
               .reset_index()
               .set_index("date")
               .asfreq("D"))
    daily["units_sold"] = daily["units_sold"].ffill()
    print(f"  Daily series shape: {daily.shape}")
    return daily

def train_sarima(series):
    print("\n[SARIMA] Checking stationarity ...")
    check_stationarity(series["units_sold"], "units_sold")

    # Train/test split — last 90 days as test
    train = series.iloc[:-90]
    test  = series.iloc[-90:]
    print(f"  Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days)")
    print(f"  Test : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} days)")

    # SARIMA(1,1,1)(1,1,1,7) — weekly seasonality
    print("\n[SARIMA] Fitting SARIMA(1,1,1)(1,1,1,7) ...")
    model = SARIMAX(
        train["units_sold"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    print(f"  AIC: {fitted.aic:.2f}  |  BIC: {fitted.bic:.2f}")

    # Forecast
    forecast = fitted.forecast(steps=90)
    forecast = np.maximum(forecast, 0)

    # Metrics
    mae  = mean_absolute_error(test["units_sold"], forecast)
    rmse = np.sqrt(mean_squared_error(test["units_sold"], forecast))
    mp   = mape(test["units_sold"].values, forecast.values)

    print(f"\n[SARIMA] Evaluation on 90-day test set:")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAPE : {mp:.2f}%")

    return fitted, train, test, forecast, {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mp,2)}

def plot_sarima(train, test, forecast, metrics):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=BG)

    # Full series + forecast
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    ax.plot(train.index, train["units_sold"], color=PALETTE[0], linewidth=1, alpha=0.7, label="Train")
    ax.plot(test.index,  test["units_sold"],  color=PALETTE[2], linewidth=1.5, label="Actual (Test)")
    ax.plot(test.index,  forecast,            color=PALETTE[1], linewidth=2, linestyle="--", label="SARIMA Forecast")
    ax.axvline(test.index[0], color="#94A3B8", linestyle=":", linewidth=1.5)
    ax.set_title("SARIMA Forecast – Daily Units Sold", fontweight="bold", fontsize=13)
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.text(0.02, 0.95,
            f"MAE={metrics['MAE']} | RMSE={metrics['RMSE']} | MAPE={metrics['MAPE']}%",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Residuals
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.yaxis.grid(True, color="#E2E8F0", linewidth=0.8)
    ax2.set_axisbelow(True)
    for sp in ["top","right"]: ax2.spines[sp].set_visible(False)

    residuals = test["units_sold"].values - forecast.values
    ax2.bar(test.index, residuals, color=PALETTE[0], alpha=0.6, width=0.8)
    ax2.axhline(0, color=PALETTE[1], linewidth=1.5)
    ax2.set_title("Forecast Residuals (Actual - Predicted)", fontweight="bold", fontsize=13)
    ax2.set_ylabel("Residual")

    plt.tight_layout()
    path = REPORT_DIR / "sarima_forecast.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ sarima_forecast.png saved")

def main():
    daily = load_and_aggregate()
    fitted, train, test, forecast, metrics = train_sarima(daily)
    plot_sarima(train, test, forecast, metrics)

    # Save metrics
    results = {"model": "SARIMA(1,1,1)(1,1,1,7)", "metrics": metrics}
    with open(MODEL_DIR / "sarima_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ SARIMA done! Metrics → {MODEL_DIR / 'sarima_metrics.json'}")
    return metrics

if __name__ == "__main__":
    main()
