"""
model_lstm.py
-------------
Milestone 2 – LSTM Deep Learning Demand Forecasting Model
"""

import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
os_environ_set = __import__("os").environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
warnings.filterwarnings("ignore")

PROCESSED_PATH = Path(__file__).parents[2] / "data" / "processed" / "retail_sales_processed.csv"
REPORT_DIR     = Path(__file__).parents[2] / "reports"
MODEL_DIR      = Path(__file__).parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BG      = "#F8FAFC"
PALETTE = ["#2563EB", "#DC2626", "#16A34A"]

LOOKBACK = 30   # use 30 days history to predict next day
EPOCHS   = 50
BATCH    = 64

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def load_daily_series():
    print("[LOAD] Aggregating daily series ...")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    daily = (df.groupby("date").agg(
        units_sold    = ("units_sold",     "sum"),
        is_promotion  = ("is_promotion",   "mean"),
        is_holiday    = ("is_holiday",     "max"),
        is_weekend    = ("is_weekend",     "max"),
        avg_price     = ("price",          "mean"),
        month_sin     = ("month_sin",      "first"),
        month_cos     = ("month_cos",      "first"),
        dayofweek_sin = ("dayofweek_sin",  "first"),
        dayofweek_cos = ("dayofweek_cos",  "first"),
    ).reset_index().sort_values("date"))
    print(f"  Daily series: {daily.shape}")
    return daily

def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])  # units_sold is column 0
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="huber",
                  metrics=["mae"])
    return model

def main():
    daily = load_daily_series()

    FEATURE_COLS = ["units_sold", "is_promotion", "is_holiday", "is_weekend",
                    "avg_price", "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos"]

    data = daily[FEATURE_COLS].values.astype(float)

    # Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, MODEL_DIR / "lstm_scaler.pkl")

    # Sequences
    X, y = create_sequences(data_scaled, LOOKBACK)
    print(f"  Sequences: X={X.shape}, y={y.shape}")

    # Train/test split — last 90 days
    split = len(X) - 90
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = daily["date"].values[LOOKBACK + split:]

    print(f"\n[LSTM] Building model ...")
    model = build_lstm((LOOKBACK, len(FEATURE_COLS)))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0),
    ]

    print(f"\n[LSTM] Training for up to {EPOCHS} epochs ...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=callbacks,
        verbose=1,
    )

    # Predict & inverse scale
    preds_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform: reconstruct full feature array, replace col 0, then extract
    dummy = np.zeros((len(preds_scaled), len(FEATURE_COLS)))
    dummy[:, 0] = preds_scaled
    preds = scaler.inverse_transform(dummy)[:, 0]
    preds = np.maximum(preds, 0)

    dummy_true = np.zeros((len(y_test), len(FEATURE_COLS)))
    dummy_true[:, 0] = y_test
    actuals = scaler.inverse_transform(dummy_true)[:, 0]

    mae  = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mp   = mape(actuals, preds)

    print(f"\n[LSTM] Evaluation:")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mp:.2f}%")

    model.save(MODEL_DIR / "lstm_model.keras")
    print(f"  ✓ Model saved → {MODEL_DIR / 'lstm_model.keras'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=BG)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # Forecast vs actual
    axes[0].plot(dates_test, actuals, color=PALETTE[1], linewidth=2,   label="Actual")
    axes[0].plot(dates_test, preds,   color=PALETTE[0], linewidth=2,   label="LSTM Forecast", linestyle="--")
    axes[0].set_title("LSTM – Demand Forecast vs Actual", fontweight="bold", fontsize=13)
    axes[0].set_ylabel("Units Sold")
    axes[0].legend()
    axes[0].text(0.02, 0.92,
                 f"MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mp:.2f}%",
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Training loss
    axes[1].plot(history.history["loss"],     color=PALETTE[0], linewidth=2, label="Train Loss")
    axes[1].plot(history.history["val_loss"], color=PALETTE[1], linewidth=2, label="Val Loss",   linestyle="--")
    axes[1].set_title("LSTM Training History", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Huber Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "lstm_forecast.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ lstm_forecast.png saved")

    metrics = {"MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE": round(mp,2)}
    with open(MODEL_DIR / "lstm_metrics.json", "w") as f:
        json.dump({"model": "LSTM(128-64-32)", "metrics": metrics}, f, indent=2)

    print(f"\n✅ LSTM done!")
    return metrics

if __name__ == "__main__":
    main()
