"""
main.py
-------
Milestone 3 – FastAPI Demand Forecasting Service

Endpoints:
  GET  /                    → Health check
  GET  /health              → Service status
  POST /predict             → Single prediction
  POST /predict/batch       → Batch predictions
  GET  /forecast/{days}     → Future forecast
  GET  /inventory/recommend → Inventory recommendations
  GET  /models/info         → Model metadata
  GET  /metrics             → Model performance metrics
"""

import json
import os
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path as FilePath
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = FilePath(__file__).parents[2]
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Demand Forecasting & Inventory Optimization API",
    description = "ML-powered demand forecasting engine with inventory recommendations.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Load models at startup ────────────────────────────────────────────────────
models_loaded = {}

@app.on_event("startup")
async def load_models():
    global models_loaded
    try:
        models_loaded["lightgbm"] = joblib.load(MODEL_DIR / "lightgbm_model.pkl")
        print("✅ LightGBM model loaded")
    except Exception as e:
        print(f"⚠️  LightGBM not found: {e}")

    try:
        models_loaded["xgboost"] = joblib.load(MODEL_DIR / "xgboost_model.pkl")
        print("✅ XGBoost model loaded")
    except Exception as e:
        print(f"⚠️  XGBoost not found: {e}")

    try:
        with open(MODEL_DIR / "evaluation_summary.json") as f:
            models_loaded["metrics"] = json.load(f)
        print("✅ Metrics loaded")
    except Exception as e:
        print(f"⚠️  Metrics not found: {e}")
        models_loaded["metrics"] = {}


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    date:             str          = Field(...,  example="2024-01-15")
    store_id:         int          = Field(...,  example=1, ge=1, le=10)
    product_id:       int          = Field(...,  example=5, ge=1, le=50)
    category:         str          = Field(...,  example="Electronics")
    store_type:       str          = Field(...,  example="Urban")
    price:            float        = Field(...,  example=149.99, gt=0)
    stock_level:      int          = Field(50,   example=50,  ge=0)
    is_promotion:     int          = Field(0,    example=0,   ge=0, le=1)
    is_holiday:       int          = Field(0,    example=0,   ge=0, le=1)
    lag_1:            Optional[float] = Field(None, example=20.0)
    lag_7:            Optional[float] = Field(None, example=18.0)
    lag_14:           Optional[float] = Field(None, example=19.0)
    lag_28:           Optional[float] = Field(None, example=17.0)
    rolling_mean_7:   Optional[float] = Field(None, example=19.5)
    rolling_mean_14:  Optional[float] = Field(None, example=18.8)
    rolling_mean_28:  Optional[float] = Field(None, example=18.2)
    rolling_std_7:    Optional[float] = Field(None, example=3.2)
    rolling_std_28:   Optional[float] = Field(None, example=3.5)
    model:            str          = Field("lightgbm", example="lightgbm")

class PredictResponse(BaseModel):
    predicted_units:      float
    prediction_date:      str
    store_id:             int
    product_id:           int
    category:             str
    model_used:           str
    confidence_interval:  dict
    inventory_status:     str
    reorder_recommended:  bool

class BatchPredictRequest(BaseModel):
    predictions: List[PredictRequest]

class ForecastResponse(BaseModel):
    forecast_horizon_days: int
    start_date:            str
    end_date:              str
    daily_forecasts:       List[dict]
    total_expected_units:  float
    peak_demand_date:      str

class InventoryRecommendation(BaseModel):
    product_id:         int
    category:           str
    current_stock:      int
    predicted_demand_7d: float
    predicted_demand_30d: float
    reorder_point:      float
    recommended_order:  float
    status:             str
    urgency:            str


# ── Helper: build feature vector ──────────────────────────────────────────────

CATEGORY_MAP   = {"Electronics": 0, "Clothing": 1, "Food": 2, "Home": 3, "Sports": 4}
STORE_TYPE_MAP = {"Urban": 3, "Suburban": 2, "Rural": 0, "Outlet": 1}

def build_features(req: PredictRequest) -> np.ndarray:
    dt = pd.Timestamp(req.date)
    month       = dt.month
    week        = dt.isocalendar().week
    day_of_week = dt.dayofweek
    quarter     = dt.quarter
    is_weekend  = int(day_of_week >= 5)
    year        = dt.year

    month_sin       = np.sin(2 * np.pi * month / 12)
    month_cos       = np.cos(2 * np.pi * month / 12)
    dayofweek_sin   = np.sin(2 * np.pi * day_of_week / 7)
    dayofweek_cos   = np.cos(2 * np.pi * day_of_week / 7)

    # Default lags/rolling if not provided
    lag1  = req.lag_1          if req.lag_1          is not None else 20.0
    lag7  = req.lag_7          if req.lag_7          is not None else 19.0
    lag14 = req.lag_14         if req.lag_14         is not None else 18.5
    lag28 = req.lag_28         if req.lag_28         is not None else 18.0
    rm7   = req.rolling_mean_7  if req.rolling_mean_7  is not None else 19.5
    rm14  = req.rolling_mean_14 if req.rolling_mean_14 is not None else 19.0
    rm28  = req.rolling_mean_28 if req.rolling_mean_28 is not None else 18.5
    rs7   = req.rolling_std_7   if req.rolling_std_7   is not None else 3.2
    rs28  = req.rolling_std_28  if req.rolling_std_28  is not None else 3.5

    price_discount = int(req.price < 80)
    stockout_flag  = int(req.stock_level == 0)
    cat_enc        = CATEGORY_MAP.get(req.category, 0)
    store_type_enc = STORE_TYPE_MAP.get(req.store_type, 0)

    features = np.array([[
        month, week, day_of_week, quarter, is_weekend,
        req.is_promotion, req.is_holiday, req.price, req.stock_level,
        month_sin, month_cos, dayofweek_sin, dayofweek_cos,
        lag1, lag7, lag14, lag28,
        rm7, rm14, rm28, rs7, rs28,
        store_type_enc, cat_enc,
        price_discount, stockout_flag, year,
    ]])
    return features


def get_inventory_status(stock: int, predicted_7d: float) -> tuple:
    days_cover = stock / max(predicted_7d / 7, 0.1)
    if days_cover < 3:
        return "CRITICAL", "🔴 Urgent"
    elif days_cover < 7:
        return "LOW", "🟡 Soon"
    elif days_cover < 14:
        return "ADEQUATE", "🟢 Normal"
    else:
        return "EXCESS", "⚪ No Action"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service":     "Demand Forecasting & Inventory Optimization API",
        "version":     "1.0.0",
        "status":      "running",
        "timestamp":   datetime.utcnow().isoformat(),
        "endpoints":   ["/docs", "/health", "/predict", "/predict/batch",
                        "/forecast/{days}", "/inventory/recommend", "/metrics"],
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status":        "healthy",
        "models_loaded": list(models_loaded.keys()),
        "timestamp":     datetime.utcnow().isoformat(),
        "uptime":        "active",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
async def predict_demand(req: PredictRequest):
    model_key = req.model.lower()
    if model_key not in models_loaded:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not loaded.")

    model    = models_loaded[model_key]
    features = build_features(req)

    try:
        pred = float(max(model.predict(features)[0], 0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Confidence interval ±15%
    ci_lower = round(pred * 0.85, 2)
    ci_upper = round(pred * 1.15, 2)

    inv_status, _ = get_inventory_status(req.stock_level, pred * 7)
    reorder = req.stock_level < pred * 3

    return PredictResponse(
        predicted_units     = round(pred, 2),
        prediction_date     = req.date,
        store_id            = req.store_id,
        product_id          = req.product_id,
        category            = req.category,
        model_used          = model_key,
        confidence_interval = {"lower": ci_lower, "upper": ci_upper, "confidence": "85%"},
        inventory_status    = inv_status,
        reorder_recommended = reorder,
    )


@app.post("/predict/batch", tags=["Forecasting"])
async def predict_batch(req: BatchPredictRequest):
    if len(req.predictions) > 100:
        raise HTTPException(status_code=400, detail="Max 100 predictions per batch.")

    results = []
    for item in req.predictions:
        model_key = item.model.lower()
        if model_key not in models_loaded:
            results.append({"error": f"Model '{model_key}' not found", "date": item.date})
            continue
        features = build_features(item)
        pred     = float(max(models_loaded[model_key].predict(features)[0], 0))
        results.append({
            "date":            item.date,
            "store_id":        item.store_id,
            "product_id":      item.product_id,
            "predicted_units": round(pred, 2),
            "model_used":      model_key,
        })

    return {"total": len(results), "predictions": results}


@app.get("/forecast/{days}", response_model=ForecastResponse, tags=["Forecasting"])
async def future_forecast(
    days:       int = Path(..., ge=1, le=365, description="Number of days to forecast"),
    store_id:   int = Query(1,   ge=1, le=10),
    product_id: int = Query(1,   ge=1, le=50),
    category:   str = Query("Electronics"),
    store_type: str = Query("Urban"),
    price:      float = Query(149.99, gt=0),
):
    if "lightgbm" not in models_loaded:
        raise HTTPException(status_code=503, detail="LightGBM model not loaded.")

    start_date  = datetime.utcnow().date()
    daily_preds = []
    lag_buffer  = [20.0] * 28   # rolling buffer for lag features

    for i in range(days):
        pred_date = start_date + timedelta(days=i)
        req = PredictRequest(
            date         = str(pred_date),
            store_id     = store_id,
            product_id   = product_id,
            category     = category,
            store_type   = store_type,
            price        = price,
            is_promotion = int(i % 14 == 0),   # promo every 2 weeks
            is_holiday   = 0,
            lag_1        = lag_buffer[-1],
            lag_7        = lag_buffer[-7],
            lag_14       = lag_buffer[-14],
            lag_28       = lag_buffer[-28] if len(lag_buffer) >= 28 else lag_buffer[0],
            rolling_mean_7  = float(np.mean(lag_buffer[-7:])),
            rolling_mean_14 = float(np.mean(lag_buffer[-14:])),
            rolling_mean_28 = float(np.mean(lag_buffer[-28:])),
            rolling_std_7   = float(np.std(lag_buffer[-7:])),
            rolling_std_28  = float(np.std(lag_buffer[-28:])),
        )
        features = build_features(req)
        pred     = float(max(models_loaded["lightgbm"].predict(features)[0], 0))
        lag_buffer.append(pred)
        if len(lag_buffer) > 60:
            lag_buffer.pop(0)

        daily_preds.append({
            "date":            str(pred_date),
            "predicted_units": round(pred, 2),
            "is_weekend":      int(pred_date.weekday() >= 5),
            "is_promotion":    int(i % 14 == 0),
        })

    units_list   = [d["predicted_units"] for d in daily_preds]
    peak_idx     = int(np.argmax(units_list))

    return ForecastResponse(
        forecast_horizon_days = days,
        start_date            = str(start_date),
        end_date              = str(start_date + timedelta(days=days - 1)),
        daily_forecasts       = daily_preds,
        total_expected_units  = round(sum(units_list), 2),
        peak_demand_date      = daily_preds[peak_idx]["date"],
    )


@app.get("/inventory/recommend", tags=["Inventory"])
async def inventory_recommendations(
    store_id:    int   = Query(1, ge=1, le=10),
    store_type:  str   = Query("Urban"),
    safety_days: int   = Query(7, ge=1, le=30, description="Safety stock days"),
):
    if "lightgbm" not in models_loaded:
        raise HTTPException(status_code=503, detail="LightGBM model not loaded.")

    categories = ["Electronics", "Clothing", "Food", "Home", "Sports"]
    today      = datetime.utcnow().date()
    recs       = []

    for pid in range(1, 11):   # Top 10 products
        cat          = categories[(pid - 1) % len(categories)]
        current_stock = int(np.random.randint(10, 200))
        price         = float(np.random.uniform(20, 200))

        # Predict 7-day demand
        week_preds = []
        for d in range(7):
            req = PredictRequest(
                date       = str(today + timedelta(days=d)),
                store_id   = store_id,
                product_id = pid,
                category   = cat,
                store_type = store_type,
                price      = price,
            )
            feat = build_features(req)
            pred = float(max(models_loaded["lightgbm"].predict(feat)[0], 0))
            week_preds.append(pred)

        pred_7d  = round(sum(week_preds), 2)
        pred_30d = round(pred_7d * (30 / 7), 2)
        daily_avg = pred_7d / 7

        reorder_point      = round(daily_avg * safety_days, 2)
        recommended_order  = round(max(pred_30d - current_stock, 0) + reorder_point, 2)
        status, urgency    = get_inventory_status(current_stock, pred_7d)

        recs.append({
            "product_id":          pid,
            "category":            cat,
            "current_stock":       current_stock,
            "predicted_demand_7d": pred_7d,
            "predicted_demand_30d":pred_30d,
            "reorder_point":       reorder_point,
            "recommended_order":   recommended_order,
            "status":              status,
            "urgency":             urgency,
        })

    recs_sorted = sorted(recs, key=lambda x: {"CRITICAL": 0, "LOW": 1, "ADEQUATE": 2, "EXCESS": 3}[x["status"]])

    return {
        "store_id":        store_id,
        "store_type":      store_type,
        "generated_at":    datetime.utcnow().isoformat(),
        "safety_days":     safety_days,
        "total_products":  len(recs_sorted),
        "critical_count":  sum(1 for r in recs_sorted if r["status"] == "CRITICAL"),
        "low_count":       sum(1 for r in recs_sorted if r["status"] == "LOW"),
        "recommendations": recs_sorted,
    }


@app.get("/models/info", tags=["Models"])
async def models_info():
    return {
        "loaded_models": list(models_loaded.keys()),
        "available_models": {
            "lightgbm": {
                "type":        "LightGBM Gradient Boosting",
                "features":    27,
                "description": "Primary production model — best MAPE",
                "recommended": True,
            },
            "xgboost": {
                "type":        "XGBoost Gradient Boosting",
                "features":    27,
                "description": "Secondary model — slightly higher MAPE",
                "recommended": False,
            },
        },
        "feature_list": [
            "month", "week", "day_of_week", "quarter", "is_weekend",
            "is_promotion", "is_holiday", "price", "stock_level",
            "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos",
            "lag_1", "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
            "rolling_std_7", "rolling_std_28",
            "store_type_enc", "category_enc", "price_discount", "stockout_flag", "year",
        ],
    }


@app.get("/metrics", tags=["Models"])
async def model_metrics():
    if "metrics" not in models_loaded:
        raise HTTPException(status_code=404, detail="Metrics not found.")
    return models_loaded["metrics"]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
