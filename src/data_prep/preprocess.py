"""
preprocess.py
-------------
Milestone 1 – Data Preprocessing Pipeline

Steps:
  1. Load raw data
  2. Handle missing values (imputation strategies per column)
  3. Detect and cap outliers (IQR method)
  4. Feature engineering (lag, rolling, date features)
  5. Encode categoricals
  6. Validate schema & dtypes
  7. Save processed dataset + preprocessing report
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RAW_PATH       = Path(__file__).parents[2] / "data" / "raw"  / "retail_sales.csv"
PROCESSED_PATH = Path(__file__).parents[2] / "data" / "processed" / "retail_sales_processed.csv"
REPORT_PATH    = Path(__file__).parents[2] / "reports" / "preprocessing_report.json"


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[LOAD]  Shape: {df.shape}")
    return df


# ── 2. Missing value handling ─────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    missing_before = df.isnull().sum().to_dict()
    report["missing_before"] = {k: int(v) for k, v in missing_before.items() if v > 0}

    # units_sold: median per (store_id, product_id, month)
    df["units_sold"] = df.groupby(
        ["store_id", "product_id", df["date"].dt.month]
    )["units_sold"].transform(lambda x: x.fillna(x.median()))

    # price: forward-fill within product (price rarely changes drastically)
    df = df.sort_values(["product_id", "date"])
    df["price"] = df.groupby("product_id")["price"].transform(
        lambda x: x.ffill().bfill()
    )

    # stock_level: fill 0 (out-of-stock interpretation)
    df["stock_level"] = df["stock_level"].fillna(0)

    # Any remaining nulls → median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    missing_after = df.isnull().sum().to_dict()
    report["missing_after"] = {k: int(v) for k, v in missing_after.items() if v > 0}
    print(f"[MISSING]  Before={sum(missing_before.values())}  After={sum(missing_after.values())}")
    return df


# ── 3. Outlier detection & capping ───────────────────────────────────────────

def handle_outliers(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    outlier_stats = {}
    for col in ["units_sold", "revenue"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        df[col] = df[col].clip(lower, upper)
        outlier_stats[col] = {"lower_cap": round(lower, 2),
                               "upper_cap": round(upper, 2),
                               "n_capped":  n_out}
    report["outliers"] = outlier_stats
    print(f"[OUTLIERS]  units_sold capped={outlier_stats['units_sold']['n_capped']}, "
          f"revenue capped={outlier_stats['revenue']['n_capped']}")
    return df


# ── 4. Feature engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

    # Date features
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["quarter"]     = df["date"].dt.quarter
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    # Cyclical encoding for month & day_of_week
    df["month_sin"]       = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]       = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dayofweek_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag features (per store-product)
    grp = df.groupby(["store_id", "product_id"])["units_sold"]
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling statistics
    for window in [7, 14, 28]:
        df[f"rolling_mean_{window}"] = (
            grp.shift(1).transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            grp.shift(1).transform(lambda x: x.rolling(window, min_periods=1).std())
        )

    # Price discount flag
    df["price_discount"] = (
        df.groupby("product_id")["price"]
        .transform(lambda x: (x < x.median()).astype(int))
    )

    # Revenue per unit
    df["revenue_per_unit"] = (df["revenue"] / df["units_sold"].replace(0, np.nan)).fillna(0)

    # Stock-out flag
    df["stockout_flag"] = (df["stock_level"] == 0).astype(int)

    # Days since last promotion per product-store
    df["days_since_promo"] = (
        df.groupby(["store_id", "product_id"])
        .apply(lambda g: g["is_promotion"]
               .replace(0, np.nan)
               .ffill()
               .isna()
               .cumsum()
               .where(g["is_promotion"] == 0, 0))
        .reset_index(drop=True)
    )

    print(f"[FEATURES]  Total columns: {df.shape[1]}")
    return df


# ── 5. Encode categoricals ────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["store_type_enc"] = pd.Categorical(df["store_type"]).codes
    df["category_enc"]   = pd.Categorical(df["category"]).codes
    return df


# ── 6. Schema validation ──────────────────────────────────────────────────────

def validate(df: pd.DataFrame, report: dict) -> None:
    assert df["units_sold"].isna().sum() == 0,  "NaN in units_sold!"
    assert df["price"].isna().sum() == 0,       "NaN in price!"
    assert (df["units_sold"] >= 0).all(),        "Negative units_sold!"
    assert (df["price"] > 0).all(),              "Non-positive price!"
    report["final_shape"]   = list(df.shape)
    report["final_columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    print(f"[VALIDATE]  ✅ All checks passed. Final shape: {df.shape}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    report = {}

    df = load_data(RAW_PATH)
    report["raw_shape"] = list(df.shape)

    df = handle_missing(df, report)
    df = handle_outliers(df, report)
    df = engineer_features(df)
    df = encode_categoricals(df)
    validate(df, report)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[SAVE]  Processed data → {PROCESSED_PATH}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVE]  Preprocessing report → {REPORT_PATH}")


if __name__ == "__main__":
    main()