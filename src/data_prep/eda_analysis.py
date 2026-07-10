"""
eda_analysis.py
---------------
Milestone 1 – Exploratory Data Analysis

Generates:
  • Statistical summary
  • Sales distribution plots
  • Time-series decomposition
  • Seasonal / weekday / holiday analysis
  • Category & store-type breakdowns
  • Correlation heatmap
  • Promotion impact analysis
  • Demand vs stock-level scatter
  • Rolling average trend chart
  • All saved as PNG to reports/
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path(__file__).parents[2] / "data" / "processed" / "retail_sales_processed.csv"
REPORT_DIR     = Path(__file__).parents[2] / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette & Style ───────────────────────────────────────────────────────────
PALETTE  = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]
BG       = "#F8FAFC"
GRID_CLR = "#E2E8F0"

def styled_fig(w=12, h=6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return fig, ax

def save(name: str):
    path = REPORT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✓ {name}")

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading processed data …")
df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
print(f"  Shape: {df.shape}")

# Aggregate daily totals for time-series plots
daily = (df.groupby("date")
           .agg(units_sold=("units_sold","sum"),
                revenue=("revenue","sum"),
                is_promotion=("is_promotion","mean"))
           .reset_index())

# ── 1. Statistical summary ────────────────────────────────────────────────────
print("\n[1] Statistical summary …")
summary = df[["units_sold","price","revenue","stock_level"]].describe().round(2)
summary.to_csv(REPORT_DIR / "statistical_summary.csv")
print(summary.to_string())


# ── 2. Sales distribution ─────────────────────────────────────────────────────
print("\n[2] Sales distribution …")
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

axes[0].hist(df["units_sold"].dropna(), bins=60, color=PALETTE[0], edgecolor="white", linewidth=0.3)
axes[0].set_title("Units Sold Distribution", fontweight="bold")
axes[0].set_xlabel("Units Sold")

axes[1].hist(df["revenue"].dropna(), bins=60, color=PALETTE[1], edgecolor="white", linewidth=0.3)
axes[1].set_title("Revenue Distribution", fontweight="bold")
axes[1].set_xlabel("Revenue ($)")

axes[2].hist(df["price"].dropna(), bins=50, color=PALETTE[2], edgecolor="white", linewidth=0.3)
axes[2].set_title("Price Distribution", fontweight="bold")
axes[2].set_xlabel("Price ($)")

plt.suptitle("Sales Metrics – Distribution Overview", fontsize=14, fontweight="bold", y=1.02)
save("01_sales_distributions.png")


# ── 3. Daily sales time-series ────────────────────────────────────────────────
print("[3] Time-series …")
fig, axes = plt.subplots(2, 1, figsize=(16, 8), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

axes[0].plot(daily["date"], daily["units_sold"], color=PALETTE[0], linewidth=0.8, alpha=0.6)
axes[0].plot(daily["date"],
             daily["units_sold"].rolling(30, center=True).mean(),
             color=PALETTE[0], linewidth=2.5, label="30-day MA")
axes[0].set_title("Daily Units Sold (2020–2023)", fontweight="bold")
axes[0].set_ylabel("Units Sold")
axes[0].legend()

axes[1].plot(daily["date"], daily["revenue"]/1e6, color=PALETTE[1], linewidth=0.8, alpha=0.6)
axes[1].plot(daily["date"],
             (daily["revenue"]/1e6).rolling(30, center=True).mean(),
             color=PALETTE[1], linewidth=2.5, label="30-day MA")
axes[1].set_title("Daily Revenue ($M)", fontweight="bold")
axes[1].set_ylabel("Revenue ($M)")
axes[1].legend()

plt.suptitle("4-Year Sales Time Series", fontsize=14, fontweight="bold")
save("02_timeseries.png")


# ── 4. Seasonality – Monthly & Weekly ────────────────────────────────────────
print("[4] Seasonality …")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

monthly = df.groupby("month")["units_sold"].mean()
months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
axes[0].bar(months, monthly.values, color=PALETTE[0], edgecolor="white")
axes[0].set_title("Avg Units Sold by Month", fontweight="bold")
axes[0].set_ylabel("Avg Units Sold")

weekday  = df.groupby("day_of_week")["units_sold"].mean()
days     = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
colors_w = [PALETTE[2] if i >= 5 else PALETTE[0] for i in range(7)]
axes[1].bar(days, weekday.values, color=colors_w, edgecolor="white")
axes[1].set_title("Avg Units Sold by Day of Week", fontweight="bold")
axes[1].set_ylabel("Avg Units Sold")
axes[1].text(5.5, weekday.max()*0.95, "Weekend", color=PALETTE[2], fontsize=9, ha="center")

plt.suptitle("Seasonality Analysis", fontsize=14, fontweight="bold")
save("03_seasonality.png")


# ── 5. Category breakdown ─────────────────────────────────────────────────────
print("[5] Category breakdown …")
cat_stats = df.groupby("category").agg(
    total_units=("units_sold","sum"),
    total_revenue=("revenue","sum"),
    avg_price=("price","mean"),
).sort_values("total_revenue", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

axes[0].barh(cat_stats.index, cat_stats["total_units"]/1e6,
             color=PALETTE[:len(cat_stats)], edgecolor="white")
axes[0].set_title("Total Units Sold by Category (M)", fontweight="bold")
axes[0].set_xlabel("Units Sold (Millions)")

axes[1].barh(cat_stats.index, cat_stats["total_revenue"]/1e9,
             color=PALETTE[:len(cat_stats)], edgecolor="white")
axes[1].set_title("Total Revenue by Category ($B)", fontweight="bold")
axes[1].set_xlabel("Revenue (Billions $)")

plt.suptitle("Category Performance Analysis", fontsize=14, fontweight="bold")
save("04_category_breakdown.png")


# ── 6. Store-type performance ────────────────────────────────────────────────
print("[6] Store-type analysis …")
store_stats = df.groupby("store_type").agg(
    avg_units=("units_sold","mean"),
    avg_revenue=("revenue","mean"),
).sort_values("avg_revenue", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

axes[0].bar(store_stats.index, store_stats["avg_units"],
            color=PALETTE[:len(store_stats)], edgecolor="white")
axes[0].set_title("Avg Units Sold by Store Type", fontweight="bold")
axes[0].set_ylabel("Avg Units Sold")

axes[1].bar(store_stats.index, store_stats["avg_revenue"],
            color=PALETTE[:len(store_stats)], edgecolor="white")
axes[1].set_title("Avg Revenue by Store Type ($)", fontweight="bold")
axes[1].set_ylabel("Avg Revenue ($)")

plt.suptitle("Store Type Performance", fontsize=14, fontweight="bold")
save("05_store_type_performance.png")


# ── 7. Promotion impact ───────────────────────────────────────────────────────
print("[7] Promotion impact …")
promo = df.groupby(["category","is_promotion"])["units_sold"].mean().unstack()
promo.columns = ["No Promo", "Promotion"]
promo["lift_pct"] = ((promo["Promotion"] - promo["No Promo"]) / promo["No Promo"] * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

x   = np.arange(len(promo))
w   = 0.35
axes[0].bar(x - w/2, promo["No Promo"],  w, label="No Promo",  color=PALETTE[0], edgecolor="white")
axes[0].bar(x + w/2, promo["Promotion"], w, label="Promotion", color=PALETTE[1], edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(promo.index, rotation=20, ha="right")
axes[0].set_title("Avg Units Sold: Promo vs No Promo", fontweight="bold")
axes[0].legend()

axes[1].bar(promo.index, promo["lift_pct"],
            color=[PALETTE[1] if v > 0 else PALETTE[2] for v in promo["lift_pct"]],
            edgecolor="white")
axes[1].axhline(0, color="#64748B", linewidth=1)
axes[1].set_title("Promotion Lift % by Category", fontweight="bold")
axes[1].set_ylabel("Lift (%)")
for i, v in enumerate(promo["lift_pct"]):
    axes[1].text(i, v + 0.5, f"{v}%", ha="center", fontsize=9, fontweight="bold")

plt.suptitle("Promotion Impact Analysis", fontsize=14, fontweight="bold")
save("06_promotion_impact.png")


# ── 8. Holiday impact ────────────────────────────────────────────────────────
print("[8] Holiday impact …")
holiday_stats = df.groupby(["category", "is_holiday"])["units_sold"].mean().unstack()
holiday_stats.columns = ["Non-Holiday", "Holiday"]

fig, ax = styled_fig(10, 6)
x = np.arange(len(holiday_stats))
w = 0.35
ax.bar(x - w/2, holiday_stats["Non-Holiday"], w, label="Non-Holiday", color=PALETTE[0], edgecolor="white")
ax.bar(x + w/2, holiday_stats["Holiday"],     w, label="Holiday",     color=PALETTE[3], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(holiday_stats.index, rotation=15, ha="right")
ax.set_title("Holiday vs Non-Holiday Demand by Category", fontweight="bold", fontsize=13)
ax.set_ylabel("Avg Units Sold")
ax.legend()
save("07_holiday_impact.png")


# ── 9. Correlation heatmap ────────────────────────────────────────────────────
print("[9] Correlation heatmap …")
corr_cols = ["units_sold","revenue","price","stock_level",
             "is_promotion","is_holiday","is_weekend",
             "month","day_of_week","lag_7","rolling_mean_7","rolling_mean_28"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
ax.set_title("Feature Correlation Heatmap", fontweight="bold", fontsize=14, pad=15)
save("08_correlation_heatmap.png")


# ── 10. Yearly growth ─────────────────────────────────────────────────────────
print("[10] Yearly growth …")
yearly = df.groupby("year").agg(
    total_units=("units_sold","sum"),
    total_revenue=("revenue","sum"),
)
yearly["yoy_unit_growth"]   = yearly["total_units"].pct_change() * 100
yearly["yoy_rev_growth"]    = yearly["total_revenue"].pct_change() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

axes[0].bar(yearly.index.astype(str), yearly["total_units"]/1e6,
            color=PALETTE[0], edgecolor="white")
axes[0].set_title("Total Units Sold per Year", fontweight="bold")
axes[0].set_ylabel("Units (Millions)")

axes[1].bar(yearly.index.astype(str), yearly["total_revenue"]/1e9,
            color=PALETTE[1], edgecolor="white")
axes[1].set_title("Total Revenue per Year ($B)", fontweight="bold")
axes[1].set_ylabel("Revenue ($B)")

plt.suptitle("Year-over-Year Growth", fontsize=14, fontweight="bold")
save("09_yearly_growth.png")


# ── 11. Top products ─────────────────────────────────────────────────────────
print("[11] Top products …")
top_prods = (df.groupby("product_id")["revenue"].sum()
               .sort_values(ascending=False).head(15))

fig, ax = styled_fig(12, 6)
ax.barh([f"Product {p}" for p in top_prods.index],
        top_prods.values / 1e6,
        color=PALETTE[0], edgecolor="white")
ax.set_title("Top 15 Products by Revenue", fontweight="bold", fontsize=13)
ax.set_xlabel("Revenue ($M)")
ax.invert_yaxis()
save("10_top_products.png")


# ── 12. Missing value heatmap ────────────────────────────────────────────────
print("[12] Missing value summary …")
raw_path = Path(__file__).parents[2] / "data" / "raw" / "retail_sales.csv"
raw_df   = pd.read_csv(raw_path)
miss     = raw_df.isnull().mean() * 100
miss     = miss[miss > 0]

fig, ax = styled_fig(8, 4)
ax.barh(miss.index, miss.values, color=PALETTE[2], edgecolor="white")
ax.set_title("Missing Values (%) – Raw Dataset", fontweight="bold", fontsize=13)
ax.set_xlabel("Missing %")
for i, v in enumerate(miss.values):
    ax.text(v + 0.02, i, f"{v:.2f}%", va="center", fontsize=10)
save("11_missing_values.png")


print("\n All EDA charts saved to reports/")
print(f"   Output directory: {REPORT_DIR.resolve()}")