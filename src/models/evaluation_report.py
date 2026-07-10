"""
evaluation_report.py
--------------------
Milestone 2 – Final Model Evaluation & Comparison Report
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MODEL_DIR  = Path(__file__).parents[2] / "models"
REPORT_DIR = Path(__file__).parents[2] / "reports"
BG         = "#F8FAFC"
PALETTE    = ["#2563EB", "#16A34A", "#DC2626", "#D97706"]

# Load all metrics
with open(MODEL_DIR / "sarima_metrics.json") as f:
    sarima = json.load(f)
with open(MODEL_DIR / "ml_metrics.json") as f:
    ml = json.load(f)
with open(MODEL_DIR / "lstm_metrics.json") as f:
    lstm = json.load(f)

models = {
    "SARIMA":   sarima["metrics"],
    "XGBoost":  ml["XGBoost"]["metrics"],
    "LightGBM": ml["LightGBM"]["metrics"],
    "LSTM":     lstm["metrics"],
}

# ── Model comparison bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
metrics_to_plot = ["MAE", "RMSE", "MAPE"]
titles = ["Mean Absolute Error (MAE) ↓", "Root Mean Squared Error (RMSE) ↓", "Mean Absolute % Error (MAPE %) ↓"]

for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[i]
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    names  = list(models.keys())
    values = [models[m][metric] for m in names]
    colors = PALETTE[:len(names)]
    bars   = ax.bar(names, values, color=colors, edgecolor="white", width=0.6)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel(metric)

    # Highlight best
    best_idx = int(np.argmin(values))
    bars[best_idx].set_edgecolor("#1E293B")
    bars[best_idx].set_linewidth(2.5)
    for j, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(values)*0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold" if j == best_idx else "normal")

plt.suptitle("Forecast Model Comparison – Milestone 2", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ model_comparison.png saved")

# ── Radar chart ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG)
ax.set_facecolor(BG)

categories = ["Low MAE", "Low RMSE", "Low MAPE"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, (name, m) in enumerate(models.items()):
    # Normalize (lower is better → invert)
    max_mae  = max(v["MAE"]  for v in models.values())
    max_rmse = max(v["RMSE"] for v in models.values())
    max_mape = max(v["MAPE"] for v in models.values())
    vals = [
        1 - m["MAE"]  / max_mae,
        1 - m["RMSE"] / max_rmse,
        1 - m["MAPE"] / max_mape,
    ]
    vals += vals[:1]
    ax.plot(angles, vals, color=PALETTE[i], linewidth=2, label=name)
    ax.fill(angles, vals, color=PALETTE[i], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
ax.set_title("Model Performance Radar\n(Higher = Better)", fontweight="bold", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(REPORT_DIR / "radar_comparison.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✓ radar_comparison.png saved")

# ── Text report ───────────────────────────────────────────────────────────────
report_lines = [
    "=" * 60,
    "   MILESTONE 2 – FORECAST MODEL EVALUATION REPORT",
    "=" * 60,
    "",
    "Models Trained:",
    "  1. SARIMA(1,1,1)(1,1,1,7) – Statistical time-series",
    "  2. XGBoost – Gradient boosted trees",
    "  3. LightGBM – Light gradient boosting",
    "  4. LSTM(128-64-32) – Deep learning",
    "",
    "Test Set: Last 90 days (Oct–Dec 2023)",
    "",
    "-" * 60,
    f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}",
    "-" * 60,
]
for name, m in models.items():
    report_lines.append(f"{name:<12} {m['MAE']:>10.2f} {m['RMSE']:>10.2f} {m['MAPE']:>9.2f}%")

best_mape_model = min(models, key=lambda x: models[x]["MAPE"])
report_lines += [
    "-" * 60,
    "",
    f" Best Model by MAPE: {best_mape_model} ({models[best_mape_model]['MAPE']}%)",
    "",
    "Observations:",
    "  • ML models (XGBoost/LightGBM) outperform SARIMA on MAPE",
    "  • LSTM achieves competitive MAPE with deep temporal learning",
    "  • SARIMA best for interpretability & explainability",
    "  • LightGBM recommended for production (speed + accuracy)",
    "",
    "Recommendation: Deploy LightGBM as primary model.",
    "Use SARIMA as baseline and LSTM for ensemble.",
    "=" * 60,
]

report_text = "\n".join(report_lines)
print("\n" + report_text)
with open(REPORT_DIR / "evaluation_report.txt", "w") as f:
    f.write(report_text)
print(f"\n✅ Full evaluation report → {REPORT_DIR / 'evaluation_report.txt'}")

# Save JSON summary
summary = {
    "milestone": "Milestone 2 – Forecast Model Development",
    "models": models,
    "best_model": best_mape_model,
    "recommendation": "LightGBM for production, SARIMA as baseline",
}
with open("Demand_Forecasting_Engine/reports/evaluation_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)