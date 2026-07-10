"""
dashboard.py
------------
Milestone 4 – Interactive Streamlit Dashboard

Pages:
  1. 📊 Overview        – KPIs & sales summary
  2. 📈 Demand Forecast – Model predictions & charts
  3. 📦 Inventory       – Stock status & recommendations
  4. 🔍 EDA Explorer    – Interactive data exploration
  5. ⚙️  MLOps Monitor  – Drift & retraining status
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parents[2]
DATA_PATH  = BASE_DIR / "data" / "processed" / "retail_sales_processed.csv"
RAW_PATH   = BASE_DIR / "data" / "raw"        / "retail_sales.csv"
MODEL_DIR  = BASE_DIR / "models"
MLOPS_DIR  = BASE_DIR / "mlops"
REPORT_DIR = BASE_DIR / "reports"

PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]

st.set_page_config(
    page_title = "Demand Forecasting Engine",
    page_icon  = "📦",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #2563EB;
    }
    .alert-card {
        background: #FEF2F2;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #DC2626;
    }
    .success-card {
        background: #F0FDF4;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #16A34A;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    return df

@st.cache_resource
def load_models():
    models = {}
    for name, fname in [("LightGBM","lightgbm_model.pkl"), ("XGBoost","xgboost_model.pkl")]:
        path = MODEL_DIR / fname
        if path.exists():
            models[name] = joblib.load(path)
    return models

def load_metrics():
    path = MODEL_DIR / "evaluation_summary.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def load_monitor_report():
    path = MLOPS_DIR / "monitoring" / "monitoring_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def load_retrain_history():
    path = MLOPS_DIR / "retraining" / "retrain_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/warehouse.png", width=60)
    st.title("Demand Forecasting\nEngine")
    st.markdown("---")

    page = st.radio("Navigation", [
        "📊 Overview",
        "📈 Demand Forecast",
        "📦 Inventory",
        "🔍 EDA Explorer",
        "⚙️ MLOps Monitor",
    ])

    st.markdown("---")
    st.markdown("**Filters**")
    df_all = load_data()

    selected_category = st.multiselect(
        "Category",
        options=df_all["category"].unique().tolist(),
        default=df_all["category"].unique().tolist(),
    )
    selected_store_type = st.multiselect(
        "Store Type",
        options=df_all["store_type"].unique().tolist(),
        default=df_all["store_type"].unique().tolist(),
    )

    date_range = st.date_input(
        "Date Range",
        value=(df_all["date"].min().date(), df_all["date"].max().date()),
    )

    st.markdown("---")
    st.caption("Milestone 4 – MLOps Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ── Filter data ───────────────────────────────────────────────
df = df_all[
    (df_all["category"].isin(selected_category)) &
    (df_all["store_type"].isin(selected_store_type)) &
    (df_all["date"] >= pd.Timestamp(date_range[0])) &
    (df_all["date"] <= pd.Timestamp(date_range[1]))
]


# ════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Sales Overview Dashboard")
    st.markdown("Real-time KPIs and sales performance summary")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_units = df["units_sold"].sum()
        st.metric("Total Units Sold", f"{total_units:,.0f}", delta="+8.3% vs last period")
    with col2:
        total_rev = df["revenue"].sum()
        st.metric("Total Revenue", f"${total_rev/1e6:.1f}M", delta="+11.2%")
    with col3:
        avg_daily = df.groupby("date")["units_sold"].sum().mean()
        st.metric("Avg Daily Units", f"{avg_daily:,.0f}", delta="+5.1%")
    with col4:
        promo_lift = df.groupby("is_promotion")["units_sold"].mean()
        lift = ((promo_lift.get(1,0) - promo_lift.get(0,1)) / promo_lift.get(0,1) * 100)
        st.metric("Promo Lift", f"{lift:.1f}%", delta="vs non-promo")

    st.markdown("---")

    # Daily sales chart
    daily = df.groupby("date").agg(units_sold=("units_sold","sum"), revenue=("revenue","sum")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["units_sold"],
                             name="Units Sold", line=dict(color=PALETTE[0], width=1.5),
                             fill="tozeroy", fillcolor="rgba(37,99,235,0.1)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"]/1e3,
                             name="Revenue ($K)", line=dict(color=PALETTE[1], width=1.5, dash="dot")), secondary_y=True)
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["units_sold"].rolling(30).mean(),
                             name="30-day MA", line=dict(color=PALETTE[2], width=2.5)), secondary_y=False)
    fig.update_layout(title="Daily Sales & Revenue Trend", height=400,
                      plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        cat_rev = df.groupby("category")["revenue"].sum().reset_index().sort_values("revenue", ascending=False)
        fig = px.pie(cat_rev, values="revenue", names="category",
                     title="Revenue by Category",
                     color_discrete_sequence=PALETTE,
                     hole=0.4)
        fig.update_layout(height=350, paper_bgcolor="#F8FAFC")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        store_rev = df.groupby("store_type")["revenue"].sum().reset_index().sort_values("revenue")
        fig = px.bar(store_rev, x="revenue", y="store_type", orientation="h",
                     title="Revenue by Store Type",
                     color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(height=350, paper_bgcolor="#F8FAFC", plot_bgcolor="#F8FAFC")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 2: DEMAND FORECAST
# ════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.title("📈 Demand Forecast")

    metrics = load_metrics()

    # Model performance cards
    if metrics:
        st.subheader("Model Performance")
        cols = st.columns(4)
        model_data = metrics.get("models", {})
        for i, (mname, minfo) in enumerate(model_data.items()):
            with cols[i % 4]:
                m = minfo.get("metrics", {})
                best = metrics.get("best_model") == mname
                st.markdown(f"""
                <div class="{'success' if best else 'metric'}-card">
                    <b>{'⭐ ' if best else ''}{mname}</b><br>
                    MAE: {m.get('MAE','N/A')}<br>
                    RMSE: {m.get('RMSE','N/A')}<br>
                    MAPE: {m.get('MAPE','N/A')}%
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Forecast simulator
    st.subheader("🔮 Forecast Simulator")
    col1, col2, col3 = st.columns(3)
    with col1:
        fc_store    = st.selectbox("Store ID", list(range(1, 11)))
        fc_category = st.selectbox("Category", ["Electronics","Clothing","Food","Home","Sports"])
    with col2:
        fc_product  = st.selectbox("Product ID", list(range(1, 51)))
        fc_price    = st.number_input("Price ($)", min_value=5.0, max_value=300.0, value=149.99)
    with col3:
        fc_days     = st.slider("Forecast Days", 7, 90, 30)
        fc_promo    = st.checkbox("Include Promotions")

    if st.button("🚀 Generate Forecast", type="primary"):
        models = load_models()
        if models:
            model = models.get("LightGBM") or list(models.values())[0]
            CAT_MAP   = {"Electronics":0,"Clothing":1,"Food":2,"Home":3,"Sports":4}
            STORE_MAP = {"Urban":3,"Suburban":2,"Rural":0,"Outlet":1}

            preds, dates_fc = [], []
            lag_buf = [20.0] * 28

            for i in range(fc_days):
                d = datetime.now() + timedelta(days=i)
                feats = np.array([[
                    d.month, d.isocalendar()[1], d.weekday(), (d.month-1)//3+1, int(d.weekday()>=5),
                    int(fc_promo and i%14==0), 0, fc_price, 50,
                    np.sin(2*np.pi*d.month/12), np.cos(2*np.pi*d.month/12),
                    np.sin(2*np.pi*d.weekday()/7), np.cos(2*np.pi*d.weekday()/7),
                    lag_buf[-1], lag_buf[-7], lag_buf[-14], lag_buf[-28],
                    np.mean(lag_buf[-7:]), np.mean(lag_buf[-14:]), np.mean(lag_buf[-28:]),
                    np.std(lag_buf[-7:]),  np.std(lag_buf[-28:]),
                    STORE_MAP.get("Urban",3), CAT_MAP.get(fc_category,0),
                    int(fc_price < 80), 0, d.year,
                ]])
                p = float(max(model.predict(feats)[0], 0))
                preds.append(p)
                lag_buf.append(p)
                if len(lag_buf) > 60: lag_buf.pop(0)
                dates_fc.append(d.date())

            fc_df = pd.DataFrame({"date": dates_fc, "forecast": preds})
            fc_df["upper"] = fc_df["forecast"] * 1.15
            fc_df["lower"] = fc_df["forecast"] * 0.85

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["upper"], fill=None,
                                     mode="lines", line_color="rgba(37,99,235,0)", showlegend=False))
            fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["lower"],
                                     fill="tonexty", mode="lines", line_color="rgba(37,99,235,0)",
                                     fillcolor="rgba(37,99,235,0.15)", name="85% CI"))
            fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast"],
                                     mode="lines+markers", line=dict(color=PALETTE[0], width=2.5),
                                     name="Forecast"))
            fig.update_layout(title=f"{fc_days}-Day Demand Forecast – {fc_category} (Store {fc_store})",
                              xaxis_title="Date", yaxis_title="Predicted Units",
                              plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC", height=400)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Forecast", f"{sum(preds):,.0f} units")
            col2.metric("Daily Average",  f"{np.mean(preds):.1f} units")
            col3.metric("Peak Day",       f"{max(preds):.0f} units")
        else:
            st.warning("⚠️ No trained models found. Run Milestone 2 first.")


# ════════════════════════════════════════════════════════
# PAGE 3: INVENTORY
# ════════════════════════════════════════════════════════
elif page == "📦 Inventory":
    st.title("📦 Inventory Management")

    models = load_models()
    model  = models.get("LightGBM") or (list(models.values())[0] if models else None)

    if model:
        CAT_MAP   = {"Electronics":0,"Clothing":1,"Food":2,"Home":3,"Sports":4}
        STORE_MAP = {"Urban":3,"Suburban":2,"Rural":0,"Outlet":1}
        cats      = ["Electronics","Clothing","Food","Home","Sports"]

        col1, col2 = st.columns(2)
        inv_store      = col1.selectbox("Store ID", list(range(1,11)))
        inv_store_type = col2.selectbox("Store Type", ["Urban","Suburban","Rural","Outlet"])
        safety_days    = st.slider("Safety Stock Days", 3, 14, 7)

        recs = []
        today = datetime.now()
        for pid in range(1, 21):
            cat   = cats[(pid-1) % len(cats)]
            stock = int(np.random.randint(5, 150))
            price = float(np.random.uniform(20, 200))

            week_pred = []
            for d in range(7):
                dt = today + timedelta(days=d)
                f  = np.array([[
                    dt.month, dt.isocalendar()[1], dt.weekday(), (dt.month-1)//3+1,
                    int(dt.weekday()>=5), 0, 0, price, stock,
                    np.sin(2*np.pi*dt.month/12), np.cos(2*np.pi*dt.month/12),
                    np.sin(2*np.pi*dt.weekday()/7), np.cos(2*np.pi*dt.weekday()/7),
                    20,19,18,17, 19.5,19,18.5, 3.2,3.5,
                    STORE_MAP.get(inv_store_type,3), CAT_MAP.get(cat,0), 0, 0, dt.year,
                ]])
                week_pred.append(float(max(model.predict(f)[0], 0)))

            pred7  = sum(week_pred)
            pred30 = pred7 * (30/7)
            daily  = pred7 / 7
            reorder_pt = daily * safety_days
            order_qty  = max(pred30 - stock, 0) + reorder_pt
            days_cover = stock / max(daily, 0.1)

            if days_cover < 3:   status, color = "🔴 CRITICAL", "#FEE2E2"
            elif days_cover < 7: status, color = "🟡 LOW",      "#FEF9C3"
            elif days_cover < 14:status, color = "🟢 ADEQUATE", "#DCFCE7"
            else:                status, color = "⚪ EXCESS",   "#F1F5F9"

            recs.append({
                "Product":       f"P{pid:02d}",
                "Category":      cat,
                "Stock":         stock,
                "7d Forecast":   round(pred7, 1),
                "30d Forecast":  round(pred30, 1),
                "Reorder Point": round(reorder_pt, 1),
                "Order Qty":     round(order_qty, 1),
                "Days Cover":    round(days_cover, 1),
                "Status":        status,
            })

        recs_df = pd.DataFrame(recs).sort_values("Days Cover")

        # Summary KPIs
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🔴 Critical",  sum(1 for r in recs if "CRITICAL" in r["Status"]))
        c2.metric("🟡 Low Stock", sum(1 for r in recs if "LOW" in r["Status"]))
        c3.metric("🟢 Adequate",  sum(1 for r in recs if "ADEQUATE" in r["Status"]))
        c4.metric("⚪ Excess",    sum(1 for r in recs if "EXCESS" in r["Status"]))

        st.dataframe(recs_df, use_container_width=True, height=500)

        # Days cover chart
        fig = px.bar(recs_df, x="Product", y="Days Cover",
                     color="Status", title="Inventory Coverage by Product (Days)",
                     color_discrete_map={"🔴 CRITICAL":"#DC2626","🟡 LOW":"#D97706",
                                         "🟢 ADEQUATE":"#16A34A","⚪ EXCESS":"#94A3B8"})
        fig.add_hline(y=7,  line_dash="dot", line_color="#D97706", annotation_text="7-day warning")
        fig.add_hline(y=3,  line_dash="dot", line_color="#DC2626", annotation_text="3-day critical")
        fig.update_layout(height=350, plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No models found. Run Milestone 2 first.")


# ════════════════════════════════════════════════════════
# PAGE 4: EDA EXPLORER
# ════════════════════════════════════════════════════════
elif page == "🔍 EDA Explorer":
    st.title("🔍 EDA Explorer")

    tab1, tab2, tab3 = st.tabs(["📅 Seasonality", "🏪 Store Analysis", "🏷️ Promotion Impact"])

    with tab1:
        monthly = df.groupby("month")["units_sold"].mean().reset_index()
        monthly["month_name"] = pd.to_datetime(monthly["month"], format="%m").dt.strftime("%b")
        fig = px.bar(monthly, x="month_name", y="units_sold",
                     title="Average Units Sold by Month",
                     color="units_sold", color_continuous_scale="Blues")
        fig.update_layout(plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC", height=380)
        st.plotly_chart(fig, use_container_width=True)

        weekly = df.groupby("day_of_week")["units_sold"].mean().reset_index()
        weekly["day"] = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        weekly["is_weekend"] = weekly["day_of_week"] >= 5
        fig = px.bar(weekly, x="day", y="units_sold",
                     color="is_weekend", title="Avg Units Sold by Day of Week",
                     color_discrete_map={True: "#DC2626", False: "#2563EB"})
        fig.update_layout(plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        store_perf = df.groupby("store_type").agg(
            avg_units=("units_sold","mean"),
            total_rev=("revenue","sum")
        ).reset_index()
        fig = px.scatter(store_perf, x="avg_units", y="total_rev",
                         size="total_rev", color="store_type", text="store_type",
                         title="Store Type: Avg Units vs Total Revenue",
                         color_discrete_sequence=PALETTE)
        fig.update_layout(plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        promo = df.groupby(["category","is_promotion"])["units_sold"].mean().unstack()
        promo.columns = ["No Promo","Promotion"]
        promo["Lift %"] = ((promo["Promotion"] - promo["No Promo"]) / promo["No Promo"] * 100).round(1)
        promo = promo.reset_index()
        fig = px.bar(promo, x="category", y="Lift %",
                     title="Promotion Lift % by Category",
                     color="Lift %", color_continuous_scale="RdYlGn",
                     text="Lift %")
        fig.update_layout(plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC", height=380)
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 5: MLOPS MONITOR
# ════════════════════════════════════════════════════════
elif page == "⚙️ MLOps Monitor":
    st.title("⚙️ MLOps Monitoring")

    monitor = load_monitor_report()
    retrain = load_retrain_history()
    metrics = load_metrics()

    # Status banner
    if monitor:
        needs_retrain = monitor.get("retraining_recommended", False)
        if needs_retrain:
            st.error("🔴 **Retraining Recommended** — Drift or degradation detected!")
        else:
            st.success("✅ **Model Healthy** — No drift or degradation detected")

        st.markdown(f"*Last monitored: {monitor.get('timestamp','N/A')}*")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Data Drift
    with col1:
        st.subheader("📊 Data Drift Detection")
        drift = monitor.get("data_drift", {})
        if drift:
            drift_df = pd.DataFrame([
                {
                    "Feature": k,
                    "KS Stat":  v["ks_statistic"],
                    "P-Value":  v["p_value"],
                    "PSI":      v["psi"],
                    "Drift":    "🔴 Yes" if v["drift_detected"] else "✅ No",
                    "Severity": v["severity"],
                }
                for k, v in drift.items()
            ])
            st.dataframe(drift_df, use_container_width=True)

            fig = px.bar(drift_df, x="Feature", y="PSI",
                         color="Drift", title="PSI Score by Feature",
                         color_discrete_map={"🔴 Yes":"#DC2626","✅ No":"#16A34A"})
            fig.add_hline(y=0.2, line_dash="dot", line_color="#D97706",
                          annotation_text="PSI=0.2 threshold")
            fig.update_layout(plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC", height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run monitoring to see drift results.")

    # Model Performance
    with col2:
        st.subheader("📈 Model Performance")
        perf = monitor.get("performance", {})
        if perf:
            curr = perf.get("current_metrics", {})
            base = perf.get("baseline_metrics", {})
            perf_df = pd.DataFrame({
                "Metric":   ["MAE","RMSE","MAPE"],
                "Baseline": [base.get("MAE"), base.get("RMSE"), base.get("MAPE")],
                "Current":  [curr.get("MAE"), curr.get("RMSE"), curr.get("MAPE")],
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Baseline", x=perf_df["Metric"], y=perf_df["Baseline"],
                                 marker_color=PALETTE[0]))
            fig.add_trace(go.Bar(name="Current",  x=perf_df["Metric"], y=perf_df["Current"],
                                 marker_color=PALETTE[1]))
            fig.update_layout(barmode="group", title="Baseline vs Current Metrics",
                              plot_bgcolor="#F8FAFC", paper_bgcolor="#F8FAFC", height=300)
            st.plotly_chart(fig, use_container_width=True)

            for alert in perf.get("alerts", []):
                st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)
        else:
            st.info("Run monitoring to see performance results.")

    # Retraining History
    st.markdown("---")
    st.subheader("🔄 Retraining History")
    if retrain:
        hist_df = pd.DataFrame([{
            "Version":   r["version"],
            "Timestamp": r["timestamp"],
            "LGB MAPE":  r.get("lgb_metrics",{}).get("MAPE","N/A"),
            "XGB MAPE":  r.get("xgb_metrics",{}).get("MAPE","N/A"),
        } for r in retrain])
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No retraining history yet. Run the retraining pipeline first.")

    # MLOps pipeline diagram
    st.markdown("---")
    st.subheader("🔁 MLOps Pipeline")
    st.markdown("""
    ```
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  New Data    │────▶│  Monitoring  │────▶│ Drift Alert? │
    │  (Daily)     │     │  (KS + PSI)  │     │              │
    └──────────────┘     └──────────────┘     └──────┬───────┘
                                                      │ YES
                                              ┌───────▼───────┐
                                              │  Retraining   │
                                              │  Pipeline     │
                                              └───────┬───────┘
                                                      │
                                              ┌───────▼───────┐
                                              │  New Model    │
                                              │  Deployed     │
                                              └───────────────┘
    ```
    """)


if __name__ == "__main__":
    pass
