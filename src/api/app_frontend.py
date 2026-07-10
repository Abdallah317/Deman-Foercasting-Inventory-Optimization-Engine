import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="DEPI - Demand Forecasting Engine", 
    layout="wide", 
    page_icon="📊"
)

# App Header
st.title("📊 Demand Forecasting & Inventory Optimization System")
st.subheader("DEPI Graduation Project — Executive Control Dashboard")
st.markdown("---")

# Navigation Tabs
tab1, tab2 = st.tabs(["🔮 Real-Time Prediction (Predict)", "📈 Future Trend Analysis (Forecast)"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: REAL-TIME PREDICTION (POST /predict)
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("🔍 Single Item Stock Analysis & Decision Support")
    st.write("Input operational parameters below to evaluate inventory risks for a specific date.")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        p_date = st.date_input("Prediction Date", value=datetime.strptime("2024-01-15", "%Y-%m-%d"))
        product_id = st.number_input("Product ID", min_value=1, max_value=999, value=5, key="p1")
        category = st.selectbox("Product Category", ["Electronics", "Clothing", "Food", "Home", "Sports"], key="c1")
        
    with col_b:
        store_id = st.selectbox("Store ID", [1, 2, 3, 4, 5], key="s1")
        store_type = st.selectbox("Store Type", ["Urban", "Suburban", "Rural", "Outlet"])
        price = st.number_input("Current Unit Price ($)", value=149.99, key="pr1")
        
    with col_c:
        stock_level = st.number_input("Current Warehouse Stock Level", value=50)
        is_promo = st.checkbox("Is Promotion Active?")
        is_holiday = st.checkbox("Is it a Public Holiday?")

    st.markdown("---")
    
    if st.button("🚀 Run AI Analysis Engine"):
        API_URL = "http://127.0.0.1:8000/predict"
        
        # Payload matching your exact ML features structure
        payload = {
            "date": str(p_date),
            "store_id": int(store_id),
            "product_id": int(product_id),
            "category": category,
            "store_type": store_type,
            "price": float(price),
            "stock_level": int(stock_level),
            "is_promotion": int(is_promo),
            "is_holiday": int(is_holiday),
            "model": "lightgbm",
            # Default fallback values for historical features
            "lag_1": 20, "lag_7": 18, "lag_14": 19, "lag_28": 17,
            "rolling_mean_7": 19.5, "rolling_mean_14": 18.8, "rolling_mean_28": 18.2,
            "rolling_std_7": 3.2, "rolling_std_28": 3.5
        }
        
        with st.spinner("Invoking LightGBM inference model..."):
            try:
                res = requests.post(API_URL, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    
                    # Performance Metrics Cards
                    c1, c2, c3 = st.columns(3)
                    c1.metric(label="🛍️ Predicted Demand", value=f"{data['predicted_units']:,} Units")
                    
                    # Color-coded Inventory Status
                    status = data['inventory_status'].upper()
                    if "CRITICAL" in status:
                        c2.error(f"🚨 Stock Status: {status}")
                    elif "LOW" in status:
                        c2.warning(f"⚠️ Stock Status: {status}")
                    else:
                        c2.success(f"✅ Stock Status: {status}")
                        
                    # Reorder recommendations
                    if data['reorder_recommended']:
                        c3.metric(label="🔄 Reorder Action", value="TRIGGER REORDER", delta="Urgent Procurement Required", delta_color="inverse")
                    else:
                        c3.metric(label="🔄 Reorder Action", value="HOLD", delta="Inventory Safe")
                    
                    # Confidence Interval Details
                    st.info(
                        f"💡 **Model Confidence Details ({data['confidence_interval']['confidence']}):** "
                        f"Actual sales are statistical bound to fall between **{data['confidence_interval']['lower']}** "
                        f"and **{data['confidence_interval']['upper']}** units."
                    )
                else:
                    st.error(f"API Error. Status Code: {res.status_code}. Response: {res.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server. Ensure it is running on port 8000. Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: FUTURE TREND ANALYSIS (GET /forecast/{days})
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("📈 Time-Series Demand Curve Forecasting")
    st.write("Generate dynamic rolling timelines to visualize upcoming supply trends over a specific horizon.")
    
    col_x, col_y = st.columns(2)
    with col_x:
        days = st.slider("Forecast Horizon (Days)", min_value=1, max_value=30, value=7)
    with col_y:
        st.write("") # Spacer

    if st.button("🔮 Generate Multi-Day Demand Curve"):
        API_URL = f"http://127.0.0.1:8000/forecast/{days}"
        params = {
            "store_id": int(store_id),
            "product_id": int(product_id),
            "category": category,
            "price": float(price)
        }
        
        with st.spinner("Processing multi-day macro forecast..."):
            try:
                response = requests.get(API_URL, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Macro stats summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="📦 Aggregate Horizon Demand", value=f"{data['total_expected_units']:,} Units")
                    col2.metric(label="🔥 Peak Demand Window", value=data['peak_demand_date'])
                    col3.metric(label="📅 Timeline Window", value=f"{data['start_date']} to {data['end_date']}")
                    
                    st.markdown("---")
                    
                    # Convert to dataframe for charting
                    df = pd.DataFrame(data["daily_forecasts"])
                    
                    # Interactive Line Chart
                    fig = px.line(
                        df, x="date", y="predicted_units",
                        title="Timeline Demand Trajectory Curve",
                        labels={"date": "Date Timeline", "predicted_units": "Expected Units Demanded"},
                        markers=True,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabular expansion for raw records
                    with st.expander("📋 Inspect Raw Daily Data Points"):
                        st.dataframe(df, use_container_width=True)
                else:
                    st.error(f"Backend failed to compute forecast. Status: {response.status_code}")
            except Exception as e:
                st.error(f"Connection refused by Backend API: {e}")