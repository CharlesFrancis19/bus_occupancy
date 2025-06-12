import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import os
import time
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from capymoa.stream import CSVStream
from capymoa.regressor import PassiveAggressiveRegressor
from capymoa.evaluation import prequential_evaluation

st.set_page_config(page_title="Live Bus Forecast", layout="wide", initial_sidebar_state="expanded")

DATA_FILE = "loader_03-05_2024.csv"
if not os.path.exists(DATA_FILE):
    st.error(f"🚫 '{DATA_FILE}' not found. Please upload or place the file next to this script.")
    st.stop()

df = pd.read_csv(DATA_FILE)
df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
stop_columns = df.columns[1:]

st.sidebar.header("🛠️ Dashboard Settings")
auto_refresh = st.sidebar.checkbox("🔁 Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 30)
selected_stops = st.sidebar.multiselect("🛑 Select Stops to Compare", stop_columns, default=[stop_columns[0]])

# Only refresh after page has rendered once
if auto_refresh and "last_refresh" in st.session_state:
    time.sleep(refresh_interval)
    st.session_state.pop("last_refresh")
    st.rerun()
else:
    st.session_state["last_refresh"] = time.time()

tab1, tab2, tab3 = st.tabs(["📈 Multi-Stop Comparison", "📉 RMSE Trend", "🔍 Search & Live Counter"])

def run_model_for_stop(stop_col):
    df_selected = df[["timestamp", stop_col]].dropna().rename(columns={stop_col: "target"})
    df_selected["hour"] = pd.to_datetime(df_selected["timestamp"]).dt.hour
    df_selected["day"] = pd.to_datetime(df_selected["timestamp"]).dt.day
    df_selected["weekday"] = pd.to_datetime(df_selected["timestamp"]).dt.weekday
    df_selected["target"] = pd.to_numeric(df_selected["target"], errors="coerce").astype(float)

    if df_selected.empty:
        return [], df_selected

    temp_csv = f"temp_{stop_col}.csv"
    df_selected[["hour", "day", "weekday", "target"]].to_csv(temp_csv, index=False)

    stream = CSVStream(
        csv_file_path=temp_csv,
        target_attribute_name="target",
        target_type="numeric",
        values_for_nominal_features={},
        delimiter=",",
        dataset_name="BusRegression"
    )
    schema = stream.get_schema()
    model = PassiveAggressiveRegressor(schema)

    try:
        results = prequential_evaluation(
            stream=stream,
            learner=model,
            max_instances=1000,
            store_predictions=True,
            store_y=True,
            optimise=True,
            restart_stream=False
        )
    except Exception as e:
        os.remove(temp_csv)
        return [], df_selected

    os.remove(temp_csv)

    y_true_raw = results.ground_truth_y()
    y_pred_raw = results.predictions()
    filtered = [(yt, yp) for yt, yp in zip(y_true_raw, y_pred_raw) if yp is not None]

    return filtered, df_selected

with tab1:
    st.subheader("📈 Multi-Stop True vs Predicted")
    fig = go.Figure()
    for stop in selected_stops:
        filtered, _ = run_model_for_stop(stop)
        if filtered:
            y_true, y_pred = zip(*filtered)
            fig.add_trace(go.Scatter(y=y_pred[:300], mode='lines', name=f"{stop} (Predicted)"))
            fig.add_trace(go.Scatter(y=y_true[:300], mode='lines', name=f"{stop} (True)", line=dict(dash='dot')))
    fig.update_layout(title="True vs Predicted Occupancy", xaxis_title="Samples", yaxis_title="Occupancy")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📉 RMSE Trend per Batch")
    batch_size = 100
    for stop in selected_stops:
        filtered, _ = run_model_for_stop(stop)
        if filtered:
            y_true, y_pred = zip(*filtered)
            batches = [(
                root_mean_squared_error(y_true[i:i+batch_size], y_pred[i:i+batch_size])
            ) for i in range(0, len(y_true) - batch_size, batch_size)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=batches, mode='lines+markers', name=stop))
            fig.update_layout(title=f"{stop} - RMSE Trend", xaxis_title="Batch", yaxis_title="RMSE")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🔍 Search Forecast by Stop and Time")
    stop_id = st.selectbox("Select Stop", stop_columns, key="search_stop")
    filtered, df_selected = run_model_for_stop(stop_id)

    if filtered:
        y_true, y_pred = zip(*filtered)
        df_selected["datetime"] = pd.to_datetime(df_selected["timestamp"])
        available_times = df_selected["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        selected_time_str = st.selectbox("Available Timestamps", available_times, key="available_times")
        selected_time = pd.to_datetime(selected_time_str)

        closest = df_selected.iloc[(df_selected["datetime"] - selected_time).abs().argsort()[:1]]
        pred_index = closest.index[0] if not closest.empty else None

        if pred_index is not None and pred_index < len(y_pred):
            st.metric("🎯 Predicted Occupancy", f"{y_pred[pred_index]:.2f}")
            st.metric("📌 Actual Occupancy", f"{y_true[pred_index]:.2f}")
        else:
            st.warning("❗ Prediction not available for selected timestamp.")
    else:
        st.warning("❗ Not enough predictions made for selected stop.")
