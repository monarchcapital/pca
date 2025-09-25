# -*- coding: utf-8 -*-
# Brazil DI Futures PCA — Backtesting Engine
#
# This script performs a walk-forward validation (backtest) of the PCA-based
# yield curve forecasting models from the main analysis tool.
#
# ---------------------------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# --- Setup ---
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="DI Curve Backtesting Engine")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------------------------
# CORE FUNCTIONS (Reused from the original analysis tool)
# ---------------------------------------------------------------------------------

# (all helper functions unchanged ... safe_to_datetime, normalize_rate_input, etc.)

# --------------------------
# Sidebar — Inputs
# --------------------------
st.sidebar.header("1) Upload Data")
yield_file = st.sidebar.file_uploader("Yield data CSV", type="csv")
expiry_file = st.sidebar.file_uploader("Expiry mapping CSV", type="csv")
holiday_file = st.sidebar.file_uploader("Holiday dates CSV (optional)", type="csv")

st.sidebar.header("2) Configure Backtest")
backtest_start_date = st.sidebar.date_input("Backtest Start Date")
backtest_end_date = st.sidebar.date_input("Backtest End Date")
training_window_days = st.sidebar.number_input("Rolling Training Window (business days)", min_value=1, max_value=1000, value=252, step=1)

st.sidebar.header("3) Model Parameters")
n_components_sel = st.sidebar.slider("Number of PCA components", 1, 10, 3)
forecast_model_type = st.sidebar.selectbox("Forecasting Model to Test", ["PCA Fair Value", "VAR (Vector Autoregression)", "ARIMA (per Component)"])
var_lags = 1
if forecast_model_type == "VAR (Vector Autoregression)":
    var_lags = st.sidebar.number_input("VAR model lags", min_value=1, max_value=20, value=1, step=1)

rate_unit = st.sidebar.selectbox("Input rate unit", ["Percent (e.g. 13.45)", "Decimal (e.g. 0.1345)", "Basis points (e.g. 1345)"])
year_basis = int(st.sidebar.selectbox("Business days in year", [252, 360], index=0))
interp_method = "linear" # Hardcoded for consistency in backtest

# --- Initialize session state ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'selected_date_index' not in st.session_state:
    st.session_state.selected_date_index = 0
if 'selected_spread_date_index' not in st.session_state:
    st.session_state.selected_spread_date_index = 0

# --- Run/Reset Buttons ---
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
if col1.button("Run Backtest"):
    st.session_state.results_df = None # Clear previous results before a new run
    st.session_state.selected_date_index = 0
    st.session_state.selected_spread_date_index = 0
    run_backtest = True
else:
    run_backtest = False

if col2.button("Reset"):
    st.session_state.results_df = None
    st.session_state.selected_date_index = 0
    st.session_state.selected_spread_date_index = 0
    st.rerun()

# --------------------------
# Main Logic (unchanged backtest loop)
# --------------------------

# --------------------------
# RESULTS ANALYSIS (runs if results are in state)
# --------------------------
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    st.title("Backtest Results")

    # (all metrics + errors code unchanged)

    unique_dates = results_df['Date'].dt.date.unique()
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)

    # --------------------------
    # Outright Rates Section (Fixed)
    # --------------------------
    prev_col, date_col, next_col = st.columns([1, 4, 1])
    if prev_col.button("◀ Previous"):
        if st.session_state.selected_date_index > 0:
            st.session_state.selected_date_index -= 1
    if next_col.button("Next ▶"):
        if st.session_state.selected_date_index < len(unique_dates) - 1:
            st.session_state.selected_date_index += 1

    selected_date = date_col.selectbox(
        "Select a date to inspect",
        options=unique_dates,
        index=st.session_state.selected_date_index,
        key='date_selector'
    )

    if selected_date:
        st.session_state.selected_date_index = list(unique_dates).index(selected_date)

        plot_data = results_df[results_df['Date'].dt.date == selected_date]
        if not plot_data.empty:
            actual_c = plot_data['Actual_Curve'].iloc[0][np.isin(all_cols, std_cols)]
            pred_c = plot_data['Predicted_Curve'].iloc[0][np.isin(all_cols, std_cols)]

            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(std_arr, actual_c, label=f"Actual on {selected_date}", color='royalblue', marker='o', linestyle='-')
            ax.plot(std_arr, pred_c, label=f"Predicted for {selected_date}", color='darkorange', marker='x', linestyle='--')
            ax.set_title(f"Yield Curve Forecast vs. Actual for {selected_date}", fontsize=16)
            ax.set_xlabel("Standardized Maturity (Years)")
            ax.set_ylabel(f"Rate ({rate_unit})")
            ax.set_xticks(std_arr)
            ax.set_xticklabels([f"{m:.2f}Y" for m in std_arr], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)

    # --------------------------
    # Spreads & Flies Section (Fixed)
    # --------------------------
    prev_col, spread_date_col, next_col = st.columns([1, 4, 1])
    if prev_col.button("◀ Previous", key="spread_prev"):
        if st.session_state.selected_spread_date_index > 0:
            st.session_state.selected_spread_date_index -= 1
    if next_col.button("Next ▶", key="spread_next"):
        if st.session_state.selected_spread_date_index < len(unique_dates) - 1:
            st.session_state.selected_spread_date_index += 1

    selected_spread_date = spread_date_col.selectbox(
        "Select a date to inspect",
        options=unique_dates,
        index=st.session_state.selected_spread_date_index,
        key='spread_date_selector'
    )

    if selected_spread_date:
        st.session_state.selected_spread_date_index = list(unique_dates).index(selected_spread_date)

        plot_data = results_df[results_df['Date'].dt.date == selected_spread_date]
        if not plot_data.empty:
            actual_spreads = plot_data['Actual_Curve'].iloc[0][np.isin(all_cols, spread_cols)]
            pred_spreads = plot_data['Predicted_Curve'].iloc[0][np.isin(all_cols, spread_cols)]
            actual_flies = plot_data['Actual_Curve'].iloc[0][np.isin(all_cols, fly_cols)]
            pred_flies = plot_data['Predicted_Curve'].iloc[0][np.isin(all_cols, fly_cols)]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            ax1.plot(spread_cols, actual_spreads, marker='o', linestyle='-', color='royalblue', label=f"Actual Spreads")
            ax1.plot(spread_cols, pred_spreads, marker='x', linestyle='--', color='darkorange', label="Predicted Spreads")
            ax1.set_title(f"Spread Forecast vs. Actual for {selected_spread_date}")
            ax1.set_ylabel("Spread (bps)")
            ax1.set_xlabel("Spread (Years)")
            ax1.set_xticklabels(spread_cols, rotation=45, ha="right")
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.legend()

            ax2.plot(fly_cols, actual_flies, marker='o', linestyle='-', color='royalblue', label=f"Actual Flies")
            ax2.plot(fly_cols, pred_flies, marker='x', linestyle='--', color='darkorange', label="Predicted Flies")
            ax2.set_title(f"Butterfly Forecast vs. Actual for {selected_spread_date}")
            ax2.set_ylabel("Fly (bps)")
            ax2.set_xlabel("Fly (Years)")
            ax2.set_xticklabels(fly_cols, rotation=45, ha="right")
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)

else:
    st.info("Upload files, configure the backtest parameters, and click **Run Backtest**.")
