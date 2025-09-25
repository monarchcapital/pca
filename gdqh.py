# -*- coding: utf-8 -*-
# Brazil DI Futures PCA — Backtesting Engine

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
# CORE FUNCTIONS (same as your original file, unchanged)
# ---------------------------------------------------------------------------------
# ... [all your helper functions here, unchanged] ...


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
interp_method = "linear"

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
    st.session_state.results_df = None
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
# Main Backtest Loop
# --------------------------
# (your original loop stays here, unchanged)

# --------------------------
# RESULTS ANALYSIS
# --------------------------
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    st.title("Backtest Results")

    unique_dates = results_df['Date'].dt.date.unique()
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)

    # --------------------------
    # Outright Rates Navigation (FIXED)
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

    # ✅ FIXED: use selected_date directly
    if selected_date:
        st.session_state.selected_date_index = list(unique_dates).index(selected_date)

    # --------------------------
    # Spreads & Flies Navigation (FIXED)
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

    # ✅ FIXED: use selected_spread_date directly
    if selected_spread_date:
        st.session_state.selected_spread_date_index = list(unique_dates).index(selected_spread_date)

else:
    st.info("Upload files, configure the backtest parameters, and click **Run Backtest**.")
