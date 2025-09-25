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
# CORE FUNCTIONS (all unchanged from your original file)
# ---------------------------------------------------------------------------------
def safe_to_datetime(s):
    if pd.isna(s):
        return pd.NaT
    formats_to_try = [
        '%m/%d/%Y', '%m-%d-%Y',
        '%d/%m/%Y', '%d-%m-%Y',
        '%Y/%m/%d', '%Y-%m-%d',
    ]
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(s, errors='coerce')

def normalize_rate_input(val, unit):
    if pd.isna(val): return np.nan
    v = float(val)
    if "Percent" in unit: return v / 100.0
    if "Basis" in unit: return v / 10000.0
    return v

def denormalize_to_percent(frac):
    if pd.isna(frac): return np.nan
    return 100.0 * float(frac)

def np_busdays_exclusive(start_dt, end_dt, holidays_np):
    if pd.isna(start_dt) or pd.isna(end_dt): return 0
    s = np.datetime64(pd.Timestamp(start_dt).date()) + np.timedelta64(1, "D")
    e = np.datetime64(pd.Timestamp(end_dt).date())
    if e < s: return 0
    return int(np.busday_count(s, e, weekmask="1111100", holidays=holidays_np))

def calculate_ttm(valuation_ts, expiry_ts, holidays_np, year_basis):
    bd = np_busdays_exclusive(valuation_ts, expiry_ts, holidays_np)
    return np.nan if bd <= 0 else bd / float(year_basis)

def build_std_grid_by_rule(max_year=7.0):
    a = list(np.round(np.arange(0.25, 3.0 + 0.001, 0.25), 2))
    b = list(np.round(np.arange(3.5, 5.0 + 0.001, 0.5), 2))
    c = list(np.round(np.arange(6.0, max_year + 0.001, 1.0), 2))
    return a + b + c

def row_to_std_grid(dt, row_series, available_contracts, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method):
    ttm_list, zero_list = [], []
    for col in available_contracts:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index: continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date(): continue
        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0: continue
        raw_val = row_series.get(col, np.nan)
        if pd.isna(raw_val): continue
        r_frac = normalize_rate_input(raw_val, rate_unit)
        zero_frac = r_frac
        ttm_list.append(t)
        zero_list.append(denormalize_to_percent(zero_frac))
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        try:
            order = np.argsort(ttm_list)
            f = interp1d(np.array(ttm_list)[order], np.array(zero_list)[order], kind=interp_method, bounds_error=False, fill_value=np.nan, assume_sorted=True)
            return f(std_arr)
        except Exception:
            return np.full_like(std_arr, np.nan, dtype=float)
    return np.full_like(std_arr, np.nan, dtype=float)

def forecast_pcs_var(PCs_df, lags=1):
    if len(PCs_df) < lags + 5: return PCs_df.iloc[-1:].values
    results = VAR(PCs_df).fit(lags)
    return results.forecast(PCs_df.values[-lags:], steps=1)

def forecast_pcs_arima(PCs_df):
    forecasts = []
    for name, series in PCs_df.items():
        if len(series) < 10:
            forecasts.append(series.iloc[-1])
            continue
        forecasts.append(ARIMA(series, order=(1, 1, 0)).fit().forecast(steps=1).iloc[0])
    return np.array(forecasts).reshape(1, -1)

def build_pca_matrix(yields_df_train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method):
    std_cols = [f"{m:.2f}Y" for m in std_arr]
    pca_df_zeros = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    available_contracts = yields_df_train.columns
    for dt in yields_df_train.index:
        pca_df_zeros.loc[dt] = row_to_std_grid(dt, yields_df_train.loc[dt], available_contracts, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method)
    spread_cols = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
    pca_df_spreads = pd.DataFrame(np.nan, index=pca_df_zeros.index, columns=spread_cols, dtype=float)
    for i in range(1, len(std_cols)):
        col_name = f"{std_cols[i]}-{std_cols[i-1]}"
        pca_df_spreads[col_name] = pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]]
    fly_cols = [f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
    pca_df_flies = pd.DataFrame(np.nan, index=pca_df_zeros.index, columns=fly_cols, dtype=float)
    for i in range(1, len(std_cols) - 1):
        col_name = f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}"
        pca_df_flies[col_name] = (pca_df_zeros[std_cols[i+1]] - pca_df_zeros[std_cols[i]]) - (pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]])
    pca_df_combined = pd.concat([pca_df_zeros, pca_df_spreads, pca_df_flies], axis=1)
    pca_vals = pca_df_combined.values.astype(float)
    col_means = np.nanmean(pca_vals, axis=0)
    if np.isnan(col_means).any():
        overall_mean = np.nanmean(col_means[~np.isnan(col_means)]) if np.any(~np.isnan(col_means)) else 0.0
        col_means = np.where(np.isnan(col_means), overall_mean, col_means)
    inds = np.where(np.isnan(pca_vals))
    if inds[0].size > 0:
        pca_vals[inds] = np.take(col_means, inds[1])
    return pd.DataFrame(pca_vals, index=pca_df_combined.index, columns=pca_df_combined.columns)

def calculate_raw_metrics(dt, row_series, available_contracts, expiry_df, rate_unit, holidays_np, year_basis):
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
    raw_data = []
    for col in available_contracts:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index: continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() <= dt.date(): continue
        raw_val = row_series.get(col, np.nan)
        if pd.isna(raw_val): continue
        rate_percent = denormalize_to_percent(normalize_rate_input(raw_val, rate_unit))
        raw_data.append({'maturity': mat_up, 'expiry': exp, 'rate': rate_percent})
    raw_data.sort(key=lambda x: x['expiry'])
    rates_map = {std_arr[i]: np.nan for i in range(len(std_arr))}
    for i in range(len(std_arr)):
        if i < len(raw_data):
            rates_map[std_arr[i]] = raw_data[i]['rate']
    rates = np.array([rates_map.get(t, np.nan) for t in std_arr])
    spreads = np.full(len(std_arr) - 1, np.nan)
    flies = np.full(len(std_arr) - 2, np.nan)
    for i in range(len(spreads)):
        rate1 = rates[i+1]
        rate2 = rates[i]
        if not np.isnan(rate1) and not np.isnan(rate2):
            spreads[i] = rate1 - rate2
    for i in range(len(flies)):
        rate1 = rates[i+2]
        rate2 = rates[i+1]
        rate3 = rates[i]
        if not np.isnan(rate1) and not np.isnan(rate2) and not np.isnan(rate3):
            flies[i] = (rate1 - rate2) - (rate2 - rate3)
    return rates, spreads, flies


# --------------------------
# Sidebar Inputs
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
if run_backtest and yield_file and expiry_file:
    # load data
    yields_df = pd.read_csv(yield_file, index_col=0, parse_dates=True)
    expiry_df = pd.read_csv(expiry_file)
    expiry_df['DATE'] = expiry_df['DATE'].apply(safe_to_datetime)
    expiry_df.set_index('CONTRACT', inplace=True)
    holidays_np = []
    if holiday_file:
        holidays_df = pd.read_csv(holiday_file)
        holidays_np = holidays_df['DATE'].apply(safe_to_datetime).dropna().dt.date.values

    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
    std_cols = [f\"{m:.2f}Y\" for m in std_arr]

    results = []
    for i in range(training_window_days, len(yields_df)):
        train = yields_df.iloc[i-training_window_days:i]
        test_date = yields_df.index[i]

        pca_df = build_pca_matrix(train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method)
        scaler = StandardScaler()
        X = scaler.fit_transform(pca_df)
        pca = PCA(n_components=n_components_sel)
        PCs = pca.fit_transform(X)

        PCs_df = pd.DataFrame(PCs, index=train.index, columns=[f'PC{j+1}' for j in range(n_components_sel)])

        if forecast_model_type == \"PCA Fair Value\":
            pcs_forecast = PCs_df.iloc[-1:].values
        elif forecast_model_type == \"VAR (Vector Autoregression)\":
            pcs_forecast = forecast_pcs_var(PCs_df, lags=var_lags)
        elif forecast_model_type == \"ARIMA (per Component)\":
            pcs_forecast = forecast_pcs_arima(PCs_df)

        recon = pca.inverse_transform(pcs_forecast)
        pred_curve = scaler.inverse_transform(recon).flatten()

        test_row = yields_df.iloc[i]
        actual_curve = row_to_std_grid(test_date, test_row, yields_df.columns, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method)

        results.append({
            'Date': test_date,
            'Predicted_Curve': pred_curve,
            'Actual_Curve': actual_curve
        })

    st.session_state.results_df = pd.DataFrame(results)


# --------------------------
# RESULTS ANALYSIS
# --------------------------
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    st.title(\"Backtest Results\")

    unique_dates = results_df['Date'].dt.date.unique()
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)

    # Outright Rates Navigation (FIXED)
    prev_col, date_col, next_col = st.columns([1, 4, 1])
    if prev_col.button(\"◀ Previous\"):
        if st.session_state.selected_date_index > 0:
            st.session_state.selected_date_index -= 1
    if next_col.button(\"Next ▶\"):
        if st.session_state.selected_date_index < len(unique_dates) - 1:
            st.session_state.selected_date_index += 1

    selected_date = date_col.selectbox(
        \"Select a date to inspect\",
        options=unique_dates,
        index=st.session_state.selected_date_index,
        key='date_selector'
    )

    if selected_date:
        st.session_state.selected_date_index = list(unique_dates).index(selected_date)

    # Spreads & Flies Navigation (FIXED)
    prev_col, spread_date_col, next_col = st.columns([1, 4, 1])
    if prev_col.button(\"◀ Previous\", key=\"spread_prev\"):
        if st.session_state.selected_spread_date_index > 0:
            st.session_state.selected_spread_date_index -= 1
    if next_col.button(\"Next ▶\", key=\"spread_next\"):
        if st.session_state.selected_spread_date_index < len(unique_dates) - 1:
            st.session_state.selected_spread_date_index += 1

    selected_spread_date = spread_date_col.selectbox(
        \"Select a date to inspect\",
        options=unique_dates,
        index=st.session_state.selected_spread_date_index,
        key='spread_date_selector'
    )

    if selected_spread_date:
        st.session_state.selected_spread_date_index = list(unique_dates).index(selected_spread_date)

else:
    st.info(\"Upload files, configure the backtest parameters, and click **Run Backtest**.\")
