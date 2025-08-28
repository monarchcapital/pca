# -*- coding: utf-8 -*-
# Brazil DI Futures PCA — Backtesting Engine
#
# This script performs a walk-forward validation (backtest) of the PCA-based
# yield curve forecasting models from the main analysis tool.
#
# How it works:
# 1. It takes a defined backtesting period (e.g., a full year).
# 2. It iterates day-by-day through this period.
# 3. On each day, it uses a "rolling window" of past data (e.g., the last 252 days)
#    to train the PCA model and the selected forecasting model (VAR, ARIMA, etc.).
# 4. It makes a one-day-ahead forecast and compares it to the actual market outcome.
# 5. After the loop, it aggregates all results and calculates performance metrics.
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
# NOTE: These functions are the core data processing engine and are kept identical
# to ensure the backtest accurately reflects the main tool's logic.

def safe_to_datetime(s):
    return pd.to_datetime(s, errors='coerce', dayfirst=False)

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
        zero_frac = r_frac # Assumes identity compounding for simplicity in backtest
        ttm_list.append(t)
        zero_list.append(denormalize_to_percent(zero_frac))
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        try:
            order = np.argsort(ttm_list)
            f = interp1d(np.array(ttm_list)[order], np.array(zero_list)[order], kind=interp_method, bounds_error=False, fill_value=np.nan, assume_sorted=True)
            return f(std_arr)
        except Exception: return np.full_like(std_arr, np.nan, dtype=float)
    return np.full_like(std_arr, np.nan, dtype=float)

def std_grid_to_contracts(dt, std_curve_rates, std_arr, expiry_df, all_contracts, holidays_np, year_basis, rate_unit, interp_method):
    contract_rates = pd.Series(index=all_contracts, dtype=float)
    interp_func = interp1d(std_arr, std_curve_rates, kind=interp_method, bounds_error=False, fill_value='extrapolate')
    for contract in all_contracts:
        mat_up = str(contract).strip().upper()
        if mat_up not in expiry_df.index: continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date(): continue
        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if np.isnan(t) or t <= 0: continue
        zero_percent = interp_func(t)
        rate_frac = zero_percent / 100.0
        if "Percent" in rate_unit: contract_rates[contract] = rate_frac * 100.0
        elif "Basis" in rate_unit: contract_rates[contract] = rate_frac * 10000.0
        else: contract_rates[contract] = rate_frac
    return contract_rates.dropna()

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
training_window_days = st.sidebar.number_input("Rolling Training Window (business days)", min_value=50, max_value=1000, value=252, step=1)

st.sidebar.header("3) Model Parameters")
n_components_sel = st.sidebar.slider("Number of PCA components", 1, 10, 3)
forecast_model_type = st.sidebar.selectbox("Forecasting Model to Test", ["PCA Fair Value", "VAR (Vector Autoregression)", "ARIMA (per Component)"])
var_lags = 1
if forecast_model_type == "VAR (Vector Autoregression)":
    var_lags = st.sidebar.number_input("VAR model lags", min_value=1, max_value=20, value=1, step=1)

rate_unit = st.sidebar.selectbox("Input rate unit", ["Percent (e.g. 13.45)", "Decimal (e.g. 0.1345)", "Basis points (e.g. 1345)"])
year_basis = int(st.sidebar.selectbox("Business days in year", [252, 360], index=0))
interp_method = "linear" # Hardcoded for consistency in backtest

# --- Run Button ---
st.sidebar.markdown("---")
if st.sidebar.button("Run Backtest"):
    st.session_state.run_backtest = True
else:
    st.session_state.run_backtest = False

if not st.session_state.run_backtest:
    st.info("Upload files, configure the backtest parameters, and click **Run Backtest**.")
    st.stop()
if not all([yield_file, expiry_file]):
    st.error("Please upload both Yield and Expiry data files.")
    st.stop()
if backtest_start_date >= backtest_end_date:
    st.error("Backtest Start Date must be before the End Date.")
    st.stop()

# --------------------------
# Data Loading
# --------------------------
@st.cache_data
def load_all_data(yield_file, expiry_file, holiday_file):
    # Yields
    yields_df = pd.read_csv(io.StringIO(yield_file.getvalue().decode("utf-8")))
    date_col = yields_df.columns[0]
    yields_df[date_col] = safe_to_datetime(yields_df[date_col])
    yields_df = yields_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    yields_df.columns = [str(c).strip() for c in yields_df.columns]
    for c in yields_df.columns:
        yields_df[c] = pd.to_numeric(yields_df[c], errors="coerce")
    
    # Expiry
    expiry_raw = pd.read_csv(io.StringIO(expiry_file.getvalue().decode("utf-8")))
    expiry_df = expiry_raw.iloc[:, :2].copy()
    expiry_df.columns = ["MATURITY", "DATE"]
    expiry_df["MATURITY"] = expiry_df["MATURITY"].astype(str).str.strip().str.upper()
    expiry_df["DATE"] = safe_to_datetime(expiry_df["DATE"])
    expiry_df = expiry_df.dropna(subset=["DATE"]).set_index("MATURITY")

    # Holidays
    holidays_np = np.array([], dtype="datetime64[D]")
    if holiday_file:
        hol_df = pd.read_csv(io.StringIO(holiday_file.getvalue().decode("utf-8")))
        hol_series = safe_to_datetime(hol_df.iloc[:, 0]).dropna()
        if not hol_series.empty:
            holidays_np = np.array(hol_series.dt.date, dtype="datetime64[D]")
            
    return yields_df, expiry_df, holidays_np

yields_df, expiry_df, holidays_np = load_all_data(yield_file, expiry_file, holiday_file)

# --------------------------
# BACKTESTING LOOP
# --------------------------
st.title("Backtest Results")

# Prepare date ranges
backtest_range = pd.bdate_range(start=backtest_start_date, end=backtest_end_date)
all_available_dates = yields_df.index
results = []

# Setup grid
std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
std_cols = [f"{m:.2f}Y" for m in std_arr]

progress_bar = st.progress(0)
status_text = st.empty()

for i, current_date in enumerate(backtest_range):
    if current_date not in all_available_dates:
        continue # Skip if it's a weekend/holiday not in our data

    # 1. Define the training window for this iteration
    training_end_date = current_date - pd.Timedelta(days=1)
    training_start_date = training_end_date - pd.DateOffset(days=training_window_days * 1.5) # Approx to get enough business days
    
    train_mask = (yields_df.index >= training_start_date) & (yields_df.index <= training_end_date)
    yields_df_train = yields_df.loc[train_mask].sort_index().tail(training_window_days)

    if len(yields_df_train) < training_window_days / 2:
        status_text.warning(f"Skipping {current_date.date()}: Not enough training data ({len(yields_df_train)} days).")
        continue

    # 2. Run PCA on the training window data
    # Build standardized matrix
    pca_df = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    for dt in yields_df_train.index:
        pca_df.loc[dt] = row_to_std_grid(dt, yields_df_train.loc[dt], yields_df_train.columns, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method)
    
    # Fill NaNs and run PCA
    pca_vals = pca_df.values
    col_means = np.nanmean(pca_vals, axis=0)
    inds = np.where(np.isnan(pca_vals))
    pca_vals[inds] = np.take(col_means, inds[1])
    pca_df_filled = pd.DataFrame(pca_vals, index=pca_df.index, columns=pca_df.columns)

    scaler = StandardScaler(with_std=False)
    X = scaler.fit_transform(pca_df_filled.values.astype(float))
    pca = PCA(n_components=n_components_sel)
    PCs = pca.fit_transform(X)
    pc_cols = [f"PC{i+1}" for i in range(n_components_sel)]
    PCs_df = pd.DataFrame(PCs, index=pca_df_filled.index, columns=pc_cols)

    # 3. Generate Forecast
    if forecast_model_type == "PCA Fair Value":
        # The forecast is the smooth, reconstructed curve of the last day in the training window.
        last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
        reconstructed_centered = pca.inverse_transform(last_pcs)
        pred_curve_std = scaler.inverse_transform(reconstructed_centered).flatten()
    else:
        if forecast_model_type == "VAR (Vector Autoregression)":
            pcs_next = forecast_pcs_var(PCs_df, lags=var_lags)
        else: # ARIMA
            pcs_next = forecast_pcs_arima(PCs_df)

        # Apply the predicted CHANGE to the last known actual curve
        last_actual_curve_on_std = pca_df_filled.iloc[-1].values
        last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
        delta_pcs = pcs_next - last_pcs
        delta_curve = pca.inverse_transform(delta_pcs)
        pred_curve_std = last_actual_curve_on_std + delta_curve.flatten()

    # 4. Convert forecast back to contracts and store results
    pred_contracts = std_grid_to_contracts(current_date, pred_curve_std, std_arr, expiry_df, yields_df.columns, holidays_np, year_basis, rate_unit, interp_method)
    actual_contracts = yields_df.loc[current_date]

    # Align and store
    common_contracts = pred_contracts.index.intersection(actual_contracts.index)
    for contract in common_contracts:
        results.append({
            "Date": current_date,
            "Contract": contract,
            "Predicted": pred_contracts[contract],
            "Actual": actual_contracts[contract]
        })

    # Update progress
    progress_bar.progress((i + 1) / len(backtest_range))
    status_text.text(f"Processing: {current_date.date()}...")

status_text.success("Backtest complete!")
progress_bar.empty()

# --------------------------
# RESULTS ANALYSIS
# --------------------------
if not results:
    st.error("No results were generated. This might be because the backtest date range has no valid data points. Please check your data and date selections.")
    st.stop()

results_df = pd.DataFrame(results)
# --- FIX: Drop rows with NaN values before calculations to prevent errors ---
results_df.dropna(inplace=True)

results_df['Error'] = results_df['Predicted'] - results_df['Actual']
results_df['Abs_Error'] = np.abs(results_df['Error'])

st.header("Overall Performance Metrics")
rmse = np.sqrt(mean_squared_error(results_df['Actual'], results_df['Predicted']))
mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])

# Directional Accuracy
results_df['Actual_Change'] = results_df.groupby('Contract')['Actual'].diff()
results_df['Predicted_Change'] = results_df.groupby('Contract')['Predicted'].diff()
results_df['Correct_Direction'] = (np.sign(results_df['Actual_Change']) == np.sign(results_df['Predicted_Change']))
directional_accuracy = results_df['Correct_Direction'].mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
col3.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")

st.markdown("---")
st.header("Performance by Contract")

# --- Error Analysis by Contract ---
error_by_contract = results_df.groupby('Contract').agg(
    MAE=('Abs_Error', 'mean'),
    RMSE=('Error', lambda x: np.sqrt(np.mean(x**2)))
).sort_values('MAE', ascending=False)

st.subheader("Average Error by Contract")
st.dataframe(error_by_contract.style.format("{:.4f}"))

# --- Visualization ---
st.markdown("---")
st.header("Predicted vs. Actual Visualization")
contracts_to_plot = results_df['Contract'].unique()
selected_contract = st.selectbox("Select a contract to visualize", options=contracts_to_plot)

plot_df = results_df[results_df['Contract'] == selected_contract].set_index('Date')

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(plot_df.index, plot_df['Actual'], label='Actual', color='royalblue', marker='.', linestyle='-')
ax.plot(plot_df.index, plot_df['Predicted'], label='Predicted', color='darkorange', marker='.', linestyle='--')
ax.set_title(f"Predicted vs. Actual Rates for {selected_contract}", fontsize=16)
ax.set_xlabel("Date")
ax.set_ylabel(f"Rate ({rate_unit})")
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Raw Results Data")
st.dataframe(results_df)

