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
# 4. It makes a one-day-ahead forecast of the ENTIRE YIELD CURVE and compares it
#    to the actual market outcome.
# 5. After the loop, it aggregates daily curve errors to calculate performance metrics.
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
    """
    Robustly converts a string to a datetime object by trying multiple common formats.
    """
    if pd.isna(s):
        return pd.NaT
    
    # List of formats to try in order of priority
    formats_to_try = [
        '%m/%d/%Y', '%m-%d-%Y',  # Month-first formats
        '%d/%m/%Y', '%d-%m-%Y',  # Day-first formats
        '%Y/%m/%d', '%Y-%m-%d',  # Year-first formats
    ]
    
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    
    # Fallback to general parser if specific formats fail
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
    # Step 1: Create a DataFrame for zero rates on the standard grid
    std_cols = [f"{m:.2f}Y" for m in std_arr]
    pca_df_zeros = pd.DataFrame(np.nan, index=yields_df_train.index, columns=std_cols, dtype=float)
    available_contracts = yields_df_train.columns
    for dt in yields_df_train.index:
        pca_df_zeros.loc[dt] = row_to_std_grid(
            dt, yields_df_train.loc[dt], available_contracts, expiry_df,
            std_arr, holidays_np, year_basis, rate_unit, interp_method
        )

    # Step 2: Calculate Spreads
    spread_cols = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
    pca_df_spreads = pd.DataFrame(np.nan, index=pca_df_zeros.index, columns=spread_cols, dtype=float)
    for i in range(1, len(std_cols)):
        col_name = f"{std_cols[i]}-{std_cols[i-1]}"
        pca_df_spreads[col_name] = pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]]

    # Step 3: Calculate Butterflies
    fly_cols = [f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
    pca_df_flies = pd.DataFrame(np.nan, index=pca_df_zeros.index, columns=fly_cols, dtype=float)
    for i in range(1, len(std_cols) - 1):
        col_name = f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}"
        pca_df_flies[col_name] = (pca_df_zeros[std_cols[i+1]] - pca_df_zeros[std_cols[i]]) - (pca_df_zeros[std_cols[i]] - pca_df_zeros[std_cols[i-1]])

    # Step 4: Combine all series into a single PCA matrix
    pca_df_combined = pd.concat([pca_df_zeros, pca_df_spreads, pca_df_flies], axis=1)

    # Fill NaN values with the column mean (after calculation)
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
    """
    Calculates rates, spreads, and flies directly from raw, non-interpolated data
    by mapping standard maturities to the first, second, third, etc. available
    live contracts.
    """
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
    
    # Create a sorted list of available raw contracts and their rates for the current day
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

    # Sort the data by expiry date to ensure we get them in the correct sequence
    raw_data.sort(key=lambda x: x['expiry'])
    
    # Map each standard maturity to the correct sequential live contract
    rates_map = {std_arr[i]: np.nan for i in range(len(std_arr))}
    for i in range(len(std_arr)):
        if i < len(raw_data):
            rates_map[std_arr[i]] = raw_data[i]['rate']
    
    rates = np.array([rates_map.get(t, np.nan) for t in std_arr])
    
    spreads = np.full(len(std_arr) - 1, np.nan)
    flies = np.full(len(std_arr) - 2, np.nan)

    # Correctly calculate spreads from the mapped rates
    for i in range(len(spreads)):
        rate1 = rates[i+1]
        rate2 = rates[i]
        if not np.isnan(rate1) and not np.isnan(rate2):
            spreads[i] = rate1 - rate2

    # Correctly calculate flies from the mapped rates
    for i in range(len(flies)):
        rate1 = rates[i+2]
        rate2 = rates[i+1]
        rate3 = rates[i]
        if not np.isnan(rate1) and not np.isnan(rate2) and not np.isnan(rate3):
            flies[i] = (rate1 - rate2) - (rate2 - rate3)
            
    return rates, spreads, flies


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
    
# --- Callbacks for Next/Prev Buttons ---
def next_date():
    if st.session_state.selected_date_index < len(st.session_state.unique_dates) - 1:
        st.session_state.selected_date_index += 1

def prev_date():
    if st.session_state.selected_date_index > 0:
        st.session_state.selected_date_index -= 1

def next_spread_date():
    if st.session_state.selected_spread_date_index < len(st.session_state.unique_dates) - 1:
        st.session_state.selected_spread_date_index += 1

def prev_spread_date():
    if st.session_state.selected_spread_date_index > 0:
        st.session_state.selected_spread_date_index -= 1

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
# Main Logic
# --------------------------

# Only run the backtest if the button was clicked AND results are not already computed
if run_backtest and st.session_state.results_df is None:
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
        yields_df[date_col] = yields_df[date_col].apply(safe_to_datetime)
        yields_df = yields_df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        yields_df.columns = [str(c).strip() for c in yields_df.columns]
        for c in yields_df.columns:
            yields_df[c] = pd.to_numeric(yields_df[c], errors="coerce")
        
        # Expiry
        expiry_raw = pd.read_csv(io.StringIO(expiry_file.getvalue().decode("utf-8")))
        expiry_df = expiry_raw.iloc[:, :2].copy()
        expiry_df.columns = ["MATURITY", "DATE"]
        expiry_df["MATURITY"] = expiry_df["MATURITY"].astype(str).str.strip().str.upper()
        expiry_df["DATE"] = expiry_df["DATE"].apply(safe_to_datetime)
        expiry_df = expiry_df.dropna(subset=["DATE"]).set_index("MATURITY")

        # Holidays
        holidays_np = np.array([], dtype="datetime64[D]")
        if holiday_file:
            hol_df = pd.read_csv(io.StringIO(holiday_file.getvalue().decode("utf-8")))
            hol_series = hol_df.iloc[:, 0].apply(safe_to_datetime).dropna()
            if not hol_series.empty:
                holidays_np = np.array(hol_series.dt.date, dtype="datetime64[D]")
                
        return yields_df, expiry_df, holidays_np

    yields_df, expiry_df, holidays_np = load_all_data(yield_file, expiry_file, holiday_file)

    # --------------------------
    # BACKTESTING LOOP
    # --------------------------
    st.title("Backtest Results")
    results = []
    
    # Prepare date ranges
    backtest_range = pd.bdate_range(start=backtest_start_date, end=backtest_end_date)
    all_available_dates = yields_df.index

    # Setup grid
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
    std_cols = [f"{m:.2f}Y" for m in std_arr]
    
    spread_cols_pca = [f"{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols))]
    fly_cols_pca = [f"{std_cols[i+1]}-{std_cols[i]}-{std_cols[i-1]}" for i in range(1, len(std_cols) - 1)]
    all_cols_full = std_cols + spread_cols_pca + fly_cols_pca

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, current_date in enumerate(backtest_range):
        if current_date not in all_available_dates:
            continue

        training_end_date = current_date - pd.Timedelta(days=1)
        training_start_date = training_end_date - pd.DateOffset(days=training_window_days * 1.5)
        
        train_mask = (yields_df.index >= training_start_date) & (yields_df.index <= training_end_date)
        yields_df_train = yields_df.loc[train_mask].sort_index().tail(training_window_days)

        if len(yields_df_train) < training_window_days / 2:
            continue

        # Build the full PCA matrix (rates, spreads, flies) from interpolated training data
        pca_df_filled = build_pca_matrix(yields_df_train, expiry_df, std_arr, holidays_np, year_basis, rate_unit, interp_method)
        
        if pca_df_filled.empty: continue

        scaler = StandardScaler(with_std=False)
        X = scaler.fit_transform(pca_df_filled.values.astype(float))
        n_components_sel_capped = min(n_components_sel, X.shape[1])
        pca = PCA(n_components=n_components_sel_capped)
        PCs = pca.fit_transform(X)
        pc_cols = [f"PC{i+1}" for i in range(n_components_sel_capped)]
        PCs_df = pd.DataFrame(PCs, index=pca_df_filled.index, columns=pc_cols)

        if forecast_model_type == "PCA Fair Value":
            last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
            reconstructed_centered = pca.inverse_transform(last_pcs)
            pred_full_curve = scaler.inverse_transform(reconstructed_centered).flatten()
        else:
            if forecast_model_type == "VAR (Vector Autoregression)":
                pcs_next = forecast_pcs_var(PCs_df, lags=var_lags)
            else: # ARIMA
                pcs_next = forecast_pcs_arima(PCs_df)
            
            last_actual_full_curve = pca_df_filled.iloc[-1].values
            last_pcs = PCs_df.iloc[-1].values.reshape(1, -1)
            delta_pcs = pcs_next - last_pcs
            delta_curve = pca.inverse_transform(delta_pcs)
            pred_full_curve = last_actual_full_curve + delta_curve.flatten()

        # Build the ACTUAL curve for comparison using RAW data for all components
        actual_rates_raw, actual_spreads_raw, actual_flies_raw = calculate_raw_metrics(
            current_date, 
            yields_df.loc[current_date], 
            yields_df.columns, 
            expiry_df, 
            rate_unit,
            holidays_np, 
            year_basis
        )
        
        actual_full_curve_series = pd.Series(
            np.concatenate([actual_rates_raw, actual_spreads_raw, actual_flies_raw]), 
            index=all_cols_full
        )
        
        actual_full_curve = actual_full_curve_series.values

        results.append({
            "Date": current_date,
            "Predicted_Curve": pred_full_curve,
            "Actual_Curve": actual_full_curve,
            "Column_Names": all_cols_full
        })

        progress_bar.progress((i + 1) / len(backtest_range))
        status_text.text(f"Processing: {current_date.date()}...")

    status_text.success("Backtest complete!")
    progress_bar.empty()
    
    if results:
        st.session_state.results_df = pd.DataFrame(results)

# --------------------------
# RESULTS ANALYSIS (runs if results are in state)
# --------------------------
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    st.title("Backtest Results")
    
    # Filter out rows where the actual curve is all NaN
    mask = results_df['Actual_Curve'].apply(lambda x: isinstance(x, float) or np.isnan(x).all())
    results_df = results_df[~mask].copy()

    if results_df.empty:
        st.error("No valid results to analyze after cleaning. The model may have failed to generate valid curves for the selected backtest period.")
        st.stop()

    # Get the column names from the first valid row
    all_cols = results_df.iloc[0]['Column_Names']
    std_cols = [c for c in all_cols if '-' not in c]
    spread_cols = [c for c in all_cols if '-' in c and len(c.split('-')) == 2]
    fly_cols = [c for c in all_cols if '-' in c and len(c.split('-')) == 3]

    # Calculate errors for each component type
    def calculate_errors(row, cols):
        pred = row['Predicted_Curve'][np.isin(all_cols, cols)]
        actual = row['Actual_Curve'][np.isin(all_cols, cols)]
        if np.isnan(actual).all(): return np.nan
        return pd.Series({
            'RMSE': np.sqrt(np.nanmean((pred - actual)**2)),
            'MAE': np.nanmean(np.abs(pred - actual))
        })
    
    results_df[['Daily_RMSE_Rates', 'Daily_MAE_Rates']] = results_df.apply(lambda row: calculate_errors(row, std_cols), axis=1)
    results_df[['Daily_RMSE_Spreads', 'Daily_MAE_Spreads']] = results_df.apply(lambda row: calculate_errors(row, spread_cols), axis=1)
    results_df[['Daily_RMSE_Flies', 'Daily_MAE_Flies']] = results_df.apply(lambda row: calculate_errors(row, fly_cols), axis=1)

    # --------------------------
    # Outright Rates Results
    # --------------------------
    st.header("Overall Curve Performance Metrics (Outright Rates)")
    overall_rmse_rates = results_df['Daily_RMSE_Rates'].mean()
    overall_mae_rates = results_df['Daily_MAE_Rates'].mean()

    col1, col2 = st.columns(2)
    col1.metric("Average Daily Curve RMSE (Rates)", f"{overall_rmse_rates:.4f}")
    col2.metric("Average Daily Curve MAE (Rates)", f"{overall_mae_rates:.4f}")

    st.markdown("---")
    st.header("Daily Curve Performance Visualization (Outright Rates)")
    st.write("Select a date from the backtest to visually inspect the model's forecast against the actual yield curve.")

    unique_dates = results_df['Date'].dt.date.unique()
    std_arr = np.array(build_std_grid_by_rule(7.0), dtype=float)
    
    st.session_state.unique_dates = unique_dates # Store unique dates in session state for callbacks
    
    prev_col, date_col, next_col = st.columns([1, 4, 1])
    prev_col.button("◀ Previous", on_click=prev_date)
    next_col.button("Next ▶", on_click=next_date)
    
    selected_date = date_col.selectbox(
        "Select a date to inspect",
        options=unique_dates,
        index=st.session_state.selected_date_index
    )
    
    if selected_date:
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

    st.markdown("---")
    st.subheader("Raw Curve Data (Outright Rates)")
    st.write("This table contains the raw predicted and actual curve vectors for each day of the backtest.")
    
    raw_rates_df = pd.DataFrame(results_df['Date']).set_index('Date')
    
    # Extract predicted and actual rates into their own columns
    for i, col in enumerate(std_cols):
        raw_rates_df[f"Predicted_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[i])
        raw_rates_df[f"Actual_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[i])

    st.dataframe(raw_rates_df)


    # --------------------------
    # Spreads and Flies Results
    # --------------------------
    st.markdown("---")
    st.header("Backtesting Results for Spreads and Flies")
    st.write("This section shows the backtesting performance on the model's ability to forecast spreads and butterflies.")
    
    col3, col4 = st.columns(2)
    col3.metric("Average Daily Curve RMSE (Spreads)", f"{results_df['Daily_RMSE_Spreads'].mean():.4f}")
    col4.metric("Average Daily Curve MAE (Spreads)", f"{results_df['Daily_MAE_Spreads'].mean():.4f}")

    col5, col6 = st.columns(2)
    col5.metric("Average Daily Curve RMSE (Flies)", f"{results_df['Daily_RMSE_Flies'].mean():.4f}")
    col6.metric("Average Daily Curve MAE (Flies)", f"{results_df['Daily_MAE_Flies'].mean():.4f}")


    st.markdown("---")
    st.subheader("Daily Performance Visualization (Spreads and Flies)")
    st.write("Select a date to visualize the predicted vs. actual spread and fly values.")

    prev_col, spread_date_col, next_col = st.columns([1, 4, 1])
    prev_col.button("◀ Previous", key="spread_prev", on_click=prev_spread_date)
    next_col.button("Next ▶", key="spread_next", on_click=next_spread_date)
    
    selected_spread_date = spread_date_col.selectbox(
        "Select a date to inspect",
        options=unique_dates,
        index=st.session_state.selected_spread_date_index
    )
    
    if selected_spread_date:
        plot_data = results_df[results_df['Date'].dt.date == selected_spread_date]
        if not plot_data.empty:
            actual_spreads = plot_data['Actual_Curve'].iloc[0][np.isin(all_cols, spread_cols)]
            pred_spreads = plot_data['Predicted_Curve'].iloc[0][np.isin(all_cols, spread_cols)]
            actual_flies = plot_data['Actual_Curve'].iloc[0][np.isin(all_cols, fly_cols)]
            pred_flies = plot_data['Predicted_Curve'].iloc[0][np.isin(all_cols, fly_cols)]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Spreads Plot
            ax1.plot(spread_cols, actual_spreads, marker='o', linestyle='-', color='royalblue', label=f"Actual Spreads")
            ax1.plot(spread_cols, pred_spreads, marker='x', linestyle='--', color='darkorange', label="Predicted Spreads")
            ax1.set_title(f"Spread Forecast vs. Actual for {selected_spread_date}")
            ax1.set_ylabel("Spread (bps)")
            ax1.set_xlabel("Spread (Years)")
            ax1.set_xticklabels(spread_cols, rotation=45, ha="right")
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.legend()
            
            # Flies Plot
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

    st.markdown("---")
    st.subheader("Raw Curve Data (Spreads and Flies)")
    st.write("This table contains the raw predicted and actual spread and fly vectors for each day of the backtest.")
    
    raw_spreads_flies_df = pd.DataFrame(results_df['Date']).set_index('Date')
    
    # Get the start and end indices for spreads and flies within the full curve vector
    spreads_start_idx = len(std_cols)
    spreads_end_idx = spreads_start_idx + len(spread_cols)
    flies_start_idx = spreads_end_idx
    flies_end_idx = flies_start_idx + len(fly_cols)

    # Extract predicted and actual spreads and flies into their own columns
    for i, col in enumerate(spread_cols):
        raw_spreads_flies_df[f"Predicted_Spread_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[spreads_start_idx + i])
        raw_spreads_flies_df[f"Actual_Spread_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[spreads_start_idx + i])
        
    for i, col in enumerate(fly_cols):
        raw_spreads_flies_df[f"Predicted_Fly_{col}"] = results_df['Predicted_Curve'].apply(lambda x: x[flies_start_idx + i])
        raw_spreads_flies_df[f"Actual_Fly_{col}"] = results_df['Actual_Curve'].apply(lambda x: x[flies_start_idx + i])

    st.dataframe(raw_spreads_flies_df)
else:
    st.info("Upload files, configure the backtest parameters, and click **Run Backtest**.")
