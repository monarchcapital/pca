import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import io
from datetime import timedelta, date
from pandas.tseries.offsets import BDay

# Set a consistent style for plots
sns.set_style("darkgrid")

def robust_date_parser(df, date_column):
    """
    Parses a date column using multiple known formats and returns the most successful result.
    This function avoids a generic parser fallback that can cause warnings and inconsistencies.
    """
    formats_to_try = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d %B %Y', '%d-%b-%y']
    best_result = None
    fewest_errors = float('inf')

    for fmt in formats_to_try:
        parsed_dates = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
        num_errors = parsed_dates.isnull().sum()

        if num_errors < fewest_errors:
            best_result = parsed_dates
            fewest_errors = num_errors

        if num_errors == 0:
            break  # found perfect format, exit early

    if best_result is not None:
        return best_result, fewest_errors

    fallback = pd.to_datetime(df[date_column], errors='coerce')
    return fallback, fallback.isnull().sum()

def run_analysis(yield_data, expiry_data, holiday_data,
                 standard_maturities_years, interpolation_method,
                 use_smoothing, n_components, use_geometric_average,
                 start_date_filter, end_date_filter):
    """
    DI-specific pipeline:
    - convert DI futures quotes -> discount factors (DF)
    - DF -> zero rates (annualized, discrete compounding consistent with DI)
    - interpolate zero rates to standard maturities
    - mean-center (no scaling) and run PCA
    Returns: pca_df (zero rates at standard maturities), pca_model, principal_components,
             scaler (mean-centering scaler), raw_yield_df, expiry_df, available_maturities, holidays_set
    """
    try:
        # Load and normalize column names
        df = pd.read_csv(io.StringIO(yield_data.getvalue().decode("utf-8")))
        df.columns = [col.strip().upper() for col in df.columns]

        expiry_df = pd.read_csv(io.StringIO(expiry_data.getvalue().decode("utf-8")))
        expiry_df.columns = [col.strip().upper() for col in expiry_df.columns]

        # Date parsing with improved logic
        df['DATE'], _ = robust_date_parser(df, 'DATE')
        expiry_df['DATE'], _ = robust_date_parser(expiry_df, 'DATE')

        df.dropna(subset=['DATE'], inplace=True)
        expiry_df.dropna(subset=['DATE'], inplace=True)

        df.set_index('DATE', inplace=True)
        # Normalize maturity strings
        expiry_df['MATURITY'] = expiry_df['MATURITY'].astype(str).str.strip().str.upper()
        expiry_df = expiry_df.set_index('MATURITY')

        # remove duplicate columns and ensure numeric yields
        df = df.loc[:, ~df.columns.duplicated()]
        yield_cols = [c for c in df.columns if c != 'DATE']  # should be maturities
        for col in yield_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # holidays
        holidays_set = set()
        if holiday_data:
            try:
                holiday_df = pd.read_csv(io.StringIO(holiday_data.getvalue().decode("utf-8")))
                holiday_df.columns = [c.strip().upper() for c in holiday_df.columns]
                holiday_parsed_dates, _ = robust_date_parser(holiday_df, 'DATE')
                holidays_set = set(holiday_parsed_dates.dropna().dt.date)
            except Exception:
                st.warning("Could not parse holidays file. Proceeding without holiday filtering.")
                holidays_set = set()

        # apply date range filter with more robust error handling
        if start_date_filter is not None and end_date_filter is not None:
            if start_date_filter > end_date_filter:
                st.error("Start date cannot be after end date.")
                return (None,) * 8
            df = df.loc[(df.index.date >= start_date_filter) & (df.index.date <= end_date_filter)]

        if df.empty:
            st.error("No yield data in selected date range after parsing.")
            return (None,) * 8

        # business day mask (remove weekends & holidays)
        is_business_day = df.index.weekday < 5
        is_not_holiday = ~pd.Series(df.index.date).isin(holidays_set).values
        df = df[is_business_day & is_not_holiday]
        if df.empty:
            st.error("No business days left after holiday/weekend filtering.")
            return (None,) * 8

        # smoothing optional (applies to raw yields)
        if use_smoothing:
            df = df.rolling(window=3, min_periods=1, center=True).mean()

        # match maturities
        yield_maturities = set(df.columns)
        expiry_maturities = set(expiry_df.index)
        available_maturities_in_df = sorted(list(yield_maturities & expiry_maturities))
        if not available_maturities_in_df:
            st.error("No maturity names match between yield file and expiry file.")
            return (None,) * 8

        st.info(f"Matched maturities: {available_maturities_in_df}")

        # Prepare output dataframe of standard maturities (zero rates)
        # Ensure standard maturities are sorted for interpolation
        std_mats = np.array(sorted(standard_maturities_years), dtype=float)
        
        pca_df = pd.DataFrame(index=df.index, columns=[f'{m:.2f}Y' for m in std_mats])

        BUSINESS_YEAR_DAYS = 252

        dates_skipped = 0
        dates_with_nan = 0
        
        for i, date in enumerate(df.index):
            ttm_list = []
            zero_list = []
            row = df.loc[date]

            for mat in available_maturities_in_df:
                expiry_date_val = expiry_df.loc[mat, 'DATE']
                if pd.isnull(expiry_date_val):
                    continue
                expiry_dt = pd.to_datetime(expiry_date_val).date()
                if expiry_dt < date.date():
                    # expired - skip
                    continue

                # Correction: Exclude valuation date from the count
                bd_range = pd.bdate_range(date + pd.Timedelta(days=1), pd.Timestamp(expiry_dt))
                # Remove holidays from the range
                if holidays_set:
                    bd_range = [d for d in bd_range if d.date() not in holidays_set]

                working_days_to_expiry = len(bd_range)
                
                if working_days_to_expiry <= 0:
                    continue

                raw_yield = row.get(mat, np.nan)
                if pd.isnull(raw_yield):
                    continue

                # DI convention: annualized rate in percent (e.g. 14.5)
                t = working_days_to_expiry / BUSINESS_YEAR_DAYS  # in years (fraction)
                r = float(raw_yield) / 100.0

                # compute discount factor consistent with DI quote:
                # DF = (1 + r)^{-t}
                DF = (1.0 + r) ** (-t)

                # implied zero annual rate (discrete compounding): DF = (1 + zero)^{-t} -> zero = DF^{-1/t} - 1
                # handle tiny t
                if t <= 0:
                    continue
                try:
                    zero = DF ** (-1.0 / t) - 1.0
                except Exception:
                    continue

                # convert back to percentage
                zero_list.append(zero * 100.0)
                ttm_list.append(t)

            # need at least 2 distinct points to interpolate
            if len(ttm_list) > 1 and len(set(np.round(ttm_list, 10))) > 1:
                # sort
                order = np.argsort(ttm_list)
                t_sorted = np.array(ttm_list)[order]
                z_sorted = np.array(zero_list)[order]

                # interpolation: do NOT extrapolate. Use fill_value=np.nan.
                try:
                    f = interp1d(t_sorted, z_sorted, kind=interpolation_method,
                                 bounds_error=False, fill_value=np.nan, assume_sorted=True)
                    interp_vals = f(std_mats)
                    
                    # Instead of skipping the date entirely, we'll keep the row but note the NaNs
                    if np.any(np.isnan(interp_vals)):
                        dates_with_nan += 1
                        
                    pca_df.loc[date] = interp_vals
                except Exception:
                    dates_skipped += 1
                    continue
            else:
                dates_skipped += 1

        st.info(f"Skipped {dates_skipped} dates due to insufficient points for interpolation.")
        if dates_with_nan > 0:
            st.info(f"Retained {dates_with_nan} dates with partial data where some standard maturities were outside the interpolation range. These will appear as NaNs.")


        # drop empty rows
        pca_df.dropna(how='all', inplace=True)
        if pca_df.empty:
            st.error("No interpolated zero-rate curves available after processing.")
            return (None,) * 8

        # Add warning for significant NaN filling before PCA
        total_elements = pca_df.size
        nan_count = pca_df.isnull().sum().sum()
        if total_elements > 0:
            nan_percentage = (nan_count / total_elements) * 100
            if nan_percentage > 5:
                st.warning(f"Warning: {nan_percentage:.2f}% of data contains missing values (NaNs). These will be filled with the column mean for PCA.")
        
        # Mean-center only (keep variance)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(with_std=False)  # center only
        
        # We need to fill NaNs for the PCA to work. A common approach is to fill with the mean.
        pca_df_filled = pca_df.fillna(pca_df.mean())
        
        scaled_data = scaler.fit_transform(pca_df_filled.values.astype(float))
        
        # PCA
        from sklearn.decomposition import PCA
        n_components = min(n_components, scaled_data.shape[1])
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)

        # return zero-rate dataframe (in percent), pca model and scores
        return pca_df.astype(float), pca, principal_components, scaler, df, expiry_df, available_maturities_in_df, holidays_set

    except Exception as e:
        st.error(f"Error in DI-specific run_analysis: {e}")
        return (None,) * 8

# New helper function to calculate TTM based on business days, which is consistent with the `run_analysis` function.
def calculate_futures_ttm(valuation_date, expiry_date, holidays_set):
    """
    Calculates the Time-to-Maturity (TTM) in years for a given contract.
    This TTM is based on the number of business days between the valuation and expiry dates.
    """
    BUSINESS_YEAR_DAYS = 252
    
    # Correction: Exclude valuation date from the count
    bd_range = pd.bdate_range(valuation_date + pd.Timedelta(days=1), pd.Timestamp(expiry_date))
    # Remove holidays from the range
    if holidays_set:
        bd_range = [d for d in bd_range if d.date() not in holidays_set]

    working_days_to_expiry = len(bd_range)
    
    if working_days_to_expiry <= 0:
        return np.nan
        
    t = working_days_to_expiry / BUSINESS_YEAR_DAYS
    return t

def show_pca_analysis_page():
    st.header("PCA Analysis")

    # Check if required data is available
    if 'pca_df' not in st.session_state or st.session_state.pca_df is None:
        st.warning("Please run the analysis first to see results.")
        return
        
    pca_df = st.session_state.pca_df
    pca_model = st.session_state.pca_model
    principal_components = st.session_state.principal_components
    scaler = st.session_state.scaler
    
    # Use st.container() to group related content, which can help with layout
    with st.container():
        st.subheader("PCA Explained Variance")
        explained_variance_df = pd.DataFrame({
            'Component': range(1, len(pca_model.explained_variance_ratio_) + 1),
            'Explained Variance': pca_model.explained_variance_ratio_,
            'Cumulative Explained Variance': np.cumsum(pca_model.explained_variance_ratio_)
        })
        # Use use_container_width=True to make the dataframe responsive
        st.dataframe(explained_variance_df, use_container_width=True)

        # Plot explained variance
        fig, ax = plt.subplots(figsize=(10, 6)) # Adjusted figsize for better display
        ax.bar(explained_variance_df['Component'], explained_variance_df['Explained Variance'], label='Individual')
        ax.plot(explained_variance_df['Component'], explained_variance_df['Cumulative Explained Variance'],
                marker='o', color='red', label='Cumulative')
        ax.set_title('Explained Variance by Principal Component')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_xticks(explained_variance_df['Component'])
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    st.divider() # Use a divider to visually separate sections

    with st.container():
        st.subheader("Principal Components (Eigenvectors)")
        pc_df = pd.DataFrame(pca_model.components_, columns=pca_df.columns,
                             index=[f'PC{i+1}' for i in range(len(pca_model.components_))]).T
        
        st.dataframe(pc_df, use_container_width=True)
        
        num_maturities = len(pc_df.index)
        # Dynamically adjust plot width based on number of maturities to prevent overlap
        fig_width = max(10, num_maturities * 0.7) 
        
        fig, ax = plt.subplots(figsize=(fig_width, 6))
        for i in range(st.session_state.n_components):
            ax.plot(pc_df.index, pc_df[f'PC{i+1}'], label=f'PC{i+1}')
        ax.set_title('Principal Component Curves (Eigenvectors)')
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Weight')
        ax.legend()
        ax.set_xticks(pc_df.index)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        st.pyplot(fig)
        
    st.divider()

    with st.container():
        st.subheader("Principal Component Loadings vs. Standard Maturities")
        pc_loadings_df = pd.DataFrame(pca_model.components_, columns=pca_df.columns, index=[f'PC{i+1}' for i in range(len(pca_model.components_))])
        num_components = len(pc_loadings_df.index)
        num_maturities = len(pc_loadings_df.columns)
        fig_width = max(12, num_maturities * 0.7)
        fig_height = max(6, num_components * 1.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # --- MODIFIED: fmt=".3f" for 3 decimal places ---
        sns.heatmap(pc_loadings_df, annot=True, cmap='viridis', fmt=".3f", ax=ax)
        ax.set_title('PCA Eigenvector Loadings by Maturity')
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Principal Component')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    st.divider()

    with st.container():
        st.subheader("Principal Component Time Series")
        pc_scores_df = pd.DataFrame(principal_components, index=pca_df.index, columns=[f'PC{i+1}' for i in range(st.session_state.n_components)])
        st.dataframe(pc_scores_df, use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # --- MODIFIED: Plot using pc_scores_df.index and pc_scores_df.values to handle gaps ---
        for i in range(st.session_state.n_components):
            # Create a new series for each PC with a continuous datetime index
            # This will have NaNs for missing dates
            reindexed_series = pc_scores_df[f'PC{i+1}'].reindex(pd.date_range(start=pc_scores_df.index.min(), end=pc_scores_df.index.max(), freq='D'))
            ax.plot(reindexed_series.index, reindexed_series.values, marker='o', linestyle='-', label=f'PC{i+1}')
            
        ax.set_title('Principal Components Time Series')
        ax.set_xlabel('Date')
        ax.set_ylabel('Score')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
    st.divider()

    with st.container():
        st.subheader("Curve Reconstruction")

        #MODIFIED: SORT DATES FOR THE SELECTBOX
        # Get all unique dates from the pca_df index and sort them in descending order
        available_dates = sorted(pca_df.index.unique().tolist(), reverse=True)
        
        # Format the sorted dates as strings for the selectbox
        available_date_strs = [d.strftime("%Y-%m-%d") for d in available_dates]
        
        # Use st.selectbox to allow the user to select a date
        selected_date_str = st.selectbox(
            "Select a date for reconstruction",
            options=available_date_strs,
            index=0 # Default to the latest date
        )
        
        if not selected_date_str:
            st.warning("No dates available for reconstruction.")
            return

        reconstruction_date = pd.to_datetime(selected_date_str)
        
        # Get the original zero rate curve for that date
        original_curve = pca_df.loc[reconstruction_date]
        mean_curve = scaler.mean_

        # Reconstruct the curve
        if st.session_state.n_components is not None:
            num_components_to_use = st.slider("Number of components to use for reconstruction", 1, st.session_state.n_components, st.session_state.n_components)
            
            # Get the principal component scores for the selected date
            pca_scores_for_date = principal_components[pca_df.index.get_loc(reconstruction_date)][:num_components_to_use]
            
            # Get the eigenvectors for the components used
            pca_components = pca_model.components_[:num_components_to_use]
            
            # Reconstruct the scaled curve
            reconstructed_scaled_curve = np.dot(pca_scores_for_date, pca_components)
            
            # Add the mean back to get the final reconstructed curve
            reconstructed_curve_values = reconstructed_scaled_curve + mean_curve
            reconstructed_curve = pd.Series(reconstructed_curve_values, index=pca_df.columns)
            
            reconstructed_df = pd.DataFrame({
                'Maturity': pca_df.columns,
                'Original Curve': original_curve.values,
                'Reconstructed Curve': reconstructed_curve_values
            }).set_index('Maturity')

            fig, ax = plt.subplots(figsize=(12, 6)) # Adjusted figsize for better display
            sns.lineplot(data=reconstructed_df, x='Maturity', y='Original Curve', marker='o', label="Original Zero-Rate Curve", ax=ax, markersize=8)
            sns.lineplot(data=reconstructed_df, x='Maturity', y='Reconstructed Curve', marker='x', label=f"Reconstructed Curve ({num_components_to_use} PCs)", ax=ax, markersize=8, linestyle='--')
            ax.set_title(f"Curve Reconstruction for {reconstruction_date.strftime('%Y-%m-%d')}")
            ax.set_xlabel("Maturity (Years)")
            ax.set_ylabel("Zero Rate (%)")
            ax.set_xticks(reconstructed_df.index)
            ax.set_xticklabels(reconstructed_df.index, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
    st.divider()

    # Futures contract reconstruction comparison
    with st.container():
        st.subheader("Futures Contract Reconstruction Comparison")
        
        # Add a radio button to select the graph type
        chart_type = st.radio(
            "Select graph type:",
            ('Bar Chart', 'Line Graph'),
            key='futures_chart_type'
        )

        try:
            # Get the necessary data from session state
            original_yield_df = st.session_state.raw_yield_df.copy()
            expiry_df = st.session_state.expiry_df.copy()
            holidays_set = st.session_state.holidays_set
            available_maturities = st.session_state.available_maturities_in_df
            
            # Filter original yield data for the reconstruction date
            actual_yields = original_yield_df.loc[reconstruction_date, available_maturities]
            
            # Calculate TTM for each available futures contract on the reconstruction date
            futures_ttm = {}
            for mat in available_maturities:
                expiry_date = expiry_df.loc[mat, 'DATE']
                ttm = calculate_futures_ttm(reconstruction_date, pd.to_datetime(expiry_date), holidays_set)
                if not np.isnan(ttm):
                    futures_ttm[mat] = ttm

            if not futures_ttm:
                st.warning("No unexpired futures contracts found for the selected date to perform comparison.")
                return

            # Sort the futures by their TTM
            sorted_futures = sorted(futures_ttm.items(), key=lambda item: item[1])
            futures_labels = [item[0] for item in sorted_futures]
            ttm_values = [item[1] for item in sorted_futures]
            
            # Get the TTMs and zero rates from the interpolated curve for interpolation
            interpolated_maturities = [float(m.replace('Y','')) for m in reconstructed_curve.index]
            interpolated_rates = reconstructed_curve.values
            
            # Reconstruct the curve for the exact futures TTMs
            if len(interpolated_maturities) > 1 and len(set(interpolated_maturities)) > 1:
                interp_func = interp1d(interpolated_maturities, interpolated_rates, kind='linear', fill_value='extrapolate')
                reconstructed_futures_values = interp_func(ttm_values)
            else:
                st.error("Not enough data points in the reconstructed curve for interpolation.")
                return
            
            # Get the actual market values for the same futures contracts
            actual_futures_values = [actual_yields.get(mat, np.nan) for mat in futures_labels]
            
            # Prepare data for plotting
            plot_data = pd.DataFrame({
                'Contract': futures_labels,
                'Actual Value': actual_futures_values,
                'Reconstructed Value': reconstructed_futures_values
            })
            
            # Plot the comparison based on user selection
            fig, ax = plt.subplots(figsize=(12, 7)) # Adjusted figsize for better display
            if chart_type == 'Bar Chart':
                plot_data.plot(x='Contract', y=['Actual Value', 'Reconstructed Value'], kind='bar', ax=ax, width=0.8, color=['dodgerblue', 'tomato'])
            else: # Line Graph
                sns.lineplot(data=plot_data, x='Contract', y='Actual Value', marker='o', ax=ax, label='Actual Value', color='dodgerblue')
                sns.lineplot(data=plot_data, x='Contract', y='Reconstructed Value', marker='x', ax=ax, label='Reconstructed Value', color='tomato', linestyle='--')
                plt.xticks(rotation=45, ha='right')

            ax.set_title(f"Actual vs. Reconstructed Values for Futures Contracts on {reconstruction_date.strftime('%Y-%m-%d')}")
            ax.set_xlabel("Futures Contract")
            ax.set_ylabel("Yield (%)")
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("Comparison Table for All Contracts")
            
            # Create a table with the values for all contracts
            if not plot_data.empty:
                comparison_table = plot_data.rename(columns={
                    'Actual Value': 'Actual Value (%)',
                    'Reconstructed Value': 'Reconstructed Value (%)'
                }).set_index('Contract')
                st.dataframe(comparison_table, use_container_width=True)
            else:
                st.info("No futures contracts available to display in the comparison table.")
            
        except Exception as e:
            st.error(f"An error occurred during futures contract reconstruction and plotting: {e}")
            st.info("Please ensure your 'expiry' data contains a 'DATE' column and that the maturity names match between your yield and expiry data files.")
            

def show_spread_analysis_page():
    """
    Displays the spread analysis graphs.
    """
    st.subheader("Spread Analysis")
    st.write("This page is for spread analysis, which is not yet implemented.")


# Main App Logic
st.set_page_config(layout="wide") # This is a key change to make the app use the full width of the screen.

st.title("Yield Curve PCA Analysis")

st.markdown("This application allows you to perform Principal Component Analysis on yield curve data.")

# --- Session State Initialization ---
# Ensure all session state variables are initialized before use.
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'page' not in st.session_state:
    st.session_state.page = "PCA Analysis"
if 'pca_df' not in st.session_state:
    st.session_state.pca_df = None
if 'reconstruction_date_str' not in st.session_state:
    st.session_state.reconstruction_date_str = None
if 'n_components' not in st.session_state:
    st.session_state.n_components = 3
if 'standard_maturities_years' not in st.session_state:
    st.session_state.standard_maturities_years = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
if 'interpolation_method' not in st.session_state:
    st.session_state.interpolation_method = 'linear'
if 'use_smoothing' not in st.session_state:
    st.session_state.use_smoothing = False
if 'use_geometric_average' not in st.session_state:
    st.session_state.use_geometric_average = False
if 'holidays_set' not in st.session_state:
    st.session_state.holidays_set = set()
if 'raw_yield_df' not in st.session_state:
    st.session_state.raw_yield_df = None
if 'expiry_df' not in st.session_state:
    st.session_state.expiry_df = None
if 'available_maturities_in_df' not in st.session_state:
    st.session_state.available_maturities_in_df = None


# Sidebar Content
st.sidebar.header("Data Upload")
col1, col2 = st.sidebar.columns(2)
with col1:
    yield_file = st.sidebar.file_uploader("Upload Yield Data (CSV)", type=["csv"])
with col2:
    expiry_file = st.sidebar.file_uploader("Upload Expiry Data (CSV)", type=["csv"])
    
holiday_file = st.sidebar.file_uploader("Upload Holidays (CSV, optional)", type=["csv"])

# --- Analysis Parameters ---
st.sidebar.header("Analysis Parameters")

maturities_str = st.sidebar.text_input(
    "Standard Maturities (years, comma-separated)", 
    value=",".join(map(str, st.session_state.standard_maturities_years))
)
try:
    st.session_state.standard_maturities_years = [float(m.strip()) for m in maturities_str.split(',') if m.strip()]
except (ValueError, IndexError):
    st.sidebar.error("Invalid format for maturities. Please use a comma-separated list of numbers.")
    st.session_state.standard_maturities_years = []

st.session_state.interpolation_method = st.sidebar.selectbox(
    "Interpolation Method",
    options=['linear', 'cubic', 'quadratic'],
    help="Select the method for interpolating zero rates to standard maturities."
)

st.session_state.use_smoothing = st.sidebar.checkbox(
    "Apply 3-day moving average smoothing to raw yields"
)

n_maturities = len(st.session_state.standard_maturities_years) if st.session_state.standard_maturities_years else 3
st.session_state.n_components = st.sidebar.slider(
    "Number of Principal Components",
    min_value=1,
    max_value=n_maturities,
    value=min(3, n_maturities),
    help="Select the number of principal components for the analysis."
)

# Date filters for PCA training
st.sidebar.header("Date Filters for PCA Training")
if yield_file:
    # Need to load a temp df to get min/max dates
    temp_df = pd.read_csv(io.StringIO(yield_file.getvalue().decode("utf-8")))
    temp_df.columns = [col.strip().upper() for col in temp_df.columns]
    temp_df['DATE'], _ = robust_date_parser(temp_df, 'DATE')
    temp_df.dropna(subset=['DATE'], inplace=True)
    temp_df.set_index('DATE', inplace=True)
    
    min_date = temp_df.index.min().date()
    # --- MODIFIED: SET END DATE TO TODAY'S DATE ---
    max_date = date.today()
    if temp_df.index.max().date() < max_date:
        max_date = temp_df.index.max().date()
    
    start_date_filter = st.sidebar.date_input("Start Date for PCA", value=min_date)
    end_date_filter = st.sidebar.date_input("End Date for PCA", value=max_date)

else:
    start_date_filter = None
    end_date_filter = None

#Action Buttons

st.sidebar.divider()
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.sidebar.button("Run Analysis", help="Click to perform PCA on the yield curve data."):
        if yield_file is not None and expiry_file is not None:
            (st.session_state.pca_df, st.session_state.pca_model, 
             st.session_state.principal_components, st.session_state.scaler, 
             st.session_state.raw_yield_df, st.session_state.expiry_df, 
             st.session_state.available_maturities_in_df, st.session_state.holidays_set) = run_analysis(
                yield_file, expiry_file, holiday_file, 
                st.session_state.standard_maturities_years, st.session_state.interpolation_method, 
                st.session_state.use_smoothing, st.session_state.n_components, 
                st.session_state.use_geometric_average,
                start_date_filter, end_date_filter
            )
            st.session_state.analysis_run = True
            
            # Set a default reconstruction date if the analysis was successful
            if st.session_state.pca_df is not None and not st.session_state.pca_df.empty:
                st.session_state.reconstruction_date_str = st.session_state.pca_df.index[0].strftime("%Y-%m-%d")
            else:
                st.session_state.analysis_run = False
        else:
            st.warning("Please upload both yield and expiry data files.")
            st.session_state.analysis_run = False
            
with col2:
    if st.sidebar.button("Reset Analysis", help="Click to clear results and start over."):
        # Clear all analysis-related session state variables
        keys_to_clear = [
            'pca_df', 'pca_model', 'principal_components', 'scaler', 'analysis_run',
            'reconstruction_date_str', 'raw_yield_df', 'expiry_df', 'holidays_set',
            'available_maturities_in_df'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun() # Rerun to clear the screen

# The main page content
if st.session_state.analysis_run:
    st.divider()
    page_selection = st.radio("Select Analysis View", ["PCA Analysis", "Spread Analysis"], index=0)
    st.session_state.page = page_selection
    
    if st.session_state.page == "PCA Analysis":
        show_pca_analysis_page()
    elif st.session_state.page == "Spread Analysis":
        show_spread_analysis_page()

