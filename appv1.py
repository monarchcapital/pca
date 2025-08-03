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

    fallback = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
    return fallback, fallback.isnull().sum()

def get_working_days(start_date, end_date, holidays_set):
    """Calculates the number of working days between two dates, excluding custom holidays."""
    dates = pd.date_range(start=start_date, end=end_date)
    is_business_day = dates.weekday < 5
    is_not_holiday = ~pd.Series(dates.date).isin(holidays_set).values
    return len(dates[is_business_day & is_not_holiday])

def run_analysis(yield_data, expiry_data, holiday_data, standard_maturities_years, interpolation_method, use_smoothing, n_components, use_geometric_average, start_date_filter, end_date_filter):
    """
    This function encapsulates the entire data processing and analysis pipeline.
    It takes all user inputs and performs the PCA.
    """
    try:
        df = pd.read_csv(io.StringIO(yield_data.getvalue().decode("utf-8")))
        df.columns = [col.strip().upper() for col in df.columns]
        
        expiry_df = pd.read_csv(io.StringIO(expiry_data.getvalue().decode("utf-8")))
        expiry_df.columns = [col.strip().upper() for col in expiry_df.columns]

        if 'DATE' not in df.columns:
            st.error("The yield data file must contain a 'DATE' column.")
            return None, None, None, None, None, None, None, None
        if 'MATURITY' not in expiry_df.columns or 'DATE' not in expiry_df.columns:
            st.error("The expiry data file must contain 'MATURITY' and 'DATE' columns.")
            return None, None, None, None, None, None, None, None

        st.sidebar.subheader("Date Parsing Status")
        df_parsed_dates, yield_date_errors = robust_date_parser(df, 'DATE')
        expiry_df_parsed_dates, expiry_date_errors = robust_date_parser(expiry_df, 'DATE')
        
        df['DATE'] = df_parsed_dates
        expiry_df['DATE'] = expiry_df_parsed_dates
        
        if yield_date_errors > 0:
            st.sidebar.warning(f"Found {yield_date_errors} invalid dates in the **yield data**. These rows will be dropped.")
        else:
            st.sidebar.success("All dates in yield data parsed successfully!")
        
        if expiry_date_errors > 0:
            st.sidebar.warning(f"Found {expiry_date_errors} invalid dates in the **expiry data**. These contracts will be ignored.")
        else:
            st.sidebar.success("All dates in expiry data parsed successfully!")
        
        df.dropna(subset=['DATE'], inplace=True)
        expiry_df.dropna(subset=['DATE'], inplace=True)

        if df.empty:
            st.error("The yield data is empty after removing rows with invalid dates. Please check your data file.")
            return None, None, None, None, None, None, None, None
        if expiry_df.empty:
            st.error("The expiry data is empty after removing rows with invalid dates. Please check your data file.")
            return None, None, None, None, None, None, None, None

        df.set_index('DATE', inplace=True)
        expiry_df.set_index('MATURITY', inplace=True)
        df = df.loc[:,~df.columns.duplicated()]

        yield_columns_to_convert = [col for col in df.columns if col != 'DATE']
        for col in yield_columns_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        holidays_set = set()
        if holiday_data:
            try:
                holiday_df = pd.read_csv(io.StringIO(holiday_data.getvalue().decode("utf-8")))
                holiday_df.columns = [col.strip().upper() for col in holiday_df.columns]
                if 'DATE' not in holiday_df.columns:
                    st.error("The holiday file must contain a 'DATE' column.")
                    return None, None, None, None, None, None, None, None
                holiday_parsed_dates, holiday_date_errors = robust_date_parser(holiday_df, 'DATE')
                if holiday_date_errors > 0:
                    st.warning(f"Found {holiday_date_errors} invalid dates in the **holiday data**. These will be ignored.")
                
                holidays_set = set(holiday_parsed_dates.dropna().dt.date)
                st.success(f"Successfully loaded {len(holidays_set)} custom holidays.")
            except Exception as e:
                st.error(f"Error loading holiday file: {e}")
                return None, None, None, None, None, None, None, None
        
        if df.index.empty:
            st.error("The date column in your yield data is invalid or empty. Please check the format.")
            return None, None, None, None, None, None, None, None
        
        # Apply user-selected date range filter
        df_filtered_by_range = df.loc[(df.index.date >= start_date_filter) & (df.index.date <= end_date_filter)]
        df = df_filtered_by_range.copy()

        if df.empty:
            st.error("No data found within the selected date range after parsing. Please adjust the dates or check your data.")
            return None, None, None, None, None, None, None, None

        is_business_day = df.index.weekday < 5
        is_not_holiday = ~pd.Series(df.index.date).isin(holidays_set).values
        
        df_filtered_dates = df[is_business_day & is_not_holiday]
        df = df_filtered_dates
        if df.empty:
            st.error("The selected date range contains no business days after filtering weekends and custom holidays.")
            return None, None, None, None, None, None, None, None
            
        if use_smoothing:
            df = df.rolling(window=3, min_periods=1, center=True).mean()
            st.info("A 3-day rolling mean smoothing has been applied to the data to reduce noise.")

        pca_df = pd.DataFrame(index=df.index, columns=[f'{m:.2f}Y' for m in standard_maturities_years])
        BUSINESS_YEAR_DAYS = 252
        
        progress_bar = st.progress(0)
        
        available_maturities_in_df = set(df.columns).intersection(set(expiry_df.index))
        if not available_maturities_in_df:
            st.error("The maturity columns in your yield data do not match the maturity index in your expiry data. Please check and correct your files.")
            return None, None, None, None, None, None, None, None
        st.info(f"Successfully matched {len(available_maturities_in_df)} maturities for processing.")

        dates_skipped_count = 0
        for i, date in enumerate(df.index):
            ttm_years = []
            yield_values = []
            
            for col in available_maturities_in_df:
                expiry_date_val = expiry_df.loc[col, 'DATE']
                if pd.isnull(expiry_date_val):
                    continue
                
                expiry_date_val_dt = pd.to_datetime(expiry_date_val).date() 
                if expiry_date_val_dt >= date.date():
                    working_days_to_expiry = get_working_days(date, expiry_date_val_dt, holidays_set)
                else:
                    working_days_to_expiry = 0

                if working_days_to_expiry > 0 and pd.notnull(df.loc[date, col]):
                    ttm = working_days_to_expiry / BUSINESS_YEAR_DAYS
                    raw_yield = df.loc[date, col]
                    
                    if use_geometric_average:
                        if raw_yield >= -100:
                            yield_value = 100 * ((1 + raw_yield / 100)**ttm - 1)
                        else:
                            yield_value = np.nan
                    else:
                        yield_value = raw_yield
                        
                    if not np.isnan(yield_value):
                        ttm_years.append(ttm)
                        yield_values.append(yield_value)

            if len(ttm_years) > 1 and len(set(ttm_years)) > 1:
                sorted_points = sorted(zip(ttm_years, yield_values))
                ttm_years_sorted, yield_values_sorted = zip(*sorted_points)
                
                f_interp = interp1d(ttm_years_sorted, yield_values_sorted, kind=interpolation_method, fill_value="extrapolate")
                interpolated_yields = f_interp(standard_maturities_years)
                
                pca_df.loc[date] = interpolated_yields
            else:
                dates_skipped_count += 1
                
            progress_bar.progress((i + 1) / len(df.index))
    
        st.info(f"Skipped {dates_skipped_count} dates due to insufficient data for interpolation.")
        progress_bar.empty()
    
        pca_df.dropna(how='all', inplace=True)
        if pca_df.empty:
            st.error("The interpolated data is empty. This may be due to missing data or an error in the interpolation process.")
            return None, None, None, None, None, None, None, None
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_df)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)

        return pca_df, pca, principal_components, scaler, df, expiry_df, available_maturities_in_df, holidays_set

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}. Please check your data files and parameters.")
        return None, None, None, None, None, None, None, None

def reconstruct_curve_for_date(date_to_reconstruct, pca_df, pca, scaler, n_components):
    """
    Reconstructs the yield curve for a given date using the PCA model.
    Returns a pandas Series of the reconstructed yields.
    """
    selected_pc_scores = pca.transform(scaler.transform(pca_df.loc[[date_to_reconstruct]]))[0, :n_components]
    components_to_use = pca.components_[:n_components, :]
    reconstructed_curve_scaled = np.dot(selected_pc_scores, components_to_use)
    reconstructed_curve = (reconstructed_curve_scaled * scaler.scale_) + scaler.mean_
    return pd.Series(reconstructed_curve, index=pca_df.columns)

def show_pca_analysis_page():
    """
    Displays all the original PCA analysis plots and tables.
    """
    pca_df = st.session_state.pca_df
    pca = st.session_state.pca_model
    principal_components = st.session_state.principal_components
    scaler = st.session_state.scaler
    n_components = st.session_state.n_components
    actual_n_components = len(pca.explained_variance_ratio_)
    num_maturities = len(pca_df.columns)

    if n_components == num_maturities:
        st.warning(
            f"The 'Reconstructed' curve is a perfect match to the 'Interpolated' curve "
            f"because you have selected **{n_components}** principal components, which is "
            f"equal to the number of maturities in the dataset. To see the smoothing "
            f"effect of the PCA, please reduce the number of components using the slider on the left."
        )

    st.header("1. Interpolated Data Preview")
    st.write("First 5 rows of the interpolated data used for PCA.")
    if st.session_state.use_geometric_average:
        st.info("The yields in the table and plots below have been calculated using the geometric average formula.")
    st.dataframe(pca_df.head())

    st.header("2. Explained Variance")
    st.subheader("Explained Variance Ratios and Scree Plot")
    variance_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(actual_n_components)],
        'Explained Variance Ratio': pca.explained_variance_ratio_
    })
    variance_df['Cumulative Explained Variance'] = variance_df['Explained Variance Ratio'].cumsum()
    st.write("This table shows the proportion of the dataset's variance that each principal component accounts for.")
    st.dataframe(variance_df.style.format({'Explained Variance Ratio': '{:.4f}', 'Cumulative Explained Variance': '{:.4f}'}))
    
    st.write(f"**Total explained variance by the first {actual_n_components} components:** {pca.explained_variance_ratio_.sum():.2%}")
    
    fig_scree, ax_scree = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Principal Component', y='Explained Variance Ratio', data=variance_df, ax=ax_scree, color='skyblue')
    ax_scree.plot(variance_df['Explained Variance Ratio'].values, 'o-', color='black')
    ax_scree.set_title('Scree Plot')
    ax_scree.set_xlabel('Principal Component')
    ax_scree.set_ylabel('Explained Variance Ratio')
    st.pyplot(fig_scree)

    st.header("3. Principal Components Over Time")
    st.write("This plot shows how the scores of each principal component (level, slope, curvature) change over time.")
    principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(actual_n_components)], index=pca_df.index)
    fig_time_series, ax_time_series = plt.subplots(figsize=(15, 8))
    for i in range(actual_n_components):
        ax_time_series.plot(principal_df.index, principal_df[f'PC{i+1}'], label=f'PC{i+1}')
    
    ax_time_series.set_title('Principal Components Over Time')
    ax_time_series.set_xlabel('Date')
    ax_time_series.set_ylabel('Principal Component Value')
    ax_time_series.legend()
    ax_time_series.grid(True)
    st.pyplot(fig_time_series)
    
    st.header("4. Yield Curve Reconstruction (Single Date)")
    st.write("Visualize the interpolated vs. reconstructed yield curve for a selected date. The reconstructed curve represents the fair value based on the PCA model's principal components.")
    st.info("The reconstructed curve is an approximation using only the dominant principal components, providing a smoothed 'fair value' representation.")

    date_options = pca_df.index.tolist()
    
    default_index_for_selectbox = 0
    if st.session_state.reconstruction_date_str:
        try:
            last_selected_date = pd.to_datetime(st.session_state.reconstruction_date_str)
            default_index_for_selectbox = date_options.index(last_selected_date)
        except ValueError:
            default_index_for_selectbox = 0

    reconstruction_date_selected = st.selectbox(
        "Select a date to reconstruct the yield curve",
        options=date_options,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        index=default_index_for_selectbox
    )
    st.session_state.reconstruction_date_str = reconstruction_date_selected.strftime("%Y-%m-%d")

    interpolated_curve = pca_df.loc[reconstruction_date_selected]
    reconstructed_curve = reconstruct_curve_for_date(reconstruction_date_selected, pca_df, pca, scaler, n_components)

    fig_curves, ax_curves = plt.subplots(figsize=(15, 8))
    ax_curves.plot(interpolated_curve.index.astype(str), interpolated_curve.values, 'o-', label='Interpolated (Actual) Yield Curve', color='blue')
    ax_curves.plot(reconstructed_curve.index.astype(str), reconstructed_curve.values, 'x--', label='Reconstructed (Fair Value) Yield Curve', color='red')
    
    ax_curves.set_title(f'Yield Curve for {reconstruction_date_selected.strftime("%Y-%m-%d")}')
    ax_curves.set_xlabel('Maturity (in years)')
    ax_curves.set_ylabel('Yield')
    ax_curves.legend()
    ax_curves.grid(True)
    st.pyplot(fig_curves)
    
    difference = interpolated_curve.astype(float) - reconstructed_curve.astype(float)
    max_diff = np.max(np.abs(difference.values))
    if max_diff < 1e-6:
        st.info(f"The maximum absolute difference between the curves is approximately zero, at {max_diff:.8f}. This is likely due to the first {n_components} components explaining virtually all the variance in your dataset.")
    else:
        st.info(f"The maximum absolute difference between the curves is {max_diff:.4f}.")


    st.header("5. Difference between Interpolated and Reconstructed Curves (Single Date)")
    st.write(
        "This plot isolates the small difference between the interpolated and reconstructed "
        "curves, allowing you to visually inspect the noise removed by the PCA model."
    )

    difference_df = pd.DataFrame({
        'Maturity': pca_df.columns,
        'Difference (Interpolated - Reconstructed)': difference.values
    }).set_index('Maturity')
    
    fig_diff, ax_diff = plt.subplots(figsize=(15, 8))
    sns.barplot(
        x=difference_df.index,
        y=difference_df['Difference (Interpolated - Reconstructed)'],
        ax=ax_diff,
        palette='viridis'
    )
    ax_diff.axhline(0, color='black', linestyle='--', linewidth=1)
    ax_diff.set_title(f'Difference (Interpolated - Reconstructed) for {reconstruction_date_selected.strftime("%Y-%m-%d")}')
    ax_diff.set_xlabel('Maturity (in years)')
    ax_diff.set_ylabel('Yield Difference (Percentage Points)')
    ax_diff.grid(True)
    st.pyplot(fig_diff)

    st.header("6. Multi-Date Difference Analysis")
    st.write(
        "Select multiple dates to compare the differences between the interpolated and "
        "reconstructed curves in a single table."
    )
    st.info("The values in the table below represent the difference in **percentage points**.")
    multi_date_options = pca_df.index.tolist()
    multi_date_selections = st.multiselect(
        "Select dates to compare",
        options=multi_date_options,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        default=st.session_state.multi_date_selections
    )
    st.session_state.multi_date_selections = multi_date_selections

    if multi_date_selections:
        multi_date_diff_data = {}
        for date_to_compare in multi_date_selections:
            interpolated_curve = pca_df.loc[date_to_compare]
            reconstructed_curve = reconstruct_curve_for_date(date_to_compare, pca_df, pca, scaler, n_components)
            difference = interpolated_curve.astype(float) - reconstructed_curve
            multi_date_diff_data[date_to_compare.strftime("%Y-%m-%d")] = difference

        diff_table = pd.DataFrame(multi_date_diff_data)
        diff_table.index = pca_df.columns
        st.dataframe(diff_table.style.format('{:.4f}'))
    else:
        st.info("Select at least one date from the dropdown to see the difference table.")


    st.header("7. Eigenvectors and Eigenvalues")
    st.markdown(
        """
        This section provides a direct view of the underlying mathematics of the PCA model.
        The eigenvectors define the new axes (the principal components), and the eigenvalues represent
        the variance explained along those axes.
        """
    )
    
    st.subheader("Eigenvector Matrix (Component Loadings)")
    loadings_df = pd.DataFrame(
        pca.components_,
        columns=pca_df.columns,
        index=[f'PC{i+1}' for i in range(actual_n_components)]
    )
    st.write("This matrix shows the weights (loadings) for each maturity on each principal component.")
    st.dataframe(loadings_df.T)
    
    st.subheader("Principal Component Loadings Heatmap")
    st.write("A visual representation of the eigenvector matrix, highlighting the positive and negative weights.")
    fig_loadings_heatmap, ax_loadings_heatmap = plt.subplots(figsize=(12, 8))
    sns.heatmap(loadings_df, cmap='viridis', annot=True, fmt=".2f", ax=ax_loadings_heatmap)
    ax_loadings_heatmap.set_title('PCA Component Loadings Heatmap')
    ax_loadings_heatmap.set_xlabel('Maturities (in years)')
    ax_loadings_heatmap.set_ylabel('Principal Components')
    st.pyplot(fig_loadings_heatmap)
    
    st.subheader("Eigenvalues")
    eigenvalues_df = pd.DataFrame({
        'Principal Component': [f'PC{i+1}' for i in range(actual_n_components)],
        'Eigenvalue (Explained Variance)': pca.explained_variance_
    })
    st.write("The eigenvalues show the magnitude of variance captured by each principal component.")
    st.dataframe(eigenvalues_df)

def show_spread_analysis_page():
    """
    Displays the spread analysis page.
    """
    st.title("Historical Spread Analysis")
    st.write("Compare the historical spread between two maturities for both the actual and reconstructed yield curves.")
    
    if st.session_state.pca_df is None or st.session_state.pca_df.empty:
        st.warning("Please run the PCA analysis on the 'PCA Analysis' page first.")
        return
    
    pca_df = st.session_state.pca_df
    maturities = pca_df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        maturity_1 = st.selectbox("Select Maturity 1", options=maturities, index=0)
    with col2:
        # Set the default index to the next item if available, otherwise stay at 0
        default_index_2 = 1 if len(maturities) > 1 else 0
        maturity_2 = st.selectbox("Select Maturity 2", options=maturities, index=default_index_2)

    if maturity_1 == maturity_2:
        st.warning("Please select two different maturities to calculate the spread.")
        return

    st.subheader(f"Historical Spread: {maturity_2} - {maturity_1}")

    # Calculate Actual Spread
    actual_spread = pca_df[maturity_2].astype(float) - pca_df[maturity_1].astype(float)

    # Calculate all reconstructed curves at once for better performance
    with st.spinner("Calculating reconstructed spreads..."):
        pca = st.session_state.pca_model
        scaler = st.session_state.scaler
        n_components = st.session_state.n_components
        
        scaled_data = scaler.transform(pca_df)
        principal_components_all = pca.transform(scaled_data)[:, :n_components]
        components_to_use = pca.components_[:n_components, :]
        reconstructed_curves_scaled = np.dot(principal_components_all, components_to_use)
        reconstructed_curves_values = (reconstructed_curves_scaled * scaler.scale_) + scaler.mean_
        
        reconstructed_df = pd.DataFrame(reconstructed_curves_values, index=pca_df.index, columns=pca_df.columns)
        
    reconstructed_yield_m1 = reconstructed_df[maturity_1].astype(float)
    reconstructed_yield_m2 = reconstructed_df[maturity_2].astype(float)
    reconstructed_spread = reconstructed_yield_m2 - reconstructed_yield_m1

    # Combine into a single DataFrame for display and plotting
    spread_df = pd.DataFrame({
        f'Actual Yield {maturity_1}': pca_df[maturity_1].astype(float),
        f'Actual Yield {maturity_2}': pca_df[maturity_2].astype(float),
        'Actual Spread': actual_spread,
        f'Fair Value Yield {maturity_1}': reconstructed_yield_m1,
        f'Fair Value Yield {maturity_2}': reconstructed_yield_m2,
        'Fair Value Spread': reconstructed_spread
    })
    
    st.info("The table below shows the historical yields and spreads, measured in percentage points.")
    st.dataframe(spread_df.style.format('{:.4f}'))

    # Plot the spreads
    fig_spread, ax_spread = plt.subplots(figsize=(15, 8))
    # Convert index to string for plotting to avoid ConversionError
    x_dates_str = spread_df.index.strftime('%Y-%m-%d')
    ax_spread.plot(x_dates_str, spread_df['Actual Spread'], label='Actual Spread', color='blue', linestyle='-')
    ax_spread.plot(x_dates_str, spread_df['Fair Value Spread'], label='Fair Value Spread', color='red', linestyle='--')
    ax_spread.set_title(f'Historical Spread for {maturity_2} - {maturity_1}')
    ax_spread.set_xlabel('Date')
    ax_spread.set_ylabel('Spread (Percentage Points)')
    ax_spread.legend()
    ax_spread.grid(True)
    ax_spread.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_spread)

def main():
    """
    Main function to run the Streamlit PCA application.
    """
    st.set_page_config(layout="wide", page_title="Yield Curve PCA App")
    st.title("Yield Curve Principal Component Analysis (PCA) App")
    st.markdown(
        """
        This application performs Principal Component Analysis on financial yield curve data.
        Upload your data files and click "Run Analysis" to get started.
        """
    )
    
    # --- Initialize Session State ---
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
        st.session_state.pca_df = None
        st.session_state.pca_model = None
        st.session_state.principal_components = None
        st.session_state.scaler = None
        st.session_state.yield_data_raw = None
        st.session_state.expiry_data_raw = None
        st.session_state.holiday_data_raw = None
        st.session_state.interpolation_method = 'cubic'
        st.session_state.n_components = 3
        st.session_state.standard_maturities_years = "0.25, 0.5, 1, 2, 3, 5, 7"
        st.session_state.use_smoothing = False
        st.session_state.use_geometric_average = False
        st.session_state.start_date_filter = None
        st.session_state.end_date_filter = None
        st.session_state.reconstruction_date_str = None
        st.session_state.multi_date_selections = []
        st.session_state.page = "PCA Analysis" # Add a state for the current page

    st.sidebar.header("User Input Panel")
    st.sidebar.markdown("---")
    
    # --- Page Navigation ---
    st.session_state.page = st.sidebar.radio("Navigation", ["PCA Analysis", "Spread Analysis"])
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader("Upload your yield data CSV file", type=["csv"], key="yield_file")
    expiry_file = st.sidebar.file_uploader("Upload contracts expiry data CSV file", type=["csv"], key="expiry_file")
    holiday_file = st.sidebar.file_uploader("Upload a holiday data CSV file (optional)", type=["csv"], key="holiday_file")

    if uploaded_file:
        st.session_state.yield_data_raw = uploaded_file
        
        try:
            df_temp = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
            df_temp.columns = [col.strip().upper() for col in df_temp.columns]
            if 'DATE' in df_temp.columns:
                parsed_dates, _ = robust_date_parser(df_temp, 'DATE')
                valid_dates = parsed_dates.dropna()
                if not valid_dates.empty:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                    
                    if st.session_state.start_date_filter is None:
                        st.session_state.start_date_filter = min_date
                    if st.session_state.end_date_filter is None:
                        st.session_state.end_date_filter = max_date
                    
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("Analysis Date Range Filter")
                    st.sidebar.info(f"Available dates: {min_date} to {max_date}")
                    selected_start_date = st.sidebar.date_input(
                        "Start Date",
                        value=st.session_state.start_date_filter,
                        min_value=min_date,
                        max_value=max_date
                    )
                    selected_end_date = st.sidebar.date_input(
                        "End Date",
                        value=st.session_state.end_date_filter,
                        min_value=min_date,
                        max_value=max_date
                    )
                    
                    st.session_state.start_date_filter = selected_start_date
                    st.session_state.end_date_filter = selected_end_date
                    
                    if selected_start_date > selected_end_date:
                        st.sidebar.error("Start date must be before or the same as the end date.")
                        st.stop()
                    st.sidebar.markdown("---")
            else:
                st.sidebar.error("The uploaded file does not contain a 'DATE' column.")
        except Exception as e:
            st.sidebar.error(f"Error processing the date column: {e}")
            st.stop()
    else:
        st.sidebar.info("Please upload a yield data CSV to enable the date range filter.")

    st.sidebar.subheader("Data & Analysis Options")
    use_smoothing = st.sidebar.checkbox("Apply 3-day rolling mean smoothing", value=st.session_state.use_smoothing)
    interpolation_method = st.sidebar.selectbox(
        "Select Interpolation Method",
        options=['cubic', 'linear'],
        index=['cubic', 'linear'].index(st.session_state.interpolation_method)
    )
    
    use_geometric_average = st.sidebar.checkbox(
        "Use Geometric Average for Yields",
        value=st.session_state.use_geometric_average,
        help="Applies the formula ( (1 + di/100)^(workingdays/252) ) - 1) * 100 to the raw yields. This is relevant for markets like Brazil."
    )
    
    standard_maturities_years = st.sidebar.text_input(
        "Standard Maturities (in working years, comma-separated)",
        value=st.session_state.standard_maturities_years
    )
    try:
        standard_maturities_years_list = [float(m.strip()) for m in standard_maturities_years.split(',') if m.strip()]
        if not standard_maturities_years_list:
            st.sidebar.warning("Please enter at least one standard maturity.")
            st.stop()
    except ValueError:
        st.sidebar.error("Please ensure the standard maturities are a comma-separated list of numbers.")
        st.stop()

    n_components = st.sidebar.slider(
        "Select number of components",
        min_value=1,
        max_value=len(standard_maturities_years_list),
        value=min(st.session_state.n_components, len(standard_maturities_years_list))
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Run Analysis", help="Click to start or re-run the analysis with current settings."):
            if uploaded_file and expiry_file and standard_maturities_years_list:
                st.session_state.interpolation_method = interpolation_method
                st.session_state.use_smoothing = use_smoothing
                st.session_state.use_geometric_average = use_geometric_average
                st.session_state.n_components = n_components
                st.session_state.standard_maturities_years = standard_maturities_years
                
                st.session_state.pca_df, st.session_state.pca_model, st.session_state.principal_components, st.session_state.scaler, _, _, _, _ = run_analysis(
                    uploaded_file, expiry_file, holiday_file, standard_maturities_years_list, interpolation_method, use_smoothing, n_components, use_geometric_average, st.session_state.start_date_filter, st.session_state.end_date_filter
                )
                st.session_state.analysis_run = True

                if st.session_state.pca_df is None or st.session_state.pca_df.empty:
                    st.session_state.analysis_run = False
                else:
                    if st.session_state.reconstruction_date_str is None and not st.session_state.pca_df.empty:
                        st.session_state.reconstruction_date_str = st.session_state.pca_df.index[0].strftime("%Y-%m-%d")
                    st.session_state.multi_date_selections = []
            else:
                st.warning("Please upload both yield and expiry data files and specify maturities before running the analysis.")
                st.session_state.analysis_run = False
                
    with col2:
        if st.button("Reset Analysis", help="Click to clear results and start over."):
            st.session_state.clear()
            st.rerun()

    if st.session_state.analysis_run:
        if st.session_state.page == "PCA Analysis":
            show_pca_analysis_page()
        elif st.session_state.page == "Spread Analysis":
            show_spread_analysis_page()
    else:
        st.info("Please upload your yield and expiry data CSV files, select your date range, and click 'Run Analysis' to get started.")

if __name__ == "__main__":
    main()
