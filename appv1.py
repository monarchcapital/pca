import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """
   PCA application.
    """
    st.set_page_config(layout="wide", page_title="Yield Curve PCA App")
    st.title("Yield Curve Principal Component Analysis (PCA) App")
    st.markdown(
        """
       
        Upload a CSV file, select date range and maturities.
        """
    )

    # Sidebar for user inputs
    st.sidebar.header("User Input Panel")
    uploaded_file = st.sidebar.file_uploader("Upload your data CSV file", type=["csv"])

    if uploaded_file:
        # Load data and preprocess
        try:
            df = pd.read_csv(uploaded_file)
            # Use dayfirst=True for more robust date parsing
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df.set_index('Date', inplace=True)
            st.sidebar.success("Data file uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading the data file: {e}")
            return

        # Get initial data count for user feedback
        initial_rows = len(df)
        
        #Filter out weekends (Saturdays and Sundays)
        # Monday is 0 and Sunday is 6, so we keep only dayofweek < 5
        df = df[df.index.dayofweek < 5]
        filtered_rows_weekends = initial_rows - len(df)
        st.sidebar.info(f"Weekends have been automatically filtered out. Removed {filtered_rows_weekends} rows.")

        # Optional holiday filter
        holiday_file = st.sidebar.file_uploader("Optional: Upload a CSV with holiday dates", type=["csv"])
        if holiday_file:
            try:
                # Read holidays.csv with no header, assigning a single "Date" column.
                # Use dayfirst=True for consistent date parsing.
                holidays_df = pd.read_csv(holiday_file, header=None, names=['Date'])
                holidays_df['Date'] = pd.to_datetime(holidays_df['Date'], dayfirst=True, errors='coerce')
                holiday_dates = holidays_df['Date'].tolist()
                
                initial_rows_after_weekends = len(df)
                df = df[~df.index.isin(holiday_dates)]
                filtered_rows_holidays = initial_rows_after_weekends - len(df)
                
                st.sidebar.success("Holidays filtered successfully!")
                st.sidebar.info(f"Removed {filtered_rows_holidays} rows corresponding to holidays.")
            except Exception as e:
                st.sidebar.error(f"Error loading or processing the holiday file: {e}")
        
        # Sort the index to ensure that date range slicing works correctly.
        df.sort_index(inplace=True)

        # Check if the dataframe is empty after all initial filtering
        if df.empty:
            st.error("The dataset became empty after filtering out weekends and holidays. Please check your data file.")
            return

        # Debugging section to show what data is left after filtering
        with st.sidebar.expander("Debugging Filtered Data"):
            st.write(f"Rows remaining after all filters: **{len(df)}**")
            st.write(f"Date range of remaining data: **{df.index.min().date()}** to **{df.index.max().date()}**")
            st.dataframe(df.head(5))
            st.dataframe(df.tail(5))


        st.sidebar.subheader("Data Selection")

        # Get the range of dates from the DataFrame
        min_date = df.index.min()
        max_date = df.index.max()

        # Date range slider
        date_range = st.sidebar.slider(
            "Select date range",
            min_value=min_date.date(),
            max_value=max_date.date(),
            value=(min_date.date(), max_date.date())
        )
        
        # Filter data based on user selected date range
        start_date_filter, end_date_filter = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered_by_date = df.loc[start_date_filter:end_date_filter]

        # Check if the date range filter resulted in an empty dataframe
        if df_filtered_by_date.empty:
            st.error("The selected date range contains no data after filtering out weekends and holidays. Please adjust your date range selection.")
            return
        
        # Maturity selection
        maturity_columns = df.columns.tolist()
        
        # Filter out any maturities that are entirely NaN within the selected date range
        valid_maturities = [col for col in maturity_columns if not df_filtered_by_date[col].isnull().all()]
        selected_maturities = st.sidebar.multiselect(
            "Select maturities for analysis",
            options=valid_maturities,
            default=valid_maturities[:min(len(valid_maturities), 10)]
        )
        
        if not valid_maturities:
            st.error("No maturities have data within the selected date range. Please adjust the date range slider or check your data file.")
            return

        # Handle missing values option
        missing_data_strategy = st.sidebar.radio(
            "How to handle missing values (NaNs)?",
            ("Drop rows", "Fill with previous value")
        )

        if not selected_maturities:
            st.sidebar.warning("Please select at least one maturity to proceed.")
            return

        # Handle the number of components
        if len(selected_maturities) == 1:
            n_components = 1
            st.sidebar.info("Only one maturity selected, so only one principal component will be calculated.")
        else:
            max_components = len(selected_maturities)
            n_components = st.sidebar.slider(
                "Select number of components",
                min_value=1,
                max_value=max_components,
                value=min(3, max_components)
            )

        #Main content area for analysis
        data_filtered = df.loc[start_date_filter:end_date_filter, selected_maturities]
        
        rows_before_handling = len(data_filtered)
        
        if missing_data_strategy == "Drop rows":
            data_filtered.dropna(inplace=True)
            rows_after_handling = len(data_filtered)
            st.info(f"Loaded {rows_after_handling} rows for analysis. {rows_before_handling - rows_after_handling} rows were removed due to missing values.")
        else: # "Fill with previous value"
            data_filtered.dropna(how='all', inplace=True)
            data_filtered.ffill(inplace=True)
            rows_after_handling = len(data_filtered)
            st.info(f"Loaded {rows_after_handling} rows for analysis. Missing values were filled with the previous day's data.")
        
        if data_filtered.empty:
            st.error("The selected date range or maturities resulted in an empty dataset. Please adjust your selections.")
            return
        
        # Standardize the data before PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_filtered)

        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)

        #NEW: Navigation for different pages
        st.sidebar.header("Navigation")
        page = st.sidebar.radio("Go to", ["Analysis", "Next Day Forecast"])

        if page == "Analysis":
            st.header("1. Raw Data Preview")
            st.write("Displaying the first 5 rows of the loaded data.")
            st.dataframe(df.head())
            
            st.header("2. PCA Results")
            st.subheader("Explained Variance")
            variance_df = pd.DataFrame({
                'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                'Explained Variance Ratio': pca.explained_variance_ratio_
            })
            variance_df['Cumulative Explained Variance'] = variance_df['Explained Variance Ratio'].cumsum()
            st.dataframe(variance_df.style.format({'Explained Variance Ratio': '{:.4f}', 'Cumulative Explained Variance': '{:.4f}'}))
            
            st.write(f"**Total explained variance by the first {n_components} components:** {pca.explained_variance_ratio_.sum():.2%}")
            
            st.subheader("Scree Plot")
            fig_scree, ax_scree = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Principal Component', y='Explained Variance Ratio', data=variance_df, ax=ax_scree, color='skyblue')
            ax_scree.plot(variance_df['Explained Variance Ratio'].values, 'o-', color='black')
            ax_scree.set_title('Scree Plot')
            ax_scree.set_xlabel('Principal Component')
            ax_scree.set_ylabel('Explained Variance Ratio')
            st.pyplot(fig_scree)

            st.subheader("Component Loadings")
            loadings_df = pd.DataFrame(pca.components_, columns=selected_maturities, index=[f'PC{i+1}' for i in range(n_components)])
            st.write("Loadings of each maturity on the principal components.")
            st.dataframe(loadings_df.T.style.background_gradient(cmap='viridis'))

            fig_loadings, ax_loadings = plt.subplots(figsize=(12, 8))
            sns.heatmap(loadings_df, cmap='viridis', annot=True, fmt=".2f", ax=ax_loadings)
            ax_loadings.set_title('PCA Component Loadings Heatmap')
            ax_loadings.set_xlabel('Maturities')
            ax_loadings.set_ylabel('Principal Components')
            st.pyplot(fig_loadings)

            st.subheader("Principal Components Over Time")
            principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)], index=data_filtered.index)
            fig_time_series, ax_time_series = plt.subplots(figsize=(15, 8))
            for i in range(n_components):
                ax_time_series.plot(principal_df.index, principal_df[f'PC{i+1}'], label=f'PC{i+1}')
            
            ax_time_series.set_title('Principal Components Over Time')
            ax_time_series.set_xlabel('Date')
            ax_time_series.set_ylabel('Principal Component Value')
            ax_time_series.legend()
            ax_time_series.grid(True)
            st.pyplot(fig_time_series)
            
            st.header("3. Yield Curve Reconstruction & Fair Value")
            st.write("Visualize the actual vs. reconstructed yield curve for a selected date. The reconstructed curve represents the fair value based on the PCA model's principal components.")

            reconstruction_date = st.selectbox(
                "Select a date to reconstruct the yield curve",
                options=data_filtered.index.tolist(),
                format_func=lambda x: x.strftime("%Y-%m-%d")
            )
            
            selected_date_index = data_filtered.index.get_loc(reconstruction_date)
            selected_pc_scores = principal_components[selected_date_index, :]
            mean_data = scaler.mean_
            reconstructed_curve = mean_data + np.dot(pca.components_.T, selected_pc_scores.T)
            
            reconstructed_df = pd.DataFrame(
                {'Maturity': selected_maturities, 'Yield': reconstructed_curve}
            ).set_index('Maturity')

            actual_curve = data_filtered.loc[reconstruction_date]

            fig_curves, ax_curves = plt.subplots(figsize=(15, 8))
            ax_curves.plot(actual_curve.index, actual_curve.values, 'o-', label='Actual Yield Curve', color='blue')
            ax_curves.plot(reconstructed_df.index, reconstructed_df.values, 'x--', label='Reconstructed (Fair Value) Yield Curve', color='red')
            
            ax_curves.set_title(f'Yield Curve for {reconstruction_date.strftime("%Y-%m-%d")}')
            ax_curves.set_xlabel('Maturity')
            ax_curves.set_ylabel('Yield')
            ax_curves.legend()
            ax_curves.grid(True)
            st.pyplot(fig_curves)
            
            st.header("4. Difference between Actual and Reconstructed Curves")
            st.write("This section shows the difference between the actual yield and the reconstructed (fair value) yield for the selected date.")

            difference = actual_curve - reconstructed_df['Yield']
            difference_df = pd.DataFrame({
                'Maturity': selected_maturities,
                'Difference (Actual - Reconstructed)': difference.values
            }).set_index('Maturity')
            st.write("Table of differences:")
            st.dataframe(difference_df)

            fig_diff, ax_diff = plt.subplots(figsize=(15, 8))
            sns.barplot(
                x=difference_df.index,
                y=difference_df['Difference (Actual - Reconstructed)'],
                ax=ax_diff,
                palette='viridis'
            )
            ax_diff.set_title(f'Difference (Actual - Reconstructed) for {reconstruction_date.strftime("%Y-%m-%d")}')
            ax_diff.set_xlabel('Maturity')
            ax_diff.set_ylabel('Yield Difference')
            ax_diff.grid(True)
            st.pyplot(fig_diff)

            st.header("5. Yield Change Curve")
            st.write("This plot shows the day-over-day change in the actual yield curve.")
            yield_changes = data_filtered.diff()

            change_date = st.selectbox(
                "Select a date to view the yield change",
                options=yield_changes.index.tolist(),
                format_func=lambda x: x.strftime("%Y-%m-%d")
            )
            
            change_for_date = yield_changes.loc[change_date]
            fig_change, ax_change = plt.subplots(figsize=(15, 8))
            sns.barplot(
                x=change_for_date.index, 
                y=change_for_date.values, 
                ax=ax_change,
                palette='coolwarm'
            )
            ax_change.set_title(f'Yield Change for {change_date.strftime("%Y-%m-%d")}')
            ax_change.set_xlabel('Maturity')
            ax_change.set_ylabel('Daily Yield Change')
            ax_change.grid(True)
            st.pyplot(fig_change)

            st.header("6. Actual vs. Fair Value Yield Change")
            st.write("This plot compares the day-over-day change in the actual yield curve with the day-over-day change in the fair value yield curve derived from the PCA model.")
            
            reconstructed_data_scaled = np.dot(principal_components, pca.components_)
            reconstructed_data = scaler.inverse_transform(reconstructed_data_scaled)
            reconstructed_df_all_dates = pd.DataFrame(reconstructed_data, columns=selected_maturities, index=data_filtered.index)
            fair_value_yield_changes = reconstructed_df_all_dates.diff()
            
            change_comparison_date = st.selectbox(
                "Select a date to view the yield change comparison",
                options=yield_changes.index.tolist(),
                format_func=lambda x: x.strftime("%Y-%m-%d")
            )
            
            actual_change_for_date = yield_changes.loc[change_comparison_date]
            fair_value_change_for_date = fair_value_yield_changes.loc[change_comparison_date]

            change_comparison_df = pd.DataFrame({
                'Maturity': selected_maturities,
                'Actual Change': actual_change_for_date,
                'Fair Value Change': fair_value_change_for_date
            }).set_index('Maturity')

            fig_change_comp, ax_change_comp = plt.subplots(figsize=(15, 8))
            change_comparison_df.plot(kind='bar', ax=ax_change_comp, rot=45)
            ax_change_comp.set_title(f'Actual vs. Fair Value Yield Change for {change_comparison_date.strftime("%Y-%m-%d")}')
            ax_change_comp.set_xlabel('Maturity')
            ax_change_comp.set_ylabel('Daily Yield Change')
            ax_change_comp.grid(axis='y')
            st.pyplot(fig_change_comp)

            st.header("7. Spread Analysis")
            st.write("Compare the actual spread between two maturities with the fair value spread from the PCA model.")
            
            if len(selected_maturities) >= 2:
                spread_maturity_1 = st.selectbox("Select first maturity", options=selected_maturities, index=0, key="m1")
                spread_maturity_2 = st.selectbox("Select second maturity", options=selected_maturities, index=1, key="m2")

                if spread_maturity_1 == spread_maturity_2:
                    st.warning("Please select two different maturities for spread analysis.")
                else:
                    actual_spread_series = data_filtered[spread_maturity_1] - data_filtered[spread_maturity_2]
                    reconstructed_data_scaled_spread = np.dot(principal_components, pca.components_)
                    reconstructed_data_spread = scaler.inverse_transform(reconstructed_data_scaled_spread)
                    reconstructed_df_all_dates_spread = pd.DataFrame(reconstructed_data_spread, columns=selected_maturities, index=data_filtered.index)
                    fair_value_spread_series = reconstructed_df_all_dates_spread[spread_maturity_1] - reconstructed_df_all_dates_spread[spread_maturity_2]

                    spread_df = pd.DataFrame({
                        'Actual Spread': actual_spread_series,
                        'Fair Value Spread': fair_value_spread_series
                    })
                    
                    st.subheader(f'Spread over time: {spread_maturity_1} - {spread_maturity_2}')
                    fig_spread_time, ax_spread_time = plt.subplots(figsize=(15, 8))
                    ax_spread_time.plot(spread_df.index, spread_df['Actual Spread'], label='Actual Spread', color='blue')
                    ax_spread_time.plot(spread_df.index, spread_df['Fair Value Spread'], label='Fair Value Spread', color='red', linestyle='--')
                    ax_spread_time.set_title(f'Actual vs. Fair Value Spread Over Time ({spread_maturity_1} - {spread_maturity_2})')
                    ax_spread_time.set_xlabel('Date')
                    ax_spread_time.set_ylabel('Spread Value')
                    ax_spread_time.legend()
                    ax_spread_time.grid(True)
                    st.pyplot(fig_spread_time)

                    last_date = data_filtered.index[-1]
                    st.write(f"**Spread values for the last date ({last_date.strftime('%Y-%m-%d')}):**")
                    st.metric("Actual Spread", f"{actual_spread_series.loc[last_date]:.4f}")
                    st.metric("Fair Value Spread", f"{fair_value_spread_series.loc[last_date]:.4f}")
                    st.metric("Difference (Actual - Fair Value)", f"{(actual_spread_series.loc[last_date] - fair_value_spread_series.loc[last_date]):.4f}")
            else:
                st.warning("Please select at least two maturities to perform spread analysis.")

        elif page == "Next Day Forecast":
            st.header("Next Day Fair Value Forecast")
            st.write("This section forecasts the fair value yield curve for the next day and presents the percentage change from the last known actual values.")

            # Get the most recent principal component scores
            last_pc_scores = principal_components[-1]
            last_date_in_data = data_filtered.index[-1]

            # Calculate a simple forecast for the next day's PC scores
            # For demonstration, we'll use a simple moving average of the last 'n' days
            forecast_period = st.slider("Select number of days for moving average forecast", min_value=1, max_value=len(principal_components), value=5)
            
            # Ensure we have enough data for the forecast period
            if len(principal_components) < forecast_period:
                st.warning(f"Not enough data points ({len(principal_components)}) for a {forecast_period}-day forecast. Using the last available data point instead.")
                forecasted_pc_scores = last_pc_scores
            else:
                forecasted_pc_scores = np.mean(principal_components[-forecast_period:], axis=0)

            # Reconstruct the forecasted yield curve using the forecasted scores
            mean_data = scaler.mean_
            forecasted_curve = mean_data + np.dot(pca.components_.T, forecasted_pc_scores.T)

            # Create a new date for the forecast, assuming it's the next business day
            next_day = last_date_in_data + pd.offsets.BDay(1)

            # Get the actual curve for the last day for comparison
            actual_curve_last_day = data_filtered.loc[last_date_in_data]
            
            # Calculate the forecasted yield change in percentage terms
            forecasted_yield_change_percent = ((forecasted_curve - actual_curve_last_day.values) / actual_curve_last_day.values) * 100

            # Create a dataframe for the forecasted change for plotting and display
            forecasted_change_df = pd.DataFrame(
                {'Maturity': selected_maturities, 'Yield Change (%)': forecasted_yield_change_percent}
            ).set_index('Maturity')

            # Plot the forecasted yield change
            fig_forecast, ax_forecast = plt.subplots(figsize=(15, 8))
            sns.barplot(
                x=forecasted_change_df.index,
                y=forecasted_change_df['Yield Change (%)'],
                ax=ax_forecast,
                palette='coolwarm'
            )
            
            ax_forecast.set_title(f'Next Day Fair Value Yield Change Forecast for {next_day.strftime("%Y-%m-%d")} (%)')
            ax_forecast.set_xlabel('Maturity')
            ax_forecast.set_ylabel('Yield Change (%)')
            ax_forecast.grid(True)
            st.pyplot(fig_forecast)
            
            st.subheader(f"Forecasted Fair Value Yield Changes (%) for {next_day.strftime('%Y-%m-%d')}")
            st.dataframe(forecasted_change_df.style.format({'Yield Change (%)': '{:.4f}%'}))
            
            st.header("Next Day Fair Value Spread Forecast")
            st.write("This section forecasts the spread between two maturities for the next day.")

            if len(selected_maturities) >= 2:
                forecast_spread_maturity_1 = st.selectbox("Select first maturity", options=selected_maturities, index=0, key="forecast_m1")
                forecast_spread_maturity_2 = st.selectbox("Select second maturity", options=selected_maturities, index=1, key="forecast_m2")

                if forecast_spread_maturity_1 == forecast_spread_maturity_2:
                    st.warning("Please select two different maturities for spread analysis.")
                else:
                    # Find the indices of the selected maturities
                    m1_index = selected_maturities.index(forecast_spread_maturity_1)
                    m2_index = selected_maturities.index(forecast_spread_maturity_2)
                    
                    # Get the last actual spread
                    last_actual_spread = data_filtered.loc[last_date_in_data, forecast_spread_maturity_1] - data_filtered.loc[last_date_in_data, forecast_spread_maturity_2]
                    
                    # Calculate the forecasted spread
                    forecasted_spread = forecasted_curve[m1_index] - forecasted_curve[m2_index]
                    
                    # Calculate the change in spread
                    forecasted_spread_change = forecasted_spread - last_actual_spread

                    st.subheader(f"Forecasted Spread for {next_day.strftime('%Y-%m-%d')}")
                    
                    # Display the metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Last Actual Spread ({last_date_in_data.strftime('%Y-%m-%d')})", f"{last_actual_spread:.4f}")
                    with col2:
                        st.metric(f"Forecasted Spread ({next_day.strftime('%Y-%m-%d')})", f"{forecasted_spread:.4f}")
                    with col3:
                        st.metric(f"Change in Spread", f"{forecasted_spread_change:.4f}")

                    # Plot a bar chart for visual comparison
                    spread_data = {
                        'Category': [f'Last Actual ({last_date_in_data.strftime("%Y-%m-%d")})', f'Forecasted ({next_day.strftime("%Y-%m-%d")})'],
                        'Spread Value': [last_actual_spread, forecasted_spread]
                    }
                    spread_df = pd.DataFrame(spread_data)

                    fig_spread, ax_spread = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Category', y='Spread Value', data=spread_df, ax=ax_spread, palette='tab10')
                    ax_spread.set_title(f'Actual vs. Forecasted Spread ({forecast_spread_maturity_1} - {forecast_spread_maturity_2})')
                    ax_spread.set_ylabel('Spread Value')
                    ax_spread.grid(axis='y')
                    st.pyplot(fig_spread)
            else:
                st.warning("Please select at least two maturities to perform spread analysis.")


    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()
