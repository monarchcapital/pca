import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

# App setup
st.set_page_config(layout="wide", page_title="Brazil DI Futures PCA")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# Helpers
QUARTERLY_CODES = {"F": 1, "J": 4, "N": 7, "V": 10}


def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def normalize_rate_input(val, unit):
    if pd.isna(val):
        return np.nan
    v = float(val)
    if "Percent" in unit:
        return v / 100.0
    if "Basis" in unit:
        return v / 10000.0
    return v


def denormalize_to_percent(frac):
    if pd.isna(frac):
        return np.nan
    return 100.0 * frac


def np_busdays_exclusive(start_dt, end_dt, holidays_np):
    if pd.isna(start_dt) or pd.isna(end_dt):
        return 0
    s = np.datetime64(pd.Timestamp(start_dt).date()) + np.timedelta64(1, "D")
    e = np.datetime64(pd.Timestamp(end_dt).date())
    if e < s:
        return 0
    return int(np.busday_count(s, e, weekmask="1111100", holidays=holidays_np))


def calculate_ttm(valuation_ts, expiry_ts, holidays_np, year_basis):
    bd = np_busdays_exclusive(valuation_ts, expiry_ts, holidays_np)
    return np.nan if bd <= 0 else bd / float(year_basis)


def ensure_business_day_back(date64, holidays_np):
    d = np.datetime64(pd.Timestamp(date64).date())
    if np.is_busday(d, weekmask="1111100", holidays=holidays_np):
        return d
    # step back until a business day
    for _ in range(31):
        d = d - np.timedelta64(1, "D")
        if np.is_busday(d, weekmask="1111100", holidays=holidays_np):
            break
    return d


def safe_busday_offset_back(date64, offset, holidays_np):
    start = ensure_business_day_back(date64, holidays_np)
    return np.busday_offset(start, -int(offset), weekmask="1111100", holidays=holidays_np)


def adjusted_roll_date(expiry_ts, roll_bd_before, holidays_np):
    if pd.isna(expiry_ts):
        return pd.NaT
    d64 = np.datetime64(pd.Timestamp(expiry_ts).date())
    try:
        roll64 = safe_busday_offset_back(d64, roll_bd_before, holidays_np)
    except Exception:
        roll64 = ensure_business_day_back(d64, holidays_np)
        for _ in range(int(roll_bd_before)):
            roll64 = ensure_business_day_back(roll64 - np.timedelta64(1, "D"), holidays_np)
    return pd.Timestamp(roll64)


def extract_month_code(maturity_name: str) -> str:
    for ch in str(maturity_name):
        if ch.isalpha():
            return ch.upper()
    return ""


def select_quarterly_generics_roll(valuation_ts, expiry_df, holidays_np, roll_bd_before=5):
    """Return front & second quarterly contracts using roll-window rule."""
    rows = []
    for mat, r in expiry_df.iterrows():
        exp = r["DATE"]
        if pd.isna(exp):
            continue
        code = extract_month_code(mat)
        if code not in QUARTERLY_CODES:
            continue
        roll = adjusted_roll_date(exp, roll_bd_before, holidays_np)
        rows.append((mat, pd.Timestamp(exp), roll))
    if not rows:
        return None, None
    rows.sort(key=lambda x: x[1])
    vdt = pd.Timestamp(valuation_ts)
    idx = None
    for i, (_, _, roll) in enumerate(rows):
        if vdt <= roll:
            idx = i
            break
    if idx is None:
        idx = len(rows) - 1
    front = rows[idx][0]
    second = rows[idx + 1][0] if idx + 1 < len(rows) else None
    return front, second


def select_quarterly_generics_calendar(valuation_ts, expiry_df):
    """Return quarterly contracts by calendar mapping J/N/V/F across months."""
    m = pd.Timestamp(valuation_ts).month
    if m <= 3:
        first_code, second_code = "J", "N"
    elif m <= 6:
        first_code, second_code = "N", "V"
    elif m <= 9:
        first_code, second_code = "V", "F"
    else:
        first_code, second_code = "F", "J"

    def pick_next_for_code(code):
        sub = expiry_df[expiry_df.index.map(lambda x: extract_month_code(x) == code)]
        if sub.empty:
            return None
        sub = sub.copy()
        sub["DATE"] = pd.to_datetime(sub["DATE"], errors="coerce")
        sub = sub[sub["DATE"] >= pd.Timestamp(valuation_ts)]
        if sub.empty:
            return None
        return sub.sort_values("DATE").index[0]

    return pick_next_for_code(first_code), pick_next_for_code(second_code)


# Sidebar: inputs (pre-flight)
st.sidebar.header("Upload / Settings")

# File uploads (unique keys)
yield_file = st.sidebar.file_uploader("1. Yield data CSV (dates + contract columns)", type="csv", key="yield_file")
expiry_file = st.sidebar.file_uploader("2. Expiry mapping CSV (maturity, date)", type="csv", key="expiry_file")
holiday_file = st.sidebar.file_uploader("3. Holiday dates CSV (optional)", type="csv", key="holiday_file")

# Parameters (unique keys)
std_maturities_txt = st.sidebar.text_input(
    "4. Standard maturities (years, comma-separated)",
    "0.25,0.50,0.75,1.00,1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,4.00,5.00,7.00",
    key="maturities_txt",
)
interpolation_method = st.sidebar.selectbox(
    "5. Interpolation method", ["linear", "cubic", "quadratic", "nearest"], key="interp_method"
)
apply_smoothing = st.sidebar.checkbox(
    "6. Apply smoothing (3-day centered)", value=False, key="apply_smoothing"
)
n_components = st.sidebar.slider("7. Number of PCA components", 1, 12, 3, key="n_components")
rate_unit = st.sidebar.selectbox(
    "8. Input rate unit",
    ["Percent (e.g. 13.45)", "Decimal (e.g. 0.1345)", "Basis points (e.g. 1345)"],
    key="rate_unit",
)
year_basis = int(
    st.sidebar.selectbox("9. Business days in year", [252, 360], index=0, key="year_basis")
)

generics_method = st.sidebar.radio(
    "10. Generics selection method",
    ["Roll-window (BDs before expiry)", "Calendar quarter mapping"],
    key="generics_method",
)
roll_bd_before = st.sidebar.number_input(
    "11. Roll-window: business days before expiry",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    key="roll_bd_before",
)

# Load minimal data early to get available date range for training selector
@st.cache_data(show_spinner=False)
def _quick_read_csv(upload):
    return pd.read_csv(io.StringIO(upload.getvalue().decode("utf-8")))

yields_df_head = None
if yield_file is not None:
    try:
        tmp = _quick_read_csv(yield_file)
        if not tmp.empty:
            dc = tmp.columns[0]
            tmp[dc] = safe_to_datetime(tmp[dc])
            tmp = tmp.dropna(subset=[dc])
            tmp = tmp.set_index(dc)
            yields_df_head = tmp
    except Exception:
        yields_df_head = None

if yields_df_head is not None and not yields_df_head.empty:
    min_date_available = yields_df_head.index.min().date()
    max_date_available = yields_df_head.index.max().date()
else:
    min_date_available = pd.to_datetime("2000-01-01").date()
    max_date_available = pd.to_datetime("2100-01-01").date()

# PCA training dates (MUST be set before starting analysis)
date_train_start = st.sidebar.date_input(
    "12. PCA Training Start Date",
    value=min_date_available,
    min_value=min_date_available,
    max_value=max_date_available,
    key="pca_train_start",
)
date_train_end = st.sidebar.date_input(
    "13. PCA Training End Date",
    value=max_date_available,
    min_value=min_date_available,
    max_value=max_date_available,
    key="pca_train_end",
)

if date_train_start > date_train_end:
    st.sidebar.error("PCA Training Start Date cannot be after End Date.")

# Start / Stop buttons (appear only when required inputs exist)
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

inputs_ready = (
    yield_file is not None
    and expiry_file is not None
    and len(std_maturities_txt.strip()) > 0
    and date_train_start <= date_train_end
)

col_start, col_stop = st.sidebar.columns(2)
if not st.session_state.analysis_started:
    if inputs_ready and col_start.button("Start Analysis", key="btn_start"):
        st.session_state.analysis_started = True
        st.rerun()
else:
    if col_stop.button("Stop Analysis", key="btn_stop"):
        st.session_state.analysis_started = False
        st.stop()

if not st.session_state.analysis_started:
    st.info("Upload files, set parameters and training date range, then click **Start Analysis**.")
    st.stop()

# Load full data (post-start)

def load_csv_file(f):
    return pd.read_csv(io.StringIO(f.getvalue().decode("utf-8")))

# Yields
yields_df = load_csv_file(yield_file)
date_col = yields_df.columns[0]
yields_df[date_col] = safe_to_datetime(yields_df[date_col])
yields_df = yields_df.dropna(subset=[date_col])
yields_df.set_index(date_col, inplace=True)
yields_df.columns = [str(c).strip() for c in yields_df.columns]
for c in yields_df.columns:
    yields_df[c] = pd.to_numeric(yields_df[c], errors="coerce")

# Expiry
expiry_raw = load_csv_file(expiry_file)
expiry_df = expiry_raw.iloc[:, :2].copy()
expiry_df.columns = ["MATURITY", "DATE"]
expiry_df["MATURITY"] = expiry_df["MATURITY"].astype(str).str.strip().str.upper()
expiry_df["DATE"] = safe_to_datetime(expiry_df["DATE"])
expiry_df = expiry_df.dropna(subset=["DATE"])  # remove bad rows
expiry_df.set_index("MATURITY", inplace=True)

# Holidays
holidays_np = np.array([], dtype="datetime64[D]")
if holiday_file:
    hol_df = load_csv_file(holiday_file)
    hol_series = safe_to_datetime(hol_df.iloc[:, 0]).dropna()
    if not hol_series.empty:
        holidays_np = np.array(hol_series.dt.date, dtype="datetime64[D]")

# Filter training window
train_mask = (yields_df.index.date >= date_train_start) & (yields_df.index.date <= date_train_end)
yields_df = yields_df.loc[train_mask]
if yields_df.empty:
    st.error("No yields in the selected PCA training date range.")
    st.stop()

if apply_smoothing:
    yields_df = yields_df.rolling(window=3, min_periods=1, center=True).mean()

# Parse standard maturities
try:
    std_maturities = [float(x.strip()) for x in std_maturities_txt.split(",") if x.strip() != ""]
    std_arr = np.array(sorted(std_maturities), dtype=float)
    std_cols = [f"{m:.2f}Y" for m in std_arr]
except Exception:
    st.error("Error parsing standard maturities.")
    st.stop()

# Build matrix for PCA (interpolated zero curves)
pca_df = pd.DataFrame(np.nan, index=yields_df.index, columns=std_cols, dtype=float)
for dt in yields_df.index:
    row = yields_df.loc[dt]
    ttm_list, zero_list = [], []
    for col in yields_df.columns:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index:
            continue
        exp = expiry_df.loc[mat_up, "DATE"]
        if pd.isna(exp) or pd.Timestamp(exp).date() < dt.date():
            continue
        bd = np_busdays_exclusive(dt, exp, holidays_np)
        if bd <= 0:
            continue
        t = bd / float(year_basis)
        raw_val = row.get(col, np.nan)
        if pd.isna(raw_val):
            continue
        r_frac = normalize_rate_input(raw_val, rate_unit)
        DF = (1.0 + r_frac) ** (-t)
        try:
            zero_frac = DF ** (-1.0 / t) - 1.0
        except Exception:
            continue
        ttm_list.append(t)
        zero_list.append(denormalize_to_percent(zero_frac))
    if len(ttm_list) > 1 and len(set(np.round(ttm_list, 12))) > 1:
        try:
            order = np.argsort(ttm_list)
            f = interp1d(
                np.array(ttm_list)[order],
                np.array(zero_list)[order],
                kind=interpolation_method,
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
            pca_df.loc[dt] = f(std_arr)
        except Exception:
            pass

# Fill NaNs by column means (fallback to overall mean)
pca_vals = pca_df.values.astype(float)
col_means = np.nanmean(pca_vals, axis=0)
if np.isnan(col_means).any():
    overall_mean = np.nanmean(col_means[~np.isnan(col_means)]) if np.any(~np.isnan(col_means)) else 0.0
    col_means = np.where(np.isnan(col_means), overall_mean, col_means)
inds = np.where(np.isnan(pca_vals))
if inds[0].size > 0:
    pca_vals[inds] = np.take(col_means, inds[1])
pca_df_filled = pd.DataFrame(pca_vals, index=pca_df.index, columns=pca_df.columns)

# PCA
scaler = StandardScaler(with_std=False)
X = scaler.fit_transform(pca_df_filled.values.astype(float))
n_comp = int(min(int(n_components), X.shape[1]))
pca = PCA(n_components=n_comp)
PCs = pca.fit_transform(X)

# PCA outputs
st.subheader("PCA Explained Variance")
ev = pd.Series(
    pca.explained_variance_ratio_, index=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))]
)
st.dataframe(ev.to_frame("Explained Variance Ratio").T, use_container_width=True)

st.subheader("PCA Loadings Heatmap")
loadings = pd.DataFrame(
    pca.components_, columns=pca_df_filled.columns, index=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
)
fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * loadings.shape[1])))
sns.heatmap(
    loadings,
    cmap="coolwarm",
    center=0,
    xticklabels=loadings.columns,
    yticklabels=loadings.index,
    annot=True,
    fmt=".4f",
    annot_kws={"size": 8},
    ax=ax,
)
st.pyplot(fig)


# Curve reconstruction (single date)

st.subheader("Original vs PCA Reconstructed Curve")
if pca_df_filled.index.size == 0:
    st.warning("No PCA output to reconstruct from.")
else:
    default_recon = pca_df_filled.index.max().date()
    recon_date = st.date_input(
        "Select date for reconstruction",
        value=default_recon,
        min_value=pca_df_filled.index.min().date(),
        max_value=pca_df_filled.index.max().date(),
        key="recon_date",
    )
    recon_ts = pd.Timestamp(recon_date)
    mask_pca = pca_df_filled.index.normalize() == recon_ts.normalize()
    mask_raw = yields_df.index.normalize() == recon_ts.normalize()

    if not mask_pca.any():
        st.warning("Selected date not available in PCA output.")
    else:
        recon_curve = pca_df_filled.loc[mask_pca].iloc[0].values
        # Build original curve on the same std maturities for comparison
        orig_on_std = np.full_like(std_arr, np.nan, dtype=float)
        if mask_raw.any():
            raw_row = yields_df.loc[mask_raw].iloc[0]
            ttm_list, zero_list = [], []
            for col in yields_df.columns:
                mat_up = str(col).strip().upper()
                if mat_up not in expiry_df.index:
                    continue
                exp = expiry_df.loc[mat_up, "DATE"]
                if pd.isna(exp) or pd.Timestamp(exp).date() < recon_ts.date():
                    continue
                t = calculate_ttm(recon_ts, exp, holidays_np, year_basis)
                if np.isnan(t) or t <= 0:
                    continue
                raw_val = raw_row.get(col, np.nan)
                if pd.isna(raw_val):
                    continue
                r_frac = normalize_rate_input(raw_val, rate_unit)
                DF = (1.0 + r_frac) ** (-t)
                try:
                    zero_frac = DF ** (-1.0 / t) - 1.0
                except Exception:
                    continue
                ttm_list.append(t)
                zero_list.append(denormalize_to_percent(zero_frac))
            if len(ttm_list) >= 2:
                try:
                    orig_fn = interp1d(
                        np.array(ttm_list),
                        np.array(zero_list),
                        kind="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    orig_on_std = orig_fn(std_arr)
                except Exception:
                    pass

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(std_arr, recon_curve, marker="o", label=f"Reconstructed ({n_comp} PCs)")
        ax.plot(std_arr, orig_on_std, marker="x", linestyle="--", label="Original (interpolated)")
        ax.set_xticks(std_arr)
        ax.set_xticklabels([f"{m:.2f}Y" for m in std_arr], rotation=45)
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Zero Rate (%)")
        ax.set_title(f"Curve Reconstruction for {recon_ts.date()}")
        ax.legend()
        ax.grid(alpha=0.4)
        st.pyplot(fig)

# ==========================
# All-maturities contract comparison
# ==========================
st.subheader("Futures Contract Reconstruction Comparison — All Maturities")
chart_type_all = st.radio("Chart type (All maturities):", ("Bar Chart", "Line Graph"), horizontal=True, key="chart_all")

contracts, ttms, actuals = [], [], []
if pca_df_filled.index.size > 0:
    # use same recon_ts chosen above if exists
    try:
        recon_ts
    except NameError:
        recon_ts = pca_df_filled.index.max()
    raw_mask_dt = yields_df.index.normalize() == pd.Timestamp(recon_ts).normalize()
    if raw_mask_dt.any():
        raw_row = yields_df.loc[raw_mask_dt].iloc[0]
        for col in yields_df.columns:
            mat_up = str(col).strip().upper()
            if mat_up not in expiry_df.index:
                continue
            exp = expiry_df.loc[mat_up, "DATE"]
            t = calculate_ttm(recon_ts, exp, holidays_np, year_basis)
            if pd.isna(t):
                continue
            val = raw_row.get(col, np.nan)
            if pd.isna(val):
                continue
            contracts.append(col)
            ttms.append(t)
            actuals.append(denormalize_to_percent(normalize_rate_input(val, rate_unit)))

if len(contracts) > 0 and pca_df_filled.index.size > 0:
    order = np.argsort(ttms)
    contracts = [contracts[i] for i in order]
    ttms = [ttms[i] for i in order]
    actuals = [actuals[i] for i in order]
    recon_fn_for_date = interp1d(std_arr, pca_df_filled.loc[pca_df_filled.index.normalize() == pd.Timestamp(recon_ts).normalize()].iloc[0].values, kind="linear", bounds_error=False, fill_value=np.nan)
    recon_vals = recon_fn_for_date(ttms)
    comp_df_all = pd.DataFrame(
        {"Contract": contracts, "TTM": ttms, "Actual (%)": actuals, "Reconstructed (%)": recon_vals}
    ).set_index("Contract")
    fig, ax = plt.subplots(figsize=(12, 5))
    if chart_type_all == "Bar Chart":
        x = np.arange(len(comp_df_all))
        w = 0.4
        ax.bar(x - w / 2, comp_df_all["Actual (%)"], w, label="Actual")
        ax.bar(x + w / 2, comp_df_all["Reconstructed (%)"], w, label="Reconstructed")
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df_all.index, rotation=45, ha="right")
    else:
        ax.plot(comp_df_all.index, comp_df_all["Actual (%)"], marker="o", label="Actual")
        ax.plot(comp_df_all.index, comp_df_all["Reconstructed (%)"], marker="x", linestyle="--", label="Reconstructed")
        ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Yield (%)")
    ax.set_title(f"All Maturities — Actual vs Reconstructed ({pd.Timestamp(recon_ts).date()})")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(comp_df_all, use_container_width=True)
else:
    st.info("No valid futures maturities found for the selected date (after filtering).")

# ==========================
# 3M & 6M generics comparison
# ==========================
st.subheader("3M & 6M Generics Comparison")
chart_type_gen = st.radio("Chart type (Generics):", ("Bar Chart", "Line Graph"), horizontal=True, key="chart_gen")

front = second = None
if generics_method.startswith("Roll"):
    front, second = select_quarterly_generics_roll(recon_ts, expiry_df, holidays_np, roll_bd_before)
else:
    front, second = select_quarterly_generics_calendar(recon_ts, expiry_df)

if front is None and second is None:
    st.warning("Could not find generic quarterly contracts for this date.")
else:
    chosen = [g for g in (front, second) if g is not None]
    gen_actual, gen_recon, gen_ttms = [], [], []
    if pca_df_filled.index.size > 0:
        recon_fn = interp1d(std_arr, pca_df_filled.loc[pca_df_filled.index.normalize() == pd.Timestamp(recon_ts).normalize()].iloc[0].values, kind="linear", bounds_error=False, fill_value=np.nan)
        raw_mask = yields_df.index.normalize() == pd.Timestamp(recon_ts).normalize()
        raw_row = yields_df.loc[raw_mask].iloc[0] if raw_mask.any() else None
        for g in chosen:
            matches = [c for c in yields_df.columns if str(c).strip().upper() == str(g).strip().upper()]
            if not matches:
                continue
            col = matches[0]
            exp = expiry_df.loc[str(g).strip().upper(), "DATE"]
            t = calculate_ttm(recon_ts, exp, holidays_np, year_basis)
            if pd.isna(t) or raw_row is None:
                continue
            val = raw_row.get(col, np.nan)
            if pd.isna(val):
                continue
            gen_actual.append(denormalize_to_percent(normalize_rate_input(val, rate_unit)))
            gen_recon.append(float(recon_fn(t)))
            gen_ttms.append(t)
        if len(gen_actual) == 0:
            st.warning("No valid generics values to plot for this date.")
        else:
            fig, ax = plt.subplots(figsize=(7, 4))
            labels = [str(g) for g in chosen[: len(gen_actual)]]
            if chart_type_gen == "Bar Chart":
                x = np.arange(len(gen_actual))
                w = 0.4
                ax.bar(x - w / 2, gen_actual, w, label="Actual")
                ax.bar(x + w / 2, gen_recon, w, label="Reconstructed")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45)
            else:
                ax.plot(labels, gen_actual, "o-", label="Actual")
                ax.plot(labels, gen_recon, "x--", label="Reconstructed")
                ax.tick_params(axis="x", rotation=45)
            ax.set_ylabel("Yield (%)")
            ax.set_title(f"3M & 6M Generics — {pd.Timestamp(recon_ts).date()}")
            ax.legend()
            st.pyplot(fig)
            gen_df = pd.DataFrame(
                {"Contract": labels, "TTM": gen_ttms, "Actual (%)": gen_actual, "Reconstructed (%)": gen_recon}
            ).set_index("Contract")
            st.dataframe(gen_df, use_container_width=True)

# ==========================
# Residual heatmap
# ==========================
st.subheader("Residual Heatmap (Actual % - Reconstructed %)")
residuals = pd.DataFrame(index=pca_df_filled.index, columns=yields_df.columns, dtype=float)

for dt in pca_df_filled.index:
    recon_row = pca_df_filled.loc[dt].values
    recon_fn = interp1d(std_arr, recon_row, kind="linear", bounds_error=False, fill_value=np.nan)
    raw_mask_dt = yields_df.index.normalize() == pd.Timestamp(dt).normalize()
    raw_row = yields_df.loc[raw_mask_dt].iloc[0] if raw_mask_dt.any() else None
    for col in yields_df.columns:
        mat_up = str(col).strip().upper()
        if mat_up not in expiry_df.index:
            residuals.loc[dt, col] = np.nan
            continue
        exp = expiry_df.loc[mat_up, "DATE"]
        t = calculate_ttm(dt, exp, holidays_np, year_basis)
        if pd.isna(t) or raw_row is None:
            residuals.loc[dt, col] = np.nan
            continue
        actual_val = raw_row.get(col, np.nan)
        actual_pct = (
            denormalize_to_percent(normalize_rate_input(actual_val, rate_unit)) if not pd.isna(actual_val) else np.nan
        )
        recon_pct = float(recon_fn(t)) if not np.isnan(t) else np.nan
        residuals.loc[dt, col] = (
            (actual_pct - recon_pct) if (not pd.isna(actual_pct) and not pd.isna(recon_pct)) else np.nan
        )

hm_start = st.date_input(
    "Heatmap start",
    value=residuals.index.min().date(),
    min_value=residuals.index.min().date(),
    max_value=residuals.index.max().date(),
    key="hm_start",
)
hm_end = st.date_input(
    "Heatmap end",
    value=residuals.index.max().date(),
    min_value=residuals.index.min().date(),
    max_value=residuals.index.max().date(),
    key="hm_end",
)
if hm_start > hm_end:
    st.error("Heatmap start must be <= end.")
else:
    hm_slice = residuals.loc[(residuals.index.date >= hm_start) & (residuals.index.date <= hm_end)]
    if hm_slice.empty:
        st.warning("No data in the selected heatmap range.")
    else:
        fig, ax = plt.subplots(figsize=(12, max(4, 0.2 * hm_slice.shape[1])))
        sns.heatmap(hm_slice.T, cmap="RdBu_r", center=0, xticklabels=False, yticklabels=True, ax=ax)
        ax.set_xlabel("Date")
        st.pyplot(fig)

# ==========================
# Downloads
# ==========================
st.markdown("---")
st.subheader("Downloads")
try:
    st.download_button("Download PCA (interpolated) CSV", pca_df_filled.to_csv().encode(), "pca_interpolated.csv")
except Exception:
    st.error("Could not prepare PCA CSV.")

try:
    st.download_button("Download Residuals CSV", residuals.to_csv().encode(), "residuals.csv")
except Exception:
    st.error("Could not prepare residuals CSV.")

