# app.py
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="US House Monte Carlo Forecast", layout="wide")


@st.cache_data(show_spinner=False)
def load_baseline_local(path: str = "A2__2024_clean.csv") -> pd.DataFrame:
    """
    Loads baked-in baseline district data from local disk (cached).
    Required columns: DistrictKey, dem_share, rep_share
    Computes base margin in points: (dem_share - rep_share) * 100
    """
    df = pd.read_csv(path)
    return _prep_baseline_df(df)


@st.cache_data(show_spinner=False)
def load_baseline_bytes(b: bytes) -> pd.DataFrame:
    """
    Loads baseline district data from uploaded bytes (cached).
    Required columns: DistrictKey, dem_share, rep_share
    """
    df = pd.read_csv(BytesIO(b))
    return _prep_baseline_df(df)


def _prep_baseline_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"DistrictKey", "dem_share", "rep_share"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Baseline file missing required columns: {sorted(missing)}")

    out = df.loc[:, ["DistrictKey", "dem_share", "rep_share"]].copy()
    out["dem_share"] = pd.to_numeric(out["dem_share"], errors="coerce")
    out["rep_share"] = pd.to_numeric(out["rep_share"], errors="coerce")
    out = out.dropna(subset=["DistrictKey", "dem_share", "rep_share"])

    out["DistrictKey"] = out["DistrictKey"].astype(str).str.strip()
    out["base_margin"] = (out["dem_share"] - out["rep_share"]) * 100.0  # points

    if out.empty:
        raise ValueError("Baseline file parsed, but no valid rows remained after cleaning.")
    return out.loc[:, ["DistrictKey", "base_margin"]]


@st.cache_data(show_spinner=False)
def load_polling_bytes(b: bytes) -> pd.DataFrame:
    """
    Loads uploaded polling file bytes (cached).
    Required columns: modeldate, dem, rep
    """
    df = pd.read_csv(BytesIO(b))

    required = {"modeldate", "dem", "rep"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Polling file missing required columns: {sorted(missing)}")

    out = df.loc[:, ["modeldate", "dem", "rep"]].copy()
    out["modeldate"] = pd.to_datetime(out["modeldate"], errors="coerce")
    out["dem"] = pd.to_numeric(out["dem"], errors="coerce")
    out["rep"] = pd.to_numeric(out["rep"], errors="coerce")
    out = out.dropna(subset=["modeldate", "dem", "rep"]).sort_values("modeldate")
    return out


@st.cache_data(show_spinner=False)
def run_monte_carlo(
    base_margins: np.ndarray,
    mu_n: float,
    sigma_n: float,
    sigma_d: float,
    runs: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Vectorized Monte Carlo simulation (no Python loops).

    N ~ Normal(mu_n, sigma_n^2)        shape (runs, 1)
    eps_i ~ Normal(0, sigma_d^2)       shape (runs, n_districts)

    simulated_margin = base_margin + N + eps
    Dem wins district if simulated_margin > 0

    Returns:
      simulated_seats: (runs,) total Dem seats per run
      district_win_probs: (n_districts,) Dem win probability (%) per district
      house_win_prob: % of runs where seats >= 218
      expected_seats: mean seats
    """
    base = np.asarray(base_margins, dtype=np.float32).reshape(1, -1)
    n_districts = base.shape[1]
    runs = int(runs)

    rng = np.random.default_rng()

    # Allocate big array once; broadcast-add in-place.
    eps = rng.normal(loc=0.0, scale=float(sigma_d), size=(runs, n_districts)).astype(np.float32)
    N = rng.normal(loc=float(mu_n), scale=float(sigma_n), size=(runs, 1)).astype(np.float32)

    eps += N
    eps += base

    wins = eps > 0.0
    simulated_seats = wins.sum(axis=1).astype(np.int32)

    district_win_probs = (wins.mean(axis=0) * 100.0).astype(np.float32)
    house_win_prob = float((simulated_seats >= 218).mean() * 100.0)
    expected_seats = float(simulated_seats.mean())

    return simulated_seats, district_win_probs, house_win_prob, expected_seats


# -----------------------------
# App UI
# -----------------------------
st.title("US House Election Forecast — Vectorized Monte Carlo")

st.sidebar.header("Inputs")

# Baseline uploader fallback (while still trying local first)
baseline_upload = st.sidebar.file_uploader(
    "Baseline districts CSV (fallback if local file missing)",
    type=["csv"],
    accept_multiple_files=False,
    help="Must include: DistrictKey, dem_share, rep_share",
)

baseline_df = None
baseline_source = None

# 1) Try baked-in local file (as requested originally)
try:
    baseline_df = load_baseline_local("A2__2024_clean.csv")
    baseline_source = "local file: A2__2024_clean.csv"
except FileNotFoundError:
    # 2) Fallback to upload
    if baseline_upload is not None:
        try:
            baseline_df = load_baseline_bytes(baseline_upload.getvalue())
            baseline_source = f"uploaded file: {baseline_upload.name}"
        except Exception as e:
            st.sidebar.error(f"Baseline upload failed: {e}")
            baseline_df = None
    else:
        baseline_df = None
except Exception as e:
    st.sidebar.error(f"Failed to load baseline local file: {e}")
    baseline_df = None

if baseline_df is None:
    st.info(
        "Baseline file not found locally. Upload a baseline CSV in the sidebar "
        "(must include DistrictKey, dem_share, rep_share)."
    )

# Polling uploader
poll_upload = st.sidebar.file_uploader(
    "Polling CSV (e.g., rfiFi (1).csv)",
    type=["csv"],
    accept_multiple_files=False,
    help="Must include: modeldate, dem, rep",
)

mu_default = 0.0
selected_date_str = None
poll_df = None

if poll_upload is not None:
    try:
        poll_df = load_polling_bytes(poll_upload.getvalue())
        if poll_df.empty:
            st.sidebar.warning("Polling file loaded, but no valid rows found after parsing.")
        else:
            date_options = (
                poll_df["modeldate"].dt.strftime("%Y-%m-%d").dropna().unique().tolist()
            )
            date_options.sort()

            selected_date_str = st.sidebar.selectbox(
                "Select modeldate",
                options=date_options,
                index=len(date_options) - 1,
            )

            day_rows = poll_df[poll_df["modeldate"].dt.strftime("%Y-%m-%d") == selected_date_str]
            mu_default = float((day_rows["dem"] - day_rows["rep"]).mean())

            st.sidebar.caption(f"Computed μN (dem - rep) on {selected_date_str}: {mu_default:.2f}")
    except Exception as e:
        st.sidebar.error(f"Failed to load/parse polling file: {e}")
        poll_df = None
        mu_default = 0.0
        selected_date_str = None
else:
    st.info("No polling CSV uploaded yet — defaulting μN to 0.0. You can still run the simulation.")

# Keep μ slider synced with selected polling date (without fighting the user override)
MU_KEY = "mu_n"
MU_SRC_KEY = "mu_n_source"

if MU_KEY not in st.session_state:
    st.session_state[MU_KEY] = float(mu_default)
    st.session_state[MU_SRC_KEY] = str(selected_date_str)
elif poll_upload is not None and str(selected_date_str) != st.session_state.get(MU_SRC_KEY):
    st.session_state[MU_KEY] = float(mu_default)
    st.session_state[MU_SRC_KEY] = str(selected_date_str)

mu_n = st.sidebar.slider(
    "Mean National Swing (μN)",
    min_value=-20.0,
    max_value=20.0,
    value=float(st.session_state[MU_KEY]),
    step=0.1,
    key=MU_KEY,
)

sigma_n = st.sidebar.slider(
    "National Volatility (σN)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1,
)

sigma_d = st.sidebar.slider(
    "District Chaos (σd)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1,
)

runs = st.sidebar.slider(
    "Runs (T)",
    min_value=1_000,
    max_value=200_000,
    value=50_000,
    step=1_000,
)

# If we have baseline, run the simulation; otherwise show placeholders
if baseline_df is None:
    c1, c2 = st.columns(2)
    c1.metric("Expected Dem Seats", "—")
    c2.metric("P(Dem Majority ≥ 218)", "—")
    st.stop()

district_keys = baseline_df["DistrictKey"].to_numpy()
base_margins = baseline_df["base_margin"].to_numpy(dtype=np.float32)
n_districts = base_margins.shape[0]

st.caption(f"Loaded baseline for {n_districts} districts ({baseline_source}).")

simulated_seats, district_win_probs, house_win_prob, expected_seats = run_monte_carlo(
    base_margins=base_margins,
    mu_n=float(mu_n),
    sigma_n=float(sigma_n),
    sigma_d=float(sigma_d),
    runs=int(runs),
)

# Top row metrics
c1, c2 = st.columns(2)
c1.metric("Expected Dem Seats", f"{expected_seats:.1f}")
c2.metric("P(Dem Majority ≥ 218)", f"{house_win_prob:.1f}%")

# Histogram (Plotly)
hist_df = pd.DataFrame({"DemSeats": simulated_seats})
fig = px.histogram(
    hist_df,
    x="DemSeats",
    nbins=60,
    title="Simulated Distribution of Democratic Seats",
    labels={"DemSeats": "Democratic Seats"},
)
fig.update_layout(bargap=0.02)
fig.add_vline(
    x=218,
    line_width=4,
    line_color="black",
    line_dash="solid",
    annotation_text="218 Majority",
    annotation_position="top right",
)
st.plotly_chart(fig, use_container_width=True)

# District table
out_df = pd.DataFrame(
    {
        "DistrictKey": district_keys,
        "DemWinProb (%)": district_win_probs.astype(float),
    }
).sort_values("DemWinProb (%)", ascending=False, ignore_index=True)

st.subheader("District Win Probabilities")
st.dataframe(
    out_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "DistrictKey": st.column_config.TextColumn("DistrictKey"),
        "DemWinProb (%)": st.column_config.NumberColumn(
            "Dem Win Probability (%)",
            format="%.1f",
            help="Percent of Monte Carlo runs where the Democrat wins the district.",
        ),
    },
)
