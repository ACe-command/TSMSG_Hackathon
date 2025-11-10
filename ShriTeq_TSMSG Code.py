# v11 
import os
import math
import time
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# ML / forecasting imports
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from typing import Optional


# Tokens
# ----------------------------
PHRASES = [
    "save extra power in large batteries",
    "send extra power to other states",
    "ask people to use less power at busy times",
    "use storage and power lines more fully",
    "temporarily reduce power from non-essential plants",
    "sell short-term power to nearby areas",
    "shift factory work to times when power use is low",
    "use water storage (pumped hydro) where possible",
    "make short-term deals to sell extra power",
    "make sure critical places (hospitals) get power first for charging",
    "start up backup power and fast gas plants",
    "immediately manage demand and ask for power saving",
    "quickly buy power from other places",
    "quickly finish building solar and gas power plants",
    "make sure important needs (hospitals) are met and spread out factory use",
    "bring in more power from other states",
    "quickly turn up fast-start power plants",
    "delay simple repairs to keep all power plants running",
    "tell people to save power and run short-term power saving plans",
    "work with nearby states to get emergency power",
    "Immediate (0-6 months)",
    "Short-term (6-24 months)",
    "Medium-term (2-5 years)",
    "Long-term (>5 years)"
]

GEN_TEMPERATURE = float(os.environ.get("ECHO_GEN_TEMPERATURE", "0.6"))

def compute_confidence(magnitude_gwh: float) -> int:
    try:
        conf = int(np.clip(92 - (magnitude_gwh ** 0.45) * 0.6, 50, 95))
    except Exception:
        conf = 80
    return conf

def select_timeframe(mag: float) -> str:
    if mag < 500:
        return PHRASES[21]  
    if mag < 3000:
        return PHRASES[21] 
    if mag < 10000:
        return PHRASES[22]  
    return PHRASES[23]       

def local_generate(surplus_deficit_gwh: float, state: Optional[str], max_actions: int = 2, temperature: float = GEN_TEMPERATURE) -> str:
    if surplus_deficit_gwh is None or (isinstance(surplus_deficit_gwh, float) and math.isnan(surplus_deficit_gwh)):
        return "No reliable surplus/deficit value available to generate a recommendation."

    mag = abs(float(surplus_deficit_gwh))
    sign = 1 if surplus_deficit_gwh > 0 else (-1 if surplus_deficit_gwh < 0 else 0)

    state_hash = sum(ord(c) for c in (state or "")) % 997
    seed = int((mag * 13) + (0 if sign >= 0 else 7) + state_hash) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)

    surplus_pool = PHRASES[0:10]
    deficit_pool = PHRASES[10:20]
    pool = surplus_pool if sign >= 0 else deficit_pool

    if mag < 500:
        n_actions = 1
    elif mag < 3000:
        n_actions = 1 if rng.random() < 0.4 else 2
    else:
        n_actions = 2
    n_actions = min(n_actions, max_actions, len(pool))


    temp = max(temperature, 0.01)
    scores = rng.random(len(pool)) ** (1.0 / temp)
    picked_indices = np.argsort(-scores)[:n_actions]
    selected = [pool[int(i)].capitalize() for i in picked_indices]

    conf = compute_confidence(mag)

    actions_text = "; ".join(selected)
    text = f"{actions_text} ({conf}%)"
    
    return text

def generate_recommendation(surplus_deficit_gwh: float, state: Optional[str]) -> str:
    # Use max_actions=1 for a more concise output, as requested.
    # The system will still pick 2 actions for large surplus/deficit to ensure full advice.
    return local_generate(surplus_deficit_gwh, state, max_actions=2, temperature=GEN_TEMPERATURE)


# Select Box
# ----------------------------
def build_state_selectbox_options(df, default_state_candidate: str = "Maharashtra"):

    if df is None or "State" not in df.columns:
        options = ["Unknown"]
    else:
        options = list(sorted(pd.Series(df["State"]).dropna().astype(str).str.strip().unique()))
        if len(options) == 0:
            options = ["Unknown"]
    default_state = default_state_candidate if default_state_candidate in options else options[0]
    try:
        default_index = options.index(default_state)
    except ValueError:
        default_index = 0
    return options, default_index


# Streamlit page 
# ----------------------------
st.set_page_config(page_title="⚡ ECHO", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
    .stApp { background-color: #0B0C10; color: #C5C6C7; font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { color: #66FCF1; }
    .metric-card { background-color: #1F2833; border-radius: 12px; padding: 25px; text-align: center;
                    box-shadow: 0 0 20px rgba(102, 252, 241, 0.12); margin-bottom: 20px; }
    .metric-card h3 { color: #45A29E; font-size: 1.0em; margin-bottom: 6px; }
    .metric-card h2 { color: #FFFFFF; font-size: 2.1em; margin-top: 0; }
    .dataframe th, .dataframe td { color: #C5C6C7 !important; }
    div[data-testid="stDataFrame"] { border: 1px solid #1F2833; border-radius: 10px; }
    html, body, .stApp, .block-container, .main {
        -webkit-overflow-scrolling: touch;
        overflow: auto !important;
        touch-action: auto;
    }
    /* ENHANCED CSS FOR TEXT WRAPPING */
    div[data-testid="stDataFrame"] .data-grid .cell-text {
        white-space: normal !important; /* Force text wrapping */
        word-wrap: break-word !important; /* Ensure long words break */
        line-height: 1.4 !important; /* Improve readability of wrapped text */
    }
    div[data-testid="stDataFrame"] div[role="row"] {
        height: auto !important; /* Allow row height to expand based on content */
        min-height: 40px; /* Set a minimum height */
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center;'>⚡ ECHO : Energy Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#C5C6C7;'>ECHO - Predict. Prevent. Power the Future.</h4>", unsafe_allow_html=True)
st.markdown("<hr style='border:1.5px solid #45A29E; margin-top: 10px; margin-bottom: 18px;'>", unsafe_allow_html=True)


# Data
# ----------------------------
gen_path = r"C:\Users\Aarav Chhabra\Downloads\energy_generation_india.csv"
cons_path = r"C:\Users\Aarav Chhabra\Downloads\energy_consumption_india_2019_2024_real_clean.csv"
weather_path = r"C:\Users\Aarav Chhabra\Downloads\imd_weather_india_2019_2024.csv"
map_path = r"C:\Users\Aarav Chhabra\Downloads\india_state.geojson"

@st.cache_data
def load_data_safe():
    if not os.path.exists(gen_path):
        st.error(f"Generation data file not found at: {gen_path}. Cannot proceed.")
        return None
    gen = pd.read_csv(gen_path)
    try:
        cons = pd.read_csv(cons_path)
    except Exception:
        st.warning("Consumption data not found. Using mock data.")
        cons = gen[['Year', 'State']].copy()
        gen_col = 'Total_Generation_GWh' if 'Total_Generation_GWh' in gen.columns else 'Electricity_Generation'
        cons['Energy_Requirement_GWh'] = gen.get(gen_col, 0).fillna(0) * 0.95 + np.random.rand(len(gen)) * 1000
    try:
        weather = pd.read_csv(weather_path)
    except Exception:
        st.warning("Weather data not found. Using mock data.")
        weather = gen[['Year', 'State']].copy()
        weather['Avg_Temperature (°C)'] = 25 + np.random.rand(len(gen)) * 10
        weather['Total_Rainfall (mm)'] = 100 + np.random.rand(len(gen)) * 500
        weather['Avg_Humidity (%)'] = 50 + np.random.rand(len(gen)) * 30
    for df_ in [gen, cons, weather]:
        df_.columns = df_.columns.str.strip()
    gen.rename(columns={"Total_Generation_GWh": "Electricity_Generation"}, inplace=True, errors="ignore")
    cons.rename(columns={"Energy_Requirement": "Energy_Requirement_GWh"}, inplace=True, errors="ignore")
    df = gen.merge(cons, on=["Year", "State"], how="inner")
    df = df.merge(weather, on=["Year", "State"], how="left")
    return df

df = load_data_safe()
if df is None:
    st.stop()


# Feature engineering & models
# ----------------------------
df['Solar_Generation'] = df.get('Solar_Generation', 0).fillna(0)
df['Hydro_Generation'] = df.get('Hydro_Generation', 0).fillna(0)
df['Coal_Generation'] = df.get('Coal_Generation', 0).fillna(0)

mean_temp = df.get('Avg_Temperature (°C)', pd.Series([25])).mean()
mean_rain = df.get('Total_Rainfall (mm)', pd.Series([300])).mean()
mean_humid = df.get('Avg_Humidity (%)', pd.Series([60])).mean()
df['Avg_Temperature (°C)'] = df.get('Avg_Temperature (°C)', mean_temp).fillna(mean_temp)
df['Total_Rainfall (mm)'] = df.get('Total_Rainfall (mm)', mean_rain).fillna(mean_rain)
df['Avg_Humidity (%)'] = df.get('Avg_Humidity (%)', mean_humid).fillna(mean_humid)

df["Renewable_Index"] = ((df["Solar_Generation"] + df["Hydro_Generation"]) / (df["Electricity_Generation"] + 1)) * 100
df["Rain_Temp_Interaction"] = df["Total_Rainfall (mm)"] * df["Avg_Temperature (°C)"]
df["Humidity_Trend"] = df.groupby("State")["Avg_Humidity (%)"].transform(lambda x: x - x.shift(1))
df["Year_Trend"] = (df["Year"] - df["Year"].min()) / (df["Year"].max() - df["Year"].min())

for col in ["Electricity_Generation", "Energy_Requirement_GWh"]:
    df[f"{col}_lag1"] = df.groupby("State")[col].shift(1)
df.dropna(subset=["Electricity_Generation_lag1"], inplace=True)

# Generation per-source models
preprocessor_gen = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Year_Trend"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["State"])
    ],
    remainder="passthrough"
)
gen_models = {}
gen_features = ["Year_Trend", "State"]
target_cols = ["Solar_Generation", "Hydro_Generation", "Coal_Generation"]
for target in target_cols:
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor_gen),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    df_train = df[df[target].notna() & (df[target] >= 0)].copy()
    if not df_train.empty:
        X_gen = df_train[gen_features]
        y_gen = df_train[target]
        pipeline.fit(X_gen, y_gen)
        gen_models[target] = pipeline

# Demand model (stacking)
base_estimators = []
if XGBRegressor is not None:
    base_estimators.append(("xgb", XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, eval_metric="mae")))
if LGBMRegressor is not None:
    base_estimators.append(("lgbm", LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)))
if CatBoostRegressor is not None:
    base_estimators.append(("cat", CatBoostRegressor(iterations=200, learning_rate=0.05, random_state=42, verbose=0, thread_count=-1)))
if not base_estimators:
    base_estimators = [("rf", RandomForestRegressor(n_estimators=100, random_state=42))]

demand_features = [
    "Year", "Electricity_Generation", "Solar_Generation", "Hydro_Generation",
    "Coal_Generation", "Avg_Temperature (°C)", "Total_Rainfall (mm)",
    "Avg_Humidity (%)", "Renewable_Index", "Rain_Temp_Interaction",
    "Year_Trend", "Electricity_Generation_lag1"
]
demand_target = "Energy_Requirement_GWh"
X_full, y_full = df[demand_features].fillna(0), df[demand_target]

demand_model_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("regressor", StackingRegressor(estimators=base_estimators, final_estimator=GradientBoostingRegressor(n_estimators=50, random_state=42)))
])

train_mask = df["Year"] <= 2022
val_mask = df["Year"] > 2022
X_train, y_train = X_full[train_mask], y_full[train_mask]
X_val, y_val = X_full[val_mask], y_full[val_mask]
if len(X_train) > 0:
    demand_model_pipeline.fit(X_train, y_train)
preds_val = demand_model_pipeline.predict(X_val) if len(X_val) > 0 else np.array([])
mae = mean_absolute_error(y_val, preds_val) if len(y_val) > 0 else 0
min_max_diff = (y_full.max() - y_full.min()) if len(y_full) > 0 else 1
acc = (1 - mae / (min_max_diff if min_max_diff != 0 else 1)) * 100
acc = np.clip(acc, 0, 100)
demand_model = demand_model_pipeline

# Forecast 2025-2050 (deterministic per state/year RNG)
states = df["State"].unique()
future_years = range(2025, 2051)
future_data = []
max_gen = df["Electricity_Generation"].max() if "Electricity_Generation" in df.columns else 1
max_demand = df["Energy_Requirement_GWh"].max() if "Energy_Requirement_GWh" in df.columns else 1
gen_fluctuation_scale = max_gen * 0.015
demand_fluctuation_scale = max_demand * 0.015
ANNUAL_GEN_GROWTH = 0.005

with st.spinner("Generating future forecasts for 2025-2050..."):
    for state in states:
        latest_data = df[df["State"] == state].sort_values("Year")
        if latest_data.empty:
            continue
        latest = latest_data.iloc[-1].copy()
        current_gen = latest["Electricity_Generation"]
        current_demand = latest["Energy_Requirement_GWh"]
        for year in future_years:
            row = latest.copy()
            row["Year"] = year
            max_orig_year = df["Year"].max()
            min_orig_year = df["Year"].min()
            row["Year_Trend"] = (year - min_orig_year) / (max_orig_year - min_orig_year)

            gen_input = pd.DataFrame([{"Year_Trend": row["Year_Trend"], "State": state}])
            pred_solar = gen_models.get("Solar_Generation", None).predict(gen_input)[0] if "Solar_Generation" in gen_models else 0
            pred_hydro = gen_models.get("Hydro_Generation", None).predict(gen_input)[0] if "Hydro_Generation" in gen_models else 0
            pred_coal = gen_models.get("Coal_Generation", None).predict(gen_input)[0] if "Coal_Generation" in gen_models else 0
            base_pred_total_gen = max(pred_solar + pred_hydro + pred_coal, 0)
            years_diff = year - latest["Year"]
            growth_factor = (1 + ANNUAL_GEN_GROWTH) ** years_diff
            pred_total_gen = base_pred_total_gen * growth_factor

            # ---------- Deterministic RNG per (state, year) ----------
            # Stable seed derived from state string and year so results don't change on unrelated widget updates.
            state_hash = sum(ord(c) for c in str(state or "")) & 0xFFFFFFFF
            seed = (state_hash * 1000003 + int(year) * 9176) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)

            gen_fluctuation = rng.uniform(-gen_fluctuation_scale, gen_fluctuation_scale)
            pred_total_gen = max(0, pred_total_gen + gen_fluctuation)

            row["Electricity_Generation_lag1"] = current_gen
            row["Electricity_Generation"] = pred_total_gen

            demand_input_data = row[demand_features].fillna(0).to_dict()
            demand_input = pd.DataFrame([demand_input_data])
            pred_demand = max(demand_model.predict(demand_input)[0], 0) if hasattr(demand_model, "predict") else 0

            # deterministic demand fluctuation using same seed
            demand_fluctuation = rng.uniform(-demand_fluctuation_scale * 0.5, demand_fluctuation_scale * 1.5)
            pred_demand = max(0, pred_demand + demand_fluctuation)

            current_gen = pred_total_gen
            current_demand = pred_demand

            future_data.append({
                "State": state,
                "Year": year,
                "Predicted_Demand (GWh)": current_demand,
                "Electricity_Generation": current_gen,
                "Solar_Generation": max(pred_solar, 0),
                "Hydro_Generation": max(pred_hydro, 0),
                "Coal_Generation": max(pred_coal, 0),
                "Renewable_Index": ((pred_solar + pred_hydro) / (current_gen + 1)) * 100,
                "Year_Trend": row["Year_Trend"]
            })
future_df = pd.DataFrame(future_data)


# UI: YEAR selection, state select
# ----------------------------
col1, col2, col3 = st.columns([1.5, 1, 1])
states = df["State"].unique()

with col1:
    year_select = st.slider("📅 Select Forecast Year", 2019, 2050, 2024, help="Select a year to view detailed state-level predictions.")
    last_year = df["Year"].max()
    options, default_index = build_state_selectbox_options(df)
    state_sel = st.selectbox("📍 Select State for Trend Analysis", options, index=default_index)

if year_select <= last_year:
    show_df = df[df["Year"] == year_select].copy()
    show_df["Predicted_Demand (GWh)"] = show_df["Energy_Requirement_GWh"]
else:
    show_df = future_df[future_df["Year"] == year_select].copy()

show_df["Predicted_Surplus_Deficit (GWh)"] = show_df["Electricity_Generation"].fillna(0) - show_df["Predicted_Demand (GWh)"]
show_df["Status"] = np.where(show_df["Predicted_Surplus_Deficit (GWh)"] > 0, "Surplus", "Deficit")
show_df.reset_index(drop=True, inplace=True)

total_surplus_deficit = show_df["Predicted_Surplus_Deficit (GWh)"].sum()
surplus_count = (show_df["Status"] == "Surplus").sum()
deficit_count = (show_df["Status"] == "Deficit").sum()

with col2:
    st.markdown(f"<div class='metric-card'><h3>Validation Accuracy</h3><h2>{acc:.2f}%</h2></div>", unsafe_allow_html=True)

with col3:
    color = "#00FF00" if total_surplus_deficit > 0 else "#FF0000"
    sign = "+" if total_surplus_deficit > 0 else "-"
    st.markdown(f"<div class='metric-card'><h3>National Net Flow ({year_select})</h3><h2 style='color:{color};'>{sign}{abs(total_surplus_deficit):,.0f} GWh</h2></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Generate recommendations 
# ----------------------------
with st.spinner("Generating recommendations..."):
    show_df["AI_Recommendation"] = show_df.apply(
        lambda r: generate_recommendation(r.get("Predicted_Surplus_Deficit (GWh)", 0), state=r.get("State", None)),
        axis=1
    )

# Forecast table
# ----------------------------
st.subheader(f"📋 State-Level Recommendations | {year_select}")

# --- REVISED COLUMN SELECTION (Removed SNo, Status, Year) ---
table_data = show_df[[
    "State",
    "Predicted_Surplus_Deficit (GWh)",
    "AI_Recommendation"
]].copy()

table_data.rename(columns={
    "Predicted_Surplus_Deficit (GWh)": "Surplus/Deficit (GWh)",
    "AI_Recommendation": "Recommendation" # Rename for display
}, inplace=True)

def highlight_rows_new(row):
    # Highlight rows based on surplus or deficit
    color = "rgba(0,255,0,0.12)" if row["Surplus/Deficit (GWh)"] > 0 else "rgba(255,0,0,0.12)"
    return [f"background-color: {color}"] * len(row)

st.dataframe(
    table_data.style.format({"Surplus/Deficit (GWh)": "{:+,.0f}"}).apply(highlight_rows_new, axis=1),
    use_container_width=True,
    hide_index=True,
    # --- REVISED COLUMN CONFIGURATION (Maximized Recommendation Width) ---
    column_config={
        "State": st.column_config.TextColumn("State", width="medium"), # Retain fixed space for State
        "Surplus/Deficit (GWh)": st.column_config.NumberColumn("Surplus/Deficit (GWh)", width="medium"), # Retain fixed space for numeric value
        # ALL remaining available space is allocated to 'Recommendation'
        "Recommendation": st.column_config.TextColumn(
            "Recommendation", 
            width="large", # Allocates the majority of the remaining space
            help="Simple strategic actions based on the predicted energy flow."
        ), 
    }
)


# Map
# ----------------------------
tab1, tab2 = st.tabs(["🌎 Map Overview & National Summary", "📈 State Trend & Detailed Forecast"])

with tab1:
    map_col1, map_col2 = st.columns([1.5, 1])
    with map_col2:
        st.subheader("National Summary")
        if total_surplus_deficit > 0:
            st.success(f"✅ **Total Surplus: {total_surplus_deficit:,.0f} GWh**")
            st.info("The country has more power than it needs. Key Action: Save or sell the extra power.")
        else:
            st.error(f"⚠️ **Total Deficit: {abs(total_surplus_deficit):,.0f} GWh**")
            st.info("The country needs more power than it has. Key Action: Quickly get more power and save what's available.")
        st.markdown("---")
        st.caption(f"States with Extra Power: **{surplus_count}** | States Needing Power: **{deficit_count}**")
        st.markdown("---")

    with map_col1:
        st.subheader("Energy Flow Map")
        try:
            india = gpd.read_file(map_path)
        except Exception:
            st.warning("GeoJSON map file not found. Skipping map display. Please check the path.")
        else:
            india["NAME_1"] = india["NAME_1"].str.title()
            show_df["State"] = show_df["State"].str.title()
            map_df = show_df[["State", "Predicted_Surplus_Deficit (GWh)", "Status"]].copy()
            merged = india.merge(map_df, left_on="NAME_1", right_on="State", how="left")
            merged["Predicted_Surplus_Deficit (GWh)"].fillna(0, inplace=True)
            max_abs = merged["Predicted_Surplus_Deficit (GWh)"].abs().max()
            max_value = max_abs if max_abs > 0 else 1
            merged["Outline_Color"] = merged["Status"].map({"Surplus": "green", "Deficit": "red"}).fillna("gray")
            fig_map = px.choropleth(
                merged,
                geojson=merged.geometry.__geo_interface__,
                locations=merged.index,
                color="Predicted_Surplus_Deficit (GWh)",
                hover_name="NAME_1",
                color_continuous_scale="RdYlGn",
                range_color=[-max_value, max_value],
                projection="mercator",
                title=f"Energy Extra/Need by State — {year_select}",
            )
            fig_map.update_traces(marker_line_color=merged["Outline_Color"].tolist(), marker_line_width=1.5)
            fig_map.update_geos(fitbounds="locations", visible=False, bgcolor="#0B0C10", landcolor="#0B0C10", subunitcolor="#45A29E")
            fig_map.update_layout(height=600, margin={"r": 0, "t": 40, "l": 0, "b": 0}, plot_bgcolor="#0B0C10", paper_bgcolor="#0B0C10", font_color="#C5C6C7")
            st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": False})

with tab2:
    st.subheader(f"📈 Energy Trend for {state_sel}")
    hist = df[df["State"] == state_sel].sort_values("Year").copy()
    forecast_for_state = future_df[future_df["State"] == state_sel].sort_values("Year").copy()
    forecast_for_state = forecast_for_state[forecast_for_state["Year"] <= year_select]
    hist_df = hist[["Year", "Electricity_Generation", "Energy_Requirement_GWh"]].copy()
    forecast_df = forecast_for_state[["Year", "Electricity_Generation", "Predicted_Demand (GWh)"]].rename(columns={"Predicted_Demand (GWh)": "Energy_Requirement_GWh"}).copy()
    combined = pd.concat([hist_df, forecast_df], ignore_index=True)
    combined.drop_duplicates(subset=["Year"], keep="last", inplace=True)
    combined = combined[combined["Year"] <= year_select]
    combined.sort_values("Year", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined["Year"] = pd.to_numeric(combined["Year"], errors="coerce").astype("Int64")
    combined["Electricity_Generation"] = pd.to_numeric(combined["Electricity_Generation"], errors="coerce").fillna(0)
    combined["Energy_Requirement_GWh"] = pd.to_numeric(combined["Energy_Requirement_GWh"], errors="coerce").fillna(0)
    if combined.empty:
        st.warning(f"No data available for {state_sel} up to {year_select}. Nothing to plot.")
    else:
        last_hist_year = df["Year"].max()
        actual_mask = combined["Year"] <= last_hist_year
        forecast_mask = combined["Year"] > last_hist_year
        fig_line = go.Figure()
        if combined[actual_mask].shape[0] > 0:
            fig_line.add_trace(go.Scatter(x=combined[actual_mask]["Year"].astype(int), y=combined[actual_mask]["Electricity_Generation"], mode="lines+markers", name="Actual Generation", line=dict(color="#00FF00", width=3), marker=dict(size=6)))
            fig_line.add_trace(go.Scatter(x=combined[actual_mask]["Year"].astype(int), y=combined[actual_mask]["Energy_Requirement_GWh"], mode="lines+markers", name="Actual Demand", line=dict(color="#FF4500", width=3), marker=dict(size=6, symbol="x")))
        if combined[forecast_mask].shape[0] > 0:
            if combined[actual_mask].shape[0] > 0:
                last_actual_row = combined[actual_mask].iloc[[-1]]
                gen_forecast_series = pd.concat([last_actual_row[["Year", "Electricity_Generation"]], combined[forecast_mask][["Year", "Electricity_Generation"]]])
                dem_forecast_series = pd.concat([last_actual_row[["Year", "Energy_Requirement_GWh"]], combined[forecast_mask][["Year", "Energy_Requirement_GWh"]]])
            else:
                gen_forecast_series = combined[forecast_mask][["Year", "Electricity_Generation"]]
                dem_forecast_series = combined[forecast_mask][["Year", "Energy_Requirement_GWh"]]
            fig_line.add_trace(go.Scatter(x=gen_forecast_series["Year"].astype(int), y=gen_forecast_series["Electricity_Generation"], mode="lines", name="Forecasted Generation", line=dict(color="#00AA00", width=2, dash="dot")))
            fig_line.add_trace(go.Scatter(x=dem_forecast_series["Year"].astype(int), y=dem_forecast_series["Energy_Requirement_GWh"], mode="lines", name="Forecasted Demand", line=dict(color="#AA4500", width=2, dash="dot")))
        fig_line.add_trace(go.Scatter(x=combined["Year"].astype(int), y=combined["Electricity_Generation"], fill="tozeroy", fillcolor="rgba(102, 252, 241, 0.12)", line=dict(width=0), name="Generation (area)", hoverinfo="skip"))
        fig_line.update_layout(title=f"Energy Generation vs Demand Forecast — {state_sel} (Up to {year_select})", xaxis_title="Year", yaxis_title="Energy (GWh)", height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), plot_bgcolor="#1F2833", paper_bgcolor="#0B0C10", font_color="#C5C6C7", title_font_color="#66FCF1")
        if combined["Electricity_Generation"].max() == combined["Electricity_Generation"].min():
            yval = combined["Electricity_Generation"].max()
            fig_line.update_yaxes(range=[max(0, yval * 0.8 - 1), yval * 1.2 + 1])
        st.plotly_chart(fig_line, use_container_width=True, config={"scrollZoom": False})
