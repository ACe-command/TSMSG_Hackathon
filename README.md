# ⚡ ECHO — Tech Stack (v10, Simplified)

This document explains the tools, structure, and technical setup of ECHO — a single-file Streamlit app (`ECHO_v10.py`).  
This version runs completely offline and uses a local transformer-like generator for AI recommendations (no Hugging Face or external models).

---

## 1) High-Level Overview
ECHO is an interactive dashboard that:
- Reads **energy generation**, **consumption**, and **weather** data from CSV files.  
- Creates new features to find patterns between weather and energy trends.  
- Trains **machine learning models** to forecast future demand and generation.  
- Predicts **surplus or deficit** for each state up to the year 2050.  
- Generates **AI recommendations** locally, using fixed action phrases.  
- Shows everything through interactive maps, charts, and tables in Streamlit.  

The app works fully offline — all predictions and AI recommendations happen inside your computer.

---

## 2) Major Components and Libraries

### Web Interface
- **Streamlit:** creates the full dashboard, sliders, charts, and layout.

### Data Handling and Math
- **Pandas:** reads CSVs, merges datasets, and creates new calculated columns.  
- **NumPy:** handles numeric operations, random number generation, and math functions.

### Geospatial Tools
- **GeoPandas:** loads the India states map (GeoJSON format) and merges it with forecast data.

### Visualization
- **Plotly Express** and **Plotly Graph Objects:** create interactive charts and maps that update live in Streamlit.

### Machine Learning
- **scikit-learn:** the main library for data preprocessing, training, and evaluation.
  - Tools used: `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`.
  - Models used:  
    - `RandomForestRegressor` — base model for generation forecasts.  
    - `GradientBoostingRegressor` — final model in the stacking ensemble.  
    - `StackingRegressor` — combines multiple models to improve accuracy.
- **Optional advanced models (used if installed):**
  - `xgboost.XGBRegressor`
  - `lightgbm.LGBMRegressor`
  - `catboost.CatBoostRegressor`
- If these are not installed, the system falls back to `RandomForestRegressor`.

### Local AI Recommendation Generator
- A **small transformer-like generator** built in Python and NumPy.
- Works on fixed phrase tokens (10 surplus actions, 10 deficit actions, 4 timeframes).  
- Picks actions and timeframes based on energy surplus or deficit magnitude.  
- Calculates a **confidence score (50–95%)** for each recommendation.  
- Fully deterministic — same input always gives the same output.

---

## 3) How Each Part Works

- **load_data_safe():** reads and cleans the three CSV datasets (generation, consumption, weather).  
- **Feature Engineering:** creates new useful columns like Renewable Index, Temperature–Rainfall Interaction, Lag Features, and Humidity Trend.  
- **Generation Models:** RandomForest models trained per energy source (Solar, Hydro, Coal).  
- **Demand Model:** a stacked model that combines multiple ML algorithms.  
  - Base models: XGBoost, LightGBM, CatBoost (if available).  
  - Final model: GradientBoostingRegressor.  
  - Includes imputing missing data and scaling features.  
- **Forecast Loop:** predicts year-by-year data (2025–2050) per state with stable random seeds for consistency.  
- **Recommendation System:** `generate_recommendation()` uses the local generator to produce actionable advice.  
- **Visualization:** displays state-level tables, maps, and trends using Streamlit and Plotly.

---

## 4) Recommendation Generator Details
- Type: Local transformer-like phrase system (no deep learning).  
- Vocabulary: 24 tokens (10 surplus, 10 deficit, 4 timeframes).  
- Selection process:
  - Uses random seed based on state name and energy magnitude.  
  - Picks 1–2 suitable actions and adds a timeframe.  
  - Confidence score depends on energy difference (GWh).  
- Always returns a short, readable sentence like:  
  *"Store excess energy in grid-scale batteries. Short-term (6–24 months). Confidence: 88%."*  
- Not a true language model — it’s rule-based and deterministic.  
- Can be upgraded to a trained transformer model later.

---

## 5) Determinism and Reruns
- Streamlit re-runs the whole script whenever a widget changes.  
- ECHO uses fixed seeds based on **(state, year)** so forecasts stay the same each time you reload.  
- This keeps numbers and recommendations consistent during use.

---

## 6) Environment Variables and Tuning
- **ECHO_GEN_TEMPERATURE:** optional, controls how random or varied recommendations are (default = `0.6`).  
- No Hugging Face or external API keys are needed — everything is local.

---

## 7) Python Dependencies

### Minimum Required:
