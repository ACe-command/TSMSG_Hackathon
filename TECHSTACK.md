# ⚡ ECHO — Source Code (v10) Explained Practically

This explanation breaks down the core functions and logic found in the `app.py` file.

---

## 1️⃣ Core Components and Setup

* **Imports:** The code brings in essential libraries: **Streamlit** (for the web app), **pandas/numpy** (for data and math), **geopandas/plotly** (for maps and charts), and **scikit-learn** (for Machine Learning).
* **Accuracy/Efficiency Boosters:** It tries to import **XGBoost, LightGBM, and CatBoost**. If these fast ML libraries aren't installed, the code is designed to **fall back** to using the standard **RandomForestRegressor** from scikit-learn instead, so the app still runs.
* **Styling:** A large `<style>` block customizes the Streamlit page with a **dark, clean theme** (`#0B0C10` background, `#66FCF1` accents) and optimizes how the app scrolls.
* **Data Loading (`@st.cache_data`):** The `load_data_safe()` function reads the CSV files. The `@st.cache_data` decorator ensures this **slow step only runs once**, even if the user clicks a widget and the rest of the script re-runs. It also includes **mock data** as a fallback if the specified CSV paths are broken.

---

## 2️⃣ Feature Engineering and Model Training

This section prepares the data and builds the forecasting tools.

* **Feature Creation:** It generates powerful new columns for the models, such as:
    * `Renewable_Index`: Measures the proportion of energy from solar/hydro.
    * `Year_Trend`: A scaled number indicating how far into the future the year is.
    * `..._lag1`: The energy generation from the **previous year** (a crucial time-series feature).
* **Generation Models (Simple):** Separate **RandomForest** pipelines are trained to predict Solar, Hydro, and Coal generation based only on the **State** (as a dummy variable/OneHotEncoder) and the **Year\_Trend**.
* **Demand Model (Complex):** This uses a sophisticated **StackingRegressor**.
    * **Base Estimators:** The model uses the faster, optional boosters (**XGBoost/LGBM/CatBoost**) if they are available.
    * **Final Estimator:** **GradientBoostingRegressor** combines the predictions from the base models for the final demand forecast.
    * **Data Pipeline:** The data is **Imputed** (fills missing values) and **Scaled** (normalized) before entering the stacking model.

---

## 3️⃣ Forecast Loop (2025–2050)

This iterative loop generates all future data displayed on the dashboard.

* **Iterative Prediction:** The loop runs from 2025 to 2050, **state by state**. It uses the predicted total generation from the previous step as a **lag feature** (`Electricity_Generation_lag1`) for the *current* year's prediction, simulating a time-series dependency.
* **Deterministic RNG (Key Feature):** A unique **seed** for the random number generator (`rng = np.random.default_rng(seed)`) is created using a hash of the **State name and the Year**.
    * *Why?* This ensures that the small fluctuations added to the Generation and Demand are **always the same** for a given (State, Year) pair, making the simulation results **stable** and **reproducible** across all dashboard interactions.

---

## 4️⃣ Local Recommendation Generator

This custom function replaces a Large Language Model (LLM).
We are planning to integrate this with transformers soon

* **Tokens:** A fixed list of **24 action phrases** (`PHRASES`) is defined (10 for surplus, 10 for deficit, 4 for timeframe).
* **Seeded Selection:** The `local_generate()` function uses the **same deterministic seed** logic (based on magnitude and state) as the forecast loop. This means the **recommendation is predictable** for a given energy outcome.
* **Mechanism:** It picks the top **1-2** actions based on a calculated score that is slightly randomized (controlled by `GEN_TEMPERATURE`) to give some variety.
* **Output:** It guarantees an output that **starts with an action** and ends with a calculated **Confidence** score and **Timeframe**.

---

## 5️⃣ UI and Visualization

* **User Inputs:** Uses a **Streamlit slider** (`st.slider`) to select the year (2019–2050) and a **selectbox** (`st.selectbox`) to pick the state for the trend chart.
* **Metrics:** It calculates and displays key metrics like **Validation Accuracy** and the **National Net Flow** (total surplus/deficit) using custom-styled **metric cards**.
* **Map (Choropleth):** It reads the `india_state.geojson` file with **geopandas**, merges the forecast data, and uses **Plotly Express** (`px.choropleth`) to color the states based on their surplus/deficit flow.
* **Trend Chart:** The time-series chart in the second tab uses **Plotly Graph Objects** (`go.Figure`) to plot **Actual** data (solid lines) alongside **Forecasted** data (dotted lines) for the selected state.


