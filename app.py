
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Configure the application
st.set_page_config(
    page_title="SEA Clean Electricity | ML Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Define project paths
PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "01_Data"
RESULTS_PATH = PROJECT_PATH / "05_Results"
MODELS_PATH = PROJECT_PATH / "04_Models"

# Load datasets and saved models
@st.cache_data
def load_data():
    historical = pd.read_csv(
        DATA_PATH / "SEA_Clean_Electricity_2021_2026.csv",
        parse_dates=["date"]
    )

    comparison_2025 = pd.read_csv(
        RESULTS_PATH / "model_comparison_2025.csv"
    )

    comparison_2026 = pd.read_csv(
        RESULTS_PATH / "model_comparison_2026.csv"
    )

    predictions_2025 = pd.read_csv(
        RESULTS_PATH / "predictions_2025.csv",
        parse_dates=["date"]
    )

    predictions_2026 = pd.read_csv(
        RESULTS_PATH / "predictions_2026.csv",
        parse_dates=["date"]
    )

    forecast = pd.read_csv(
        RESULTS_PATH / "next_month_forecast.csv",
        parse_dates=["forecast_month"]
    )

    return (
        historical,
        comparison_2025,
        comparison_2026,
        predictions_2025,
        predictions_2026,
        forecast
    )


@st.cache_resource
def load_model():
    return joblib.load(
        MODELS_PATH / "linear_regression_model.joblib"
    )


(
    historical_df,
    comparison_2025,
    comparison_2026,
    predictions_2025,
    predictions_2026,
    forecast_df
) = load_data()

# Sidebar controls
st.sidebar.title("⚡ ML Forecasting Controls")

countries = sorted(historical_df["country"].unique())

selected_country = st.sidebar.selectbox(
    "Select Country",
    countries
)

selected_period = st.sidebar.selectbox(
    "Evaluation Period",
    ["2025 Test", "2026 Validation"]
)

# Filter historical data
country_df = historical_df[
    historical_df["country"] == selected_country
].sort_values("date")

latest_record = country_df.iloc[-1]

# Dashboard header
st.title("⚡ Southeast Asia Clean Electricity Forecasting")

st.write(
    "Machine learning-based analysis and next-month prediction "
    "of clean electricity share in five Southeast Asian countries."
)

st.caption(
    "Models trained on 2022–2024 data | "
    "2025 testing | 2026 validation"
)

# SECTION 1: Prediction Overview
st.divider()
st.header("1. Prediction Overview")

selected_forecast = forecast_df[
    forecast_df["country"] == selected_country
].iloc[0]

forecast_month = selected_forecast[
    "forecast_month"
].strftime("%Y-%m")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Forecast Month",
        forecast_month
    )

with col2:
    st.metric(
        "Predicted Clean Share",
        f"{selected_forecast['predicted_clean_share']:.2f}%",
        delta=(
            f"{selected_forecast['predicted_change_pp']:+.2f} pp"
        )
    )

with col3:
    st.metric(
        "Latest Actual Share",
        f"{selected_forecast['latest_actual_share']:.2f}%"
    )

st.info(
    "Forecasts are one-step-ahead predictions based on "
    "the latest available monthly observations. "
    "Forecast months differ across countries due to "
    "different data availability."
)

# SECTION 2: Model Performance
st.divider()
st.header("2. Machine Learning Model Performance")

comparison = (
    comparison_2025
    if selected_period == "2025 Test"
    else comparison_2026
)

st.subheader(f"Model Comparison: {selected_period}")

st.dataframe(
    comparison.round(4),
    hide_index=True,
    use_container_width=True
)

st.subheader("Model Prediction Error (MAE)")

st.bar_chart(
    comparison.set_index("Model")["MAE"],
    y_label="MAE (percentage points)"
)

best_model = comparison.loc[
    comparison["MAE"].idxmin()
]

st.success(
    f"Best overall model: {best_model['Model']} "
    f"(MAE = {best_model['MAE']:.4f} percentage points)"
)


# SECTION 3: Actual vs Predicted
st.divider()
st.header("3. Actual vs Predicted")

prediction_df = (
    predictions_2025
    if selected_period == "2025 Test"
    else predictions_2026
)

country_predictions = prediction_df[
    prediction_df["country"] == selected_country
].sort_values("date")

st.write(
    f"Actual and predicted monthly clean electricity share "
    f"for {selected_country} during {selected_period}."
)

# Prepare the actual and predicted values
chart_columns = [
    "clean_share",
    "Seasonal Naive",
    "Linear Regression",
    "Random Forest"
]

chart_data = country_predictions[
    ["date"] + chart_columns
].copy()

chart_data = chart_data.rename(
    columns={"clean_share": "Actual"}
)

chart_data = chart_data.set_index("date")

# Display all model predictions against actual values
st.line_chart(
    chart_data,
    y_label="Clean Electricity Share (%)"
)

st.caption(
    "Actual values are compared with one-step-ahead "
    "predictions from three models using observed historical "
    "lag features."
)

# Calculate country-level prediction errors
st.subheader("Prediction Error by Model")

actual_values = country_predictions["clean_share"]

error_summary = pd.DataFrame({
    "Model": [
        "Seasonal Naive",
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        (
            actual_values - country_predictions[model_name]
        ).abs().mean()
        for model_name in [
            "Seasonal Naive",
            "Linear Regression",
            "Random Forest"
        ]
    ]
})

st.dataframe(
    error_summary.round(4),
    hide_index=True,
    use_container_width=True
)

best_country_model = error_summary.loc[
    error_summary["MAE"].idxmin()
]

st.info(
    f"Best model for {selected_country} in {selected_period}: "
    f"{best_country_model['Model']} "
    f"(MAE = {best_country_model['MAE']:.4f} percentage points)."
)

# SECTION 4: Interactive Prediction
st.divider()
st.header("4. Interactive Next-Month Prediction")

st.write(
    "Click the button to generate a prediction using "
    "the trained Linear Regression model."
)

if st.button("Generate Prediction", type="primary"):

    model = load_model()

    next_month = (
        latest_record["date"] + pd.DateOffset(months=1)
    )

    history = country_df.set_index("date")["clean_share"]

    required_dates = {
        "lag_1": next_month - pd.DateOffset(months=1),
        "lag_3": next_month - pd.DateOffset(months=3),
        "lag_12": next_month - pd.DateOffset(months=12)
    }

    if not all(
        date in history.index
        for date in required_dates.values()
    ):
        st.error(
            "Insufficient historical data for prediction."
        )

    else:
        input_data = pd.DataFrame([{
            "country": selected_country,
            "lag_1": history.loc[required_dates["lag_1"]],
            "lag_3": history.loc[required_dates["lag_3"]],
            "lag_12": history.loc[required_dates["lag_12"]],
            "month": next_month.month
        }])

        predicted_share = model.predict(input_data)[0]

        st.success("Prediction generated successfully!")

        st.metric(
            "Predicted Clean Electricity Share",
            f"{predicted_share:.2f}%"
        )

        st.write(
            "Forecast Month:",
            next_month.strftime("%Y-%m")
        )

# SECTION 5: Historical Data Explorer
st.divider()
st.header("5. Historical Data Explorer")

st.subheader("Historical Clean Electricity Share")

st.line_chart(
    country_df.set_index("date")["clean_share"],
    y_label="Clean Electricity Share (%)"
)

st.subheader("Historical Clean Electricity Generation")

st.line_chart(
    country_df.set_index("date")["clean_generation_twh"],
    y_label="Generation (TWh)"
)

# Footer
st.divider()

st.caption(
    "Data source: Ember | "
    "Models: Seasonal Naive, Linear Regression, Random Forest | "
    "Study period: 2021–2026 (2026 data are incomplete)"
)
