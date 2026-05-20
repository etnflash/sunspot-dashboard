import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Sunspot Forecast", layout="wide")
st.title("Prophet Forecast with Preprocessed Sunspot Data")

@st.cache_data
def load_data():
    df = pd.read_csv("data/sunspots_for_prophet.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    return df

@st.cache_resource
def fit_model(data):
    model = Prophet(
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
    )
    model.add_seasonality(name="sunspot_cycle", period=11, fourier_order=5)
    model.fit(data)
    return model

df = load_data()

st.subheader("Data Preview")
st.dataframe(df.head())

model = fit_model(df)
future = model.make_future_dataframe(periods=30, freq="Y")
forecast = model.predict(future)

st.subheader("Prophet Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)

st.subheader("Forecast Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

st.subheader("Actual vs Predicted with Prediction Intervals")
fig3, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["ds"], df["y"], label="Actual", color="black")
ax.plot(forecast["ds"], forecast["yhat"], label="Predicted", color="blue")
ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    color="skyblue",
    alpha=0.3,
    label="Prediction Interval",
)
ax.set_xlabel("Year")
ax.set_ylabel("Sunspot Activity")
ax.legend()
ax.grid(True)
st.pyplot(fig3)

st.subheader("Residual Analysis")
merged = pd.merge(df, forecast[["ds", "yhat"]], on="ds", how="inner")
merged["residual"] = merged["y"] - merged["yhat"]

fig4, ax2 = plt.subplots(figsize=(14, 4))
ax2.plot(merged["ds"], merged["residual"], label="Residual", color="red")
ax2.axhline(0, color="black", linestyle="--")
ax2.set_xlabel("Year")
ax2.set_ylabel("Residual")
ax2.legend()
ax2.grid(True)
st.pyplot(fig4)

st.subheader("Residual Summary Statistics")
st.write(merged["residual"].describe())
