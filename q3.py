import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    if "YEAR" in df.columns:
        df["YEAR_INT"] = df["YEAR"].astype(int)
        df["DATE"] = pd.to_datetime(df["YEAR_INT"].astype(str), format="%Y")
        df.set_index("DATE", inplace=True)
    return df

def plot_advanced_sunspot_visualizations(
    df,
    sunactivity_col="SUNACTIVITY",
    hist_bins=30,
    trend_degree=1,
    point_size=10,
    point_alpha=0.5,
):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Sunspots Data Advanced Visualization", fontsize=18)

    axs[0, 0].plot(df.index, df[sunactivity_col], color="blue")
    axs[0, 0].set_title("Sunspot Activity Over Time")
    axs[0, 0].set_xlabel("Year")
    axs[0, 0].set_ylabel("Sunspot Count")
    axs[0, 0].grid(True)

    data = df[sunactivity_col].dropna().values
    if len(data) > 0:
        xs = np.linspace(data.min(), data.max(), 200)
        density = gaussian_kde(data)
        axs[0, 1].hist(data, bins=hist_bins, density=True, alpha=0.6, color="gray", label="Histogram")
        axs[0, 1].plot(xs, density(xs), color="red", linewidth=2, label="Density")
    axs[0, 1].set_title("Distribution of Sunspot Activity")
    axs[0, 1].set_xlabel("Sunspot Count")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    try:
        df_20th = df.loc["1900":"2000"]
        if not df_20th.empty:
            axs[1, 0].boxplot(df_20th[sunactivity_col], vert=False)
    except Exception:
        pass
    axs[1, 0].set_title("Boxplot of Sunspot Activity (1900-2000)")
    axs[1, 0].set_xlabel("Sunspot Count")

    years = df["YEAR"].values
    sun_activity = df[sunactivity_col].values
    mask = ~np.isnan(sun_activity)
    years_clean = years[mask]
    sun_activity_clean = sun_activity[mask]

    if len(years_clean) > trend_degree:
        axs[1, 1].scatter(years_clean, sun_activity_clean, s=point_size, alpha=point_alpha, label="Data Points")
        coef = np.polyfit(years_clean, sun_activity_clean, trend_degree)
        trend = np.poly1d(coef)
        x_trend = np.linspace(years_clean.min(), years_clean.max(), 100)
        axs[1, 1].plot(x_trend, trend(x_trend), color="red", linewidth=2, label="Trend Line")
    axs[1, 1].set_title("Trend of Sunspot Activity")
    axs[1, 1].set_xlabel("Year")
    axs[1, 1].set_ylabel("Sunspot Count")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

st.title("Sunspot Data Analysis Dashboard")
st.markdown("This dashboard shows sunspot data with interactive controls.")

try:
    df = load_data("data/sunspots.csv")

    st.sidebar.header("Visualization Options")
    min_year = int(df["YEAR"].min())
    max_year = int(df["YEAR"].max())
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
    hist_bins = st.sidebar.slider("Histogram Bins", min_value=5, max_value=100, value=30)
    trend_degree = st.sidebar.slider("Trend Degree", min_value=1, max_value=5, value=1)
    point_size = st.sidebar.slider("Scatter Point Size", min_value=1, max_value=50, value=10)
    point_alpha = st.sidebar.slider("Scatter Point Alpha", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    filtered_df = df[(df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])]

    if not filtered_df.empty:
        st.subheader("Sunspot Data Advanced Visualization")
        fig = plot_advanced_sunspot_visualizations(
            filtered_df,
            hist_bins=hist_bins,
            trend_degree=trend_degree,
            point_size=point_size,
            point_alpha=point_alpha,
        )
        st.pyplot(fig)
    else:
        st.warning("No data in the selected period.")
except Exception as e:
    st.error(f"Error: {e}")
    st.info("Check that data/sunspots.csv exists and has YEAR and SUNACTIVITY columns.")
