import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
import nolds
import matplotlib.pyplot as plt
from scipy.fft import fft
from numpy import pi, sin

# Configuration
DATA_DIR = Path("data")
STABLECOINS = ["USDT", "USDC", "TUSD", "DAI", "PAX"]

# Rolling d estimator using Hurst
def rolling_d_estimates(prices, window=150):
    hurst_vals = []
    dates = []
    for i in range(len(prices) - window + 1):
        window_data = prices[i:i+window]
        if window_data.nunique() <= 1:
            hurst_vals.append(np.nan)
        else:
            h = nolds.hurst_rs(window_data)
            hurst_vals.append(h - 0.5)
        dates.append(prices.index[i + window - 1])
    return pd.Series(hurst_vals, index=dates)

def get_coin_name(ticker):
    names = {
        "USDT": "Tether",
        "USDC": "USD Coin",
        "TUSD": "TrueUSD",
        "DAI": "DAI",
        "PAX": "PAX Dollar"
    }
    return names.get(ticker, ticker)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Price Analysis", "Local Whittle Analysis"])

# ---------------- HOME ----------------
if page == "Home":
    st.title("🪙 The Instability of StableCoins")
    st.markdown("Bellini Alessandro, Bruscagin Alessandro, Cartella Gabriele, Gallazzi Luca")
    st.markdown("### Introduction to Python CLASS project ")
    st.markdown("""
    This web-app replicates key findings from the paper **_"The Instability of Stablecoins"_** by Duan & Urquhart (2023).  
    Use the sidebar to:
    - 🔍 **View current and historical prices**
    - 📈 **Analyze stability and stationarity**
    - 📊 **Replicate the tables from the paper**
    """)

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("Pic1.png", use_container_width=True)
    with col2:
        st.markdown("### 📂 Data Info")
        st.markdown("- ✅ **Data source:** Binance API (via `fetch_data.py`)")
        st.markdown("- 📁 **CSV folder:** `./data/`")
        st.markdown("- 🕒 **Frequency:** Hourly historical prices")
        st.code("python fetch_data.py", language="bash")

    st.markdown("---")
    st.markdown("🚀 **Start exploring from the sidebar.** Choose a section like `Price Analysis` or `Local Whittle Analysis`.")

# ---------------- PRICES ----------------
elif page == "Price Analysis":
    st.title("Current & Historical Prices")
    coin = st.selectbox("Select stablecoin", STABLECOINS)

    try:
        df_cur = pd.read_csv(DATA_DIR / "prices.csv", index_col="Coin")
        cur = df_cur.at[coin, 'Price'] if coin in df_cur.index else None
        st.metric(f"{coin} Price (USD)", f"${cur:.4f}" if cur else "N/A")
    except FileNotFoundError:
        st.error("Could not find 'prices.csv'. Please run `fetch_data.py` first.")

    try:
        df_hist = pd.read_csv(DATA_DIR / f"history_{coin}.csv", parse_dates=['Timestamp']).set_index('Timestamp')
        fig = px.line(df_hist, y='Price', title=f"{coin} Price History")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Price Clustering around $1.000")
        bins = [0.988, 0.990, 0.992, 0.994, 0.996, 0.998, 1.000, 1.002, 1.004, 1.006, 1.008, 1.010, 1.012]
        hist_fig = px.histogram(df_hist.reset_index(), x='Price', nbins=len(bins), title=f"{coin} Price Clustering")
        st.plotly_chart(hist_fig, use_container_width=True)

    except FileNotFoundError:
        st.warning(f"Price history for {coin} not found.")

# ---------------- DESCRIPTIVE STATS ----------------
elif page == "Descriptive Statistics":
    st.title("Descriptive Statistics of Stablecoins")

    st.subheader("Table 1: Descriptive Statistics")
    rows = []
    for coin in STABLECOINS:
        try:
            df = pd.read_csv(DATA_DIR / f"history_{coin}.csv", parse_dates=["Timestamp"])
            rows.append({
                "Name": get_coin_name(coin),
                "Ticker": coin,
                "Market cap": "N/A",
                "Start date": df["Timestamp"].min().date(),
                "End date": df["Timestamp"].max().date(),
                "No. obs": len(df),
                "Mean": df["Price"].mean(),
                "Median": df["Price"].median(),
                "Std. Dev": df["Price"].std(),
                "Max": df["Price"].max(),
                "Min": df["Price"].min()
            })
        except Exception as e:
            st.warning(f"Error with {coin}: {e}")
    df_table1 = pd.DataFrame(rows)
    df_table1 = df_table1[["Name", "Ticker", "Market cap", "Start date", "End date", "No. obs", "Mean", "Median", "Std. Dev", "Max", "Min"]]
    st.dataframe(df_table1.style.format("{:.4f}", subset=["Mean", "Median", "Std. Dev", "Max", "Min"]))

    st.subheader("Table 2: ADF Stationarity Test")
    adf_rows = []
    for coin in STABLECOINS:
        try:
            df = pd.read_csv(DATA_DIR / f"history_{coin}.csv")
            dev = df["Price"] - 1.0
            if dev.nunique() <= 1:
                adf_rows.append({"Coin": coin, "ADF Stat": np.nan, "p-value": np.nan, "Stability": "Stable", "Note": "Constant series"})
            else:
                stat, pval, *_ = adfuller(dev.dropna())
                stability = "Stable" if pval < 0.05 else "Unstable"
                adf_rows.append({"Coin": coin, "ADF Stat": stat, "p-value": pval, "Stability": stability, "Note": ""})
        except:
            adf_rows.append({"Coin": coin, "ADF Stat": np.nan, "p-value": np.nan, "Stability": "Unknown", "Note": "Error"})
    df_table2 = pd.DataFrame(adf_rows).set_index("Coin")
    df_table2_numeric = df_table2.copy()
    for col in ["ADF Stat", "p-value"]:
        df_table2_numeric[col] = pd.to_numeric(df_table2_numeric[col], errors="coerce")
    st.dataframe(df_table2_numeric.style.format("{:.4f}", subset=["ADF Stat", "p-value"]))

    st.subheader("Table 3: Fractional Integration Estimates (Hurst-based)")
    hurst_rows = []
    for coin in STABLECOINS:
        try:
            df = pd.read_csv(DATA_DIR / f"history_{coin}.csv")
            dev = df["Price"] - 1.0
            if dev.nunique() <= 1:
                continue
            hurst = nolds.hurst_rs(dev.dropna().values)
            d_est = hurst - 0.5
            hurst_rows.append({"Coin": coin, "Hurst (H)": hurst, "Estimated d (H-0.5)": d_est})
        except Exception as e:
            st.warning(f"Fractional integration failed for {coin}: {e}")
    df_hurst = pd.DataFrame(hurst_rows).set_index("Coin")
    st.dataframe(df_hurst.style.format("{:.4f}"))
    st.markdown("*Note: d is approximated as Hurst exponent (H) - 0.5 using `nolds`.*")

# ---------------- LOCAL WHITTLE ANALYSIS ----------------
elif page == "Local Whittle Analysis":
    st.title("Stationarity Analysis: Estimate of the d parameter (Local Whittle)")

    M_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8]
    M_FACTOR = 0.7

    def load_price_data(coin):
        df = pd.read_csv(DATA_DIR / f"history_{coin}.csv", parse_dates=['Timestamp'])
        df.set_index('Timestamp', inplace=True)
        return df['Price']

    def estimate_d_local_whittle(yt, m):
        T = len(yt)
        freqs = 2 * pi * np.arange(1, m + 1) / T
        periodogram = np.abs(fft(yt - np.mean(yt))[1:m + 1]) ** 2 / T
        x = np.log(4 * (np.sin(freqs / 2) ** 2))
        y = np.log(periodogram)
        b = np.polyfit(x, y, 1)
        return -b[0] / 2

    def estimate_d_multiple_m(yt, m_factors):
        return [estimate_d_local_whittle(yt, int(mf * len(yt))) for mf in m_factors]

    def rolling_lwest(yt, window_size, m_factor):
        T = len(yt)
        num_windows = T - window_size + 1
        d_rolling = np.zeros(num_windows)
        for i in range(num_windows):
            window = yt[i:i + window_size]
            m = min(int(m_factor * window_size), window_size // 2)
            d_rolling[i] = estimate_d_local_whittle(window, m)
        return d_rolling

    selected_coins = st.multiselect("Select the Stablecoins", STABLECOINS, default=STABLECOINS)
    show_rolling = st.checkbox("Show rolling d", value=True)

    summary_stats = []
    d_estimates_all = {}

    for coin in selected_coins:
        try:
            prices = load_price_data(coin)
        except FileNotFoundError:
            st.warning(f"Missing for {coin}")
            continue

        deviations = prices - 1.0
        deviations = deviations.fillna(method='ffill')

        stats = {
            'Coin': coin,
            'Start': prices.index.min(),
            'End': prices.index.max(),
            'N': len(prices),
            'Mean': prices.mean(),
            'Median': prices.median(),
            'StdDev': prices.std(),
            'Min': prices.min(),
            'Max': prices.max()
        }
        summary_stats.append(stats)

        d_estimates = estimate_d_multiple_m(deviations.values, M_VALUES)
        d_estimates_all[coin] = d_estimates

    if summary_stats:
        st.subheader("Summary Statistics")
        st.dataframe(pd.DataFrame(summary_stats).set_index('Coin'))

    if d_estimates_all:
        st.subheader("d Parameter estimation (Local Whittle)")

        d_table = pd.DataFrame(d_estimates_all, index=M_VALUES)
        d_table.index.name = 'm'

        try:
            representative_row = d_table.loc[0.7]
        except KeyError:
            st.error("m = 0.7 is not found in the d_table. Check M_VALUES consistency.")
            representative_row = d_table.iloc[len(d_table) // 2]

        def classify_stability(coin, d_val):
            if coin == "USDT":
                return "-"
            return "Stable" if d_val < 0.5 else "Unstable"

        stability_row = pd.Series(
            {coin: classify_stability(coin, d_val) for coin, d_val in representative_row.items()},
            name="Stability"
        )

        d_table_with_stability = pd.concat([d_table, pd.DataFrame([stability_row])])
        numeric_rows = d_table_with_stability.index[:-1]

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                d_table_with_stability.style.format("{:.4f}", subset=pd.IndexSlice[numeric_rows, :])
            )
        with col2:
            st.markdown(
                """
                <div style="background-color:#fff3cd; color:#333333; padding:15px; border-radius:10px; border-left:6px solid #ffa500">
                <strong style="color:#000000;">Interpretation:</strong><br>
                The estimated fractional integration order <code style="background:none; color:#000000;">d</code> for different bandwidth factors provides insights into the long-memory properties of price deviations.<br><br>
                • Lower <code style="background:none; color:#000000;">d</code> values suggest stronger mean reversion and higher stability.<br>
                • Higher <code style="background:none; color:#000000;">d</code> values indicate prolonged deviations and potential instability.
                </div>
                """,
                unsafe_allow_html=True
            )

    if show_rolling:
        st.subheader("Rolling d Estimate (Local Whittle)")
        window_slider = st.slider("Select rolling window size:", min_value=50, max_value=500, value=150, step=10)

        d_rolling_all = {}
        for coin in selected_coins:
            try:
                prices = load_price_data(coin)
                deviations = prices - 1.0
                deviations = deviations.fillna(method="ffill")
                d_rolling = rolling_lwest(deviations.values, window_slider, M_FACTOR)
                d_rolling_all[coin] = pd.Series(d_rolling, index=prices.index[window_slider - 1:])
            except Exception as e:
                st.warning(f"Could not compute rolling d for {coin}: {e}")

        if d_rolling_all:
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='none')
            ax.set_facecolor("none")
            for coin in d_rolling_all:
                ax.plot(d_rolling_all[coin], label=coin)
            ax.set_title(f"Rolling Local Whittle d Estimates (window = {window_slider})", color="white")
            ax.set_xlabel("Timestamp", color="white")
            ax.set_ylabel("Estimated d", color="white")
            ax.tick_params(colors='white')
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
            ax.legend()
            st.pyplot(fig)
