import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# Directory to store CSVs
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Configuration: number of hours to keep in historical CSVs
HIST_HOURS = 24 * 30  # last 30 days of hourly data

# Stablecoins and their Binance trading symbols
STABLECOINS = {
    "USDT": "USDTUSDT",   # dummy 1.0
    "USDC": "USDCUSDT",
    "TUSD": "TUSDUSDT",
    "DAI":  "USDTDAI",     # will invert
    "PAX":  "USDPUSDT",
}

# Binance API endpoints
BASE_URL = "https://api.binance.com"
PRICE_ENDPOINT = "/api/v3/ticker/price"
KLINES_ENDPOINT = "/api/v3/klines"

# Fetch current prices and save to prices.csv
def fetch_current_prices():
    records = []
    for coin, symbol in STABLECOINS.items():
        if coin == 'USDT':
            price = 1.0
        else:
            try:
                resp = requests.get(BASE_URL + PRICE_ENDPOINT, params={"symbol": symbol})
                resp.raise_for_status()
                raw_price = float(resp.json().get('price', None))
                price = 1.0 / raw_price if coin == 'DAI' else raw_price
            except Exception as e:
                print(f"Error fetching price for {coin} ({symbol}): {e}")
                price = None
        records.append({"Coin": coin, "Price": price})

    df = pd.DataFrame(records).set_index('Coin')
    df.to_csv(data_dir / 'prices.csv')
    print(f"Saved current prices to {data_dir / 'prices.csv'}")

# Fetch hourly historical prices for each coin (up to HIST_HOURS)
def fetch_historical(symbol: str, limit: int = HIST_HOURS, inverse: bool = False):
    try:
        params = {'symbol': symbol, 'interval': '1h', 'limit': limit}
        resp = requests.get(BASE_URL + KLINES_ENDPOINT, params=params)
        resp.raise_for_status()
        klines = resp.json()
        records = []
        for k in klines:
            ts = int(k[0]) // 1000
            timestamp = datetime.utcfromtimestamp(ts)
            price = float(k[4])
            if inverse:
                price = 1.0 / price
            records.append({'Timestamp': timestamp, 'Price': price})
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return pd.DataFrame(columns=['Timestamp', 'Price'])

# Main: fetch current + historical
if __name__ == '__main__':
    fetch_current_prices()

    for coin, symbol in STABLECOINS.items():
        if coin == 'USDT':
            timestamps = pd.date_range(end=datetime.utcnow(), periods=HIST_HOURS, freq='H')
            df_hist = pd.DataFrame({'Timestamp': timestamps, 'Price': 1.0})
            print(f"Prepared dummy hourly history for {coin} ({len(df_hist)} rows)")
        else:
            inverse = (coin == 'DAI')
            df_hist = fetch_historical(symbol, limit=HIST_HOURS, inverse=inverse)
            print(f"Fetched hourly history for {coin} ({len(df_hist)} rows)")

        df_hist.to_csv(data_dir / f'history_{coin}.csv', index=False)
        print(f"Saved hourly history for {coin} to {data_dir / f'history_{coin}.csv'}")
