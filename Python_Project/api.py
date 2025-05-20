from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Stablecoin Price API (CSV)",
    description="Serve USD prices from local CSV files",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("data")
STABLECOINS = ["USDT", "USDC", "BUSD", "DAI", "PAX"]

@app.get("/prices")
def get_prices():
    """
    Read `data/prices.csv` and return current prices as JSON.
    """
    csv_path = DATA_DIR / "prices.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=503, detail="Prices CSV not found")
    df = pd.read_csv(csv_path, index_col="Coin")
    return df["Price"].to_dict()

@app.get("/history/{symbol}")
def get_history(symbol: str):
    """
    Read `data/history_{symbol}.csv` and return historical dates & prices.
    """
    symbol = symbol.upper()
    if symbol not in STABLECOINS:
        raise HTTPException(status_code=404, detail="Symbol not supported")
    csv_path = DATA_DIR / f"history_{symbol}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=503, detail="History CSV not found")
    df = pd.read_csv(csv_path)
    return {"dates": df["Date"].tolist(), "prices": df["Price"].tolist()}
