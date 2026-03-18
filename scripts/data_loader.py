import requests
import pandas as pd
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

#------------------- Fetch Historical Data from Upstox API and Store in CSV File -------------------
def gen_session():
    with open("data\\access_token.txt", "r") as f:
        return f.read().strip()


def get_historical_data(token, symbol,a, tf, start_date, end_date):
    #a =  minutes / days / weeks / months
    url = f"https://api.upstox.com/v3/historical-candle/{symbol}/{a}/{tf}/{start_date}/{end_date}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()["data"]["candles"]
    else:
        print(f"{symbol} | {response.status_code} | {response.text}")
        return None

def fetch_history(token, row, start_date, end_date):
    instrument_key = row["instrument_key"]
    trading_symbol = row["trading_symbol"]

    candles = get_historical_data(
        token,
        instrument_key,
        a="days",
        tf="1",
        start_date=start_date,
        end_date=end_date
    )

    if not candles:
        return None

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    df["symbol"] = trading_symbol
    df["instrument_key"] = instrument_key

    return df[["symbol", "instrument_key", "open", "high", "low", "close", "volume", "timestamp"]].sort_values("timestamp")

def get_data():
    token = gen_session()
    symbols_df = pd.read_csv("data\\symbols.csv")
    all_data = []

    for _, row in symbols_df.iterrows():
        df = fetch_history(token, row, "2026-02-01", "2020-12-01")
        if df is not None:
            all_data.append(df)

    if all_data:
        final_df_day = pd.concat(all_data, ignore_index=True)
        final_df_day.to_csv("data\\historical_data.csv", index=False)
        print("Done")
    else:
        print("No data fetched.")

#----------------------------------------------------------------------------------------

if __name__ == "__main__":
    #get_data()
    data = pd.read_csv("data\\historical_data.csv")
    print(data.head())
    
