import datetime
import os
from pathlib import Path
import yfinance as yf


def fetch_stock_data(ticker: str, start: str, end: str):
    """Fetch historical stock data for a given ticker between start and end dates."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    return df


def save_to_csv(df, ticker: str, data_dir: str = "data"):
    """Save DataFrame to CSV file in the data directory."""
    # Create data directory if it doesn't exist
    Path(data_dir).mkdir(exist_ok=True)
    
    # Define the file path
    file_path = os.path.join(data_dir, f"{ticker}_history.csv")
    
    # Save to CSV
    df.to_csv(file_path)
    print(f"✓ Saved: {file_path}")


def main():
    # ここで取得したい銘柄と期間を指定します
    tickers = ["NVDA", "SONY"]
    start_date = "2023-01-01"
    end_date = datetime.date.today().isoformat()

    for ticker in tickers:
        print(f"=== {ticker} ===")
        df = fetch_stock_data(ticker, start_date, end_date)
        print(f"Data shape: {df.shape}")
        print(df.head())
        
        # Save to CSV
        save_to_csv(df, ticker)
        print()


if __name__ == "__main__":
    main()
