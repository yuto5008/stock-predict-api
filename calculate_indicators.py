import pandas as pd
import numpy as np
import os
from pathlib import Path


def load_csv(ticker: str, data_dir: str = "data"):
    """Load stock data from CSV file."""
    file_path = os.path.join(data_dir, f"{ticker}_history.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded: {file_path}")
    return df


def calculate_ma(df, column: str = 'Close', periods: list = None):
    """Calculate moving averages."""
    if periods is None:
        periods = [5, 20, 200]
    
    for period in periods:
        df[f'MA_{period}'] = df[column].rolling(window=period).mean()
    
    return df


def calculate_rsi(df, column: str = 'Close', period: int = 14):
    """Calculate RSI (Relative Strength Index)."""
    delta = df[column].diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi
    return df


def calculate_macd(df, column: str = 'Close', fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = df[column].ewm(span=fast).mean()
    ema_slow = df[column].ewm(span=slow).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_histogram
    
    return df


def calculate_bollinger_bands(df, column: str = 'Close', period: int = 20, std_dev: float = 2):
    """Calculate Bollinger Bands."""
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    df['BB_Upper'] = upper_band
    df['BB_Middle'] = sma
    df['BB_Lower'] = lower_band
    
    return df


def calculate_volume_ma(df, period: int = 20):
    """Calculate Volume Moving Average."""
    df['Volume_MA_20'] = df['Volume'].rolling(window=period).mean()
    return df


def calculate_technical_indicators(df):
    """Calculate technical indicators and add them as new columns."""
    
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # 1. 移動平均 (5, 20, 200日)
    df_copy = calculate_ma(df_copy, 'Close', [5, 20, 200])
    
    # 2. RSI (14日)
    df_copy = calculate_rsi(df_copy, 'Close', 14)
    
    # 3. MACD
    df_copy = calculate_macd(df_copy, 'Close')
    
    # 4. ボリンジャーバンド
    df_copy = calculate_bollinger_bands(df_copy, 'Close', 20, 2)
    
    # 5. 出来高の移動平均
    df_copy = calculate_volume_ma(df_copy, 20)
    
    return df_copy




def save_processed_csv(df, ticker: str, data_dir: str = "data"):
    """Save processed data with technical indicators."""
    file_path = os.path.join(data_dir, f"processed_{ticker}_history.csv")
    df.to_csv(file_path)
    print(f"✓ Saved: {file_path}")


def main():
    tickers = ["NVDA", "SONY"]
    
    for ticker in tickers:
        print(f"\n=== Processing {ticker} ===")
        
        # Load data
        df = load_csv(ticker)
        if df is None:
            continue
        
        # Calculate indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        # Show preview
        print(f"\nData shape: {df_with_indicators.shape}")
        print(f"\nColumns: {list(df_with_indicators.columns)}")
        print(f"\nLast 5 rows:")
        print(df_with_indicators.tail())
        
        # Save processed data
        save_processed_csv(df_with_indicators, ticker)
        print()


if __name__ == "__main__":
    main()
