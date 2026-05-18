import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import yfinance as yf
from calculate_indicators import calculate_technical_indicators
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TICKER_ALIASES = {
    "BMW": "BMWYY",
    "SONY": "SONY",
    "NVDA": "NVDA",
}


def normalize_ticker(ticker: str) -> str:
    ticker_code = ticker.strip().upper()
    return TICKER_ALIASES.get(ticker_code, ticker_code)


def download_stock_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    ticker_symbol = normalize_ticker(ticker)
    try:
        df = yf.download(
            tickers=ticker_symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if df is None or df.empty:
            raise FileNotFoundError(f"yfinance returned no data for ticker: {ticker_symbol}")

        print(f"DEBUG: downloaded {len(df)} rows for {ticker_symbol} using period={period} interval={interval}")
        print(df.tail())

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        elif len(df.columns) > 0 and isinstance(df.columns[0], tuple):
            df.columns = [col[0] for col in df.columns]

        # マルチインデックス平坦化後の重複カラムを排除
        df = df.loc[:, ~df.columns.duplicated()]

        col_map = {}
        for c in df.columns:
            c_str = str(c)
            key = c_str.strip().lower()
            if key == "open":
                col_map[c] = "Open"
            elif key == "high":
                col_map[c] = "High"
            elif key == "low":
                col_map[c] = "Low"
            elif key in ("close", "adj close", "adj_close", "adjclose"):
                col_map[c] = "Close"
            elif key == "volume":
                col_map[c] = "Volume"
            else:
                col_map[c] = c

        df = df.rename(columns=col_map)

        # さらに同名カラムが重複していれば先頭のみ残す
        df = df.loc[:, ~df.columns.duplicated()]

        if "Close" in df.columns:
            df = df.dropna(subset=["Close"])

        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns from yfinance for {ticker_symbol}: {missing}. Available columns: {list(df.columns)}")

        df = df[expected_columns].copy()
        df = df.dropna(subset=expected_columns)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.index.name = "Date"
        print(f"DEBUG: cleaned {len(df)} rows after dropna for {ticker_symbol}")
        print(df.tail())
        return df
    except Exception as e:
        print(f"ERROR: download_stock_history failed for {ticker_symbol}: {e}")
        raise


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Create target and feature matrix with comprehensive technical indicators.

    This function expects technical indicators to already be present on `df`.
    It computes additional time-series features, creates the `Target` column,
    and returns the cleaned dataframe and feature list.
    """
    df = df.copy()

    # 完全なテクニカル特徴量リスト
    features = [
        # 基本列
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        # 移動平均線
        "MA_5",
        "MA_20",
        "MA_200",
        # RSI
        "RSI",
        # MACD
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        # ボリンジャーバンド
        "BB_Upper",
        "BB_Middle",
        "BB_Lower",
        # 出来高
        "Volume_MA_20",
    ]

    if "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError("Dataframe must contain 'Close' and 'Volume' columns to prepare features")

    # 追加の時系列特徴量
    df["Return_1"] = df["Close"].pct_change(1)
    df["Return_3"] = df["Close"].pct_change(3)
    df["Volume_Change"] = df["Volume"].pct_change(1)
    features.extend(["Return_1", "Return_3", "Volume_Change"])

    # Create targets for multiple timelines
    # 明日（1営業日後）
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    # 約1ヶ月後（20営業日後）
    df["Target_1m"] = (df["Close"].shift(-20) > df["Close"]).astype(int)
    # 約3ヶ月後（60営業日後）
    df["Target_3m"] = (df["Close"].shift(-60) > df["Close"]).astype(int)

    # 利用可能な特徴量のみを確保
    available_features = [col for col in features if col in df.columns]
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        print(f"WARNING: Missing technical indicators (will proceed with available): {missing_features}")

    # NaNを排除（技術指標計算による最初の行は除去される）
    # 3つのターゲット列すべてに対応
    df = df.dropna(subset=available_features + ["Target", "Target_1m", "Target_3m"]).copy()
    # Target_3m の最後の 60 行を除去（3ヶ月後の目標がない）
    df = df.iloc[:-60]

    if df.empty:
        raise ValueError("予測に必要な十分なデータがありません。有効な行数が足りません。")

    return df, available_features


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def get_model_path(ticker: str, timeline: str = "1d") -> Path:
    """Get model path for a given ticker and timeline.
    
    Args:
        ticker: Stock ticker symbol
        timeline: "1d" (tomorrow), "1m" (1 month), "3m" (3 months)
    """
    return MODEL_DIR / f"{ticker.strip().lower()}_{timeline}_stock_classifier.joblib"


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and drop initial NaNs created by those indicators.

    This function does NOT create the `Target` column.
    """
    # 1. Build a clean DataFrame containing only 1D Series for primary columns
    clean_df = pd.DataFrame(index=df.index)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            series_data = df[col]
            # If the column itself is a DataFrame (e.g., from MultiIndex flattening), take first sub-column
            if isinstance(series_data, pd.DataFrame):
                series_data = series_data.iloc[:, 0]
            # If Series but unexpectedly multi-dimensional, squeeze to 1D
            if hasattr(series_data, "ndim") and getattr(series_data, "ndim", 1) > 1:
                series_data = series_data.squeeze()

            # Store as a plain 1D Series in clean_df
            clean_df[col] = pd.Series(series_data.values, index=df.index, name=col)
        else:
            raise ValueError(f"Required column {col} missing for technical analysis.")

    # 2. Compute technical indicators on the clean DataFrame and drop initial NaNs
    clean_df = calculate_technical_indicators(clean_df)
    clean_df = clean_df.dropna()
    return clean_df


def load_processed_csv(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    file_path = data_dir / f"processed_{ticker}_history.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    return df


def train_model_from_history(df: pd.DataFrame, ticker: str, timeline: str = "1d") -> Pipeline:
    """Train a model for a specific timeline.
    
    Args:
        df: Historical price dataframe
        ticker: Stock ticker symbol
        timeline: "1d" (tomorrow), "1m" (1 month), "3m" (3 months)
    """
    df_features = compute_technical_features(df)
    df_prepared, features = prepare_data(df_features)

    if df_prepared.empty:
        raise ValueError("学習用データが不足しています。過去の株価データを増やしてください。")

    X = df_prepared[features]
    # Select target column based on timeline
    target_col_map = {"1d": "Target", "1m": "Target_1m", "3m": "Target_3m"}
    target_col = target_col_map.get(timeline, "Target")
    y = df_prepared[target_col]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    model_path = get_model_path(ticker, timeline)
    joblib.dump(pipeline, model_path)
    return pipeline


def safe_number(value):
    try:
        return float(value)
    except Exception:
        return None


def get_financial_metrics(ticker: str) -> dict:
    ticker_symbol = normalize_ticker(ticker)
    try:
        info = yf.Ticker(ticker_symbol).info or {}
    except Exception:
        info = {}

    return {
        "trailingPE": safe_number(info.get("trailingPE")),
        "priceToBook": safe_number(info.get("priceToBook")),
        "dividendYield": safe_number(info.get("dividendYield")),
    }


def build_ai_analysis_text(ticker: str, metrics: dict, predictions: dict) -> str:
    parts = []
    pe = metrics.get("trailingPE")
    pb = metrics.get("priceToBook")
    dy = metrics.get("dividendYield")

    if pe is not None:
        parts.append(f"PERは約{pe:.1f}倍で、収益力に対する株価の評価を確認しています。")
    if pb is not None:
        parts.append(f"PBRは約{pb:.2f}倍で、資産価値とのバランスを見ています。")
    if dy is not None:
        parts.append(f"配当利回りは{dy * 100:.2f}%で、配当収入の魅力も見ています。")
    if not parts:
        parts.append("主要財務指標が取得できないため、価格動向とテクニカル指標を中心に分析しました。")

    horizon_texts = []
    for label, key in [("翌日", "prediction_1d"), ("1ヶ月後", "prediction_1m"), ("3ヶ月後", "prediction_3m")]:
        pred = predictions.get(key)
        if pred is None:
            horizon_texts.append(f"{label}の方向性は予測できませんでした。")
        else:
            horizon_texts.append(f"{label}は{'強含み' if pred == 1 else '弱含み'}の見通しです。")

    parts.append(" ".join(horizon_texts))
    return " ".join(parts)


def load_or_train_model(ticker: str, timeline: str = "1d", history_df: pd.DataFrame = None) -> Pipeline:
    """Load or train model for a specific timeline."""
    model_path = get_model_path(ticker, timeline)
    if model_path.exists():
        return joblib.load(model_path)

    if history_df is None:
        history_df = download_stock_history(ticker, period="5y")

    return train_model_from_history(history_df, ticker, timeline)


def train_and_evaluate(ticker: str, test_size: float = 0.2, random_state: int = 42):
    df = load_processed_csv(ticker)
    df_prepared, features = prepare_data(df)

    X = df_prepared[features]
    y = df_prepared["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(f"=== {ticker} モデル評価 ===")
    print(f"学習データ: {X_train.shape[0]} 行, テストデータ: {X_test.shape[0]} 行")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = get_model_path(ticker)
    joblib.dump(pipeline, model_path)
    print(f"モデルを保存しました: {model_path}\n")

    return pipeline, features


def predict_next_day(ticker: str, model_path: Path = None) -> Tuple[int, float]:
    if model_path is None:
        model_path = get_model_path(ticker)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    df = load_processed_csv(ticker)
    df_prepared, features = prepare_data(df)

    X_latest = df_prepared[features].iloc[[-1]]
    probability = pipeline.predict_proba(X_latest)[0][1]
    prediction = int(pipeline.predict(X_latest)[0])

    print(f"{ticker} の翌日予測: {'上がる(1)' if prediction == 1 else '下がる(0)'}")
    print(f"上昇確率: {probability:.4f}")
    return prediction, probability


def predict_next_day_from_yfinance(ticker: str, period: str = "3y") -> Tuple[dict, dict, str]:
    history_df = download_stock_history(ticker, period=period)
    if history_df.empty:
        raise ValueError("最新の株価データが取得できませんでした。")

    latest_row_series = history_df.iloc[-1]
    latest_date = latest_row_series.name.strftime("%Y-%m-%d")

    df_features = compute_technical_features(history_df)
    df_prepared, features = prepare_data(df_features)

    if df_prepared.empty:
        raise ValueError("予測に必要な十分なデータがありません。")

    X_latest = df_prepared[features].iloc[[-1]]

    prediction_1d_model = load_or_train_model(ticker, timeline="1d", history_df=history_df)
    prediction_1m_model = load_or_train_model(ticker, timeline="1m", history_df=history_df)
    prediction_3m_model = load_or_train_model(ticker, timeline="3m", history_df=history_df)

    probability_1d = float(prediction_1d_model.predict_proba(X_latest)[0][1])
    prediction_1d = int(prediction_1d_model.predict(X_latest)[0])

    probability_1m = float(prediction_1m_model.predict_proba(X_latest)[0][1])
    prediction_1m = int(prediction_1m_model.predict(X_latest)[0])

    probability_3m = float(prediction_3m_model.predict_proba(X_latest)[0][1])
    prediction_3m = int(prediction_3m_model.predict(X_latest)[0])

    latest_row_dict = {
        "latest_open": float(latest_row_series["Open"]),
        "latest_high": float(latest_row_series["High"]),
        "latest_low": float(latest_row_series["Low"]),
        "latest_close": float(latest_row_series["Close"]),
        "latest_volume": int(latest_row_series["Volume"]),
    }

    metrics = get_financial_metrics(ticker)
    ai_analysis = build_ai_analysis_text(ticker, metrics, {
        "prediction_1d": prediction_1d,
        "prediction_1m": prediction_1m,
        "prediction_3m": prediction_3m,
    })

    return {
        "prediction_1d": prediction_1d,
        "probability_1d": probability_1d,
        "prediction_1m": prediction_1m,
        "probability_1m": probability_1m,
        "prediction_3m": prediction_3m,
        "probability_3m": probability_3m,
        "ai_analysis": ai_analysis,
    }, latest_row_dict, latest_date


def main():
    parser = argparse.ArgumentParser(description="Train or predict stock direction for a ticker.")
    parser.add_argument("ticker", nargs="?", default="NVDA", help="Ticker symbol to process (e.g. NVDA, SONY, BMW)")
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="both", help="Operation mode")
    args = parser.parse_args()

    ticker = args.ticker.strip().upper()

    if args.mode in ["train", "both"]:
        history_df = download_stock_history(ticker, period="720d")
        train_model_from_history(history_df, ticker)

    if args.mode in ["predict", "both"]:
        prediction, probability, latest_row, latest_date = predict_next_day_from_yfinance(ticker)
        print(f"{ticker} prediction: {prediction} ({'up' if prediction == 1 else 'down'})")
        print(f"probability: {probability:.4f}")
        print(f"latest_close: {latest_row['Close']}")


if __name__ == "__main__":
    main()

# AIアナリスト風の解説を生成する関数（OpenAI APIを使用）
def generate_ai_analysis(ticker, pred_1d, pred_1m, pred_3m, pe, pbr, yield_pct):
    """純粋にプロのアナリスト解説文（3行）だけを生成する"""
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    from pathlib import Path

    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        pe_str = f"{float(pe):.2f}" if pe != "N/A" else "N/A"
        pbr_str = f"{float(pbr):.2f}" if pbr != "N/A" else "N/A"
    except Exception:
        pe_str, pbr_str = str(pe), str(pbr)

    # 元の簡易判定ロジック
    financial_notes = []
    try:
        if pe != "N/A" and float(pe) < 15: financial_notes.append("PER基準で割安圏内")
        elif pe != "N/A" and float(pe) > 25: financial_notes.append("PER基準で割高警戒")
        if pbr != "N/A" and float(pbr) < 1.0: financial_notes.append("PBR1倍割れ（解散価値以下）")
    except Exception: pass
    financial_status = "、".join(financial_notes) if financial_notes else "財務指標は概ねセクター平均水準"
    
    if not api_key or api_key.startswith("your-"):
        return "【システムシグナル】テクニカルモデルに基づき、現在トレンドを検証中です。"
        
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
        あなたは日系大手証券会社のチーフ・クオンツ・アナリストです。
        以下のデータを厳密に評価し、機関投資家向けのレポートとして通用する、簡潔でプロフェッショナルな見通しを【3行の美しい日本語】で出力してください。
        対象銘柄: {ticker} | 翌日={pred_1d}, 1ヶ月後={pred_1m}, 3ヶ月後={pred_3m} | PER={pe_str}倍, PBR={pbr_str}倍, 利回り={yield_pct} | 財務評: {financial_status}
        ■ 制約条件: 「〜と考えられます」は禁止。「〜を示唆」「〜に留意したい」等の硬派な文体で、1行目に財務と短期、2行目に中長期、3行目にリスクを述べること。
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "マルチタイムフレーム予測モデルに基づき、現在のトレンドと財務健全性を検証中です。"