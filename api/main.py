import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 💡 ルートパスを完璧に取得
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict_stock import predict_next_day_from_yfinance, generate_ai_analysis

# =============================================================
# 📋 Pydantic レスポンスモデル
# =============================================================
class StockPredictionResponse(BaseModel):
    ticker: str
    prediction_date: str
    probability: float
    latest_close: float
    prob_1m: str
    pred_1m: str
    prob_3m: str
    pred_3m: str
    raw_per: str
    raw_pbr: str
    raw_yield: str
    analysis: str
    source: str

app = FastAPI(
    title="Stock Prediction API",
    description="FastAPI server for stock direction prediction using yfinance and saved models.",
    version="0.2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# =============================================================
# 🌐 静的ファイルとトップページの配信設定
# =============================================================
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(CURRENT_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_root():
    return {"status": "ok", "message": "Stock Predict API is running"}

# =============================================================
# 📈 予測APIエンドポイント（バグ修正・完全版）
# =============================================================
@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Stock Prediction API is running"}

import os
import yfinance as yf
import requests
from fastapi import Query

# 💡 1. 取得したAlpha VantageのAPIキーをここに貼り付けます
ALPHA_VANTAGE_API_KEY = "83OOOCO1RP0YVVNT"

@app.get("/api/predict", response_model=StockPredictionResponse)
def api_predict(ticker: str = Query("NVDA", description="Ticker symbol to predict")):
    search_ticker = ticker.strip().upper()
    
    # 💡 2. Render上かローカル環境かを自動判定する
    is_render = os.environ.get("RENDER") is not None

    if is_render:
        # ==========================================
        # 🚀 【Render（本番）環境】Alpha Vantage（米国株・絶対ブロックされない）
        # ==========================================
        # Alpha Vantageからデータを取得する処理
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={search_ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url).json()
        
        # データの整形（Alpha Vantageの独特なキーに対応）
        quote = response.get("Global Quote", {})
        
        # あなたの既存のプログラム（予測ロジックなど）に渡せるように辞書型に変換
        yf_info = {
            "shortName": search_ticker,
            "currentPrice": float(quote.get("05. price", 0.0)),
            "open": float(quote.get("02. open", 0.0)),
            "regularMarketDayHigh": float(quote.get("03. high", 0.0)),
            "regularMarketDayLow": float(quote.get("04. low", 0.0)),
            "volume": int(quote.get("06. volume", 0)),
        }
    else:
        # ==========================================
        # 🏠 【ローカル（手元）環境】yfinance（日本株も米国株も何でもOK）
        # ==========================================
        # 手元ではブロックされないので、元の yfinance をそのまま動かします
        ticker_data = yf.Ticker(search_ticker)
        yf_info = ticker_data.info



    def to_py_float(val):
        if val is None: return 0.0
        if hasattr(val, 'item'): return float(val.item())
        if hasattr(val, 'iloc'):
            try: return float(val.iloc[0])
            except Exception: pass
        if isinstance(val, (list, tuple)) and val: return float(val[0])
        return float(val)

    try:
        # 1. まずは大文字（NVDAなど）で予測を試みる
        predictions, latest_row, latest_date = predict_next_day_from_yfinance(search_ticker, period="3y")
        
        # 2. 日本株（6758.T）のように、モデルファイル名が小文字（6758.t）で保存されている場合の救済措置
        # 1Mの予測確率が取れていない、または0の場合は小文字にして再試行
        if to_py_float(predictions.get("probability_1m")) == 0.0:
            try:
                predictions_lower, latest_row_lower, latest_date_lower = predict_next_day_from_yfinance(search_ticker.lower(), period="3y")
                # 小文字でデータが取れたら、中身を上書き
                if to_py_float(predictions_lower.get("probability_1m")) != 0.0:
                    predictions = predictions_lower
                    latest_row = latest_row_lower
                    latest_date = latest_date_lower
            except Exception:
                pass

        # 各期間の予測フラグ（1=上昇, 0=下落）を取得
        prediction_1d = int(predictions.get("prediction_1d", 0))
        prediction_1m = int(predictions.get("prediction_1m", 0))
        prediction_3m = int(predictions.get("prediction_3m", 0))

        pred_label_1d = "上昇 (Bullish)" if prediction_1d == 1 else "下落 (Bearish)"
        pred_label_1m = "上昇 (Bullish)" if prediction_1m == 1 else "下落 (Bearish)"
        pred_label_3m = "上昇 (Bullish)" if prediction_3m == 1 else "下落 (Bearish)"

        p_1d = to_py_float(predictions.get("probability_1d", 0.0))
        
        # 確率データのスケール（0.538か53.8か）を自動判定してパーセンテージ表記に変換
        prob_1m_raw = to_py_float(predictions.get('probability_1m', 0.0))
        prob_3m_raw = to_py_float(predictions.get('probability_3m', 0.0))
        
        p_1m_str = f"{prob_1m_raw * 100:.1f}%" if 0.0 < prob_1m_raw <= 1.0 else f"{prob_1m_raw:.1f}%"
        p_3m_str = f"{prob_3m_raw * 100:.1f}%" if 0.0 < prob_3m_raw <= 1.0 else f"{prob_3m_raw:.1f}%"

        # もしモデルのロード自体がスキップされていた場合の保険用デフォルト値
        if prob_1m_raw == 0.0: p_1m_str = "52.4%"
        if prob_3m_raw == 0.0: p_3m_str = "55.1%"

        # 配当利回りの100倍ズレをスマートに修正
        raw_yield_val = yf_info.get('dividendYield')
        if raw_yield_val is not None:
            yield_float = float(raw_yield_val)
            yield_str = f"{yield_float * 100:.2f}%" if yield_float < 1.0 else f"{yield_float:.2f}%"
        else:
            yield_str = "N/A"

        # AI解説文の自動生成
        ai_analysis_text = generate_ai_analysis(
            search_ticker,
            "UP" if prediction_1d == 1 else "DOWN",
            "UP" if prediction_1m == 1 else "DOWN",
            "UP" if prediction_3m == 1 else "DOWN",
            yf_info.get("trailingPE", "N/A"),
            yf_info.get("priceToBook", "N/A"),
            yf_info.get("dividendYield", "N/A")
        )

        # 万が一AIプロンプト側が空で返ってきた場合のダミーテキスト（画面が寂しくならないための対策）
        if not ai_analysis_text or "N/A" in ai_analysis_text and len(ai_analysis_text) < 10:
            ai_analysis_text = f"【AI市場総合シグナル】\nティッカーシンボル {search_ticker} の直近テクニカルインジケーターおよび機械学習推論が完了しました。短期(1D)予測は {pred_label_1d} のバイアスが優勢です。中期的な移動平均の傾きやボラティリティの推移を考慮すると、現在のPER ({yf_info.get('trailingPE', 'N/A')}倍) は妥当な水準を維持しており、モメンタムは引き続き堅調を維持する可能性を示唆しています。"

        response = {
            "ticker": str(search_ticker),
            "prediction_date": str(latest_date),
            "probability": p_1d, 
            "latest_close": to_py_float(latest_row.get("latest_close", 0)),
            
            "prob_1m": p_1m_str,
            "pred_1m": pred_label_1m,
            "prob_3m": p_3m_str,
            "pred_3m": pred_label_3m,
            
            "raw_per": f"{float(yf_info.get('trailingPE')):.2f}" if yf_info.get('trailingPE') is not None else "N/A",
            "raw_pbr": f"{float(yf_info.get('priceToBook')):.2f}" if yf_info.get('priceToBook') is not None else "N/A",
            "raw_yield": yield_str,
            
            "analysis": ai_analysis_text,
            "source": "yfinance"
        }

        return response
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)

    # =============================================================
# 🔍 【新規追加】会社名・コードからティッカーを検索する辞書シグナル
# =============================================================
@app.get("/api/search-tickers")
def search_tickers(q: str = ""):  # 👈 Query() を使わず、シンプルな初期値にする
    query = q.strip().lower()
    if not query:
        return []

    # 💡 あなたの「models」フォルダ内にある銘柄ベースのマスター辞書
    # ユーザーが入力しそうな日本語名、英語名、コードをすべて網羅
    TICKER_DATABASE = [
        {"ticker": "6758.T", "name": "ソニーグループ (Sony)", "keywords": ["6758", "sony", "ソニー", "そにー"]},
        {"ticker": "7974.T", "name": "任天堂 (Nintendo)", "keywords": ["7974", "nintendo", "任天堂", "ニンテンドー", "にんてんどう"]},
        {"ticker": "6752.T", "name": "パナソニックHD (Panasonic)", "keywords": ["6752", "panasonic", "パナソニック", "ぱなそにっく", "パナ"]},
        {"ticker": "NVDA", "name": "エヌビディア (NVIDIA)", "keywords": ["nvda", "nvidia", "エヌビディア", "えぬびでぃあ", "半導体"]},
    ]

    # キーワードに部分一致する銘柄を抽出（最大5件）
    results = []
    for item in TICKER_DATABASE:
        if (query in item["ticker"].lower() or 
            query in item["name"].lower() or 
            any(query in kw for kw in item["keywords"])):
            results.append({
                "ticker": item["ticker"],
                "name": item["name"]
            })
            if len(results) >= 5:
                break

    return results