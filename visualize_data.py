import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def load_processed_csv(ticker: str, data_dir: str = "data"):
    """Load processed stock data with technical indicators."""
    file_path = os.path.join(data_dir, f"processed_{ticker}_history.csv")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded: {file_path}")
    return df


def create_chart(df, ticker: str):
    """Create interactive chart with technical indicators using Plotly."""
    
    # Create subplots: 4 rows (Price, RSI, MACD, Volume)
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=(f"{ticker} - Price & Bollinger Bands", "RSI", "MACD", "Volume")
    )
    
    # ===== Row 1: Price Chart with Moving Averages and Bollinger Bands =====
    
    # Candlestick or Line chart for Close price
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='black', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Moving Averages
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA_5'],
            mode='lines',
            name='MA 5',
            line=dict(color='blue', width=1.5),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>MA5: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1.5),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>MA20: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MA_200'],
            mode='lines',
            name='MA 200',
            line=dict(color='red', width=1.5),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>MA200: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands - Upper Band
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='lightgray', width=1, dash='dash'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>BB Upper: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands - Lower Band
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='lightgray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.2)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>BB Lower: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # ===== Row 2: RSI =====
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>RSI: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # RSI reference lines (30 and 70)
    fig.add_hline(y=30, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Oversold")
    fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Overbought")
    
    # ===== Row 3: MACD =====
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>MACD: %{y:.4f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            mode='lines',
            name='MACD Signal',
            line=dict(color='red', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Signal: %{y:.4f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # MACD Histogram
    colors = ['green' if val > 0 else 'red' for val in df['MACD_Histogram']]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['MACD_Histogram'],
            name='MACD Histogram',
            marker_color=colors,
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Histogram: %{y:.4f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # ===== Row 4: Volume =====
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='steelblue',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,}<extra></extra>'
        ),
        row=4, col=1
    )
    
    # Add Volume Moving Average
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Volume_MA_20'],
            mode='lines',
            name='Volume MA 20',
            line=dict(color='red', width=2),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Vol MA: %{y:,}<extra></extra>'
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"<b>{ticker} - Stock Price Analysis</b>",
        height=1200,
        width=1400,
        hovermode='x unified',
        template='plotly_white',
        font=dict(size=10),
        showlegend=True
    )
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig


def save_chart(fig, ticker: str, output_dir: str = "charts"):
    """Save chart as HTML file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{ticker}_chart.html")
    fig.write_html(file_path)
    print(f"✓ Saved: {file_path}")
    return file_path


def main():
    tickers = ["NVDA", "SONY"]
    
    for ticker in tickers:
        print(f"\n=== Creating chart for {ticker} ===")
        
        # Load processed data
        df = load_processed_csv(ticker)
        if df is None:
            continue
        
        # Remove NaN rows (from indicator calculations)
        df = df.dropna()
        
        # Create chart
        fig = create_chart(df, ticker)
        
        # Save and display
        file_path = save_chart(fig, ticker)
        print(f"📊 Open this file in your browser: {file_path}")


if __name__ == "__main__":
    main()
