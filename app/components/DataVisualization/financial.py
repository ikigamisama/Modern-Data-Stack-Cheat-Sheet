import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots


class Financial:

    def __init__(self):
        st.set_page_config(
            page_title="Financial Visualizations", layout="wide")
        self.title = "💹 Financial Visualizations Dashboard"
        self.chart_types = [
            "Candlestick Chart",
            "OHLC Chart",
            "Bollinger Bands",
            "MACD"
        ]
        self.assets = [
            "Apple (AAPL)",
            "Bitcoin (BTC)",
            "Gold Futures",
            "EUR/USD Forex",
            "Tesla (TSLA)"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Financial charts are specialized for **trading, market analysis, and price movements**. 
        They incorporate open, high, low, and close values with technical indicators.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **When to Use:**
            - Stock market analysis
            - Cryptocurrency trading
            - Commodity price tracking
            - Technical analysis
            - Day trading decisions
            """)

        with col2:
            st.markdown("""
            **Key Characteristics:**
            - Shows OHLC (Open, High, Low, Close) data
            - Includes volume information
            - Incorporates technical indicators
            - Enables pattern recognition
            """)

        st.markdown("""
        **Supported Charts:**
        - **Candlestick Chart** – Classic price action with body and wicks
        - **OHLC Chart** – Open-High-Low-Close bars
        - **Bollinger Bands** – Volatility bands around moving average
        - **MACD** – Momentum indicator with signal line and histogram
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Chart Settings")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            asset = st.selectbox("Asset", self.assets, key="asset")

        with col3:
            period = st.selectbox(
                "Time Period", ["3 Months", "6 Months", "1 Year"], index=1)

        return chart_type, asset, period

    def generate_ohlc_data(self, asset: str, period: str):
        np.random.seed(42)
        periods = {"3 Months": 60, "6 Months": 120, "1 Year": 252}
        n = periods[period]

        dates = pd.date_range("2025-01-01", periods=n,
                              freq='B')  # Business days

        # Base price trend
        if "Bitcoin" in asset:
            base_price = 60000
            volatility = 0.04
        elif "Apple" in asset or "Tesla" in asset:
            base_price = 180 if "Apple" in asset else 250
            volatility = 0.02
        elif "Gold" in asset:
            base_price = 2300
            volatility = 0.01
        else:  # Forex
            base_price = 1.08
            volatility = 0.005

        prices = [base_price]
        for _ in range(n-1):
            change = np.random.normal(0, volatility)
            prices.append(prices[-1] * (1 + change))

        prices = np.array(prices)

        # Generate OHLC
        df = pd.DataFrame(index=dates)
        df["Open"] = prices * np.random.uniform(0.99, 1.01, n)
        df["Close"] = prices * np.random.uniform(0.99, 1.01, n)
        df["High"] = np.maximum(df["Open"], df["Close"]) * \
            np.random.uniform(1.00, 1.03, n)
        df["Low"] = np.minimum(df["Open"], df["Close"]) * \
            np.random.uniform(0.97, 1.00, n)
        df["Volume"] = np.random.randint(500000, 5000000, n)

        return df.round(2)

    def create_candlestick(self, df: pd.DataFrame, asset: str) -> go.Figure:
        fig = go.Figure(data=go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=asset
        ))

        fig.update_layout(
            title=f"{asset} – Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                

fig = go.Figure(data=go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name=asset
))

fig.update_layout(
    title=f"{asset} – Candlestick Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    height=600,
    xaxis_rangeslider_visible=True
)
fig.show()                
""", language="python")

    def create_ohlc(self, df: pd.DataFrame, asset: str) -> go.Figure:
        fig = go.Figure(data=go.Ohlc(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=asset
        ))

        fig.update_layout(
            title=f"{asset} – OHLC Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                

fig = go.Figure(data=go.Ohlc(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name=asset
))

fig.update_layout(
    title=f"{asset} – OHLC Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    height=600,
    xaxis_rangeslider_visible=True
)
fig.show()                
""", language="python")

    def create_bollinger(self, df: pd.DataFrame, asset: str) -> go.Figure:
        df["MA20"] = df["Close"].rolling(20).mean()
        df["STD20"] = df["Close"].rolling(20).std()
        df["Upper"] = df["MA20"] + 2 * df["STD20"]
        df["Lower"] = df["MA20"] - 2 * df["STD20"]

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Price"
        ))

        fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Upper Band", line=dict(
            color="#7f8c8d", dash="dash")))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA20"], name="20-Day MA", line=dict(color="#3498db")))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Lower Band", line=dict(color="#7f8c8d", dash="dash"),
                                 fill=None))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Bollinger Band",
                                 fill='tonexty', fillcolor='rgba(173,216,230,0.2)'))

        fig.update_layout(
            title=f"{asset} – Bollinger Bands (20, 2)",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
df["MA20"] = df["Close"].rolling(20).mean()
df["STD20"] = df["Close"].rolling(20).std()
df["Upper"] = df["MA20"] + 2 * df["STD20"]
df["Lower"] = df["MA20"] - 2 * df["STD20"]

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="Price"
))

fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Upper Band", line=dict(
    color="#7f8c8d", dash="dash")))
fig.add_trace(go.Scatter(
    x=df.index, y=df["MA20"], name="20-Day MA", line=dict(color="#3498db")))
fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Lower Band", line=dict(color="#7f8c8d", dash="dash"),
                            fill=None))
fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Bollinger Band",
                            fill='tonexty', fillcolor='rgba(173,216,230,0.2)'))

fig.update_layout(
    title=f"{asset} – Bollinger Bands (20, 2)",
    xaxis_title="Date",
    yaxis_title="Price",
    height=600,
    xaxis_rangeslider_visible=False
)
fig.show()                
""", language="python")

    def create_macd(self, df: pd.DataFrame, asset: str) -> go.Figure:
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9).mean()
        df["Histogram"] = df["MACD"] - df["Signal"]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=(
                                f"{asset} – Price", "MACD Indicator"),
                            row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Price"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(
            color="#3498db")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal Line", line=dict(
            color="#e67e22")), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["Histogram"], name="Histogram",
                             marker_color=np.where(df["Histogram"] > 0, '#27ae60', '#e74c3c')), row=2, col=1)

        fig.update_layout(
            title=f"{asset} – MACD (12, 26, 9)",
            height=700,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
ema12 = df["Close"].ewm(span=12).mean()
ema26 = df["Close"].ewm(span=26).mean()
df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9).mean()
df["Histogram"] = df["MACD"] - df["Signal"]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(
                        f"{asset} – Price", "MACD Indicator"),
                    row_heights=[0.7, 0.3])

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name="Price"
), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(
    color="#3498db")), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal Line", line=dict(
    color="#e67e22")), row=2, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["Histogram"], name="Histogram",
                        marker_color=np.where(df["Histogram"] > 0, '#27ae60', '#e74c3c')), row=2, col=1)

fig.update_layout(
    title=f"{asset} – MACD (12, 26, 9)",
    height=700,
    xaxis_rangeslider_visible=False,
    showlegend=True
)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="MACD", row=2, col=1)
fig.show()                
""", language="python")

    def render_chart(self, chart_type: str, asset: str, period: str):
        st.markdown(f"### {chart_type}: {asset} ({period})")

        df = self.generate_ohlc_data(asset, period)

        if chart_type == "Candlestick Chart":
            self.create_candlestick(df, asset)
        elif chart_type == "OHLC Chart":
            self.create_ohlc(df, asset)
        elif chart_type == "Bollinger Bands":
            self.create_bollinger(df, asset)
        else:  # MACD
            self.create_macd(df, asset)

        if chart_type == "Bollinger Bands":
            st.info(
                "**Insight**: Price near upper band = overbought | near lower = oversold | squeeze = low volatility.")
        elif chart_type == "MACD":
            st.info(
                "**Insight**: MACD above signal = bullish momentum | histogram growth = strengthening trend.")

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Budget Analysis": "Actual spending vs. budgeted amounts",
            "Performance Revie": "Results compared to targets",
            "Survey Results": "Net promoter scores (positive minus negative)",
            "Financial Reporting": "FRevenue variance from forecast",
            "Temperature Analysis": "Daily temperature vs. historical average"
        }

        for example, description in examples.items():
            with st.expander(f"💭 {example}"):
                st.write(description)

    def render_key_characteristics(self):
        st.markdown("### 📈 Understanding Deviation Analysis")

        st.markdown("""
        Deviation analysis shows **how values differ from a benchmark or reference point**.
        It emphasizes relative performance rather than absolute numbers alone.
        """)

        st.markdown("#### ➕➖ Clear Positive/Negative Distinction")
        st.markdown("""
        Deviations are visually distinguished:
        - **Positive** (above benchmark)  
        - **Negative** (below benchmark)  

        This makes interpretation immediate.
        """)

        st.markdown("#### 📏 Emphasizes Magnitude of Deviation")
        st.markdown("""
        The size of the deviation communicates importance:
        - Larger deviations demand attention  
        - Smaller deviations may be less critical  

        Visual encodings reinforce this.
        """)

        st.markdown("#### 🎨 Often Uses Contrasting Colors")
        st.markdown("""
        Color is a pre-attentive cue:
        - Green/Red  
        - Blue/Orange  
        - Diverging palettes  

        It enables users to instantly identify performance direction.
        """)

        st.markdown("#### ⚡ Facilitates Quick Assessment")
        st.markdown("""
        Deviation views summarize performance against expectations:
        - Spot winners and laggards  
        - Prioritize attention efficiently  
        - Identify risk or opportunity quickly
        """)

        st.divider()

        st.markdown("#### 🎯 Why Deviation Analysis Matters")
        st.markdown("""
        Deviation analysis turns raw metrics into actionable insight.
        It supports:
        - KPI monitoring and dashboards  
        - Target tracking  
        - Performance evaluation  
        - Risk and opportunity identification  
        """)

    def output(self):
        self.render_header()
        chart_type, asset, period = self.render_configuration()
        self.render_chart(chart_type, asset, period)
        self.render_examples()
        self.render_key_characteristics()
