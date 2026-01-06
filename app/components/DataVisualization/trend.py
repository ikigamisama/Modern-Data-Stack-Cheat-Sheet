import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


class Trend:
    def __init__(self):

        self.chart_type = None
        self.dataset_type = None
        self.time_period = None

        self.periods_map = {
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730
        }

        self.df = None
        self.chart_functions = {
            "Line Chart": self.render_line_chart,
            "Area Chart": self.render_area_chart,
            "Sparkline": self.render_sparkline,
            "Cycle Plot": self.render_cycle_plot,
            "Timeline": self.render_timeline,
            "Run Chart": self.render_run_chart,
            "Streamgraph": self.render_streamgraph
        }

    def render_configuration(self):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                [
                    "Line Chart", "Area Chart", "Sparkline",
                    "Cycle Plot", "Timeline", "Run Chart", "Streamgraph"
                ]
            )

        with col2:
            dataset_type = st.selectbox(
                "Dataset Type",
                ["Sales", "Website Traffic", "Temperature",
                    "Stock Prices", "Social Media"]
            )

        with col3:
            time_period = st.selectbox(
                "Time Period",
                ["3 Months", "6 Months", "1 Year", "2 Years"]
            )

        return chart_type, dataset_type, time_period

    def generate_trend_data(self, data_type, periods=365):
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="D")

        if data_type == "Sales":
            base = np.linspace(1000, 5000, periods)
            seasonal = 500 * np.sin(2 * np.pi * np.arange(periods) / 30)
            noise = np.random.normal(0, 200, periods)
            return pd.DataFrame({"Date": dates, "Revenue": base + seasonal + noise})

        elif data_type == "Website Traffic":
            base = np.linspace(500, 3000, periods)
            weekly = 300 * np.sin(2 * np.pi * np.arange(periods) / 7)
            monthly = 150 * np.sin(2 * np.pi * np.arange(periods) / 30)
            noise = np.random.normal(0, 100, periods)
            return pd.DataFrame({"Date": dates, "Visitors": base + weekly + monthly + noise})

        elif data_type == "Temperature":
            base = np.full(periods, 20)
            seasonal = 15 * np.sin(2 * np.pi * np.arange(periods) / 365)
            noise = np.random.normal(0, 3, periods)
            return pd.DataFrame({"Date": dates, "Temperature": base + seasonal + noise})

        elif data_type == "Stock Prices":
            returns = np.random.normal(0.001, 0.02, periods)
            prices = 100 * np.cumprod(1 + returns)
            return pd.DataFrame({"Date": dates, "Price": prices})

        else:  # Social Media
            base = np.linspace(1000, 10000, periods)
            spikes = np.random.poisson(0.05, periods) * 500
            noise = np.random.normal(0, 200, periods)
            return pd.DataFrame({"Date": dates, "Followers": base + spikes + noise})

    def render_chart(self):
        if self.chart_type in self.chart_functions:
            self.chart_functions[self.chart_type](self.df, self.dataset_type)

    def render_line_chart(self, df, data_type):
        self._render_basic_chart(
            df,
            data_type,
            "📊 Line Chart - Basic Trend Visualization",
            px.line,
            "Line Chart",
            "**When to use:** Showing continuous data over time, identifying trends and patterns",
            """
import plotly.express as px

fig = px.line(df, x='Date', y='Revenue', title='Sales Trend Over Time')
fig.show()
            """
        )

    def render_area_chart(self, df, data_type):
        self._render_basic_chart(
            df,
            data_type,
            "📈 Area Chart - Cumulative Trend Visualization",
            px.area,
            "Area Chart",
            "**When to use:** Showing cumulative totals over time, emphasizing volume",
            """
import plotly.express as px

fig = px.area(df, x='Date', y='Revenue', title='Cumulative Revenue')
fig.show()
            """
        )

    def render_sparkline(self, df, data_type):
        st.subheader("⚡ Sparkline - Compact Trend Indicator")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df.iloc[:, 1], mode="lines"))
        fig.update_layout(
            height=100,
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_showticklabels=False,
            yaxis_showticklabels=False
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            "**When to use:** Inline trend indicators, dashboards with limited space")

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code("""
import plotly.graph_objects as go
                           
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df.iloc[:, 1], mode="lines"))
fig.update_layout(
    height=100,
    showlegend=False,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_showticklabels=False,
    yaxis_showticklabels=False
)
fig.show()
        """, language="python")

    def render_cycle_plot(self, df, data_type):
        st.subheader("🔄 Cycle Plot - Seasonal Pattern Analysis")

        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year

        fig = px.line(df, x="Month", y=df.columns[1], color="Year")
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            "**When to use:** Comparing seasonal patterns across multiple years")

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""
import plotly.express as px
                           
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
fig = px.line(df, x='Month', y='Revenue', color='Year')
fig.show()
        """, language="python")

    def render_timeline(self, df, data_type):
        st.subheader("⏰ Timeline - Event-based Trend Visualization")

        df["Date"] = pd.to_datetime(df["Date"])
        y_col = df.columns[1]
        y_max = df[y_col].max()

        events = {
            "Product Launch": df["Date"].iloc[len(df) // 4],
            "Marketing Campaign": df["Date"].iloc[len(df) // 2],
            "Holiday Season": df["Date"].iloc[3 * len(df) // 4],
        }

        fig = px.line(df, x="Date", y=y_col)

        for name, date in events.items():
            # Vertical line (NO math involved)
            fig.add_shape(
                type="line",
                x0=date, x1=date,
                y0=0, y1=y_max,
                line=dict(dash="dash")
            )

            # Annotation (explicit placement)
            fig.add_annotation(
                x=date,
                y=y_max,
                text=name,
                showarrow=True,
                arrowhead=2,
                yshift=10
            )

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            "**When to use:** Showing trends with key events and milestones"
        )

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""
import plotly.express as px                

df["Date"] = pd.to_datetime(df["Date"])
y_col = df.columns[1]
y_max = df[y_col].max()

events = {
    "Product Launch": df["Date"].iloc[len(df) // 4],
    "Marketing Campaign": df["Date"].iloc[len(df) // 2],
    "Holiday Season": df["Date"].iloc[3 * len(df) // 4],
}

fig = px.line(df, x="Date", y=y_col)

for name, date in events.items():
    # Vertical line (NO math involved)
    fig.add_shape(
        type="line",
        x0=date, x1=date,
        y0=0, y1=y_max,
        line=dict(dash="dash")
    )

    # Annotation (explicit placement)
    fig.add_annotation(
        x=date,
        y=y_max,
        text=name,
        showarrow=True,
        arrowhead=2,
        yshift=10
    )
fig.show()
""", language="python")

    def render_run_chart(self, df, data_type):
        st.subheader("📏 Run Chart - Process Monitoring")

        median_val = df.iloc[:, 1].median()
        fig = px.line(df, x="Date", y=df.columns[1])
        fig.add_hline(y=median_val, line_dash="dash",
                      annotation_text=f"Median: {median_val:.0f}")
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            "**When to use:** Statistical process control, monitoring process stability")

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.express as px
                
median_val = df.iloc[:, 1].median()
fig = px.line(df, x="Date", y=df.columns[1])
fig.add_hline(y=median_val, line_dash="dash", annotation_text=f"Median: {median_val:.0f}")
""", language="python")

    def render_streamgraph(self, df, data_type):
        st.subheader("🌊 Streamgraph - Stacked Area Visualization")

        categories = ["Category A", "Category B", "Category C", "Category D"]
        records = []

        for i, cat in enumerate(categories):
            base = (i + 1) * 1000 + np.linspace(0, 2000, 365)
            seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 30)
            noise = np.random.normal(0, 100, 365)
            for d, v in zip(pd.date_range(end=datetime.now(), periods=365), base + seasonal + noise):
                records.append({"Date": d, "Category": cat, "Value": v})

        stream_df = pd.DataFrame(records)
        fig = px.area(stream_df, x="Date", y="Value", color="Category")
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            "**When to use:** Showing composition changes over time across multiple categories")

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.express as px
import numpy as np
import pandas as pd
                
categories = ["Category A", "Category B", "Category C", "Category D"]
records = []

for i, cat in enumerate(categories):
    base = (i + 1) * 1000 + np.linspace(0, 2000, 365)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 30)
    noise = np.random.normal(0, 100, 365)
    for d, v in zip(pd.date_range(end=datetime.now(), periods=365), base + seasonal + noise):
        records.append({"Date": d, "Category": cat, "Value": v})

stream_df = pd.DataFrame(records)
fig = px.area(stream_df, x="Date", y="Value", color="Category")
fig.show()""", language="python")

    def _render_basic_chart(self, df, data_type, subtitle, plot_func, chart_name, explanation, code):
        st.subheader(subtitle)
        fig = plot_func(
            df, x="Date", y=df.columns[1], title=f"{data_type} - {chart_name}")
        st.plotly_chart(fig, width="stretch")
        st.markdown(explanation)
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code(code, language="python")

    def render_key_characteristics(self):
        st.markdown("### 📈 Understanding Temporal Patterns in Data")

        st.markdown("""
        Data does not exist in isolation—it evolves over time. One of the most powerful ways to
        understand a dataset is by examining its **temporal behavior**, which reveals how values
        change, accelerate, slow down, or abruptly shift.
        """)

        st.markdown("#### ⏳ Emphasis on Temporal Progression")
        st.markdown("""
        Temporal progression focuses on how data unfolds across time—seconds, days, months,
        or years. Rather than asking *“What is the value?”*, it asks *“How did we get here?”*.

        This perspective enables:
        - Trend detection  
        - Seasonality analysis  
        - Long-term behavioral insights  
        """)

        st.markdown("#### 📐 Highlighting the Rate of Change")
        st.markdown("""
        The rate of change measures **how fast** values are increasing or decreasing.
        Two trends may look similar, yet differ significantly in velocity.

        Examples include:
        - Revenue growth speed  
        - System load increases  
        - User adoption velocity  
        """)

        st.markdown("#### 🚀 Revealing Acceleration or Deceleration")
        st.markdown("""
        Acceleration examines how the **rate of change itself evolves**.
        It uncovers momentum and second-order effects.

        - Acceleration → compounding growth or escalation  
        - Deceleration → saturation, resistance, or constraints  

        This insight is critical for forecasting and capacity planning.
        """)

        st.markdown("#### ⚠️ Continuity vs. Disruption in Patterns")
        st.markdown("""
        Temporal data often follows predictable rhythms such as daily, weekly,
        or seasonal cycles.

        - **Continuity** signals stability and reliability  
        - **Disruption** indicates anomalies, shocks, or behavioral shifts  

        Identifying these breaks helps organizations respond quickly.
        """)

        st.divider()

        st.markdown("#### 🎯 Why Temporal Analysis Matters")
        st.markdown("""
        Time-aware analysis transforms raw metrics into **actionable intelligence**.
        It supports:
        - Better decision-making  
        - Early anomaly detection  
        - More accurate forecasting  """)

    def render_examples(self, chart_type: str, dataset_type: str):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Sales Performance": "Track monthly revenue growth across quarters",
            "Website Analytics": "Monitor daily visitor counts and page views",
            "Temperature Data": "Display climate changes over decades",
            "Stock Market": "Show price movements throughout trading days",
            "Social Media": "Track follower growth and engagement rates over time"
        }

        for example, description in examples.items():
            with st.expander(f"📊 {example}"):
                st.write(description)
                best = chart_type if dataset_type == example else "All density charts suitable"
                st.write(f"**Best chart type:** {best}")

    def output(self):
        st.markdown("## 📈 Trend Visualization Dashboard")
        st.markdown("""
        ### Purpose
        Trend visualizations reveal patterns, changes, and movements in data over time. They help identify upward or downward trajectories, seasonal patterns, and cyclical behaviors.
        """)

        self.chart_type, self.dataset_type, self.time_period = self.render_configuration()
        self.df = self.generate_trend_data(
            self.dataset_type,
            self.periods_map[self.time_period]
        )

        self.render_chart()
        self.render_examples(self.chart_type, self.dataset_type)
        self.render_key_characteristics()
