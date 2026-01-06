import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd


class Anomaly:
    def __init__(self):
        st.set_page_config(page_title="Anomaly Detection", layout="wide")
        self.title = "🚨 Anomaly Detection Visualizations Dashboard"
        self.chart_types = [
            "Control Chart",
            "Time Series Anomaly"
        ]
        self.scenarios = [
            "Manufacturing Process",
            "Server Response Time",
            "Daily Sales Transactions",
            "Patient Heart Rate",
            "Network Traffic Volume"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Anomaly detection charts highlight **unusual patterns, outliers, and deviations** from expected behavior. 
        They're essential for quality control, fraud detection, and system monitoring.
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            scenario = st.selectbox("Scenario", self.scenarios, key="scenario")

        with col3:
            num_points = st.slider(
                "Number of Data Points", 30, 100, 50, key="points")
            anomaly_strength = st.slider(
                "Anomaly Strength", 1.0, 4.0, 2.5, step=0.5, key="strength")

        return chart_type, scenario, num_points, anomaly_strength

    def generate_control_chart_data(self, scenario: str, n_points: int, strength: float):
        np.random.seed(42)
        dates = pd.date_range("2025-01-01", periods=n_points, freq='D')

        # Base stable process
        if "Manufacturing" in scenario:
            mean = 50.0
            std = 3.0
            process_name = "Product Dimension (mm)"
        elif "Response Time" in scenario:
            mean = 120.0
            std = 15.0
            process_name = "Server Response Time (ms)"
        elif "Sales" in scenario:
            mean = 200.0
            std = 20.0
            process_name = "Daily Transactions"
        elif "Heart Rate" in scenario:
            mean = 72.0
            std = 5.0
            process_name = "Heart Rate (bpm)"
        else:
            mean = 100.0
            std = 10.0
            process_name = "Metric Value"

        values = np.random.normal(mean, std, n_points)

        # Inject anomalies
        anomaly_indices = np.random.choice(
            n_points, size=max(3, n_points//15), replace=False)
        for idx in anomaly_indices:
            if np.random.rand() > 0.5:
                values[idx] += strength * std  # High outlier
            else:
                values[idx] -= strength * std  # Low outlier

        df = pd.DataFrame({"Date": dates, "Value": values})
        df["Mean"] = mean
        df["UCL"] = mean + 3 * std
        df["LCL"] = mean - 3 * std
        df["Anomaly"] = ~df["Value"].between(df["LCL"], df["UCL"])

        return df, process_name

    def create_control_chart(self, df: pd.DataFrame, process_name: str) -> go.Figure:
        fig = go.Figure()

        # Main data points
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Value"],
            mode='lines+markers',
            name='Measurements',
            line=dict(color='#3498db'),
            marker=dict(
                size=8,
                color=np.where(df["Anomaly"], '#e74c3c', '#3498db'),
                line=dict(width=1, color='white')
            )
        ))

        # Control limits
        fig.add_trace(go.Scatter(x=df["Date"], y=df["UCL"], mode='lines',
                      name='UCL (+3σ)', line=dict(dash='dash', color='#e74c3c')))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["LCL"], mode='lines',
                      name='LCL (-3σ)', line=dict(dash='dash', color='#e74c3c')))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Mean"], mode='lines',
                      name='Center Line', line=dict(color='#27ae60', width=2)))

        # Highlight anomalies
        anomalies = df[df["Anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["Date"], y=anomalies["Value"],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#e74c3c', size=12,
                            symbol='x', line=dict(width=3))
            ))

        fig.update_layout(
            title=f"Control Chart – {process_name}<br><sub>Red points and dashed lines indicate out-of-control conditions</sub>",
            xaxis_title="Date",
            yaxis_title=process_name,
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code("""import pandas as pd
import numpy as np
import plotly.graph_objects as go
                           
fig = go.Figure()

# Main data points
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Value"],
    mode='lines+markers',
    name='Measurements',
    line=dict(color='#3498db'),
    marker=dict(
        size=8,
        color=np.where(df["Anomaly"], '#e74c3c', '#3498db'),
        line=dict(width=1, color='white')
    )
))

# Control limits
fig.add_trace(go.Scatter(x=df["Date"], y=df["UCL"], mode='lines',
                name='UCL (+3σ)', line=dict(dash='dash', color='#e74c3c')))
fig.add_trace(go.Scatter(x=df["Date"], y=df["LCL"], mode='lines',
                name='LCL (-3σ)', line=dict(dash='dash', color='#e74c3c')))
fig.add_trace(go.Scatter(x=df["Date"], y=df["Mean"], mode='lines',
                name='Center Line', line=dict(color='#27ae60', width=2)))

# Highlight anomalies
anomalies = df[df["Anomaly"]]
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies["Date"], y=anomalies["Value"],
        mode='markers',
        name='Anomalies',
        marker=dict(color='#e74c3c', size=12,
                    symbol='x', line=dict(width=3))
    ))

fig.update_layout(
    title=f"Control Chart – {process_name}<br><sub>Red points and dashed lines indicate out-of-control conditions</sub>",
    xaxis_title="Date",
    yaxis_title=process_name,
    height=600,
    showlegend=True
)
fig.show()""", language="python")

    def generate_time_series_anomalies(self, scenario: str, n_points: int, strength: float):
        np.random.seed(42)
        dates = pd.date_range("2025-01-01", periods=n_points, freq='H')

        # Trend + seasonality
        trend = np.linspace(100, 120, n_points)
        seasonality = 20 * np.sin(np.linspace(0, 4*np.pi, n_points))
        noise = np.random.normal(0, 5, n_points)
        values = trend + seasonality + noise

        # Inject anomalies
        anomaly_indices = np.random.choice(
            n_points, size=max(4, n_points//12), replace=False)
        for idx in anomaly_indices:
            # Strong deviation
            values[idx] += np.random.choice([-1, 1]) * strength * 20

        df = pd.DataFrame({"Timestamp": dates, "Value": values})

        # Simple anomaly detection: Z-score > 3
        rolling_mean = df["Value"].rolling(window=20, min_periods=1).mean()
        rolling_std = df["Value"].rolling(window=20, min_periods=1).std()
        df["Z"] = (df["Value"] - rolling_mean) / (rolling_std + 1e-6)
        df["Anomaly"] = df["Z"].abs() > 3

        metric_name = "Network Traffic" if "Network" in scenario else "Metric Value"
        return df, metric_name

    def create_time_series_anomaly(self, df: pd.DataFrame, metric_name: str) -> go.Figure:
        fig = go.Figure()

        # Main line
        fig.add_trace(go.Scatter(
            x=df["Timestamp"], y=df["Value"],
            mode='lines',
            name='Time Series',
            line=dict(color='#3498db')
        ))

        # Anomalies
        anomalies = df[df["Anomaly"]]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["Timestamp"], y=anomalies["Value"],
                mode='markers',
                name='Detected Anomalies',
                marker=dict(color='#e74c3c', size=10,
                            symbol='circle-open', line=dict(width=3))
            ))

        fig.update_layout(
            title=f"Time Series Anomaly Detection – {metric_name}<br><sub>Red circles highlight statistically significant outliers</sub>",
            xaxis_title="Timestamp",
            yaxis_title=metric_name,
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code("""import pandas as pd
import numpy as np
import plotly.graph_objects as go
                           
fig = go.Figure()

# Main line
fig.add_trace(go.Scatter(
    x=df["Timestamp"], y=df["Value"],
    mode='lines',
    name='Time Series',
    line=dict(color='#3498db')
))

# Anomalies
anomalies = df[df["Anomaly"]]
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies["Timestamp"], y=anomalies["Value"],
        mode='markers',
        name='Detected Anomalies',
        marker=dict(color='#e74c3c', size=10,
                    symbol='circle-open', line=dict(width=3))
    ))

fig.update_layout(
    title=f"Time Series Anomaly Detection – {metric_name}<br><sub>Red circles highlight statistically significant outliers</sub>",
    xaxis_title="Timestamp",
    yaxis_title=metric_name,
    height=600,
    showlegend=True
)
fig.show()""", language="python")

    def render_chart(self, chart_type: str, scenario: str, n_points: int, strength: float):
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Control Chart":
            df, name = self.generate_control_chart_data(
                scenario, n_points, strength)
            self.create_control_chart(df, name)
        else:
            df, name = self.generate_time_series_anomalies(
                scenario, n_points, strength)
            self.create_time_series_anomaly(df, name)

        # Summary stats
        if chart_type == "Control Chart":
            anomalies = df[df["Anomaly"]]
            st.info(
                f"**Detected {len(anomalies)} out-of-control points** out of {len(df)} measurements")
        else:
            anomalies = df[df["Anomaly"]]
            st.info(
                f"**Detected {len(anomalies)} anomalies** using rolling Z-score method")

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Manufacturing": "Detecting defective products in production",
            "Cybersecurity": "Identifying unusual network traffic patterns",
            "Finance": "Flagging suspicious transactions",
            "Healthcare": "Monitoring vital signs for abnormal readings",
            "Process Control": "Detecting when processes go out of specification"
        }

        for example, description in examples.items():
            with st.expander(f"💭 {example}"):
                st.write(description)

    def render_key_characteristics(self):
        st.markdown("### 🚨 Understanding Anomaly Analysis")

        st.markdown("""
        Anomaly analysis identifies **unexpected deviations** from normal behavior.
        Its purpose is not explanation, but **early detection and response**.
        """)

        st.markdown("#### 📉 Highlights Deviations from Normal")
        st.markdown("""
        Anomalies are detected relative to a baseline such as:
        - Historical patterns  
        - Seasonal trends  
        - Learned system behavior  

        Anything that significantly deviates is flagged.
        """)

        st.markdown("#### 📊 Uses Statistical Thresholds")
        st.markdown("""
        Detection commonly relies on:
        - Standard deviation bands  
        - Z-scores  
        - Percentile thresholds  
        - Model-based confidence intervals  

        This ensures objectivity and consistency.
        """)

        st.markdown("#### ⚡ Enables Quick Identification of Problems")
        st.markdown("""
        Anomaly analysis filters noise and directs attention to
        high-impact irregularities, speeding up investigation and resolution.
        """)

        st.markdown("#### 🛡️ Supports Proactive Intervention")
        st.markdown("""
        By detecting issues early, teams can act before failures occur,
        reducing downtime, losses, and user impact.
        """)

        st.divider()

        st.markdown("#### 🎯 Why Anomaly Analysis Matters")
        st.markdown("""
        Anomaly analysis transforms monitoring into an early warning system.
        It is essential for:
        - System reliability  
        - Fraud detection  
        - Data quality assurance  
        - Operational resilience  
        """)

    def output(self):
        self.render_header()
        chart_type, scenario, num_points, anomaly_strength = self.render_configuration()
        self.render_chart(chart_type, scenario, num_points, anomaly_strength)
        self.render_examples()
        self.render_key_characteristics()
