import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class TimeSeries:
    def __init__(self):
        self.title = "⏰ Time Series Visualization Dashboard"
        self.chart_types = ["Time Series Plot",
                            "Gantt Chart", "Horizon Chart", "Timetable"]
        self.dataset_types = [
            "Stock Market", "Project Management", "Weather Forecasting",
            "Business Metrics", "IoT Monitoring", "Web Analytics"
        ]
        self.time_ranges = ["1 Month", "3 Months",
                            "6 Months", "1 Year", "2 Years"]
        self.frequencies = ["Daily", "Weekly", "Monthly", "Quarterly"]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Time series visualizations are specifically designed for temporal data, showing how values 
        change across time intervals. They're optimized for revealing trends, seasonality, and temporal patterns.
        """)

    def render_configuration(self):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            dataset_type = st.selectbox(
                "Dataset Type", self.dataset_types, key="dataset_type")

        with col3:
            time_range = st.selectbox(
                "Time Range", self.time_ranges, key="time_range")

        with col4:
            frequency = st.selectbox(
                "Frequency", self.frequencies, key="frequency")
            show_forecast = st.checkbox(
                "Show Forecast/Trend", value=False, key="show_forecast")

        return chart_type, dataset_type, time_range, frequency, show_forecast

    def generate_timeseries_data(self, data_type: str, time_range: str, frequency: str) -> pd.DataFrame:
        periods_map = {
            "1 Month": 30, "3 Months": 90, "6 Months": 180,
            "1 Year": 365, "2 Years": 730
        }
        freq_map = {"Daily": "D", "Weekly": "W",
                    "Monthly": "M", "Quarterly": "Q"}

        periods = periods_map[time_range]
        freq_str = freq_map[frequency]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=periods)
        dates = pd.date_range(start=start_date, end=end_date, freq=freq_str)

        np.random.seed(42)

        if data_type == "Stock Market":
            trend = np.linspace(100, 180, len(dates))
            noise = np.cumsum(np.random.normal(0, 3, len(dates)))
            prices = trend + noise
            return pd.DataFrame({
                'Date': dates,
                'Price': prices,
                'Volume': np.random.randint(1_000_000, 5_000_000, len(dates)),
                'Daily_Change': np.random.normal(0, 2, len(dates))
            })

        elif data_type == "Project Management":
            tasks = ['Planning', 'Design', 'Development',
                     'Testing', 'Deployment', 'Review']
            task_data = []
            current = start_date
            for task in tasks:
                duration = np.random.randint(10, 45)
                end = current + timedelta(days=duration)
                task_data.append({
                    'Task': task,
                    'Start': current.date(),
                    'End': end.date(),
                    'Duration_Days': duration,
                    'Progress': np.random.uniform(20, 100),
                    'Team': np.random.choice(['Alpha', 'Beta', 'Gamma'])
                })
                current = end + timedelta(days=np.random.randint(2, 7))
            return pd.DataFrame(task_data)

        elif data_type == "Weather Forecasting":
            seasonal = 12 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            trend = np.linspace(-2, 2, len(dates))
            temp = 20 + seasonal + trend + np.random.normal(0, 4, len(dates))
            return pd.DataFrame({
                'Date': dates,
                'Temperature': temp,
                'Humidity': np.random.uniform(40, 90, len(dates)),
                'Precipitation_mm': np.random.exponential(3, len(dates))
            })

        elif data_type == "Business Metrics":
            growth = np.linspace(10_000, 60_000, len(dates))
            seasonal = 5_000 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
            revenue = growth + seasonal + \
                np.random.normal(0, 2_000, len(dates))
            return pd.DataFrame({
                'Date': dates,
                'Revenue': revenue,
                'New_Customers': np.random.randint(50, 300, len(dates)),
                'Conversion_Rate': np.random.uniform(2.5, 6.0, len(dates))
            })

        elif data_type == "IoT Monitoring":
            daily_cycle = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
            spikes = np.random.poisson(
                0.05, len(dates)) * np.random.uniform(15, 40, len(dates))
            reading = 50 + daily_cycle + spikes + \
                np.random.normal(0, 3, len(dates))
            return pd.DataFrame({
                'Timestamp': dates,
                'Sensor_Reading': reading,
                'Battery_Percent': np.linspace(100, 30, len(dates)) + np.random.normal(0, 5, len(dates)),
                'Status': np.random.choice(['Normal', 'Warning', 'Alert'], len(dates), p=[0.9, 0.08, 0.02])
            })

        else:  # Web Analytics
            trend = np.linspace(2_000, 8_000, len(dates))
            weekly = 800 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
            visitors = trend + weekly + np.random.normal(0, 400, len(dates))
            return pd.DataFrame({
                'Date': dates,
                'Visitors': visitors,
                'Page_Views': visitors * np.random.uniform(3, 6, len(dates)),
                'Bounce_Rate': np.random.uniform(35, 55, len(dates))
            })

    def render_timeseries_plot(self, df: pd.DataFrame, data_type: str, show_forecast: bool):
        st.markdown("### 📈 Time Series Plot - Trends & Patterns")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = st.selectbox("Value to Plot", numeric_cols, index=0)
        date_col = 'Date' if 'Date' in df.columns else 'Timestamp'

        fig = px.line(df, x=date_col, y=value_col,
                      title=f"{data_type} - {value_col} Over Time")

        if show_forecast:
            x_num = np.arange(len(df))
            trend = np.polyfit(x_num, df[value_col], 1)
            trend_line = np.poly1d(trend)(x_num)
            future_x = np.arange(len(df), len(df) + 30)
            future_y = np.poly1d(trend)(future_x)

            fig.add_scatter(x=df[date_col], y=trend_line, mode='lines',
                            name='Trend Line', line=dict(dash='dash', color='orange'))
            future_dates = pd.date_range(start=df[date_col].iloc[-1] + pd.Timedelta(
                days=1), periods=30, freq=df[date_col].diff().median())
            fig.add_scatter(x=future_dates, y=future_y, mode='lines',
                            name='30-Day Forecast', line=dict(dash='dot', color='red'))

        # Moving average
        if len(df) > 7:
            ma = df[value_col].rolling(window=7).mean()
            fig.add_scatter(x=df[date_col], y=ma, mode='lines',
                            name='7-Period MA', line=dict(color='green'))

        st.plotly_chart(fig, width='stretch')

        st.markdown("### 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Period Start", df[date_col].min().strftime("%b %d, %Y"))
        col2.metric("Period End", df[date_col].max().strftime("%b %d, %Y"))
        col3.metric("Average", f"{df[value_col].mean():,.1f}")
        col4.metric(
            "Trend", "↗️ Upward" if df[value_col].iloc[-1] > df[value_col].mean() else "↘️ Downward")

        st.markdown("""
        **When to use:** Continuous metrics over time — sales, sensor data, stock prices.
        
        **Key Features:** Trend lines, moving averages, optional short-term forecast.
        """)

    def render_gantt_chart(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 📅 Gantt Chart - Project Timeline")

        if data_type != "Project Management":
            st.info(
                "Gantt works best with Project Management data. Showing sample project timeline.")
            sample_tasks = pd.DataFrame({
                'Task': ['Research', 'Design', 'Build', 'Test', 'Launch'],
                'Start': pd.date_range(start=datetime.now() - timedelta(days=60), periods=5, freq='14D'),
                'End': pd.date_range(start=datetime.now() - timedelta(days=46), periods=5, freq='14D'),
                'Progress': [100, 100, 80, 40, 0]
            })
            tasks_df = sample_tasks
        else:
            tasks_df = df

        fig = px.timeline(
            tasks_df, x_start="Start", x_end="End", y="Task",
            color="Progress" if "Progress" in tasks_df.columns else None,
            title=f"{data_type} - Project Gantt Chart",
            color_continuous_scale="Blues"
        )
        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, width='stretch')

        total_days = (tasks_df['End'].max() - tasks_df['Start'].min()).days
        avg_progress = tasks_df['Progress'].mean(
        ) if 'Progress' in tasks_df.columns else None

        col1, col2 = st.columns(2)
        col1.metric("Total Tasks", len(tasks_df))
        col2.metric("Project Span", f"{total_days} days")
        if avg_progress:
            st.metric("Average Progress", f"{avg_progress:.1f}%")

        st.markdown("""
        **When to use:** Project planning, task scheduling, milestone tracking.
        
        **Key Features:** Clear duration, overlaps, progress overlay.
        """)

    def render_horizon_chart(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🌅 Horizon Chart - Compact Long-Term View")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = st.selectbox("Value for Horizon", numeric_cols, index=0)
        date_col = 'Date' if 'Date' in df.columns else 'Timestamp'

        data = df[value_col].values
        dates = df[date_col]
        normalized = (data - data.mean()) / data.std()

        fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
        colors = ['#457b9d', '#1d3557', '#a8dadc', '#f1faee']

        for i in range(4):
            lower = i * 1.5
            upper = (i + 1) * 1.5
            band = np.clip(normalized, lower, upper) - lower

            axes[i].fill_between(dates, 0, band, color=colors[i], alpha=0.9)
            axes[i].set_ylim(0, 1.5)
            axes[i].set_ylabel(f"{lower:+.1f} to {upper:+.1f}σ")
            axes[i].grid(alpha=0.3)

        axes[-1].set_xlabel("Date")
        fig.suptitle(f"{data_type} - Horizon Chart ({value_col})")
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** Monitoring many long time series in limited space (dashboards).
        
        **Key Features:** Stacked bands compress vertical space while preserving detail.
        """)

    def render_timetable(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🕒 Timetable - Key Events & Milestones")

        # For non-project data, extract significant events
        if data_type != "Project Management":
            value_col = df.select_dtypes(include=[np.number]).columns[0]
            date_col = 'Date' if 'Date' in df.columns else 'Timestamp'
            values = df[value_col].values
            changes = np.abs(np.diff(values))
            threshold = np.mean(changes) + 2 * np.std(changes)
            event_indices = np.where(changes > threshold)[0] + 1

            events = []
            for idx in event_indices[:10]:
                events.append({
                    'Event': f"Significant {value_col} Change",
                    'Time': df[date_col].iloc[idx],
                    'Value': values[idx],
                    'Change': changes[idx-1]
                })
            events_df = pd.DataFrame(events)
        else:
            events_df = df[['Task', 'Start']].rename(
                columns={'Task': 'Event', 'Start': 'Time'})

        if events_df.empty:
            st.info("No significant events detected.")
            return

        fig, ax = plt.subplots(figsize=(12, len(events_df) * 0.6 + 2))
        for i, (_, event) in enumerate(events_df.iterrows()):
            time = pd.to_datetime(event['Time'])
            ax.plot(time, i, 'o', markersize=12, color='teal')
            ax.text(time, i + 0.2, event['Event'],
                    ha='center', va='bottom', fontsize=10)

        ax.set_yticks([])
        ax.set_title(f"{data_type} - Event Timetable")
        ax.grid(axis='x', alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(events_df, width='stretch')

        st.markdown("""
        **When to use:** Highlighting important dates, releases, spikes, or milestones.
        
        **Key Features:** Clean event timeline with context.
        """)

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Time Series Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Chronological Order**
            - Time always progresses left to right
            - Preserves sequence and causality
            """)
            st.markdown("""
            **Pattern Detection**
            - Reveals trends, cycles, seasonality
            - Highlights anomalies and shifts
            """)

        with col2:
            st.markdown("""
            **Forecasting Ready**
            - Supports trend extrapolation
            - Enables predictive modeling
            """)
            st.markdown("""
            **Real-Time Capable**
            - Ideal for live monitoring
            - Works with streaming data
            """)

    def render_examples(self, dataset_type: str, df: pd.DataFrame):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Stock Market": "Daily closing prices and volume",
            "Project Management": "Task timelines and dependencies",
            "Weather Forecasting": "Temperature and precipitation trends",
            "Business Metrics": "Revenue and customer growth",
            "IoT Monitoring": "Sensor readings over time",
            "Web Analytics": "Daily visitor traffic patterns"
        }

        for example, desc in examples.items():
            with st.expander(f"⏰ {example}"):
                st.write(desc)
                if dataset_type == example:
                    date_col = 'Date' if 'Date' in df.columns else 'Timestamp'
                    st.success(
                        f"**Period:** {df[date_col].min().strftime('%b %d, %Y')} → {df[date_col].max().strftime('%b %d, %Y')}")
                    st.info(f"**Data Points:** {len(df)}")

    def render_data_table(self, df: pd.DataFrame):
        st.markdown("### 📊 Time Series Data Sample")
        st.dataframe(df.head(20), width='stretch')

    def output(self):
        self.render_header()
        chart_type, dataset_type, time_range, frequency, show_forecast = self.render_configuration()

        df = self.generate_timeseries_data(dataset_type, time_range, frequency)

        chart_map = {
            "Time Series Plot": lambda: self.render_timeseries_plot(df, dataset_type, show_forecast),
            "Gantt Chart": lambda: self.render_gantt_chart(df, dataset_type),
            "Horizon Chart": lambda: self.render_horizon_chart(df, dataset_type),
            "Timetable": lambda: self.render_timetable(df, dataset_type)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_key_characteristics()
        self.render_examples(dataset_type, df)
        self.render_data_table(df)
