import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


class Trend:
    def __init__(self):
        # Page configuration
        st.markdown("## 📈 Trend Visualization Dashboard")

        # Configuration
        self.chart_type, self.dataset_type, self.time_period = self._get_user_inputs()
        self.periods_map = {"3 Months": 90,
                            "6 Months": 180, "1 Year": 365, "2 Years": 730}
        self.df = self.generate_trend_data(
            self.dataset_type, self.periods_map[self.time_period])

        # Chart functions mapping
        self.chart_functions = {
            "Line Chart": self.render_line_chart,
            "Area Chart": self.render_area_chart,
            "Sparkline": self.render_sparkline,
            "Cycle Plot": self.render_cycle_plot,
            "Timeline": self.render_timeline,
            "Run Chart": self.render_run_chart,
            "Streamgraph": self.render_streamgraph
        }

    def _get_user_inputs(self):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Line Chart", "Area Chart", "Sparkline", "Cycle Plot",
                 "Timeline", "Run Chart", "Streamgraph"]
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
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')

        if data_type == "Sales":
            base_trend = np.linspace(1000, 5000, periods)
            seasonal = 500 * np.sin(2 * np.pi * np.arange(periods) / 30)
            noise = np.random.normal(0, 200, periods)
            values = base_trend + seasonal + noise
            return pd.DataFrame({'Date': dates, 'Revenue': values})

        elif data_type == "Website Traffic":
            base_trend = np.linspace(500, 3000, periods)
            weekly_seasonal = 300 * np.sin(2 * np.pi * np.arange(periods) / 7)
            monthly_seasonal = 150 * \
                np.sin(2 * np.pi * np.arange(periods) / 30)
            noise = np.random.normal(0, 100, periods)
            values = base_trend + weekly_seasonal + monthly_seasonal + noise
            return pd.DataFrame({'Date': dates, 'Visitors': values})

        elif data_type == "Temperature":
            base = 20 * np.ones(periods)
            seasonal = 15 * np.sin(2 * np.pi * np.arange(periods) / 365)
            noise = np.random.normal(0, 3, periods)
            values = base + seasonal + noise
            return pd.DataFrame({'Date': dates, 'Temperature': values})

        elif data_type == "Stock Prices":
            returns = np.random.normal(0.001, 0.02, periods)
            prices = 100 * np.cumprod(1 + returns)
            return pd.DataFrame({'Date': dates, 'Price': prices})

        else:  # Social Media
            base_trend = np.linspace(1000, 10000, periods)
            viral_spikes = np.random.poisson(0.05, periods) * 500
            noise = np.random.normal(0, 200, periods)
            values = base_trend + viral_spikes + noise
            return pd.DataFrame({'Date': dates, 'Followers': values})

    def render_chart(self):
        if self.chart_type in self.chart_functions:
            self.chart_functions[self.chart_type](self.df, self.dataset_type)

    # Chart rendering functions
    def render_line_chart(self, df, data_type):
        self._render_basic_chart(
            df, data_type, "📊 Line Chart - Basic Trend Visualization",
            px.line, "Line Chart", "Revenue",
            "**When to use:** Showing continuous data over time, identifying trends and patterns",
            '''
import plotly.express as px
fig = px.line(df, x='Date', y='Revenue', title='Sales Trend Over Time')
fig.show()
            '''
        )

    def render_area_chart(self, df, data_type):
        self._render_basic_chart(
            df, data_type, "📈 Area Chart - Cumulative Trend Visualization",
            px.area, "Area Chart", "Revenue",
            "**When to use:** Showing cumulative totals over time, emphasizing volume",
            '''
import plotly.express as px
fig = px.area(df, x='Date', y='Revenue', title='Cumulative Revenue')
fig.show()
            '''
        )

    def render_sparkline(self, df, data_type):
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        st.subheader("⚡ Sparkline - Compact Trend Indicator")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df.iloc[:, 1], mode='lines', line=dict(width=2)))
        fig.update_layout(height=100, showlegend=False,
                          margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_showticklabels=False, yaxis_showticklabels=False)
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            "**When to use:** Inline trend indicators, dashboards with limited space")
        code = '''
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Revenue'], mode='lines', line=dict(width=2)))
fig.update_layout(height=100, showlegend=False,
                 margin=dict(l=10, r=10, t=30, b=10))
fig.show()
        '''
        self._display_code(code)
        st.markdown('</div>', unsafe_allow_html=True)

    def render_cycle_plot(self, df, data_type):
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        st.subheader("🔄 Cycle Plot - Seasonal Pattern Analysis")

        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year

        fig = px.line(
            df, x='Month', y=df.columns[1], color='Year', title=f"{data_type} - Seasonal Cycle Plot")
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            "**When to use:** Comparing seasonal patterns across multiple years")
        code = '''
import plotly.express as px
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
fig = px.line(df, x='Month', y='Revenue', color='Year', title='Seasonal Revenue Patterns')
fig.show()
        '''
        self._display_code(code)
        st.markdown('</div>', unsafe_allow_html=True)

    def render_timeline(self, df, data_type):
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        st.subheader("⏰ Timeline - Event-based Trend Visualization")

        events = {
            'Product Launch': df['Date'].iloc[len(df)//4],
            'Marketing Campaign': df['Date'].iloc[len(df)//2],
            'Holiday Season': df['Date'].iloc[3*len(df)//4]
        }

        fig = px.line(df, x='Date', y=df.columns[1])
        for event, date in events.items():
            fig.add_vline(x=date, line_dash="dash", line_color="red",
                          annotation_text=event, annotation_position="top")
        st.plotly_chart(fig, width='stretch')
        st.markdown(
            "**When to use:** Showing trends with key events and milestones")
        code = '''
import plotly.express as px
fig = px.line(df, x='Date', y='Revenue')
fig.add_vline(x=pd.Timestamp('2024-06-01'), line_dash="dash", annotation_text="Product Launch")
fig.show()
        '''
        self._display_code(code)
        st.markdown('</div>', unsafe_allow_html=True)

    def render_run_chart(self, df, data_type):
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        st.subheader("📏 Run Chart - Process Monitoring")

        median_value = df.iloc[:, 1].median()
        fig = px.line(df, x='Date', y=df.columns[1])
        fig.add_hline(y=median_value, line_dash="dash",
                      annotation_text=f"Median: {median_value:.0f}")
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            "**When to use:** Statistical process control, monitoring process stability")
        code = f'''
import plotly.express as px
median_val = df['Revenue'].median()
fig = px.line(df, x='Date', y='Revenue')
fig.add_hline(y=median_val, line_dash="dash", annotation_text=f"Median: {{median_val:.0f}}")
fig.show()
        '''
        self._display_code(code)
        st.markdown('</div>', unsafe_allow_html=True)

    def render_streamgraph(self, df, data_type):
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        st.subheader("🌊 Streamgraph - Stacked Area Visualization")

        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        categories = ['Category A', 'Category B', 'Category C', 'Category D']

        stream_data = []
        for i, category in enumerate(categories):
            base = (i+1)*1000 + np.linspace(0, 2000, 365)
            seasonal = 200*np.sin(2*np.pi*np.arange(365)/30 + i*np.pi/2)
            noise = np.random.normal(0, 100, 365)
            values = base + seasonal + noise
            for date, value in zip(dates, values):
                stream_data.append(
                    {'Date': date, 'Category': category, 'Value': value})

        stream_df = pd.DataFrame(stream_data)
        fig = px.area(stream_df, x='Date', y='Value', color='Category',
                      title="Streamgraph - Multiple Categories")
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            "**When to use:** Showing composition changes over time across multiple categories")
        code = '''
import plotly.express as px
import pandas as pd
categories = ['Cat A', 'Cat B', 'Cat C']
data = []
for cat in categories:
    for date in pd.date_range('2024-01-01', periods=365):
        data.append({'Date': date, 'Category': cat, 'Value': np.random.randint(100,1000)})
df_stream = pd.DataFrame(data)
fig = px.area(df_stream, x='Date', y='Value', color='Category')
fig.show()
        '''
        self._display_code(code)
        st.markdown('</div>', unsafe_allow_html=True)

    # Helper functions
    def _render_basic_chart(self, df, data_type, subtitle, plot_func, chart_name, y_col, explanation, code):
        st.markdown('<div class="chart-section">', unsafe_allow_html=True)
        st.subheader(subtitle)
        fig = plot_func(
            df, x='Date', y=df.columns[1], title=f"{data_type} - {chart_name}")
        st.plotly_chart(fig, width='stretch')
        st.markdown(explanation)
        self._display_code(code)
        st.markdown('</div>', unsafe_allow_html=True)

    def _display_code(self, code):
        st.markdown('<div class="code-block">', unsafe_allow_html=True)
        st.code(code, language='python')
        st.markdown('</div>', unsafe_allow_html=True)

    def render_guide(self):
        with st.expander("📚 Trend Visualization Guide"):
            st.markdown("""
            **Trend Visualization Characteristics:**
            - **Temporal Progression**: Shows data evolution over time
            - **Rate of Change**: Highlights growth/decline patterns
            - **Pattern Recognition**: Identifies cycles and seasonality
            - **Anomaly Detection**: Spots outliers and disruptions
            
            **Best Practices:**
            - Choose appropriate time intervals
            - Use consistent scaling
            - Highlight important events
            - Consider seasonality adjustments
            - Use annotations for context
            """)

    def output(self):
        self.render_chart()
        self.render_guide()
