import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd


class Distribution:

    def __init__(self):
        self.title = "📊 Distribution Visualizations Dashboard"
        self.chart_types = [
            "Histogram",
            "Violin Plot",
            "Dot Plot",
            "Ridge Plot",
            "Strip Plot",
            "Bubble Timeline"
        ]
        self.scenarios = [
            "Product Measurement Quality",
            "Employee Salary Distribution",
            "Student Test Scores",
            "Patient Wait Times",
            "E-Commerce Order Values"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Distribution charts reveal **how values are spread across a range**, showing central tendencies, 
        variability, and the shape of data distributions.
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
            num_samples = st.slider(
                "Number of Samples", 200, 2000, 800, step=200)

        return chart_type, scenario, num_samples

    def generate_data(self, scenario: str, n: int):
        np.random.seed(42)

        if scenario == "Product Measurement Quality":
            # Near-normal with slight skew
            data = np.random.normal(50, 5, n)
            data = np.append(data, np.random.normal(
                62, 3, int(n*0.05)))  # Outliers
            categories = None
            title = "Product Dimension (mm)"

        elif scenario == "Employee Salary Distribution":
            # Right-skewed (log-normal)
            data = np.random.lognormal(mean=4.8, sigma=0.5, size=n) * 10000
            categories = np.random.choice(
                ["Engineering", "Sales", "Marketing", "Operations"], n)
            title = "Annual Salary ($)"

        elif scenario == "Student Test Scores":
            # Bimodal
            data = np.concatenate([
                np.random.normal(65, 10, int(n*0.6)),
                np.random.normal(90, 8, int(n*0.4))
            ])
            categories = np.random.choice(["Class A", "Class B", "Class C"], n)
            title = "Test Score (%)"

        elif scenario == "Patient Wait Times":
            # Exponential-like
            data = np.random.exponential(scale=20, size=n) + 5
            categories = np.random.choice(
                ["Morning", "Afternoon", "Evening"], n)
            title = "Wait Time (minutes)"

        else:  # E-Commerce Order Values
            # Heavy right skew
            data = np.random.pareto(a=2, size=n) * 50 + 20
            categories = None
            dates = pd.date_range(
                "2025-01-01", periods=n, freq='H')[np.random.choice(n, n, replace=False)]
            df = pd.DataFrame({"Value": data, "Date": dates})
            return df, "Order Value ($)", "Bubble Timeline"

        df = pd.DataFrame({"Value": data})
        if categories is not None:
            df["Category"] = categories

        return df, title, scenario

    def create_histogram(self, df: pd.DataFrame, title: str):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["Value"],
            nbinsx=30,
            marker_color='#3498db',
            opacity=0.7
        ))
        fig.update_layout(
            title=f"Histogram – {title}",
            xaxis_title=title,
            yaxis_title="Count",
            bargap=0.05,
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=df["Value"],
    nbinsx=30,
    marker_color='#3498db',
    opacity=0.7
))
fig.update_layout(
    title=f"Histogram – {title}",
    xaxis_title=title,
    yaxis_title="Count",
    bargap=0.05,
    height=600
)
fig.show()           
""", language="python")

    def create_violin(self, df: pd.DataFrame, title: str):
        if "Category" in df.columns:
            categories = df["Category"].unique()
            fig = go.Figure()
            for cat in categories:
                fig.add_trace(go.Violin(
                    y=df[df["Category"] == cat]["Value"],
                    name=cat,
                    box_visible=True,
                    meanline_visible=True
                ))
        else:
            fig = go.Figure(
                go.Violin(y=df["Value"], box_visible=True, meanline_visible=True))

        fig.update_layout(
            title=f"Violin Plot – {title}",
            yaxis_title=title,
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

if "Category" in df.columns:
    categories = df["Category"].unique()
    fig = go.Figure()
    for cat in categories:
        fig.add_trace(go.Violin(
            y=df[df["Category"] == cat]["Value"],
            name=cat,
            box_visible=True,
            meanline_visible=True
        ))
else:
    fig = go.Figure(
        go.Violin(y=df["Value"], box_visible=True, meanline_visible=True))

fig.update_layout(
    title=f"Violin Plot – {title}",
    yaxis_title=title,
    height=600
)
fig.show()           
""", language="python")

    def create_dot_plot(self, df: pd.DataFrame, title: str):
        fig = go.Figure()
        if "Category" in df.columns:
            for cat in df["Category"].unique():
                subset = df[df["Category"] == cat]
                fig.add_trace(go.Scatter(
                    x=subset["Value"],
                    y=[cat] * len(subset),
                    mode='markers',
                    marker=dict(size=8, opacity=0.6),
                    name=cat
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df["Value"],
                y=np.zeros(len(df)),
                mode='markers',
                marker=dict(size=8)
            ))

        fig.update_layout(
            title=f"Dot Plot – {title}",
            height=600,
            showlegend="Category" in df.columns
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

fig = go.Figure()
if "Category" in df.columns:
    for cat in df["Category"].unique():
        subset = df[df["Category"] == cat]
        fig.add_trace(go.Scatter(
            x=subset["Value"],
            y=[cat] * len(subset),
            mode='markers',
            marker=dict(size=8, opacity=0.6),
            name=cat
        ))
else:
    fig.add_trace(go.Scatter(
        x=df["Value"],
        y=np.zeros(len(df)),
        mode='markers',
        marker=dict(size=8)
    ))

fig.update_layout(
    title=f"Dot Plot – {title}",
    height=600,
    showlegend="Category" in df.columns
)
fig.show()           
""", language="python")

    def create_ridge_plot(self, df: pd.DataFrame, title: str):
        if "Category" not in df.columns:
            st.warning(
                "Ridge Plot requires categorical data. Falling back to Violin.")
            return self.create_violin(df, title)

        categories = sorted(df["Category"].unique())
        fig = go.Figure()

        colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]
        for i, cat in enumerate(categories):
            subset = df[df["Category"] == cat]["Value"]
            hist_data = np.histogram(subset, bins=40)
            hist_x = (hist_data[1][:-1] + hist_data[1][1:]) / 2
            hist_y = hist_data[0]
            hist_y_smooth = np.convolve(hist_y, np.ones(5)/5, mode='same')

            fig.add_trace(go.Scatter(
                x=hist_x,
                y=hist_y_smooth + i,
                fill='tozeroy',
                mode='lines',
                name=cat,
                fillcolor=colors[i % len(colors)],
                opacity=0.6
            ))

        fig.update_layout(
            title=f"Ridge Plot – {title} by Category",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(categories))),
                ticktext=categories
            ),
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

categories = sorted(df["Category"].unique())
fig = go.Figure()

colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]
for i, cat in enumerate(categories):
    subset = df[df["Category"] == cat]["Value"]
    hist_data = np.histogram(subset, bins=40)
    hist_x = (hist_data[1][:-1] + hist_data[1][1:]) / 2
    hist_y = hist_data[0]
    hist_y_smooth = np.convolve(hist_y, np.ones(5)/5, mode='same')

    fig.add_trace(go.Scatter(
        x=hist_x,
        y=hist_y_smooth + i,
        fill='tozeroy',
        mode='lines',
        name=cat,
        fillcolor=colors[i % len(colors)],
        opacity=0.6
    ))

fig.update_layout(
    title=f"Ridge Plot – {title} by Category",
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(len(categories))),
        ticktext=categories
    ),
    height=600,
    showlegend=True
)
fig.show()           
""", language="python")

    def create_strip_plot(self, df: pd.DataFrame, title: str):
        if "Category" not in df.columns:
            st.warning("Strip Plot works best with categories.")
            return self.create_dot_plot(df, title)

        fig = go.Figure()
        categories = df["Category"].unique()
        for i, cat in enumerate(categories):
            subset = df[df["Category"] == cat]["Value"]
            jitter = np.random.uniform(-0.2, 0.2, len(subset))
            fig.add_trace(go.Scatter(
                x=subset + jitter,
                y=[cat] * len(subset),
                mode='markers',
                marker=dict(size=8, opacity=0.7),
                name=cat
            ))

        fig.update_layout(
            title=f"Strip Plot – {title} by Category",
            xaxis_title=title,
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

fig = go.Figure()
categories = df["Category"].unique()
for i, cat in enumerate(categories):
    subset = df[df["Category"] == cat]["Value"]
    jitter = np.random.uniform(-0.2, 0.2, len(subset))
    fig.add_trace(go.Scatter(
        x=subset + jitter,
        y=[cat] * len(subset),
        mode='markers',
        marker=dict(size=8, opacity=0.7),
        name=cat
    ))

fig.update_layout(
    title=f"Strip Plot – {title} by Category",
    xaxis_title=title,
    height=600
)
fig.show()           
""", language="python")

    def create_bubble_timeline(self, df: pd.DataFrame, title: str):
        fig = go.Figure()
        sizes = np.sqrt(df["Value"]) * 2

        if "Date" not in df.columns:
            df["Date"] = pd.date_range("2025-01-01", periods=len(df))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Value"],
            mode='markers',
            marker=dict(
                size=sizes,
                sizemode='area',
                sizeref=2. * max(sizes) / (80**2),
                sizemin=4,
                color=df["Value"],
                colorscale='Viridis',
                showscale=True
            ),
            text=df["Value"].round(0),
            hoverinfo='text+x'
        ))

        fig.update_layout(
            title=f"Bubble Timeline – {title} Over Time<br><sub>Bubble size = Order Value</sub>",
            xaxis_title="Date",
            yaxis_title=title,
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

fig = go.Figure()
sizes = np.sqrt(df["Value"]) * 2

if "Date" not in df.columns:
    df["Date"] = pd.date_range("2025-01-01", periods=len(df))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Value"],
    mode='markers',
    marker=dict(
        size=sizes,
        sizemode='area',
        sizeref=2. * max(sizes) / (80**2),
        sizemin=4,
        color=df["Value"],
        colorscale='Viridis',
        showscale=True
    ),
    text=df["Value"].round(0),
    hoverinfo='text+x'
))

fig.update_layout(
    title=f"Bubble Timeline – {title} Over Time<br><sub>Bubble size = Order Value</sub>",
    xaxis_title="Date",
    yaxis_title=title,
    height=600
)
fig.show()           
""", language="python")

    def render_chart(self, chart_type: str, scenario: str, n_samples: int):
        st.markdown(f"### {chart_type}: {scenario}")

        data_result = self.generate_data(scenario, n_samples)
        df = data_result[0]
        title = data_result[1]

        if chart_type == "Histogram":
            self.create_histogram(df, title)
        elif chart_type == "Violin Plot":
            self.create_violin(df, title)
        elif chart_type == "Dot Plot":
            self.create_dot_plot(df, title)
        elif chart_type == "Ridge Plot":
            self.create_ridge_plot(df, title)
        elif chart_type == "Strip Plot":
            self.create_strip_plot(df, title)
        else:  # Bubble Timeline
            self.create_bubble_timeline(df, title)

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Quality Control": "Distribution of product measurements",
            "Human Resources": "Salary distributions across the organization",
            "Education": "Test score distributions across students",
            "Healthcare": "Patient wait times distribution",
            "E-commerce": "Order value distribution"
        }

        for example, description in examples.items():
            with st.expander(f"💭 {example}"):
                st.write(description)

    def render_key_characteristics(self):
        st.markdown("### 📊 Understanding Data Distribution")

        st.markdown("""
        Distribution analysis examines **how data points are spread across a range of values**.
        It focuses on the shape, concentration, and variability of data.
        """)

        st.markdown("#### 📈 Shows Data Spread")
        st.markdown("""
        Visualizes how values are dispersed, helping to identify:
        - Wide vs narrow ranges  
        - Dense vs sparse regions  
        - Variability in measurements
        """)

        st.markdown("#### 🎯 Reveals Distribution Shape")
        st.markdown("""
        Determines the underlying shape of the data:
        - Symmetric vs skewed  
        - Normal, uniform, exponential, or custom patterns  
        - Helps select appropriate statistical tests and models
        """)

        st.markdown("#### ⚖️ Identifies Central Tendency")
        st.markdown("""
        Highlights key measures like:
        - Mean  
        - Median  
        - Mode  

        Provides insight into where most data points are concentrated.
        """)

        st.markdown("#### 🚨 Highlights Outliers and Skewness")
        st.markdown("""
        Detects unusual or extreme values that may affect analysis:
        - Outliers that deviate from patterns  
        - Skewness indicating asymmetry  
        - Guides data cleaning and transformation decisions
        """)

        st.divider()

        st.markdown("#### 🎯 Why Distribution Analysis Matters")
        st.markdown("""
        Understanding distribution is critical for:
        - Statistical modeling  
        - Feature engineering  
        - Detecting anomalies  
        - Making data-driven decisions  
        """)

    def output(self):
        self.render_header()
        chart_type, scenario, num_samples = self.render_configuration()
        self.render_chart(chart_type, scenario, num_samples)
        self.render_examples()
        self.render_key_characteristics()
