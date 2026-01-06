import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class Comparison:
    def __init__(self):
        st.set_page_config(
            page_title="Comparison Visualizations", layout="wide")
        self.title = "📊 Comparison Visualizations Dashboard"
        self.chart_scenario_map = {
            "Bar Chart (Vertical)": [
                "Regional Sales Revenue",
                "Product Category Performance",
                "A/B Test Conversion Rates"
            ],
            "Lollipop Chart": [
                "Regional Sales Revenue"
            ],
            "Grouped Bar Chart": [
                "Product Category Performance",
                "Quarterly Metrics Comparison"
            ],
            "100% Stacked Column Chart": [
                "Market Share by Brand"
            ]
        }
        self.chart_types = [
            "Bar Chart (Vertical)", "Lollipop Chart", "Grouped Bar Chart", "100% Stacked Column Chart"]
        self.scenarios = ["Regional Sales Revenue", "Product Category Performance",
                          "Quarterly Metrics Comparison", "Market Share by Brand", "A/B Test Conversion Rates"]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Comparison charts facilitate **direct, clear comparison** between categories, groups, or time periods.
        They make differences and similarities immediately apparent.
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2 = st.columns(2)

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type",
                self.chart_types,
                key="chart_type"
            )

        valid_scenarios = self.chart_scenario_map[chart_type]

        with col2:
            scenario = st.selectbox(
                "Scenario",
                valid_scenarios,
                key="scenario"
            )

        return chart_type, scenario

    def get_sample_data(self, scenario: str) -> pd.DataFrame:
        np.random.seed(42)

        if scenario == "Regional Sales Revenue":
            regions = ["North America", "Europe", "Asia-Pacific",
                       "Latin America", "Middle East & Africa"]
            revenue = [450, 320, 280, 150, 100]
            df = pd.DataFrame({"Region": regions, "Revenue ($M)": revenue})

        elif scenario == "Product Category Performance":
            categories = ["Electronics", "Clothing",
                          "Home & Garden", "Sports", "Beauty", "Books"]
            sales = [320, 280, 220, 180, 140, 90]
            profit_margin = [22, 35, 28, 30, 42, 18]
            df = pd.DataFrame(
                {"Category": categories, "Sales ($K)": sales, "Profit Margin (%)": profit_margin})

        elif scenario == "Quarterly Metrics Comparison":
            quarters = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
            metrics = ["Revenue ($M)", "New Customers",
                       "Customer Satisfaction"]
            data = np.array([
                [45, 52, 58, 64],      # Revenue
                [12000, 14500, 13800, 16200],  # New Customers
                [8.7, 8.9, 9.1, 9.3]    # CSAT
            ])
            df = pd.DataFrame(data.T, columns=metrics,
                              index=quarters).reset_index(names="Quarter")

        elif scenario == "Market Share by Brand":
            brands = ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E"]
            segments = ["Segment 1", "Segment 2", "Segment 3"]
            # Random proportional data
            raw = np.random.dirichlet(
                np.ones(len(segments)), size=len(brands)) * 100
            df = pd.DataFrame(raw.round(1), columns=segments,
                              index=brands).reset_index(names="Brand")

        else:  # A/B Test Conversion Rates
            variants = ["Control", "Variant A", "Variant B", "Variant C"]
            traffic = [50000, 48000, 52000, 49000]
            conversions = [2450, 2880, 3120, 2695]
            df = pd.DataFrame({
                "Variant": variants,
                "Visitors": traffic,
                "Conversions": conversions
            })
            df["Conversion Rate (%)"] = (
                df["Conversions"] / df["Visitors"] * 100).round(2)

        return df

    def create_vertical_bar(self, df: pd.DataFrame, scenario: str) -> go.Figure:
        if scenario == "Regional Sales Revenue":
            x_col, y_col = "Region", "Revenue ($M)"
            title = "Revenue by Region"

        elif scenario == "Product Category Performance":
            x_col, y_col = "Category", "Sales ($K)"
            title = "Sales by Category"

        elif scenario == "A/B Test Conversion Rates":
            x_col, y_col = "Variant", "Conversion Rate (%)"
            title = "A/B Test Conversion Rates"

        else:
            raise ValueError("Unsupported scenario for Vertical Bar Chart")

        fig = go.Figure(go.Bar(
            x=df[x_col],
            y=df[y_col],
            text=df[y_col],
            textposition="outside"
        ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=600
        )

        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go

if scenario == "Regional Sales Revenue":
    x_col, y_col = "Region", "Revenue ($M)"
    title = "Revenue by Region"

elif scenario == "Product Category Performance":
    x_col, y_col = "Category", "Sales ($K)"
    title = "Sales by Category"

elif scenario == "A/B Test Conversion Rates":
    x_col, y_col = "Variant", "Conversion Rate (%)"
    title = "A/B Test Conversion Rates"

else:
    raise ValueError("Unsupported scenario for Vertical Bar Chart")

fig = go.Figure(go.Bar(
    x=df[x_col],
    y=df[y_col],
    text=df[y_col],
    textposition="outside"
))

fig.update_layout(
    title=title,
    xaxis_title=x_col,
    yaxis_title=y_col,
    height=600
)
fig.show()
""", language="python")

    def create_lollipop(self, df: pd.DataFrame, scenario: str) -> go.Figure:
        if "Profit Margin" in df.columns:
            x_col = "Category"
            y_col = "Profit Margin (%)"
            title = "Profit Margin by Category"
        else:
            x_col = "Region" if "Region" in df.columns else "Variant"
            y_col = df.columns[1]
            title = "Performance Comparison"

        fig = go.Figure()

        # Stem
        fig.add_trace(go.Scatter(
            x=df[y_col],
            y=df[x_col],
            mode='lines',
            line=dict(color='#7f8c8d', width=2),
            hoverinfo='none'
        ))

        # Lollipop head
        fig.add_trace(go.Scatter(
            x=df[y_col],
            y=df[x_col],
            mode='markers+text',
            marker=dict(size=16, color='#3498db'),
            text=df[y_col],
            textposition='top right'
        ))

        fig.update_layout(
            title=f"Lollipop Chart – {title}",
            xaxis_title=y_col,
            yaxis_title="",
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go

if "Profit Margin" in df.columns:
    x_col = "Category"
    y_col = "Profit Margin (%)"
    title = "Profit Margin by Category"
else:
    x_col = "Region" if "Region" in df.columns else "Variant"
    y_col = df.columns[1]
    title = "Performance Comparison"

fig = go.Figure()

# Stem
fig.add_trace(go.Scatter(
    x=df[y_col],
    y=df[x_col],
    mode='lines',
    line=dict(color='#7f8c8d', width=2),
    hoverinfo='none'
))

# Lollipop head
fig.add_trace(go.Scatter(
    x=df[y_col],
    y=df[x_col],
    mode='markers+text',
    marker=dict(size=16, color='#3498db'),
    text=df[y_col],
    textposition='top right'
))

fig.update_layout(
    title=f"Lollipop Chart – {title}",
    xaxis_title=y_col,
    yaxis_title="",
    height=600,
    showlegend=False
)
fig.show()
""", language="python")

    def create_grouped_bar(self, df: pd.DataFrame, scenario: str) -> go.Figure:
        if "Quarter" in df.columns:
            fig = go.Figure()
            for metric in ["Revenue ($M)", "New Customers", "Customer Satisfaction"]:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=df["Quarter"],
                    y=df[metric]
                ))
            title = "Quarterly Metrics"
        else:  # Product categories with Sales & Margin
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Sales ($K)",
                x=df["Category"],
                y=df["Sales ($K)"]
            ))
            fig.add_trace(go.Bar(
                name="Profit Margin (%)",
                x=df["Category"],
                y=df["Profit Margin (%)"]
            ))
            title = "Sales vs Profit Margin"

        fig.update_layout(
            title=f"Grouped Bar Chart – {title}",
            barmode='group',
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go

if "Quarter" in df.columns:
    fig = go.Figure()
    for metric in ["Revenue ($M)", "New Customers", "Customer Satisfaction"]:
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Quarter"],
            y=df[metric]
        ))
    title = "Quarterly Metrics"
else:  # Product categories with Sales & Margin
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Sales ($K)",
        x=df["Category"],
        y=df["Sales ($K)"]
    ))
    fig.add_trace(go.Bar(
        name="Profit Margin (%)",
        x=df["Category"],
        y=df["Profit Margin (%)"]
    ))
    title = "Sales vs Profit Margin"

fig.update_layout(
    title=f"Grouped Bar Chart – {title}",
    barmode='group',
    height=600
)
fig.show()
""", language="python")

    def create_stacked_100(self, df: pd.DataFrame, scenario: str) -> go.Figure:
        if scenario != "Market Share by Brand":
            raise ValueError(
                "100% Stacked Chart only supports Market Share by Brand")

        segments = [c for c in df.columns if c != "Brand"]

        fig = go.Figure()
        for seg in segments:
            fig.add_trace(go.Bar(
                name=seg,
                y=df["Brand"],
                x=df[seg],
                orientation="h"
            ))

        fig.update_layout(
            title="100% Stacked Column Chart – Market Share by Brand",
            barmode="stack",
            xaxis_title="Percentage",
            yaxis_title="Brand",
            height=600
        )

        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go

segments = [c for c in df.columns if c != "Brand"]

fig = go.Figure()
for seg in segments:
    fig.add_trace(go.Bar(
        name=seg,
        y=df["Brand"],
        x=df[seg],
        orientation="h"
    ))

fig.update_layout(
    title="100% Stacked Column Chart – Market Share by Brand",
    barmode="stack",
    xaxis_title="Percentage",
    yaxis_title="Brand",
    height=600
)

fig.show()
""", language="python")

    def render_chart(self, chart_type: str, scenario: str):
        df = self.get_sample_data(scenario)
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Bar Chart (Vertical)":
            self.create_vertical_bar(df, scenario)
        elif chart_type == "Lollipop Chart":
            self.create_lollipop(df, scenario)
        elif chart_type == "Grouped Bar Chart":
            self.create_grouped_bar(df, scenario)
        else:  # 100% Stacked
            self.create_stacked_100(df, scenario)

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Sales Analysis": "Revenue comparison across regions",
            "Product Performance": "Market share of competing products",
            "Period Comparison": "Q1 vs Q2 vs Q3 vs Q4 performance",
            "Benchmark Studies": "Company metrics vs industry average",
            "A/B Testing": "Conversion rates between variants"
        }

        for example, description in examples.items():
            with st.expander(f"💭 {example}"):
                st.write(description)

    def render_key_characteristics(self):
        st.markdown("### ⚖️ Understanding Data Comparison")

        st.markdown("""
        Comparison analysis examines **differences and similarities between data points or groups**.
        It helps uncover performance gaps, trends, and relative strengths.
        """)

        st.markdown("#### 📊 Highlights Differences")
        st.markdown("""
        Shows how metrics vary across categories, regions, or periods:
        - Spot high and low performers  
        - Identify gaps in performance  
        - Detect shifts over time
        """)

        st.markdown("#### 🎯 Supports Benchmarking")
        st.markdown("""
        Compares your data against standards or targets:
        - Industry averages  
        - Competitor performance  
        - Internal KPIs  
        Helps measure success relative to goals.
        """)

        st.markdown("#### ⏱️ Tracks Period Over Period")
        st.markdown("""
        Measures changes across time:
        - Quarterly or yearly comparisons  
        - Month-over-month growth  
        - Pre/post intervention metrics  
        Useful for trend detection and seasonality analysis.
        """)

        st.markdown("#### 🧪 Enables A/B Testing Insights")
        st.markdown("""
        Evaluates the impact of variations:
        - Conversion rates between different versions  
        - Feature performance comparisons  
        - Campaign effectiveness  
        Helps make data-driven decisions on changes or optimizations.
        """)

        st.divider()

        st.markdown("#### 🎯 Why Comparison Analysis Matters")
        st.markdown("""
        Comparison analysis is essential for:
        - Performance evaluation  
        - Strategic decision making  
        - Identifying opportunities for improvement  
        - Validating experiments and initiatives
        """)

        st.markdown("#### 🔹 Examples of Comparison Analysis")
        st.markdown("""
        - **Sales Analysis:** Revenue comparison across regions  
        - **Product Performance:** Market share of competing products  
        - **Period Comparison:** Q1 vs Q2 vs Q3 vs Q4 performance  
        - **Benchmark Studies:** Company metrics vs industry average  
        - **A/B Testing:** Conversion rates between variants
        """)

    def output(self):
        self.render_header()
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
        self.render_examples()
        self.render_key_characteristics()
