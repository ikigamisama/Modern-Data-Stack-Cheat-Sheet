import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class Deviation:

    def __init__(self):
        st.set_page_config(
            page_title="Deviation Visualizations", layout="wide")
        self.title = "↔️ Deviation Visualizations Dashboard"
        self.chart_types = [
            "Diverging Bar Chart",
            "Back-to-Back Bar Chart"
        ]
        self.scenarios = [
            "Budget vs Actual Spending",
            "Sales Performance vs Target",
            "Customer Satisfaction Survey",
            "Regional Revenue Variance",
            "Employee KPI Deviation"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Deviation charts highlight **differences from a reference point, baseline, or average**. 
        They make positive and negative variances immediately visible.
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2 = st.columns([2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            scenario = st.selectbox("Scenario", self.scenarios, key="scenario")

        return chart_type, scenario

    def get_sample_data(self, scenario: str) -> pd.DataFrame:
        np.random.seed(42)

        if scenario == "Budget vs Actual Spending":
            categories = ["Marketing", "R&D", "Salaries",
                          "Operations", "Facilities", "Travel", "Training"]
            budget = [120, 180, 450, 90, 60, 40, 25]
            actual = [135, 165, 460, 85, 70, 25, 30]
            df = pd.DataFrame(
                {"Category": categories, "Budget": budget, "Actual": actual})
            df["Variance"] = df["Actual"] - df["Budget"]
            df["Variance %"] = (df["Variance"] / df["Budget"] * 100).round(1)

        elif scenario == "Sales Performance vs Target":
            reps = ["North", "South", "East", "West", "Central"]
            target = [500, 420, 380, 460, 410]
            actual = [520, 390, 420, 440, 430]
            df = pd.DataFrame(
                {"Region": reps, "Target": target, "Actual": actual})
            df["Variance"] = df["Actual"] - df["Target"]

        elif scenario == "Customer Satisfaction Survey":
            statements = [
                "Easy to use", "Good value for money", "Reliable performance",
                "Excellent support", "Fast delivery", "Modern design",
                "Meets my needs"
            ]
            # Net score: % Agree - % Disagree
            net_scores = [42, 28, 35, -15, 18, 51, 38]
            df = pd.DataFrame(
                {"Statement": statements, "Net Score": net_scores})

        elif scenario == "Regional Revenue Variance":
            regions = ["Americas", "EMEA", "APAC", "Japan", "China"]
            forecast = [1200, 850, 680, 320, 410]
            actual = [1180, 890, 620, 350, 430]
            df = pd.DataFrame(
                {"Region": regions, "Forecast": forecast, "Actual": actual})
            df["Variance"] = df["Actual"] - df["Forecast"]

        else:  # Employee KPI Deviation
            employees = ["A. Johnson", "B. Lee", "C. Rivera",
                         "D. Patel", "E. Kim", "F. Garcia"]
            target_score = 85
            scores = [92, 78, 88, 70, 95, 82]
            df = pd.DataFrame({"Employee": employees, "Score": scores})
            df["Deviation"] = df["Score"] - target_score

        return df

    def create_diverging_bar(self, df: pd.DataFrame, scenario: str) -> go.Figure:
        if "Survey" in scenario:
            x_col = "Net Score"
            categories = "Statement"
            title = "Customer Satisfaction – Net Promoter Scores"
        elif "Employee" in scenario:
            x_col = "Deviation"
            categories = "Employee"
            title = "Employee KPI Deviation from Target (85)"
        else:
            df["Deviation"] = df["Variance"]
            x_col = "Deviation"
            categories = "Category" if "Budget" in scenario else "Region"
            title = "Variance from Target/Forecast"

        # Color: green for positive, red for negative
        colors = ['#e74c3c' if v < 0 else '#27ae60' for v in df[x_col]]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            y=df[categories],
            x=df[x_col],
            orientation='h',
            marker_color=colors,
            text=df[x_col],
            textposition='outside',
            hoverinfo='text',
            hovertext=df[categories] + ": " + df[x_col].astype(str)
        ))

        fig.add_vline(x=0, line=dict(color="black", width=2))

        fig.update_layout(
            title=f"Diverging Bar Chart – {title}<br><sub>Positive (green) = above target | Negative (red) = below target</sub>",
            xaxis_title="Deviation",
            yaxis_title="",
            height=600,
            showlegend=False,
            plot_bgcolor="white"
        )

        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
if "Survey" in scenario:
    x_col = "Net Score"
    categories = "Statement"
    title = "Customer Satisfaction – Net Promoter Scores"
elif "Employee" in scenario:
    x_col = "Deviation"
    categories = "Employee"
    title = "Employee KPI Deviation from Target (85)"
else:
    df["Deviation"] = df["Variance"]
    x_col = "Deviation"
    categories = "Category" if "Budget" in scenario else "Region"
    title = "Variance from Target/Forecast"

# Color: green for positive, red for negative
colors = ['#e74c3c' if v < 0 else '#27ae60' for v in df[x_col]]

fig = go.Figure()

fig.add_trace(go.Bar(
    y=df[categories],
    x=df[x_col],
    orientation='h',
    marker_color=colors,
    text=df[x_col],
    textposition='outside',
    hoverinfo='text',
    hovertext=df[categories] + ": " + df[x_col].astype(str)
))

fig.add_vline(x=0, line=dict(color="black", width=2))

fig.update_layout(
    title=f"Diverging Bar Chart – {title}<br><sub>Positive (green) = above target | Negative (red) = below target</sub>",
    xaxis_title="Deviation",
    yaxis_title="",
    height=600,
    showlegend=False,
    plot_bgcolor="white"
)
fig.show()
""", language="python")

    def create_back_to_back_bar(self, df: pd.DataFrame, scenario: str) -> go.Figure:
        fig = go.Figure()
        title = scenario

        if "Budget" in scenario or "Sales" in scenario or "Regional" in scenario:
            actual_col = "Actual"
            plan_col = "Budget" if "Budget" in df.columns else "Target" if "Target" in df.columns else "Forecast"
            category_col = df.columns[0]

            # Negative bars for plan (left)
            fig.add_trace(go.Bar(
                y=df[category_col],
                x=-df[plan_col],
                orientation='h',
                name=plan_col,
                marker_color='#3498db',
                text=df[plan_col],
                textposition='inside',
                hoverinfo='text'
            ))

            # Positive bars for actual (right)
            fig.add_trace(go.Bar(
                y=df[category_col],
                x=df[actual_col],
                orientation='h',
                name="Actual",
                marker_color='#e67e22',
                text=df[actual_col],
                textposition='inside',
                hoverinfo='text'
            ))

            title = f"{scenario} – Actual vs. {plan_col}"

        fig.update_layout(
            title=f"Back-to-Back Bar Chart – {title}<br><sub>Blue = Plan/Target | Orange = Actual</sub>",
            barmode='relative',
            xaxis_title="Amount",
            yaxis_title="",
            height=600,
            showlegend=True,
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
fig = go.Figure()
title = scenario

if "Budget" in scenario or "Sales" in scenario or "Regional" in scenario:
    actual_col = "Actual"
    plan_col = "Budget" if "Budget" in df.columns else "Target" if "Target" in df.columns else "Forecast"
    category_col = df.columns[0]

    # Negative bars for plan (left)
    fig.add_trace(go.Bar(
        y=df[category_col],
        x=-df[plan_col],
        orientation='h',
        name=plan_col,
        marker_color='#3498db',
        text=df[plan_col],
        textposition='inside',
        hoverinfo='text'
    ))

    # Positive bars for actual (right)
    fig.add_trace(go.Bar(
        y=df[category_col],
        x=df[actual_col],
        orientation='h',
        name="Actual",
        marker_color='#e67e22',
        text=df[actual_col],
        textposition='inside',
        hoverinfo='text'
    ))

    title = f"{scenario} – Actual vs. {plan_col}"

fig.update_layout(
    title=f"Back-to-Back Bar Chart – {title}<br><sub>Blue = Plan/Target | Orange = Actual</sub>",
    barmode='relative',
    xaxis_title="Amount",
    yaxis_title="",
    height=600,
    showlegend=True,
    plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom",
                y=1.02, xanchor="right", x=1)
)
fig.show()        
""", language="python")

    def render_chart(self, chart_type: str, scenario: str):
        df = self.get_sample_data(scenario)
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Diverging Bar Chart":
            self.create_diverging_bar(df, scenario)
            st.info(
                "**Insight**: Immediate visual of which items exceed (right/green) or fall short (left/red) of the baseline.")
        else:
            self.create_back_to_back_bar(df, scenario)
            st.info(
                "**Insight**: Side-by-side comparison makes variance magnitude and direction easy to assess.")

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Budget Analysis": "Actual spending vs. budgeted amounts",
            "Performance Revie": "Results compared to targets",
            "Survey Results": "Net promoter scores (positive minus negative)",
            "Financial Reporting": "Revenue variance from forecast",
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
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
        self.render_examples()
        self.render_key_characteristics()
