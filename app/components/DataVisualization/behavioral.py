import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np


class Behavioral:
    def __init__(self):
        st.set_page_config(
            page_title="Behavioral Visualizations", layout="wide")
        self.title = "🎯 Behavioral Visualizations Dashboard"
        self.chart_types = [
            "Funnel Chart",
            "Cohort Chart"
        ]
        self.funnel_scenarios = [
            "E-Commerce Purchase",
            "SaaS Onboarding",
            "Mobile App Registration",
            "Email Campaign",
            "Lead to Customer"
        ]
        self.cohort_scenarios = [
            "Subscription Service Retention",
            "SaaS Monthly Cohorts",
            "E-Commerce Customer Loyalty",
            "App User Engagement",
            "Online Course Completion"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Behavioral charts track **user actions, conversion patterns, and engagement metrics**. 
        They reveal how people interact with products, services, or systems over time.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **When to Use:**
            - Analyzing user conversion funnels
            - Tracking customer retention
            - Understanding user engagement
            - Identifying drop-off points
            - Measuring campaign effectiveness
            """)

        with col2:
            st.markdown("""
            **Key Characteristics:**
            - Shows user progression
            - Identifies friction points
            - Measures conversion efficiency
            - Tracks longitudinal behavior
            """)

        st.markdown("""
        **Supported Charts:**
        - **Funnel Chart** – Visualizes step-by-step user drop-off in a process
        - **Cohort Chart** – Heatmap showing retention or behavior over time by user groups
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2 = st.columns([2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            if chart_type == "Funnel Chart":
                scenario = st.selectbox(
                    "Scenario", self.funnel_scenarios, key="scenario")
            else:
                scenario = st.selectbox(
                    "Scenario", self.cohort_scenarios, key="scenario")

        return chart_type, scenario

    def create_funnel_chart(self, scenario: str) -> go.Figure:
        if scenario == "E-Commerce Purchase":
            stages = ["Visit Site", "View Product", "Add to Cart",
                      "Begin Checkout", "Complete Purchase"]
            values = [10000, 7200, 4200, 2500, 1800]
            conversion = [100, 72, 58, 60, 72]
            colors = ["#3498db", "#2980b9", "#1f618d", "#154360", "#1e3799"]
            title = "E-Commerce Purchase Funnel"

        elif scenario == "SaaS Onboarding":
            stages = ["Sign Up", "Verify Email",
                      "Connect Account", "Import Data", "First Report"]
            values = [5000, 4500, 3600, 2200, 1600]
            conversion = [100, 90, 80, 61, 73]
            colors = ["#27ae60", "#229954", "#1e8449", "#196f3d", "#148f77"]
            title = "SaaS Product Onboarding"

        elif scenario == "Mobile App Registration":
            stages = ["Install App", "Open App", "Create Profile",
                      "Grant Permissions", "Complete Tutorial"]
            values = [25000, 18000, 12000, 9000, 6500]
            conversion = [100, 72, 67, 75, 72]
            colors = ["#9b59b6", "#8e44ad", "#7d3c98", "#6c3483", "#5e3370"]
            title = "Mobile App Registration Flow"

        else:
            stages = ["Step 1", "Step 2", "Step 3", "Step 4", "Conversion"]
            values = [10000, 7000, 4500, 2800, 2000]
            conversion = [100, 70, 64, 62, 71]
            colors = ["#34495e"] * 5
            title = "Generic Conversion Funnel"

        fig = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textposition="inside",
            textinfo="value+percent initial+percent previous",
            marker={"color": colors},
            connector={"line": {"color": "gray", "width": 2}}
        ))

        fig.update_layout(
            title=f"{title}<br><sub>Drop-off analysis across user journey stages</sub>",
            height=600
        )
        return fig

    def create_cohort_chart(self, scenario: str) -> go.Figure:
        np.random.seed(42)
        # Define months and cohorts
        months = ["Month 0", "Month 1", "Month 2",
                  "Month 3", "Month 4", "Month 5", "Month 6"]
        cohorts = ["Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "May 2025"]

        # Base retention lists (variable length per cohort)
        base_retention = [
            [100, 82, 70, 62, 58, 55, 52],
            [100, 85, 73, 65, 60, 57],
            [100, 88, 75, 68, 63],
            [100, 83, 72, 66],
            [100, 86, 74]
        ]

        # -----------------------
        # Pad each row to match months length
        padded_retention = []
        for row in base_retention:
            row_list = list(row)  # ensure it's a list
            padded_row = row_list + [np.nan] * (len(months) - len(row_list))
            padded_retention.append(padded_row)

        retention = np.array(padded_retention)  # shape will be (5, 7)
        # -----------------------

        # Scenario-based adjustments
        if "Subscription" in scenario:
            title = "Subscription Service Cohort Retention"
        elif "SaaS" in scenario:
            # Add random variation only where retention exists
            noise = np.random.randint(-5, 8, retention.shape)
            retention = np.where(np.isnan(retention),
                                 np.nan, retention + noise)
            title = "SaaS Monthly Active User Cohorts"
        elif "E-Commerce" in scenario:
            # Lower retention typical for E-Commerce
            retention = np.where(np.isnan(retention), np.nan, retention * 0.9)
            title = "E-Commerce Repeat Purchase Cohorts"
        else:
            title = "User Retention by Cohort"

        # Clip retention values between 20 and 100, ignoring np.nan
        retention = np.where(np.isnan(retention), np.nan,
                             np.clip(retention, 20, 100))

        # Create the Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=retention,
            x=months,
            y=cohorts,
            colorscale="YlGnBu",
            text=np.where(np.isnan(retention), "", retention.astype(int)),
            texttemplate="%{text}%",
            textfont={"size": 12},
            hoverongaps=False
        ))

        # Update layout
        fig.update_layout(
            title=f"{title}<br><sub>Percentage of users retained over time by signup cohort</sub>",
            xaxis_title="Months Since Signup",
            yaxis_title="Signup Cohort",
            height=600
        )

        return fig

    def render_chart(self, chart_type: str, scenario: str):
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Funnel Chart":
            fig = self.create_funnel_chart(scenario)
        else:
            fig = self.create_cohort_chart(scenario)

        st.plotly_chart(fig, width='stretch')

        if chart_type == "Funnel Chart":
            st.info(
                "**Insight**: Shows where users drop off most — focus optimization on high-drop stages.")
        else:
            st.info(
                "**Insight**: Diagonal pattern shows retention trends. Stronger cohorts retain better over time.")

    def output(self):
        self.render_header()
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
