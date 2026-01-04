import streamlit as st
import pandas as pd
import numpy as np


class TextBased:
    def __init__(self):
        st.set_page_config(
            page_title="Text-Based Visualizations", layout="wide")
        self.title = "📋 Text-Based Visualizations Dashboard"
        self.chart_types = [
            "Table",
            "Highlight Table"
        ]
        self.scenarios = [
            "Financial Performance",
            "Product Specifications",
            "Sales Team Metrics",
            "Customer Support KPIs",
            "Employee Directory"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Text-based charts present data in **tabular or structured text formats**, offering precision and detailed comparison 
        when exact values matter more than visual patterns.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **When to Use:**
            - Presenting precise numerical values
            - Comparing many detailed metrics
            - Creating reference materials
            - Displaying specifications or parameters
            - When exact values are critical
            """)

        with col2:
            st.markdown("""
            **Key Characteristics:**
            - Emphasizes precision over pattern
            - Supports detailed comparison
            - Enables sorting and filtering
            - Preserves exact values
            """)

        st.markdown("""
        **Supported Visualizations:**
        - **Table** – Clean, structured data presentation with optional sorting
        - **Highlight Table** – Color-coded cells to emphasize high/low values or categories
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2 = st.columns([2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Visualization Type", self.chart_types, key="chart_type")

        with col2:
            scenario = st.selectbox("Scenario", self.scenarios, key="scenario")

        return chart_type, scenario

    def get_sample_data(self, scenario: str) -> pd.DataFrame:
        np.random.seed(42)

        if scenario == "Financial Performance":
            data = {
                "Quarter": ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"],
                "Revenue ($M)": [45.2, 52.8, 58.1, 64.3],
                "Net Profit ($M)": [8.1, 10.5, 12.3, 15.7],
                "Gross Margin (%)": [42.8, 44.1, 45.3, 47.2],
                "Operating Expenses ($M)": [28.4, 30.1, 31.8, 33.5],
                "YoY Growth (%)": [12.4, 16.8, 19.2, 23.5]
            }
            df = pd.DataFrame(data)

        elif scenario == "Product Specifications":
            data = {
                "Model": ["Pro X1", "Pro X2", "Elite S", "Standard M", "Lite"],
                "Screen Size": ["15.6\"", "16\"", "14\"", "15.6\"", "13.3\""],
                "Processor": ["Intel i9", "AMD Ryzen 9", "Intel i7", "Intel i5", "Intel i3"],
                "RAM (GB)": [32, 64, 32, 16, 8],
                "Storage (TB)": [2, 4, 1, 1, 0.5],
                "Battery Life (hrs)": [12, 14, 10, 9, 8],
                "Weight (kg)": [1.8, 2.1, 1.4, 2.0, 1.3],
                "Price ($)": [2499, 3499, 1899, 1299, 899]
            }
            df = pd.DataFrame(data)

        elif scenario == "Sales Team Metrics":
            names = ["Alice Johnson", "Bob Smith", "Carol Lee",
                     "David Park", "Emma Wilson", "Frank Chen"]
            data = {
                "Sales Rep": names,
                "Region": ["North", "South", "East", "West", "Central", "North"],
                "Deals Closed": np.random.randint(15, 45, 6),
                "Revenue Generated ($K)": np.random.randint(180, 650, 6),
                "Average Deal Size ($K)": [0] * 6,
                "Conversion Rate (%)": np.random.uniform(22, 38, 6).round(1),
                "Quota Achievement (%)": np.random.uniform(85, 135, 6).round(1)
            }
            df = pd.DataFrame(data)
            df["Average Deal Size ($K)"] = (
                df["Revenue Generated ($K)"] / df["Deals Closed"]).round(1)

        elif scenario == "Customer Support KPIs":
            data = {
                "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                "Tickets Resolved": np.random.randint(800, 1200, 6),
                "Avg Resolution Time (hrs)": np.random.uniform(2.1, 4.8, 6).round(1),
                "First Contact Resolution (%)": np.random.uniform(68, 85, 6).round(1),
                "Customer Satisfaction (CSAT)": np.random.uniform(8.2, 9.4, 6).round(2),
                "Escalations": np.random.randint(20, 60, 6)
            }
            df = pd.DataFrame(data)

        else:  # Employee Directory
            data = {
                "Name": ["Sarah Miller", "James Brown", "Lisa Davis", "Mike Taylor", "Anna White"],
                "Role": ["Product Manager", "Senior Developer", "UX Designer", "Data Analyst", "Marketing Lead"],
                "Department": ["Product", "Engineering", "Design", "Analytics", "Marketing"],
                "Email": ["sarah@company.com", "james@company.com", "lisa@company.com", "mike@company.com", "anna@company.com"],
                "Location": ["New York", "Remote", "San Francisco", "New York", "London"],
                "Years at Company": [5, 3, 4, 2, 6]
            }
            df = pd.DataFrame(data)

        return df

    def render_table(self, df: pd.DataFrame, scenario: str):
        st.markdown(f"### Table: {scenario}")
        st.dataframe(df, width='stretch', hide_index=True)
        st.info(
            "**Insight**: Precise values for reference, comparison, and detailed analysis.")

    def render_highlight_table(self, df: pd.DataFrame, scenario: str):
        st.markdown(f"### Highlight Table: {scenario}")

        styled_df = df.style

        # Apply conditional formatting based on scenario
        if "Financial" in scenario:
            styled_df = styled_df.format({
                "Revenue ($M)": "${:.1f}",
                "Net Profit ($M)": "${:.1f}",
                "Operating Expenses ($M)": "${:.1f}"
            }).background_gradient(subset=["Revenue ($M)", "Net Profit ($M)"], cmap="Greens") \
                .background_gradient(subset=["YoY Growth (%)"], cmap="Blues")

        elif "Product Specifications" in scenario:
            styled_df = styled_df.format({"Price ($)": "${:.0f}"}).background_gradient(subset=["RAM (GB)", "Storage (TB)", "Battery Life (hrs)"], cmap="YlGn") \
                .background_gradient(subset=["Price ($)"], cmap="Reds_r")

        elif "Sales Team Metrics" in scenario:
            styled_df = styled_df.format({
                "Revenue Generated ($K)": "${:.0f}",
                "Average Deal Size ($K)": "${:.1f}"
            }).background_gradient(subset=["Revenue Generated ($K)", "Quota Achievement (%)"], cmap="Greens") \
                .background_gradient(subset=["Conversion Rate (%)"], cmap="Blues")

        elif "Customer Support KPIs" in scenario:
            styled_df = styled_df.background_gradient(subset=["Tickets Resolved"], cmap="Blues") \
                .background_gradient(subset=["Avg Resolution Time (hrs)"], cmap="Reds_r") \
                .background_gradient(subset=["Customer Satisfaction (CSAT)"], cmap="Greens")

        else:  # Employee Directory
            styled_df = styled_df.background_gradient(
                subset=["Years at Company"], cmap="Purples")

        st.dataframe(styled_df, width='stretch', hide_index=True)
        st.info(
            "**Insight**: Color highlights draw attention to high/low values for quick pattern recognition.")

    def render_chart(self, chart_type: str, scenario: str):
        df = self.get_sample_data(scenario)

        if chart_type == "Table":
            self.render_table(df, scenario)
        else:
            self.render_highlight_table(df, scenario)

    def output(self):
        self.render_header()
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
