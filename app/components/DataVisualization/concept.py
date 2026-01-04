import streamlit as st
import plotly.graph_objects as go
import numpy as np


class Concept:

    def __init__(self):
        st.set_page_config(page_title="Concept Visualizations", layout="wide")
        self.title = "🔷 Concept Visualizations Dashboard"
        self.chart_types = [
            "Fishbone Diagram",
            "Pyramid Diagram",
            "Step-by-Step Diagram"
        ]
        self.scenarios = [
            "Customer Complaint Analysis",
            "Business Strategy Hierarchy",
            "Product Development Process",
            "Employee Onboarding Journey",
            "Sales Methodology"
        ]

    def render_header(self):
        st.markdown(f"# {self.title}")
        st.markdown("""
        ### Purpose
        Concept charts illustrate **ideas, processes, and theoretical frameworks**. 
        They help communicate abstract concepts and facilitate understanding of complex systems.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **When to Use:**
            - Educational materials and training
            - Process documentation
            - Problem-solving frameworks
            - Strategic planning
            - Communication of complex ideas
            """)

        with col2:
            st.markdown("""
            **Key Characteristics:**
            - Clarifies abstract concepts
            - Shows logical relationships
            - Supports teaching and learning
            - Facilitates communication
            """)

        st.markdown("""
        **Supported Diagrams:**
        - **Fishbone Diagram** – Root cause analysis (Ishikawa)
        - **Pyramid Diagram** – Hierarchical importance or foundation
        - **Step-by-Step Diagram** – Sequential process flow
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Diagram Settings")
        col1, col2 = st.columns([2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Diagram Type", self.chart_types, key="chart_type")

        with col2:
            scenario = st.selectbox("Scenario", self.scenarios, key="scenario")

        return chart_type, scenario

    def create_fishbone(self, scenario: str) -> go.Figure:
        if "Customer Complaint" in scenario:
            problem = "High Customer Churn Rate"
            categories = ["People", "Process", "Product",
                          "Policy", "Platform", "Promotion"]
            causes = {
                "People": ["Poor training", "High turnover", "Low motivation"],
                "Process": ["Long wait times", "Complex procedures", "No follow-up"],
                "Product": ["Bugs in app", "Missing features", "Slow performance"],
                "Policy": ["Strict cancellation", "Hidden fees", "No refunds"],
                "Platform": ["Outdated UI", "Mobile issues", "Downtime"],
                "Promotion": ["Misleading ads", "Overpromising", "Poor targeting"]
            }
            colors = ["#3498db", "#e74c3c", "#27ae60",
                      "#f39c12", "#9b59b6", "#e67e22"]

        else:
            problem = "Main Problem"
            categories = ["Category 1", "Category 2",
                          "Category 3", "Category 4"]
            causes = {cat: ["Cause A", "Cause B", "Cause C"]
                      for cat in categories}
            colors = ["#34495e"] * 4

        fig = go.Figure()

        # Main horizontal line (spine)
        fig.add_trace(go.Scatter(x=[-5, 5], y=[0, 0], mode='lines',
                      line=dict(color="black", width=4), hoverinfo='none'))

        # Problem box
        fig.add_trace(go.Scatter(x=[5.5], y=[0], mode='text', text=[
                      f"<b>{problem}</b>"], textposition="middle right"))

        angle_step = 60
        for i, cat in enumerate(categories):
            angle = 180 - \
                (i + 1) * angle_step if i >= len(categories)//2 else (i + 1) * angle_step
            rad = np.deg2rad(angle)

            # Category line
            x_end = 3 * np.cos(rad)
            y_end = 3 * np.sin(rad)
            fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end], mode='lines', line=dict(
                color=colors[i], width=3), hoverinfo='none'))

            # Category label
            fig.add_annotation(
                x=x_end*1.1, y=y_end*1.1, text=f"<b>{cat}</b>", showarrow=False, font=dict(size=14, color=colors[i]))

            # Causes
            sub_causes = causes[cat]
            for j, cause in enumerate(sub_causes):
                sub_angle = angle + (j - len(sub_causes)/2 + 0.5) * 15
                sub_rad = np.deg2rad(sub_angle)
                x_sub = x_end + 1.5 * np.cos(sub_rad)
                y_sub = y_end + 1.5 * np.sin(sub_rad)

                fig.add_trace(go.Scatter(x=[x_end, x_sub], y=[y_end, y_sub], mode='lines', line=dict(
                    color=colors[i], width=2), hoverinfo='none'))

                fig.add_annotation(x=x_sub, y=y_sub, text=cause,
                                   showarrow=False, font=dict(size=11))

        fig.update_layout(
            title=f"Fishbone Diagram – {problem}<br><sub>Root cause analysis categories and contributing factors</sub>",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, range=[-8, 8]),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, scaleanchor="x", scaleratio=1),
            height=700,
            plot_bgcolor="white"
        )
        return fig

    def create_pyramid(self, scenario: str) -> go.Figure:
        if "Business Strategy" in scenario:
            levels = [
                "Vision & Mission",
                "Strategic Objectives",
                "Key Initiatives",
                "Operational Plans",
                "Daily Activities"
            ]
            values = [10, 20, 30, 40, 50]
            colors = ["#2c3e50", "#34495e", "#7f8c8d", "#95a5a6", "#bdc3c7"]
            title = "Business Strategy Pyramid"

        elif "Product Development" in scenario:
            levels = [
                "Customer Needs",
                "Product Vision",
                "Core Features",
                "Supporting Features",
                "Technical Implementation"
            ]
            values = [8, 16, 28, 40, 52]
            colors = ["#27ae60", "#2ecc71", "#3498db", "#2980b9", "#2c3e50"]
            title = "Product Development Hierarchy"

        else:
            levels = ["Top Level", "Level 2", "Level 3", "Level 4", "Base"]
            values = [10, 20, 30, 40, 50]
            colors = ["#34495e"] * 5
            title = "Concept Pyramid"

        fig = go.Figure(go.Funnel(
            y=levels,
            x=values,
            textinfo="value+text",
            text=levels,
            marker=dict(color=colors),
            connector=dict(line=dict(color="black", width=2))
        ))

        fig.update_layout(
            title=f"{title}<br><sub>Hierarchical foundation – bottom supports top</sub>",
            height=700
        )
        return fig

    def create_step_by_step(self, scenario: str) -> go.Figure:
        if "Employee Onboarding" in scenario:
            steps = ["1. Pre-boarding", "2. Welcome Day", "3. Training Week",
                     "4. Team Integration", "5. 30-Day Review", "6. Full Productivity"]
            descriptions = [
                "Paperwork & setup",
                "Orientation & culture",
                "Skills & tools training",
                "Meet team & projects",
                "Feedback session",
                "Ongoing support"
            ]
            colors = ["#3498db", "#2980b9", "#27ae60",
                      "#2ecc71", "#f39c12", "#e67e22"]
            title = "Employee Onboarding Journey"

        elif "Sales Methodology" in scenario:
            steps = ["1. Prospecting", "2. Discovery", "3. Solution Demo",
                     "4. Proposal", "5. Negotiation", "6. Close"]
            descriptions = ["Find leads", "Understand needs", "Show value",
                            "Present offer", "Handle objections", "Win deal"]
            colors = ["#9b59b6", "#8e44ad", "#3498db",
                      "#2980b9", "#27ae60", "#2ecc71"]
            title = "Sales Process"

        else:
            steps = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
            descriptions = ["Description A", "Description B",
                            "Description C", "Description D", "Description E"]
            colors = ["#34495e"] * 5
            title = "Step-by-Step Process"

        x_pos = np.arange(len(steps))

        fig = go.Figure()

        for i, (step, desc, color) in enumerate(zip(steps, descriptions, colors)):
            fig.add_trace(go.Scatter(
                x=[x_pos[i]], y=[0],
                mode='markers+text',
                marker=dict(size=80, color=color),
                text=[step],
                textposition="middle center",
                textfont=dict(size=14, color="white"),
                hoverinfo="text",
                hovertext=f"<b>{step}</b><br>{desc}"
            ))

            if i < len(steps) - 1:
                fig.add_trace(go.Scatter(
                    x=[x_pos[i], x_pos[i+1]], y=[0, 0],
                    mode='lines',
                    line=dict(color="#7f8c8d", width=6),
                    hoverinfo='none'
                ))
                fig.add_shape(
                    type="path",
                    path=f"M {x_pos[i]+0.5} 0 L {x_pos[i]+0.8} 0.3 L {x_pos[i]+0.8} -0.3 Z",
                    line=dict(color="#7f8c8d", width=2),
                    fillcolor="#7f8c8d"
                )

        fig.update_layout(
            title=f"{title}<br><sub>Sequential process flow with key activities</sub>",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, range=[-1, 1]),
            height=400,
            plot_bgcolor="white"
        )
        return fig

    def render_chart(self, chart_type: str, scenario: str):
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Fishbone Diagram":
            fig = self.create_fishbone(scenario)
        elif chart_type == "Pyramid Diagram":
            fig = self.create_pyramid(scenario)
        else:
            fig = self.create_step_by_step(scenario)

        st.plotly_chart(fig, width='stretch')

    def output(self):
        self.render_header()
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
