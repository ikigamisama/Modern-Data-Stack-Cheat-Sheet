import streamlit as st
import numpy as np
import plotly.graph_objects as go


class Qualitative:
    def __init__(self):
        st.set_page_config(
            page_title="Qualitative Visualizations", layout="wide")
        self.title = "🎨 Qualitative Visualizations Dashboard"
        self.chart_types = [
            "Mind Map",
            "SWOT Analysis",
            "Customer Journey Map"
        ]
        self.mind_scenarios = [
            "Product Launch Ideas", "Career Development", "Content Strategy", "Personal Goals"
        ]
        self.swot_scenarios = [
            "New Startup", "E-Commerce Business", "Personal Brand", "Team Project"
        ]
        self.journey_scenarios = [
            "Online Shopping", "Booking a Flight", "Restaurant Visit", "Software Onboarding"
        ]

    def render_header(self):
        st.markdown(f"# {self.title}")
        st.markdown("""
        ### Purpose
        Qualitative charts represent **non-numerical concepts, strategies, and ideas**. 
        They help organize thoughts, plan initiatives, and communicate abstract concepts visually.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **When to Use:**
            - Strategic planning sessions
            - Brainstorming and ideation
            - User experience mapping
            - Business model development
            - Problem-solving workshops
            """)

        with col2:
            st.markdown("""
            **Key Characteristics:**
            - Represents concepts, not numbers
            - Facilitates thinking and planning
            - Organizes complex ideas
            - Supports collaborative work
            """)

        st.markdown("""
        **Supported Visualizations:**
        - **Mind Map** – Radial idea generation and connections
        - **SWOT Analysis** – Strengths, Weaknesses, Opportunities, Threats
        - **Customer Journey Map** – User experience stages and touchpoints
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2 = st.columns([2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Visualization Type", self.chart_types, key="chart_type")

        with col2:
            if chart_type == "Mind Map":
                scenario = st.selectbox(
                    "Scenario", self.mind_scenarios, key="scenario")
            elif chart_type == "SWOT Analysis":
                scenario = st.selectbox(
                    "Scenario", self.swot_scenarios, key="scenario")
            else:
                scenario = st.selectbox(
                    "Scenario", self.journey_scenarios, key="scenario")

        return chart_type, scenario

    def create_mind_map(self, scenario: str) -> go.Figure:
        if scenario == "Product Launch Ideas":
            center = "New AI App Launch"
            main_branches = ["Target Audience",
                             "Features", "Marketing", "Monetization"]
            sub_branches = {
                "Target Audience": ["Developers", "Small Businesses", "Students"],
                "Features": ["Chat Assistant", "Image Generation", "Code Helper", "API Access"],
                "Marketing": ["Social Media", "Content Marketing", "Partnerships", "Beta Waitlist"],
                "Monetization": ["Freemium", "Subscription Tiers", "Enterprise Plan"]
            }
            colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

        else:  # Generic fallback
            center = "Main Idea"
            main_branches = ["Branch 1", "Branch 2", "Branch 3"]
            sub_branches = {
                b: [f"Sub {i}" for i in range(1, 4)] for b in main_branches}
            colors = ["#a8e6cf", "#ffd3b6", "#ffaaa5"]

        # Radial layout approximation
        fig = go.Figure()

        # Center node
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            text=[center],
            textposition="middle center",
            marker=dict(size=60, color="#34495e"),
            textfont=dict(size=16, color="white"),
            hoverinfo="text"
        ))

        angle_step = 360 / len(main_branches)
        for i, branch in enumerate(main_branches):
            angle = i * angle_step
            rad = angle * 3.14159 / 180
            x_main = 2 * np.cos(rad)
            y_main = 2 * np.sin(rad)

            # Main branch line
            fig.add_trace(go.Scatter(x=[0, x_main], y=[0, y_main], mode='lines', line=dict(
                color="#7f8c8d", width=3), hoverinfo='none'))

            # Main branch node
            fig.add_trace(go.Scatter(
                x=[x_main], y=[y_main],
                mode='markers+text',
                text=[branch],
                marker=dict(size=45, color=colors[i]),
                textfont=dict(size=14),
                hoverinfo="text"
            ))

            # Sub-branches
            subs = sub_branches.get(branch, [])
            sub_step = 1.2 if len(subs) > 0 else 0
            for j, sub in enumerate(subs):
                sub_angle = angle + (j - (len(subs)-1)/2) * 20
                sub_rad = sub_angle * 3.14159 / 180
                x_sub = (3.5 + j*0.3) * np.cos(sub_rad)
                y_sub = (3.5 + j*0.3) * np.sin(sub_rad)

                fig.add_trace(go.Scatter(x=[x_main, x_sub], y=[y_main, y_sub], mode='lines', line=dict(
                    color="#95a5a6", width=2), hoverinfo='none'))

                fig.add_trace(go.Scatter(
                    x=[x_sub], y=[y_sub],
                    mode='markers+text',
                    text=[sub],
                    marker=dict(size=35, color=colors[i], opacity=0.8),
                    textfont=dict(size=12),
                    hoverinfo="text"
                ))

        fig.update_layout(
            title=f"{scenario} – Mind Map (Radial Layout)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            plot_bgcolor="white"
        )
        return fig

    def create_swot_analysis(self, scenario: str) -> go.Figure:
        if scenario == "New Startup":
            strengths = [
                "Innovative AI tech<br>Strong founding team<br>First-mover advantage"]
            weaknesses = [
                "Limited funding<br>No brand recognition<br>Small team size"]
            opportunities = [
                "Growing AI market<br>Remote work trend<br>Partnership potential"]
            threats = [
                "Big tech competitors<br>Regulatory changes<br>Rapid tech evolution"]

        else:
            strengths = ["Advantage 1<br>Advantage 2<br>Advantage 3"]
            weaknesses = ["Challenge 1<br>Challenge 2"]
            opportunities = ["Opportunity 1<br>Opportunity 2<br>Opportunity 3"]
            threats = ["Risk 1<br>Risk 2"]

        fig = go.Figure()

        quadrants = [
            ("Strengths 💪", strengths, "#a8e6cf"),
            ("Weaknesses ⚠️", weaknesses, "#ffccc9"),
            ("Opportunities 🚀", opportunities, "#c2e0c2"),
            ("Threats ⚡", threats, "#ffd3b6")
        ]

        positions = [(0, 1), (1, 1), (0, 0), (1, 0)]

        for (title, items, color), (x, y) in zip(quadrants, positions):
            text = f"<b>{title}</b><br><br>" + "<br>".join(items)
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[text],
                textposition="middle center",
                marker=dict(size=300, color=color, opacity=0.9),
                textfont=dict(size=14, color='white'),
                hoverinfo="text"
            ))

        fig.update_layout(
            title=f"{scenario} – SWOT Analysis",
            showlegend=False,
            xaxis=dict(range=[-0.5, 1.5], showgrid=False,
                       zeroline=False, showticklabels=False),
            yaxis=dict(range=[-0.5, 1.5], showgrid=False,
                       zeroline=False, showticklabels=False),
            height=700,
            plot_bgcolor="white"
        )
        return fig

    def create_journey_map(self, scenario: str) -> go.Figure:
        if scenario == "Online Shopping":
            stages = ["Awareness", "Consideration",
                      "Purchase", "Delivery", "Post-Purchase"]
            touchpoints = ["Social Ad<br>Search Engine", "Product Page<br>Reviews<br>Comparison",
                           "Cart<br>Checkout<br>Payment", "Tracking<br>Email Updates", "Support<br>Review Request<br>Loyalty Program"]
            emotions = ["Curious", "Interested",
                        "Excited", "Anxious", "Relieved / Happy"]
            emotion_y = [0.2, 0.6, 0.8, 0.4, 0.7]

        else:
            stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
            touchpoints = [
                "Touchpoint A", "Touchpoint B<br>Touchpoint C", "Touchpoint D", "Touchpoint E"]
            emotions = ["Neutral", "Positive", "Frustrated", "Satisfied"]
            emotion_y = [0.3, 0.7, 0.2, 0.8]

        x_positions = [i for i in range(len(stages))]

        fig = go.Figure()

        # Emotion line
        fig.add_trace(go.Scatter(
            x=x_positions, y=emotion_y,
            mode='lines+markers',
            name='Emotional Journey',
            line=dict(color="#e74c3c", width=4),
            marker=dict(size=12)
        ))

        # Stage labels
        for i, stage in enumerate(stages):
            fig.add_annotation(
                x=i, y=1.1,
                text=f"<b>{stage}</b>",
                showarrow=False,
                font=dict(size=16, color="black")
            )

            fig.add_annotation(
                x=i, y=0.9,
                text=touchpoints[i],
                showarrow=False,
                font=dict(size=12, color="black"),
                align="center"
            )

        fig.update_layout(
            title=f"{scenario} – Customer Journey Map",
            showlegend=False,
            xaxis=dict(showgrid=False, tickvals=x_positions, ticktext=stages),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, range=[0, 1.2]),
            height=600,
            plot_bgcolor="white"
        )
        return fig

    def render_chart(self, chart_type: str, scenario: str):
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Mind Map":
            fig = self.create_mind_map(scenario)
        elif chart_type == "SWOT Analysis":
            fig = self.create_swot_analysis(scenario)
        else:
            fig = self.create_journey_map(scenario)

        st.plotly_chart(fig, width='stretch')

    def output(self):
        self.render_header()
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
