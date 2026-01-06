import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd


class Proportional:

    def __init__(self):
        self.title = "🔶 Proportional Visualizations Dashboard"
        self.chart_types = [
            "Venn Diagram",
            "Rose Chart (Nightingale)",
            "Marimekko Chart",
            "Bubble Cloud"
        ]
        self.scenarios = [
            "Market Segment Overlap",
            "Sales by Month and Region",
            "Budget Allocation by Department",
            "Company Portfolio Analysis",
            "Product Feature Popularity"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Proportional charts emphasize **relative sizes and relationships** between values, 
        making size comparisons intuitive and immediate.
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

    def create_venn(self, scenario: str) -> go.Figure:
        # Using approximate proportional Venn with 3 circles
        if "Market Segment" in scenario:
            labels = ['Online', 'Retail', 'Wholesale']
            only1 = 120
            only2 = 80
            only3 = 60
            intersect12 = 45
            intersect13 = 30
            intersect23 = 25
            all3 = 15
            title = "Customer Segment Overlap"
        else:
            labels = ['Set A', 'Set B', 'Set C']
            only1, only2, only3 = 100, 80, 70
            intersect12, intersect13, intersect23 = 40, 30, 20
            all3 = 10
            title = "Proportional Set Overlap"

        fig = go.Figure()

        # Simplified proportional areas using scatter with sized markers (approximation)
        positions = [(0.3, 0.5), (0.7, 0.5), (0.5, 0.3)]
        sizes = [only1, only2, only3]
        colors = ['#3498db', '#e74c3c', '#27ae60']

        for i, (pos, size, color, label) in enumerate(zip(positions, sizes, colors, labels)):
            fig.add_trace(go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode='markers+text',
                marker=dict(size=np.sqrt(size)*10, color=color,
                            opacity=0.4, line=dict(width=2)),
                text=[label],
                textposition="middle center",
                textfont=dict(size=14, color="white")
            ))

        # Overlap annotations (approximate)
        fig.add_annotation(
            x=0.5, y=0.5, text=f"Overlap<br>{all3}", showarrow=False, font=dict(size=12))

        fig.update_layout(
            title=f"Venn Diagram – {title}<br><sub>Circle size proportional to unique segment count</sub>",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, range=[0, 1]),
            yaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, range=[0, 1]),
            height=600,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

if "Market Segment" in scenario:
    labels = ['Online', 'Retail', 'Wholesale']
    only1 = 120
    only2 = 80
    only3 = 60
    intersect12 = 45
    intersect13 = 30
    intersect23 = 25
    all3 = 15
    title = "Customer Segment Overlap"
else:
    labels = ['Set A', 'Set B', 'Set C']
    only1, only2, only3 = 100, 80, 70
    intersect12, intersect13, intersect23 = 40, 30, 20
    all3 = 10
    title = "Proportional Set Overlap"

fig = go.Figure()

# Simplified proportional areas using scatter with sized markers (approximation)
positions = [(0.3, 0.5), (0.7, 0.5), (0.5, 0.3)]
sizes = [only1, only2, only3]
colors = ['#3498db', '#e74c3c', '#27ae60']

for i, (pos, size, color, label) in enumerate(zip(positions, sizes, colors, labels)):
    fig.add_trace(go.Scatter(
        x=[pos[0]], y=[pos[1]],
        mode='markers+text',
        marker=dict(size=np.sqrt(size)*10, color=color,
                    opacity=0.4, line=dict(width=2)),
        text=[label],
        textposition="middle center",
        textfont=dict(size=14, color="white")
    ))

# Overlap annotations (approximate)
fig.add_annotation(
    x=0.5, y=0.5, text=f"Overlap<br>{all3}", showarrow=False, font=dict(size=12))

fig.update_layout(
    title=f"Venn Diagram – {title}<br><sub>Circle size proportional to unique segment count</sub>",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False,
                showticklabels=False, range=[0, 1]),
    yaxis=dict(showgrid=False, zeroline=False,
                showticklabels=False, range=[0, 1]),
    height=600,
    plot_bgcolor="white"
)
fig.show()
""", language="python")

    def create_rose_chart(self, scenario: str) -> go.Figure:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        if "Sales by Month" in scenario:
            region1 = [120, 135, 150, 160, 170,
                       185, 200, 195, 180, 165, 145, 130]
            region2 = [100, 110, 120, 130, 140,
                       150, 160, 155, 145, 130, 115, 105]
            title = "Monthly Sales by Region"
        else:
            values = np.random.randint(80, 200, 12)
            region1 = values.tolist()
            region2 = (values * 0.8).astype(int).tolist()
            title = "Cyclic Proportional Comparison"

        theta = months
        r1 = region1
        r2 = region2

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(r=r1, theta=theta, fill='toself',
                      name='Region A', fillcolor='rgba(52,152,219,0.6)'))
        fig.add_trace(go.Scatterpolar(r=r2, theta=theta, fill='toself',
                      name='Region B', fillcolor='rgba(231,76,60,0.6)'))

        fig.update_layout(
            title=f"Rose Chart – {title}<br><sub>Area proportional to value (Nightingale style)</sub>",
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

if "Sales by Month" in scenario:
    region1 = [120, 135, 150, 160, 170,
                185, 200, 195, 180, 165, 145, 130]
    region2 = [100, 110, 120, 130, 140,
                150, 160, 155, 145, 130, 115, 105]
    title = "Monthly Sales by Region"
else:
    values = np.random.randint(80, 200, 12)
    region1 = values.tolist()
    region2 = (values * 0.8).astype(int).tolist()
    title = "Cyclic Proportional Comparison"

theta = months
r1 = region1
r2 = region2

fig = go.Figure()

fig.add_trace(go.Scatterpolar(r=r1, theta=theta, fill='toself',
                name='Region A', fillcolor='rgba(52,152,219,0.6)'))
fig.add_trace(go.Scatterpolar(r=r2, theta=theta, fill='toself',
                name='Region B', fillcolor='rgba(231,76,60,0.6)'))

fig.update_layout(
    title=f"Rose Chart – {title}<br><sub>Area proportional to value (Nightingale style)</sub>",
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=True,
    height=600
)
fig.show()
""", language="python")

    def create_marimekko(self, scenario: str) -> go.Figure:
        if "Budget Allocation" in scenario:
            departments = ["Engineering", "Marketing",
                           "Sales", "Operations", "HR", "R&D"]
            total_budget = [400, 250, 300, 150, 100, 200]
            sub_categories = ["Salary", "Tools", "Travel", "Training"]
            # Percentage breakdown per department
            data = np.array([
                [60, 20, 15, 5],   # Engineering
                [50, 30, 10, 10],  # Marketing
                [70, 10, 15, 5],   # Sales
                [55, 25, 15, 5],   # Operations
                [65, 15, 10, 10],  # HR
                [45, 35, 10, 10]   # R&D
            ])
            title = "Budget Allocation by Department & Category"
        else:
            departments = ["Cat A", "Cat B", "Cat C", "Cat D"]
            total_budget = [300, 200, 250, 150]
            sub_categories = ["Sub1", "Sub2", "Sub3"]
            data = np.random.dirichlet(np.ones(3), size=len(departments)) * 100
            title = "Marimekko Proportional Share"

        # Calculate widths
        total = sum(total_budget)
        widths = [x / total for x in total_budget]

        fig = go.Figure()

        colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

        bottom = [0] * len(departments)
        for i, cat in enumerate(sub_categories):
            heights = data[:, i] / 100
            fig.add_trace(go.Bar(
                name=cat,
                y=departments,
                x=heights,
                offset=0,
                width=widths,
                orientation='h',
                marker_color=colors[i],
                text=np.round(data[:, i], 1),
                textposition='inside'
            ))
            bottom += heights

        fig.update_layout(
            title=f"Marimekko Chart – {title}<br><sub>Width = total share | Height = subcategory proportion</sub>",
            barmode='stack',
            height=600,
            xaxis_title="Proportion of Total",
            yaxis_title="Department"
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

if "Budget Allocation" in scenario:
    departments = ["Engineering", "Marketing",
                    "Sales", "Operations", "HR", "R&D"]
    total_budget = [400, 250, 300, 150, 100, 200]
    sub_categories = ["Salary", "Tools", "Travel", "Training"]
    # Percentage breakdown per department
    data = np.array([
        [60, 20, 15, 5],   # Engineering
        [50, 30, 10, 10],  # Marketing
        [70, 10, 15, 5],   # Sales
        [55, 25, 15, 5],   # Operations
        [65, 15, 10, 10],  # HR
        [45, 35, 10, 10]   # R&D
    ])
    title = "Budget Allocation by Department & Category"
else:
    departments = ["Cat A", "Cat B", "Cat C", "Cat D"]
    total_budget = [300, 200, 250, 150]
    sub_categories = ["Sub1", "Sub2", "Sub3"]
    data = np.random.dirichlet(np.ones(3), size=len(departments)) * 100
    title = "Marimekko Proportional Share"

# Calculate widths
total = sum(total_budget)
widths = [x / total for x in total_budget]

fig = go.Figure()

colors = ["#3498db", "#e74c3c", "#27ae60", "#f39c12"]

bottom = [0] * len(departments)
for i, cat in enumerate(sub_categories):
    heights = data[:, i] / 100
    fig.add_trace(go.Bar(
        name=cat,
        y=departments,
        x=heights,
        offset=0,
        width=widths,
        orientation='h',
        marker_color=colors[i],
        text=np.round(data[:, i], 1),
        textposition='inside'
    ))
    bottom += heights

fig.update_layout(
    title=f"Marimekko Chart – {title}<br><sub>Width = total share | Height = subcategory proportion</sub>",
    barmode='stack',
    height=600,
    xaxis_title="Proportion of Total",
    yaxis_title="Department"
)
fig.show()
""", language="python")

    def create_bubble_cloud(self, scenario: str) -> go.Figure:
        if "Company Portfolio" in scenario:
            companies = ["Alpha", "Beta", "Gamma", "Delta",
                         "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
            revenue = [450, 320, 280, 220, 180, 150, 120, 90, 70, 50]
            profit_margin = [18, 12, 22, 8, 25, 15, 10, 30, 14, 20]
            growth = [15, 8, 20, 5, 28, 12, 6, 35, 10, 18]
            title = "Company Portfolio – Revenue vs Growth vs Margin"
        else:
            companies = [f"Item {i}" for i in range(1, 11)]
            revenue = np.random.randint(50, 500, 10)
            profit_margin = np.random.uniform(5, 30, 10)
            growth = np.random.uniform(5, 40, 10)
            title = "Proportional Bubble Cloud"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=growth,
            y=profit_margin,
            mode='markers+text',
            marker=dict(
                size=np.sqrt(revenue)*2,
                color=revenue,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Revenue")
            ),
            text=companies,
            textposition="top center",
            hoverinfo='text',
            hovertext=[f"{c}<br>Revenue: {r}<br>Growth: {g}%<br>Margin: {m}%" for c, r, g, m in zip(
                companies, revenue, growth, profit_margin)]
        ))

        fig.update_layout(
            title=f"Bubble Cloud – {title}<br><sub>Bubble size = Revenue</sub>",
            xaxis_title="Growth Rate (%)",
            yaxis_title="Profit Margin (%)",
            height=600,
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import numpy as np

if "Company Portfolio" in scenario:
    companies = ["Alpha", "Beta", "Gamma", "Delta",
                    "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
    revenue = [450, 320, 280, 220, 180, 150, 120, 90, 70, 50]
    profit_margin = [18, 12, 22, 8, 25, 15, 10, 30, 14, 20]
    growth = [15, 8, 20, 5, 28, 12, 6, 35, 10, 18]
    title = "Company Portfolio – Revenue vs Growth vs Margin"
else:
    companies = [f"Item {i}" for i in range(1, 11)]
    revenue = np.random.randint(50, 500, 10)
    profit_margin = np.random.uniform(5, 30, 10)
    growth = np.random.uniform(5, 40, 10)
    title = "Proportional Bubble Cloud"

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=growth,
    y=profit_margin,
    mode='markers+text',
    marker=dict(
        size=np.sqrt(revenue)*2,
        color=revenue,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Revenue")
    ),
    text=companies,
    textposition="top center",
    hoverinfo='text',
    hovertext=[f"{c}<br>Revenue: {r}<br>Growth: {g}%<br>Margin: {m}%" for c, r, g, m in zip(
        companies, revenue, growth, profit_margin)]
))

fig.update_layout(
    title=f"Bubble Cloud – {title}<br><sub>Bubble size = Revenue</sub>",
    xaxis_title="Growth Rate (%)",
    yaxis_title="Profit Margin (%)",
    height=600,
    showlegend=False,
    plot_bgcolor="white"
)
fig.show()
""", language="python")

    def render_chart(self, chart_type: str, scenario: str):
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Venn Diagram":
            self.create_venn(scenario)
        elif chart_type == "Rose Chart (Nightingale)":
            self.create_rose_chart(scenario)
        elif chart_type == "Marimekko Chart":
            self.create_marimekko(scenario)
        else:
            self.create_bubble_cloud(scenario)

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Market Analysis": "Relative size of market segments",
            "Budget Allocation": "Department budgets sized proportionally",
            "Population Studies": "City sizes compared visually",
            "Risk Assessment": "Probability and impact sized proportionally",
            "Resource Allocation": "Team sizes across projects"
        }

        for example, description in examples.items():
            with st.expander(f"💭 {example}"):
                st.write(description)

    def render_key_characteristics(self):
        st.markdown("### 📊 Understanding Proportional Analysis")

        st.markdown("""
        Proportional analysis visualizes **values through relative size**.
        It enables instant understanding of magnitude and importance.
        """)

        st.markdown("#### 📏 Size Represents Value")
        st.markdown("""
        Elements such as bars, slices, or bubbles are scaled to their value.
        Larger elements indicate higher values, smaller elements indicate lower values.
        """)

        st.markdown("#### ⚡ Immediate Visual Comparison")
        st.markdown("""
        Users can compare multiple items instantly:
        - Larger vs smaller elements  
        - Relative contributions to totals  
        - Visual rankings without reading numbers  
        """)

        st.markdown("#### 🧠 Intuitive Understanding")
        st.markdown("""
        Size encoding leverages human perception:
        - People interpret relative size faster than exact numbers  
        - Makes charts accessible and easy to digest  
        """)

        st.markdown("#### 🎯 Shows Relative Importance")
        st.markdown("""
        Highlights which items contribute most to a total:
        - Market share  
        - Budget allocation  
        - Task prioritization  
        """)

        st.divider()

        st.markdown("#### 🎯 Why Proportional Analysis Matters")
        st.markdown("""
        Proportional visualization transforms complex numbers into intuitive insights:
        - Supports quick decision-making  
        - Simplifies dashboards  
        - Enhances communication of priorities  
        """)

    def output(self):
        self.render_header()
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
        self.render_examples()
        self.render_key_characteristics()
