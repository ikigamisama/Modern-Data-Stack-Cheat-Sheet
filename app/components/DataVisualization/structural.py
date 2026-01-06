import streamlit as st
import plotly.graph_objects as go


class Structural:
    def __init__(self):
        self.title = "🏗️ Structural Visualizations Dashboard"
        self.chart_types = [
            "Organizational Chart",
            "Entity-Relationship (ER) Diagram"
        ]
        self.org_scenarios = [
            "Tech Startup", "Large Corporation", "Non-Profit Organization",
            "Government Department", "University Structure"
        ]
        self.er_scenarios = [
            "E-Commerce Platform", "Library Management System",
            "Hospital Database", "School Enrollment System", "Inventory Management"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Structural charts represent the **architecture, organization, and logical relationships** within systems. 
        They show how components are arranged and relate **hierarchically or functionally**.
        """, unsafe_allow_html=True)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            if chart_type == "Organizational Chart":
                scenario = st.selectbox(
                    "Scenario", self.org_scenarios, key="scenario")
            else:
                scenario = st.selectbox(
                    "Scenario", self.er_scenarios, key="scenario")

        with col3:
            detail_level = st.select_slider(
                "Detail Level",
                options=["Simple", "Medium", "Detailed"],
                value="Medium",
                key="detail"
            )

        return chart_type, scenario, detail_level

    def create_org_treemap(self, scenario: str, detail: str) -> go.Figure:
        # Define hierarchy data based on scenario and detail
        if scenario == "Tech Startup":
            labels = ["CEO", "CTO (Technology)",
                      "CPO (Product)", "CMO (Marketing)"]
            parents = ["", "CEO", "CEO", "CEO"]
            values = [20, 8, 6, 6]  # Relative sizes
            colors = ["#a8e6cf", "#ffd3b6", "#ffd3b6", "#ffd3b6"]

            if detail in ["Medium", "Detailed"]:
                labels += ["Engineering Lead", "Design Lead",
                           "Development Team (12)", "QA Team (4)"]
                parents += ["CTO (Technology)", "CPO (Product)",
                            "Engineering Lead", "Engineering Lead"]
                values += [6, 4, 4, 2]
                colors += ["#ffaaa5", "#ffaaa5", "#d5d6ea", "#d5d6ea"]

                if detail == "Detailed":
                    labels += ["Frontend Team", "Backend Team"]
                    parents += ["Development Team (12)",
                                "Development Team (12)"]
                    values += [2, 2]
                    colors += ["#d5d6ea", "#d5d6ea"]

        elif scenario == "Large Corporation":
            labels = ["CEO", "COO", "CFO", "CHRO"]
            parents = ["", "CEO", "CEO", "CEO"]
            values = [30, 12, 10, 8]
            colors = ["#a8e6cf", "#ffd3b6", "#ffd3b6", "#ffd3b6"]

            if detail in ["Medium", "Detailed"]:
                labels += ["VP Sales", "VP Marketing", "VP Engineering"]
                parents += ["COO", "COO", "COO"]
                values += [8, 7, 8]
                colors += ["#ffaaa5"] * 3

                if detail == "Detailed":
                    labels += ["Regional Sales Teams"]
                    parents += ["VP Sales"]
                    values += [6]
                    colors += ["#d5d6ea"]

        else:  # Generic fallback
            labels = ["CEO", "Dept Head 1", "Dept Head 2"]
            parents = ["", "CEO", "CEO"]
            values = [15, 7, 7]
            colors = ["#a8e6cf", "#ffd3b6", "#ffd3b6"]
            if detail != "Simple":
                labels += ["Team A", "Team B"]
                parents += ["Dept Head 1", "Dept Head 2"]
                values += [4, 4]
                colors += ["#d5d6ea", "#d5d6ea"]

        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker_colors=colors,
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><extra></extra>",
            root_color="#f0f0f0"
        ))

        fig.update_layout(
            title=f"{scenario} - Organizational Hierarchy (Interactive Treemap)",
            margin=dict(t=50, l=25, r=25, b=25),
            height=700
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### 🛠️ Sample Code")
        if scenario == "Tech Startup":
            st.code("""
import plotly.graph_objects as go                    

labels = ["CEO", "CTO (Technology)",
            "CPO (Product)", "CMO (Marketing)"]
parents = ["", "CEO", "CEO", "CEO"]
values = [20, 8, 6, 6]  # Relative sizes
colors = ["#a8e6cf", "#ffd3b6", "#ffd3b6", "#ffd3b6"]

if detail in ["Medium", "Detailed"]:
    labels += ["Engineering Lead", "Design Lead",
                "Development Team (12)", "QA Team (4)"]
    parents += ["CTO (Technology)", "CPO (Product)",
                "Engineering Lead", "Engineering Lead"]
    values += [6, 4, 4, 2]
    colors += ["#ffaaa5", "#ffaaa5", "#d5d6ea", "#d5d6ea"]

    if detail == "Detailed":
        labels += ["Frontend Team", "Backend Team"]
        parents += ["Development Team (12)",
                    "Development Team (12)"]
        values += [2, 2]
        colors += ["#d5d6ea", "#d5d6ea"]
                    
fig = go.Figure(go.Treemap(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    marker_colors=colors,
    textinfo="label+value",
    hovertemplate="<b>%{label}</b><extra></extra>",
    root_color="#f0f0f0"
))

fig.update_layout(
    title=f"{scenario} - Organizational Hierarchy (Interactive Treemap)",
    margin=dict(t=50, l=25, r=25, b=25),
    height=700
)
fig.show()

""", language="python")
        elif scenario == "Large Corporation":
            st.code("""import plotly.graph_objects as go     
                    
labels = ["CEO", "COO", "CFO", "CHRO"]
parents = ["", "CEO", "CEO", "CEO"]
values = [30, 12, 10, 8]
colors = ["#a8e6cf", "#ffd3b6", "#ffd3b6", "#ffd3b6"]

if detail in ["Medium", "Detailed"]:
    labels += ["VP Sales", "VP Marketing", "VP Engineering"]
    parents += ["COO", "COO", "COO"]
    values += [8, 7, 8]
    colors += ["#ffaaa5"] * 3

    if detail == "Detailed":
        labels += ["Regional Sales Teams"]
        parents += ["VP Sales"]
        values += [6]
        colors += ["#d5d6ea"]

fig = go.Figure(go.Treemap(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    marker_colors=colors,
    textinfo="label+value",
    hovertemplate="<b>%{label}</b><extra></extra>",
    root_color="#f0f0f0"
))

fig.update_layout(
    title=f"{scenario} - Organizational Hierarchy (Interactive Treemap)",
    margin=dict(t=50, l=25, r=25, b=25),
    height=700
)
fig.show()
""", language="python")
        else:
            st.code("""
labels = ["CEO", "Dept Head 1", "Dept Head 2"]
parents = ["", "CEO", "CEO"]
values = [15, 7, 7]
colors = ["#a8e6cf", "#ffd3b6", "#ffd3b6"]
if detail != "Simple":
    labels += ["Team A", "Team B"]
    parents += ["Dept Head 1", "Dept Head 2"]
    values += [4, 4]
    colors += ["#d5d6ea", "#d5d6ea"]
                    
fig = go.Figure(go.Treemap(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    marker_colors=colors,
    textinfo="label+value",
    hovertemplate="<b>%{label}</b><extra></extra>",
    root_color="#f0f0f0"
))

fig.update_layout(
    title=f"{scenario} - Organizational Hierarchy (Interactive Treemap)",
    margin=dict(t=50, l=25, r=25, b=25),
    height=700
)
fig.show()
""", language="python")

    def create_er_network(self, scenario: str, detail: str) -> go.Figure:
        # Manual layout for classic ER diagram feel (left to right)
        if scenario == "E-Commerce Platform":
            nodes = ["Customer", "Order", "OrderItem", "Product"]
            x = [0, 0.33, 0.66, 0.66]
            y = [0.8, 0.8, 0.5, 0.8]
            colors = ["#fad7a0"] * 4
            texts = [
                "Customer<br>customer_id (PK)<br>name<br>email<br>address<br>phone",
                "Order<br>order_id (PK)<br>customer_id (FK)<br>order_date<br>total_amount<br>status",
                "OrderItem<br>order_id (PK, FK)<br>product_id (PK, FK)<br>quantity<br>unit_price",
                "Product<br>product_id (PK)<br>name<br>description<br>price<br>stock_quantity"
            ]

            # Edges (x/y for lines)
            edge_x = [0, 0.33, None, 0.33, 0.66, None, 0.66, 0.66]
            edge_y = [0.8, 0.8, None, 0.5, 0.5, None, 0.8, 0.5]

            annotations = [
                dict(x=0.165, y=0.85, text="places<br>1 → *",
                     showarrow=False, font=dict(size=12, color="#2c3e50")),
                dict(x=0.495, y=0.65, text="contains<br>1 → *",
                     showarrow=False, font=dict(size=12, color="#2c3e50")),
                dict(x=0.66, y=0.65, text="included in<br>* → *",
                     showarrow=False, font=dict(size=12, color="#2c3e50"))
            ]

            if detail in ["Medium", "Detailed"]:
                nodes += ["Category", "Payment"]
                x += [0.66, 0.33]
                y += [1.0, 0.2]
                colors += ["#fad7a0", "#fad7a0"]
                texts += [
                    "Category<br>category_id (PK)<br>name<br>description",
                    "Payment<br>payment_id (PK)<br>order_id (FK)<br>amount<br>method"
                ]
                edge_x += [0.66, 0.66, None, 0.33, 0.33]
                edge_y += [0.8, 1.0, None, 0.8, 0.2]
                annotations += [
                    dict(x=0.66, y=0.9, text="belongs to<br>* → 1",
                         showarrow=False, font=dict(size=12)),
                    dict(x=0.33, y=0.5, text="has<br>1 → 1",
                         showarrow=False, font=dict(size=12))
                ]

        else:  # Default simple ER
            nodes = ["Entity1", "Entity2"]
            x = [0, 0.5]
            y = [0.5, 0.5]
            colors = ["#fad7a0"] * 2
            texts = ["Entity1<br>id (PK)<br>name<br>description",
                     "Entity2<br>id (PK)<br>entity1_id (FK)<br>value"]
            edge_x = [0, 0.5]
            edge_y = [0.5, 0.5]
            annotations = [
                dict(x=0.25, y=0.55, text="has many<br>1 → *", showarrow=False)]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(
            width=3, color='#888'), hoverinfo='none')

        node_trace = go.Scatter(
            x=x, y=y,
            mode='markers+text',
            text=nodes,
            textposition="middle center",
            marker=dict(size=80, color=colors, line_width=2),
            hoverinfo='text',
            hovertext=texts
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f"{scenario} - ER Diagram (Interactive)",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=20, r=20, t=60),
                            xaxis=dict(showgrid=False, zeroline=False,
                                       showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False,
                                       showticklabels=False),
                            annotations=annotations,
                            height=700
        ))
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### 🛠️ Sample Code")
        if scenario == "E-Commerce Platform":
            st.code("""
import plotly.graph_objects as go                    

nodes = ["Customer", "Order", "OrderItem", "Product"]
x = [0, 0.33, 0.66, 0.66]
y = [0.8, 0.8, 0.5, 0.8]
colors = ["#fad7a0"] * 4
texts = [
    "Customer<br>customer_id (PK)<br>name<br>email<br>address<br>phone",
    "Order<br>order_id (PK)<br>customer_id (FK)<br>order_date<br>total_amount<br>status",
    "OrderItem<br>order_id (PK, FK)<br>product_id (PK, FK)<br>quantity<br>unit_price",
    "Product<br>product_id (PK)<br>name<br>description<br>price<br>stock_quantity"
]

# Edges (x/y for lines)
edge_x = [0, 0.33, None, 0.33, 0.66, None, 0.66, 0.66]
edge_y = [0.8, 0.8, None, 0.5, 0.5, None, 0.8, 0.5]

annotations = [
    dict(x=0.165, y=0.85, text="places<br>1 → *",
            showarrow=False, font=dict(size=12, color="#2c3e50")),
    dict(x=0.495, y=0.65, text="contains<br>1 → *",
            showarrow=False, font=dict(size=12, color="#2c3e50")),
    dict(x=0.66, y=0.65, text="included in<br>* → *",
            showarrow=False, font=dict(size=12, color="#2c3e50"))
]

if detail in ["Medium", "Detailed"]:
    nodes += ["Category", "Payment"]
    x += [0.66, 0.33]
    y += [1.0, 0.2]
    colors += ["#fad7a0", "#fad7a0"]
    texts += [
        "Category<br>category_id (PK)<br>name<br>description",
        "Payment<br>payment_id (PK)<br>order_id (FK)<br>amount<br>method"
    ]
    edge_x += [0.66, 0.66, None, 0.33, 0.33]
    edge_y += [0.8, 1.0, None, 0.8, 0.2]
    annotations += [
        dict(x=0.66, y=0.9, text="belongs to<br>* → 1",
                showarrow=False, font=dict(size=12)),
        dict(x=0.33, y=0.5, text="has<br>1 → 1",
                showarrow=False, font=dict(size=12))
    ]

edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(
    width=3, color='#888'), hoverinfo='none')

node_trace = go.Scatter(
    x=x, y=y,
    mode='markers+text',
    text=nodes,
    textposition="middle center",
    marker=dict(size=80, color=colors, line_width=2),
    hoverinfo='text',
    hovertext=texts
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"{scenario} - ER Diagram (Interactive)",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=20, r=20, t=60),
                    xaxis=dict(showgrid=False, zeroline=False,
                                showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False,
                                showticklabels=False),
                    annotations=annotations,
                    height=700
))
fig.show()
""", language="python")
        else:
            st.code("""
import plotly.graph_objects as go                    

nodes = ["Entity1", "Entity2"]
x = [0, 0.5]
y = [0.5, 0.5]
colors = ["#fad7a0"] * 2
texts = ["Entity1<br>id (PK)<br>name<br>description",
            "Entity2<br>id (PK)<br>entity1_id (FK)<br>value"]
edge_x = [0, 0.5]
edge_y = [0.5, 0.5]
annotations = [dict(x=0.25, y=0.55, text="has many<br>1 → *", showarrow=False)]

edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(
    width=3, color='#888'), hoverinfo='none')

node_trace = go.Scatter(
    x=x, y=y,
    mode='markers+text',
    text=nodes,
    textposition="middle center",
    marker=dict(size=80, color=colors, line_width=2),
    hoverinfo='text',
    hovertext=texts
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"{scenario} - ER Diagram (Interactive)",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=20, r=20, t=60),
                    xaxis=dict(showgrid=False, zeroline=False,
                                showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False,
                                showticklabels=False),
                    annotations=annotations,
                    height=700
))
fig.show()
""", language="python")

    def render_chart(self, chart_type: str, scenario: str, detail: str):
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Organizational Chart":
            self.create_org_treemap(scenario, detail)
        else:
            self.create_er_network(scenario, detail)

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div class='stats-box'>Detail: {detail}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(
                f"<div class='stats-box'>Scenario: {scenario}</div>", unsafe_allow_html=True)
        with col3:
            st.markdown(
                f"<div class='stats-box'>Type: {chart_type}</div>", unsafe_allow_html=True)

    def render_key_characteristics(self):
        st.markdown(
            "### 🏗️ Understanding Structural and System Organization Analysis")

        st.markdown("""
        Structural analysis documents **how a system is organized**.
        It focuses on formal relationships, hierarchy, and logical structure
        rather than behavior or variation.
        """)

        st.markdown("#### 📐 Showing Formal Relationships")
        st.markdown("""
        Structural views represent explicit, designed relationships such as:
        - Parent–child hierarchies  
        - Ownership and responsibility  
        - Component dependencies  
        """)

        st.markdown("#### 🧭 Clear Hierarchical or Logical Structure")
        st.markdown("""
        Hierarchies and logical groupings make complex systems understandable.
        They clarify roles, boundaries, and how components fit together.
        """)

        st.markdown("#### 📄 Documenting System Organization")
        st.markdown("""
        Structural analysis serves as system documentation.
        It provides a stable reference for:
        - Onboarding  
        - Auditing  
        - Maintenance and change management  
        """)

        st.markdown("#### 🗣️ Supporting Planning and Communication")
        st.markdown("""
        Clear structure improves cross-team communication.
        It enables better planning by exposing dependencies, constraints,
        and system boundaries.
        """)

        st.divider()

        st.markdown("#### 🎯 Why Structural Analysis Matters")
        st.markdown("""
        Structural clarity reduces complexity and architectural drift.
        It supports:
        - System design and governance  
        - Scalability and maintainability  
        - Long-term sustainability  
        """)

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Company Structure": "Departments, teams, and reporting lines",
            "Database Design": "Tables, relationships, and keys",
            "Software Architecture": "Components, modules, and dependencies",
            "Family Trees": "Genealogical relationships across generations",
            "Military Hierarchy": "Ranks and chain of command"
        }

        for example, description in examples.items():
            with st.expander(f"🏗️ {example}"):
                st.write(description)

    def output(self):
        self.render_header()
        chart_type, scenario, detail_level = self.render_configuration()

        self.render_chart(chart_type, scenario, detail_level)
        self.render_examples()
        self.render_key_characteristics()
