import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd


class Hierarchical:
    def __init__(self):
        self.title = "🌳 Hierarchical Visualizations Dashboard"
        self.chart_types = [
            "Treemap",
            "Sunburst Chart",
            "Dendrogram",
            "Icicle Chart",
            "Circle Packing"
        ]
        self.scenarios = [
            "Company Organization",
            "Product Category Sales",
            "Website Navigation Structure",
            "Biological Taxonomy",
            "File System Structure"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Hierarchical charts represent **nested, tree-like structures** where elements have parent-child relationships. 
        They show how items are organized from general to specific.
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

    def get_hierarchy_data(self, scenario: str) -> pd.DataFrame:
        """Return hierarchy as a DataFrame with columns: label, parent, value"""
        if scenario == "Company Organization":
            data = {
                "label": ["CEO", "CTO", "CFO", "CMO", "VP Engineering", "VP Product", "VP Sales", "VP Marketing",
                          "Engineering Team", "Dev Team", "QA Team", "Product Team", "Sales Team", "Marketing Team"],
                "parent": ["", "CEO", "CEO", "CEO", "CTO", "CTO", "CFO", "CMO",
                           "VP Engineering", "VP Engineering", "VP Engineering", "VP Product", "VP Sales", "VP Marketing"],
                "value": [100, 40, 30, 30, 25, 20, 20, 20, 15, 10, 10, 15, 15, 15]
            }
            title = "Company Organization Structure"

        elif scenario == "Product Category Sales":
            data = {
                "label": ["All Products", "Electronics", "Clothing", "Home", "Sports",
                          "Phones", "Laptops", "TVs", "Men", "Women", "Kitchen", "Furniture", "Fitness", "Outdoor"],
                "parent": ["", "All Products", "All Products", "All Products", "All Products",
                           "Electronics", "Electronics", "Electronics", "Clothing", "Clothing", "Home", "Home", "Sports", "Sports"],
                "value": [1000, 450, 300, 150, 100, 200, 150, 100, 160, 140, 80, 70, 60, 40]
            }
            title = "Product Category Sales ($K)"

        elif scenario == "Website Navigation Structure":
            data = {
                "label": ["Home", "Products", "About", "Support", "Blog",
                          "Electronics", "Clothing", "Company", "Team", "Contact", "FAQ", "Docs", "News", "Tutorials"],
                "parent": ["", "Home", "Home", "Home", "Home",
                           "Products", "Products", "About", "About", "Support", "Support", "Support", "Blog", "Blog"],
                "value": [500, 300, 100, 120, 80, 150, 150, 50, 50, 70, 50, 60, 40, 40]
            }
            title = "Website Navigation Hierarchy"

        elif scenario == "Biological Taxonomy":
            data = {
                "label": ["Life", "Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Canidae",
                          "Felis", "Panthera", "Canis", "Cat", "Lion", "Tiger", "Dog", "Wolf"],
                "parent": ["", "Life", "Animalia", "Chordata", "Mammalia", "Carnivora", "Carnivora",
                           "Felidae", "Felidae", "Canidae", "Felis", "Panthera", "Panthera", "Canis", "Canis"],
                "value": [1000, 800, 700, 500, 300, 150, 150, 80, 70, 100, 50, 40, 30, 60, 40]
            }
            title = "Biological Classification (Simplified)"

        else:  # File System Structure
            data = {
                "label": ["Root", "Home", "Documents", "Pictures", "Work", "Personal", "Projects", "Vacation", "Reports"],
                "parent": ["", "Root", "Home", "Home", "Documents", "Documents", "Documents", "Pictures", "Work"],
                "value": [800, 500, 300, 200, 150, 150, 120, 120, 100]
            }
            title = "File System Hierarchy"

        df = pd.DataFrame(data)
        df.attrs["title"] = title
        return df

    def create_treemap(self, df: pd.DataFrame):
        fig = go.Figure(go.Treemap(
            labels=df["label"],
            parents=df["parent"],
            values=df["value"],
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            marker_colorscale="Viridis"
        ))
        fig.update_layout(
            title=f"Treemap – {df.attrs['title']}<br><sub>Rectangle size proportional to value</sub>",
            height=700
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
fig = go.Figure(go.Treemap(
    labels=df["label"],
    parents=df["parent"],
    values=df["value"],
    textinfo="label+value",
    hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
    marker_colorscale="Viridis"
))
fig.update_layout(
    title=f"Treemap – {df.attrs['title']}<br><sub>Rectangle size proportional to value</sub>",
    height=700
)
fig.show()""", language="python")

    def create_sunburst(self, df: pd.DataFrame):
        fig = go.Figure(go.Sunburst(
            labels=df["label"],
            parents=df["parent"],
            values=df["value"],
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            marker_colorscale="Plasma"
        ))
        fig.update_layout(
            title=f"Sunburst Chart – {df.attrs['title']}<br><sub>Click to drill down into levels</sub>",
            height=700
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
fig = go.Figure(go.Treemap(
    labels=df["label"],
    parents=df["parent"],
    values=df["value"],
    textinfo="label+value",
    hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
    marker_colorscale="Viridis"
))
fig.update_layout(
    title=f"Treemap – {df.attrs['title']}<br><sub>Rectangle size proportional to value</sub>",
    height=700
)
fig.show()""", language="python")

    def create_dendrogram(self, df: pd.DataFrame):
        fig = go.Figure()
        level_map = {}
        positions = {}
        max_level = 0

        for _, row in df.iterrows():
            label, parent = row["label"], row["parent"]
            if parent == "":
                level_map[label] = 0
                positions[label] = (0, 0.5)
            else:
                level_map[label] = level_map[parent] + 1
                max_level = max(max_level, level_map[label])
                parent_x, parent_y = positions[parent]
                child_x = level_map[label] / (max_level + 1)
                child_y = np.random.uniform(parent_y - 0.1, parent_y + 0.1)
                positions[label] = (child_x, child_y)
                fig.add_trace(go.Scatter(x=[parent_x, child_x], y=[parent_y, child_y],
                                         mode='lines', line=dict(color="#7f8c8d"), hoverinfo='none'))

        xs = [positions[l][0] for l in df["label"]]
        ys = [positions[l][1] for l in df["label"]]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            text=df["label"],
            textposition="middle right",
            marker=dict(size=df["value"]/10 + 10,
                        color=df["value"], colorscale="Blues"),
            hoverinfo='text'
        ))
        fig.update_layout(
            title=f"Dendrogram – {df.attrs['title']}<br><sub>Horizontal tree layout</sub>",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
fig = go.Figure()
level_map = {}
positions = {}
max_level = 0

for _, row in df.iterrows():
    label, parent = row["label"], row["parent"]
    if parent == "":
        level_map[label] = 0
        positions[label] = (0, 0.5)
    else:
        level_map[label] = level_map[parent] + 1
        max_level = max(max_level, level_map[label])
        parent_x, parent_y = positions[parent]
        child_x = level_map[label] / (max_level + 1)
        child_y = np.random.uniform(parent_y - 0.1, parent_y + 0.1)
        positions[label] = (child_x, child_y)
        fig.add_trace(go.Scatter(x=[parent_x, child_x], y=[parent_y, child_y],
                                    mode='lines', line=dict(color="#7f8c8d"), hoverinfo='none'))

xs = [positions[l][0] for l in df["label"]]
ys = [positions[l][1] for l in df["label"]]

fig.add_trace(go.Scatter(
    x=xs, y=ys,
    mode='markers+text',
    text=df["label"],
    textposition="middle right",
    marker=dict(size=df["value"]/10 + 10, color=df["value"], colorscale="Blues"),
    hoverinfo='text'
))
fig.update_layout(
    title=f"Dendrogram – {df.attrs['title']}<br><sub>Horizontal tree layout</sub>",
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=700,
    plot_bgcolor="white"
)
fig.show()""", language="python")

    def create_icicle(self, df: pd.DataFrame):
        fig = go.Figure(go.Icicle(
            labels=df["label"],
            parents=df["parent"],
            values=df["value"],
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            marker_colorscale="Earth"
        ))
        fig.update_layout(
            title=f"Icicle Chart – {df.attrs['title']}<br><sub>Horizontal partitioned layout</sub>",
            height=700
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
fig = go.Figure(go.Icicle(
    labels=df["label"],
    parents=df["parent"],
    values=df["value"],
    branchvalues="total",
    hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
    marker_colorscale="Earth"
))
fig.update_layout(
    title=f"Icicle Chart – {df.attrs['title']}<br><sub>Horizontal partitioned layout</sub>",
    height=700
)
fig.show()""", language="python")

    def create_circle_packing(self, df: pd.DataFrame):
        fig = go.Figure(go.Sunburst(
            labels=df["label"],
            parents=df["parent"],
            values=df["value"],
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            marker_colorscale="Rainbow",
            maxdepth=2
        ))
        fig.update_traces(insidetextorientation='radial')
        fig.update_layout(
            title=f"Circle Packing – {df.attrs['title']}<br><sub>Nested circles proportional to value</sub>",
            height=700
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
fig = go.Figure(go.Sunburst(
    labels=df["label"],
    parents=df["parent"],
    values=df["value"],
    branchvalues="total",
    hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
    marker_colorscale="Rainbow",
    maxdepth=2
))
fig.update_traces(insidetextorientation='radial')
fig.update_layout(
    title=f"Circle Packing – {df.attrs['title']}<br><sub>Nested circles proportional to value</sub>",
    height=700
)
fig.show()""", language="python")

    def render_chart(self, chart_type: str, scenario: str):
        df = self.get_hierarchy_data(scenario)
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Treemap":
            self.create_treemap(df)
        elif chart_type == "Sunburst Chart":
            self.create_sunburst(df)
        elif chart_type == "Dendrogram":
            self.create_dendrogram(df)
        elif chart_type == "Icicle Chart":
            self.create_icicle(df)
        else:
            self.create_circle_packing(df)

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Company Organization": "From CEO down to individual contributors",
            "Product Categories": "Main categories to subcategories to items",
            "File Systems": "Folders and subfolders on computers",
            "Biological Classification": "Kingdom, phylum, class, order, family",
            "Website Navigation": "Site map from homepage to individual pages"
        }

        for example, description in examples.items():
            with st.expander(f"💭 {example}"):
                st.write(description)

    def render_key_characteristics(self):
        st.markdown("### 🌳 Understanding Hierarchical Analysis")

        st.markdown("""
        Hierarchical analysis visualizes **parent-child relationships and multi-level structures**.
        It helps users explore both high-level organization and detailed components.
        """)

        st.markdown("#### 🔗 Clear Parent-Child Relationships")
        st.markdown("""
        Each element is linked to its parent:
        - Shows dependencies and ownership  
        - Traces lineage of items  
        - Supports organizational clarity  
        """)

        st.markdown("#### 📂 Multiple Levels of Nesting")
        st.markdown("""
        Hierarchies can extend across many layers:
        - Top-level categories  
        - Subcategories and subcomponents  
        - Detailed nested structures  
        """)

        st.markdown("#### 🌐 Shows Both Breadth and Depth")
        st.markdown("""
        Users can see:
        - The scope of items at each level (breadth)  
        - Detailed composition of each node (depth)  
        """)

        st.markdown("#### 🧭 Supports Drill-Down Exploration")
        st.markdown("""
        Interactive hierarchical views allow users to:
        - Expand or collapse nodes  
        - Explore details without overwhelming the display  
        - Focus on areas of interest efficiently  
        """)

        st.divider()

        st.markdown("#### 🎯 Why Hierarchical Analysis Matters")
        st.markdown("""
        Hierarchical visualizations reveal structure and dependencies that flat views cannot.
        They are essential for:
        - Organizational charts  
        - Taxonomy exploration  
        - File system or product category analysis  
        - Planning and decision-making  
        """)

    def output(self):
        self.render_header()
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
        self.render_examples()
        self.render_key_characteristics()
