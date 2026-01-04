import streamlit as st
import plotly.graph_objects as go
import numpy as np


class Hierarchical:

    def __init__(self):
        st.set_page_config(
            page_title="Hierarchical Visualizations", layout="wide")
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

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **When to Use:**
            - Displaying organizational structures
            - Showing file system organization
            - Representing category taxonomies
            - Illustrating family relationships
            - Mapping website navigation
            """)

        with col2:
            st.markdown("""
            **Key Characteristics:**
            - Clear parent-child relationships
            - Multiple levels of nesting
            - Shows both breadth and depth
            - Supports drill-down exploration
            """)

        st.markdown("""
        **Supported Charts:**
        - **Treemap** – Space-filling rectangles for size and hierarchy
        - **Sunburst Chart** – Radial hierarchy with drill-down
        - **Dendrogram** – Tree diagram showing relationships
        - **Icicle Chart** – Horizontal partitioned rectangles
        - **Circle Packing** – Nested circles proportional to value
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

    def get_hierarchy_data(self, scenario: str):
        if scenario == "Company Organization":
            labels = ["CEO", "CTO", "CFO", "CMO", "VP Engineering", "VP Product", "VP Sales", "VP Marketing",
                      "Engineering Team", "Dev Team", "QA Team", "Product Team", "Sales Team", "Marketing Team"]
            parents = ["", "CEO", "CEO", "CEO", "CTO", "CTO", "CFO", "CMO",
                       "VP Engineering", "VP Engineering", "VP Engineering", "VP Product", "VP Sales", "VP Marketing"]
            values = [100, 40, 30, 30, 25, 20, 20, 20, 15, 10, 10, 15, 15, 15]
            title = "Company Organization Structure"

        elif scenario == "Product Category Sales":
            labels = ["All Products", "Electronics", "Clothing", "Home", "Sports",
                      "Phones", "Laptops", "TVs", "Men", "Women", "Kitchen", "Furniture", "Fitness", "Outdoor"]
            parents = ["", "All Products", "All Products", "All Products", "All Products",
                       "Electronics", "Electronics", "Electronics", "Clothing", "Clothing", "Home", "Home", "Sports", "Sports"]
            values = [1000, 450, 300, 150, 100,
                      200, 150, 100, 160, 140, 80, 70, 60, 40]
            title = "Product Category Sales ($K)"

        elif scenario == "Website Navigation Structure":
            labels = ["Home", "Products", "About", "Support", "Blog",
                      "Electronics", "Clothing", "Company", "Team", "Contact", "FAQ", "Docs", "News", "Tutorials"]
            parents = ["", "Home", "Home", "Home", "Home",
                       "Products", "Products", "About", "About", "Support", "Support", "Support", "Blog", "Blog"]
            values = [500, 300, 100, 120, 80,
                      150, 150, 50, 50, 70, 50, 60, 40, 40]
            title = "Website Navigation Hierarchy"

        elif scenario == "Biological Taxonomy":
            labels = ["Life", "Animalia", "Chordata", "Mammalia", "Carnivora", "Felidae", "Canidae",
                      "Felis", "Panthera", "Canis", "Cat", "Lion", "Tiger", "Dog", "Wolf"]
            parents = ["", "Life", "Animalia", "Chordata", "Mammalia", "Carnivora", "Carnivora",
                       "Felidae", "Felidae", "Canidae", "Felis", "Panthera", "Panthera", "Canis", "Canis"]
            values = [1000, 800, 700, 500, 300, 150, 150,
                      80, 70, 100, 50, 40, 30, 60, 40]
            title = "Biological Classification (Simplified)"

        else:  # File System Structure
            labels = ["Root", "Home", "Documents", "Pictures",
                      "Work", "Personal", "Projects", "Vacation", "Reports"]
            parents = ["", "Root", "Home", "Home", "Documents",
                       "Documents", "Documents", "Pictures", "Work"]
            values = [800, 500, 300, 200, 150, 150, 120, 120, 100]
            title = "File System Hierarchy"

        return labels, parents, values, title

    def create_treemap(self, labels, parents, values, title):
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            marker_colorscale="Viridis"
        ))

        fig.update_layout(
            title=f"Treemap – {title}<br><sub>Rectangle size proportional to value</sub>",
            height=700
        )
        return fig

    def create_sunburst(self, labels, parents, values, title):
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            marker_colorscale="Plasma"
        ))

        fig.update_layout(
            title=f"Sunburst Chart – {title}<br><sub>Click to drill down into levels</sub>",
            height=700
        )
        return fig

    def create_dendrogram(self, labels, parents, values, title):
        # Simplified dendrogram using scatter with lines
        fig = go.Figure()

        level_map = {}
        x_pos = 0
        positions = {}

        for i, label in enumerate(labels):
            if parents[i] == "":
                level_map[label] = 0
                positions[label] = (0, 0)
            else:
                parent_level = level_map[parents[i]]
                level_map[label] = parent_level + 1

        max_level = max(level_map.values())
        y_step = 1 / (max_level + 1)

        for label, parent in zip(labels, parents):
            if parent == "":
                positions[label] = (0, 0.5)
            else:
                parent_x, parent_y = positions[parent]
                child_x = level_map[label] / (max_level + 1)
                child_y = np.random.uniform(parent_y - 0.1, parent_y + 0.1)
                positions[label] = (child_x, child_y)

                fig.add_trace(go.Scatter(x=[parent_x, child_x], y=[parent_y, child_y],
                                         mode='lines', line=dict(color="#7f8c8d"), hoverinfo='none'))

        xs = [positions[l][0] for l in labels]
        ys = [positions[l][1] for l in labels]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            text=labels,
            textposition="middle right",
            marker=dict(size=[v/10 + 10 for v in values],
                        color=values, colorscale="Blues"),
            hoverinfo='text'
        ))

        fig.update_layout(
            title=f"Dendrogram – {title}<br><sub>Horizontal tree layout</sub>",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            plot_bgcolor="white"
        )
        return fig

    def create_icicle(self, labels, parents, values, title):
        fig = go.Figure(go.Icicle(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            marker_colorscale="Earth"
        ))

        fig.update_layout(
            title=f"Icicle Chart – {title}<br><sub>Horizontal partitioned layout</sub>",
            height=700
        )
        return fig

    def create_circle_packing(self, labels, parents, values, title):
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
            marker_colorscale="Rainbow",
            maxdepth=2  # Limits depth for clearer packing view
        ))

        fig.update_traces(insidetextorientation='radial')
        fig.update_layout(
            title=f"Circle Packing – {title}<br><sub>Nested circles proportional to value</sub>",
            height=700
        )
        return fig

    def render_chart(self, chart_type: str, scenario: str):
        labels, parents, values, title = self.get_hierarchy_data(scenario)
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Treemap":
            fig = self.create_treemap(labels, parents, values, title)
        elif chart_type == "Sunburst Chart":
            fig = self.create_sunburst(labels, parents, values, title)
        elif chart_type == "Dendrogram":
            fig = self.create_dendrogram(labels, parents, values, title)
        elif chart_type == "Icicle Chart":
            fig = self.create_icicle(labels, parents, values, title)
        else:  # Circle Packing
            fig = self.create_circle_packing(labels, parents, values, title)

        st.plotly_chart(fig, width='stretch')

    def output(self):
        self.render_header()
        chart_type, scenario = self.render_configuration()
        self.render_chart(chart_type, scenario)
