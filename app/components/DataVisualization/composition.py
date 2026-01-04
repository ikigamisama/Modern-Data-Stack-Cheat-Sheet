import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import squarify
import matplotlib.colors as mcolors
from scipy.spatial import Voronoi, voronoi_plot_2d


class Composition:
    def __init__(self):
        self.title = "🥧 Composition Visualization Dashboard"
        self.chart_types = [
            "Pie Chart", "Donut Chart", "Stacked Area Chart",
            "Waffle Chart", "Mosaic Plot", "Voronoi Diagram"
        ]
        self.dataset_types = [
            "Budget Planning", "Market Analysis", "Revenue Breakdown",
            "Demographics", "Time Management", "Portfolio Allocation"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Composition charts break down a whole into its constituent parts, showing how different 
        components contribute to the total. They answer "what is this made of?" questions.
        """)

    def render_configuration(self):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            dataset_type = st.selectbox(
                "Dataset Type", self.dataset_types, key="dataset_type")

        with col3:
            show_percentages = st.checkbox(
                "Show Percentages", value=True, key="show_pct")
            explode_slices = st.checkbox(
                "Explode Slices", value=False, key="explode")

        return chart_type, dataset_type, show_percentages, explode_slices

    @staticmethod
    def generate_composition_data(data_type: str) -> pd.DataFrame:
        if data_type == "Budget Planning":
            categories = ['R&D', 'Marketing',
                          'Operations', 'HR', 'IT', 'Facilities']
            values = [35, 25, 20, 10, 8, 2]
            unit = "Million USD"

        elif data_type == "Market Analysis":
            categories = ['Apple', 'Samsung',
                          'Huawei', 'Xiaomi', 'Oppo', 'Others']
            values = [25, 22, 15, 12, 8, 18]
            unit = "Market Share %"

        elif data_type == "Revenue Breakdown":
            categories = ['Smartphones', 'Laptops',
                          'Tablets', 'Accessories', 'Services']
            values = [45, 30, 12, 8, 5]
            unit = "Revenue %"

        elif data_type == "Demographics":
            categories = ['0-18', '19-35', '36-50', '51-65', '65+']
            values = [22, 35, 25, 12, 6]
            unit = "Population %"

        elif data_type == "Time Management":
            categories = ['Work', 'Sleep', 'Meals',
                          'Leisure', 'Commute', 'Exercise']
            values = [9, 7, 2, 3, 1, 2]  # hours
            unit = "Hours per Day"

        else:  # Portfolio Allocation
            categories = ['Stocks', 'Bonds',
                          'Real Estate', 'Cash', 'Commodities']
            values = [45, 30, 15, 5, 5]
            unit = "Allocation %"

        total = sum(values)
        percentages = [v / total * 100 for v in values]

        return pd.DataFrame({
            'Category': categories,
            'Value': values,
            'Percentage': percentages,
            'Unit': [unit] * len(categories)
        })

    def render_pie_chart(self, df: pd.DataFrame, data_type: str, show_percentages: bool, explode_slices: bool):
        st.markdown("### 🥧 Pie Chart - Classic Composition")

        fig = px.pie(
            df, values='Value', names='Category',
            title=f"{data_type} - Pie Chart",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        if show_percentages:
            fig.update_traces(textposition='inside', textinfo='percent+label')
        else:
            fig.update_traces(textposition='inside', textinfo='label+value')

        if explode_slices:
            fig.update_traces(pull=[0.1] * len(df))  # Explode all slightly

        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        **When to use:** Simple part-to-whole relationships with few categories (3–7).
        
        **Key Features:** Instantly recognizable, great for highlighting dominant segments.
        """)

        st.code('''
import plotly.express as px
fig = px.pie(df, values='Value', names='Category',
             title='Composition Pie Chart',
             color_discrete_sequence=px.colors.qualitative.Set3)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
        ''', language='python')

    def render_donut_chart(self, df: pd.DataFrame, data_type: str, show_percentages: bool):
        st.markdown("### 🍩 Donut Chart - Modern Composition")

        fig = px.pie(
            df, values='Value', names='Category',
            title=f"{data_type} - Donut Chart",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        if show_percentages:
            fig.update_traces(textposition='inside', textinfo='percent+label')

        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        **When to use:** Same as pie chart but with a cleaner, modern look and space for central text.
        
        **Key Features:** Less visual clutter, center can display total or key metric.
        """)

        st.code('''
import plotly.express as px
fig = px.pie(df, values='Value', names='Category', hole=0.4,
             title='Donut Chart')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
        ''', language='python')

    def render_stacked_area_chart(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 📊 Stacked Area Chart - Temporal Composition")

        categories = df['Category'].tolist()
        base_values = df['Value'].values
        months = pd.date_range('2024-01-01', periods=12, freq='M')

        data = []
        for month in months:
            seasonal = 1 + 0.15 * np.sin(2 * np.pi * (month.month - 1) / 12)
            noise = np.random.normal(1, 0.08, len(categories))
            row = {'Month': month}
            for i, cat in enumerate(categories):
                row[cat] = max(0.1, base_values[i] * seasonal * noise[i])
            data.append(row)

        trend_df = pd.DataFrame(data)

        fig = px.area(
            trend_df, x='Month', y=categories,
            title=f"{data_type} - Composition Over 12 Months",
            color_discrete_sequence=px.colors.qualitative.Safe
        )

        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        **When to use:** Showing how composition changes over time while preserving the total.
        
        **Key Features:** Reveals trends in individual components and shifting proportions.
        """)

        st.code('''
import plotly.express as px
import pandas as pd
df_trend = pd.DataFrame({ ... })  # time-series data
fig = px.area(df_trend, x='Month', y=['Cat1', 'Cat2', 'Cat3'],
              title='Composition Over Time')
fig.show()
        ''', language='python')

    def render_waffle_chart(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🧇 Waffle Chart - Grid-Based Composition")
        fig, ax = plt.subplots(figsize=(10, 6))

        cmap = cm.get_cmap("Set3", len(df))
        colors = [cmap(i) for i in range(len(df))]

        labels = [f"{cat}\n{val:.1f}%" for cat,
                  val in zip(df['Category'], df['Percentage'])]

        squarify.plot(
            sizes=df['Value'],
            label=labels,
            color=colors,
            alpha=0.8,
            ax=ax,
            text_kwargs={'fontsize': 10}
        )

        ax.set_title(f"{data_type} - Waffle Chart")
        ax.axis('off')

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** When precise percentage comparison matters more than angular judgment.
        
        **Key Features:** Each block represents equal share, easier to compare than pie slices.
        """)

        st.code('''
import squarify
import matplotlib.pyplot as plt
squarify.plot(sizes=[40, 30, 20, 10], label=['A\\n40%', 'B\\n30%', 'C\\n20%', 'D\\n10%'],
              color=['#FF9999','#66B2FF','#99FF99','#FFD700'])
plt.axis('off')
plt.show()
        ''', language='python')

    def render_mosaic_plot(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🧩 Mosaic Plot - Hierarchical Composition")

        # Create hierarchical data: main category × quarter
        main_cats = df['Category'].tolist()
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']

        mosaic_data = []
        for cat in main_cats:
            base = df[df['Category'] == cat]['Value'].iloc[0] / 4
            variations = base * np.array([0.9, 1.1, 0.85, 1.15])
            for q, val in zip(quarters, variations):
                mosaic_data.append(
                    {'Category': cat, 'Quarter': q, 'Value': val})

        mosaic_df = pd.DataFrame(mosaic_data)
        pivot = mosaic_df.pivot(
            index='Category', columns='Quarter', values='Value')

        fig, ax = plt.subplots(figsize=(12, 7))

        # Use Matplotlib colormap for quarters
        cmap = cm.get_cmap("Set2", 4)  # 4 colors for 4 quarters
        colors = [cmap(i) for i in range(4)]  # RGBA tuples

        pivot.plot(kind='bar', stacked=True, ax=ax, color=colors)
        ax.set_title(f"{data_type} - Mosaic Plot (by Quarter)")
        ax.set_ylabel("Contribution")
        ax.legend(title="Quarter")

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** Showing composition across two categorical dimensions.
        
        **Key Features:** Area proportional to value, great for contingency tables.
        """)

        st.code('''
pivot_df.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Mosaic Plot')
plt.show()
        ''', language='python')

    def render_voronoi_diagram(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🔺 Voronoi Diagram - Spatial Composition")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Use Matplotlib colormap instead of Plotly
        cmap = cm.get_cmap("Set3", len(df))
        colors = [cmap(i) for i in range(len(df))]  # RGBA tuples

        points = []
        labels = []

        total_value = sum(df['Value'])
        for i, row in df.iterrows():
            n = max(3, int(row['Value'] / total_value * 60))
            x_base = i * 1.5
            for _ in range(n):
                points.append(
                    [x_base + np.random.normal(0, 0.3), np.random.normal(0, 0.4)])
                labels.append(i)

        points = np.array(points)

        if len(points) >= 3:
            vor = Voronoi(points)
            voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                            line_colors='white', line_width=1)

            for r_idx, region in enumerate(vor.regions):
                if -1 not in region and len(region) > 0:
                    polygon = [vor.vertices[i] for i in region]
                    cat_idx = labels[r_idx % len(labels)]
                    ax.fill(*zip(*polygon),
                            color=colors[cat_idx % len(colors)], alpha=0.6)

        # Legend
        for i, cat in enumerate(df['Category']):
            ax.plot([], [], color=colors[i], label=cat, linewidth=10)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_title(f"{data_type} - Voronoi Spatial Composition")
        ax.axis('off')

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** Creative spatial representation of influence or territory by proportion.
        
        **Key Features:** Larger categories claim more space via proximity.
        """)

        st.code('''
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(points)
voronoi_plot_2d(vor, ax=ax)
ax.fill(*zip(*polygon), alpha=0.6)
plt.show()
        ''', language='python')

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Composition Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Parts-to-Whole Relationship**
            - Always sums to 100% or a meaningful total
            - Shows relative contribution of each part
            """)
            st.markdown("""
            **Comparative Analysis**
            - Easy to spot dominant vs minor components
            - Visual hierarchy through size/area
            """)

        with col2:
            st.markdown("""
            **Temporal Composition**
            - Tracks how shares evolve over time
            - Highlights rising/falling components
            """)
            st.markdown("""
            **Multi-dimensional Composition**
            - Supports hierarchy and subgroups
            - Handles nested breakdowns
            """)

    def render_examples(self, dataset_type: str, df: pd.DataFrame):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Budget Planning": "How organizational funds are distributed across programs",
            "Market Analysis": "Each competitor's share of the total market",
            "Revenue Breakdown": "Contribution of each product line to total sales",
            "Demographics": "Population distribution by age groups",
            "Time Management": "How daily hours are spent across activities",
            "Portfolio Allocation": "Asset class allocation in an investment portfolio"
        }

        for example, description in examples.items():
            with st.expander(f"📊 {example}"):
                st.write(description)
                if dataset_type == example:
                    total = df['Value'].sum()
                    unit = df['Unit'].iloc[0]
                    st.success(f"**Current Total:** {total:.1f} {unit}")

    def output(self):
        self.render_header()
        chart_type, dataset_type, show_percentages, explode_slices = self.render_configuration()

        df = self.generate_composition_data(dataset_type)

        chart_map = {
            "Pie Chart": lambda: self.render_pie_chart(df, dataset_type, show_percentages, explode_slices),
            "Donut Chart": lambda: self.render_donut_chart(df, dataset_type, show_percentages),
            "Stacked Area Chart": lambda: self.render_stacked_area_chart(df, dataset_type),
            "Waffle Chart": lambda: self.render_waffle_chart(df, dataset_type),
            "Mosaic Plot": lambda: self.render_mosaic_plot(df, dataset_type),
            "Voronoi Diagram": lambda: self.render_voronoi_diagram(df, dataset_type)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_key_characteristics()
        self.render_examples(dataset_type, df)
