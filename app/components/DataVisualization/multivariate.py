import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


class Multivariate:
    def __init__(self):
        self.title = "📊 Multivariate Visualization Dashboard"
        self.chart_types = ["Parallel Coordinates",
                            "Scatter Plot Matrix", "Radar Chart", "Heatmap"]
        self.dataset_types = [
            "Product Comparison", "Candidate Assessment", "Investment Analysis",
            "Supplier Selection", "Performance Review", "Customer Segmentation"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Multivariate charts handle three or more variables simultaneously, revealing complex patterns 
        and relationships that aren't visible when examining variables in isolation.
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
            num_items = st.slider("Number of Items", 5,
                                  30, 15, key="num_items")
            num_dimensions = st.slider(
                "Number of Dimensions", 4, 10, 6, key="num_dims")
            normalize_data = st.checkbox(
                "Normalize Variables", value=True, key="normalize")

        return chart_type, dataset_type, num_items, num_dimensions, normalize_data

    def generate_multivariate_data(self, data_type: str, n_items: int, n_dims: int) -> tuple[pd.DataFrame, list]:
        np.random.seed(42)

        dim_sets = {
            "Product Comparison": ['Price', 'Quality', 'Features', 'Reviews', 'Durability', 'Design', 'Support', 'Innovation'],
            "Candidate Assessment": ['Technical_Skills', 'Communication', 'Leadership', 'Experience', 'Creativity', 'Teamwork', 'Problem_Solving'],
            "Investment Analysis": ['Return_Rate', 'Risk_Level', 'Growth_Potential', 'Liquidity', 'Volatility', 'Dividends', 'Sustainability'],
            "Supplier Selection": ['Cost', 'Quality', 'Delivery_Time', 'Reliability', 'Service', 'Flexibility', 'Capacity'],
            "Performance Review": ['Productivity', 'Quality', 'Teamwork', 'Innovation', 'Communication', 'Reliability', 'Adaptability'],
            "Customer Segmentation": ['Spending_Score', 'Frequency', 'Recency', 'Loyalty', 'Satisfaction', 'Engagement', 'Referral_Rate']
        }

        dimensions = dim_sets[data_type][:n_dims]
        items = [f"{data_type.split()[0]}_{i+1:02d}" for i in range(n_items)]

        # Generate correlated data with realistic structure
        base = np.random.randn(n_items)
        data = {}

        for i, dim in enumerate(dimensions):
            corr = max(0.2, 0.9 - i * 0.1)  # decreasing correlation
            noise = np.random.randn(n_items)
            values = base * corr + noise * (1 - corr)

            # Scale appropriately
            if any(k in dim for k in ['Price', 'Cost']):
                values = np.interp(
                    values, (values.min(), values.max()), (50, 800))
            elif any(k in dim for k in ['Score', 'Rating', 'Level']):
                values = np.interp(
                    values, (values.min(), values.max()), (1, 10))
            else:
                values = np.interp(
                    values, (values.min(), values.max()), (0, 100))

            data[dim] = values

        df = pd.DataFrame(data)
        df.insert(0, 'Item', items)

        # Add outliers
        for dim in dimensions[:min(3, len(dimensions))]:
            outliers = np.random.choice(
                n_items, size=min(3, n_items//5), replace=False)
            df.loc[outliers, dim] *= np.random.uniform(1.8, 3.0)

        return df, dimensions

    def render_parallel_coordinates(self, df: pd.DataFrame, dimensions: list, data_type: str, normalize: bool):
        st.markdown("### 📈 Parallel Coordinates - Multi-Dimensional Patterns")

        selected_dims = st.multiselect(
            "Select Dimensions", dimensions, default=dimensions[:min(7, len(dimensions))])
        if len(selected_dims) < 2:
            st.warning("Select at least 2 dimensions.")
            return

        df_plot = df[['Item'] + selected_dims].copy()
        if normalize:
            scaler = MinMaxScaler()
            df_plot[selected_dims] = scaler.fit_transform(
                df_plot[selected_dims])

        fig = px.parallel_coordinates(
            df_plot,
            dimensions=selected_dims,
            color=selected_dims[0],
            labels={d: d.replace('_', ' ') for d in selected_dims},
            title=f"{data_type} - Parallel Coordinates",
            color_continuous_scale=px.colors.diverging.Tealrose
        )
        st.plotly_chart(fig, width='stretch')

        # Insights
        st.markdown("### 🔍 Key Patterns")
        corr = df[selected_dims].corr()
        strong = [(c1, c2, corr.loc[c1, c2])
                  for c1 in selected_dims for c2 in selected_dims if c1 < c2 and abs(corr.loc[c1, c2]) > 0.6]

        if strong:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Strong Positive Links:**")
                for c1, c2, r in [s for s in strong if s[2] > 0.6]:
                    st.success(f"{c1} ↔ {c2}: {r:+.3f}")
            with col2:
                st.write("**Strong Negative Links:**")
                for c1, c2, r in [s for s in strong if s[2] < -0.6]:
                    st.error(f"{c1} ↔ {c2}: {r:+.3f}")

    def render_scatter_plot_matrix(self, df: pd.DataFrame, dimensions: list, data_type: str, normalize: bool):
        st.markdown("### 🔁 Scatter Plot Matrix - Pairwise Relationships")

        dims = dimensions[:min(6, len(dimensions))]
        df_plot = df[dims].copy()
        if normalize:
            df_plot = pd.DataFrame(
                MinMaxScaler().fit_transform(df_plot), columns=dims)

        fig = px.scatter_matrix(
            df_plot,
            dimensions=dims,
            color=df_plot[dims[0]],
            title=f"{data_type} - Scatter Matrix",
            height=800
        )
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, width='stretch')

        # Correlation heatmap
        st.markdown("### Correlation Overview")
        corr = df[dims].corr()
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm',
                    center=0, square=True, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    def render_radar_chart(self, df: pd.DataFrame, dimensions: list, data_type: str, normalize: bool):
        st.markdown("### 🎯 Radar Chart - Profile Comparison")

        selected_items = st.multiselect(
            "Select Items", df['Item'].tolist(), default=df['Item'].head(4).tolist())
        if not selected_items:
            st.warning("Select at least one item.")
            return

        radar_dims = dimensions[:min(8, len(dimensions))]
        df_plot = df.set_index('Item').loc[selected_items, radar_dims].copy()

        if normalize:
            df_plot = pd.DataFrame(MinMaxScaler().fit_transform(
                df_plot), index=df_plot.index, columns=df_plot.columns)

        fig = go.Figure()
        colors = px.colors.qualitative.Bold

        for i, item in enumerate(df_plot.index):
            values = df_plot.loc[item].tolist() + [df_plot.loc[item].iloc[0]]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_dims + [radar_dims[0]],
                fill='toself',
                name=item,
                line_color=colors[i % len(colors)]
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=f"{data_type} - Radar Comparison",
            height=600
        )
        st.plotly_chart(fig, width='stretch')

        # Profile summary
        st.markdown("### Profile Summary")
        df_plot['Overall'] = df_plot.mean(axis=1)
        df_plot['Balance'] = 1 - (df_plot.std(axis=1) / df_plot.mean(axis=1))
        summary = df_plot[['Overall', 'Balance']].round(3)
        st.dataframe(summary.style.highlight_max(axis=0))

    def render_heatmap(self, df: pd.DataFrame, dimensions: list, data_type: str, normalize: bool):
        st.markdown("### 🌡️ Heatmap - Value Intensity Matrix")

        heatmap_data = df.set_index('Item')[dimensions].copy()
        if normalize:
            heatmap_data = pd.DataFrame(MinMaxScaler().fit_transform(heatmap_data),
                                        index=heatmap_data.index, columns=heatmap_data.columns)

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            center=0.5 if normalize else None,
            cbar_kws={"label": "Normalized Value" if normalize else "Raw Value"},
            ax=ax
        )
        ax.set_title(f"{data_type} - Multivariate Heatmap")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)

        # Clustering
        st.markdown("### Clustering Insights")
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(heatmap_data)
        df['Cluster'] = clusters

        cluster_means = df.groupby('Cluster')[dimensions].mean()
        st.write("**Cluster Profiles (Average Values):**")
        st.dataframe(cluster_means.round(2).style.highlight_max(axis=0))

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Multivariate Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **High-Dimensional Insight**
            - Handles 3+ variables at once
            - Reveals hidden interactions
            """)
            st.markdown("""
            **Pattern Discovery**
            - Detects clusters and outliers
            - Shows correlation structures
            """)
        with col2:
            st.markdown("""
            **Comparative Power**
            - Enables multi-criteria decisions
            - Highlights trade-offs
            """)
            st.markdown("""
            **Profile Analysis**
            - Compares items holistically
            - Identifies strengths/weaknesses
            """)

    def render_examples(self, dataset_type: str, df: pd.DataFrame):
        st.markdown("### 💡 Real-world Examples")
        examples = {
            "Product Comparison": "Evaluating products across features, price, reviews",
            "Candidate Assessment": "Comparing applicants on skills, experience, fit",
            "Investment Analysis": "Stocks rated on return, risk, growth, sustainability",
            "Supplier Selection": "Vendors evaluated on cost, quality, delivery",
            "Performance Review": "Employees assessed across multiple KPIs",
            "Customer Segmentation": "Customers profiled by behavior and value"
        }
        for ex, desc in examples.items():
            with st.expander(f"📊 {ex}"):
                st.write(desc)
                if dataset_type == ex:
                    st.success(
                        f"**Current Dataset:** {len(df)} items × {len(df.columns)-1} dimensions")

    def render_data_table(self, df: pd.DataFrame):
        st.markdown("### 📋 Multivariate Data Table")
        st.dataframe(df, width='stretch')

    def output(self):
        self.render_header()
        chart_type, dataset_type, num_items, num_dimensions, normalize_data = self.render_configuration()

        df, dimensions = self.generate_multivariate_data(
            dataset_type, num_items, num_dimensions)

        chart_map = {
            "Parallel Coordinates": lambda: self.render_parallel_coordinates(df, dimensions, dataset_type, normalize_data),
            "Scatter Plot Matrix": lambda: self.render_scatter_plot_matrix(df, dimensions, dataset_type, normalize_data),
            "Radar Chart": lambda: self.render_radar_chart(df, dimensions, dataset_type, normalize_data),
            "Heatmap": lambda: self.render_heatmap(df, dimensions, dataset_type, normalize_data)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_key_characteristics()
        self.render_examples(dataset_type, df)
        self.render_data_table(df)
