import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


class Correlation:
    def __init__(self):
        self.title = "🔗 Correlation Visualization Dashboard"
        self.chart_types = ["Correlogram", "Euler Diagram", "Fan Chart"]
        self.dataset_types = [
            "Financial Analysis", "Marketing Mix", "Scientific Research",
            "Quality Control", "HR Analytics", "Healthcare Data"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Correlation charts specifically measure and display the strength and direction of relationships 
        between variables. They quantify how variables move together or independently.
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
            num_variables = st.slider(
                "Number of Variables", 3, 10, 5, key="num_vars")
            corr_method = st.selectbox("Correlation Method", [
                                       "Pearson", "Spearman"], key="corr_method")

        return chart_type, dataset_type, num_variables, corr_method

    def generate_correlation_data(self, data_type: str, n_vars: int) -> pd.DataFrame:
        np.random.seed(42)

        var_lists = {
            "Financial Analysis": ['Stock_A', 'Stock_B', 'Bond_Yield', 'GDP_Growth', 'Inflation',
                                   'Unemployment', 'Consumer_Confidence', 'Oil_Price'],
            "Marketing Mix": ['TV_Ads', 'Social_Media', 'Email_Campaigns', 'SEO_Traffic', 'PPC_Clicks',
                              'Sales', 'Brand_Awareness', 'Customer_Retention'],
            "Scientific Research": ['Temperature', 'Pressure', 'Reaction_Time', 'Catalyst_Amount', 'Yield',
                                    'Purity', 'Energy_Input', 'Efficiency'],
            "Quality Control": ['Machine_Speed', 'Temperature', 'Humidity', 'Raw_Material_Batch',
                                'Operator_Shift', 'Defect_Rate', 'Product_Weight', 'Tolerance'],
            "HR Analytics": ['Salary', 'Experience', 'Education_Level', 'Training_Hours', 'Satisfaction_Score',
                             'Productivity', 'Tenure', 'Promotion_Count'],
            "Healthcare Data": ['Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Exercise_Hours',
                                'Sleep_Hours', 'Stress_Level', 'Medication_Dosage']
        }

        variables = var_lists[data_type][:n_vars]
        n_samples = 200

        # Generate base latent factors
        factors = np.random.randn(n_samples, max(2, n_vars // 3))

        data = {}
        for i, var in enumerate(variables):
            # Mix of shared and unique variance
            weights = np.random.uniform(-1.2, 1.2, factors.shape[1])
            shared = factors @ weights
            unique = np.random.randn(n_samples) * np.random.uniform(0.5, 1.5)
            data[var] = shared + unique

        df = pd.DataFrame(data)

        # Add realistic outliers
        for col in df.columns[:min(3, len(df.columns))]:
            outliers = np.random.choice(df.index, size=5, replace=False)
            df.loc[outliers, col] *= np.random.uniform(2, 4)

        return df

    def calculate_correlation_matrix(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        return df.corr(method=method.lower())

    def render_correlogram(self, df: pd.DataFrame, corr_matrix: pd.DataFrame, data_type: str, method: str):
        st.markdown("### 📊 Correlogram - Correlation Matrix Overview")

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax1)
        ax1.set_title(f"Correlation Heatmap ({method})")

        # Scatter matrix (sample for speed)
        ax2 = fig.add_subplot(gs[0, 1])
        sample = df.sample(min(80, len(df)))
        pd.plotting.scatter_matrix(
            sample, alpha=0.6, diagonal='kde', ax=ax2, figsize=(8, 8))
        ax2.set_title("Scatter Matrix (Sample)")

        # Distribution of correlations
        ax3 = fig.add_subplot(gs[1, :])
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_vals = upper_tri.stack().values
        ax3.hist(corr_vals, bins=20, color='skyblue',
                 edgecolor='black', alpha=0.8)
        ax3.axvline(corr_vals.mean(), color='red', linestyle='--',
                    label=f'Mean: {corr_vals.mean():.3f}')
        ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel("Correlation Coefficient")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Pairwise Correlations")
        ax3.legend()
        ax3.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        # Strong correlations
        st.markdown("### 🔗 Key Correlation Insights")
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                c = corr_matrix.iloc[i, j]
                if abs(c) > 0.3:
                    pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], c))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Strongest Positive Correlations:**")
            pos = [p for p in pairs if p[2] > 0.3][:6]
            for v1, v2, c in pos:
                st.success(f"**{v1} ↔ {v2}**: {c:+.3f}")

        with col2:
            st.write("**Strongest Negative Correlations:**")
            neg = [p for p in pairs if p[2] < -0.3][:6]
            for v1, v2, c in neg:
                st.error(f"**{v1} ↔ {v2}**: {c:+.3f}")

        if not pos and not neg:
            st.info(
                "All correlations are weak to moderate — variables are relatively independent.")

        st.markdown("""
        **When to use:** Comprehensive multivariate relationship exploration.
        
        **Key Features:** Heatmap + scatter matrix + distribution for full insight.
        """)

    def render_euler_diagram(self, df: pd.DataFrame, data_type: str):
        st.markdown("### ⭕ Euler Diagram - Categorical Overlaps")

        # Convert top variables to binary (above/below median)
        n_show = min(5, len(df.columns))
        binary = pd.DataFrame()
        for col in df.columns[:n_show]:
            median = df[col].median()
            binary[f"{col}_High"] = (df[col] > median).astype(int)

        variables = binary.columns.tolist()

        fig, ax = plt.subplots(figsize=(12, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(variables)))

        # Draw overlapping circles
        centers = np.array([[np.cos(2*np.pi*i/len(variables)), np.sin(2*np.pi*i/len(variables))]
                           for i in range(len(variables))]) * 1.2
        radius = 0.6

        for i, (var, (cx, cy)) in enumerate(zip(variables, centers)):
            circle = plt.Circle(
                (cx, cy), radius, color=colors[i], alpha=0.5, label=var)
            ax.add_patch(circle)
            count = binary[var].sum()
            pct = count / len(df) * 100
            ax.text(cx, cy, var.replace('_High', '\nHigh'), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

        # Intersections
        intersections = {}
        for i in range(len(variables)):
            for j in range(i+1, len(variables)):
                overlap = binary[[variables[i], variables[j]]].all(
                    axis=1).sum()
                if overlap > 0:
                    intersections[f"{variables[i]} ∩ {variables[j]}"] = overlap

        # Legend
        legend_items = [f"{v.replace('_High', ' High')}: {binary[v].sum()} ({binary[v].mean()*100:.1f}%)"
                        for v in variables]
        for inter, count in list(intersections.items())[:5]:
            legend_items.append(f"{inter}: {count}")

        ax.text(1.4, 0, "\n".join(legend_items), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{data_type} - Euler Diagram of High-Value Overlaps")

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** Understanding co-occurrence of conditions (e.g., high values in multiple metrics).
        
        **Key Features:** Visual set overlap, useful for categorical/binary relationships.
        """)

    def render_fan_chart(self, df: pd.DataFrame, corr_matrix: pd.DataFrame, data_type: str):
        st.markdown("### 🎐 Fan Chart - Correlation Strength Network")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Fan chart: sorted correlations
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        pairs = upper_tri.stack()
        pairs = pairs[pairs.abs() > 0.1]  # filter weak
        sorted_pairs = pairs.abs().sort_values(ascending=False)

        y_pos = np.arange(len(sorted_pairs))
        colors = ['red' if pairs[idx] <
                  0 else 'blue' for idx in sorted_pairs.index]

        bars = ax1.barh(y_pos, sorted_pairs.values, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(
            [f"{i} ↔ {j}" for (i, j) in sorted_pairs.index], fontsize=9)
        ax1.axvline(0, color='black', linewidth=0.8)
        ax1.set_xlabel("Absolute Correlation Strength")
        ax1.set_title("Ranked Pairwise Correlations")

        # Network graph
        G = nx.Graph()
        threshold = 0.3
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                c = corr_matrix.iloc[i, j]
                if abs(c) > threshold:
                    G.add_edge(
                        corr_matrix.columns[i], corr_matrix.columns[j], weight=abs(c), signed=c)

        pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

        weights = [G[u][v]['weight'] * 6 for u, v in G.edges()]
        edge_colors = ['red' if G[u][v]['signed']
                       < 0 else 'blue' for u, v in G.edges()]

        nx.draw_networkx_nodes(G, pos, node_size=800,
                               node_color='lightgreen', alpha=0.9, ax=ax2)
        nx.draw_networkx_edges(G, pos, width=weights,
                               edge_color=edge_colors, alpha=0.7, ax=ax2)
        nx.draw_networkx_labels(G, pos, font_size=10,
                                font_weight='bold', ax=ax2)

        edge_labels = {
            (u, v): f"{G[u][v]['signed']:+.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=8, ax=ax2)

        ax2.set_title(f"Correlation Network (|r| > {threshold})")
        ax2.axis('off')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### 💡 Key Insights")
        strong_pos = len(
            [e for e in G.edges(data=True) if e[2]['signed'] > 0.5])
        strong_neg = len(
            [e for e in G.edges(data=True) if e[2]['signed'] < -0.5])

        col1, col2 = st.columns(2)
        col1.metric("Strong Positive Links", strong_pos)
        col2.metric("Strong Negative Links", strong_neg)

        st.markdown("""
        **When to use:** Visualizing complex interrelationships and clustering.
        
        **Key Features:** Fan ranking + network view for structural insight.
        """)

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Correlation Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Quantifies Relationships**
            - Numerical strength and direction
            - Statistical significance testing
            """)
            st.markdown("""
            **Multivariate Support**
            - All pairwise relationships at once
            - Identifies clusters and redundancy
            """)

        with col2:
            st.markdown("""
            **Predictive Value**
            - Guides feature selection
            - Reveals potential multicollinearity
            """)
            st.markdown("""
            **Insight Generation**
            - Hypothesis validation
            - Discovery of hidden patterns
            """)

    def render_examples(self, dataset_type: str, corr_matrix: pd.DataFrame):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Financial Analysis": "How assets move together (portfolio risk)",
            "Marketing Mix": "Channel interdependence and cannibalization",
            "Scientific Research": "Variable interactions in experiments",
            "Quality Control": "Process parameters affecting defects",
            "HR Analytics": "Factors driving performance and retention",
            "Healthcare Data": "Risk factors and health outcomes"
        }

        for example, desc in examples.items():
            with st.expander(f"🔗 {example}"):
                st.write(desc)
                if dataset_type == example:
                    mean_abs = corr_matrix.abs().mean().mean()
                    st.success(f"**Average |r|:** {mean_abs:.3f}")
                    st.info(f"**Variables:** {len(corr_matrix)}")

    def render_data_preview(self, df: pd.DataFrame):
        st.markdown("### 📊 Data Preview (First 10 Rows)")
        st.dataframe(df.head(10), width='stretch')

    def output(self):
        self.render_header()
        chart_type, dataset_type, num_variables, corr_method = self.render_configuration()

        df = self.generate_correlation_data(dataset_type, num_variables)
        corr_matrix = self.calculate_correlation_matrix(df, corr_method)

        chart_map = {
            "Correlogram": lambda: self.render_correlogram(df, corr_matrix, dataset_type, corr_method),
            "Euler Diagram": lambda: self.render_euler_diagram(df, dataset_type),
            "Fan Chart": lambda: self.render_fan_chart(df, corr_matrix, dataset_type)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_key_characteristics()
        self.render_examples(dataset_type, corr_matrix)
        self.render_data_preview(df)
