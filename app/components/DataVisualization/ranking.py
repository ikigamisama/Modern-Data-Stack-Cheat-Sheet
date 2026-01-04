import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


class Ranking:
    def __init__(self):
        self.title = "🏆 Ranking Visualization Dashboard"
        self.chart_types = ["Bar Chart", "Slope Chart",
                            "Bullet Graph", "Step Chart"]
        self.dataset_types = [
            "Sales Leaderboard", "Product Comparison", "Sports Statistics",
            "College Rankings", "Employee Performance", "Website Analytics"
        ]

    def render_header(self):
        st.markdown(f"{self.title}")
        st.markdown("""
        ### Purpose
        Ranking charts order items from highest to lowest (or vice versa), making it easy to identify 
        top performers, compare relative positions, and track changes in standing over time.
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
                                  20, 10, key="num_items")
            show_comparison = st.checkbox(
                "Show Time Comparison", value=True, key="show_comp")

        return chart_type, dataset_type, num_items, show_comparison

    def generate_ranking_data(self, data_type: str, n_items: int) -> pd.DataFrame:
        np.random.seed(42)

        if data_type == "Sales Leaderboard":
            reps = [f"Sales Rep {chr(65 + i)}" for i in range(n_items)]
            sales = np.random.randint(50_000, 200_000, n_items)
            sales.sort()
            sales = sales[::-1]
            return pd.DataFrame({
                'sales_rep': reps,
                'sales_amount': sales,
                'growth_rate': np.random.uniform(-5, 25, n_items),
                'quota_achievement': np.random.uniform(80, 150, n_items)
            })

        elif data_type == "Product Comparison":
            products = [f"Product {chr(65 + i)}" for i in range(n_items)]
            satisfaction = np.random.uniform(3.5, 5.0, n_items)
            satisfaction.sort()
            satisfaction = satisfaction[::-1]
            return pd.DataFrame({
                'product': products,
                'satisfaction_score': satisfaction,
                'feature_count': np.random.randint(5, 20, n_items),
                'price': np.random.uniform(99, 999, n_items)
            })

        elif data_type == "Sports Statistics":
            teams = [f"Team {chr(65 + i)}" for i in range(n_items)]
            wins = np.random.randint(10, 50, n_items)
            wins.sort()
            wins = wins[::-1]
            return pd.DataFrame({
                'team': teams,
                'wins': wins,
                'points': wins * 2 + np.random.randint(0, 10, n_items),
                'performance_index': np.random.uniform(0.4, 0.9, n_items)
            })

        elif data_type == "College Rankings":
            colleges = [f"University {chr(65 + i)}" for i in range(n_items)]
            score = np.random.uniform(60, 100, n_items)
            score.sort()
            score = score[::-1]
            return pd.DataFrame({
                'college': colleges,
                'overall_score': score,
                'acceptance_rate': np.random.uniform(5, 50, n_items),
                'ranking': list(range(1, n_items + 1))
            })

        elif data_type == "Employee Performance":
            employees = [f"Employee {chr(65 + i)}" for i in range(n_items)]
            performance = np.random.uniform(3.0, 5.0, n_items)
            performance.sort()
            performance = performance[::-1]
            return pd.DataFrame({
                'employee': employees,
                'performance_score': performance,
                'projects_completed': np.random.randint(3, 12, n_items),
                'satisfaction_rating': np.random.uniform(4.0, 5.0, n_items)
            })

        else:  # Website Analytics
            sites = [f"Website {chr(65 + i)}" for i in range(n_items)]
            traffic = np.random.randint(10_000, 500_000, n_items)
            traffic.sort()
            traffic = traffic[::-1]
            return pd.DataFrame({
                'website': sites,
                'monthly_traffic': traffic,
                'avg_engagement': np.random.uniform(2.0, 8.0, n_items),
                'bounce_rate': np.random.uniform(20, 60, n_items)
            })

    def generate_previous_data(self, current_df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        prev = current_df.copy()
        np.random.seed(123)

        # Identify main value column (second column usually)
        value_col = current_df.columns[1]

        # Apply realistic variation
        prev[value_col] = current_df[value_col] * \
            np.random.uniform(0.8, 1.2, len(current_df))

        if prev[value_col].dtype == 'int':
            prev[value_col] = prev[value_col].astype(int)

        return prev

    def render_bar_chart(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 📊 Bar Chart - Ranked Comparison")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = st.selectbox("Value to Rank", numeric_cols, index=0)
        ascending = st.checkbox("Sort Ascending (Low to High)", value=False)

        sorted_df = df.sort_values(
            value_col, ascending=ascending).reset_index(drop=True)

        fig = px.bar(
            sorted_df,
            x=value_col,
            y=sorted_df.columns[0],
            orientation='h',
            title=f"{data_type} - Horizontal Bar Ranking",
            color=value_col,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=600, yaxis={
                          'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')

        st.markdown("### 🏆 Current Leaderboard")
        for i, row in sorted_df.iterrows():
            rank = i + 1
            val = row[value_col]
            name = row.iloc[0]
            st.markdown(f"**#{rank}** — **{name}**: {val:,.0f}" if isinstance(
                val, int) else f"**#{rank}** — **{name}**: {val:.2f}")

        st.markdown("""
        **When to use:** Clear top-to-bottom comparison, identifying leaders and laggards.
        
        **Key Features:** Horizontal bars for easy label reading, color intensity shows magnitude.
        """)

        st.code('''
import plotly.express as px
sorted_df = df.sort_values('sales_amount', ascending=False)
fig = px.bar(sorted_df, x='sales_amount', y='sales_rep', orientation='h',
             color='sales_amount', color_continuous_scale='Viridis')
fig.show()
        ''', language='python')

    def render_slope_chart(self, current_df: pd.DataFrame, previous_df: pd.DataFrame, data_type: str, show_comparison: bool):
        st.markdown("### 📈 Slope Chart - Ranking Changes Over Time")

        if not show_comparison:
            st.info("Enable 'Show Time Comparison' in sidebar to see changes.")
            return

        value_col = st.selectbox("Metric to Compare", current_df.select_dtypes(
            include=[np.number]).columns, index=1)

        # Prepare long-format data
        slope_data = []
        for _, row in current_df.iterrows():
            name = row.iloc[0]
            prev_val = previous_df.loc[previous_df.iloc[:, 0]
                                       == name, value_col].iloc[0]
            slope_data.extend([
                {'item': name, 'period': 'Previous', 'value': prev_val},
                {'item': name, 'period': 'Current', 'value': row[value_col]}
            ])

        slope_df = pd.DataFrame(slope_data)

        fig = px.line(
            slope_df, x='period', y='value', color='item',
            markers=True, title=f"{data_type} - Changes Over Time"
        )
        fig.update_traces(marker=dict(size=10), line=dict(width=3))
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
        st.plotly_chart(fig, width='stretch')

        # Ranking change table
        changes = []
        for name in current_df.iloc[:, 0]:
            curr_rank = current_df[current_df.iloc[:, 0] == name].index[0] + 1
            prev_rank = previous_df[previous_df.iloc[:, 0]
                                    == name].index[0] + 1
            delta = prev_rank - curr_rank
            changes.append({
                'Item': name,
                'Previous Rank': prev_rank,
                'Current Rank': curr_rank,
                'Change': f"↑ {abs(delta)}" if delta < 0 else f"↓ {delta}" if delta > 0 else "→ No change"
            })

        st.markdown("### 🔄 Ranking Movement")
        st.dataframe(pd.DataFrame(changes), width='stretch')

        st.markdown("""
        **When to use:** Showing how rankings shift between two time periods.
        
        **Key Features:** Clear upward/downward movement, great for performance reviews.
        """)

        st.code('''
fig = px.line(slope_df, x='period', y='value', color='item', markers=True)
fig.show()
        ''', language='python')

    def render_bullet_graph(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🎯 Bullet Graph - Performance vs Targets")

        value_col = st.selectbox("Performance Metric", df.select_dtypes(
            include=[np.number]).columns, index=1)

        sorted_df = df.sort_values(
            value_col, ascending=False).reset_index(drop=True)
        values = sorted_df[value_col].values
        min_val, max_val = values.min(), values.max()

        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.6)))

        for i, val in enumerate(values):
            y = len(df) - i - 1
            ranges = [min_val, min_val + (max_val - min_val) * 0.6,
                      min_val + (max_val - min_val) * 0.8, max_val]

            # Background ranges
            ax.barh(y, ranges[1] - ranges[0], left=ranges[0],
                    height=0.7, color='#ffcccc', alpha=0.7)
            ax.barh(y, ranges[2] - ranges[1], left=ranges[1],
                    height=0.7, color='#fff2cc', alpha=0.7)
            ax.barh(y, ranges[3] - ranges[2], left=ranges[2],
                    height=0.7, color='#ccffcc', alpha=0.7)

            # Performance bar
            ax.barh(y, val - min_val, left=min_val,
                    height=0.4, color='#2E86AB', alpha=0.9)

            # Target line (80th percentile)
            target = ranges[2]
            ax.plot([target, target], [y - 0.35, y + 0.35],
                    color='red', linewidth=3)

            # Label
            name = sorted_df.iloc[len(df)-1-i, 0]
            ax.text(max_val * 1.02, y, f"{name} ({val:,.1f})",
                    va='center', fontsize=10, fontweight='bold')

        ax.set_xlim(min_val, max_val * 1.15)
        ax.set_ylim(-0.5, len(df) - 0.5)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"#{i+1}" for i in range(len(df))])
        ax.set_xlabel(value_col)
        ax.set_title(f"{data_type} - Bullet Graph Performance Ranking")
        ax.grid(axis='x', alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** KPI dashboards, performance against benchmarks.
        
        **Key Features:** Compact, shows actual vs target + qualitative ranges.
        """)

    def render_step_chart(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🪜 Step Chart - Cumulative Contribution (Pareto)")

        value_col = st.selectbox("Value for Cumulative", df.select_dtypes(
            include=[np.number]).columns, index=1)

        sorted_df = df.sort_values(
            value_col, ascending=False).reset_index(drop=True)
        cumulative = sorted_df[value_col].cumsum()
        total = sorted_df[value_col].sum()
        cum_pct = cumulative / total * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Step chart
        ranks = list(range(1, len(sorted_df) + 1))
        ax1.step(ranks, cumulative, where='mid', linewidth=3, color='#f5576c')
        ax1.fill_between(ranks, cumulative, alpha=0.3, color='#f5576c')
        ax1.set_xlabel("Rank Position")
        ax1.set_ylabel(f"Cumulative {value_col}")
        ax1.set_title("Cumulative Contribution")
        ax1.grid(alpha=0.3)

        # Pareto
        ax2.bar(ranks, sorted_df[value_col] / total * 100,
                color='skyblue', alpha=0.7, label='Individual %')
        ax2.plot(ranks, cum_pct, color='red', marker='o',
                 linewidth=3, label='Cumulative %')
        ax2.axhline(80, color='orange', linestyle='--',
                    linewidth=2, label='80% Threshold')
        ax2.set_xlabel("Rank Position")
        ax2.set_ylabel("Percentage of Total")
        ax2.set_title("Pareto Analysis (80/20 Rule)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Highlight 80% point
        if (cum_pct >= 80).any():
            top_n = np.argmax(cum_pct >= 80) + 1
            ax2.axvline(top_n, color='green', linestyle=':', linewidth=2)

        st.pyplot(fig)
        plt.close(fig)

        if (cum_pct >= 80).any():
            top_n = np.argmax(cum_pct >= 80) + 1
            st.success(
                f"**Top {top_n} items account for 80% of total {value_col}**")

        st.markdown("""
        **When to use:** Identifying where most value is concentrated (Pareto principle).
        
        **Key Features:** Shows diminishing returns, great for prioritization.
        """)

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Ranking Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Clear Hierarchical Order**
            - Instantly see top and bottom performers
            - Natural ordering creates strong visual hierarchy
            """)
            st.markdown("""
            **Performance Tracking**
            - Track changes over time
            - Spot rising stars and declining items
            """)

        with col2:
            st.markdown("""
            **Comparative Analysis**
            - Easy magnitude and gap comparison
            - Benchmark against targets or peers
            """)
            st.markdown("""
            **Decision Support**
            - Reward top performers
            - Focus improvement efforts
            - Allocate resources effectively
            """)

    def render_examples(self, dataset_type: str, df: pd.DataFrame):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Sales Leaderboard": "Top performing sales representatives",
            "Product Comparison": "Customer satisfaction ratings across products",
            "Sports Statistics": "Team rankings in a league",
            "College Rankings": "Universities ordered by various metrics",
            "Employee Performance": "Staff evaluation scores",
            "Website Analytics": "Traffic ranking across websites"
        }

        for example, desc in examples.items():
            with st.expander(f"🏆 {example}"):
                st.write(desc)
                if dataset_type == example:
                    top_name = df.iloc[0, 0]
                    top_val = df.iloc[0, 1]
                    st.success(f"**Current #1:** {top_name} — {top_val:,.0f}")

    def render_data_table(self, df: pd.DataFrame):
        st.markdown("### 📊 Current Ranking Data")
        st.dataframe(df.sort_values(df.columns[1], ascending=False).reset_index(
            drop=True), width='stretch')

    def output(self):
        self.render_header()
        chart_type, dataset_type, num_items, show_comparison = self.render_configuration()

        current_df = self.generate_ranking_data(dataset_type, num_items)
        previous_df = self.generate_previous_data(current_df, dataset_type)

        chart_map = {
            "Bar Chart": lambda: self.render_bar_chart(current_df, dataset_type),
            "Slope Chart": lambda: self.render_slope_chart(current_df, previous_df, dataset_type, show_comparison),
            "Bullet Graph": lambda: self.render_bullet_graph(current_df, dataset_type),
            "Step Chart": lambda: self.render_step_chart(current_df, dataset_type)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_key_characteristics()
        self.render_examples(dataset_type, current_df)
        self.render_data_table(current_df)
