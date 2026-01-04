import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class PartToWhole:
    def __init__(self):
        self.title = "📊 Part-to-Whole Visualization Dashboard"
        self.chart_types = ["Stacked Bar Chart",
                            "Nested Bar Chart", "Waterfall Chart"]
        self.dataset_types = [
            "Financial Reporting", "Survey Results", "Project Budget",
            "Time Tracking", "Market Segmentation", "Portfolio Allocation"
        ]

    def render_header(self):
        st.markdown("## {self.title}")
        st.markdown("""
        ### Purpose
        Part-to-whole charts emphasize how individual segments contribute to a complete total, 
        making proportions and relative sizes immediately apparent. They answer 
        "how much of the total does each part represent?"
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
            num_categories = st.slider(
                "Number of Categories", 3, 10, 5, key="num_cats")
            show_percentages = st.checkbox(
                "Show Percentages", value=True, key="show_pct")

        return chart_type, dataset_type, num_categories, show_percentages

    def generate_partwhole_data(self, data_type: str, n_categories: int) -> pd.DataFrame:
        np.random.seed(42)

        category_lists = {
            "Financial Reporting": ['Salaries', 'Marketing', 'Rent', 'Utilities', 'Software', 'Travel', 'Training'],
            "Survey Results": ['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied'],
            "Project Budget": ['Design', 'Development', 'Testing', 'Deployment', 'Training', 'Maintenance'],
            "Time Tracking": ['Meetings', 'Coding', 'Testing', 'Documentation', 'Research', 'Break'],
            "Market Segmentation": ['Age 18-25', 'Age 26-35', 'Age 36-45', 'Age 46-55', 'Age 55+'],
            "Portfolio Allocation": ['Stocks', 'Bonds', 'Real Estate', 'Cash', 'Commodities', 'Crypto']
        }

        categories = category_lists[data_type][:n_categories]

        if data_type in ["Financial Reporting", "Project Budget", "Portfolio Allocation"]:
            values = np.random.randint(10000, 50000, n_categories)
        elif data_type == "Time Tracking":
            values = np.random.randint(5, 40, n_categories)
        else:
            values = np.random.randint(50, 500, n_categories)

        total = values.sum()
        percentages = values / total * 100

        col_name = {
            "Financial Reporting": "Amount",
            "Survey Results": "Count",
            "Project Budget": "Cost",
            "Time Tracking": "Hours",
            "Market Segmentation": "Customers",
            "Portfolio Allocation": "Value"
        }[data_type]

        cat_col = {
            "Financial Reporting": "Category",
            "Survey Results": "Response",
            "Project Budget": "Phase",
            "Time Tracking": "Activity",
            "Market Segmentation": "Segment",
            "Portfolio Allocation": "Asset"
        }[data_type]

        return pd.DataFrame({
            cat_col: categories,
            col_name: values,
            'Percentage': percentages
        })

    def render_stacked_bar_chart(self, df: pd.DataFrame, data_type: str, show_percentages: bool):
        st.markdown("### 📊 Stacked Bar Chart - Temporal Composition")

        periods = ['Q1 2025', 'Q2 2025', 'Q3 2025', 'Q4 2025']
        time_data = []

        cat_col = df.columns[0]
        val_col = df.columns[1]

        base_values = df[val_col].values

        for period in periods:
            variation = np.random.uniform(0.85, 1.15, len(df))
            period_vals = (base_values * variation).astype(int)
            period_total = period_vals.sum()

            for cat, val in zip(df[cat_col], period_vals):
                time_data.append({
                    'Period': period,
                    'Category': cat,
                    'Value': val,
                    'Percentage': val / period_total * 100
                })

        time_df = pd.DataFrame(time_data)

        fig = px.bar(
            time_df, x='Period', y='Value', color='Category',
            title=f"{data_type} - Quarterly Stacked Breakdown",
            color_discrete_sequence=px.colors.qualitative.Bold,
            height=600
        )

        if show_percentages:
            for period in periods:
                period_data = time_df[time_df['Period']
                                      == period].sort_values('Value')
                cumulative = 0
                for _, row in period_data.iterrows():
                    fig.add_annotation(
                        x=period,
                        y=cumulative + row['Value'] / 2,
                        text=f"{row['Percentage']:.1f}%",
                        showarrow=False,
                        font=dict(color="white", size=10, weight="bold"),
                        align="center"
                    )
                    cumulative += row['Value']

        st.plotly_chart(fig, width='stretch')

        total = df[val_col].sum()
        st.success(f"**Current Period Total:** {total:,} {val_col.lower()}")

        st.markdown("**Current Breakdown:**")
        for _, row in df.iterrows():
            pct = row['Percentage']
            st.markdown(f"**{row[cat_col]}**: {row[val_col]:,} ({pct:.1f}%)")
            st.progress(pct / 100)

        st.markdown("""
        **When to use:** Comparing composition across time periods while preserving the total.
        
        **Key Features:** Easy to see shifts in proportion over time.
        """)

    def render_nested_bar_chart(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 📈 Nested Bar Chart - Hierarchical Breakdown")

        cat_col = df.columns[0]
        val_col = df.columns[1]
        total = df[val_col].sum()

        nested_data = []
        for _, row in df.iterrows():
            main_cat = row[cat_col]
            main_val = row[val_col]
            n_subs = np.random.randint(2, 4)
            sub_names = [
                f"{main_cat} - Sub {chr(65+i)}" for i in range(n_subs)]

            sub_vals = np.random.randint(
                main_val // (n_subs + 1), main_val // n_subs * 2, n_subs)
            sub_vals = np.append(sub_vals, main_val -
                                 sub_vals.sum())  # ensure exact total

            for sub_name, sub_val in zip(sub_names, sub_vals):
                nested_data.append({
                    'Main Category': main_cat,
                    'Sub Category': sub_name,
                    'Value': max(0, sub_val),
                    'Pct of Main': sub_val / main_val * 100,
                    'Pct of Total': sub_val / total * 100
                })

        nested_df = pd.DataFrame(nested_data)

        fig = px.bar(
            nested_df,
            x='Main Category',
            y='Value',
            color='Sub Category',
            title=f"{data_type} - Hierarchical Nested View",
            barmode='stack',
            height=600
        )

        st.plotly_chart(fig, width='stretch')

        st.success(f"**Grand Total:** {total:,}")

        for main_cat in df[cat_col]:
            main_total = df[df[cat_col] == main_cat][val_col].iloc[0]
            sub_data = nested_df[nested_df['Main Category'] == main_cat]

            with st.expander(f"🔽 {main_cat} ({main_total:,} — {main_total/total*100:.1f}% of total)"):
                for _, row in sub_data.iterrows():
                    st.write(
                        f"• **{row['Sub Category']}**: {row['Value']:,} ({row['Pct of Main']:.1f}% of {main_cat})")

        st.markdown("""
        **When to use:** Showing multi-level hierarchy within a total (e.g., departments → teams).
        
        **Key Features:** Drill-down structure, preserves both part-to-main and part-to-total proportions.
        """)

    def render_waterfall_chart(self, df: pd.DataFrame, data_type: str, show_percentages: bool):
        st.markdown("### 💧 Waterfall Chart - Cumulative Build-Up")

        cat_col = df.columns[0]
        val_col = df.columns[1]
        total = df[val_col].sum()

        categories = df[cat_col].tolist()
        values = df[val_col].tolist()

        # Waterfall structure
        measures = ["absolute"] + ["relative"] * len(categories) + ["total"]
        x_labels = ["Start"] + categories + ["Total"]
        y_values = [0] + values + [total]

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_values,
            textposition="outside",
            text=["" if i == 0 or i == len(
                x_labels)-1 else f"+{v:,}" for i, v in enumerate(y_values)],
            connector=dict(line=dict(color="rgb(63, 63, 63)", width=2)),
        ))

        fig.update_layout(
            title=f"{data_type} - How Components Build the Total",
            showlegend=False,
            height=600,
            xaxis_title="",
            yaxis_title=val_col
        )

        if show_percentages:
            cumulative = 0
            for i, val in enumerate(values):
                cumulative += val
                pct = cumulative / total * 100
                fig.add_annotation(
                    x=categories[i],
                    y=cumulative - val / 2,
                    text=f"{pct:.1f}%",
                    showarrow=False,
                    font=dict(color="black", size=11),
                    bgcolor="rgba(255,255,255,0.7)"
                )

        st.plotly_chart(fig, width='stretch')

        st.success(f"**Final Total: {total:,}**")

        cumulative = 0
        st.markdown("**Step-by-Step Accumulation:**")
        for cat, val in zip(categories, values):
            cumulative += val
            pct = cumulative / total * 100
            st.markdown(
                f"**+ {cat}** → {val:,} → **Running: {cumulative:,} ({pct:.1f}% of total)**")

        st.markdown("""
        **When to use:** Explaining how individual positive contributions build up to a final total.
        
        **Key Features:** Shows incremental addition, clear path from zero to total.
        """)

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Part-to-Whole Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Proportional Clarity**
            - Instantly see relative contribution
            - Parts always sum to meaningful whole
            """)
            st.markdown("""
            **Hierarchical Support**
            - Handles nested breakdowns
            - Shows both sub- and grand totals
            """)

        with col2:
            st.markdown("""
            **Temporal Comparison**
            - Track changing composition over time
            - Spot rising/falling segments
            """)
            st.markdown("""
            **Cumulative Insight**
            - Waterfall shows build-up sequence
            - Highlights key contributors
            """)

    def render_examples(self, dataset_type: str, df: pd.DataFrame):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Financial Reporting": "Operating expense breakdown by department",
            "Survey Results": "Response distribution across satisfaction levels",
            "Project Budget": "Cost allocation across project phases",
            "Time Tracking": "Daily hours spent on different activities",
            "Market Segmentation": "Customer base by age groups",
            "Portfolio Allocation": "Investment distribution across asset classes"
        }

        for example, desc in examples.items():
            with st.expander(f"📊 {example}"):
                st.write(desc)
                if dataset_type == example:
                    total = df.iloc[:, 1].sum()
                    unit = df.columns[1]
                    st.success(f"**Current Total:** {total:,} {unit}")

    def render_data_table(self, df: pd.DataFrame):
        st.markdown("### 📋 Current Part-to-Whole Data")
        st.dataframe(df.sort_values('Percentage', ascending=False),
                     width='stretch')

    def output(self):
        self.render_header()
        chart_type, dataset_type, num_categories, show_percentages = self.render_configuration()

        df = self.generate_partwhole_data(dataset_type, num_categories)

        chart_map = {
            "Stacked Bar Chart": lambda: self.render_stacked_bar_chart(df, dataset_type, show_percentages),
            "Nested Bar Chart": lambda: self.render_nested_bar_chart(df, dataset_type),
            "Waterfall Chart": lambda: self.render_waterfall_chart(df, dataset_type, show_percentages)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_key_characteristics()
        self.render_examples(dataset_type, df)
        self.render_data_table(df)
