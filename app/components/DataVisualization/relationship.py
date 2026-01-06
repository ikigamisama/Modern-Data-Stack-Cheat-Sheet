import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


class Relationship:
    def __init__(self):
        self.title = "🔗 Relationship Visualization Dashboard"
        self.chart_types = ["Scatter Plot",
                            "Bubble Plot", "Matrix Plot", "Pair Plot"]
        self.dataset_types = [
            "Marketing Analysis", "Health Studies", "Education",
            "Economics", "Real Estate", "Customer Behavior"
        ]
        self.sample_size_range = (100, 2000, 500)

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Relationship charts uncover connections, correlations, and associations between two or more variables. 
        They help identify whether variables move together, oppose each other, or remain independent.
        """)

    def render_configuration(self):
        col1, col2, col3 = st.columns(3)

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            dataset_type = st.selectbox(
                "Dataset Type", self.dataset_types, key="dataset_type")

        with col3:
            sample_size = st.slider(
                "Sample Size", *self.sample_size_range, key="sample_size")
            show_correlation = st.checkbox(
                "Show Correlation Metrics", value=True, key="show_corr")

        return chart_type, dataset_type, sample_size, show_correlation

    @staticmethod
    def generate_relationship_data(data_type: str, n_points: int) -> pd.DataFrame:
        np.random.seed(42)

        if data_type == "Marketing Analysis":
            ad_spend = np.random.uniform(1000, 10000, n_points)
            sales = 50 + 0.8 * ad_spend + np.random.normal(0, 1000, n_points)
            customer_satisfaction = np.random.uniform(1, 10, n_points)
            return pd.DataFrame({
                'ad_spend': ad_spend,
                'sales': sales,
                'customer_satisfaction': customer_satisfaction,
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_points)
            })

        elif data_type == "Health Studies":
            exercise_hours = np.random.uniform(0, 10, n_points)
            health_score = 80 + 2 * exercise_hours + \
                np.random.normal(0, 10, n_points)
            age = np.random.randint(20, 70, n_points)
            bmi = np.random.uniform(18, 35, n_points)
            return pd.DataFrame({
                'exercise_hours': exercise_hours,
                'health_score': health_score,
                'age': age,
                'bmi': bmi,
                'gender': np.random.choice(['Male', 'Female'], n_points)
            })

        elif data_type == "Education":
            study_hours = np.random.uniform(1, 20, n_points)
            exam_score = 50 + 2 * study_hours + \
                np.random.normal(0, 10, n_points)
            attendance = np.random.uniform(60, 100, n_points)
            prev_grades = np.random.uniform(60, 95, n_points)
            return pd.DataFrame({
                'study_hours': study_hours,
                'exam_score': exam_score,
                'attendance': attendance,
                'prev_grades': prev_grades,
                'major': np.random.choice(['Science', 'Arts', 'Business'], n_points)
            })

        elif data_type == "Economics":
            unemployment = np.random.uniform(3, 12, n_points)
            inflation = 2 + 0.3 * unemployment + 0.1 * \
                unemployment**2 + np.random.normal(0, 0.5, n_points)
            gdp_growth = np.random.uniform(-2, 5, n_points)
            interest_rates = np.random.uniform(0.5, 5, n_points)
            return pd.DataFrame({
                'unemployment': unemployment,
                'inflation': inflation,
                'gdp_growth': gdp_growth,
                'interest_rates': interest_rates,
                'country': np.random.choice(['US', 'UK', 'Germany', 'Japan'], n_points)
            })

        elif data_type == "Real Estate":
            sq_ft = np.random.uniform(500, 3000, n_points)
            price = 50000 + 150 * sq_ft + np.random.normal(0, 20000, n_points)
            bedrooms = np.random.randint(1, 5, n_points)
            bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], n_points)
            year_built = np.random.randint(1950, 2023, n_points)
            return pd.DataFrame({
                'sq_ft': sq_ft,
                'price': price,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'year_built': year_built,
                'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_points)
            })

        else:  # Customer Behavior
            time_on_site = np.random.uniform(1, 60, n_points)
            purchase_amount = 10 + 0.5 * time_on_site + \
                np.random.normal(0, 20, n_points)
            pages_visited = np.random.randint(1, 20, n_points)
            customer_age = np.random.randint(18, 70, n_points)
            return pd.DataFrame({
                'time_on_site': time_on_site,
                'purchase_amount': purchase_amount,
                'pages_visited': pages_visited,
                'customer_age': customer_age,
                'device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_points)
            })

    @staticmethod
    def calculate_correlations(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].corr()

    def render_scatter_plot(self, df: pd.DataFrame, data_type: str, show_correlation: bool):
        st.markdown("### 📊 Scatter Plot - Bivariate Relationships")

        col1, col2 = st.columns(2)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        with col1:
            x_var = st.selectbox("X Variable", numeric_cols, index=0)
        with col2:
            y_var = st.selectbox("Y Variable", numeric_cols,
                                 index=1 if len(numeric_cols) > 1 else 0)

        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns
        color_var = st.selectbox("Color by", ['None'] + list(categorical_cols))

        fig = px.scatter(
            df, x=x_var, y=y_var,
            color=color_var if color_var != 'None' else None,
            title=f"{data_type} - {x_var} vs {y_var}",
            trendline="ols" if show_correlation else None
        )

        st.plotly_chart(fig, width='stretch')

        if show_correlation:
            corr = df[x_var].corr(df[y_var])
            strength = (
                "Strong Positive" if corr > 0.7 else
                "Moderate Positive" if corr > 0.3 else
                "Weak Positive" if corr > 0 else
                "Weak Negative" if corr > -0.3 else
                "Moderate Negative" if corr > -0.7 else
                "Strong Negative"
            )
            st.success(
                f"**Correlation Coefficient:** {corr:.3f} → **{strength} Relationship**")

        st.markdown("""
        **When to use:** Exploring relationships between two continuous variables, detecting trends, outliers.
        
        **Key Features:** Interactive, supports trendlines, grouping by category.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import plotly.express as px
                
fig = px.scatter(df, x='x_column', y='y_column', color='category',
                 trendline='ols', title='Scatter Plot')
fig.show()
        ''', language='python')

    def render_bubble_plot(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🎈 Bubble Plot - Multivariate Relationships")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        col1, col2, col3 = st.columns(3)

        with col1:
            x_var = st.selectbox("X Variable", numeric_cols,
                                 index=0, key="bubble_x")
        with col2:
            y_var = st.selectbox("Y Variable", numeric_cols, index=1 if len(
                numeric_cols) > 1 else 0, key="bubble_y")
        with col3:
            size_var = st.selectbox(
                "Bubble Size", numeric_cols, index=2 if len(numeric_cols) > 2 else 0)

        categorical_cols = df.select_dtypes(include=['object']).columns
        color_var = st.selectbox(
            "Color by", ['None'] + list(categorical_cols), key="bubble_color")

        fig = px.scatter(
            df, x=x_var, y=y_var, size=size_var,
            color=color_var if color_var != 'None' else None,
            hover_data=df.columns,
            title=f"{data_type} - Bubble Plot"
        )

        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        **When to use:** Visualizing three (or four) variables at once — two on axes, one as size, one as color.
        
        **Key Features:** Rich multivariate insight, interactive hover details.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import plotly.express as px
                
fig = px.scatter(df, x='x', y='y', size='size_var', color='category',
                 hover_data=['extra_info'], title='Bubble Plot')
fig.show()
        ''', language='python')

    def render_matrix_plot(self, df: pd.DataFrame, data_type: str, corr_matrix: pd.DataFrame):
        st.markdown("### 🔢 Matrix Plot - Correlation Overview")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title(f"{data_type} - Correlation Matrix Heatmap")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** Quick overview of all pairwise correlations in a dataset.
        
        **Key Features:** Color intensity shows strength and direction of relationships.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import seaborn as sns
import matplotlib.pyplot as plt
                
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0)
plt.title('Correlation Matrix')
plt.show()
        ''', language='python')

    def render_pair_plot(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 📈 Pair Plot - Multi-dimensional Relationships")

        numeric_cols = df.select_dtypes(
            include=[np.number]).columns[:5]  # Limit for performance
        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns

        hue_var = st.selectbox("Color groups by (Hue)", [
                               'None'] + list(categorical_cols))
        hue = hue_var if hue_var != 'None' else None

        g = sns.pairplot(
            df[numeric_cols.tolist() + ([hue_var] if hue else [])],
            hue=hue,
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 50}
        )

        g.fig.suptitle(f"{data_type} - Pair Plot", y=1.02)

        # Pass the actual figure object to Streamlit
        st.pyplot(g.fig)

        # Close the figure properly
        plt.close(g.fig)

        st.markdown("""
        **When to use:** Exploratory analysis of multiple variables, spotting clusters and trends.
        
        **Key Features:** All pairwise scatter plots + distributions on diagonal.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import seaborn as sns
import matplotlib.pyplot as plt
                
sns.pairplot(df, hue='category', diag_kind='kde', plot_kws={'alpha': 0.6})
plt.show()
        ''', language='python')

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Relationship Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Correlation Patterns**
            - Positive, negative, or no correlation
            - Linear vs non-linear trends
            - Strength of association
            """)
            st.markdown("""
            **Outlier Detection**
            - Identifies anomalies
            - Shows data dispersion
            - Reveals unusual patterns
            """)

        with col2:
            st.markdown("""
            **Predictive Insights**
            - Foundation for regression models
            - Supports hypothesis testing
            - Guides feature selection
            """)
            st.markdown("""
            **Multi-variable Analysis**
            - Handles interactions between factors
            - Comprehensive exploration
            - Reveals hidden relationships
            """)

    def render_examples(self, chart_type: str, dataset_type: str):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Marketing Analysis": "Relationship between advertising spend and sales",
            "Health Studies": "Correlation between exercise frequency and health outcomes",
            "Education": "Connection between study hours and exam performance",
            "Economics": "Relationship between unemployment and inflation rates",
            "Real Estate": "How square footage relates to property price",
            "Customer Behavior": "Time spent on site vs purchase amount"
        }

        for example, description in examples.items():
            with st.expander(f"📊 {example}"):
                st.write(description)
                if dataset_type == example:
                    st.success(f"**Current Chart Type:** {chart_type}")
                    st.info("This dataset is currently active in your view.")

    def render_key_characteristics(self):
        st.markdown("### 🔗 Understanding Relationships in Data")
        st.markdown("""
        Analyzing relationships between variables is key to uncovering **how one factor
        influences another**. This analysis helps identify patterns, dependencies, and
        predictive potential.
        """)

        st.markdown("#### 📈 Showing Positive or Negative Correlations")
        st.markdown("""
        - **Positive correlation:** Both variables move in the same direction  
        - **Negative correlation:** Variables move in opposite directions  

        Example applications:
        - Marketing spend ↔ Sales revenue  
        - Temperature ↔ Heating usage
        """)

        st.markdown("#### 📐 Identifying Linear or Non-linear Relationships")
        st.markdown("""
        Linear relationships show proportional change, while non-linear relationships
        exhibit varying rates of change. Detecting the correct type is crucial for:
        - Accurate modeling  
        - Forecasting  
        - Hypothesis testing
        """)

        st.markdown("#### ⚠️ Revealing Outliers and Anomalies")
        st.markdown("""
        Points that deviate from expected patterns are outliers. These can indicate:
        - Data errors  
        - Rare events  
        - Hidden opportunities  

        Outlier detection ensures robust analysis and predictive modeling.
        """)

        st.markdown("#### 🤖 Supporting Predictive Modeling")
        st.markdown("""
        Relationships form the foundation of predictive analytics. Understanding how
        variables interact improves:
        - Feature selection  
        - Model accuracy  
        - Actionable insights
        """)

        st.divider()

        st.markdown("#### 🎯 Why Relationship Analysis Matters")
        st.markdown("""
        Exploring relationships turns raw data into meaningful knowledge. This enables:
        - Better operational decisions  
        - Outcome anticipation  
        - Evidence-based strategies
        """)

    def output(self):
        self.render_header()
        chart_type, dataset_type, sample_size, show_correlation = self.render_configuration()

        df = self.generate_relationship_data(dataset_type, sample_size)
        corr_matrix = self.calculate_correlations(df)

        chart_map = {
            "Scatter Plot": lambda: self.render_scatter_plot(df, dataset_type, show_correlation),
            "Bubble Plot": lambda: self.render_bubble_plot(df, dataset_type),
            "Matrix Plot": lambda: self.render_matrix_plot(df, dataset_type, corr_matrix),
            "Pair Plot": lambda: self.render_pair_plot(df, dataset_type)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_examples(chart_type, dataset_type)
        self.render_key_characteristics()
