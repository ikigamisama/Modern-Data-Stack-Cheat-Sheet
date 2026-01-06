import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy.stats as stats


class Statistical:
    def __init__(self):
        self.title = "📈 Statistical Visualizations Dashboard"
        self.chart_types = [
            "Box Plot",
            "Density Plot (KDE)",
            "Q-Q Plot",
            "Normal Distribution"
        ]
        self.scenarios = [
            "Treatment Effect in Medical Trial",
            "Process Measurement in Manufacturing",
            "Survey Response Scores",
            "Exam Results Across Classes",
            "Revenue per Customer Segment"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Statistical charts display **probability distributions, confidence intervals, and other statistical concepts**. 
        They support rigorous data analysis and hypothesis testing.
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            scenario = st.selectbox("Scenario", self.scenarios, key="scenario")

        with col3:
            num_samples = st.slider(
                "Number of Samples", 200, 2000, 1000, step=200)

        return chart_type, scenario, num_samples

    def generate_data(self, scenario: str, n: int):
        np.random.seed(42)

        if scenario == "Treatment Effect in Medical Trial":
            # Control group vs Treatment group
            control = np.random.normal(50, 10, n)
            # Higher mean, more variance
            treatment = np.random.normal(58, 12, n)
            df = pd.DataFrame({
                "Response": np.concatenate([control, treatment]),
                "Group": ["Control"] * n + ["Treatment"] * n
            })
            title = "Patient Response Score"

        elif scenario == "Process Measurement in Manufacturing":
            # Normal process + occasional shift
            data = np.random.normal(100, 5, n)
            data[::20] += 15  # Outliers
            df = pd.DataFrame({"Measurement": data})
            title = "Product Dimension (mm)"

        elif scenario == "Survey Response Scores":
            # Skewed data
            data = np.random.beta(2, 5, n) * 100
            df = pd.DataFrame({"Score": data})
            title = "Satisfaction Score (0-100)"

        elif scenario == "Exam Results Across Classes":
            classes = ["Class A", "Class B", "Class C", "Class D"]
            data = []
            labels = []
            for i, cls in enumerate(classes):
                mean = [75, 82, 68, 90][i]
                std = [12, 10, 15, 8][i]
                samples = np.random.normal(mean, std, n//4)
                data.extend(samples)
                labels.extend([cls] * (n//4))
            df = pd.DataFrame({"Score": data, "Class": labels})
            title = "Exam Score (%)"

        else:  # Revenue per Customer Segment
            # Log-normal
            data = np.random.lognormal(mean=5, sigma=0.8, size=n)
            df = pd.DataFrame({"Revenue": data})
            title = "Revenue per Customer ($)"

        return df, title

    def create_box_plot(self, df: pd.DataFrame, title: str):
        if "Group" in df.columns or "Class" in df.columns:
            cat_col = "Group" if "Group" in df.columns else "Class"
            fig = go.Figure()
            for cat in df[cat_col].unique():
                fig.add_trace(go.Box(
                    y=df[df[cat_col] == cat][df.columns[0]],
                    name=cat,
                    boxpoints='outliers'
                ))
        else:
            fig = go.Figure(go.Box(y=df.iloc[:, 0], boxpoints='outliers'))

        fig.update_layout(
            title=f"Box Plot – {title}<br><sub>Shows median, quartiles, and outliers</sub>",
            yaxis_title=title,
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
if "Group" in df.columns or "Class" in df.columns:
    cat_col = "Group" if "Group" in df.columns else "Class"
    fig = go.Figure()
    for cat in df[cat_col].unique():
        fig.add_trace(go.Box(
            y=df[df[cat_col] == cat][df.columns[0]],
            name=cat,
            boxpoints='outliers'
        ))
else:
    fig = go.Figure(go.Box(y=df.iloc[:, 0], boxpoints='outliers'))

fig.update_layout(
    title=f"Box Plot – {title}<br><sub>Shows median, quartiles, and outliers</sub>",
    yaxis_title=title,
    height=600
)
fig.show()
""", language="python")

    def create_density_plot(self, df: pd.DataFrame, title: str):
        fig = go.Figure()
        value_col = df.columns[0]
        if len(df.columns) > 1:  # Categorical
            cat_col = df.columns[1]
            for cat in df[cat_col].unique():
                subset = df[df[cat_col] == cat][value_col]
                fig.add_trace(go.Violin(
                    y=subset,
                    name=cat,
                    side='positive' if "Control" in cat else 'negative',
                    box_visible=True,
                    meanline_visible=True
                ))
        else:
            # KDE using histogram + density
            x = df[value_col]
            kde = stats.gaussian_kde(x)
            x_range = np.linspace(x.min(), x.max(), 200)
            density = kde(x_range)

            fig.add_trace(go.Scatter(
                x=x_range,
                y=density,
                fill='tozeroy',
                mode='lines',
                line=dict(color='#3498db'),
                name='KDE'
            ))

        fig.update_layout(
            title=f"Density Plot (KDE) – {title}",
            yaxis_title="Density" if len(df.columns) == 1 else title,
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
fig = go.Figure()
value_col = df.columns[0]
if len(df.columns) > 1:  # Categorical
    cat_col = df.columns[1]
    for cat in df[cat_col].unique():
        subset = df[df[cat_col] == cat][value_col]
        fig.add_trace(go.Violin(
            y=subset,
            name=cat,
            side='positive' if "Control" in cat else 'negative',
            box_visible=True,
            meanline_visible=True
        ))
else:
    # KDE using histogram + density
    x = df[value_col]
    kde = stats.gaussian_kde(x)
    x_range = np.linspace(x.min(), x.max(), 200)
    density = kde(x_range)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=density,
        fill='tozeroy',
        mode='lines',
        line=dict(color='#3498db'),
        name='KDE'
    ))

fig.update_layout(
    title=f"Density Plot (KDE) – {title}",
    yaxis_title="Density" if len(df.columns) == 1 else title,
    height=600
)
fig.show()
""", language="python")

    def create_qq_plot(self, df: pd.DataFrame, title: str):
        value_col = df.columns[0]

        # Clean numeric data
        data = (
            df[value_col]
            .dropna()
            .astype(float)
            .to_numpy()
        )

        if data.size < 2:
            raise ValueError("Q-Q plot requires at least 2 data points")

        # ---- SciPy (defensive handling) ----
        result = stats.probplot(data, dist="norm", fit=False)

        theoretical = np.asarray(result[0][0]).ravel()
        sample = np.asarray(result[0][1]).ravel()
        # -----------------------------------

        fig = go.Figure()

        # Q-Q points
        fig.add_trace(go.Scatter(
            x=theoretical.tolist(),
            y=sample.tolist(),
            mode="markers",
            marker=dict(color="#3498db"),
            name="Data Quantiles"
        ))

        # 45-degree reference line
        line_min = min(theoretical.min(), sample.min())
        line_max = max(theoretical.max(), sample.max())

        fig.add_trace(go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Normal Line"
        ))

        fig.update_layout(
            title=f"Q-Q Plot – {title}<br><sub>Deviation from line indicates non-normality</sub>",
            xaxis_title="Theoretical Quantiles (Normal)",
            yaxis_title="Sample Quantiles",
            height=600,
            showlegend=True
        )

        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
value_col = df.columns[0]

# Clean numeric data
data = (
    df[value_col]
    .dropna()
    .astype(float)
    .to_numpy()
)

if data.size < 2:
    raise ValueError("Q-Q plot requires at least 2 data points")

# ---- SciPy (defensive handling) ----
result = stats.probplot(data, dist="norm", fit=False)

theoretical = np.asarray(result[0][0]).ravel()
sample = np.asarray(result[0][1]).ravel()
# -----------------------------------

fig = go.Figure()

# Q-Q points
fig.add_trace(go.Scatter(
    x=theoretical.tolist(),
    y=sample.tolist(),
    mode="markers",
    marker=dict(color="#3498db"),
    name="Data Quantiles"
))

# 45-degree reference line
line_min = min(theoretical.min(), sample.min())
line_max = max(theoretical.max(), sample.max())

fig.add_trace(go.Scatter(
    x=[line_min, line_max],
    y=[line_min, line_max],
    mode="lines",
    line=dict(color="red", dash="dash"),
    name="Normal Line"
))

fig.update_layout(
    title=f"Q-Q Plot – {title}<br><sub>Deviation from line indicates non-normality</sub>",
    xaxis_title="Theoretical Quantiles (Normal)",
    yaxis_title="Sample Quantiles",
    height=600,
    showlegend=True
)
fig.show()
""", language="python")

    def create_normal_distribution(self, df: pd.DataFrame, title: str):
        value_col = df.columns[0]
        data = df[value_col]
        mean = data.mean()
        std = data.std()

        x = np.linspace(data.min(), data.max(), 200)
        normal_pdf = stats.norm.pdf(x, mean, std)

        fig = go.Figure()

        # Histogram + normal curve
        fig.add_trace(go.Histogram(
            x=data,
            histnorm='probability density',
            name='Data',
            marker_color='#3498db',
            opacity=0.6
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=normal_pdf,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#e74c3c', width=3)
        ))

        # Shade 95% confidence interval
        ci_low = mean - 1.96 * std
        ci_high = mean + 1.96 * std
        x_ci = np.linspace(ci_low, ci_high, 100)
        y_ci = stats.norm.pdf(x_ci, mean, std)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_ci, x_ci[::-1]]),
            y=np.concatenate([y_ci, [0]*len(y_ci)]),
            fill='toself',
            fillcolor='rgba(231,76,60,0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Interval'
        ))

        fig.update_layout(
            title=f"Normal Distribution Overlay – {title}<br><sub>Mean: {mean:.1f} | Std: {std:.1f}</sub>",
            xaxis_title=title,
            yaxis_title="Probability Density",
            height=600
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")
        st.code("""import plotly.graph_objects as go
import pandas as pd
                
value_col = df.columns[0]
data = df[value_col]
mean = data.mean()
std = data.std()

x = np.linspace(data.min(), data.max(), 200)
normal_pdf = stats.norm.pdf(x, mean, std)

fig = go.Figure()

# Histogram + normal curve
fig.add_trace(go.Histogram(
    x=data,
    histnorm='probability density',
    name='Data',
    marker_color='#3498db',
    opacity=0.6
))

fig.add_trace(go.Scatter(
    x=x,
    y=normal_pdf,
    mode='lines',
    name='Normal Distribution',
    line=dict(color='#e74c3c', width=3)
))

# Shade 95% confidence interval
ci_low = mean - 1.96 * std
ci_high = mean + 1.96 * std
x_ci = np.linspace(ci_low, ci_high, 100)
y_ci = stats.norm.pdf(x_ci, mean, std)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_ci, x_ci[::-1]]),
    y=np.concatenate([y_ci, [0]*len(y_ci)]),
    fill='toself',
    fillcolor='rgba(231,76,60,0.3)',
    line=dict(color='rgba(255,255,255,0)'),
    name='95% Interval'
))

fig.update_layout(
    title=f"Normal Distribution Overlay – {title}<br><sub>Mean: {mean:.1f} | Std: {std:.1f}</sub>",
    xaxis_title=title,
    yaxis_title="Probability Density",
    height=600
)
fig.show()
""", language="python")

    def render_chart(self, chart_type: str, scenario: str, n_samples: int):
        st.markdown(f"### {chart_type}: {scenario}")

        df, title = self.generate_data(scenario, n_samples)

        if chart_type == "Box Plot":
            self.create_box_plot(df, title)
        elif chart_type == "Density Plot (KDE)":
            self.create_density_plot(df, title)
        elif chart_type == "Q-Q Plot":
            self.create_qq_plot(df, title)
        else:  # Normal Distribution
            self.create_normal_distribution(df, title)

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Research Studies": "Confidence intervals for experimental results",
            "Quality Control": "Process capability analysis",
            "Medical Trials": "Treatment effect distributions",
            "Survey Research": "Margin of error visualization",
            "Risk Analysis": "Probability distribution of outcomes"
        }

        for example, description in examples.items():
            with st.expander(f"💭 {example}"):
                st.write(description)

    def render_key_characteristics(self):
        st.markdown("### 📐 Understanding Statistical Analysis")

        st.markdown("""
        Statistical analysis examines **data using formal statistical methods**.
        It focuses on variability, uncertainty, and inference to support rigorous decision-making.
        """)

        st.markdown("#### 📊 Incorporates Statistical Concepts")
        st.markdown("""
        Leverages fundamental concepts such as:
        - Mean, median, and variance  
        - Probability distributions  
        - Correlations and hypothesis testing  
        Ensures analysis is grounded in sound methodology.
        """)

        st.markdown("#### ⚖️ Shows Uncertainty and Variability")
        st.markdown("""
        Highlights the range and reliability of data:
        - Confidence intervals  
        - Error margins  
        - Variability across samples or groups  
        Helps avoid overinterpretation of results.
        """)

        st.markdown("#### 🎯 Supports Inferential Reasoning")
        st.markdown("""
        Enables conclusions beyond the observed data:
        - Hypothesis testing  
        - Predictive modeling  
        - Generalizing insights to larger populations  
        Facilitates evidence-based decision making.
        """)

        st.markdown("#### 🧪 Enables Rigorous Analysis")
        st.markdown("""
        Provides structured and reproducible approaches:
        - Ensures statistical validity  
        - Supports experimentation and validation  
        - Guides critical evaluation of results
        """)

        st.divider()

        st.markdown("#### 🎯 Why Statistical Analysis Matters")
        st.markdown("""
        Statistical analysis is essential for:
        - Making informed, data-driven decisions  
        - Quantifying risk and uncertainty  
        - Validating hypotheses and experiments  
        - Supporting rigorous research and business intelligence
        """)

    def output(self):
        self.render_header()
        chart_type, scenario, num_samples = self.render_configuration()
        self.render_chart(chart_type, scenario, num_samples)
