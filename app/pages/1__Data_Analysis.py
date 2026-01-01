import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Analysis Types Guide",
                   page_icon="📊", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    /* ===============================
       THEME VARIABLES
    =============================== */
    [data-theme="light"] {
        --primary: #1f77b4;
        --text-muted: #666;
        --bg-card: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --bg-example: #f8f9fa;
        --bg-code: #282c34;
        --text-code: #abb2bf;
        --shadow: rgba(0,0,0,0.1);
    }

    [data-theme="dark"] {
        --primary: #4fa3ff;
        --text-muted: #aaa;
        --bg-card: linear-gradient(135deg, #2b2f77 0%, #3b1d5a 100%);
        --bg-example: #1e1e1e;
        --bg-code: #0d1117;
        --text-code: #c9d1d9;
        --shadow: rgba(0,0,0,0.6);
    }

    /* ===============================
       HEADERS
    =============================== */
    .main-header {
        font-size: 3rem;
        color: var(--primary);
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }

    .sub-header {
        text-align: center;
        color: var(--text-muted);
        margin-bottom: 3rem;
    }

    /* ===============================
       CARDS
    =============================== */
    .card {
        background: var(--bg-card);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 6px 12px var(--shadow);
    }

    /* ===============================
       EXAMPLE BOX
    =============================== */
    .example-box {
        background-color: var(--bg-example);
        padding: 0.5rem 1rem;       /* smaller vertical padding */
        border-left: 3px solid var(--primary); /* thinner accent bar */
        border-radius: 6px;         /* slightly tighter corners */
        margin: 0.4rem 0;           /* tighter vertical spacing */
        font-size: 0.95rem;         /* slightly smaller font for dense lists */
        line-height: 1.4;           /* tighter but readable */
    }

    /* ===============================
       CODE BOX
    =============================== */
    .code-box {
        background-color: var(--bg-code);
        color: var(--text-code);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<h1 class="main-header">📊 Data Analysis Types</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Complete Learning Guide with Examples & Code</p>',
            unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 Navigation")
st.sidebar.markdown("---")
analysis_type = st.sidebar.radio(
    "Choose an analysis type:",
    ["🏠 Overview", "📈 Descriptive", "🔍 Diagnostic", "🧭 Exploratory",
     "📊 Inferential", "🔮 Predictive", "💡 Prescriptive", "🔗 Causal", "⚙️ Mechanistic"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Click on each type to learn more!")

# Overview Section
if analysis_type == "🏠 Overview":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 What is Data Analysis?")
        st.write("""
        Data analysis is the process of inspecting, cleaning, transforming, and modeling data 
        to discover useful information, draw conclusions, and support decision-making.
        """)

        st.markdown("### 📚 8 Types of Analysis")
        st.write("""
        1. **Descriptive** - What happened?
        2. **Diagnostic** - Why did it happen?
        3. **Exploratory** - What patterns exist?
        4. **Inferential** - What about the population?
        5. **Predictive** - What will happen?
        6. **Prescriptive** - What should we do?
        7. **Causal** - Does X cause Y?
        8. **Mechanistic** - How does X cause Y?
        """)

    with col2:
        st.markdown("### 🔄 Analysis Flow")
        st.info("""
        **Typical Data Analysis Journey:**
        
        1️⃣ Start with **Descriptive** (understand your data)
        
        2️⃣ Move to **Diagnostic** (find causes)
        
        3️⃣ **Explore** patterns (discover insights)
        
        4️⃣ Make **Inferences** (generalize findings)
        
        5️⃣ **Predict** future outcomes
        
        6️⃣ **Prescribe** actions
        
        7️⃣ Establish **Causation**
        
        8️⃣ Understand **Mechanisms**
        """)

    st.markdown("---")
    st.success("👈 Select an analysis type from the sidebar to learn more!")

# Descriptive Analysis
elif analysis_type == "📈 Descriptive":
    st.markdown('<div class="card"><h2>📈 Descriptive Analysis</h2><h3>What happened?</h3></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Definition")
        st.write("""
        Summarizes and describes the main features of a dataset. It's about understanding what 
        the data shows without drawing conclusions about why.
        """)

        st.markdown("### ✨ Key Characteristics")
        st.write("""
        - Uses summary statistics (mean, median, mode, standard deviation)
        - Visualizes data patterns
        - No interpretation of causes
        """)

    with col2:
        st.markdown("### 🛠️ Common Tools")
        st.write("""
        - Summary tables
        - Histograms
        - Bar charts
        - Pie charts
        - Box plots
        """)

    st.markdown("### 💼 Real-World Examples")

    examples = [
        ("🛒", "E-commerce", "50,000 visitors, avg session 3.5 min"),
        ("🏥", "Healthcare", "Avg patient wait time: 22 min (Q1 2024)"),
        ("🎓", "Education", "Exam avg 78% (45–98%)"),
        ("💰", "Sales", "Quarterly revenue: $2.5M"),
        ("📱", "Social Media", "1,200 likes · 150 comments · 80 shares"),
    ]

    # Render each example in a minimal, dense style
    for icon, title, text in examples:
        st.markdown(
            f"""
            <div class="example-box">
                <h5><strong>{icon} {title}</strong> — {text}</h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💻 Python Code Example")
    code = """import pandas as pd
import numpy as np

# Sample e-commerce data
data = {
    'visitor_id': range(1, 101),
    'session_duration': np.random.normal(3.5, 1.2, 100),
    'pages_viewed': np.random.randint(1, 15, 100),
    'purchase': np.random.choice([0, 1], 100, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

# Descriptive statistics
print("=== Descriptive Analysis ===")
print(f"Total visitors: {len(df)}")
print(f"Average session duration: {df['session_duration'].mean():.2f} minutes")
print(f"Median pages viewed: {df['pages_viewed'].median()}")
print(f"Conversion rate: {df['purchase'].sum() / len(df) * 100:.1f}%")
print("\\nSummary Statistics:")
print(df.describe())"""

    st.code(code, language="python")

# Diagnostic Analysis
elif analysis_type == "🔍 Diagnostic":
    st.markdown('<div class="card"><h2>🔍 Diagnostic Analysis</h2><h3>Why did it happen?</h3></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Definition")
        st.write("""
        Investigates the causes behind patterns found in descriptive analysis. 
        It drills down to identify relationships and root causes.
        """)

        st.markdown("### ✨ Key Characteristics")
        st.write("""
        - Examines anomalies and patterns
        - Uses drill-down, data mining, and correlations
        - Answers "why" questions
        """)

    with col2:
        st.markdown("### 🛠️ Common Tools")
        st.write("""
        - Drill-down reports
        - Correlation analysis
        - Data mining
        - Root cause analysis
        """)

    st.markdown("### 💼 Real-World Examples")

    examples = [
        ("🛒", "E-commerce", "Website traffic dropped 30% due to marketing ending & site outage on the 15th"),
        ("🏥", "Healthcare",
         "Patient wait times increased due to staff shortages & flu outbreak"),
        ("🏭", "Manufacturing",
         "Product defects rose 15% because a machine calibration was off during the night shift"),
        ("🛍️", "Retail", "Sales declined in Store #5 due to nearby competitor opening & parking construction"),
        ("📱", "App Development",
         "User churn increased as latest update caused crashes on older devices"),
    ]

    for icon, title, text in examples:
        st.markdown(
            f"""
            <div class="example-box">
                <h5><strong>{icon} {title}</strong> — {text}</h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💻 Python Code Example")
    code = """import pandas as pd
import numpy as np

# Sample website traffic data
dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
data = {
    'date': dates,
    'traffic': np.random.randint(45000, 55000, len(dates)),
    'campaign_active': [1 if d < pd.Timestamp('2024-02-15') else 0 for d in dates],
    'site_outage': [1 if d == pd.Timestamp('2024-02-15') else 0 for d in dates]
}
df = pd.DataFrame(data)

# Simulate traffic drop
df.loc[df['campaign_active'] == 0, 'traffic'] *= 0.7
df.loc[df['site_outage'] == 1, 'traffic'] *= 0.4

print("=== Diagnostic Analysis ===")
# Compare periods
before = df[df['date'] < '2024-02-15']['traffic'].mean()
after = df[df['date'] >= '2024-02-15']['traffic'].mean()
print(f"Average traffic before Feb 15: {before:.0f}")
print(f"Average traffic after Feb 15: {after:.0f}")
print(f"Traffic change: {(after - before) / before * 100:.1f}%")

# Correlation analysis
print(f"\\nCorrelation with campaign: {df['traffic'].corr(df['campaign_active']):.3f}")
print(f"Correlation with outage: {df['traffic'].corr(df['site_outage']):.3f}")"""

    st.code(code, language="python")

# Exploratory Analysis
elif analysis_type == "🧭 Exploratory":
    st.markdown('<div class="card"><h2>🧭 Exploratory Analysis</h2><h3>What patterns exist in the data?</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Definition")
        st.write("""
        Discovers unknown patterns, relationships, or anomalies in data without a specific hypothesis. 
        It's about exploration and discovery.
        """)

        st.markdown("### ✨ Key Characteristics")
        st.write("""
        - Open-ended investigation
        - Identifies trends, clusters, and outliers
        - Generates hypotheses for further testing
        """)

    with col2:
        st.markdown("### 🛠️ Common Tools")
        st.write("""
        - Scatter plots
        - Clustering algorithms
        - PCA
        - Heat maps
        - Data visualization
        """)

    st.markdown("### 💼 Real-World Examples")

    examples = [
        ("🧑‍💼", "Customer Data",
         "Purchase data reveals 3 segments: budget, premium, seasonal shoppers"),
        ("🔬", "Scientific Research",
         "Gene expression data explored to find co-expressed genes under different conditions"),
        ("📣", "Marketing", "Social media analysis shows video posts perform 3x better than image posts on Thursdays"),
        ("💵", "Finance", "Transaction patterns analyzed to identify unusual clusters indicating potential fraud"),
        ("🏙️", "Urban Planning",
         "Traffic patterns reveal unexpected congestion hotspots at non-peak hours"),
    ]

    for icon, title, text in examples:
        st.markdown(
            f"""
            <div class="example-box">
                <h5><strong>{icon} {title}</strong> — {text}</h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💻 Python Code Example")
    code = """import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Sample customer purchase data
np.random.seed(42)
n_customers = 200
data = {
    'customer_id': range(1, n_customers + 1),
    'avg_purchase': np.concatenate([
        np.random.normal(30, 10, 70),   # Budget shoppers
        np.random.normal(150, 30, 70),  # Premium buyers
        np.random.normal(80, 15, 60)    # Mid-tier
    ]),
    'frequency': np.concatenate([
        np.random.randint(15, 30, 70),
        np.random.randint(2, 8, 70),
        np.random.randint(8, 15, 60)
    ])
}
df = pd.DataFrame(data)

print("=== Exploratory Analysis ===")
# Clustering to discover segments
X = df[['avg_purchase', 'frequency']].values
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X)

# Analyze discovered segments
for i in range(3):
    segment = df[df['segment'] == i]
    print(f"\\nSegment {i + 1}:")
    print(f"  Size: {len(segment)} customers")
    print(f"  Avg Purchase: ${segment['avg_purchase'].mean():.2f}")
    print(f"  Avg Frequency: {segment['frequency'].mean():.1f} purchases/year")"""

    st.code(code, language="python")

# Inferential Analysis
elif analysis_type == "📊 Inferential":
    st.markdown('<div class="card"><h2>📊 Inferential Analysis</h2><h3>What can we conclude about the population?</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Definition")
        st.write("""
        Uses sample data to make inferences and generalizations about a larger population. 
        It tests hypotheses and estimates parameters.
        """)

        st.markdown("### ✨ Key Characteristics")
        st.write("""
        - Uses statistical tests and confidence intervals
        - Accounts for uncertainty and sampling error
        - Makes probabilistic statements
        """)

    with col2:
        st.markdown("### 🛠️ Common Tools")
        st.write("""
        - T-tests
        - ANOVA
        - Confidence intervals
        - Regression analysis
        - Chi-square tests
        """)

    st.markdown("### 💼 Real-World Examples")

    examples = [
        ("🗳️", "Political Polling",
         "Survey of 1,000 voters estimates 52% ± 3% support the candidate"),
        ("🧪", "Medical Research",
         "Clinical trial with 500 patients shows new drug reduces symptoms with 95% confidence (p < 0.05)"),
        ("⚙️", "Quality Control",
         "Testing 100 products from 10,000 indicates defect rate likely 2–4%"),
        ("🆚", "A/B Testing",
         "New website design increases conversions by 5–8% from test with 5,000 users"),
        ("📊", "Market Research",
         "Surveying 2,000 consumers suggests 65% of target market would buy the product"),
    ]

    for icon, title, text in examples:
        st.markdown(
            f"""
            <div class="example-box">
                <h5><strong>{icon} {title}</strong> — {text}</h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💻 Python Code Example")
    code = """import pandas as pd
import numpy as np
from scipy import stats

# Sample: A/B test data (sample from larger population)
np.random.seed(42)
control_group = np.random.binomial(1, 0.10, 500)  # 10% conversion
treatment_group = np.random.binomial(1, 0.15, 500)  # 15% conversion

print("=== Inferential Analysis ===")
# Calculate sample statistics
control_rate = control_group.mean()
treatment_rate = treatment_group.mean()

print(f"Control conversion rate (sample): {control_rate:.1%}")
print(f"Treatment conversion rate (sample): {treatment_rate:.1%}")

# Statistical test (inference about population)
statistic, p_value = stats.ttest_ind(treatment_group, control_group)
print(f"\\nT-test p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant (p < 0.05)")
    print("Conclusion: We can infer the treatment improves conversion in the population")
else:
    print("Result: Not statistically significant")

# Confidence interval for difference
diff = treatment_rate - control_rate
se = np.sqrt(control_rate * (1 - control_rate) / len(control_group) + 
             treatment_rate * (1 - treatment_rate) / len(treatment_group))
ci_lower = diff - 1.96 * se
ci_upper = diff + 1.96 * se
print(f"\\n95% CI for difference: [{ci_lower:.1%}, {ci_upper:.1%}]")"""

    st.code(code, language="python")

# Predictive Analysis
elif analysis_type == "🔮 Predictive":
    st.markdown('<div class="card"><h2>🔮 Predictive Analysis</h2><h3>What will happen in the future?</h3></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Definition")
        st.write("""
        Uses historical data and statistical models to forecast future outcomes. 
        It identifies likely future scenarios based on patterns.
        """)

        st.markdown("### ✨ Key Characteristics")
        st.write("""
        - Uses machine learning and statistical models
        - Provides probability estimates
        - Based on historical patterns
        """)

    with col2:
        st.markdown("### 🛠️ Common Tools")
        st.write("""
        - Regression models
        - Time series forecasting
        - Neural networks
        - Decision trees
        - Random forests
        """)

    st.markdown("### 💼 Real-World Examples")

    examples = [
        ("🛍️", "Retail", "Based on past trends, sales predicted to increase 20% during holiday season"),
        ("☔", "Weather Forecasting",
         "70% chance of rain tomorrow based on atmospheric models"),
        ("🏥", "Healthcare", "Patient has 35% risk of developing diabetes in next 5 years based on health metrics"),
        ("💵", "Finance", "Stock price predicted to reach $150 within 6 months from historical patterns & indicators"),
        ("🔄", "Churn Prediction",
         "Customers with certain characteristics have 80% probability of canceling subscription next month"),
    ]

    for icon, title, text in examples:
        st.markdown(
            f"""
            <div class="example-box">
                <h5><strong>{icon} {title}</strong> — {text}</h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💻 Python Code Example")
    code = """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Historical sales data
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
data = {
    'date': dates,
    'month': [d.month for d in dates],
    'day_of_week': [d.dayofweek for d in dates],
    'is_holiday': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
    'sales': np.random.normal(10000, 2000, len(dates))
}
df = pd.DataFrame(data)

# Boost holiday sales
df.loc[df['is_holiday'] == 1, 'sales'] *= 1.5
# Seasonal pattern
df['sales'] *= (1 + 0.3 * np.sin(2 * np.pi * df['month'] / 12))

print("=== Predictive Analysis ===")
# Prepare features
X = df[['month', 'day_of_week', 'is_holiday']]
y = df['sales']

# Train model on historical data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict future sales
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print(f"Model trained on {len(X_train)} historical days")
print(f"Prediction accuracy (MAE): ${mae:.2f}")

# Forecast next week
future_data = pd.DataFrame({
    'month': [1] * 7,
    'day_of_week': range(7),
    'is_holiday': [0] * 7
})
forecast = model.predict(future_data)
print(f"\\nNext 7 days forecast:")
for i, pred in enumerate(forecast):
    print(f"  Day {i+1}: ${pred:.2f}")"""

    st.code(code, language="python")

# Prescriptive Analysis
elif analysis_type == "💡 Prescriptive":
    st.markdown('<div class="card"><h2>💡 Prescriptive Analysis</h2><h3>What should we do about it?</h3></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Definition")
        st.write("""
        Recommends actions to achieve desired outcomes. It combines predictions with business rules 
        and optimization to suggest the best course of action.
        """)

        st.markdown("### ✨ Key Characteristics")
        st.write("""
        - Provides actionable recommendations
        - Often uses optimization and simulation
        - Considers constraints and objectives
        """)

    with col2:
        st.markdown("### 🛠️ Common Tools")
        st.write("""
        - Optimization algorithms
        - Simulation
        - Decision analysis
        - What-if scenarios
        - Prescriptive AI
        """)

    st.markdown("### 💼 Real-World Examples")

    examples = [
        ("🚚", "Supply Chain", "Order 5,000 units from Supplier A & 3,000 from Supplier B via route C to minimize costs while meeting demand"),
        ("🏥", "Healthcare", "Reduce readmissions by discharging patient with home care visits on days 3 & 7 plus specific medications"),
        ("📣", "Marketing", "Allocate 40% budget to digital ads, 30% to email, 30% to influencer partnerships to maximize ROI"),
        ("💲", "Dynamic Pricing",
         "Set price at $127 for maximum profit based on demand, competitor pricing & inventory levels"),
        ("🧑‍⚕️", "Staff Scheduling",
         "Schedule 8 nurses Mon, 10 Fri, 6 Wed to optimize coverage and minimize overtime costs"),
    ]

    for icon, title, text in examples:
        st.markdown(
            f"""
            <div class="example-box">
                <h5><strong>{icon} {title}</strong> — {text}</h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💻 Python Code Example")
    code = """import pandas as pd
import numpy as np
from scipy.optimize import linprog

print("=== Prescriptive Analysis ===")
# Supply chain optimization problem
# Suppliers A and B, need 8000 total units
# Minimize cost: A costs $10/unit, B costs $12/unit
# Constraints: A max 5000 units, B max 5000 units, shipping routes

# Objective: minimize cost
# Decision variables: [units_from_A, units_from_B]
c = [10, 12]  # Costs

# Inequality constraints (A_ub @ x <= b_ub)
# -units_A <= -2000 (min from A)
# units_A <= 5000 (max from A)
# units_B <= 5000 (max from B)
A_ub = [
    [-1, 0],   # Min from A
    [1, 0],    # Max from A
    [0, 1]     # Max from B
]
b_ub = [-2000, 5000, 5000]

# Equality constraints: total must equal 8000
A_eq = [[1, 1]]
b_eq = [8000]

# Solve optimization
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                 method='highs', bounds=[(0, None), (0, None)])

if result.success:
    print("Optimal Solution Found:")
    print(f"  Order from Supplier A: {result.x[0]:.0f} units")
    print(f"  Order from Supplier B: {result.x[1]:.0f} units")
    print(f"  Total cost: ${result.fun:.2f}")
    print(f"  Cost savings vs. buying all from B: ${8000*12 - result.fun:.2f}")
    print("\\nRecommendation: This allocation minimizes cost while meeting demand")
else:
    print("No optimal solution found")"""

    st.code(code, language="python")

# Causal Analysis
elif analysis_type == "🔗 Causal":
    st.markdown('<div class="card"><h2>🔗 Causal Analysis</h2><h3>Does X actually cause Y?</h3></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Definition")
        st.write("""
        Establishes cause-and-effect relationships between variables. It goes beyond correlation 
        to prove that changes in one variable directly cause changes in another.
        """)

        st.markdown("### ✨ Key Characteristics")
        st.write("""
        - Uses controlled experiments or quasi-experimental methods
        - Isolates causal effects from confounding factors
        - Establishes directionality
        """)

    with col2:
        st.markdown("### 🛠️ Common Tools")
        st.write("""
        - Randomized controlled trials
        - Instrumental variables
        - Difference-in-differences
        - Regression discontinuity
        - Causal inference
        """)

    st.warning(
        "⚠️ **Important**: Correlation ≠ Causation! Causal analysis proves true cause-effect relationships.")

    st.markdown("### 💼 Real-World Examples")

    examples = [
        ("💊", "Medicine", "Randomized trial shows vaccine reduces infection rates by 90%, not just correlation"),
        ("📈", "Economics", "Raising minimum wage reduces employee turnover by 2%, controlling for economic conditions"),
        ("🎓", "Education", "New teaching method increases test scores by 15 points via controlled experiment"),
        ("📧", "Marketing", "Reminder emails increase purchase completion by 12%, demonstrated via A/B testing"),
        ("🏥", "Public Health",
         "Smoking causes lung cancer, proven through decades of controlled & natural experiments"),
    ]

    for icon, title, text in examples:
        st.markdown(
            f"""
            <div class="example-box">
                <h5><strong>{icon} {title}</strong> — {text}</h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💻 Python Code Example")
    code = """import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(42)

print("=== Causal Analysis ===")
# Randomized Controlled Trial (RCT) - Gold standard for causation
# Question: Does sending reminder emails CAUSE increased purchases?

# Randomly assign customers to treatment vs control
n_customers = 1000
customers = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'treatment': np.random.choice([0, 1], n_customers)  # Random assignment!
})

# Simulate outcomes
# True causal effect: reminder emails increase purchase rate by 12%
baseline_purchase_rate = 0.20
causal_effect = 0.12

customers['purchased'] = np.where(
    customers['treatment'] == 1,
    np.random.binomial(1, baseline_purchase_rate + causal_effect, n_customers),
    np.random.binomial(1, baseline_purchase_rate, n_customers)
)

# Analyze causal effect
control = customers[customers['treatment'] == 0]
treatment = customers[customers['treatment'] == 1]

control_rate = control['purchased'].mean()
treatment_rate = treatment['purchased'].mean()
causal_effect_estimate = treatment_rate - control_rate

print(f"Control group purchase rate: {control_rate:.1%}")
print(f"Treatment group purchase rate: {treatment_rate:.1%}")
print(f"Estimated causal effect: +{causal_effect_estimate:.1%}")

# Statistical significance
t_stat, p_value = stats.ttest_ind(treatment['purchased'], control['purchased'])
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("\\nConclusion: Reminder emails CAUSE increased purchases")
    print("(Random assignment eliminates confounding variables)")
else:
    print("\\nNo significant causal effect detected")"""

    st.code(code, language="python")

# Mechanistic Analysis
elif analysis_type == "⚙️ Mechanistic":
    st.markdown('<div class="card"><h2>⚙️ Mechanistic Analysis</h2><h3>How exactly does X cause Y?</h3></div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Definition")
        st.write("""
        Explains the underlying mechanisms and processes by which cause-and-effect relationships occur. 
        It's about understanding the "how" at a detailed, often physical or biological level.
        """)

        st.markdown("### ✨ Key Characteristics")
        st.write("""
        - Focuses on precise, deterministic relationships
        - Often uses first principles and domain expertise
        - Minimal random variation expected
        """)

    with col2:
        st.markdown("### 🛠️ Common Tools")
        st.write("""
        - Mathematical models
        - Simulations
        - Physical equations
        - Biochemical pathways
        - Deterministic models
        """)

    st.info("🔬 **Note**: Mechanistic analysis is common in physics, chemistry, biology, and engineering where exact processes can be modeled.")

    st.markdown("### 💼 Real-World Examples")

    examples = [
        ("⚛️", "Physics", "Increasing temperature causes gas molecules to move faster, increasing pressure: P = nRT/V"),
        ("💊", "Pharmacology", "Aspirin inhibits COX enzymes, preventing prostaglandin synthesis, reducing inflammation & pain"),
        ("⚡", "Engineering",
         "Higher voltage increases current through resistor via Ohm's Law: V = IR"),
        ("🧬", "Biology", "Insulin binds receptors → activates PI3K → triggers GLUT4 translocation → glucose enters cells"),
        ("💻", "Computer Science",
         "Quicksort partitions arrays & recursively sorts subarrays with O(n log n) average complexity"),
    ]

    for icon, title, text in examples:
        st.markdown(
            f"""
            <div class="example-box">
                <h5><strong>{icon} {title}</strong> — {text}</h5>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 💻 Python Code Example")
    code = """import pandas as pd
import numpy as np

print("=== Mechanistic Analysis ===")
# Physics example: Ideal Gas Law mechanism
# Explains exactly HOW temperature causes pressure increase

def ideal_gas_pressure(n_moles, temperature_K, volume_L):
    \"\"\"
    Mechanistic model: PV = nRT
    Explains the exact mechanism of how temperature affects pressure
    
    R = 0.0821 L·atm/(mol·K) - Gas constant
    \"\"\"
    R = 0.0821  # Gas constant
    pressure = (n_moles * R * temperature_K) / volume_L
    return pressure

# Fixed conditions
n_moles = 1.0  # 1 mole of gas
volume = 10.0  # 10 liters

# Simulate temperature increase
temperatures = np.linspace(273, 373, 20)  # 0°C to 100°C in Kelvin
pressures = [ideal_gas_pressure(n_moles, T, volume) for T in temperatures]

print("Mechanistic Relationship: P = nRT/V")
print(f"\\nFixed: n={n_moles} mol, V={volume} L")
print("\\nHow temperature causes pressure change:")
print("Temperature → Molecular kinetic energy → Collision frequency → Pressure")
print("\\nPredicted values (deterministic):")
for i in [0, 9, 19]:
    print(f"  T={temperatures[i]:.1f}K → P={pressures[i]:.2f} atm")

# Pharmacology example: Drug concentration decay
def drug_concentration(initial_dose, time_hours, half_life):
    \"\"\"
    Mechanistic model: First-order kinetics
    C(t) = C₀ × e^(-kt) where k = ln(2)/t½
    
    Explains HOW drug is eliminated through metabolic pathways
    \"\"\"
    k = np.log(2) / half_life  # Elimination rate constant
    concentration = initial_dose * np.exp(-k * time_hours)
    return concentration

print("\\n\\nPharmacokinetic Mechanism:")
print("Drug → Liver metabolism (CYP450) → Inactive metabolites → Elimination")
initial = 100  # mg
half_life = 6  # hours
time_points = [0, 3, 6, 12]
for t in time_points:
    conc = drug_concentration(initial, t, half_life)
    print(f"  t={t}h → Concentration={conc:.2f} mg")"""

    st.code(code, language="python")

# Footer with comparison table
st.markdown("---")
st.markdown("## 📊 Quick Comparison Table")

comparison_data = {
    "Type": ["Descriptive", "Diagnostic", "Exploratory", "Inferential", "Predictive", "Prescriptive", "Causal", "Mechanistic"],
    "Core Question": [
        "What happened?",
        "Why did it happen?",
        "What patterns exist?",
        "What about the population?",
        "What will happen?",
        "What should we do?",
        "Does X cause Y?",
        "How does X cause Y?"
    ],
    "Uncertainty": ["None", "Low", "Medium", "Medium-High", "High", "Medium-High", "Low-Medium", "Very Low"],
    "Timeframe": ["Past", "Past", "Past/Present", "Present", "Future", "Future", "Any", "Any"]
}

df_comparison = pd.DataFrame(comparison_data)
st.dataframe(df_comparison, width='stretch', hide_index=True)

st.markdown("---")
st.markdown("## 💡 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔍 Looking Backward
    - **Descriptive** and **Diagnostic** examine what happened and why
    - **Exploratory** discovers unknown patterns without preconceptions
    - **Inferential** generalizes from samples to populations
    """)

with col2:
    st.markdown("""
    ### 🔮 Looking Forward
    - **Predictive** and **Prescriptive** forecast and recommend actions
    - **Causal** establishes true cause-effect (correlation ≠ causation!)
    - **Mechanistic** explains the exact processes underlying causation
    """)

st.success(
    "✨ **Remember**: Most real-world projects combine multiple types of analysis!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>📚 Data Analysis Types Learning Guide</p>
        <p>Built with Streamlit 🎈 | © 2026</p>
    </div>
""", unsafe_allow_html=True)
