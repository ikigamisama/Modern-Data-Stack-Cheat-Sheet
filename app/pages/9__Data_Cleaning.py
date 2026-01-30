import streamlit as st
from pathlib import Path
from components.data_cleaning import DataCleaning

APP_DIR = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Data Cleaning",
    page_icon="🧹",
    layout="wide"
)

st.title("🧹 Data Cleaning Cheat Sheet Universal Data")
st.caption(
    "This section focuses on validating, cleaning, and standardizing raw data to ensure accuracy, consistency, and reliability before analysis."
)

st.markdown("""### Overview
This section ensures the dataset is accurate, consistent, and analysis-ready by systematically identifying and resolving data quality issues.

### Objectives
- Improve data accuracy and consistency  
- Reduce noise and bias introduced by poor data quality  
- Prepare a reliable foundation for analysis and modeling  
""")

st.divider()

DATASETS = {
    'Marketing Campaign': APP_DIR / 'data' / 'marketing_campaign_data.csv',
    'Airbnb Open Data': APP_DIR / 'data' / 'Airbnb_Open_Data.csv',
    'Credit Customers': APP_DIR / 'data' / 'credit_customers.csv',
    'Amazon Product': APP_DIR / 'data' / 'amazon_product.csv',
    'Netflix': APP_DIR / 'data' / 'netflix.csv',
    'Fashion': APP_DIR / 'data' / 'FashionDataset.csv',
    'Online Foods': APP_DIR / 'data' / 'onlinefoods.csv',
    'HR Employee Attrition': APP_DIR / 'data' / 'HR-Employee-Attrition.csv'
}

DATABASE_SELECTION = [i for i, v in DATASETS.items()]

dataframe = st.selectbox('Datasets:', DATABASE_SELECTION)
DataCleaning(DATASETS[dataframe]).output()

st.markdown("""### Key Checks & Actions

#### 1. Dataset Snapshot
- Review dataset shape and structure  
- Inspect data types and schema consistency  
- Generate basic summary statistics  

#### 2. Missing Values
- Quantify missingness per feature  
- Identify patterns of missing data  
- Apply appropriate strategies: imputation, removal, or flagging  

#### 3. Duplicates & Data Integrity
- Detect duplicate rows or records  
- Validate unique identifiers and keys  
- Ensure row-level integrity  

#### 4. Data Type & Format Standardization
- Convert and validate numerical types  
- Parse and standardize datetime fields  
- Normalize categorical and text data (case, whitespace, encoding)  

#### 5. Outlier Sanity Checks
- Flag extreme or suspicious values  
- Distinguish true anomalies from valid rare events  
- Avoid premature removal without context  

#### 6. Final Validation
- Re-run schema and summary checks  
- Confirm consistency across features  
- Ensure reproducibility of cleaning steps  

**Outcome:**  
A clean, reliable dataset suitable for unbiased exploration and downstream modeling.
""")


with st.expander("📖 Data Cleaning Workflow Overview"):
    st.markdown("""
    ### Systematic Data Cleaning Process
    
    This tool follows best practices to ensure accuracy, consistency, and reliability:
    
    1. **Schema & Header Standardization** - Clean and consistent column naming conventions
    2. **Data Type Inspection** - Review current types and schema consistency
    3. **Missing Value Quantification** - Identify patterns and extent of missingness
    4. **Duplicate Detection** - Ensure row-level integrity and unique identifiers
    5. **Type & Format Standardization** - Convert numerical, datetime, and categorical data
    6. **Outlier Sanity Checks** - Flag anomalies while preserving valid rare events
    7. **Missing Value Imputation** - Apply context-appropriate strategies
    8. **Text Normalization** - Standardize case, whitespace, and encoding
    9. **Feature Quality Assessment** - Remove low-variance or redundant columns
    10. **Final Validation** - Confirm consistency and reproducibility
    11. **Clean Dataset Export** - Deliver analysis-ready data
    
    **Imputation Strategy:**
    - Numeric columns → Median (robust to outliers)
    - Categorical columns → Mode (most frequent value)
    - Datetime columns → Forward fill (temporal continuity)
    
    **Column Removal Criteria:**
    - Constant values (zero variance)
    - Excessive missingness (>95%)
    - All unique values (likely identifiers)
    
    **Outcome:** A clean, reliable dataset suitable for unbiased exploration and downstream modeling.
    """)

with st.expander("💡 Best Practices & Recommendations"):
    st.markdown("""
    **Critical Guidelines:**
    - **Context matters**: Don't blindly apply transformations without understanding your data
    - **Domain knowledge**: Some "outliers" may be legitimate rare events
    - **Preserve raw data**: Always maintain an untouched backup of original data
    - **Validate assumptions**: Question automated decisions based on business logic
    - **Iterate systematically**: Clean in stages and verify at each checkpoint
    - **Document everything**: Track all transformations for reproducibility
    - **Bias awareness**: Be mindful of how cleaning choices might introduce bias
    - **Stakeholder alignment**: Confirm cleaning decisions align with analysis objectives
    
    **Common Pitfalls to Avoid:**
    - Removing outliers without investigation
    - Imputing missing values when absence is informative
    - Over-standardizing text that contains meaningful variation
    - Dropping features without understanding their potential value
    """)
