import streamlit as st
import pandas as pd
import numpy as np


class DataCleaning:
    def __init__(self, df):
        self.df = pd.read_csv(df, low_memory=False)
        self.original_df = self.df.copy()

    def dataframe_info(self):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{self.df.shape[0]:,}")
            st.metric("Total Columns", self.df.shape[1])
        with col2:
            st.metric(
                "Memory Usage", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.metric("Duplicate Rows", self.df.duplicated().sum())
        with col3:
            st.metric("Total Missing", f"{self.df.isnull().sum().sum():,}")
            st.metric(
                "Missing %", f"{(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100):.2f}%")

        st.dataframe(self.df)

    def make_columns_unique(self, df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            dup_idx = cols[cols == dup].index.tolist()
            for i, idx in enumerate(dup_idx[1:], start=1):
                cols[idx] = f"{cols[idx]}_{i}"
        df.columns = cols
        return df

    def steps(self):
        with st.container():
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>🏷️ Step 1 – Schema & Header Standardization</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Review and standardize column names: lowercase, underscores, remove special characters")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Before
print("Original columns:")
print(df.columns.tolist())

# Standardize headers
df.columns = (
    df.columns
    .str.strip()                          # Remove spaces
    .str.lower()                          # Lowercase
    .str.replace(r'\\s+', '_', regex=True) # Spaces to underscore
    .str.replace(r'[^a-z0-9_]', '', regex=True) # Remove special chars
    .str.replace(r'_+', '_', regex=True)   # Multiple underscores to one
    .str.strip('_')                       # Remove leading/trailing _
)

print(df.columns.tolist())""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Before**")
                before_cols = self.df.columns.tolist()
                st.code(str(before_cols), language=None)

                # Apply transformation
                self.df.columns = (
                    self.df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r'\s+', '_', regex=True)
                    .str.replace(r'[^a-z0-9_]', '', regex=True)
                    .str.replace(r'_+', '_', regex=True)
                    .str.strip('_')
                )

                st.markdown("**After**")
                after_cols = self.df.columns.tolist()
                st.code(str(after_cols), language=None)

        st.divider()

        with st.container():
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>🔍 Step 2 – Data Type Inspection & Validation</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Inspect current data types and schema consistency across all features")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Analyze data types
print("Current data types:")
print(df.dtypes.value_counts())
print("\\n" + "="*50)

# Detailed breakdown
for dtype in df.dtypes.unique():
    cols = df.select_dtypes(include=[dtype]).columns.tolist()
    print(f"\\n{dtype}: {len(cols)} columns")
    print(cols[:3])  # Show first 3""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Data Type Distribution**")

                type_counts = self.df.dtypes.value_counts()

                for dtype, count in type_counts.items():
                    st.metric(f"{dtype}", count)

                # Show sample columns for each type
                with st.expander("View columns by type"):
                    for dtype in self.df.dtypes.unique():
                        cols = self.df.select_dtypes(
                            include=[dtype]).columns.tolist()
                        st.text(f"{dtype}: {cols[:5]}")

        st.divider()

        with st.container():
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>📊 Step 3 – Missing Value Quantification</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Quantify missingness per feature and identify patterns of missing data")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Check for missing values
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df) * 100)

missing_df = pd.DataFrame({
    'column': missing_counts.index,
    'missing_count': missing_counts.values,
    'missing_pct': missing_pct.values
})

# Filter columns with missing values
missing_df = missing_df[
    missing_df['missing_count'] > 0
].sort_values('missing_count', ascending=False)

print(f"\\nColumns with missing data: {len(missing_df)}")
print(f"Total missing values: {missing_counts.sum():,}")
print("\\n", missing_df.head(10))""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Missing Values Report**")

                missing_counts = self.df.isnull().sum()
                missing_pct = (missing_counts / len(self.df) * 100)

                missing_df_display = pd.DataFrame({
                    'Column': missing_counts.index,
                    'Missing': missing_counts.values,
                    'Percentage': missing_pct.values
                })

                missing_df_display = missing_df_display[
                    missing_df_display['Missing'] > 0
                ].sort_values('Missing', ascending=False)

                if len(missing_df_display) > 0:
                    st.metric("Columns with Missing Data",
                              len(missing_df_display))
                    st.metric("Total Missing Values",
                              f"{missing_counts.sum():,}")

                    st.dataframe(
                        missing_df_display.head(10).style.format(
                            {'Percentage': '{:.2f}%'}),
                        height=250
                    )
                else:
                    st.success("✓ No missing values detected!")

        st.divider()

        with st.container():
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>🗑️ Step 4 – Duplicate Detection & Data Integrity</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Detect duplicate rows and validate unique identifiers for row-level integrity")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows found: {duplicate_count:,}")

if duplicate_count > 0:
    # Show sample duplicates
    print("\\nSample duplicate rows:")
    print(df[df.duplicated(keep=False)].head())
    
    # Remove duplicates (keep first occurrence)
    before_shape = df.shape[0]
    df = df.drop_duplicates()
    after_shape = df.shape[0]
    
    print(f"\\n✓ DUPLICATES REMOVED")
    print(f"Rows removed: {before_shape - after_shape:,}")
    print(f"New shape: {df.shape}")
else:
    print("✓ No duplicate rows found")""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Duplicate Analysis**")

                duplicate_count = self.df.duplicated().sum()

                if duplicate_count > 0:
                    before_shape = self.df.shape[0]
                    st.metric("Duplicates Found", f"{duplicate_count:,}")

                    # Remove duplicates
                    self.df = self.df.drop_duplicates()
                    after_shape = self.df.shape[0]

                    st.metric("Rows Removed",
                              f"{before_shape - after_shape:,}")
                    st.metric("Rows Remaining", f"{after_shape:,}")
                    st.success("✓ Duplicates removed")
                else:
                    st.info("✓ No duplicate rows found")
                    st.metric("Total Unique Rows", f"{self.df.shape[0]:,}")

        st.divider()

        with st.container():
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>🔧 Step 5 – Data Type & Format Standardization</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Convert boolean-like values (Yes/No, True/False, 1/0), validate numerical types, parse datetime fields")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Auto-detect and convert data types
conversions = []

# 1. Try to convert boolean-like columns
bool_map = {
    'Yes': True, 'Y': True, 'yes': True, 'y': True,
    'True': True, 'true': True, 'TRUE': True,
    '1': True, 1: True,
    'No': False, 'N': False, 'no': False, 'n': False,
    'False': False, 'false': False, 'FALSE': False,
    '0': False, 0: False
}

for col in df.select_dtypes(include=['object']).columns:
    # Check if column contains boolean-like values
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) <= 10:  # Check small cardinality
        # Check if all values are in bool_map
        unique_str = [str(v).strip() for v in unique_vals]
        if all(v in bool_map for v in unique_str):
            df[col] = df[col].astype(str).str.strip().map(bool_map)
            conversions.append(f"{col}: object → boolean")

# 2. Try to convert object columns to numeric
for col in df.select_dtypes(include=['object']).columns:
    # Remove common currency symbols and commas
    test_col = df[col].astype(str).str.replace(
        r'[$,€£¥]', '', regex=True
    ).str.strip()
    
    # Try numeric conversion
    converted = pd.to_numeric(test_col, errors='coerce')
    
    # If >50% successfully converted, apply
    if converted.notna().sum() / len(df) > 0.5:
        df[col] = converted
        conversions.append(f"{col}: object → numeric")

# 3. Try to detect datetime columns
for col in df.select_dtypes(include=['object']).columns:
    if df[col].notna().sum() > 0:
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() / len(df) > 0.5:
                df[col] = converted
                conversions.append(f"{col}: object → datetime")
        except:
            pass

print("✓ AUTO TYPE CONVERSION COMPLETE")
print(f"Conversions made: {len(conversions)}")
for c in conversions:
    print(f"  - {c}")""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Type Conversions Applied**")

                conversions = []

                # 1. Boolean conversion
                bool_map = {
                    'Yes': True, 'Y': True, 'yes': True, 'y': True,
                    'True': True, 'true': True, 'TRUE': True,
                    '1': True, 1: True,
                    'No': False, 'N': False, 'no': False, 'n': False,
                    'False': False, 'false': False, 'FALSE': False,
                    '0': False, 0: False
                }

                for col in self.df.select_dtypes(include=['object']).columns:
                    # Check if column contains boolean-like values
                    unique_vals = self.df[col].dropna().unique()
                    if len(unique_vals) <= 10:  # Small cardinality check
                        # Check if all values are in bool_map
                        unique_str = [str(v).strip() for v in unique_vals]
                        if all(v in bool_map for v in unique_str):
                            self.df[col] = self.df[col].astype(
                                str).str.strip().map(bool_map)
                            conversions.append(f"{col}: object → boolean")

                # 2. Numeric conversion
                for col in self.df.select_dtypes(include=['object']).columns:
                    test_col = self.df[col].astype(str).str.replace(
                        r'[$,€£¥]', '', regex=True).str.strip()
                    converted = pd.to_numeric(test_col, errors='coerce')

                    if converted.notna().sum() / len(self.df) > 0.5:
                        self.df[col] = converted
                        conversions.append(f"{col}: object → numeric")

                # 3. Datetime conversion
                for col in self.df.select_dtypes(include=['object']).columns:
                    if self.df[col].notna().sum() > 0:
                        try:
                            converted = pd.to_datetime(
                                self.df[col], errors='coerce', format='mixed')
                            if converted.notna().sum() / len(self.df) > 0.5:
                                self.df[col] = converted
                                conversions.append(f"{col}: object → datetime")
                        except:
                            pass

                if conversions:
                    st.metric("Columns Converted", len(conversions))
                    st.code('\n'.join(conversions[:10]), language=None)
                    st.success("✓ Types converted")
                else:
                    st.info("No automatic conversions needed")

        st.divider()

        with st.container():
            self.df = self.make_columns_unique(self.df)

            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>📉 Step 6 – Outlier Sanity Checks</h3>",
                unsafe_allow_html=True
            )
            st.caption(
                "Flag extreme values and distinguish anomalies from valid rare events (IQR method)"
            )

            col_code, col_result = st.columns([5, 4], gap="medium")

            # Show code snippet in one column
            with col_code:
                st.code("""
# Detect outliers using IQR method
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
outlier_report = []

for col in numeric_cols:
    non_null_count = df[col].dropna().shape[0]
    if non_null_count > 0:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        
        if outliers > 0:
            outlier_pct = (outliers / len(df)) * 100
            outlier_report.append({
                'Column': col,
                'Outliers': outliers,
                'Percentage': f"{outlier_pct:.2f}%",
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })

print("✓ OUTLIER DETECTION COMPLETE")
print(f"Columns with outliers: {len(outlier_report)}")
for item in outlier_report[:5]:
    print(f"  {item['Column']}: {item['Outliers']} outliers")
                """, language="python", line_numbers=True)

            # Run outlier detection and display results
            with col_result:
                st.markdown("**Outlier Detection Report**")

                numeric_cols = self.df.select_dtypes(
                    include=['float64', 'int64']).columns
                outlier_report = []

                for col in numeric_cols:
                    non_null_count = self.df[col].dropna().shape[0]
                    if non_null_count > 0:
                        Q1 = self.df[col].quantile(0.25)
                        Q3 = self.df[col].quantile(0.75)
                        IQR = Q3 - Q1

                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        # Ensure boolean indexing works even with duplicates handled
                        outliers = self.df[(self.df[col] < lower_bound) | (
                            self.df[col] > upper_bound)].shape[0]

                        if outliers > 0:
                            outlier_pct = (outliers / len(self.df)) * 100
                            outlier_report.append({
                                'Column': col,
                                'Outliers': outliers,
                                'Percentage': f"{outlier_pct:.2f}%",
                                'Lower Bound': lower_bound,
                                'Upper Bound': upper_bound
                            })

                if outlier_report:
                    st.metric("Columns with Outliers", len(outlier_report))
                    outlier_df = pd.DataFrame(outlier_report)
                    st.dataframe(outlier_df.head(10))
                    st.warning("⚠️ Review outliers before removal")
                else:
                    st.success("✓ No significant outliers detected")

            st.divider()

        with st.container():
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>💉 Step 7 – Missing Value Imputation Strategy</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Apply appropriate strategies: imputation (median/mode), removal, or flagging based on context")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Strategy: Numeric → Median, Categorical → Mode
missing_before = df.isnull().sum().sum()

# Handle numeric columns
numeric_cols = df.select_dtypes(
include=['float64', 'int64']
).columns

for col in numeric_cols:
if df[col].isnull().sum() > 0:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)
    print(f"{col}: filled with median ({median_val})")

# Handle categorical/object columns
categorical_cols = df.select_dtypes(
include=['object']
).columns

for col in categorical_cols:
if df[col].isnull().sum() > 0:
    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
    df[col].fillna(mode_val, inplace=True)
    print(f"{col}: filled with mode ({mode_val})")

# Handle datetime columns (forward fill)
datetime_cols = df.select_dtypes(
include=['datetime64']
).columns

for col in datetime_cols:
if df[col].isnull().sum() > 0:
    df[col].fillna(method='ffill', inplace=True)
    print(f"{col}: filled with forward fill")

missing_after = df.isnull().sum().sum()

print(f"Before: {missing_before:,} missing values")
print(f"After: {missing_after:,} missing values")""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Imputation Results**")

                missing_before = self.df.isnull().sum().sum()

                # Numeric columns - median
                numeric_cols = self.df.select_dtypes(
                    include=['float64', 'int64']).columns
                numeric_filled = 0
                for col in numeric_cols:
                    if self.df[col].isnull().sum() > 0:
                        self.df[col] = self.df[col].fillna(
                            self.df[col].median())
                        numeric_filled += 1

                # Categorical - mode
                categorical_cols = self.df.select_dtypes(
                    include=['object']).columns
                categorical_filled = 0
                for col in categorical_cols:
                    if self.df[col].isnull().sum() > 0:
                        mode_val = self.df[col].mode()[0] if len(
                            self.df[col].mode()) > 0 else 'Unknown'
                        self.df[col] = self.df[col].fillna(mode_val)
                        categorical_filled += 1

                # Datetime - forward fill
                datetime_cols = self.df.select_dtypes(
                    include=['datetime64']).columns
                datetime_filled = 0
                for col in datetime_cols:
                    if self.df[col].isnull().sum() > 0:
                        self.df[col] = self.df[col].fillna(method='ffill')
                        datetime_filled += 1

                missing_after = self.df.isnull().sum().sum()

                st.metric("Missing Before", f"{missing_before:,}")
                st.metric("Missing After", f"{missing_after:,}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Numeric Filled", numeric_filled)
                col2.metric("Categorical Filled", categorical_filled)
                col3.metric("Datetime Filled", datetime_filled)

                if missing_after == 0:
                    st.success("✓ All missing values handled!")
                else:
                    st.warning(
                        f"⚠️ {missing_after:,} values remain missing")

        st.divider()

        with st.container():
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>📝 Step 8 – Text & Categorical Normalization</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Normalize text data: trim whitespace, fix casing, standardize encoding and formatting")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Standardize text columns
text_cols = df.select_dtypes(include=['object']).columns
cleaned_cols = 0

for col in text_cols:
    # Strip whitespace
    df[col] = df[col].astype(str).str.strip()
    
    # Replace multiple spaces with single space
    df[col] = df[col].str.replace(r'\\s+', ' ', regex=True)
    
    # Standardize to title case (optional - comment if not needed)
    # df[col] = df[col].str.title()
    
    # Replace string representations of null
    null_strings = ['nan', 'NaN', 'none', 'None', 
                    'null', 'NULL', 'N/A', 'n/a', '']
    df[col] = df[col].replace(null_strings, np.nan)
    
    cleaned_cols += 1

print("✓ TEXT STANDARDIZATION COMPLETE")
print(f"Columns cleaned: {cleaned_cols}")
print(f"Sample values: {df[text_cols[0]].head(3).tolist()}")""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Text Cleaning Results**")

                text_cols = self.df.select_dtypes(include=['object']).columns

                if len(text_cols) > 0:
                    # Apply standardization
                    for col in text_cols:
                        self.df[col] = self.df[col].astype(str).str.strip()
                        self.df[col] = self.df[col].str.replace(
                            r'\s+', ' ', regex=True)

                        null_strings = ['nan', 'NaN', 'none',
                                        'None', 'null', 'NULL', 'N/A', 'n/a', '']
                        self.df[col] = self.df[col].replace(
                            null_strings, np.nan)

                    st.metric("Text Columns Cleaned", len(text_cols))

                    # Show sample before/after
                    st.code(
                        f"Example column: {text_cols[0]}\n{self.df[text_cols[0]].head(3).tolist()}", language=None)
                    st.success("✓ Text standardized")
                else:
                    st.info("No text columns found")

        st.divider()

        with st.container(border=True):
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>🗂️ Step 9 – Feature Quality Assessment</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Remove low-variance, high-missingness, or redundant columns to improve data quality")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Find columns to remove
cols_to_drop = []

# 1. Constant columns (all same value)
for col in df.columns:
    if df[col].nunique() <= 1:
        cols_to_drop.append(col)
        print(f"Constant column: {col}")

# 2. Columns with >95% missing data
missing_threshold = 0.95
for col in df.columns:
    missing_pct = df[col].isnull().sum() / len(df)
    if missing_pct > missing_threshold:
        cols_to_drop.append(col)
        print(f"High missing column: {col} ({missing_pct:.1%})")

# 3. Columns with all unique values (likely IDs)
for col in df.columns:
    if df[col].nunique() == len(df) and len(df) > 100:
        cols_to_drop.append(col)
        print(f"All unique (ID column): {col}")

# Remove duplicates from list
cols_to_drop = list(set(cols_to_drop))

# Drop columns
if cols_to_drop:
    df = df.drop(columns=cols_to_drop)
    print(f"\\n✓ COLUMNS REMOVED: {len(cols_to_drop)}")
else:
    print("✓ No columns to remove")

print(f"Remaining columns: {df.shape[1]}")""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Column Removal Analysis**")

                cols_before = self.df.shape[1]
                cols_to_drop = []
                reasons = []

                # Constant columns
                for col in self.df.columns:
                    if self.df[col].nunique() <= 1:
                        cols_to_drop.append(col)
                        reasons.append(f"{col}: constant value")

                # High missing
                missing_threshold = 0.95
                for col in self.df.columns:
                    missing_pct = self.df[col].isnull().sum() / len(self.df)
                    if missing_pct > missing_threshold:
                        if col not in cols_to_drop:
                            cols_to_drop.append(col)
                            reasons.append(f"{col}: {missing_pct:.1%} missing")

                # All unique (IDs)
                for col in self.df.columns:
                    if self.df[col].nunique() == len(self.df) and len(self.df) > 100:
                        if col not in cols_to_drop:
                            cols_to_drop.append(col)
                            reasons.append(f"{col}: unique ID")

                if cols_to_drop:
                    self.df = self.df.drop(columns=cols_to_drop)

                    st.metric("Columns Removed", len(cols_to_drop))
                    st.code('\n'.join(reasons[:10]), language=None)
                    st.success(f"✓ {len(cols_to_drop)} columns dropped")
                else:
                    st.info("✓ No columns to remove")

                st.metric("Columns Remaining", self.df.shape[1])

        st.divider()

        with st.container(border=True):
            st.markdown(
                "<h3 style='color: #1976d2; margin: 0.3rem 0;'>✅ Step 10 – Final Validation & Consistency Check</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Re-run schema checks, confirm consistency across features, ensure reproducibility")

            col_code, col_result = st.columns([5, 4], gap="medium")

            with col_code:
                st.code("""# Final validation checks
validations = []

# Check 1: No missing values
missing_total = df.isnull().sum().sum()
validations.append(
    ("Missing values", missing_total == 0, f"{missing_total:,} found")
)

# Check 2: No duplicates
duplicate_count = df.duplicated().sum()
validations.append(
    ("Duplicate rows", duplicate_count == 0, f"{duplicate_count:,} found")
)

# Check 3: Numeric columns have valid ranges
numeric_cols = df.select_dtypes(
    include=['float64', 'int64']
).columns
for col in numeric_cols:
    has_inf = np.isinf(df[col]).any()
    if has_inf:
        validations.append(
            (f"{col} - no infinity", False, "Infinity values found")
        )
    else:
        validations.append(
            (f"{col} - no infinity", True, "OK")
        )

# Check 4: Data type consistency
type_issues = 0
for col in df.columns:
    if df[col].dtype == 'object':
        # Check if column should be numeric
        try:
            pd.to_numeric(df[col].head(100))
            type_issues += 1
        except:
            pass

validations.append(
    ("Type consistency", type_issues == 0, f"{type_issues} potential issues")
)

print("✓ VALIDATION COMPLETE")
for check, passed, msg in validations:
    status = "✓" if passed else "✗"
    print(f"{status} {check}: {msg}")""", language="python", line_numbers=True)

            with col_result:
                st.markdown("**Quality Validation Report**")

                validations = []

                # Missing values check
                missing_total = self.df.isnull().sum().sum()
                validations.append({
                    'Check': 'Missing Values',
                    'Status': '✓ Pass' if missing_total == 0 else '✗ Fail',
                    'Details': f"{missing_total:,} found"
                })

                # Duplicates check
                duplicate_count = self.df.duplicated().sum()
                validations.append({
                    'Check': 'No Duplicates',
                    'Status': '✓ Pass' if duplicate_count == 0 else '✗ Fail',
                    'Details': f"{duplicate_count:,} found"
                })

                # Infinity check
                numeric_cols = self.df.select_dtypes(
                    include=['float64', 'int64']).columns
                inf_count = sum([np.isinf(self.df[col]).any()
                                for col in numeric_cols])
                validations.append({
                    'Check': 'No Infinity Values',
                    'Status': '✓ Pass' if inf_count == 0 else '✗ Fail',
                    'Details': f"{inf_count} columns with inf"
                })

                # Display validation results
                validation_df = pd.DataFrame(validations)
                st.dataframe(
                    validation_df,  hide_index=True)

                passed = validation_df['Status'].str.contains('✓').sum()
                total = len(validation_df)

                if passed == total:
                    st.success(f"✓ All {total} validations passed!")
                else:
                    st.warning(f"⚠️ {passed}/{total} validations passed")

        st.divider()

        with st.container(border=True):
            st.markdown(
                "<h3 style='color: #28a745; margin: 0.3rem 0;'>💾 Step 11 – Export Clean Dataset</h3>",
                unsafe_allow_html=True
            )

            st.caption(
                "Deliver a clean, reliable dataset suitable for unbiased exploration and modeling")

            # Cleaning Summary
            st.markdown("**📊 Cleaning Summary**")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Rows", f"{self.original_df.shape[0]:,}")
                st.metric(
                    "Final Rows", f"{self.df.shape[0]:,}", delta=f"{self.df.shape[0] - self.original_df.shape[0]:,}")
            with col2:
                st.metric("Original Columns", self.original_df.shape[1])
                st.metric(
                    "Final Columns", self.df.shape[1], delta=f"{self.df.shape[1] - self.original_df.shape[1]}")
            with col3:
                st.metric("Missing Values",
                          f"{self.df.isnull().sum().sum():,}")
                st.metric("Duplicates", self.df.duplicated().sum())
            with col4:
                st.metric(
                    "Memory Usage", f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                st.metric("Data Quality", "✓ Clean" if self.df.isnull().sum(
                ).sum() == 0 and self.df.duplicated().sum() == 0 else "⚠️ Review")

            st.markdown("---")

            # Display cleaned dataset
            st.markdown("**🗂️ Cleaned Dataset Output**")
            st.dataframe(self.df,  height=400)

            # Data types summary
            st.markdown("**📋 Final Schema**")
            schema_df = pd.DataFrame({
                'Column': self.df.columns,
                'Data Type': self.df.dtypes.values,
                'Non-Null Count': self.df.count().values,
                'Null Count': self.df.isnull().sum().values,
                'Unique Values': [self.df[col].nunique() for col in self.df.columns]
            })
            st.dataframe(schema_df,  height=300)

    def output(self):
        self.dataframe_info()
        self.steps()
