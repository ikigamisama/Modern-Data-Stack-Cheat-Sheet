import streamlit as st


def import_setup():
    st.header("🚀 Import and Setup")
    st.code("""
import polars as pl
import polars.selectors as cs
from datetime import datetime, timedelta
import numpy as np

# Optimized display settings for analysis
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(120)
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_streaming_chunk_size(50000)

# Enable verbose logging for debugging
pl.Config.set_verbose(True)
""", language='python')


def create_dataframe():
    st.header("📊 Creating DataFrames and Series")
    st.markdown("### From Various Sources")
    st.code("""
# DataFrame from dictionary with explicit schema
df = pl.DataFrame({
    'user_id': [1, 2, 3],
    'email': ['a@test.com', 'b@test.com', 'c@test.com'],
    'revenue': [100.5, 200.3, 150.7]
}, schema={
    'user_id': pl.UInt32,
    'email': pl.Utf8,
    'revenue': pl.Float64
})

# From numpy arrays (fast)
df = pl.from_numpy(
    np.array([[1, 2], [3, 4]]),
    schema=['col1', 'col2'],
    orient='row'
)

# From pandas (efficient conversion)
import pandas as pd
pandas_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
df = pl.from_pandas(pandas_df)

# From Arrow tables (zero-copy)
import pyarrow as pa
arrow_table = pa.table({'a': [1, 2, 3]})
df = pl.from_arrow(arrow_table)

# From records (list of dicts)
records = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
]
df = pl.DataFrame(records)

# Empty DataFrame with schema
df = pl.DataFrame(
    schema={
        'id': pl.UInt64,
        'timestamp': pl.Datetime('ms'),
        'value': pl.Float64,
        'category': pl.Categorical
    }
)
""", language='python')

    st.markdown("### Reading Data (Eager and Lazy)")
    st.code("""
# CSV with advanced options
df = pl.read_csv(
    'file.csv',
    separator=',',
    has_header=True,
    skip_rows=1,
    n_rows=1000,
    columns=['col1', 'col2'],  # Read only specific columns
    dtypes={'col1': pl.UInt32},
    null_values=['NA', 'null', ''],
    try_parse_dates=True,
    infer_schema_length=10000,
    low_memory=False  # Better type inference
)

# Lazy reading for large files (recommended)
lazy_df = pl.scan_csv('large_file.csv')
lazy_df = pl.scan_parquet('large_file.parquet')
lazy_df = pl.scan_ndjson('large_file.jsonl')

# Read with glob patterns
df = pl.read_csv('data/*.csv')
lazy_df = pl.scan_parquet('data/year=*/month=*/*.parquet')

# Read from cloud storage (S3, GCS, Azure)
df = pl.read_parquet('s3://bucket/path/file.parquet')
lazy_df = pl.scan_parquet('gs://bucket/path/*.parquet')

# Read from database
df = pl.read_database(
    query="SELECT * FROM table WHERE date > '2024-01-01'",
    connection="postgresql://user:pass@localhost/db"
)

# Read compressed files
df = pl.read_csv('file.csv.gz')
df = pl.read_parquet('file.parquet.zstd')

# Read Excel with specific options
df = pl.read_excel(
    'file.xlsx',
    sheet_name='Sheet1',
    engine='xlsx2csv',  # or 'calamine'
    read_csv_options={'has_header': True}
)

# Streaming read for massive files
chunks = pl.read_csv_batched(
    'huge_file.csv',
    batch_size=100000
)
for batch in chunks:
    process_batch(batch)
""", language='python')


def data_profiling():
    st.header("🔍 Data Profiling & Quality Assessment")
    st.code("""
# Comprehensive data profile
def profile_dataframe(df: pl.DataFrame) -> None:
    # Generate comprehensive data profile

    print(f"{'='*60}")
    print(f"DataFrame Profile")
    print(f"{'='*60}")
    print(f"Shape: {df.shape} (rows × columns)")
    print(f"Memory: {df.estimated_size('mb'):.2f} MB")
    print(f"Columns: {df.width}")

    # Data types distribution
    print("Data Types:")
    print(f"{'='*60}")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"{col:30} {dtype}")

    # Null analysis

    print("Missing Values:")
    print(f"{'='*60}")
    null_counts = df.null_count()
    for col in df.columns:
        null_pct = (null_counts[col][0] / df.height) * 100
        print(f"{col:30} {null_counts[col][0]:8} ({null_pct:5.2f}%)")

    # Cardinality analysis
    
    print("Cardinality (Unique Values):")
    print(f"{'='*60}")
    for col in df.columns:
        unique = df[col].n_unique()
        pct = (unique / df.height) * 100
        print(f"{col:30} {unique:8} ({pct:5.2f}%)")

    # Basic statistics for numeric columns
    numeric_cols = df.select(cs.numeric()).columns
    if numeric_cols:
        
        print("Numeric Statistics:")
        print(f"{'='*60}")
        print(df.select(numeric_cols).describe())

# Quick data quality checks
quality_report = df.select([
    pl.len().alias('total_rows'),
    pl.all().null_count().name.suffix('_nulls'),
    pl.all().n_unique().name.suffix('_unique'),
    (pl.all().null_count() / pl.len() * 100).name.suffix('_null_pct')
])

# Duplicate detection
duplicate_rows = df.filter(df.is_duplicated())
duplicate_count = df.is_duplicated().sum()

# Find duplicates by key columns
dup_by_key = df.group_by(['id', 'date']).agg([
    pl.len().alias('count')
]).filter(pl.col('count') > 1)

# Identify potential data quality issues
quality_checks = df.select([
    # Null checks
    pl.col('email').is_null().sum().alias('null_emails'),

    # Pattern validation
    pl.col('email').str.contains(r'^[^@]+@[^@]+\.[^@]+$')
        .sum().alias('valid_emails'),

    # Range checks
    (pl.col('age') < 0).sum().alias('negative_ages'),
    (pl.col('amount') < 0).sum().alias('negative_amounts'),

    # Outlier detection (IQR method)
    ((pl.col('value') < pl.col('value').quantile(0.25) -
      1.5 * (pl.col('value').quantile(0.75) - pl.col('value').quantile(0.25))) |
     (pl.col('value') > pl.col('value').quantile(0.75) +
      1.5 * (pl.col('value').quantile(0.75) - pl.col('value').quantile(0.25))))
    .sum().alias('outliers')
])

# Statistical outlier detection using z-score
df_with_outliers = df.with_columns([
    ((pl.col('value') - pl.col('value').mean()) / pl.col('value').std())
    .abs().alias('z_score')
]).filter(pl.col('z_score') > 3)
""", language='python')


def advance_column_selection():
    st.header("🎯 Advanced Column Selection & Expressions")
    st.code("""
# Column selectors (powerful utility)
import polars.selectors as cs

df.select([
    cs.numeric(),           # All numeric columns
    cs.string(),           # All string columns
    cs.temporal(),         # Date/datetime columns
    cs.categorical(),      # Categorical columns
    cs.float(),           # Float columns only
    cs.integer(),         # Integer columns only
    cs.boolean(),         # Boolean columns
])

# Complex selections
df.select([
    cs.starts_with('user_'),
    cs.ends_with('_id'),
    cs.contains('date'),
    cs.matches(r'^col_\d+$'),  # Regex pattern
    ~cs.string(),              # Inverse selection
])

# Combine selectors
df.select(cs.numeric() & ~cs.float())  # Integer columns only
df.select(cs.string() | cs.categorical())  # String or categorical

# Advanced expressions
df.select([
    # Arithmetic operations
    (pl.col('price') * pl.col('quantity')).alias('total'),

    # Conditional expressions
    pl.when(pl.col('status') == 'active')
      .then(pl.col('value'))
      .otherwise(0)
      .alias('active_value'),

    # Multiple conditions
    pl.when(pl.col('score') >= 90).then('A')
      .when(pl.col('score') >= 80).then('B')
      .when(pl.col('score') >= 70).then('C')
      .otherwise('F')
      .alias('grade'),

    # Complex calculations
    ((pl.col('revenue') - pl.col('cost')) / pl.col('revenue') * 100)
    .alias('profit_margin'),

    # Chained string operations
    pl.col('name')
      .str.to_lowercase()
      .str.strip_chars()
      .str.replace_all(r'\s+', ' ')
      .alias('clean_name')
])

# Working with expressions
expr_list = [
    pl.col('value').sum().alias('total'),
    pl.col('value').mean().alias('average'),
    pl.col('value').std().alias('std_dev')
]
df.select(expr_list)

# Dynamic column generation
numeric_cols = df.select(cs.numeric()).columns
normalization_exprs = [
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std())
    .alias(f'{col}_normalized')
    for col in numeric_cols
]
df_normalized = df.with_columns(normalization_exprs)
""", language='python')


def advance_data_types():
    st.header("🔧 Advanced Data Types & Schema Management")
    st.code("""
# Explicit schema definition
schema = {
    'id': pl.UInt64,
    'user_id': pl.UInt32,
    'timestamp': pl.Datetime('us', 'UTC'),
    'event_type': pl.Categorical,
    'properties': pl.Struct({
        'browser': pl.Utf8,
        'os': pl.Utf8,
        'version': pl.Utf8
    }),
    'tags': pl.List(pl.Utf8),
    'metadata': pl.Struct([
        pl.Field('key', pl.Utf8),
        pl.Field('value', pl.Int64)
    ]),
    'amount': pl.Decimal(precision=10, scale=2)
}

# Cast with error handling
df = df.with_columns([
    pl.col('amount').cast(pl.Float64, strict=False),
    pl.col('date_str').str.strptime(pl.Date, '%Y-%m-%d', strict=False),
    pl.col('category').cast(pl.Categorical)
])

# Categorical encoding (memory efficient)
df = df.with_columns([
    pl.col('category').cast(pl.Categorical),
    pl.col('country').cast(pl.Enum(['US', 'UK', 'CA', 'AU']))
])

# Working with nested structures
df = df.with_columns([
    # Create struct
    pl.struct([
        pl.col('street'),
        pl.col('city'),
        pl.col('zipcode')
    ]).alias('address'),

    # Create list
    pl.concat_list([
        pl.col('tag1'),
        pl.col('tag2'),
        pl.col('tag3')
    ]).alias('all_tags')
])

# Extract from struct
df = df.with_columns([
    pl.col('address').struct.field('city').alias('city_extract'),
    pl.col('address').struct.field('zipcode').alias('zip_extract')
])

# Unnest struct columns
df = df.unnest('address')  # Flattens struct into separate columns

# Advanced date/time types
df = df.with_columns([
    pl.col('timestamp').cast(pl.Datetime('ms', 'UTC')),
    pl.duration(days=30).alias('duration'),
    pl.date(2024, 1, 1).alias('fixed_date')
])
""", language='python')


def advance_filtering():
    st.header("🚰 Advanced Filtering & Complex Conditions")
    st.code("""
# Multiple filter strategies
df_filtered = (
    df
    .filter(pl.col('status').is_in(['active', 'pending']))
    .filter(pl.col('amount').is_between(100, 1000, closed='both'))
    .filter(pl.col('created_date') >= datetime(2024, 1, 1))
    .filter(~pl.col('email').str.contains('@test.com'))
)

# Complex boolean logic
df_filtered = df.filter(
    (
        (pl.col('category') == 'A') &
        (pl.col('value') > 100)
    ) | (
        (pl.col('category') == 'B') &
        (pl.col('value') > 200)
    )
)

# Filter with aggregations
df_filtered = df.filter(
    pl.col('value') > pl.col('value').mean() + 2 * pl.col('value').std()
)

# Filter using window functions
df_filtered = (
    df
    .with_columns([
        pl.col('value').rank().over('category').alias('rank')
    ])
    .filter(pl.col('rank') <= 10)
)

# Null handling in filters
df_filtered = df.filter(
    pl.col('optional_field').fill_null(0) > 10
)

# String pattern matching
df_filtered = df.filter(
    pl.col('text')
    .str.contains(r'^[A-Z]{3}\d{4}$')  # Regex pattern
)

# Date range filtering
df_filtered = df.filter(
    pl.col('date').is_between(
        datetime(2024, 1, 1),
        datetime(2024, 12, 31)
    )
)

# Anti-patterns filtering
excluded_emails = ['spam@test.com', 'test@test.com']
df_clean = df.filter(~pl.col('email').is_in(excluded_emails))

# Semi-join pattern (exists in another dataframe)
valid_users = pl.DataFrame({'user_id': [1, 2, 3]})
df_filtered = df.join(valid_users, on='user_id', how='semi')

# Anti-join pattern (not exists in another dataframe)
blocked_users = pl.DataFrame({'user_id': [4, 5, 6]})
df_filtered = df.join(blocked_users, on='user_id', how='anti')
""", language='python')


def advance_aggregations():
    st.header("📈 Advanced Aggregations & Analytics")
    st.code("""
# Complex aggregations with multiple metrics
agg_result = df.group_by('category').agg([
    # Counts
    pl.len().alias('count'),
    pl.col('user_id').n_unique().alias('unique_users'),

    # Central tendency
    pl.col('amount').mean().alias('avg_amount'),
    pl.col('amount').median().alias('median_amount'),
    pl.col('amount').mode().first().alias('mode_amount'),

    # Spread
    pl.col('amount').std().alias('std_amount'),
    pl.col('amount').var().alias('var_amount'),

    # Range
    pl.col('amount').min().alias('min_amount'),
    pl.col('amount').max().alias('max_amount'),
    (pl.col('amount').max() - pl.col('amount').min()).alias('range'),

    # Percentiles
    pl.col('amount').quantile(0.25).alias('q25'),
    pl.col('amount').quantile(0.50).alias('q50'),
    pl.col('amount').quantile(0.75).alias('q75'),
    pl.col('amount').quantile(0.95).alias('q95'),

    # Lists
    pl.col('product').unique().alias('unique_products'),
    pl.col('user_id').list().alias('all_users'),

    # Conditional aggregations
    pl.col('amount').filter(pl.col('status') == 'completed').sum()
        .alias('completed_amount'),
    pl.len().filter(pl.col('is_premium')).alias('premium_count'),

    # First and last values
    pl.col('timestamp').min().alias('first_seen'),
    pl.col('timestamp').max().alias('last_seen'),

    # Struct aggregations
    pl.struct([
        pl.col('amount').sum().alias('total'),
        pl.col('amount').mean().alias('average')
    ]).alias('summary')
])

# Weighted aggregations
df_weighted = df.select([
    (pl.col('value') * pl.col('weight')).sum() / pl.col('weight').sum()
    .alias('weighted_average')
])

# Rolling aggregations with group_by
rolling_metrics = df.group_by_dynamic(
    'date',
    every='1d',
    period='7d',
    by='user_id'
).agg([
    pl.col('value').sum().alias('7day_sum'),
    pl.col('value').mean().alias('7day_avg')
])

# Pivot with multiple aggregations
pivot_result = df.pivot(
    index='date',
    columns='category',
    values=['amount', 'quantity'],
    aggregate_function=['sum', 'mean']
)

# Custom aggregations using expressions
custom_agg = df.group_by('category').agg([
    # Coefficient of variation
    (pl.col('value').std() / pl.col('value').mean() * 100)
    .alias('cv'),

    # Range as percentage of mean
    ((pl.col('value').max() - pl.col('value').min()) /
     pl.col('value').mean() * 100)
    .alias('range_pct'),

    # Percent above threshold
    (pl.col('value').filter(pl.col('value') > 100).len() / pl.len() * 100)
    .alias('pct_above_100')
])

# Multi-level aggregations
df_multi_agg = (
    df
    .group_by(['year', 'quarter', 'category'])
    .agg([pl.col('amount').sum().alias('total')])
    .group_by(['year', 'quarter'])
    .agg([
        pl.col('total').sum().alias('quarter_total'),
        pl.col('category').n_unique().alias('categories')
    ])
)
""", language='python')


def advance_window_function():
    st.header("🪟 Advanced Window Functions")
    st.code("""
# Comprehensive window operations
df_windowed = df.with_columns([
    # Ranking functions
    pl.col('value').rank('ordinal').over('category').alias('rank'),
    pl.col('value').rank('dense').over('category').alias('dense_rank'),

    # Row number
    pl.int_range(pl.len()).over('category').alias('row_num'),

    # Percentile rank
    pl.col('value').rank('average').over('category')
    .truediv(pl.col('value').count().over('category'))
    .alias('percentile'),

    # Cumulative operations
    pl.col('value').cum_sum().over('category').alias('cumsum'),
    pl.col('value').cum_count().over('category').alias('cumcount'),
    pl.col('value').cum_max().over('category').alias('cummax'),
    pl.col('value').cum_min().over('category').alias('cummin'),

    # Lead and lag
    pl.col('value').shift(1).over('category').alias('prev_value'),
    pl.col('value').shift(-1).over('category').alias('next_value'),
    pl.col('value').shift(2).over('category').alias('prev_2_value'),

    # First and last in group
    pl.col('value').first().over('category').alias('first_in_group'),
    pl.col('value').last().over('category').alias('last_in_group'),

    # Running aggregations
    pl.col('value').sum().over('category').alias('group_total'),
    pl.col('value').mean().over('category').alias('group_avg'),

    # Percentage of total
    (pl.col('value') / pl.col('value').sum().over('category') * 100)
    .alias('pct_of_group_total')
])

# Rolling windows with custom periods
df_rolling = df.sort('date').with_columns([
    # Fixed window
    pl.col('value').rolling_mean(window_size=7).alias('7day_ma'),
    pl.col('value').rolling_sum(window_size=30).alias('30day_sum'),
    pl.col('value').rolling_std(window_size=7).alias('7day_std'),
    pl.col('value').rolling_max(window_size=7).alias('7day_max'),

    # Exponential moving average
    pl.col('value').ewm_mean(span=7).alias('ema_7'),

    # Rolling with custom function
    pl.col('value').rolling_map(
        lambda s: s.max() - s.min(),
        window_size=7
    ).alias('7day_range')
])

# Time-based rolling windows
df_time_rolling = df.with_columns([
    pl.col('value')
    .rolling_mean_by('timestamp', window_size='7d')
    .alias('7day_rolling_avg'),

    pl.col('value')
    .rolling_sum_by('timestamp', window_size='30d')
    .alias('30day_rolling_sum')
])

# Multiple window specifications
from polars import Expr

df_multi_window = df.with_columns([
    # Partition by one column, order by another
    pl.col('value').rank().over(['category'], order_by='date')
    .alias('rank_by_date'),

    # Complex window with multiple order columns
    pl.col('value').rank().over(
        partition_by=['category', 'subcategory'],
        order_by=['date', 'value']
    ).alias('complex_rank')
])

# Window with frame specification
df_frame = df.with_columns([
    pl.col('value')
    .rolling_mean(window_size=3, center=True)
    .alias('centered_ma_3')
])
""", language='python')


def advance_joins_relationship():
    st.header("🔗 Advanced Joins & Relationships")
    st.code("""
# All join types with conditions
df_joined = df1.join(
    df2,
    on='key',
    how='inner'  # inner, left, outer, semi, anti, cross
)

# Multiple key joins
df_joined = df1.join(
    df2,
    on=['key1', 'key2', 'key3'],
    how='left'
)

# Join with different column names
df_joined = df1.join(
    df2,
    left_on='user_id',
    right_on='id',
    how='left'
)

# Join with suffix handling
df_joined = df1.join(
    df2,
    on='key',
    suffix='_right'
)

# Coalesce duplicate columns after join
df_joined = df1.join(df2, on='key', how='outer').with_columns([
    pl.coalesce(['col1', 'col1_right']).alias('col1'),
    pl.coalesce(['col2', 'col2_right']).alias('col2')
]).drop(['col1_right', 'col2_right'])

# Join with inequality conditions (using cross join + filter)
df_intervals = df1.join(df2, how='cross').filter(
    (pl.col('start_date') <= pl.col('event_date')) &
    (pl.col('event_date') <= pl.col('end_date'))
)

# Self-join for hierarchical data
df_hierarchy = df.join(
    df,
    left_on='manager_id',
    right_on='employee_id',
    suffix='_manager'
)

# Asof join (time-series joins)
df_asof = df1.join_asof(
    df2,
    on='timestamp',
    by='symbol',
    strategy='backward'  # backward, forward, nearest
)

# Semi-join (filter rows that exist in another df)
df_exists = df1.join(df2, on='key', how='semi')

# Anti-join (filter rows that don't exist in another df)
df_not_exists = df1.join(df2, on='key', how='anti')

# Join with aggregations
df_with_stats = df.join(
    df.group_by('category').agg([
        pl.col('value').mean().alias('category_avg'),
        pl.col('value').std().alias('category_std')
    ]),
    on='category'
).with_columns([
    ((pl.col('value') - pl.col('category_avg')) / pl.col('category_std'))
    .alias('z_score')
])

# Deduplication before join (avoid cartesian product)
df_joined = (
    df1.unique(subset=['key'])
    .join(
        df2.unique(subset=['key']),
        on='key'
    )
)

# Multiple joins in pipeline
result = (
    df1
    .join(df2, on='user_id', how='left')
    .join(df3, on='product_id', how='left')
    .join(df4, left_on='category_id', right_on='id', how='left')
)
""", language='python')


def advance_data_cleaning():
    st.header("🧹 Advanced Data Cleaning & Transformation")
    st.code("""# Comprehensive text cleaning
df_clean = df.with_columns([
    # Standardize text
    pl.col('text')
    .str.to_lowercase()
    .str.strip_chars()
    .str.replace_all(r'\s+', ' ')
    .str.replace_all(r'[^\w\s]', '')
    .alias('clean_text'),

    # Clean emails
    pl.col('email')
    .str.to_lowercase()
    .str.strip_chars()
    .str.replace_all(r'\s+', '')
    .alias('clean_email'),

    # Extract domain from email
    pl.col('email')
    .str.extract(r'@(.+)$', 1)
    .alias('email_domain'),

    # Phone number cleaning
    pl.col('phone')
    .str.replace_all(r'[^\d]', '')
    .alias('phone_clean'),

    # URL extraction
    pl.col('text')
    .str.extract_all(r'https?://[^\s]+')
    .alias('urls')
])

# Handle mixed data types
df_parsed = df.with_columns([
    # Parse amounts with various formats
    pl.when(pl.col('amount').str.contains(r'^\d+\.?\d*$'))
      .then(pl.col('amount').cast(pl.Float64))
      .when(pl.col('amount').str.contains(r'^\$[\d,]+\.?\d*$'))
      .then(pl.col('amount').str.replace_all(r'[\$,]', '').cast(pl.Float64))
      .when(pl.col('amount').str.contains(r'^\([\d,]+\.?\d*\)$'))
      .then(
          pl.col('amount')
          .str.replace_all(r'[(),]', '')
          .cast(pl.Float64) * -1
      )
      .otherwise(None)
      .alias('amount_parsed')
])

# Advanced missing value imputation
df_imputed = df.with_columns([
    # Forward fill
    pl.col('value').fill_null(strategy='forward'),

    # Backward fill
    pl.col('value').fill_null(strategy='backward'),

    # Mean imputation
    pl.col('value').fill_null(pl.col('value').mean()),

    # Median imputation
    pl.col('value').fill_null(pl.col('value').median()),

    # Group-wise imputation
    pl.col('value').fill_null(
        pl.col('value').mean().over('category')
    ).alias('value_imputed'),

    # Interpolation
    pl.col('value').interpolate(),

    # Custom conditional imputation
    pl.when(pl.col('value').is_null() & (pl.col('category') == 'A'))
      .then(100)
      .when(pl.col('value').is_null() & (pl.col('category') == 'B'))
      .then(200)
      .otherwise(pl.col('value'))
      .alias('value_conditional')
])

# Outlier handling
df_outlier_handled = df.with_columns([
    # Cap outliers using IQR
    pl.when(
        pl.col('value') > pl.col('value').quantile(0.75) +
        1.5 * (pl.col('value').quantile(0.75) - pl.col('value').quantile(0.25))
    ).then(
        pl.col('value').quantile(0.75) +
        1.5 * (pl.col('value').quantile(0.75) - pl.col('value').quantile(0.25))
    ).when(
        pl.col('value') < pl.col('value').quantile(0.25) -
        1.5 * (pl.col('value').quantile(0.75) - pl.col('value').quantile(0.25))
    ).then(
        pl.col('value').quantile(0.25) -
        1.5 * (pl.col('value').quantile(0.75) - pl.col('value').quantile(0.25))
    ).otherwise(pl.col('value'))
    .alias('value_capped')
])

# Data standardization and normalization
df_normalized = df.with_columns([
    # Z-score normalization
    ((pl.col('value') - pl.col('value').mean()) / pl.col('value').std())
    .alias('value_zscore'),

    # Min-max normalization
    ((pl.col('value') - pl.col('value').min()) /
     (pl.col('value').max() - pl.col('value').min()))
    .alias('value_minmax'),

    # Log transformation
    pl.col('value').log().alias('value_log'),

    # Group-wise normalization
    ((pl.col('value') - pl.col('value').mean().over('category')) /
     pl.col('value').std().over('category'))
    .alias('value_zscore_by_category')
])

# Deduplication with priority
df_dedup = (
    df
    .with_columns([
        pl.col('updated_at').rank('ordinal', descending=True)
        .over('id')
        .alias('recency_rank')
    ])
    .filter(pl.col('recency_rank') == 1)
    .drop('recency_rank')
)""", language='python')


def advance_date_time():
    st.header("📅 Advanced Date & Time Operations")
    st.code("""
# Comprehensive date operations
df_dates = df.with_columns([
    # Extract components
    pl.col('date').dt.year().alias('year'),
    pl.col('date').dt.month().alias('month'),
    pl.col('date').dt.quarter().alias('quarter'),
    pl.col('date').dt.week().alias('week'),
    pl.col('date').dt.weekday().alias('weekday'),
    pl.col('date').dt.day().alias('day'),
    pl.col('date').dt.ordinal_day().alias('day_of_year'),

    # Time components
    pl.col('timestamp').dt.hour().alias('hour'),
    pl.col('timestamp').dt.minute().alias('minute'),
    pl.col('timestamp').dt.second().alias('second'),

    # Date calculations
    pl.col('date').dt.offset_by('1mo').alias('next_month'),
    pl.col('date').dt.offset_by('-7d').alias('week_ago'),
    (pl.col('end_date') - pl.col('start_date')).dt.total_days().alias('days_diff'),
    (pl.col('end_date') - pl.col('start_date')).dt.total_hours().alias('hours_diff'),

    # Start/end of periods
    pl.col('date').dt.month_start().alias('month_start'),
    pl.col('date').dt.month_end().alias('month_end'),
    pl.col('date').dt.truncate('1mo').alias('month_truncated'),
    pl.col('date').dt.truncate('1w').alias('week_truncated'),

    # Business logic
    pl.when(pl.col('date').dt.weekday().is_in([6, 7]))
      .then(True)
      .otherwise(False)
      .alias('is_weekend'),

    # Fiscal calendar
    pl.when(pl.col('date').dt.month() <= 3)
      .then(pl.col('date').dt.year())
      .otherwise(pl.col('date').dt.year() + 1)
      .alias('fiscal_year'),
])

# Parse multiple date formats
df_parsed_dates = df.with_columns([
    pl.coalesce([
        pl.col('date_str').str.strptime(pl.Date, '%Y-%m-%d', strict=False),
        pl.col('date_str').str.strptime(pl.Date, '%m/%d/%Y', strict=False),
        pl.col('date_str').str.strptime(pl.Date, '%d-%b-%Y', strict=False),
        pl.col('date_str').str.strptime(pl.Date, '%Y%m%d', strict=False)
    ]).alias('parsed_date')
])

# Timezone operations
df_tz = df.with_columns([
    pl.col('timestamp').dt.replace_time_zone('UTC').alias('utc_time'),
    pl.col('timestamp').dt.convert_time_zone('America/New_York').alias('ny_time'),
    pl.col('timestamp').dt.cast_time_unit('ms').alias('ms_timestamp')
])

# Date ranges and sequences
date_range = pl.date_range(
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    interval='1d'
)

# Business day calendar
df_business = pl.DataFrame({
    'date': date_range
}).filter(
    ~pl.col('date').dt.weekday().is_in([6, 7])
)

# Cohort age calculation
df_cohort = df.with_columns([
    (pl.col('current_date') - pl.col('signup_date'))
    .dt.total_days()
    .truediv(30)
    .floor()
    .alias('months_since_signup')
])

# Rolling date windows for analytics
df_rolling_dates = df.with_columns([
    pl.col('value')
    .rolling_mean_by('date', window_size='30d')
    .alias('trailing_30d_avg'),

    pl.col('value')
    .rolling_sum_by('date', window_size='90d')
    .alias('trailing_90d_sum')
])
""", language='python')


def advance_io_operations():
    st.header("💾 Advanced I/O Operations")
    st.code("""
# Optimized Parquet writing
df.write_parquet(
    'output.parquet',
    compression='zstd',  # or 'snappy', 'gzip', 'lz4'
    compression_level=3,
    statistics=True,
    row_group_size=None,  # Auto
    data_page_size=None
)

# Partitioned writes
df.write_parquet(
    'partitioned_output',
    partition_by=['year', 'month']
)

# Streaming sink for large datasets (memory efficient)
(
    pl.scan_parquet('large_input.parquet')
    .filter(pl.col('date') >= datetime(2024, 1, 1))
    .group_by('category')
    .agg([pl.col('value').sum()])
    .sink_parquet('output.parquet')
)

# Multiple sinks in one pass
query = pl.scan_csv('input.csv')
query.sink_parquet('output.parquet')
query.sink_csv('output.csv')

# Delta Lake integration
df.write_delta(
    'path/to/delta',
    mode='append',  # or 'overwrite'
    overwrite_schema=False
)

# Read with predicate pushdown
lazy_df = (
    pl.scan_parquet('data/*.parquet')
    .filter(pl.col('year') == 2024)  # Pushed to file scan
    .filter(pl.col('month') >= 6)    # Pushed to file scan
)

# Optimized CSV reading for large files
df = pl.read_csv(
    'large_file.csv',
    low_memory=False,
    rechunk=True,
    use_pyarrow=False,
    n_threads=8
)

# Streaming CSV read
batches = pl.read_csv_batched(
    'huge_file.csv',
    batch_size=1_000_000
)

for batch_df in batches:
    # Process each batch
    result = process_batch(batch_df)
    result.write_parquet('output.parquet', append=True)

# Database operations
# Write to database
df.write_database(
    table_name='my_table',
    connection='postgresql://user:pass@localhost/db',
    if_table_exists='append'  # or 'replace', 'fail'
)

# Read with SQL query
df = pl.read_database(
    query="SELECT * FROM table WHERE date >= '2024-01-01' AND status IN('active', 'pending')",
    connection='postgresql://user:pass@localhost/db'
)

# Cloud storage operations
# S3
df = pl.read_parquet('s3://bucket/path/*.parquet')
df.write_parquet('s3://bucket/output/')

# GCS
df = pl.scan_parquet('gs://bucket/data/*.parquet').collect()

# Azure
df = pl.read_parquet('az://container/path/file.parquet')
""", language='python')


def analytics_patterns():
    st.header("📊 Analytics Patterns")
    st.code("""
ohort_df = (
    df
    .with_columns([
        pl.col('signup_date').dt.truncate('1mo').alias('cohort_month'),
        ((pl.col('event_date') - pl.col('signup_date'))
         .dt.total_days() / 30).floor().cast(pl.Int32).alias('months_since_signup')
    ])
    .group_by(['cohort_month', 'months_since_signup'])
    .agg([
        pl.col('user_id').n_unique().alias('active_users')
    ])
    .pivot(
        index='cohort_month',
        columns='months_since_signup',
        values='active_users'
    )
)

# Retention analysis
retention = (
    df
    .group_by('signup_month')
    .agg([
        pl.col('user_id').n_unique().alias('cohort_size'),
        pl.col('user_id')
          .filter(pl.col('months_active') >= 1)
          .n_unique().alias('retained_month_1'),
        pl.col('user_id')
          .filter(pl.col('months_active') >= 3)
          .n_unique().alias('retained_month_3'),
        pl.col('user_id')
          .filter(pl.col('months_active') >= 6)
          .n_unique().alias('retained_month_6')
    ])
    .with_columns([
        (pl.col('retained_month_1') / pl.col('cohort_size') * 100)
        .alias('retention_rate_month_1'),
        (pl.col('retained_month_3') / pl.col('cohort_size') * 100)
        .alias('retention_rate_month_3'),
        (pl.col('retained_month_6') / pl.col('cohort_size') * 100)
        .alias('retention_rate_month_6')
    ])
)

# RFM Analysis
rfm = (
    df
    .group_by('customer_id')
    .agg([
        (pl.col('order_date').max().dt.epoch_days() -
         pl.lit(datetime.now()).dt.epoch_days()).abs().alias('recency'),
        pl.col('order_id').n_unique().alias('frequency'),
        pl.col('amount').sum().alias('monetary')
    ])
    .with_columns([
        pl.col('recency').qcut(5, labels=['1','2','3','4','5']).alias('r_score'),
        pl.col('frequency').qcut(5, labels=['1','2','3','4','5']).alias('f_score'),
        pl.col('monetary').qcut(5, labels=['1','2','3','4','5']).alias('m_score')
    ])
    .with_columns([
        (pl.col('r_score').cast(pl.Int32) +
         pl.col('f_score').cast(pl.Int32) +
         pl.col('m_score').cast(pl.Int32)).alias('rfm_score')
    ])
)

# Funnel analysis
funnel = (
    df
    .group_by('user_id')
    .agg([
        pl.col('event').filter(pl.col('event') == 'page_view')
          .count().alias('step_1_views'),
        pl.col('event').filter(pl.col('event') == 'add_to_cart')
          .count().alias('step_2_carts'),
        pl.col('event').filter(pl.col('event') == 'checkout')
          .count().alias('step_3_checkouts'),
        pl.col('event').filter(pl.col('event') == 'purchase')
          .count().alias('step_4_purchases')
    ])
    .select([
        (pl.col('step_1_views') > 0).sum().alias('users_viewed'),
        (pl.col('step_2_carts') > 0).sum().alias('users_added_cart'),
        (pl.col('step_3_checkouts') > 0).sum().alias('users_checkout'),
        (pl.col('step_4_purchases') > 0).sum().alias('users_purchased')
    ])
    .with_columns([
        (pl.col('users_added_cart') / pl.col('users_viewed') * 100)
        .alias('view_to_cart_rate'),
        (pl.col('users_checkout') / pl.col('users_added_cart') * 100)
        .alias('cart_to_checkout_rate'),
        (pl.col('users_purchased') / pl.col('users_checkout') * 100)
        .alias('checkout_to_purchase_rate')
    ])
)

# Time series decomposition
ts_decomp = (
    df
    .sort('date')
    .with_columns([
        # Trend (rolling average)
        pl.col('value').rolling_mean(window_size=30).alias('trend'),
    ])
    .with_columns([
        # Detrended
        (pl.col('value') - pl.col('trend')).alias('detrended')
    ])
    .with_columns([
        # Seasonal component (by month)
        pl.col('detrended').mean().over(pl.col('date').dt.month())
        .alias('seasonal')
    ])
    .with_columns([
        # Residual
        (pl.col('detrended') - pl.col('seasonal')).alias('residual')
    ])
)

# ABC Analysis
abc_analysis = (
    df
    .group_by('product_id')
    .agg([pl.col('revenue').sum().alias('total_revenue')])
    .sort('total_revenue', descending=True)
    .with_columns([
        pl.col('total_revenue').cum_sum().alias('cumulative_revenue'),
        pl.col('total_revenue').sum().alias('total')
    ])
    .with_columns([
        (pl.col('cumulative_revenue') / pl.col('total') * 100)
        .alias('cumulative_pct')
    ])
    .with_columns([
        pl.when(pl.col('cumulative_pct') <= 80).then('A')
          .when(pl.col('cumulative_pct') <= 95).then('B')
          .otherwise('C')
          .alias('abc_class')
    ])
)
""", language='python')


def performance_optimization():
    st.header("⚡ Performance Optimization")
    st.code("""
# Query optimization with explain
query = (
    pl.scan_parquet('data/*.parquet')
    .filter(pl.col('date') >= datetime(2024, 1, 1))
    .group_by('category')
    .agg([pl.col('value').sum()])
)

# Show query plan
print(query.explain())

# Execute with streaming
result = query.collect(streaming=True)

# Predicate pushdown (automatic in lazy mode)
optimized_query = (
    pl.scan_parquet('data.parquet')
    .filter(pl.col('year') == 2024)  # Pushed to file read
    .filter(pl.col('month') >= 6)    # Pushed to file read
    .select(['id', 'value'])          # Column pruning
)

# Efficient joins with proper ordering
large_df = pl.scan_parquet('large.parquet')
small_df = pl.scan_parquet('small.parquet').collect()

# Join small to large (not vice versa)
result = large_df.join(small_df, on='key').collect()

# Use categorical for high-cardinality strings
df = df.with_columns([
    pl.col('category').cast(pl.Categorical)
])

# Rechunk for better performance
df = df.rechunk()

# Memory-mapped files for large datasets
df = pl.read_parquet('large.parquet', memory_map=True)

# Parallel execution settings
pl.Config.set_global_string_cache(True)  # For categorical across dataframes
pl.thread_pool_size()  # Check thread count

# Optimize dtypes
df = df.with_columns([
    pl.col('small_int').cast(pl.UInt8),      # 0-255
    pl.col('medium_int').cast(pl.UInt16),    # 0-65535
    pl.col('id').cast(pl.UInt32),            # 0-4B
    pl.col('large_id').cast(pl.UInt64)       # 0-18 quintillion
])

# Batch processing for memory efficiency
def process_in_batches(file_path: str, batch_size: int = 1_000_000):
    # Process large file in batches

    scanner = pl.scan_parquet(file_path)
    total_rows = scanner.collect().height

    for offset in range(0, total_rows, batch_size):
        batch = (
            scanner
            .slice(offset, batch_size)
            .collect()
        )
        # Process batch
        yield batch

# Use sink for write-only workflows
(
    pl.scan_csv('input.csv')
    .filter(pl.col('valid') == True)
    .select(['id', 'value', 'date'])
    .sink_parquet('output.parquet')  # Never materializes in memory
)
""", language='python')


def security_data_governance():
    st.header("🔐 Security & Data Governance")
    st.code("""
# PII masking
df_masked = df.with_columns([
    # Mask email
    pl.col('email').str.replace(r'^(.{2}).*(@.*), r'\1***\2').alias('email_masked'),

    # Mask phone
    pl.col('phone').str.replace(r'(\d{3})\d{3}(\d{4})', r'\1***\2').alias('phone_masked'),

    # Hash sensitive data
    pl.col('ssn').hash().alias('ssn_hash'),

    # Redact credit card
    pl.col('cc_number').str.slice(-4).str.concat(pl.lit('************'))
    .alias('cc_last_4')
])

# Row-level security
def apply_row_level_security(df: pl.DataFrame,
                              user_role: str,
                              user_id: int) -> pl.DataFrame:
    # Filter data based on user permissions

    if user_role == 'admin':
        return df
    elif user_role == 'manager':
        return df.filter(pl.col('department_id') == user_id)
    else:
        return df.filter(pl.col('created_by') == user_id)

# Audit logging
df_with_audit = df.with_columns([
    pl.lit('etl_pipeline').alias('processed_by'),
    pl.lit(datetime.now()).alias('processed_at'),
    pl.lit('v1.0').alias('pipeline_version')
])

# Data lineage tracking
metadata = {
    'source': 'raw_events',
    'transform': 'aggregation',
    'destination': 'analytics_mart',
    'timestamp': datetime.now(),
    'row_count': df.height
}
""", language='python')


def best_practice():
    st.header("🎯 Best Practices for Data Engineers")
    st.markdown("### Lazy Evaluation")
    st.code("""
# GOOD: Use lazy evaluation for large datasets
result = (
    pl.scan_parquet('large_file.parquet')
    .filter(pl.col('date') >= datetime(2024, 1, 1))
    .group_by('category')
    .agg([pl.col('value').sum()])
    .collect()  # Only execute at the end
)

# BAD: Eager evaluation forces intermediate materialization
df = pl.read_parquet('large_file.parquet')
df = df.filter(pl.col('date') >= datetime(2024, 1, 1))
df = df.group_by('category').agg([pl.col('value').sum()])
""", language='python')

    st.markdown("### Schema Validation")
    st.code("""
# Robust ETL with error handling
def safe_etl(source: str, target: str):
    try:
        df = pl.scan_parquet(source)

        # Validate schema
        required_cols = {'id', 'date', 'value'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

        # Transform with strict=False for casting
        result = (
            df
            .with_columns([
                pl.col('value').cast(pl.Float64, strict=False)
            ])
            .filter(pl.col('value').is_not_null())
        )

        # Write with validation
        result.sink_parquet(target)

    except Exception as e:
        logging.error(f"ETL failed: {e}")
        raise
""", language='python')


def quick_refences():
    st.header("📚 Quick Reference")
    st.markdown("""
### Polars vs Pandas

```python
# Pandas                              # Polars
df.groupby('col').sum()               df.group_by('col').agg(pl.all().sum())
df.apply(lambda x: x**2)              df.with_columns(pl.all().pow(2))
df.fillna(0)                          df.fill_null(0)
df.drop_duplicates()                  df.unique()
df.query('col > 5')                   df.filter(pl.col('col') > 5)
df.merge(df2, on='key')               df.join(df2, on='key')
df['col'].rolling(7).mean()           df.with_columns(pl.col('col').rolling_mean(7))
```

### Common Patterns

```python
# Column exists check
'column_name' in df.columns

# Convert to different format
df.to_pandas()
df.to_numpy()
df.to_dict()
df.write_json('output.json')

# Sample data
df.sample(n=1000, seed=42)
df.head(10)
df.tail(10)

# Shape and size
df.shape  # (rows, cols)
df.height  # rows
df.width   # cols
df.estimated_size('mb')  # memory usage
```

## 💡 Pro Tips

1. **Always use lazy evaluation** for large datasets - it's Polars' superpower
2. **Prefer Parquet over CSV** - 10-100x faster reads/writes
3. **Use categorical types** for repeated strings - massive memory savings
4. **Leverage predicate pushdown** - filters before reading = faster queries
5. **Cast to optimal dtypes** - UInt8/16/32 instead of Int64 where possible
6. **Use `scan_*` instead of `read_*`** for files > 1GB
7. **Sink for write-only** workflows - never materializes in memory
8. **Rechunk before expensive operations** - better performance
9. **Use `pl.col()` instead of strings** - better type safety
10. **Enable streaming** for queries on massive datasets
11. **Profile with `.explain()`** - understand query execution
12. **Use column selectors (cs)** - cleaner, more maintainable code
13. **Batch process huge files** - avoid OOM errors
14. **Validate schemas explicitly** - catch errors early
15. **Use expressions over map_elements** - 10-1000x faster
""")
