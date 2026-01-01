import streamlit as st


def essential_commands():
    st.header("🚀 Getting Started")
    st.code("""
# Connect to PostgreSQL
psql -U username -d database_name

# Connect with password
psql -U username -W -d database_name

# Exit
\q
""", language='bash')

    st.markdown("### Essential psql Commands")
    st.code(r"""
\l                          -- List databases
\c database_name            -- Switch database
\dt                         -- List tables
\d table_name               -- Show table structure
\d+ table_name              -- Detailed table info with sizes

\x                          -- Toggle vertical display (great for wide results)
\timing on                  -- Show query execution time
\i script.sql               -- Run SQL file
\copy table TO 'file.csv' CSV HEADER  -- Export to CSV
""", language='bash')


def exploring_data():
    st.header("🔍 Exploring Data")
    st.code("""
-- See table structure
\d table_name

-- Count rows
SELECT COUNT(*) FROM table_name;

-- Sample data
SELECT * FROM table_name LIMIT 10;

-- Random sample
SELECT * FROM table_name ORDER BY RANDOM() LIMIT 100;

-- Column data types and nulls
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'your_table';

-- Basic statistics
SELECT
    COUNT(*) as total_rows,
    COUNT(DISTINCT user_id) as unique_users,
    MIN(created_at) as earliest_date,
    MAX(created_at) as latest_date
FROM table_name;
""", language='sql')

    st.markdown("### Checking for Data Quality Issues")
    st.code("""
-- Find null values
SELECT
    COUNT(*) as total,
    COUNT(column_name) as non_null,
    COUNT(*) - COUNT(column_name) as null_count,
    ROUND(100.0 * (COUNT(*) - COUNT(column_name)) / COUNT(*), 2) as null_percent
FROM table_name;

-- Find duplicates
SELECT
    email,
    COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1
ORDER BY count DESC;

-- Check value ranges
SELECT
    MIN(age) as min_age,
    MAX(age) as max_age,
    AVG(age) as avg_age,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY age) as median_age
FROM users;

-- Find outliers (values beyond 3 standard deviations)
WITH stats AS (
    SELECT
        AVG(value) as mean,
        STDDEV(value) as stddev
    FROM measurements
)
SELECT *
FROM measurements, stats
WHERE ABS(value - mean) > 3 * stddev;

-- Check for distinct values (useful for categorical data)
SELECT
    column_name,
    COUNT(*) as frequency,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as percentage
FROM table_name
GROUP BY column_name
ORDER BY frequency DESC;
""", language='sql')


def filtering_selecting_data():
    st.header("🎯 Filtering & Selecting Data")
    st.markdown("### WHERE Clause Essentials")

    st.code("""
-- Basic comparisons
SELECT * FROM sales WHERE amount > 1000;
SELECT * FROM sales WHERE status = 'completed';
SELECT * FROM sales WHERE created_at >= '2024-01-01';

-- Multiple conditions
SELECT * FROM sales
WHERE amount > 1000
  AND status = 'completed'
  AND created_at >= '2024-01-01';

-- OR conditions
SELECT * FROM products
WHERE category = 'electronics'
   OR category = 'computers';

-- IN operator (cleaner than multiple ORs)
SELECT * FROM products
WHERE category IN ('electronics', 'computers', 'phones');

-- NOT IN
SELECT * FROM users
WHERE status NOT IN ('banned', 'suspended');

-- BETWEEN
SELECT * FROM sales
WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';

-- Pattern matching
SELECT * FROM users WHERE email LIKE '%@gmail.com';
SELECT * FROM users WHERE email ILIKE '%@gmail.com';  -- Case insensitive
SELECT * FROM products WHERE name LIKE 'iPhone%';      -- Starts with

-- NULL checks
SELECT * FROM users WHERE phone_number IS NULL;
SELECT * FROM users WHERE phone_number IS NOT NULL;

-- Date filtering
SELECT * FROM sales WHERE created_at > NOW() - INTERVAL '7 days';
SELECT * FROM sales WHERE created_at > NOW() - INTERVAL '1 month';
SELECT * FROM sales WHERE EXTRACT(YEAR FROM created_at) = 2024;
SELECT * FROM sales WHERE DATE_TRUNC('month', created_at) = '2024-01-01';
""", language='sql')


def aggregations_group():
    st.header("📊 Aggregations & GROUP BY")
    st.markdown("### Basic Aggregations")
    st.code("""
-- Common aggregate functions
SELECT
    COUNT(*) as total_orders,
    SUM(amount) as total_revenue,
    AVG(amount) as average_order,
    MIN(amount) as smallest_order,
    MAX(amount) as largest_order,
    STDDEV(amount) as std_deviation
FROM orders;

-- Count distinct
SELECT
    COUNT(DISTINCT user_id) as unique_customers,
    COUNT(DISTINCT product_id) as unique_products
FROM orders;
""", language='sql')

    st.markdown("### GROUP BY for Summaries")
    st.code("""
- Group by single column
SELECT
    category,
    COUNT(*) as product_count,
    AVG(price) as avg_price
FROM products
GROUP BY category
ORDER BY product_count DESC;

-- Group by multiple columns
SELECT
    category,
    brand,
    COUNT(*) as product_count,
    AVG(price) as avg_price
FROM products
GROUP BY category, brand
ORDER BY category, product_count DESC;

-- Group by date parts
SELECT
    DATE_TRUNC('month', order_date) as month,
    COUNT(*) as order_count,
    SUM(amount) as monthly_revenue
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- Group by day of week
SELECT
    TO_CHAR(order_date, 'Day') as day_name,
    EXTRACT(DOW FROM order_date) as day_number,
    COUNT(*) as order_count
FROM orders
GROUP BY day_name, day_number
ORDER BY day_number;

-- HAVING clause (filter after grouping)
SELECT
    user_id,
    COUNT(*) as order_count,
    SUM(amount) as total_spent
FROM orders
GROUP BY user_id
HAVING COUNT(*) >= 5  -- Only users with 5+ orders
ORDER BY total_spent DESC;
""", language='sql')

    st.markdown("### Advanced Aggregations")
    st.code("""
-- Percentiles
SELECT
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) as q1,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) as q3,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY salary) as p90
FROM employees;

-- Mode (most common value)
SELECT
    MODE() WITHIN GROUP (ORDER BY category) as most_common_category
FROM products;

-- String aggregation
SELECT
    user_id,
    STRING_AGG(product_name, ', ' ORDER BY order_date) as products_ordered
FROM orders
GROUP BY user_id;

-- Array aggregation
SELECT
    category,
    ARRAY_AGG(product_name ORDER BY price DESC) as products
FROM products
GROUP BY category;

-- Conditional aggregation (FILTER clause)
SELECT
    DATE_TRUNC('month', order_date) as month,
    COUNT(*) as total_orders,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_orders,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled_orders,
    SUM(amount) FILTER (WHERE status = 'completed') as completed_revenue
FROM orders
GROUP BY month
ORDER BY month;

-- Using CASE for pivoting
SELECT
    DATE_TRUNC('month', order_date) as month,
    COUNT(*) as total_orders,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
    SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) as revenue
FROM orders
GROUP BY month;
""", language='sql')


def join_combine():
    st.header("🔗 JOINs for Combining Data")
    st.markdown("### Basic JOIN Types")
    st.code("""
-- INNER JOIN (only matching rows)
SELECT
    o.order_id,
    o.order_date,
    u.username,
    u.email
FROM orders o
INNER JOIN users u ON o.user_id = u.id;

-- LEFT JOIN (all from left, matching from right)
SELECT
    u.username,
    COUNT(o.order_id) as order_count,
    COALESCE(SUM(o.amount), 0) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username;

-- Multiple joins
SELECT
    o.order_id,
    u.username,
    p.product_name,
    p.price
FROM orders o
INNER JOIN users u ON o.user_id = u.id
INNER JOIN products p ON o.product_id = p.id;

-- Self join (compare rows within same table)
SELECT
    e1.name as employee,
    e2.name as manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.id;
""", language='sql')

    st.markdown("### Practical JOIN Examples")
    st.code("""
-- Find customers with no orders
SELECT u.*
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.order_id IS NULL;

-- Find products never ordered
SELECT p.*
FROM products p
LEFT JOIN order_items oi ON p.id = oi.product_id
WHERE oi.order_id IS NULL;

-- Customer lifetime value
SELECT
    u.username,
    u.email,
    COUNT(o.order_id) as total_orders,
    SUM(o.amount) as lifetime_value,
    AVG(o.amount) as avg_order_value,
    MAX(o.order_date) as last_order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.username, u.email
ORDER BY lifetime_value DESC NULLS LAST;
""", language='sql')


def window_function():
    st.header("📈 Window Functions (Advanced Analytics)")
    st.markdown("### Running Totals & Moving Averages")
    st.code("""
-- Running total
SELECT
    date,
    revenue,
    SUM(revenue) OVER (ORDER BY date) as running_total
FROM daily_sales
ORDER BY date;

-- Moving average (7-day)
SELECT
    date,
    revenue,
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7day
FROM daily_sales
ORDER BY date;

-- Year-to-date total
SELECT
    date,
    revenue,
    SUM(revenue) OVER (
        PARTITION BY EXTRACT(YEAR FROM date)
        ORDER BY date
    ) as ytd_total
FROM daily_sales
ORDER BY date;
""", language='sql')

    st.markdown("### Ranking & Row Numbers")
    st.code("""
-- Row number
SELECT
    product_name,
    sales,
    ROW_NUMBER() OVER (ORDER BY sales DESC) as rank
FROM products;

-- Rank with ties
SELECT
    product_name,
    sales,
    RANK() OVER (ORDER BY sales DESC) as rank,
    DENSE_RANK() OVER (ORDER BY sales DESC) as dense_rank
FROM products;

-- Rank within groups
SELECT
    category,
    product_name,
    sales,
    RANK() OVER (PARTITION BY category ORDER BY sales DESC) as rank_in_category
FROM products;

-- Top N per group
SELECT *
FROM (
    SELECT
        category,
        product_name,
        sales,
        RANK() OVER (PARTITION BY category ORDER BY sales DESC) as rank
    FROM products
) ranked
WHERE rank <= 3;  -- Top 3 per category

-- Percentile ranking
SELECT
    product_name,
    sales,
    PERCENT_RANK() OVER (ORDER BY sales) as percentile,
    NTILE(4) OVER (ORDER BY sales) as quartile
FROM products;
""", language='sql')

    st.markdown("### LAG & LEAD (Compare with Previous/Next Rows)")
    st.code("""
-- Compare with previous period
SELECT
    date,
    revenue,
    LAG(revenue) OVER (ORDER BY date) as prev_day_revenue,
    revenue - LAG(revenue) OVER (ORDER BY date) as day_change,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY date)) /
          LAG(revenue) OVER (ORDER BY date), 2) as percent_change
FROM daily_sales
ORDER BY date;

-- Month-over-month growth
SELECT
    DATE_TRUNC('month', date) as month,
    SUM(revenue) as monthly_revenue,
    LAG(SUM(revenue)) OVER (ORDER BY DATE_TRUNC('month', date)) as prev_month,
    SUM(revenue) - LAG(SUM(revenue)) OVER (ORDER BY DATE_TRUNC('month', date)) as change
FROM daily_sales
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;

-- Next value (LEAD)
SELECT
    date,
    revenue,
    LEAD(revenue) OVER (ORDER BY date) as next_day_revenue
FROM daily_sales;
""", language='sql')

    st.markdown("### First & Last Values")
    st.code("""
-- First and last in each group
SELECT
    user_id,
    order_date,
    amount,
    FIRST_VALUE(order_date) OVER (
        PARTITION BY user_id ORDER BY order_date
    ) as first_order_date,
    LAST_VALUE(order_date) OVER (
        PARTITION BY user_id
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as last_order_date
FROM orders;
""", language='sql')


def date_time_operations():
    st.header("📅 Date & Time Operations")
    st.markdown("### Date Calculations")
    st.code("""
-- Current date/time
SELECT NOW();                           -- Current timestamp
SELECT CURRENT_DATE;                    -- Current date
SELECT CURRENT_TIME;                    -- Current time

-- Date arithmetic
SELECT NOW() + INTERVAL '7 days';       -- Add 7 days
SELECT NOW() - INTERVAL '1 month';      -- Subtract 1 month
SELECT NOW() + INTERVAL '2 years';      -- Add 2 years
SELECT NOW() - INTERVAL '3 hours';      -- Subtract 3 hours

-- Date difference
SELECT AGE('2024-12-31', '2024-01-01');  -- Interval between dates
SELECT '2024-12-31'::date - '2024-01-01'::date as days_between;

-- Extract parts
SELECT
    EXTRACT(YEAR FROM order_date) as year,
    EXTRACT(MONTH FROM order_date) as month,
    EXTRACT(DAY FROM order_date) as day,
    EXTRACT(DOW FROM order_date) as day_of_week,  -- 0=Sunday
    EXTRACT(QUARTER FROM order_date) as quarter,
    EXTRACT(WEEK FROM order_date) as week_number
FROM orders;

-- Truncate dates
SELECT
    DATE_TRUNC('year', order_date) as year,
    DATE_TRUNC('month', order_date) as month,
    DATE_TRUNC('week', order_date) as week,
    DATE_TRUNC('day', order_date) as day
FROM orders;

-- Format dates
SELECT
    TO_CHAR(order_date, 'YYYY-MM-DD') as date_iso,
    TO_CHAR(order_date, 'Mon DD, YYYY') as date_readable,
    TO_CHAR(order_date, 'Day') as day_name,
    TO_CHAR(order_date, 'Month') as month_name,
    TO_CHAR(order_date, 'HH24:MI:SS') as time_24h
FROM orders;
""", language='sql')

    st.markdown("### Common Date Filters")
    st.code("""
-- Today
SELECT * FROM orders WHERE order_date = CURRENT_DATE;

-- Last 7 days
SELECT * FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '7 days';

-- Last 30 days
SELECT * FROM orders WHERE order_date >= CURRENT_DATE - INTERVAL '30 days';

-- This month
SELECT * FROM orders
WHERE DATE_TRUNC('month', order_date) = DATE_TRUNC('month', CURRENT_DATE);

-- Last month
SELECT * FROM orders
WHERE DATE_TRUNC('month', order_date) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month');

-- This year
SELECT * FROM orders WHERE EXTRACT(YEAR FROM order_date) = EXTRACT(YEAR FROM CURRENT_DATE);

-- Specific date range
SELECT * FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';

-- Weekends only
SELECT * FROM orders WHERE EXTRACT(DOW FROM order_date) IN (0, 6);

-- Weekdays only
SELECT * FROM orders WHERE EXTRACT(DOW FROM order_date) BETWEEN 1 AND 5;
""", language='sql')


def cte():
    st.header("🔄 Common Table Expressions (CTEs)")
    st.markdown("### Basic CTEs")
    st.code("""
-- Simple CTE
WITH high_value_customers AS (
    SELECT
        user_id,
        SUM(amount) as total_spent
    FROM orders
    GROUP BY user_id
    HAVING SUM(amount) > 1000
)
SELECT
    u.username,
    u.email,
    hvc.total_spent
FROM high_value_customers hvc
JOIN users u ON hvc.user_id = u.id
ORDER BY hvc.total_spent DESC;

-- Multiple CTEs
WITH
monthly_sales AS (
    SELECT
        DATE_TRUNC('month', order_date) as month,
        SUM(amount) as revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
),
sales_with_growth AS (
    SELECT
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as prev_month_revenue,
        revenue - LAG(revenue) OVER (ORDER BY month) as growth
    FROM monthly_sales
)
SELECT
    TO_CHAR(month, 'YYYY-MM') as month,
    revenue,
    prev_month_revenue,
    growth,
    ROUND(100.0 * growth / NULLIF(prev_month_revenue, 0), 2) as growth_pct
FROM sales_with_growth
ORDER BY month;
""", language='sql')

    st.markdown("### Recursive CTEs (Hierarchies)")
    st.code("""
WITH RECURSIVE org_chart AS (
    -- Base case: top-level managers
    SELECT
        id,
        name,
        manager_id,
        0 as level,
        name::TEXT as path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: employees under managers
    SELECT
        e.id,
        e.name,
        e.manager_id,
        oc.level + 1,
        oc.path || ' > ' || e.name
    FROM employees e
    INNER JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY path;

-- Generate date series
WITH RECURSIVE date_series AS (
    SELECT '2024-01-01'::date as date
    UNION ALL
    SELECT date + INTERVAL '1 day'
    FROM date_series
    WHERE date < '2024-12-31'
)
SELECT date FROM date_series;
""", language='sql')


def case_statements():
    st.header("🔀 CASE Statements")
    st.markdown("### Basic CASE")

    st.code("""
-- Simple categorization
SELECT
    product_name,
    price,
    CASE
        WHEN price < 20 THEN 'Budget'
        WHEN price BETWEEN 20 AND 100 THEN 'Mid-range'
        ELSE 'Premium'
    END as price_category
FROM products;

-- Multiple conditions
SELECT
    user_id,
    total_orders,
    total_spent,
    CASE
        WHEN total_orders >= 10 AND total_spent > 1000 THEN 'VIP'
        WHEN total_orders >= 5 THEN 'Regular'
        WHEN total_orders >= 1 THEN 'New'
        ELSE 'Inactive'
    END as customer_segment
FROM user_stats;
""", language='sql')

    st.markdown("### CASE in Aggregations")
    st.code("""
- Pivot-style reporting
SELECT
    DATE_TRUNC('month', order_date) as month,
    COUNT(*) as total_orders,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
    SUM(CASE WHEN status = 'completed' THEN amount ELSE 0 END) as revenue
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

-- Conditional averages
SELECT
    category,
    AVG(CASE WHEN status = 'new' THEN price END) as avg_new_price,
    AVG(CASE WHEN status = 'used' THEN price END) as avg_used_price
FROM products
GROUP BY category;
""", language='sql')


def subqueries():
    st.header("🔎 Subqueries")
    st.markdown("### Scalar Subqueries")
    st.code("""
-- Compare to average
SELECT
    product_name,
    price,
    (SELECT AVG(price) FROM products) as avg_price,
    price - (SELECT AVG(price) FROM products) as diff_from_avg
FROM products;
""", language='sql')

    st.markdown("### IN / NOT IN Subqueries")
    st.code("""
-- Find users who made purchases
SELECT * FROM users
WHERE id IN (SELECT DISTINCT user_id FROM orders);

-- Find users who never purchased
SELECT * FROM users
WHERE id NOT IN (SELECT DISTINCT user_id FROM orders WHERE user_id IS NOT NULL);
""", language='sql')

    st.markdown("### EXISTS Subqueries")
    st.code("""
-- More efficient than IN for large datasets
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id AND o.status = 'completed'
);

-- Users with no completed orders
SELECT * FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id AND o.status = 'completed'
);
""", language='sql')


def userful_string_functions():
    st.header("🔤 Useful String Functions")
    st.code("""
-- Concatenation
SELECT first_name || ' ' || last_name as full_name FROM users;
SELECT CONCAT(first_name, ' ', last_name) as full_name FROM users;

-- Case conversion
SELECT UPPER(email), LOWER(email), INITCAP(name) FROM users;

-- Trimming
SELECT TRIM(name), LTRIM(name), RTRIM(name) FROM users;

-- Substring
SELECT SUBSTRING(email FROM 1 FOR POSITION('@' IN email) - 1) as username FROM users;
SELECT SPLIT_PART(email, '@', 1) as username FROM users;

-- String length
SELECT LENGTH(description), CHAR_LENGTH(name) FROM products;

-- Replace
SELECT REPLACE(description, 'old', 'new') FROM products;

-- Pattern matching
SELECT * FROM users WHERE email LIKE '%@gmail.com';
SELECT * FROM users WHERE email ILIKE '%@GMAIL.COM';""", language='sql')


def data_export_import():
    st.header("💾 Data Export & Import")
    st.markdown("### Export Data")
    st.code("""
- Export to CSV from psql
\copy (SELECT * FROM users WHERE created_at >= '2024-01-01') TO '/path/to/users.csv' CSV HEADER;

-- Export query results
\copy (
    SELECT
        u.username,
        COUNT(o.id) as order_count,
        SUM(o.amount) as total_spent
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    GROUP BY u.username
) TO '/path/to/report.csv' CSV HEADER;
""", language='sql')

    st.markdown("### Import Data")
    st.code("""
-- Import CSV to existing table
\copy table_name FROM '/path/to/data.csv' CSV HEADER;

-- Import to temp table first (safer)
CREATE TEMP TABLE temp_import (
    column1 TEXT,
    column2 TEXT,
    column3 INTEGER
);

\copy temp_import FROM '/path/to/data.csv' CSV HEADER;

-- Then validate and insert
INSERT INTO main_table
SELECT * FROM temp_import
WHERE column3 > 0;
""", language='sql')


def quick_tips():
    st.header("💡 Quick Tips & Best Practices")
    st.markdown("### Performance")
    st.code("""
-- Use EXPLAIN to understand query performance
EXPLAIN ANALYZE
SELECT * FROM large_table WHERE indexed_column = 'value';

-- Limit results when exploring
SELECT * FROM large_table LIMIT 100;

-- Use DISTINCT carefully (expensive operation)
SELECT DISTINCT user_id FROM orders;  -- Fine
SELECT DISTINCT * FROM large_table;
""", language='sql')

    st.markdown("### Writing Clean Queries")
    st.code("""
-- Use meaningful aliases
SELECT
    u.username,
    o.order_date,
    p.product_name
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN products p ON o.product_id = p.id;

-- Format for readability
SELECT
    category,
    COUNT(*) as product_count,
    AVG(price) as avg_price
FROM products
WHERE price > 0
GROUP BY category
HAVING COUNT(*) >= 10
ORDER BY product_count DESC;

-- Comment your complex queries
-- Calculate customer lifetime value by cohort
WITH user_cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', first_order_date) as cohort_month
    FROM ...
)
SELECT ...
""", language='sql')

    st.markdown("### Common Mistakes to Avoid")
    st.code("""
- ❌ Forgetting NULLIF when dividing
SELECT revenue / user_count FROM table;  -- Division by zero error!

-- ✅ Safe division
SELECT revenue / NULLIF(user_count, 0) FROM table;

-- ❌ Not handling NULLs in calculations
SELECT SUM(amount) FROM orders;  -- NULLs are ignored (good)
SELECT amount * 1.1 FROM orders;  -- NULL * 1.1 = NULL (be aware)

-- ✅ Handle NULLs explicitly
SELECT COALESCE(amount, 0) * 1.1 FROM orders;

-- ❌ Using SELECT * in production queries
SELECT * FROM huge_table;  -- Slow and wasteful

-- ✅ Select only what you need
SELECT id, name, created_at FROM huge_table;
""", language='sql')

    st.markdown("### Cohort Analysis")
    st.code("""
WITH user_cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('month', MIN(order_date)) as cohort_month
    FROM orders
    GROUP BY user_id
),
cohort_activity AS (
    SELECT
        uc.cohort_month,
        DATE_TRUNC('month', o.order_date) as activity_month,
        COUNT(DISTINCT o.user_id) as active_users
    FROM user_cohorts uc
    JOIN orders o ON uc.user_id = o.user_id
    GROUP BY uc.cohort_month, DATE_TRUNC('month', o.order_date)
)
SELECT
    cohort_month,
    activity_month,
    active_users,
    EXTRACT(MONTH FROM AGE(activity_month, cohort_month)) as months_since_cohort
FROM cohort_activity
ORDER BY cohort_month, activity_month;
""", language='sql')

    st.markdown("### RFM Analysis (Recency, Frequency, Monetary)")
    st.code("""
WITH rfm AS (
    SELECT
        user_id,
        CURRENT_DATE - MAX(order_date) as recency_days,
        COUNT(*) as frequency,
        SUM(amount) as monetary
    FROM orders
    WHERE status = 'completed'
    GROUP BY user_id
),
rfm_scores AS (
    SELECT
        user_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC) as r_score,
        NTILE(5) OVER (ORDER BY frequency) as f_score,
        NTILE(5) OVER (ORDER BY monetary) as m_score
    FROM rfm
)
SELECT
    user_id,
    r_score,
    f_score,
    m_score,
    r_score + f_score + m_score as rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'New'
        WHEN r_score <= 2 THEN 'At Risk'
        ELSE 'Regular'
    END as segment
FROM rfm_scores;
""", language='sql')

    st.markdown("### Funnel Analysis")
    st.code("""
WITH funnel_steps AS (
    SELECT
        COUNT(DISTINCT CASE WHEN event = 'page_view' THEN user_id END) as step_1_viewed,
        COUNT(DISTINCT CASE WHEN event = 'add_to_cart' THEN user_id END) as step_2_added,
        COUNT(DISTINCT CASE WHEN event = 'checkout' THEN user_id END) as step_3_checkout,
        COUNT(DISTINCT CASE WHEN event = 'purchase' THEN user_id END) as step_4_purchased
    FROM events
    WHERE event_date >= CURRENT_DATE - INTERVAL '30 days'
)
SELECT
    step_1_viewed,
    step_2_added,
    ROUND(100.0 * step_2_added / NULLIF(step_1_viewed, 0), 2) as view_to_cart_rate,
    step_3_checkout,
    ROUND(100.0 * step_3_checkout / NULLIF(step_2_added, 0), 2) as cart_to_checkout_rate,
    step_4_purchased,
    ROUND(100.0 * step_4_purchased / NULLIF(step_3_checkout, 0), 2) as checkout_to_purchase_rate,
    ROUND(100.0 * step_4_purchased / NULLIF(step_1_viewed, 0), 2) as overall_conversion_rate
FROM funnel_steps;
""", language='sql')
