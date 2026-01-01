import streamlit as st


def connection_authentication():
    st.header("🔌 Connection & Authentication")
    st.code("""
-- Connect to MySQL
mysql -u username -p
mysql -u username -p database_name
mysql -h hostname -u username -p database_name

-- Connect with specific port
mysql -h hostname -P 3306 -u username -p

-- Exit MySQL
EXIT; or QUIT; or \q
""", language='bash')


def database_operations():
    st.header("🗄️ Database Operations")
    st.code("""
-- Show all databases
SHOW DATABASES;

-- Create database
CREATE DATABASE database_name;
CREATE DATABASE IF NOT EXISTS database_name;

-- Use/Select database
USE database_name;

-- Drop database
DROP DATABASE database_name;
DROP DATABASE IF EXISTS database_name;

-- Show current database
SELECT DATABASE();
""", language='sql')


def table_operations():
    st.header("📊 Table Operations")
    st.code("""
-- Show all tables
SHOW TABLES;

-- Create table
CREATE TABLE table_name (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    age INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Show table structure
DESCRIBE table_name;
SHOW COLUMNS FROM table_name;

-- Show create table statement
SHOW CREATE TABLE table_name;

-- Drop table
DROP TABLE table_name;
DROP TABLE IF EXISTS table_name;

-- Rename table
RENAME TABLE old_name TO new_name;
ALTER TABLE old_name RENAME TO new_name;

-- Truncate table (delete all data)
TRUNCATE TABLE table_name;
""", language='sql')


def column_operations():
    st.header("🔧 Column Operations")
    st.code("""
-- Add column
ALTER TABLE table_name ADD COLUMN column_name datatype;
ALTER TABLE table_name ADD column_name VARCHAR(255) AFTER existing_column;

-- Modify column
ALTER TABLE table_name MODIFY COLUMN column_name new_datatype;
ALTER TABLE table_name CHANGE old_column_name new_column_name datatype;

-- Drop column
ALTER TABLE table_name DROP COLUMN column_name;

-- Add primary key
ALTER TABLE table_name ADD PRIMARY KEY (column_name);

-- Drop primary key
ALTER TABLE table_name DROP PRIMARY KEY;
""", language='sql')


def data_types():
    st.header("📝 Data Types")
    st.code("""
-- Numeric
TINYINT, SMALLINT, MEDIUMINT, INT, BIGINT
DECIMAL(precision, scale), NUMERIC(precision, scale)
FLOAT, DOUBLE

-- String
CHAR(length)        -- Fixed length
VARCHAR(length)     -- Variable length
TEXT, MEDIUMTEXT, LONGTEXT
BINARY, VARBINARY
BLOB, MEDIUMBLOB, LONGBLOB

-- Date/Time
DATE                -- YYYY-MM-DD
TIME                -- HH:MM:SS
DATETIME            -- YYYY-MM-DD HH:MM:SS
TIMESTAMP           -- YYYY-MM-DD HH:MM:SS
YEAR                -- YYYY

-- JSON (MySQL 5.7+)
JSON
""", language='sql')


def crud_operations():
    st.header("✏️ CRUD Operations")
    st.markdown("### INSERT")
    st.code("""- Insert single record
INSERT INTO table_name (column1, column2) VALUES (value1, value2);

-- Insert multiple records
INSERT INTO table_name (column1, column2) VALUES
    (value1, value2),
    (value3, value4);

-- Insert with SELECT
INSERT INTO table_name (column1, column2)
SELECT column1, column2 FROM other_table WHERE condition;

-- Insert or update (upsert)
INSERT INTO table_name (id, name) VALUES (1, 'John')
ON DUPLICATE KEY UPDATE name = VALUES(name);""", language='sql')

    st.markdown("### SELECT")
    st.code("""-- Basic select
SELECT * FROM table_name;
SELECT column1, column2 FROM table_name;

-- With conditions
SELECT * FROM table_name WHERE condition;
SELECT * FROM table_name WHERE column1 = 'value' AND column2 > 10;

-- Sorting
SELECT * FROM table_name ORDER BY column1 ASC, column2 DESC;

-- Limiting results
SELECT * FROM table_name LIMIT 10;
SELECT * FROM table_name LIMIT 10 OFFSET 20;

-- Grouping
SELECT column1, COUNT(*) FROM table_name GROUP BY column1;
SELECT column1, COUNT(*) FROM table_name GROUP BY column1 HAVING COUNT(*) > 5;

-- Distinct values
SELECT DISTINCT column1 FROM table_name;""", language='sql')

    st.markdown("### UPDATE")
    st.code("""-- Update records
UPDATE table_name SET column1 = value1 WHERE condition;
UPDATE table_name SET column1 = value1, column2 = value2 WHERE condition;

-- Update with JOIN
UPDATE table1 t1
JOIN table2 t2 ON t1.id = t2.id
SET t1.column1 = t2.column1
WHERE condition;""", language='sql')

    st.markdown("### DELETE")
    st.code("""DELETE FROM table_name WHERE condition;

-- Delete all records (keep structure)
DELETE FROM table_name;

-- Delete with JOIN
DELETE t1 FROM table1 t1
JOIN table2 t2 ON t1.id = t2.id
WHERE condition;""", language='sql')


def where_clauses():
    st.header("🔎 WHERE Clauses")
    st.code("""-- Comparison operators
WHERE column = value
WHERE column != value (or <> value)
WHERE column > value
WHERE column >= value
WHERE column < value
WHERE column <= value

-- Pattern matching
WHERE column LIKE 'pattern%'      -- Starts with
WHERE column LIKE '%pattern'      -- Ends with
WHERE column LIKE '%pattern%'     -- Contains
WHERE column REGEXP 'pattern'     -- Regular expression

-- Range and lists
WHERE column BETWEEN value1 AND value2
WHERE column IN (value1, value2, value3)
WHERE column NOT IN (value1, value2)

-- NULL checks
WHERE column IS NULL
WHERE column IS NOT NULL

-- Logical operators
WHERE condition1 AND condition2
WHERE condition1 OR condition2
WHERE NOT condition

-- Date filters
WHERE date_column >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
WHERE YEAR(date_column) = 2024
WHERE date_column BETWEEN '2024-01-01' AND '2024-12-31'""", language='sql')


def joins():
    st.header("🔗 Joins")
    st.code("""-- INNER JOIN
SELECT * FROM table1 t1
INNER JOIN table2 t2 ON t1.id = t2.foreign_id;

-- LEFT JOIN
SELECT * FROM table1 t1
LEFT JOIN table2 t2 ON t1.id = t2.foreign_id;

-- RIGHT JOIN
SELECT * FROM table1 t1
RIGHT JOIN table2 t2 ON t1.id = t2.foreign_id;

-- FULL OUTER JOIN (not directly supported, use UNION)
SELECT * FROM table1 t1 LEFT JOIN table2 t2 ON t1.id = t2.foreign_id
UNION
SELECT * FROM table1 t1 RIGHT JOIN table2 t2 ON t1.id = t2.foreign_id;

-- CROSS JOIN
SELECT * FROM table1 CROSS JOIN table2;

-- Self JOIN
SELECT a.name, b.name as manager
FROM employees a
LEFT JOIN employees b ON a.manager_id = b.id;""", language='sql')


def window_functions():
    st.header("🪟 Window Functions")
    st.code("""
    -- ROW_NUMBER - Assign unique row numbers
SELECT
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as row_num
FROM employees;

-- RANK and DENSE_RANK - Handle ties differently
SELECT
    name,
    score,
    RANK() OVER (ORDER BY score DESC) as rank,
    DENSE_RANK() OVER (ORDER BY score DESC) as dense_rank
FROM students;

-- LAG and LEAD - Access previous/next rows
SELECT
    date,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY date) as prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY date) as next_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY date) as day_over_day_change
FROM daily_sales;

-- Running totals and moving averages
SELECT
    date,
    amount,
    SUM(amount) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as running_total,
    AVG(amount) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7day
FROM transactions;

-- NTILE - Divide into buckets (quartiles, deciles, etc.)
SELECT
    customer_id,
    total_spend,
    NTILE(4) OVER (ORDER BY total_spend DESC) as quartile,
    NTILE(10) OVER (ORDER BY total_spend DESC) as decile
FROM customer_summary;

-- FIRST_VALUE and LAST_VALUE
SELECT
    product_id,
    date,
    price,
    FIRST_VALUE(price) OVER (PARTITION BY product_id ORDER BY date) as initial_price,
    LAST_VALUE(price) OVER (PARTITION BY product_id ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as latest_price
FROM product_prices;""", language='sql')


def analytics_use_cast_example():
    st.header("🎯 Analytics Use Case Examples")
    st.code("""
-- Customer RFM Analysis (Recency, Frequency, Monetary)
WITH customer_rfm AS (
    SELECT
        customer_id,
        DATEDIFF(CURDATE(), MAX(order_date)) as recency,
        COUNT(order_id) as frequency,
        SUM(amount) as monetary
    FROM orders
    GROUP BY customer_id
)
SELECT
    customer_id,
    recency,
    frequency,
    monetary,
    NTILE(5) OVER (ORDER BY recency DESC) as r_score,
    NTILE(5) OVER (ORDER BY frequency) as f_score,
    NTILE(5) OVER (ORDER BY monetary) as m_score
FROM customer_rfm;

-- Funnel Analysis
SELECT
    step,
    COUNT(DISTINCT user_id) as users,
    ROUND(100.0 * COUNT(DISTINCT user_id) /
        FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (ORDER BY step), 2) as conversion_rate
FROM (
    SELECT user_id, 1 as step FROM page_views WHERE page = 'landing'
    UNION ALL
    SELECT user_id, 2 FROM page_views WHERE page = 'product'
    UNION ALL
    SELECT user_id, 3 FROM cart_additions
    UNION ALL
    SELECT user_id, 4 FROM purchases
) funnel
GROUP BY step
ORDER BY step;

-- Moving Average for Trend Analysis
SELECT
    date,
    revenue,
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as ma_7day,
    AVG(revenue) OVER (
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as ma_30day
FROM daily_revenue
ORDER BY date;
""", language='sql')


def advanced_aggregations():
    st.header("📈 Advanced Aggregations")
    st.code("""
- ROLLUP - Generate subtotals and grand totals
SELECT
    COALESCE(region, 'All Regions') as region,
    COALESCE(category, 'All Categories') as category,
    SUM(sales) as total_sales
FROM sales_data
GROUP BY region, category WITH ROLLUP;

-- Multiple aggregations with CASE
SELECT
    product_category,
    COUNT(*) as total_orders,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_orders,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled_orders,
    AVG(CASE WHEN status = 'completed' THEN order_value END) as avg_completed_value,
    SUM(CASE WHEN order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) THEN 1 ELSE 0 END) as orders_last_30d
FROM orders
GROUP BY product_category;

-- Conditional aggregations for pivot-like results
SELECT
    DATE_FORMAT(date, '%Y-%m') as month,
    SUM(CASE WHEN product = 'A' THEN sales ELSE 0 END) as product_a_sales,
    SUM(CASE WHEN product = 'B' THEN sales ELSE 0 END) as product_b_sales,
    SUM(CASE WHEN product = 'C' THEN sales ELSE 0 END) as product_c_sales
FROM sales
GROUP BY DATE_FORMAT(date, '%Y-%m');

-- Percentile calculations
SELECT
    department,
    AVG(salary) as avg_salary,
    MIN(salary) as min_salary,
    MAX(salary) as max_salary,
    (SELECT salary FROM employees e2
     WHERE e2.department = e1.department
     ORDER BY salary
     LIMIT 1 OFFSET FLOOR(COUNT(*) * 0.5)) as median_salary
FROM employees e1
GROUP BY department;""", language='sql')


def cte():
    st.header("🔄 CTEs - Clean Complex Queries")
    st.code("""
-- Basic CTE
WITH monthly_sales AS (
    SELECT
        DATE_FORMAT(order_date, '%Y-%m') as month,
        SUM(amount) as total_sales
    FROM orders
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
)
SELECT * FROM monthly_sales WHERE total_sales > 10000;

-- Multiple CTEs
WITH
customer_totals AS (
    SELECT customer_id, SUM(amount) as total_spent
    FROM orders
    GROUP BY customer_id
),
customer_segments AS (
    SELECT
        customer_id,
        total_spent,
        CASE
            WHEN total_spent > 1000 THEN 'High Value'
            WHEN total_spent > 500 THEN 'Medium Value'
            ELSE 'Low Value'
        END as segment
    FROM customer_totals
)
SELECT segment, COUNT(*) as customer_count, AVG(total_spent) as avg_spent
FROM customer_segments
GROUP BY segment;

-- Recursive CTE (for hierarchical data)
WITH RECURSIVE employee_hierarchy AS (
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    SELECT e.id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM employee_hierarchy ORDER BY level, name;
""", language='sql')


def data_quality_checks():
    st.header("✅ Data Quality Checks")
    st.code("""-- Check for duplicates
SELECT column1, column2, COUNT(*) as duplicate_count
FROM table_name
GROUP BY column1, column2
HAVING COUNT(*) > 1;

-- Check for NULL values
SELECT
    COUNT(*) as total_rows,
    SUM(CASE WHEN column1 IS NULL THEN 1 ELSE 0 END) as column1_nulls,
    SUM(CASE WHEN column2 IS NULL THEN 1 ELSE 0 END) as column2_nulls,
    ROUND(100.0 * SUM(CASE WHEN column1 IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as column1_null_pct
FROM table_name;

-- Check for data consistency
SELECT
    'Total records' as check_type,
    COUNT(*) as count
FROM orders
UNION ALL
SELECT
    'Records with invalid dates',
    COUNT(*)
FROM orders
WHERE order_date > CURDATE() OR order_date < '2000-01-01'
UNION ALL
SELECT
    'Records with negative amounts',
    COUNT(*)
FROM orders
WHERE amount < 0;

-- Find outliers using IQR method
WITH quartiles AS (
    SELECT
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY value) as q1,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY value) as q3
    FROM measurements
)
SELECT m.*
FROM measurements m
CROSS JOIN quartiles q
WHERE m.value < (q.q1 - 1.5 * (q.q3 - q.q1))
   OR m.value > (q.q3 + 1.5 * (q.q3 - q.q1));""", language='sql')


def data_profling():
    st.header("🔍 Data Profiling")
    st.code("""-- Get table statistics
SELECT
    table_name,
    table_rows,
    ROUND(data_length / 1024 / 1024, 2) as data_size_mb,
    ROUND(index_length / 1024 / 1024, 2) as index_size_mb,
    ROUND((data_length + index_length) / 1024 / 1024, 2) as total_size_mb
FROM information_schema.tables
WHERE table_schema = 'your_database'
ORDER BY (data_length + index_length) DESC;

-- Column statistics
SELECT
    column_name,
    COUNT(*) as total_count,
    COUNT(DISTINCT column_name) as unique_count,
    SUM(CASE WHEN column_name IS NULL THEN 1 ELSE 0 END) as null_count,
    MIN(column_name) as min_value,
    MAX(column_name) as max_value
FROM table_name;

-- Value frequency distribution
SELECT
    column_name,
    COUNT(*) as frequency,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM table_name), 2) as percentage
FROM table_name
GROUP BY column_name
ORDER BY frequency DESC
LIMIT 20;

-- Cardinality check for all columns
SELECT
    'column1' as column_name,
    COUNT(DISTINCT column1) as distinct_count,
    COUNT(*) as total_count,
    ROUND(100.0 * COUNT(DISTINCT column1) / COUNT(*), 2) as uniqueness_pct
FROM table_name
UNION ALL
SELECT 'column2', COUNT(DISTINCT column2), COUNT(*),
    ROUND(100.0 * COUNT(DISTINCT column2) / COUNT(*), 2)
FROM table_name;""", language='sql')


def date_time_analytics():
    st.header("📅 Date/Time Analytics")
    st.code("""
-- Date dimension generation
WITH RECURSIVE date_series AS (
    SELECT DATE('2024-01-01') as date
    UNION ALL
    SELECT DATE_ADD(date, INTERVAL 1 DAY)
    FROM date_series
    WHERE date < '2024-12-31'
)
SELECT
    date,
    DAYNAME(date) as day_name,
    WEEK(date) as week_number,
    MONTH(date) as month,
    QUARTER(date) as quarter,
    YEAR(date) as year,
    CASE WHEN DAYOFWEEK(date) IN (1, 7) THEN 1 ELSE 0 END as is_weekend
FROM date_series;

-- Time-based cohort analysis
SELECT
    DATE_FORMAT(first_purchase_date, '%Y-%m') as cohort_month,
    TIMESTAMPDIFF(MONTH, first_purchase_date, purchase_date) as months_since_first,
    COUNT(DISTINCT customer_id) as active_customers,
    SUM(amount) as total_revenue
FROM (
    SELECT
        customer_id,
        purchase_date,
        MIN(purchase_date) OVER (PARTITION BY customer_id) as first_purchase_date,
        amount
    FROM purchases
) cohort_data
GROUP BY cohort_month, months_since_first
ORDER BY cohort_month, months_since_first;

-- Year-over-year comparison
SELECT
    DATE_FORMAT(date, '%Y-%m') as month,
    SUM(revenue) as current_revenue,
    LAG(SUM(revenue), 12) OVER (ORDER BY DATE_FORMAT(date, '%Y-%m')) as prev_year_revenue,
    ROUND(100.0 * (SUM(revenue) - LAG(SUM(revenue), 12) OVER (ORDER BY DATE_FORMAT(date, '%Y-%m'))) /
        LAG(SUM(revenue), 12) OVER (ORDER BY DATE_FORMAT(date, '%Y-%m')), 2) as yoy_growth_pct
FROM sales
GROUP BY DATE_FORMAT(date, '%Y-%m');
""", language='sql')


def performance_optimization():
    st.header("⚡ Performance Optimization for Analytics")
    st.code("""
-- EXPLAIN - Analyze query execution plan
EXPLAIN SELECT * FROM large_table WHERE indexed_column = 'value';
EXPLAIN FORMAT=JSON SELECT * FROM table_name WHERE condition;

-- Show query profile
SET profiling = 1;
SELECT * FROM table_name WHERE condition;
SHOW PROFILES;
SHOW PROFILE FOR QUERY 1;

-- Index usage analysis
SELECT
    table_name,
    index_name,
    seq_in_index,
    column_name,
    cardinality
FROM information_schema.statistics
WHERE table_schema = 'your_database'
ORDER BY table_name, index_name, seq_in_index;

-- Identify missing indexes (slow queries without index)
SELECT * FROM information_schema.tables
WHERE table_schema = 'your_database'
AND table_rows > 10000
AND table_name NOT IN (
    SELECT DISTINCT table_name
    FROM information_schema.statistics
    WHERE table_schema = 'your_database'
);

-- Composite index recommendation
CREATE INDEX idx_composite ON table_name (frequently_filtered_col, join_col, sorted_col);

-- Covering index (include all columns in query)
CREATE INDEX idx_covering ON orders (customer_id, order_date, status, amount);
""", language='sql')


def sampling_techniques():
    st.header("🎲 Sampling Techniques")
    st.code("""
-- Random sampling (percentage)
SELECT * FROM large_table
WHERE RAND() < 0.01  -- 1% sample
LIMIT 10000;

-- Systematic sampling
SELECT * FROM large_table
WHERE id % 100 = 0;  -- Every 100th row

-- Stratified sampling
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY RAND()) as rn
    FROM products
) sampled
WHERE rn <= 10;  -- 10 samples per category
""", language='sql')


def string_functions_data_clearning():
    st.header("🧹 String Functions for Data Cleaning")
    st.code("""
-- Clean and standardize text
SELECT
    TRIM(BOTH ' ' FROM name) as cleaned_name,
    UPPER(TRIM(email)) as standardized_email,
    REPLACE(phone, '-', '') as phone_digits_only,
    REGEXP_REPLACE(address, '[^a-zA-Z0-9 ]', '') as clean_address
FROM customers;

-- Extract patterns
SELECT
    email,
    SUBSTRING_INDEX(email, '@', 1) as username,
    SUBSTRING_INDEX(email, '@', -1) as domain,
    CASE
        WHEN email REGEXP '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$'
        THEN 'Valid'
        ELSE 'Invalid'
    END as email_validity
FROM users;

-- Fuzzy matching (Levenshtein distance approximation)
SELECT
    a.name as name1,
    b.name as name2,
    LENGTH(a.name) + LENGTH(b.name) - 2 * LENGTH(
        REGEXP_REPLACE(
            CONCAT(a.name, b.name),
            CONCAT('[^', a.name, ']'),
            ''
        )
    ) as similarity_score
FROM names a
CROSS JOIN names b
WHERE a.id < b.id
HAVING similarity_score < 5;
""", language='sql')


def materialize_views():
    st.header("💾 Materialized Views (using tables)")
    st.code("""-- Create summary table for fast analytics
CREATE TABLE sales_summary AS
SELECT
    DATE(order_date) as date,
    product_id,
    SUM(quantity) as total_quantity,
    SUM(amount) as total_revenue,
    COUNT(DISTINCT customer_id) as unique_customers
FROM orders
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
GROUP BY DATE(order_date), product_id;

-- Add index for performance
CREATE INDEX idx_date_product ON sales_summary(date, product_id);

-- Refresh materialized view
TRUNCATE TABLE sales_summary;
INSERT INTO sales_summary
SELECT
    DATE(order_date) as date,
    product_id,
    SUM(quantity) as total_quantity,
    SUM(amount) as total_revenue,
    COUNT(DISTINCT customer_id) as unique_customers
FROM orders
WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
GROUP BY DATE(order_date), product_id;""", language='sql')


def data_export_analysis():
    st.header("📤 Data Export for Analysis Tools")
    st.code("""-- Export to CSV
SELECT 'column1', 'column2', 'column3'  -- Header row
UNION ALL
SELECT column1, column2, column3
FROM table_name
INTO OUTFILE '/tmp/export.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

-- Export with custom delimiter for tools like Tableau
SELECT * FROM analytics_table
INTO OUTFILE '/tmp/tableau_export.txt'
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n';""", language='sql')


def indexes():
    st.header("🔖 Indexes (Analytics-Focused)")
    st.code("""-- Create index
CREATE INDEX index_name ON table_name (column1, column2);
CREATE UNIQUE INDEX index_name ON table_name (column1);

-- Composite index (column order matters!)
CREATE INDEX idx_analytics ON fact_table (date_key, product_key, customer_key);

-- Show indexes
SHOW INDEX FROM table_name;

-- Drop index
DROP INDEX index_name ON table_name;
ALTER TABLE table_name DROP INDEX index_name;

-- Analyze table (update statistics)
ANALYZE TABLE table_name;""", language='sql')


def views():
    st.header("👁️ Views")
    st.code("""-- Create view
    CREATE VIEW view_name AS
    SELECT column1, column2 FROM table_name WHERE condition;

    -- Create or replace view
    CREATE OR REPLACE VIEW customer_metrics AS
    SELECT
        customer_id,
        COUNT(*) as order_count,
        SUM(amount) as lifetime_value,
        AVG(amount) as avg_order_value,
        MAX(order_date) as last_order_date
    FROM orders
    GROUP BY customer_id;

    -- Show views
    SHOW FULL TABLES WHERE Table_type = 'VIEW';

    -- Drop view
    DROP VIEW view_name;""", language='sql')


def stored_procedures():
    st.header("⚙️ Stored Procedures & Functions")
    st.code("""
-- Create stored procedure
DELIMITER //
CREATE PROCEDURE procedure_name(IN param1 INT, OUT param2 VARCHAR(255))
BEGIN
    -- procedure body
    SELECT column1 INTO param2 FROM table_name WHERE id = param1;
END //
DELIMITER ;

-- Call procedure
CALL procedure_name(1, @result);
SELECT @result;

-- Create function
DELIMITER //
CREATE FUNCTION function_name(param1 INT) RETURNS VARCHAR(255)
READS SQL DATA
BEGIN
    DECLARE result VARCHAR(255);
    SELECT column1 INTO result FROM table_name WHERE id = param1;
    RETURN result;
END //
DELIMITER ;

-- Drop procedure/function
DROP PROCEDURE procedure_name;
DROP FUNCTION function_name;
""", language='sql')


def transactions():
    st.header("🔄 Transactions")
    st.code("""-- Start transaction
START TRANSACTION;
BEGIN;

-- Commit transaction
COMMIT;

-- Rollback transaction
ROLLBACK;

-- Savepoints
SAVEPOINT savepoint_name;
ROLLBACK TO savepoint_name;""", language='sql')


def user_management():
    st.header("👤 User Management")
    st.code("""-- Create user
CREATE USER 'username'@'localhost' IDENTIFIED BY 'password';
CREATE USER 'username'@'%' IDENTIFIED BY 'password';

-- Grant privileges
GRANT ALL PRIVILEGES ON database_name.* TO 'username'@'localhost';
GRANT SELECT, INSERT ON table_name TO 'username'@'localhost';

-- Grant read-only access (common for analysts)
GRANT SELECT ON database_name.* TO 'analyst'@'%';

-- Show grants
SHOW GRANTS FOR 'username'@'localhost';

-- Revoke privileges
REVOKE ALL PRIVILEGES ON database_name.* FROM 'username'@'localhost';

-- Drop user
DROP USER 'username'@'localhost';

-- Change password
ALTER USER 'username'@'localhost' IDENTIFIED BY 'new_password';""", language='sql')


def system_information():
    st.header("ℹ️ System Information")
    st.code("""
-- Show MySQL version
SELECT VERSION();

-- Show current user
SELECT USER();

-- Show processlist
SHOW PROCESSLIST;

-- Show variables
SHOW VARIABLES LIKE 'variable_name%';

-- Show status
SHOW STATUS LIKE 'status_name%';

-- Show engines
SHOW ENGINES;

-- Show database size
SELECT
    table_schema as database_name,
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) as size_mb
FROM information_schema.tables
GROUP BY table_schema;
""", language='sql')


def json_functions():
    st.header("📋 JSON Functions (MySQL 5.7+)")
    st.code("""
-- JSON creation
JSON_OBJECT('key', 'value')
JSON_ARRAY(value1, value2)

-- JSON extraction
JSON_EXTRACT(json_column, '$.key')
json_column->'$.key'              -- Shorthand
json_column->>'$.key'             -- Unquoted

-- JSON modification
JSON_SET(json_column, '$.key', 'new_value')
JSON_INSERT(json_column, '$.new_key', 'value')
JSON_REPLACE(json_column, '$.key', 'new_value')
JSON_REMOVE(json_column, '$.key')

-- JSON aggregation
SELECT
    category,
    JSON_ARRAYAGG(product_name) as products,
    JSON_OBJECTAGG(product_id, price) as product_prices
FROM products
GROUP BY category;
""", language='sql')
