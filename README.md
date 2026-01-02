# 📊🧱⚙️ The Modern Data Stack Cheat Sheet

A practical, engineering-first reference for **Data Analysts, Analytics Engineers, and Data Engineers**.

This is not a tutorial. It's a **working knowledge base** of patterns, syntax, and system-level thinking across the modern data stack.

## 🎯 Purpose

Built as a living reference for real-world data systems — not just syntax. This cheat sheet provides quick access to common patterns and approaches across the most important tools in modern data work.

## 🧠 What's Inside

### 📊 DataFrame & Processing

- **🐼 Pandas** – Local analytics and data manipulation
- **⚡ Polars** – High-performance columnar processing
- **🔥 PySpark** – Distributed data processing at scale

### 🗄️ SQL & Databases

- **🐘 PostgreSQL** – Production-grade relational database
- **🐬 MySQL** – Popular open-source RDBMS

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Streamlit

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd modern-data-stack-cheatsheet

# Install dependencies
pip install streamlit

# Run the app
streamlit run app/Home.py
```

## 🧭 How to Use

- Use the **sidebar** to navigate between tools and layers
- Treat each page as a **quick reference**, not a walkthrough
- Focus on **patterns**, not memorization
- Apply what you see directly to **real pipelines and data apps**

## 📂 Project Structure

```
.
app/
├── components/
│   ├── mysql.py
│   ├── pandas.py
│   ├── polaris.py
│   ├── postgresql.py
│   └── pyspark.py
├── pages/
│   ├── 1_Data_Analysis.py
│   ├── 2_Pandas.py
│   ├── 3_PostgreSQL.py
│   ├── 4_PySpark.py
│   ├── 5_Polaris.py
│   ├── 6_MySQL.py
│   └── 7_Kafka.py
└── Home.py
└── README.md           # This file
```

## 🚀 Roadmap

Planned additions:

- 🧱 **Data Engineering Design Patterns** – Common architectural patterns and best practices
- 📐 **Analytics Engineering Modeling Patterns** – dbt, dimensional modeling, and metrics layers
- 🧪 **Performance & Optimization Playbooks** – Query optimization, indexing strategies, and scaling patterns

## 🤝 Contributing

This is a living reference. Contributions, corrections, and expansions are welcome.

## 📄 License

MIT

## 👤 Author

[Ikigami](https://github.com/ikigamisama)

---

**Built for practitioners who ship data products.**
