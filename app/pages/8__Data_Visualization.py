import streamlit as st
from components import sidebar

from components.data_visualization import main_page
from components.DataVisualization import (
    Trend,
    Density,
    Relationship,
    Composition,
    Geospatial,
    Ranking,
    Flow,
    PartToWhole,
    TimeSeries,
    Correlation,
    Network,
    Multivariate
)

# Page configuration
st.set_page_config(
    page_title="Data Visualization Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        padding: 2rem 0;
        animation: fadeInDown 1s ease-in;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #6c757d;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-in;
    }
    
    /* Stats card styling */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        margin: 1rem 0;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Feature card styling */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        transform: translateX(5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #6c757d;
        line-height: 1.6;
    }
    
    /* Category card styling */
    .category-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .category-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    
    .category-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .category-name {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Button styling */
    .cta-button {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        text-align: center;
        display: inline-block;
        text-decoration: none;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Animation */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Remove padding from main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown('# 📊 Data Visualization Hub')

categories = [
    ("📈", "Trend", "Track changes over time"),
    ("🎯", "Density", "Show concentration patterns"),
    ("🔗", "Relationship", "Discover correlations"),
    ("🥧", "Composition", "Break down the whole"),
    ("🗺️", "Geospatial", "Map your data"),
    ("🏆", "Ranking", "Compare and order"),
    ("🌊", "Flow", "Track movements"),
    ("📊", "Part-to-Whole", "Show proportions"),
    ("⏰", "Time Series", "Analyze temporal data"),
    ("🔗", "Correlation", "Measure relationships"),
    ("🕸️", "Network", "Visualize connections"),
    ("📊", "Multivariate", "Handle complexity"),
]

categories2 = [
    ("🏗️", "Structural", "Show organization"),
    ("🎨", "Qualitative", "Visualize concepts"),
    ("📏", "Gauge", "Track progress"),
    ("🚨", "Anomaly", "Detect outliers"),
    ("🎯", "Behavioral", "Understand users"),
    ("📝", "Text Analysis", "Mine insights"),
    ("📋", "Text-Based", "Present data"),
    ("↔️", "Deviation", "Show variance"),
    ("💹", "Financial", "Track markets"),
    ("🔷", "Concept", "Illustrate ideas"),
    ("🔶", "Proportional", "Compare sizes"),
    ("🌳", "Hierarchical", "Show structure"),
    ("📊", "Distribution", "Understand spread"),
    ("📊", "Comparison", "Evaluate options"),
    ("📈", "Statistical", "Analyze rigorously"),
]


sidebar()
navigation_link_category = [
    i + " " + c for i, c, _ in categories + categories2]


page = st.sidebar.selectbox(
    "Pages:",
    ["🏠 Main Page"] + navigation_link_category,
)

if page == "🏠 Main Page":
    main_page(categories, categories2)
elif page == "📈 Trend":
    Trend().output()
elif page == "🎯 Density":
    Density().output()
elif page == "🔗 Relationship":
    Relationship().output()
elif page == "🥧 Composition":
    Composition().output()
elif page == "🗺️ Geospatial":
    Geospatial().output()
elif page == "🏆 Ranking":
    Ranking().output()
elif page == "🌊 Flow":
    Flow().output()
elif page == "📊 Part-to-Whole":
    PartToWhole().output()
elif page == "⏰ Time Series":
    TimeSeries().output()
elif page == "🔗 Correlation":
    Correlation().output()
elif page == "🕸️ Network":
    Network().output()
elif page == "📊 Multivariate":
    Multivariate().output()
