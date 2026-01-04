import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import squarify

from matplotlib.patches import Polygon
from datetime import datetime, timedelta
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d


def main_page(categories, categories2):
    st.markdown('#### Transform Your Data Into Stunning Visual Stories')
    # Stats Section
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">100+</div>
                <div class="stat-label">Chart Types</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">27</div>
                <div class="stat-label">Categories</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">∞</div>
                <div class="stat-label">Possibilities</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">100%</div>
                <div class="stat-label">Python Powered</div>
            </div>
        """, unsafe_allow_html=True)

    # Features Section
    st.markdown("---")
    st.markdown("## 🎯 Explore Visualization Categories")
    st.markdown("")

    cols = st.columns(4)
    for idx, (icon, name, desc) in enumerate(categories):
        with cols[idx % 4]:
            st.markdown(f"""
                <div class="category-card">
                    <div class="category-icon">{icon}</div>
                    <div class="category-name">{name}</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.9;">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    # More categories
    cols = st.columns(5)
    for idx, (icon, name, desc) in enumerate(categories2):
        with cols[idx % 5]:
            st.markdown(f"""
                <div class="category-card">
                    <div class="category-icon">{icon}</div>
                    <div class="category-name">{name}</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.9;">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style="text-align: center;">
                <h2 style="margin-bottom: 1.5rem;">Ready to Start Visualizing?</h2>
                <p style="font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem;">
                    Choose from 100+ chart types and transform your data into compelling visual stories
                </p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Explore All Visualizations", width='stretch', type="primary"):
            st.balloons()
            st.success(
                "🎉 Let's create something amazing! Navigate to the visualization categories to get started.")
