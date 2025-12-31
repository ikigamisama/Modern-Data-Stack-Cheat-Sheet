import streamlit as st
from components import sidebar

st.set_page_config(
    page_title="Polars Cheat Sheet",
    page_icon="⚡📊",
    layout="wide"
)

st.title("⚡📊  Polars Cheat Sheet")
st.caption("Analytics/Data Engineer Patterns")
sidebar()
