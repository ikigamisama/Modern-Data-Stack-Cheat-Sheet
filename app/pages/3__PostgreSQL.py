import streamlit as st
from components import sidebar

st.set_page_config(
    page_title="PostgreSQL Cheat Sheet",
    page_icon="🐘🗄️",
    layout="wide"
)


st.title("🐘🗄️  PostgreSQL Cheat Sheet")
st.caption("Analytics/Data Engineer Patterns")
sidebar()