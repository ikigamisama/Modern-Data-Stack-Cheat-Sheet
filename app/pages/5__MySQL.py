import streamlit as st
from components import sidebar, reset_layout, init_layout_state

st.set_page_config(
    page_title="MySQL Cheat Sheet",
    page_icon="🐬🗄️",
    layout="wide"
)


st.title("🐬🗄️  MySQL Cheat Sheet")
st.caption("Analytics/Data Engineer Patterns")
sidebar()

left_column_defaults = [

]
right_column_defaults = [

]

all_segments = left_column_defaults + right_column_defaults
init_layout_state(left_column_defaults, right_column_defaults)

custom_layout = st.sidebar.expander("🧑‍🎨 Customize Layout")
with custom_layout:
    st.button(
        "Default Layout",
        on_click=reset_layout,
        args=(left_column_defaults, right_column_defaults),
    )
    side_left_col, side_right_col = st.columns(2)
    left_col_segments = side_left_col.multiselect("Left Column",
                                                  options=all_segments,
                                                  default=left_column_defaults,
                                                  key="layout_left_column")

    right_col_segments = side_right_col.multiselect("Right Column",
                                                    options=all_segments,
                                                    default=right_column_defaults,
                                                    key="layout_right_column")

segment_dict = {}
col1, col2 = st.columns(2)

with col1:
    for seg in left_col_segments:
        segment_dict[seg]()

with col2:
    for seg in right_col_segments:
        segment_dict[seg]()
