import streamlit as st
from components import sidebar, reset_layout, init_layout_state
from components.pyspark import (
    setup_initial,
    create_dataframes,
    data_profiling,
    advance_column_operations,
    data_type_conversions,
    advanced_filtering,
    advance_aggregations,
    advance_window,
    advnace_joins_merge,
    advance_data_cleaning,
    advance_data_time,
    advnace_read_write_pattern,
    debugging_optimization,
    analytics_patterns,
    security_governance,
    performance_tuning,
    quick_refence
)

st.set_page_config(
    page_title="PySpark Cheat Sheet",
    page_icon="🔥⚡",
    layout="wide"
)

st.title("🔥⚡  PySpark Cheat Sheet")
st.caption("Analytics/Data Engineer Patterns")
sidebar()

left_column_defaults = [
    'setup_initial',
    'data_profiling',
    'data_type_conversions',
    'advance_aggregations',
    'advnace_joins_merge',
    'advance_data_time',
    'debugging_optimization',
    'security_governance'
]
right_column_defaults = [
    'create_dataframes',
    'advance_column_operations',
    'advanced_filtering',
    'advance_window',
    'advance_data_cleaning',
    'advnace_read_write_pattern',
    'analytics_patterns',
    'performance_tuning'
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


segment_dict = {
    'setup_initial': setup_initial,
    'create_dataframes': create_dataframes,
    'data_profiling': data_profiling,
    'advance_column_operations': advance_column_operations,
    'data_type_conversions': data_type_conversions,
    'advanced_filtering': advanced_filtering,
    'advance_aggregations': advance_aggregations,
    'advance_window': advance_window,
    'advnace_joins_merge': advnace_joins_merge,
    'advance_data_cleaning': advance_data_cleaning,
    'advance_data_time': advance_data_time,
    'advnace_read_write_pattern': advnace_read_write_pattern,
    'debugging_optimization': debugging_optimization,
    'analytics_patterns': analytics_patterns,
    'security_governance': security_governance,
    'performance_tuning': performance_tuning,
    'quick_refence': quick_refence
}

col1, col2 = st.columns(2)

with col1:
    for seg in left_col_segments:
        segment_dict[seg]()

with col2:
    for seg in right_col_segments:
        segment_dict[seg]()


quick_refence()
