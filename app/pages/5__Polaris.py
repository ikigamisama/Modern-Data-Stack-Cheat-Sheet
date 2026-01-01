import streamlit as st
from components import sidebar, reset_layout, init_layout_state
from components.polaris import (
    import_setup,
    create_dataframe,
    data_profiling,
    advance_column_selection,
    advance_data_types,
    advance_filtering,
    advance_aggregations,
    advance_window_function,
    advance_joins_relationship,
    advance_data_cleaning,
    advance_date_time,
    advance_io_operations,
    analytics_patterns,
    performance_optimization,
    security_data_governance,
    best_practice,
    quick_refences
)

st.set_page_config(
    page_title="Polars Cheat Sheet",
    page_icon="⚡📊",
    layout="wide"
)

st.title("⚡📊  Polars Cheat Sheet")
st.caption("Analytics/Data Engineer Patterns")
sidebar()

left_column_defaults = [
    'import_setup',
    'data_profiling',
    'advance_data_types',
    'advance_aggregations',
    'advance_joins_relationship',
    'advance_date_time',
    'analytics_patterns',
    'security_data_governance'
]
right_column_defaults = [
    'create_dataframe',
    'advance_column_selection',
    'advance_filtering',
    'advance_window_function',
    'advance_data_cleaning',
    'advance_io_operations',
    'performance_optimization',
    'best_practice'
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
    'import_setup': import_setup,
    'create_dataframe': create_dataframe,
    'data_profiling': data_profiling,
    'advance_column_selection': advance_column_selection,
    'advance_data_types': advance_data_types,
    'advance_filtering': advance_filtering,
    'advance_aggregations': advance_aggregations,
    'advance_window_function': advance_window_function,
    'advance_joins_relationship': advance_joins_relationship,
    'advance_data_cleaning': advance_data_cleaning,
    'advance_date_time': advance_date_time,
    'advance_io_operations': advance_io_operations,
    'analytics_patterns': analytics_patterns,
    'performance_optimization': performance_optimization,
    'security_data_governance': security_data_governance,
    'best_practice': best_practice,
}
col1, col2 = st.columns(2)

with col1:
    for seg in left_col_segments:
        segment_dict[seg]()

with col2:
    for seg in right_col_segments:
        segment_dict[seg]()

quick_refences()
