import streamlit as st

from components import sidebar, reset_layout, init_layout_state
from components.pandas import (
    import_setup,
    create_dataframes,
    data_profiling,
    advanced_index,
    advance_data_types,
    advamce_data_cleaning,
    advance_aggregation,
    advance_window_function,
    advance_join_merges,
    advance_date_time,
    advance_IO_operations,
    analytics_patterns,
    performance_optimization,
    security_data_governance,
    best_practices,
    quick_reference
)

st.set_page_config(
    page_title="Pandas Cheat Sheet",
    page_icon="🐼📊",
    layout="wide"
)

st.title("🐼📊  Pandas Cheat Sheet")
st.caption("Analytics/Data Engineer Patterns")
sidebar()

left_column_defaults = [
    'import',
    'data',
    'advanced_data_types',
    'advance_aggregation',
    'advance_join_merges',
    'advance_IO_operations',
    'performance_optimization',
    'best_practices'
]
right_column_defaults = [
    'dataframe',
    'advance_indexing',
    'advance_data_cleaning',
    'advance_window_function',
    'advance_date_time',
    'analytics_patterns',
    'security_data_governance'
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
    'import': import_setup,
    'dataframe': create_dataframes,
    'data': data_profiling,
    'advance_indexing': advanced_index,
    'advanced_data_types': advance_data_types,
    'advance_data_cleaning': advamce_data_cleaning,
    'advance_aggregation': advance_aggregation,
    'advance_window_function': advance_window_function,
    'advance_join_merges': advance_join_merges,
    'advance_date_time': advance_date_time,
    'advance_IO_operations': advance_IO_operations,
    'analytics_patterns': analytics_patterns,
    'performance_optimization': performance_optimization,
    'security_data_governance': security_data_governance,
    'best_practices': best_practices
}
col1, col2 = st.columns(2)

with col1:
    for seg in left_col_segments:
        segment_dict[seg]()


with col2:
    for seg in right_col_segments:
        segment_dict[seg]()

quick_reference()
