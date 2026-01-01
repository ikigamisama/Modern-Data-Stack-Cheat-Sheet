import streamlit as st
from components import sidebar, reset_layout, init_layout_state
from components.postgresql import (
    essential_commands,
    exploring_data,
    filtering_selecting_data,
    aggregations_group,
    join_combine,
    window_function,
    date_time_operations,
    cte,
    case_statements,
    subqueries,
    userful_string_functions,
    data_export_import,
    quick_tips
)

st.set_page_config(
    page_title="PostgreSQL Cheat Sheet",
    page_icon="🐘🗄️",
    layout="wide"
)

st.title("🐘🗄️  PostgreSQL Cheat Sheet")
st.caption("Analytics/Data Engineer Patterns")
sidebar()

left_column_defaults = [
    'essential_commands',
    'filtering_selecting_data',
    'join_combine',
    'date_time_operations',
    'case_statements',
    'userful_string_functions'
]
right_column_defaults = [
    'exploring_data',
    'aggregations_group',
    'window_function',
    'cte',
    'subqueries',
    'data_export_import'
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
    'essential_commands': essential_commands,
    'exploring_data': exploring_data,
    'filtering_selecting_data': filtering_selecting_data,
    'aggregations_group': aggregations_group,
    'join_combine': join_combine,
    'window_function': window_function,
    'date_time_operations': date_time_operations,
    'cte': cte,
    'case_statements': case_statements,
    'subqueries': subqueries,
    'userful_string_functions': userful_string_functions,
    'data_export_import': data_export_import,
}
col1, col2 = st.columns(2)

with col1:
    for seg in left_col_segments:
        segment_dict[seg]()

with col2:
    for seg in right_col_segments:
        segment_dict[seg]()

quick_tips()
