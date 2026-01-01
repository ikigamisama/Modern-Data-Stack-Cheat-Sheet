import streamlit as st
from components import sidebar, reset_layout, init_layout_state
from components.mysql import (
    connection_authentication,
    database_operations,
    table_operations,
    column_operations,
    data_types,
    crud_operations,
    where_clauses,
    joins,
    window_functions,
    analytics_use_cast_example,
    advanced_aggregations,
    cte,
    data_quality_checks,
    data_profling,
    date_time_analytics,
    performance_optimization,
    sampling_techniques,
    string_functions_data_clearning,
    materialize_views,
    data_export_analysis,
    indexes,
    views,
    stored_procedures,
    transactions,
    user_management,
    system_information,
    json_functions
)

st.set_page_config(
    page_title="MySQL Cheat Sheet",
    page_icon="🐬🗄️",
    layout="wide"
)


st.title("🐬🗄️  MySQL Cheat Sheet")
st.caption("Analytics/Data Engineer Patterns")
sidebar()

left_column_defaults = [
    'connection_authentication',
    'table_operations',
    'data_types',
    'where_clauses',
    'window_functions',
    'advanced_aggregations',
    'data_quality_checks',
    'date_time_analytics',
    'sampling_techniques',
    'materialize_views',
    'indexes',
    'stored_procedures',
    'user_management',
    'json_functions'
]
right_column_defaults = [
    'database_operations',
    'column_operations',
    'crud_operations',
    'joins',
    'analytics_use_cast_example',
    'cte',
    'data_profling',
    'performance_optimization',
    'string_functions_data_clearning',
    'data_export_analysis',
    'views',
    'transactions',
    'system_information'
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
    'connection_authentication': connection_authentication,
    'database_operations': database_operations,
    'table_operations': table_operations,
    'column_operations': column_operations,
    'data_types': data_types,
    'crud_operations': crud_operations,
    'where_clauses': where_clauses,
    'joins': joins,
    'window_functions': window_functions,
    'analytics_use_cast_example': analytics_use_cast_example,
    'advanced_aggregations': advanced_aggregations,
    'cte': cte,
    'data_quality_checks': data_quality_checks,
    'data_profling': data_profling,
    'date_time_analytics': date_time_analytics,
    'performance_optimization': performance_optimization,
    'sampling_techniques': sampling_techniques,
    'string_functions_data_clearning': string_functions_data_clearning,
    'materialize_views': materialize_views,
    'data_export_analysis': data_export_analysis,
    'indexes': indexes,
    'views': views,
    'stored_procedures': stored_procedures,
    'transactions': transactions,
    'user_management': user_management,
    'system_information': system_information,
    'json_functions': json_functions
}
col1, col2 = st.columns(2)

with col1:
    for seg in left_col_segments:
        segment_dict[seg]()

with col2:
    for seg in right_col_segments:
        segment_dict[seg]()
