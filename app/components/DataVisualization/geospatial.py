import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class Geospatial:
    def __init__(self):
        self.title = "🗺️ Geospatial Visualization Dashboard"
        self.map_types = ["Choropleth Map",
                          "Bubble Map", "Point Map", "Grid Map"]
        self.dataset_types = [
            "Sales Territory", "Public Health", "Political Analysis",
            "Retail Planning", "Environmental", "Population Density"
        ]
        self.region_scopes = ["US States", "World Countries", "European Union"]

        # Predefined approximate centroids for major US states (for bubble/point maps)
        self.us_state_centroids = {
            'California': (36.7783, -119.4179),
            'Texas': (31.9686, -99.9018),
            'Florida': (27.9943, -81.7603),
            'New York': (43.2994, -74.2179),
            'Illinois': (40.6331, -89.3985),
            'Ohio': (40.4173, -82.9071),
            'Georgia': (32.9006, -83.4321),
            'Michigan': (44.1822, -84.5068),
            'North Carolina': (35.7822, -80.7935),
            'Arizona': (34.2744, -111.6602),
            'Colorado': (39.1130, -105.3589),
            'Washington': (47.7511, -120.7401),
            'Pennsylvania': (41.2033, -77.1945),
            'Virginia': (37.4316, -78.6569),
        }

        # Major cities for point maps
        self.us_city_points = {
            # LA, SF, SD
            'California': [[34.05, -118.25], [37.77, -122.42], [32.72, -117.16]],
            # Houston, Dallas, Austin
            'Texas': [[29.76, -95.36], [32.78, -96.80], [30.27, -97.74]],
            # Miami, Orlando, Jacksonville
            'Florida': [[25.76, -80.19], [28.54, -81.38], [30.33, -81.66]],
            # NYC, Buffalo, Rochester
            'New York': [[40.71, -74.01], [42.89, -78.88], [43.16, -77.61]],
        }

    def render_header(self):
        st.title(f"{self.title}")
        st.markdown("""
        ### Purpose
        Geospatial visualizations display data in geographic context, revealing spatial patterns, 
        regional differences, and location-based insights. They connect data values to physical places.
        """)

    def render_configuration(self):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            map_type = st.selectbox(
                "Select Map Type", self.map_types, key="map_type")

        with col2:
            dataset_type = st.selectbox(
                "Dataset Type", self.dataset_types, key="dataset_type")

        with col3:
            region_scope = st.selectbox(
                "Region Scope", self.region_scopes, key="region_scope")
            show_legend = st.checkbox(
                "Show Legend", value=True, key="show_legend")

        return map_type, dataset_type, region_scope, show_legend

    def generate_geospatial_data(self, data_type: str, scope: str) -> pd.DataFrame:
        np.random.seed(42)

        if scope == "US States":
            states = list(self.us_state_centroids.keys())

            common_cols = {'state': states}

            if data_type == "Sales Territory":
                return pd.DataFrame({
                    **common_cols,
                    'revenue': np.random.randint(1_000_000, 5_000_000, len(states)),
                    'growth_rate': np.random.uniform(-5, 15, len(states)),
                    'sales_team_size': np.random.randint(5, 50, len(states))
                })

            elif data_type == "Public Health":
                return pd.DataFrame({
                    **common_cols,
                    'cases': np.random.randint(1_000, 10_000, len(states)),
                    'vaccination_rate': np.random.uniform(60, 95, len(states)),
                    'hospitals_per_capita': np.random.uniform(0.1, 0.5, len(states))
                })

            elif data_type == "Political Analysis":
                dem = np.random.randint(1_000_000, 3_000_000, len(states))
                rep = np.random.randint(800_000, 2_800_000, len(states))
                return pd.DataFrame({
                    **common_cols,
                    'democrat_votes': dem,
                    'republican_votes': rep,
                    'margin': dem - rep
                })

            elif data_type == "Retail Planning":
                return pd.DataFrame({
                    **common_cols,
                    'store_count': np.random.randint(10, 200, len(states)),
                    'avg_sales_per_store': np.random.uniform(50_000, 200_000, len(states)),
                    'market_potential': np.random.uniform(0.5, 1.5, len(states))
                })

            elif data_type == "Environmental":
                return pd.DataFrame({
                    **common_cols,
                    'pollution_index': np.random.uniform(20, 100, len(states)),
                    'green_area_percent': np.random.uniform(10, 50, len(states)),
                    'renewable_energy': np.random.uniform(10, 60, len(states))
                })

            else:  # Population Density
                return pd.DataFrame({
                    **common_cols,
                    'population': np.random.randint(1_000_000, 40_000_000, len(states)),
                    'density_per_sq_mile': np.random.uniform(50, 1200, len(states)),
                    'urban_population_pct': np.random.uniform(60, 95, len(states))
                })

        elif scope == "World Countries":
            countries = ['United States', 'China', 'India', 'Brazil', 'Russia', 'Germany',
                         'United Kingdom', 'France', 'Japan', 'Canada', 'Australia', 'Mexico']

            base = pd.DataFrame({'country': countries})

            if data_type == "Sales Territory":
                return pd.concat([base, pd.DataFrame({
                    'revenue': np.random.randint(5_000_000, 50_000_000, len(countries)),
                    'growth_rate': np.random.uniform(-2, 20, len(countries))
                })], axis=1)

            elif data_type == "Public Health":
                return pd.concat([base, pd.DataFrame({
                    'cases_per_million': np.random.randint(10_000, 1_000_000, len(countries)),
                    'life_expectancy': np.random.uniform(65, 85, len(countries))
                })], axis=1)

            else:
                value = np.random.randint(1_000, 100_000, len(countries))
                return pd.concat([base, pd.DataFrame({
                    'value': value,
                    'growth': np.random.uniform(-10, 30, len(countries))
                })], axis=1)

        else:  # European Union fallback
            return self.generate_geospatial_data(data_type, "World Countries")

    def render_choropleth_map(self, df: pd.DataFrame, data_type: str, scope: str):
        st.markdown("### 🎨 Choropleth Map - Regional Coloring")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = st.selectbox("Color by", numeric_cols, index=0)

        if scope == "US States":
            fig = px.choropleth(
                df,
                locations='state',
                locationmode='USA-states',
                color=value_col,
                scope='usa',
                color_continuous_scale='Viridis',
                title=f"{data_type} - {value_col} by US State"
            )
        else:  # World or EU
            fig = px.choropleth(
                df,
                locations='country' if 'country' in df.columns else 'state',
                locationmode='country names',
                color=value_col,
                color_continuous_scale='Plasma',
                title=f"{data_type} - {value_col} by Country"
            )

        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        **When to use:** Comparing aggregated values across regions (e.g., totals, rates, averages).
        
        **Key Features:** Color intensity reflects data magnitude; intuitive for spatial trends.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import plotly.express as px
                
numeric_cols = df.select_dtypes(include=[np.number]).columns
value_col = st.selectbox("Color by", numeric_cols, index=0)

if scope == "US States":
    fig = px.choropleth(
        df,
        locations='state',
        locationmode='USA-states',
        color=value_col,
        scope='usa',
        color_continuous_scale='Viridis',
        title=f"{data_type} - {value_col} by US State"
    )
else:  # World or EU
    fig = px.choropleth(
        df,
        locations='country' if 'country' in df.columns else 'state',
        locationmode='country names',
        color=value_col,
        color_continuous_scale='Plasma',
        title=f"{data_type} - {value_col} by Country"
    )

fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
fig.show()''', language='python')

    def render_bubble_map(self, df: pd.DataFrame, data_type: str, scope: str):
        st.markdown("### 💭 Bubble Map - Proportional Symbols")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        col1, col2 = st.columns(2)
        with col1:
            size_col = st.selectbox("Bubble Size", numeric_cols, index=0)
        with col2:
            color_col = st.selectbox(
                "Bubble Color", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

        if scope == "US States":
            df['lat'] = df['state'].map(
                {k: v[0] for k, v in self.us_state_centroids.items()})
            df['lon'] = df['state'].map(
                {k: v[1] for k, v in self.us_state_centroids.items()})
            df = df.dropna(subset=['lat', 'lon'])

            fig = px.scatter_geo(
                df, lat='lat', lon='lon',
                size=size_col, color=color_col,
                hover_name='state',
                scope='north america',
                title=f"{data_type} - Bubble Map (US)"
            )
        else:
            location_col = 'country' if 'country' in df.columns else 'state'
            fig = px.scatter_geo(
                df, locations=location_col, locationmode='country names',
                size=size_col, color=color_col,
                hover_name=location_col,
                title=f"{data_type} - Bubble Map (World)"
            )

        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        **When to use:** Showing magnitude + secondary metric across locations.
        
        **Key Features:** Size = quantity, color = another variable (e.g., growth).
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''import plotly.express as px
                
fig = px.scatter_geo(df, lat='lat', lon='lon', size='revenue', color='growth',
                     hover_name='state', scope='north america')
fig.show()
        ''', language='python')

    def render_point_map(self, df: pd.DataFrame, data_type: str, scope: str):
        st.markdown("### 📍 Point Map - Location Markers")

        points_data = []
        np.random.seed(42)

        if scope == "US States":
            for state in df['state'].dropna().unique():
                if state in self.us_city_points:
                    for lat, lon in self.us_city_points[state]:
                        points_data.append({
                            'state': state,
                            'lat': lat,
                            'lon': lon,
                            'value': np.random.randint(100, 1000)
                        })

        points_df = pd.DataFrame(points_data)

        if not points_df.empty:
            fig = px.scatter_geo(
                points_df,
                lat='lat', lon='lon',
                size='value', color='value',
                hover_name='state',
                scope='usa',
                color_continuous_scale='Reds',
                title=f"{data_type} - Key Locations (Cities)"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Point-level city data available only for US States scope.")

        st.markdown("""
        **When to use:** Displaying individual events, facilities, or precise locations.
        
        **Key Features:** Exact coordinates, great for stores, hospitals, incidents.
        """)
        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import plotly.express as px
                
fig = px.scatter_geo(df_points, lat='lat', lon='lon', size='value',
                     hover_name='location', scope='usa')
fig.show()
        ''', language='python')

    def render_grid_map(self, df: pd.DataFrame, data_type: str, scope: str):
        st.markdown("### 🔲 Grid Map - Equal Area Representation")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = st.selectbox("Value for Size/Color", numeric_cols, index=0)

        fig = go.Figure()

        # Simple 4x4 grid layout
        grid_coords = {}
        row, col = 0, 0
        for i, state in enumerate(df['state' if 'state' in df.columns else 'country']):
            grid_coords[state] = (col, row)
            col += 1
            if col > 3:
                col = 0
                row += 1

        max_val = df[value_col].max()
        for _, row in df.iterrows():
            location = row['state'] if 'state' in df.columns else row['country']
            if location in grid_coords:
                x, y = grid_coords[location]
                size = row[value_col] / max_val * 100
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=size, color=row[value_col], colorscale='Viridis', showscale=True),
                    text=location,
                    textposition="bottom center",
                    name=location,
                    hoverinfo='text',
                    hovertext=f"{location}<br>{value_col}: {row[value_col]:,.0f}"
                ))

        fig.update_layout(
            title=f"{data_type} - Grid Map (Equal Area View)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=600,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        **When to use:** When fair visual comparison matters more than geographic accuracy.
        
        **Key Features:** Each region gets equal space — avoids bias from large land areas.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import plotly.graph_objects as go
                
fig = go.Figure()
# Add scattered points on grid with size/color encoding
fig.add_trace(go.Scatter(x=[x], y=[y], marker=dict(size=scaled_value)))
fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False)
fig.show()
        ''', language='python')

    def render_key_characteristics(self):
        st.markdown(
            "### 🗺️ Understanding Geospatial and Location-Based Analysis")
        st.markdown("""
        Geospatial analysis connects data to real-world locations, allowing analysts to
        leverage **spatial intuition** to uncover patterns, clusters, and regional differences.
        """)

        st.markdown("#### 🧠 Leveraging Spatial Intuition")
        st.markdown("""
        People naturally understand space, distance, and proximity.
        Geospatial visualizations convert abstract data into familiar geographic forms,
        making insights faster to interpret and easier to communicate.
        """)

        st.markdown("####📍 Showing Geographic Patterns and Clusters")
        st.markdown("""
        Mapping data reveals spatial clustering—areas where values or events concentrate.

        Examples include:
        - High-demand regions  
        - Risk or incident hotspots  
        - Population or infrastructure density  
        """)

        st.markdown("#### ⚖️ Revealing Regional Disparities")
        st.markdown("""
        Geospatial views expose differences between regions, such as performance gaps,
        economic inequality, or uneven resource distribution.

        This enables side-by-side regional comparison and targeted analysis.
        """)

        st.markdown("#### 📌 Enabling Location-Based Decision Making")
        st.markdown("""
        Location-aware insights support decisions that depend on **where** something happens.

        Common applications:
        - Site selection  
        - Logistics and routing  
        - Market expansion  
        - Urban and infrastructure planning  
        """)

        st.divider()

        st.markdown("#### 🎯 Why Geospatial Analysis Matters")
        st.markdown("""
        Embedding data in physical space transforms metrics into actionable intelligence.
        Geospatial analysis enables smarter resource allocation, risk mitigation,
        and geographically informed strategies.
        """)

    def render_examples(self, dataset_type: str, map_type: str, region_scope: str):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Sales Territory": "Revenue performance across different regions",
            "Public Health": "Disease incidence and vaccination rates by state",
            "Political Analysis": "Election results and voting margins",
            "Retail Planning": "Store density and sales performance",
            "Environmental": "Pollution levels and green coverage",
            "Population Density": "Urban vs rural population distribution"
        }

        for example, description in examples.items():
            with st.expander(f"🗺️ {example}"):
                st.write(description)
                if dataset_type == example:
                    st.success(f"**Current Map Type:** {map_type}")
                    st.info(f"**Region Scope:** {region_scope}")

    def output(self):
        self.render_header()
        map_type, dataset_type, region_scope, show_legend = self.render_configuration()

        df = self.generate_geospatial_data(dataset_type, region_scope)

        chart_map = {
            "Choropleth Map": lambda: self.render_choropleth_map(df, dataset_type, region_scope),
            "Bubble Map": lambda: self.render_bubble_map(df, dataset_type, region_scope),
            "Point Map": lambda: self.render_point_map(df, dataset_type, region_scope),
            "Grid Map": lambda: self.render_grid_map(df, dataset_type, region_scope)
        }

        if map_type in chart_map:
            chart_map[map_type]()

        self.render_examples(dataset_type, map_type, region_scope)
        self.render_key_characteristics()
