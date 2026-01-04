import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx


class Flow:
    def __init__(self):
        self.title = "🌊 Flow Visualization Dashboard"
        self.chart_types = ["Sankey Diagram", "Alluvial Diagram", "Flow Chart"]
        self.dataset_types = [
            "Budget Flow", "Customer Journey", "Supply Chain",
            "Energy Systems", "Migration Patterns", "Website Flow"
        ]

    def render_header(self):
        st.markdown(f"{self.title}")
        st.markdown("""
        ### Purpose
        Flow charts track movement, transitions, and progression through states, processes, or categories. 
        They show how quantities move from one point to another and where they accumulate or disperse.
        """)

    def render_configuration(self):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            dataset_type = st.selectbox(
                "Dataset Type", self.dataset_types, key="dataset_type")

        with col3:
            flow_scale = st.slider("Flow Scale", 1, 10, 5, key="flow_scale")
            show_values = st.checkbox(
                "Show Values", value=True, key="show_values")

        return chart_type, dataset_type, flow_scale, show_values

    def generate_flow_data(self, data_type: str, scale_factor: int = 1) -> pd.DataFrame:
        np.random.seed(42)

        if data_type == "Budget Flow":
            sources = ['Tax Revenue', 'Bonds', 'Fed Funding', 'Tax Revenue', 'Tax Revenue',
                       'Bonds', 'Fed Funding', 'Health', 'Education', 'Infrastructure']
            targets = ['Health', 'Education', 'Infrastructure', 'Defense', 'Debt Service',
                       'Debt Service', 'State Grants', 'Hospitals', 'Schools', 'Roads']
            values = [300, 200, 150, 250, 100, 80, 70, 180, 220, 120]

        elif data_type == "Customer Journey":
            sources = ['Homepage', 'Homepage', 'Homepage', 'Product Page', 'Product Page',
                       'Cart', 'Cart', 'Checkout', 'Checkout', 'Payment']
            targets = ['Product Page', 'Category Page', 'Exit', 'Cart', 'Exit',
                       'Checkout', 'Exit', 'Payment', 'Exit', 'Confirmation']
            values = [1000, 600, 400, 700, 300, 500, 200, 450, 50, 400]

        elif data_type == "Supply Chain":
            sources = ['Supplier A', 'Supplier B', 'Supplier C', 'Manufacturer', 'Manufacturer',
                       'Distributor', 'Distributor', 'Warehouse', 'Warehouse', 'Retailer']
            targets = ['Manufacturer', 'Manufacturer', 'Manufacturer', 'Distributor', 'Warehouse',
                       'Warehouse', 'Retailer', 'Retailer', 'Customer', 'Customer']
            values = [5000, 3000, 2000, 7000,
                      3000, 6000, 4000, 5000, 4500, 8000]

        elif data_type == "Energy Systems":
            sources = ['Solar', 'Wind', 'Natural Gas', 'Coal', 'Hydro',
                       'Grid', 'Grid', 'Grid', 'Grid']
            targets = ['Grid', 'Grid', 'Grid', 'Grid', 'Grid',
                       'Residential', 'Commercial', 'Industrial', 'Losses']
            values = [200, 150, 500, 300, 100, 800, 400, 600, 150]

        elif data_type == "Migration Patterns":
            sources = ['North', 'North', 'North', 'South', 'South', 'South',
                       'East', 'East', 'East', 'West', 'West']
            targets = ['South', 'East', 'West', 'North', 'East', 'West',
                       'North', 'South', 'West', 'North', 'South']
            values = [50000, 30000, 20000, 40000, 25000, 15000,
                      35000, 45000, 20000, 30000, 40000]

        else:  # Website Flow
            sources = ['Home', 'Home', 'Home', 'Blog', 'Blog',
                       'Products', 'Products', 'Products', 'About', 'Contact']
            targets = ['Blog', 'Products', 'About', 'Products', 'Contact',
                       'Cart', 'Contact', 'Exit', 'Contact', 'Lead Form']
            values = [800, 600, 200, 500, 200, 400, 150, 50, 150, 120]

        values = [v * scale_factor for v in values]

        return pd.DataFrame({
            'source': sources,
            'target': targets,
            'value': values
        })

    def render_sankey_diagram(self, df: pd.DataFrame, data_type: str, show_values: bool):
        st.markdown("### 🌊 Sankey Diagram - Flow Quantities")

        labels = list(set(df['source']) | set(df['target']))
        label_dict = {label: i for i, label in enumerate(labels)}

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="steelblue"
            ),
            link=dict(
                source=[label_dict[s] for s in df['source']],
                target=[label_dict[t] for t in df['target']],
                value=df['value'],
                label=[str(v) if show_values else "" for v in df['value']],
                color="rgba(135, 206, 235, 0.6)"
            )
        )])

        fig.update_layout(
            title_text=f"{data_type} - Sankey Flow Diagram", font_size=12)
        st.plotly_chart(fig, use_container_width=True)

        total_flow = df['value'].sum()
        st.markdown("### 📊 Flow Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Flow", f"{total_flow:,}")
        col2.metric("Nodes", len(labels))
        col3.metric("Connections", len(df))

        # Top flows
        top_flows = df.sort_values('value', ascending=False).head(5)
        st.markdown("**Top 5 Flows:**")
        for _, row in top_flows.iterrows():
            pct = row['value'] / total_flow * 100
            st.markdown(
                f"**{row['source']} → {row['target']}**: {row['value']:,} ({pct:.1f}%)")

        st.markdown("""
        **When to use:** Visualizing magnitude of flows between categories (energy, money, materials).
        
        **Key Features:** Link width = flow volume, conservation of quantity.
        """)

        st.code('''
import plotly.graph_objects as go
fig = go.Figure(go.Sankey(
    node=dict(label=labels),
    link=dict(source=sources_idx, target=targets_idx, value=values)
))
fig.show()
        ''', language='python')

    def render_alluvial_diagram(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🔄 Alluvial Diagram - Category Transitions")

        if data_type == "Customer Journey":
            stages = ['Entry', 'Awareness',
                      'Consideration', 'Decision', 'Outcome']

            n_paths = min(800, max(100, 100 * df['value'].sum() // 1000))
            paths = []

            for _ in range(n_paths):
                path = []
                path.append(np.random.choice(
                    ['Homepage', 'Search', 'Social Media'], p=[0.5, 0.3, 0.2]))
                path.append(np.random.choice(
                    ['Product Page', 'Category Page', 'Blog'], p=[0.5, 0.3, 0.2]))

                if path[-1] == 'Product Page':
                    path.append(np.random.choice(
                        ['Cart', 'Wishlist', 'Exit'], p=[0.6, 0.2, 0.2]))
                else:
                    path.append('Exit')

                if path[-1] == 'Cart':
                    path.append(np.random.choice(['Checkout', 'Abandon']))
                elif path[-1] == 'Wishlist':
                    path.append(np.random.choice(['Abandon', 'Cart']))
                else:
                    path.append('Lost')

                if path[-1] == 'Checkout':
                    path.append(np.random.choice(['Purchase', 'One-time']))
                else:
                    path.append('Lost')

                paths.append(path[:len(stages)])

            alluvial_df = pd.DataFrame(paths, columns=stages)

            # Map Outcome (categorical) to numeric codes for coloring
            alluvial_df['Outcome_code'] = alluvial_df['Outcome'].astype(
                'category').cat.codes

            fig = px.parallel_categories(
                alluvial_df,
                dimensions=stages,
                color='Outcome_code',  # numeric for coloring
                color_continuous_scale=px.colors.sequential.Inferno,
                labels={'Outcome_code': 'Outcome'},
                title=f"{data_type} - User Path Transitions"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info(
                "Alluvial diagram best suited for Customer Journey. Showing Sankey instead:")
            self.render_sankey_diagram(df, data_type, True)

        st.markdown("""
        **When to use:** Tracking categorical changes over stages (e.g., user journeys, cohort transitions).
        
        **Key Features:** Shows individual path volume, highlights common routes and drop-offs.
        """)

    def render_flow_chart(self, df: pd.DataFrame, data_type: str, show_values: bool):
        st.markdown("### 📋 Flow Chart - Process Network")

        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(row['source'], row['target'], weight=row['value'])

        pos = nx.spring_layout(G, k=4, iterations=100, seed=42)

        fig, ax = plt.subplots(figsize=(14, 10))

        # Nodes
        node_sizes = [G.degree(n, weight='weight') *
                      10 + 500 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                               node_color='skyblue', alpha=0.9, ax=ax)

        # Edges
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(weights) if weights else 1
        edge_widths = [w / max_weight * 10 for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='gray',
                               arrows=True, arrowsize=20, arrowstyle='->', ax=ax)

        # Labels
        nx.draw_networkx_labels(G, pos, font_size=10,
                                font_weight='bold', ax=ax)

        if show_values:
            edge_labels = {
                (u, v): f"{G[u][v]['weight']:,}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

        ax.set_title(f"{data_type} - Directed Flow Network", fontsize=16)
        ax.axis('off')
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # Analysis
        st.markdown("### 🔍 Flow Analysis")
        in_flow = dict(G.in_degree(weight='weight'))
        out_flow = dict(G.out_degree(weight='weight'))

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Receivers (Inflow):**")
            for node, flow in sorted(in_flow.items(), key=lambda x: x[1], reverse=True)[:6]:
                st.markdown(f"📥 **{node}**: {flow:,}")

        with col2:
            st.write("**Top Senders (Outflow):**")
            for node, flow in sorted(out_flow.items(), key=lambda x: x[1], reverse=True)[:6]:
                st.markdown(f"📤 **{node}**: {flow:,}")

        # Bottlenecks
        st.markdown("### 🚧 Potential Bottlenecks")
        bottlenecks = []
        for node in G.nodes():
            inflow = in_flow.get(node, 0)
            outflow = out_flow.get(node, 0)
            if inflow > 0 and outflow > 0 and outflow / inflow < 0.8:
                loss = inflow - outflow
                bottlenecks.append((node, outflow / inflow, loss))

        if bottlenecks:
            for node, eff, loss in sorted(bottlenecks, key=lambda x: x[2], reverse=True)[:5]:
                st.warning(
                    f"**{node}** → {eff:.1%} efficiency (loses {loss:,} units)")
        else:
            st.success("No major bottlenecks detected")

        st.markdown("""
        **When to use:** Process mapping, workflow analysis, identifying constraints.
        
        **Key Features:** Directed edges, node centrality, bottleneck detection.
        """)

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Flow Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Directionality**
            - Clear source-to-destination movement
            - Tracks progression through stages
            """)
            st.markdown("""
            **Quantity Preservation**
            - Total input = total output (conservation)
            - Shows accumulation and dispersion
            """)

        with col2:
            st.markdown("""
            **Bottleneck Detection**
            - Identifies process constraints
            - Highlights inefficiencies
            """)
            st.markdown("""
            **Path Analysis**
            - Reveals major and minor pathways
            - Shows connection strength
            """)

    def render_examples(self, dataset_type: str, df: pd.DataFrame):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Budget Flow": "How funds move from revenue sources to spending categories",
            "Customer Journey": "User navigation paths through a website or app",
            "Supply Chain": "Material/product movement from suppliers to customers",
            "Energy Systems": "Power flow from generation to consumption",
            "Migration Patterns": "Population movement between regions",
            "Website Flow": "Traffic flow between site sections"
        }

        for example, desc in examples.items():
            with st.expander(f"🌊 {example}"):
                st.write(desc)
                if dataset_type == example:
                    total = df['value'].sum()
                    st.success(f"**Total Flow Volume:** {total:,} units")

    def render_data_table(self, df: pd.DataFrame):
        st.markdown("### 📊 Flow Data (Source → Target)")
        st.dataframe(df.sort_values('value', ascending=False),
                     use_container_width=True)

    def output(self):
        self.render_header()
        chart_type, dataset_type, flow_scale, show_values = self.render_configuration()

        df = self.generate_flow_data(dataset_type, flow_scale)

        chart_map = {
            "Sankey Diagram": lambda: self.render_sankey_diagram(df, dataset_type, show_values),
            "Alluvial Diagram": lambda: self.render_alluvial_diagram(df, dataset_type),
            "Flow Chart": lambda: self.render_flow_chart(df, dataset_type, show_values)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_key_characteristics()
        self.render_examples(dataset_type, df)
        self.render_data_table(df)
