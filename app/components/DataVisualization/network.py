import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pyvis.network import Network
from matplotlib.patches import Arc


class Network:
    def __init__(self):
        self.title = "🕸️ Network Visualization Dashboard"
        self.chart_types = ["Force-Directed Graph",
                            "Chord Diagram", "Arc Diagram", "Sociogram"]
        self.dataset_types = [
            "Social Media", "Organization Charts", "Computer Networks",
            "Citation Analysis", "Transportation", "Knowledge Graphs"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Network charts display connections and relationships between entities as nodes and edges. 
        They reveal community structures, influence patterns, and the architecture of complex systems.
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
            num_nodes = st.slider("Number of Nodes", 10,
                                  50, 20, key="num_nodes")
            show_communities = st.checkbox(
                "Detect Communities", value=True, key="show_comm")

        return chart_type, dataset_type, num_nodes, show_communities

    def generate_network_data(self, data_type: str, n_nodes: int) -> nx.Graph:
        np.random.seed(42)
        G = nx.Graph()

        # Node naming
        if data_type == "Social Media":
            nodes = [f"User_{i:02d}" for i in range(n_nodes)]
            nodes[0] = "Influencer_A"
            nodes[5] = "Brand_Official"
            nodes[10] = "Celebrity_X"
        elif data_type == "Organization Charts":
            roles = ['CEO', 'CTO', 'CFO', 'VP_Sales', 'VP_Marketing'] + \
                ['Manager'] * 5 + ['Employee'] * (n_nodes - 10)
            nodes = roles[:n_nodes]
            nodes[0] = "CEO"
        elif data_type == "Computer Networks":
            types = ['Server', 'Router', 'Switch', 'Workstation', 'Laptop']
            nodes = [
                f"{np.random.choice(types)}_{i:02d}" for i in range(n_nodes)]
            nodes[0] = "Main_Server"
            nodes[5] = "Core_Router"
        elif data_type == "Citation Analysis":
            nodes = [
                f"Paper_{chr(65 + i//10)}{i % 10}" for i in range(n_nodes)]
            nodes[0] = "Foundational_Paper"
            nodes[5] = "Highly_Cited_Review"
        elif data_type == "Transportation":
            nodes = [
                f"Station_{chr(65 + i//10)}{i % 10}" for i in range(n_nodes)]
            nodes[0] = "Central_Hub"
            nodes[8] = "International_Airport"
        else:  # Knowledge Graphs
            types = ['Person', 'Organization', 'Location', 'Concept', 'Event']
            nodes = [
                f"{np.random.choice(types)}_{i:02d}" for i in range(n_nodes)]

        G.add_nodes_from(nodes)

        # Edge generation based on domain
        if data_type == "Social Media":
            influencers = [0, 5, 10]
            for i in range(n_nodes):
                connections = np.random.randint(
                    4, 12) if i in influencers else np.random.randint(2, 7)
                targets = np.random.choice(
                    [j for j in range(n_nodes) if j != i], connections, replace=False)
                for t in targets:
                    weight = np.random.randint(
                        3, 15) if i in influencers or t in influencers else np.random.randint(1, 5)
                    G.add_edge(nodes[i], nodes[t], weight=weight)

        elif data_type == "Organization Charts":
            # Hierarchical structure
            levels = {'CEO': 0, 'CTO': 1, 'CFO': 1, 'VP_Sales': 1, 'VP_Marketing': 1,
                      'Manager': 2, 'Employee': 3}
            for i, node in enumerate(nodes):
                level = levels.get(node.split(
                    '_')[0] if '_' in node else node, 3)
                possible = [j for j, n in enumerate(nodes) if levels.get(
                    n.split('_')[0] if '_' in n else n, 3) < level]
                if possible:
                    G.add_edge(
                        node, nodes[np.random.choice(possible)], weight=1)

        elif data_type == "Computer Networks":
            hubs = [0, 5, 10]
            for i in range(n_nodes):
                if i in hubs:
                    targets = np.random.choice(
                        [j for j in range(n_nodes) if j != i], np.random.randint(10, 20), replace=False)
                else:
                    targets = np.random.choice(hubs + [j for j in range(
                        n_nodes) if j != i and j not in hubs], np.random.randint(2, 6), replace=False)
                for t in targets:
                    G.add_edge(nodes[i], nodes[t],
                               weight=np.random.randint(1, 10))

        elif data_type == "Citation Analysis":
            for i in range(1, n_nodes):
                cites = np.random.randint(2, 8)
                older = np.random.choice(range(i), cites, replace=False)
                for o in older:
                    G.add_edge(nodes[i], nodes[o], weight=1)

        else:  # General/Transportation/Knowledge
            hubs = list(range(0, n_nodes, max(5, n_nodes//6)))
            for i in range(n_nodes):
                connections = np.random.randint(
                    6, 12) if i in hubs else np.random.randint(2, 6)
                targets = np.random.choice(
                    [j for j in range(n_nodes) if j != i], connections, replace=False)
                for t in targets:
                    G.add_edge(nodes[i], nodes[t],
                               weight=np.random.randint(1, 8))

        return G

    def render_force_directed_graph(self, G: nx.Graph, data_type: str, show_communities: bool):
        st.markdown("### 🌀 Force-Directed Graph - Interactive Layout")

        # Compute force-directed layout
        pos = nx.spring_layout(G, seed=42)  # force-directed positions

        # Edges
        edge_x = []
        edge_y = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        # Nodes
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = [f"{n}<br>Degree: {G.degree(n)}" for n in G.nodes()]
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n for n in G.nodes()],
            textposition='top center',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=[5 + G.degree(n)*3 for n in G.nodes()],
                color=[G.degree(n) for n in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Degree")
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False,
                                       showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False,
                                       showticklabels=False),
                            height=700
        ))

        st.plotly_chart(fig, use_container_width=True)

    def render_chord_diagram(self, G: nx.Graph, data_type: str):
        st.markdown("### 🎵 Chord Diagram - Flow Between Entities")

        matrix = nx.to_numpy_array(G)
        nodes = list(G.nodes())

        fig = go.Figure()

        # Create chord-like heatmap
        fig.add_trace(go.Heatmap(
            z=matrix,
            x=nodes,
            y=nodes,
            colorscale='Viridis',
            hovertemplate='From %{y} to %{x}: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=f"{data_type} - Connection Matrix (Chord Representation)",
            xaxis_title="Target",
            yaxis_title="Source",
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

        # Flow analysis
        in_strength = matrix.sum(axis=0)
        out_strength = matrix.sum(axis=1)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Top Receivers (Incoming):**")
            top_in = sorted(zip(nodes, in_strength),
                            key=lambda x: x[1], reverse=True)[:6]
            for node, val in top_in:
                st.success(f"📥 {node}: {val:.0f}")

        with col2:
            st.write("**Top Senders (Outgoing):**")
            top_out = sorted(zip(nodes, out_strength),
                             key=lambda x: x[1], reverse=True)[:6]
            for node, val in top_out:
                st.info(f"📤 {node}: {val:.0f}")

    def render_arc_diagram(self, G: nx.Graph, data_type: str):
        st.markdown("### 〰️ Arc Diagram - Linear Connection View")

        nodes = list(G.nodes())
        pos = {node: i for i, node in enumerate(nodes)}

        fig, ax = plt.subplots(figsize=(14, 8))

        # Nodes
        degrees = dict(G.degree())
        sizes = [100 + degrees[n] * 30 for n in nodes]
        ax.scatter(range(len(nodes)), [
                   0]*len(nodes), s=sizes, c='skyblue', alpha=0.8, edgecolors='black')

        # Labels
        for i, node in enumerate(nodes):
            ax.text(i, -0.05, node, rotation=45,
                    ha='right', va='top', fontsize=9)

        # Arcs
        for u, v in G.edges():
            i, j = pos[u], pos[v]
            if i > j:
                i, j = j, i
            height = (j - i) / 4
            center = (i + j) / 2
            arc = Arc((center, height), (j - i)*1.1, height*2,
                      theta1=0, theta2=180, linewidth=2, color='gray', alpha=0.6)
            ax.add_patch(arc)

        ax.set_xlim(-1, len(nodes))
        ax.set_ylim(-0.3, len(nodes)//3)
        ax.axis('off')
        ax.set_title(f"{data_type} - Arc Diagram")

        st.pyplot(fig)
        plt.close(fig)

    def render_sociogram(self, G: nx.Graph, data_type: str):
        st.markdown("### 👥 Sociogram - Social Structure")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Spring layout
        pos1 = nx.spring_layout(G, k=3, iterations=100, seed=42)
        degrees = dict(G.degree())
        sizes = [200 + degrees[n] * 40 for n in G.nodes()]

        # Role coloring
        colors = []
        for n in G.nodes():
            if any(k in n for k in ['CEO', 'Influencer', 'Celebrity', 'Server', 'Hub']):
                colors.append('gold')
            elif any(k in n for k in ['VP', 'Manager', 'Router']):
                colors.append('lightblue')
            else:
                colors.append('lightgreen')

        nx.draw_networkx_nodes(G, pos1, node_size=sizes,
                               node_color=colors, alpha=0.9, ax=ax1)
        nx.draw_networkx_edges(G, pos1, alpha=0.5, ax=ax1)
        nx.draw_networkx_labels(G, pos1, font_size=9, ax=ax1)
        ax1.set_title("Spring Layout - Natural Clusters")
        ax1.axis('off')

        # Circular layout
        pos2 = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos2, node_size=sizes,
                               node_color=colors, alpha=0.9, ax=ax2)
        nx.draw_networkx_edges(G, pos2, alpha=0.5, ax=ax2)
        nx.draw_networkx_labels(G, pos2, font_size=9, ax=ax2)
        ax2.set_title("Circular Layout - Clear Individual Links")
        ax2.axis('off')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Social roles
        st.markdown("### 👑 Key Social Roles")
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Most Connected (Influencers):**")
            top_deg = sorted(deg_cent.items(),
                             key=lambda x: x[1], reverse=True)[:5]
            for n, c in top_deg:
                st.success(f"{n} — Degree: {G.degree(n)}")

        with col2:
            st.write("**Key Bridges (Connectors):**")
            top_bet = sorted(bet_cent.items(),
                             key=lambda x: x[1], reverse=True)[:5]
            for n, c in top_bet:
                st.info(f"{n} — Betweenness: {c:.4f}")

    def render_key_characteristics(self):
        st.markdown("### 🎯 Key Characteristics of Network Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Interconnectedness**
            - Reveals hidden relationships
            - Shows direct and indirect links
            """)
            st.markdown("""
            **Centrality & Influence**
            - Identifies key players
            - Measures node importance
            """)

        with col2:
            st.markdown("""
            **Community Structure**
            - Detects natural clusters
            - Shows group formation
            """)
            st.markdown("""
            **Flow & Bottlenecks**
            - Highlights critical paths
            - Identifies vulnerabilities
            """)

    def render_examples(self, dataset_type: str, G: nx.Graph):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Social Media": "Follower networks and influence spread",
            "Organization Charts": "Reporting lines and collaboration",
            "Computer Networks": "Device connectivity and traffic",
            "Citation Analysis": "Academic influence and knowledge flow",
            "Transportation": "Routes and transfer points",
            "Knowledge Graphs": "Entity relationships and semantics"
        }

        for ex, desc in examples.items():
            with st.expander(f"🕸️ {ex}"):
                st.write(desc)
                if dataset_type == ex:
                    st.success(
                        f"**Current Network:** {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    def render_edge_preview(self, G: nx.Graph):
        st.markdown("### 📋 Sample Connections")
        edges = [(u, v, d.get('weight', 1)) for u, v, d in G.edges(data=True)]
        df = pd.DataFrame(edges[:20], columns=['Source', 'Target', 'Weight'])
        st.dataframe(df, use_container_width=True)

    def output(self):
        self.render_header()
        chart_type, dataset_type, num_nodes, show_communities = self.render_configuration()

        G = self.generate_network_data(dataset_type, num_nodes)

        chart_map = {
            "Force-Directed Graph": lambda: self.render_force_directed_graph(G, dataset_type, show_communities),
            "Chord Diagram": lambda: self.render_chord_diagram(G, dataset_type),
            "Arc Diagram": lambda: self.render_arc_diagram(G, dataset_type),
            "Sociogram": lambda: self.render_sociogram(G, dataset_type)
        }

        if chart_type in chart_map:
            chart_map[chart_type]()

        self.render_key_characteristics()
        self.render_examples(dataset_type, G)
        self.render_edge_preview(G)
