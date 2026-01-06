import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Density:
    def __init__(self):
        self.title = "🎯 Density Visualization Dashboard"
        self.chart_types = ["Hexbin Plot", "Contour Plot", "Dot Density Map"]
        self.dataset_types = [
            "Crime Analysis", "Customer Distribution", "Scientific Research",
            "Real Estate", "Weather Patterns", "Custom Data"
        ]
        self.sample_sizes = (100, 10000, 2000)

    def render_header(self):
        st.markdown(f"### {self.title}")
        st.markdown("""
        ### Purpose
        Density charts show the concentration and distribution of data points across a space, 
        revealing where values cluster or disperse. They're essential for understanding spatial 
        patterns and point concentrations.
        """)

    def render_configuration(self):
        col1, col2, col3 = st.columns(3)

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            dataset_type = st.selectbox(
                "Dataset Type", self.dataset_types, key="dataset_type")

        with col3:
            sample_size = st.slider(
                "Sample Size", *self.sample_sizes, key="sample_size")

        return chart_type, dataset_type, sample_size

    @staticmethod
    def generate_density_data(data_type: str, n_points: int) -> pd.DataFrame:
        np.random.seed(42)

        if data_type == "Crime Analysis":
            x1 = np.random.normal(0.3, 0.1, n_points // 3)
            y1 = np.random.normal(0.3, 0.1, n_points // 3)
            x2 = np.random.normal(0.7, 0.15, n_points // 3)
            y2 = np.random.normal(0.7, 0.15, n_points // 3)
            x3 = np.random.uniform(0, 1, n_points // 3)
            y3 = np.random.uniform(0, 1, n_points // 3)

            x = np.concatenate([x1, x2, x3])
            y = np.concatenate([y1, y2, y3])
            intensity = np.concatenate([
                np.random.normal(8, 1, n_points // 3),
                np.random.normal(6, 1, n_points // 3),
                np.random.normal(2, 1, n_points // 3)
            ])

        elif data_type == "Customer Distribution":
            centers = [(0.2, 0.2), (0.8, 0.8), (0.5, 0.5)]
            x, y = [], []
            for cx, cy in centers:
                n = n_points // len(centers)
                x.extend(np.random.normal(cx, 0.1, n))
                y.extend(np.random.normal(cy, 0.1, n))
            x.extend(np.random.uniform(0, 1, n_points // 4))
            y.extend(np.random.uniform(0, 1, n_points // 4))
            intensity = np.random.gamma(2, 1, len(x))
            x, y, intensity = x[:n_points], y[:n_points], intensity[:n_points]

        elif data_type == "Scientific Research":
            x = np.random.uniform(0, 1, n_points)
            y = np.random.uniform(0, 1, n_points)
            intensity = (np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / 0.05) +
                         np.exp(-((x - 0.7)**2 + (y - 0.7)**2) / 0.08))

        elif data_type == "Real Estate":
            x = np.random.uniform(0, 1, n_points)
            y = np.random.uniform(0, 1, n_points)
            intensity = 100 + 50 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

        elif data_type == "Weather Patterns":
            x = np.random.uniform(0, 1, n_points)
            y = np.random.uniform(0, 1, n_points)
            intensity = (np.exp(-((x - 0.2)**2 + (y - 0.8)**2) / 0.02) +
                         np.exp(-((x - 0.8)**2 + (y - 0.2)**2) / 0.03) +
                         np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.04))

        else:  # Custom Data
            x = np.random.uniform(0, 1, n_points)
            y = np.random.uniform(0, 1, n_points)
            intensity = np.random.gamma(2, 1, n_points)

        return pd.DataFrame({'x': x, 'y': y, 'intensity': intensity})

    def render_hexbin_plot(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 📊 Hexbin Plot - Bivariate Density")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        hexbin = ax1.hexbin(df['x'], df['y'], gridsize=30,
                            cmap='Blues', alpha=0.8)
        ax1.set_title(f'{data_type} - Hexbin Density Plot')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        plt.colorbar(hexbin, ax=ax1, label='Point Density')

        ax2.scatter(df['x'], df['y'], alpha=0.3, s=1, color='blue')
        ax2.set_title('Underlying Data Points')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** Handling large datasets, reducing overplotting, identifying clusters.
        
        **Key Features:** Efficient with thousands of points, clear density gradients.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
x = np.random.normal(0.5, 0.2, 2000)
y = np.random.normal(0.5, 0.2, 2000)
df = pd.DataFrame({'x': x, 'y': y})

fig, ax = plt.subplots(figsize=(10, 8))
hb = ax.hexbin(df['x'], df['y'], gridsize=30, cmap='Blues')
ax.set_title('Hexbin Density Plot')
plt.colorbar(hb, label='Point Density')
plt.show()
        ''', language='python')

    def render_contour_plot(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🌊 Contour Plot - Density Contours")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.kdeplot(x=df['x'], y=df['y'], fill=True, cmap='viridis', ax=ax1)
        ax1.set_title(f'{data_type} - Filled Contour Density')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        sns.kdeplot(x=df['x'], y=df['y'], levels=10,
                    color='black', linewidths=0.5, ax=ax2)
        ax2.set_title('Contour Lines Only')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** Scientific analysis, showing smooth gradients, probability density.
        
        **Key Features:** Professional appearance, clear level sets.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
x = np.random.normal(0.5, 0.2, 2000)
y = np.random.normal(0.5, 0.2, 2000)
df = pd.DataFrame({'x': x, 'y': y})

plt.figure(figsize=(10, 8))
sns.kdeplot(x=df['x'], y=df['y'], fill=True, cmap='viridis')
plt.title('Contour Density Plot')
plt.show()
        ''', language='python')

    def render_dot_density_map(self, df: pd.DataFrame, data_type: str):
        st.markdown("### 🔴 Dot Density Map - Point Concentration")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(df['x'], df['y'], alpha=0.3, s=10, color='red')
        ax1.set_title(f'{data_type} - Basic Dot Density')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')

        scatter = ax2.scatter(
            df['x'], df['y'], c=df['intensity'], alpha=0.5, s=20, cmap='hot')
        ax2.set_title('Intensity-based Dot Density')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        plt.colorbar(scatter, ax=ax2, label='Intensity')

        st.pyplot(fig)
        plt.close(fig)

        st.markdown("""
        **When to use:** Geographic analysis, hotspot identification, individual point visibility.
        
        **Key Features:** Intuitive, supports color/size encoding for intensity.
        """)

        st.markdown("#### 🛠️ Dataset")
        st.dataframe(df)
        st.markdown("#### 🛠️ Sample Code")

        st.code('''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
x = np.random.normal(0.5, 0.2, 2000)
y = np.random.normal(0.5, 0.2, 2000)
intensity = np.random.gamma(2, 1, 2000)
df = pd.DataFrame({'x': x, 'y': y, 'intensity': intensity})

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['x'], df['y'], c=df['intensity'], alpha=0.5, s=20, cmap='hot')
plt.colorbar(scatter, label='Intensity')
plt.title('Dot Density Map with Intensity')
plt.show()
        ''', language='python')

    def render_key_characteristics(self):
        st.markdown("### 🎯 Understanding Density and Concentration in Data")
        st.markdown("""
As datasets grow in size, individual data points become less informative.
Density analysis focuses on **where data accumulates**, helping analysts
identify meaningful patterns within large volumes of information.
""")

        st.markdown("#### 📍 Revealing Concentration Patterns")
        st.markdown("""
        Density visualizations highlight regions where data points cluster together.
        These clusters often represent dominant behaviors, frequent events, or
        high-demand areas.

        Instead of examining each point individually, analysts can quickly identify
        hotspots and areas of interest.
        """)

        st.markdown("#### 📊 Handling Large Datasets Effectively")
        st.markdown("""
        Large datasets often overwhelm traditional scatter plots.
        Density-based techniques aggregate data into regions, making them easier
        to interpret at scale.

        Common approaches include:
        - Heatmaps  
        - Hexbin plots  
        - Kernel density estimation (KDE)  
        """)

        st.markdown("#### 🌈 Showing Gradients of Intensity")
        st.markdown("""
        Rather than binary classifications, density analysis displays **continuous
        gradients of intensity**. These gradients reveal how concentration changes
        smoothly across space or value ranges.

        This helps analysts understand transitions, boundaries, and probabilistic behavior.
        """)

        st.markdown("#### 🧹 Reducing Visual Clutter from Overlapping Points")
        st.markdown("""
        Overlapping data points can hide important patterns.
        Density techniques replace overlapping markers with color intensity or contours,
        resulting in cleaner and more interpretable visuals—without losing information.
        """)

        st.divider()

        st.markdown("#### 🎯 Why Density Analysis Matters")
        st.markdown("""
        Density-focused views improve pattern recognition, anomaly detection,
        and decision-making across domains such as:
        - Geospatial analytics  
        - User behavior analysis  
        - System performance monitoring  
        - Market segmentation  
""")

    def render_examples(self, chart_type: str, dataset_type: str):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Crime Analysis": "Show concentration of incidents across city neighborhoods",
            "Customer Distribution": "Map where customers are geographically clustered",
            "Scientific Research": "Display protein concentrations in cellular studies",
            "Real Estate": "Visualize property density and pricing patterns",
            "Weather Patterns": "Show precipitation intensity across regions"
        }

        for example, description in examples.items():
            with st.expander(f"📊 {example}"):
                st.write(description)
                best = chart_type if dataset_type == example else "All density charts suitable"
                st.write(f"**Best chart type:** {best}")

    def output(self):
        self.render_header()
        chart_type, dataset_type, sample_size = self.render_configuration()

        df = self.generate_density_data(dataset_type, sample_size)

        chart_map = {
            "Hexbin Plot": self.render_hexbin_plot,
            "Contour Plot": self.render_contour_plot,
            "Dot Density Map": self.render_dot_density_map
        }

        if chart_type in chart_map:
            chart_map[chart_type](df, dataset_type)

        self.render_examples(chart_type, dataset_type)
        self.render_key_characteristics()
