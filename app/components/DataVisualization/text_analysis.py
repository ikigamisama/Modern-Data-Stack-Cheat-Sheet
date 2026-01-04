import streamlit as st
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re


class TextAnalysis:

    def __init__(self):
        st.set_page_config(page_title="Text Analysis", layout="wide")
        self.title = "📝 Text Analysis Visualizations Dashboard"
        self.chart_types = [
            "Word Cloud",
            "Concordance Plot"
        ]
        self.scenarios = [
            "Customer Product Reviews",
            "Social Media Feedback",
            "Survey Open-Ended Responses",
            "News Article Comments",
            "Support Tickets"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Text analysis charts transform **textual data into visual insights**, revealing patterns in word usage, 
        document themes, and content structure.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **When to Use:**
            - Analyzing customer feedback and reviews
            - Exploring large text corpora
            - Identifying key themes in documents
            - Comparing text sources
            - Mining social media content
            """)

        with col2:
            st.markdown("""
            **Key Characteristics:**
            - Visualizes unstructured text data
            - Shows word frequency and importance
            - Reveals document themes
            - Enables pattern discovery
            """)

        st.markdown("""
        **Supported Visualizations:**
        - **Word Cloud** – Displays most frequent words with size proportional to occurrence
        - **Concordance Plot** – Shows keyword occurrences in context (keyword-in-context lines)
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Visualization Settings")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            chart_type = st.selectbox(
                "Select Chart Type", self.chart_types, key="chart_type")

        with col2:
            scenario = st.selectbox("Scenario", self.scenarios, key="scenario")

        with col3:
            if chart_type == "Concordance Plot":
                keyword = st.text_input(
                    "Keyword to Search", value="battery", key="keyword")
            else:
                keyword = None

        return chart_type, scenario, keyword

    def get_sample_text(self, scenario: str):
        texts = {
            "Customer Product Reviews": [
                "The battery life is amazing, lasts all day easily.",
                "Great camera quality but the phone gets hot quickly.",
                "Battery drains too fast when using GPS.",
                "Love the design and screen, very premium feel.",
                "Speaker sound is mediocre, expected better.",
                "Fast charging is a game changer, charges in 30 minutes.",
                "The battery degrades after a few months of use.",
                "Excellent performance, no lag at all.",
                "Camera is good in daylight but poor in low light.",
                "Battery life could be better, especially with 5G on."
            ] * 10,

            "Social Media Feedback": [
                "Just got the new update and battery is worse now 😠",
                "Anyone else having issues with overheating?",
                "The new camera features are incredible! 📸",
                "Battery life has improved so much after the patch.",
                "Why does my phone drain battery so fast on social apps?",
                "Love the new design, feels premium.",
                "Fast charging saves me every day.",
                "Sound quality on calls is crystal clear.",
                "The screen is stunning, best I've seen.",
                "Battery anxiety is real with this phone."
            ] * 12,

            "Survey Open-Ended Responses": [
                "I wish the battery lasted longer during video calls.",
                "The fast charging feature is very convenient.",
                "Phone performance is smooth and responsive.",
                "Camera takes beautiful photos, especially portraits.",
                "I often worry about battery life when traveling.",
                "The design is sleek and modern.",
                "Customer support was helpful with my battery issue.",
                "Screen brightness is excellent outdoors.",
                "Would love better low-light camera performance.",
                "Overall very satisfied with battery management."
            ] * 15,

            "News Article Comments": [
                "Battery technology needs to improve across all brands.",
                "This phone has the best camera in its price range.",
                "Overheating during gaming is a dealbreaker.",
                "Fast charging is now essential for me.",
                "Battery life is the most important feature.",
                "The software updates have optimized battery usage.",
                "Premium build quality justifies the price.",
                "Sound from speakers could be louder.",
                "Excellent display, vibrant colors.",
                "Concerned about long-term battery health."
            ] * 10,

            "Support Tickets": [
                "My battery is draining rapidly even when idle.",
                "Phone shuts down at 30% battery unexpectedly.",
                "Charging takes too long now, used to be faster.",
                "Device gets very hot while charging.",
                "Battery percentage jumps from 50% to 20% suddenly.",
                "After update, battery life significantly worse.",
                "Requesting battery replacement under warranty.",
                "Fast charger not working properly.",
                "Background apps killing battery quickly.",
                "Need help calibrating the battery."
            ] * 8
        }
        return " ".join(texts.get(scenario, texts["Customer Product Reviews"]))

    def create_word_cloud(self, text: str) -> plt.Figure:
        # Simple preprocessing
        text = re.sub(r'[^\w\s]', '', text.lower())
        stopwords = {'the', 'and', 'for', 'with', 'this',
                     'that', 'was', 'very', 'have', 'but', 'not', 'after'}
        words = [w for w in text.split() if w not in stopwords and len(w) > 3]

        word_freq = Counter(words)
        if not word_freq:
            word_freq = Counter(
                {"battery": 20, "camera": 15, "charging": 12, "phone": 10})

        wc = WordCloud(
            width=800, height=500,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
        return fig

    def create_concordance_plot(self, text: str, keyword: str) -> go.Figure:
        keyword = keyword.lower()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        matches = []

        for i, line in enumerate(lines):
            lower_line = line.lower()
            if keyword in lower_line:
                start = lower_line.find(keyword)
                context_before = line[max(0, start-40):start]
                context_after = line[start+len(keyword):start+len(keyword)+40]
                matches.append({
                    "line_num": i+1,
                    "before": context_before,
                    "keyword": keyword.capitalize(),
                    "after": context_after
                })

        if not matches:
            st.warning(
                f"No occurrences of '{keyword}' found in the sample text.")
            return go.Figure()

        # Sort by line number
        matches.sort(key=lambda x: x["line_num"])

        y_positions = list(range(len(matches)))
        texts = [
            f"{m['before']}<b>{m['keyword']}</b>{m['after']}" for m in matches]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[0] * len(matches),
            y=y_positions,
            mode='text',
            text=texts,
            textposition='middle right',
            hoverinfo='text',
            textfont=dict(size=12)
        ))

        fig.add_trace(go.Scatter(
            x=[0] * len(matches),
            y=y_positions,
            mode='markers',
            marker=dict(color='#e74c3c', size=10),
            hoverinfo='none'
        ))

        fig.update_layout(
            title=f"Concordance Plot – Occurrences of '{keyword.capitalize()}'<br><sub>Keyword in context from sample text</sub>",
            xaxis=dict(showgrid=False, zeroline=False,
                       showticklabels=False, range=[-1, 1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300 + len(matches)*30,
            showlegend=False,
            plot_bgcolor="white"
        )
        return fig

    def render_chart(self, chart_type: str, scenario: str, keyword: str):
        st.markdown(f"### {chart_type}: {scenario}")

        text = self.get_sample_text(scenario)

        if chart_type == "Word Cloud":
            fig = self.create_word_cloud(text)
            st.pyplot(fig)
            plt.close(fig)
            st.info(
                "**Insight**: Larger words appear more frequently in the text corpus.")

        else:  # Concordance Plot
            fig = self.create_concordance_plot(text, keyword or "battery")
            if fig.data:  # Only plot if there are matches
                st.plotly_chart(fig, width='stretch')
                st.info(
                    "**Insight**: Shows real contextual usage of the keyword across different responses.")

    def output(self):
        self.render_header()
        chart_type, scenario, keyword = self.render_configuration()
        self.render_chart(chart_type, scenario, keyword)
