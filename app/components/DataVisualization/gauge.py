import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Gauge:
    def __init__(self):
        st.set_page_config(page_title="Gauge Indicators", layout="wide")
        self.title = "📏 Gauge Indicator Visualizations Dashboard"
        self.chart_types = [
            "Gauge Chart",
            "Thermometer",
            "Progress Bar"
        ]
        self.scenarios = [
            "Monthly Sales Target",
            "Project Completion",
            "Server CPU Usage",
            "Daily Steps Goal",
            "Battery Level",
            "Customer Satisfaction Score"
        ]

    def render_header(self):
        st.markdown(f"## {self.title}")
        st.markdown("""
        ### Purpose
        Gauge indicators provide **at-a-glance status updates**, showing current values in relation to targets or capacity. 
        They're designed for quick interpretation of performance status.
        """)

    def render_configuration(self):
        st.markdown("### ⚙️ Indicator Settings")
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            chart_type = st.selectbox(
                "Select Indicator Type", self.chart_types, key="chart_type")

        with col2:
            scenario = st.selectbox("Scenario", self.scenarios, key="scenario")

        with col3:
            current_value = st.slider(
                "Current Value (%)", 0, 100, 65, key="value")

        return chart_type, scenario, current_value

    def create_gauge_chart(self, scenario: str, value: int) -> go.Figure:
        # Color logic based on value
        if value < 40:
            color = "#e74c3c"  # Red
        elif value < 75:
            color = "#f39c12"  # Orange
        else:
            color = "#27ae60"  # Green

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>{scenario}</b>"},
            delta={'reference': 80, 'increasing': {'color': "#27ae60"},
                   'decreasing': {'color': "#e74c3c"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffebee'},
                    {'range': [40, 70], 'color': '#fff3e0'},
                    {'range': [70, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig.update_layout(height=500, margin=dict(t=80, b=20, l=20, r=20))
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### 🛠️ Sample Code")
        st.code(f"""
import plotly.graph_objects as go

# Color logic based on value
if value < 40:
    color = "#e74c3c"  # Red
elif value < 75:
    color = "#f39c12"  # Orange
else:
    color = "#27ae60"  # Green

fig = go.Figure(
    go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={{"x": [0, 1], "y": [0, 1]}},
        title={{"text": "<b>{scenario}</b>"}},
        delta={{
            "reference": 80,
            "increasing": {{"color": "#27ae60"}},
            "decreasing": {{"color": "#e74c3c"}},
        }},
        gauge={{
            "axis": {{
                "range": [None, 100],
                "tickwidth": 1,
                "tickcolor": "darkblue",
            }},
            "bar": {{"color": color}},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {{"range": [0, 40], "color": "#ffebee"}},
                {{"range": [40, 70], "color": "#fff3e0"}},
                {{"range": [70, 100], "color": "#e8f5e9"}},
            ],
            "threshold": {{
                "line": {{"color": "red", "width": 4}},
                "thickness": 0.75,
                "value": 90,
            }},
        }},
    )
)

fig.update_layout(
    height=500,
    margin=dict(t=80, b=20, l=20, r=20),
)

fig.show()""", language="python")

    def create_thermometer(self, scenario: str, value: int) -> go.Figure:
        fill_color = "#e74c3c" if value > 80 else "#3498db"

        fig = go.Figure()

        # Thermometer bulb
        fig.add_trace(go.Scatter(
            x=[0, 1, 1, 0],
            y=[0, 0, 0.3, 0.3],
            fill='toself',
            fillcolor=fill_color,
            line=dict(width=0),
            mode='none',
            hoverinfo='none'
        ))

        # Tube
        fig.add_trace(go.Scatter(
            x=[0.3, 0.7, 0.7, 0.3],
            y=[0.3, 0.3, 3.5, 3.5],
            fill='toself',
            fillcolor='#ecf0f1',
            line=dict(color='gray', width=2),
            mode='lines',
            hoverinfo='none'
        ))

        # Fill level
        fill_height = 0.3 + (value / 100) * 3.2
        fig.add_trace(go.Scatter(
            x=[0.35, 0.65, 0.65, 0.35],
            y=[0.3, 0.3, fill_height, fill_height],
            fill='toself',
            fillcolor=fill_color,
            line=dict(width=0),
            mode='none',
            name=f"{value}%"
        ))

        # Labels
        for i in [0, 25, 50, 75, 100]:
            y_pos = 0.3 + (i / 100) * 3.2
            fig.add_annotation(
                x=0.9, y=y_pos, text=f"{i}%", showarrow=False, font=dict(size=12))
            fig.add_shape(type="line", x0=0.7, x1=0.8, y0=y_pos,
                          y1=y_pos, line=dict(color="gray", width=2))

        fig.update_layout(
            title=f"<b>{scenario}</b><br>{value}% Achieved",
            xaxis=dict(range=[-0.2, 1.2], showgrid=False,
                       zeroline=False, showticklabels=False),
            yaxis=dict(range=[0, 4], showgrid=False,
                       zeroline=False, showticklabels=False),
            height=600,
            showlegend=False,
            plot_bgcolor="white"
        )
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### 🛠️ Sample Code")
        st.code(f"""import plotly.graph_objects as go
                
fill_color = "#e74c3c" if value > 80 else "#3498db"

fig = go.Figure()

# Thermometer bulb
fig.add_trace(go.Scatter(
    x=[0, 1, 1, 0],
    y=[0, 0, 0.3, 0.3],
    fill='toself',
    fillcolor=fill_color,
    line=dict(width=0),
    mode='none',
    hoverinfo='none'
))

# Tube
fig.add_trace(go.Scatter(
    x=[0.3, 0.7, 0.7, 0.3],
    y=[0.3, 0.3, 3.5, 3.5],
    fill='toself',
    fillcolor='#ecf0f1',
    line=dict(color='gray', width=2),
    mode='lines',
    hoverinfo='none'
))

# Fill level
fill_height = 0.3 + (value / 100) * 3.2
fig.add_trace(go.Scatter(
    x=[0.35, 0.65, 0.65, 0.35],
    y=[0.3, 0.3, fill_height, fill_height],
    fill='toself',
    fillcolor=fill_color,
    line=dict(width=0),
    mode='none',
    name=f"{value}%"
))

# Labels
for i in [0, 25, 50, 75, 100]:
    y_pos = 0.3 + (i / 100) * 3.2
    fig.add_annotation(
        x=0.9, y=y_pos, text=f"{i}%", showarrow=False, font=dict(size=12))
    fig.add_shape(type="line", x0=0.7, x1=0.8, y0=y_pos,
                    y1=y_pos, line=dict(color="gray", width=2))

fig.update_layout(
    title=f"<b>{scenario}</b><br>{value}% Achieved",
    xaxis=dict(range=[-0.2, 1.2], showgrid=False,
                zeroline=False, showticklabels=False),
    yaxis=dict(range=[0, 4], showgrid=False,
                zeroline=False, showticklabels=False),
    height=600,
    showlegend=False,
    plot_bgcolor="white"
)
fig.show()""", language="python")

    def create_progress_bar(self, scenario: str, value: int) -> go.Figure:
        status = "On Track" if value >= 70 else "At Risk" if value >= 40 else "Critical"
        if value < 40:
            color = "#e74c3c"
        elif value < 70:
            color = "#f39c12"
        else:
            color = "#27ae60"

        fig = go.Figure(go.Indicator(
            mode="number+gauge",
            value=value,
            number={'suffix': "%"},
            title={
                'text': f"<b>{scenario}</b><br><span style='font-size:0.8em'>{status}</span>"},
            gauge={
                'shape': "bullet",
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'bgcolor': "#ecf0f1",
                'steps': [
                    {'range': [0, 40], 'color': '#ffebee'},
                    {'range': [40, 70], 'color': '#fff3e0'},
                    {'range': [70, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'value': 90
                }
            }
        ))

        fig.update_layout(height=300, margin=dict(t=100, b=20, l=50, r=50))
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### 🛠️ Sample Code")
        st.code(f"""import plotly.graph_objects as go

status = "On Track" if value >= 70 else "At Risk" if value >= 40 else "Critical"
if value < 40:
    color = "#e74c3c"
elif value < 70:
    color = "#f39c12"
else:
    color = "#27ae60"

fig = go.Figure(
    go.Indicator(
        mode="number+gauge",
        value=value,
        number={{"suffix": "%"}},
        title={{
            "text": "<b>{scenario}</b><br><span style='font-size:0.8em'>{status}</span>"
        }},
        gauge={{
            "shape": "bullet",
            "axis": {{"range": [None, 100]}},
            "bar": {{"color": color}},
            "bgcolor": "#ecf0f1",
            "steps": [
                {{"range": [0, 40], "color": "#ffebee"}},
                {{"range": [40, 70], "color": "#fff3e0"}},
                {{"range": [70, 100], "color": "#e8f5e9"}},
            ],
            "threshold": {{
                "line": {{"color": "red", "width": 4}},
                "value": 90,
            }},
        }},
    )
)

fig.update_layout(height=300, margin=dict(t=100, b=20, l=50, r=50))
fig.show()        
""", language="python")

    def render_chart(self, chart_type: str, scenario: str, value: int):
        st.markdown(f"### {chart_type}: {scenario}")

        if chart_type == "Gauge Chart":
            self.create_gauge_chart(scenario, value)
        elif chart_type == "Thermometer":
            self.create_thermometer(scenario, value)
        else:
            self.create_progress_bar(scenario, value)

    def render_examples(self):
        st.markdown("### 💡 Real-world Examples")

        examples = {
            "Sales Dashboards": "Current monthly sales vs. target",
            "Manufacturing": "Production output vs. capacity",
            "Project Management": "Percent completion of deliverables",
            "Fitness Tracking": "Daily steps toward goal",
            "System Monitoring": "CPU usage, memory, disk space"
        }

        for example, description in examples.items():
            with st.expander(f"📏 {example}"):
                st.write(description)

    def render_key_characteristics(self):
        st.markdown("### ⏱️ Understanding Gauge and Status Indicator Analysis")

        st.markdown("""
        Gauge analysis communicates **current status at a glance**.
        It answers a simple but critical question:
        *Is this metric within acceptable bounds right now?*
        """)

        st.markdown("#### ⚡ Immediate Status Interpretation")
        st.markdown("""
        Gauges are designed for instant comprehension.
        Users should understand status within seconds, without interpretation
        or detailed analysis.
        """)

        st.markdown("#### 🎯 Clear Target or Benchmark")
        st.markdown("""
        Every gauge references a target, threshold, or acceptable range.
        This context enables evaluation relative to expectations,
        not raw values.
        """)

        st.markdown("#### 🚦 Visual Alert Through Color or Position")
        st.markdown("""
        Color and position act as pre-attentive signals:
        - Green → acceptable  
        - Yellow → caution  
        - Red → action required  

        Urgency is communicated instantly.
        """)

        st.markdown("#### 🧠 Minimal Cognitive Load")
        st.markdown("""
        Gauges intentionally reduce complexity.
        They prioritize clarity over detail, making them ideal for dashboards,
        monitoring, and executive reporting.
        """)

        st.divider()

        st.markdown("#### 🎯 Why Gauge Analysis Matters")
        st.markdown("""
        Gauge analysis enables fast, confident decisions.
        It supports:
        - Operational monitoring  
        - Alerting and escalation  
        - Executive and real-time dashboards  
        """)

    def output(self):
        self.render_header()
        chart_type, scenario, current_value = self.render_configuration()
        self.render_chart(chart_type, scenario, current_value)
        self.render_examples()
        self.render_key_characteristics()
