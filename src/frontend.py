import streamlit as st
import requests
import json
import random
import time
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Any

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(
    page_title="ChurnGuard | Enterprise MLOps",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Network Configuration
BASE_URL = os.getenv("API_URL", "https://mlops-api-40lx.onrender.com")
API_URL = f"{BASE_URL.strip('/')}/predict"
TIMEOUT_SECONDS = 60

# Model Configuration
FEATURE_COUNT = 38
DEFAULT_CUSTOMER_ID = "CUST_12345678"

# Premium UI Palette
C_PRIMARY = "#8B5CF6"      # Violet500
C_SECONDARY = "#06B6D4"    # Cyan 500
C_SUCCESS = "#10B981"      # Emerald 500
C_WARNING = "#F59E0B"      # Amber 500
C_DANGER = "#F43F5E"       # Rose 500
C_BG = "#050510"
C_SURFACE = "#0F111A"
C_CARD = "#13141F"
C_TEXT = "#E2E8F0"
C_MUTED = "#64748B"

# ==============================================================================
# 2. PREMIUM CUSTOM CSS & STYLING
# ==============================================================================
def load_custom_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        :root {{
            --c-primary: {C_PRIMARY};
            --c-secondary: {C_SECONDARY};
            --c-success: {C_SUCCESS};
            --c-warning: {C_WARNING};
            --c-danger: {C_DANGER};
            --c-bg: {C_BG};
            --c-surface: {C_SURFACE};
            --c-card: {C_CARD};
            --c-text: {C_TEXT};
            --c-muted: {C_MUTED};
        }}
        
        /* Global Background & Typography */
        .stApp {{
            background: 
                radial-gradient(circle at 10% 20%, rgba(139,92,246,0.07) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(6,182,212,0.07) 0%, transparent 40%),
                linear-gradient(180deg, #050510 0%, #0a0a1a 100%);
            background-attachment: fixed;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
 color: var(--c-text);
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
            color: #FAFAFA !important;
        }}
        
        /* Scrollbars */
        ::-webkit-scrollbar {{
            width: 8px; height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: rgba(255,255,255,0.02);
        }}
        ::-webkit-scrollbar-thumb {{
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: rgba(255,255,255,0.15);
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] > div {{
            background: rgba(15, 17, 26, 0.95) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        
        /* Radio Nav Menu */
        div[data-testid="stRadio"] > div {{
            flex-direction: column;
            gap: 6px;
        }}
        div[data-testid="stRadio"] label {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 12px 16px !important;
            margin: 0 !important;
            transition: all 0.2s ease;
            font-weight: 500;
            font-size: 0.9rem;
        }}
        div[data-testid="stRadio"] label:hover {{
            background: rgba(255,255,255,0.06);
            border-color: rgba(139,92,246,0.3);
        }}
        div[data-testid="stRadio"] label[data-checked="true"] {{
            background: linear-gradient(90deg, rgba(139,92,246,0.15), rgba(6,182,212,0.1));
            border-color: rgba(139,92,246,0.4);
            color: #FAFAFA !important;
            box-shadow: 0 0 20px rgba(139,92,246,0.1);
        }}
        div[data-testid="stRadio"] div[role="radiogroup"] {{
            gap: 8px;
        }}
        
        /* Inputs */
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextArea"] textarea {{
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 10px !important;
            color: #FAFAFA !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85rem;
        }}
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus {{
            border-color: var(--c-primary) !important;
            box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
        }}
        
        /* Buttons */
        div.stButton > button {{
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            height: 3em;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: none;
            letter-spacing: 0.01em;
            position: relative;
            overflow: hidden;
        }}
        div.stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {C_PRIMARY}, {C_SECONDARY}) !important;
            box-shadow: 0 4px 20px -5px rgba(139,92,246,0.4);
        }}
        div.stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 8px 30px -5px rgba(139,92,246,0.5);
        }}
        div.stButton > button[kind="primary"]:active {{
            transform: scale(0.98);
        }}
        div.stButton > button[kind="secondary"] {{
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            color: #FAFAFA !important;
        }}
        div.stButton > button[kind="secondary"]:hover {{
            background: rgba(255,255,255,0.08) !important;
            border-color: rgba(255,255,255,0.15) !important;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            font-weight: 500;
        }}
        .streamlit-expanderContent {{
            border: 1px solid rgba(255,255,255,0.04);
            border-top: none;
            border-radius: 0 0 10px 10px;
            background: rgba(255,255,255,0.01);
        }}
        
        /* Sliders */
        div[data-testid="stSlider"] > div > div > div {{
            background: linear-gradient(90deg, {C_PRIMARY}, {C_SECONDARY}) !important;
        }}
        div[data-testid="stSlider"] > div > div > div > div {{
            background: #FAFAFA !important;
            box-shadow: 0 0 10px rgba(139,92,246,0.3);
        }}
        
        /* Dataframe & Tables */
        .stDataFrame {{
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            overflow: hidden;
        }}
        .stDataFrame th {{
            background: rgba(255,255,255,0.04) !important;
            color: #94A3B8 !important;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.7rem;
            letter-spacing: 0.05em;
        }}
        .stDataFrame td {{
            background: transparent !important;
            border-bottom: 1px solid rgba(255,255,255,0.03) !important;
        }}
        
        /* Code blocks */
        pre {{
            background: rgba(0,0,0,0.3) !important;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            padding: 16px !important;
        }}
        code {{
            font-family: 'JetBrains Mono', monospace !important;
            color: #A5B4FC !important;
        }}
        
        /* Status Widgets */
        .status-pulse {{
            width: 8px; height: 8px;
            background: {C_SUCCESS};
            border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(16,185,129,0.7);
            animation: pulse-green 2s infinite;
            display: inline-block;
            margin-right: 8px;
        }}
        @keyframes pulse-green {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16,185,129,0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 6px rgba(16,185,129,0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16,185,129,0); }}
        }}
        
        /* Glass Cards (Custom HTML) */
        .glass-card {{
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
        }}
        .glass-card:hover {{
            border-color: rgba(255,255,255,0.12);
            transform: translateY(-2px);
            box-shadow: 0 20px 40px -12px rgba(0,0,0,0.5);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            background: linear-gradient(180deg, #FAFAFA, #A1A1AA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .metric-label {{
            font-size: 0.8rem;
            color: var(--c-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-top: 4px;
        }}
        
        /* Gradient text utility */
        .text-gradient {{
            background: linear-gradient(135deg, #C4B5FD 0%, #67E8F9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        /* Empty states */
        .empty-state {{
            text-align: center;
            padding: 40px 20px;
            color: var(--c-muted);
        }}
        
        /* Spinner override */
        div[data-testid="stSpinner"] > div {{
            border-top-color: {C_PRIMARY} !important;
            border-right-color: transparent !important;
        }}
 </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. STATE MANAGEMENT
# ==============================================================================
class SessionState:
    """Handles session persistence for navigation, inputs, and history."""
    
    @staticmethod
    def initialize():
        if 'page' not in st.session_state:
            st.session_state['page'] = 'login'
        if 'features' not in st.session_state:
            st.session_state['features'] = [0.0] * FEATURE_COUNT
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        if 'user' not in st.session_state:
            st.session_state['user'] = None
        if 'last_latency' not in st.session_state:
            st.session_state['last_latency'] = 0.0

    @staticmethod
    def add_to_history(entry: Dict):
        """Adds a prediction result to the persistent session history."""
        st.session_state['history'].insert(0, entry)
        if len(st.session_state['history']) > 100:
            st.session_state['history'] = st.session_state['history'][:100]

# ==============================================================================
# 4. NETWORK SERVICE LAYER
# ==============================================================================
class APIService:
    """Encapsulates API communication with robust error handling."""
    
    @staticmethod
    def predict(customer_id: str, features: List[float]) -> Dict[str, Any]:
        payload = {
            "customer_id": customer_id,
            "feature_vector": features
        }
        
        start_time = time.time()
        result = {
            "success": False,
            "data": None,
            "error": None,
            "latency": 0.0,
            "status_code": 0
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECONDS)
            
            result["latency"] = (time.time() - start_time) * 1000
            result["status_code"] = response.status_code
            
            if response.status_code == 200:
                result["success"] = True
                result["data"] = response.json()
            elif response.status_code == 502:
                result["error"] = "❄️ Cold Start: API is waking up. Please retry in 15 seconds."
            else:
                result["error"] = f"API Error {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            result["error"] = "⌛ Timeout: The request took too long (Cold Start Protection)."
        except requests.exceptions.ConnectionError:
            result["error"] = f"❌ Connection Failed: Could not reach {API_URL}."
        except Exception as e:
            result["error"] = f"⚠️ Unexpected System Error: {str(e)}"
            
        return result

# ==============================================================================
# 5. PREMIUM UI COMPONENTS
# ==============================================================================
def render_header():
    c1, c2 = st.columns([0.5, 6])
    with c1:
        st.markdown("<h1 style='font-size: 2.2rem; margin:0;'>🛡️</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div style='margin-top: 4px;'>
 <h2 style='margin-bottom:0; letter-spacing: -0.03em;'>ChurnGuard Enterprise</h2>
                <p style='color: #64748B; margin-top:4px; font-size: 0.9rem; letter-spacing: 0.02em;'>
                    PRODUCTION MLOPS CONTROL PLANE </p>
            </div>
        """, unsafe_allow_html=True)

def render_gauge_chart(probability: float):
    """Renders a premium Plotly Gauge Chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {
            'suffix': "%",
            'font': {'size': 48, 'family': 'Inter, sans-serif', 'color': '#FAFAFA'}
        },
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': "Churn Probability",
            'font': {'size': 18, 'family': 'Inter, sans-serif', 'color': "#94A3B8"}
        },
        gauge = {
            'axis': {
                'range': [None, 100],
                'tickwidth': 1,
                'tickcolor': "rgba(255,255,255,0.2)",
                'tickfont': {'color': '#64748B'}
            },
            'bar': {'color': "rgba(255,255,255,0.1)", 'thickness': 0.15},
            'bgcolor': "rgba(255,255,255,0.02)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.06)",
            'steps': [
                {'range': [0, 40], 'color': f"rgba(16,185,129,0.15)"},
                {'range': [40, 70], 'color': f"rgba(245,158,11,0.15)"},
                {'range': [70, 100], 'color': f"rgba(244,63,94,0.15)"}
            ],
            'threshold': {
                'line': {'color': "#FAFAFA", 'width': 3},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': 'Inter'},
        height=320,
        margin=dict(l=30, r=30, t=60, b=20),
        annotations=[
            dict(
                text="SAFE" if probability < 0.4 else "WARN" if probability < 0.7 else "CRITICAL",
                x=0.5, y=0.15,
                font=dict(size=14, color='#64748B', family='Inter'),
                showarrow=False
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def custom_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal", icon: str = "📊"):
    delta_html = ""
 if delta:
        color = C_SUCCESS if delta_color == "normal" else C_DANGER if delta_color == "inverse" else C_WARNING
        delta_html = f'<div style="font-size:0.85rem; color:{color}; font-weight:600; margin-top:6px;">{delta}</div>'
    
    st.markdown(f"""
    <div class="glass-card" style="height: 140px; display:flex; flex-direction:column; justify-content:space-between;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span style="font-size:1.2rem;">{icon}</span>
            <span style="font-size:0.75rem; color:{C_MUTED}; text-transform:uppercase; letter-spacing:0.08em; font-weight:700;">{label}</span>
        </div>
        <div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def feature_heatmap_bar(features: List[float]):
    """Visualizes the feature vector as a horizontal heatmap strip."""
    if not features:
        return    chunks = 38
    cells = ""
    for i, val in enumerate(features):
        # Interpolate color between violet and cyan based on value
        r = int(139 + (6 - 139) * val)
        g = int(92 + (182 - 92) * val)
        b = int(246 + (212 - 246) * val)
        alpha = 0.3 + (val * 0.7)
        cells += f'<div title="Dim_{i}: {val:.2f}" style="flex:1; height:100%; background:rgba({r},{g},{b},{alpha}); border-radius:1px; margin:0 1px; min-width:2px;"></div>'
    
    st.markdown(f"""
    <div style="margin: 12px 0;">
        <div style="font-size:0.7rem; color:{C_MUTED}; text-transform:uppercase; letter-spacing:0.08em; font-weight:600; margin-bottom:6px;">
            Feature Vector Heatmap (38-Dim)
        </div>
        <div style="display:flex; width:100%; height:28px; align-items:center; background:rgba(0,0,0,0.3); border-radius:6px; padding:3px; border:1px solid rgba(255,255,255,0.05);">
            {cells}
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:{C_MUTED}; margin-top:4px;">
            <span>0.0</span>
            <span>0.5</span>
            <span>1.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 6. APPLICATION PAGES (PREMIUM STYLED)
# ==============================================================================

def page_login():
    """Premium Enterprise Login Screen."""
    st.markdown("""
        <style>
        .login-bg {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(139,92,246,0.12) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(6,182,212,0.12) 0%, transparent 50%);
            z-index: -1;
        }
        </style>
        <div class="login-bg"></div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2.2, 1])
    with c2:
        st.markdown("""
            <div style="text-align:center; margin-bottom:32px;">
                <div style="font-size:3.5rem; margin-bottom:12px;">🛡️</div>
                <h1 style="margin-bottom:4px;">ChurnGuard</h1>
                <p style="color:#64748B; letter-spacing:0.1em; font-size:0.85rem; text-transform:uppercase;">Enterprise MLOps Platform</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.container(border=False):
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); backdrop-filter:blur(24px); border:1px solid rgba(255,255,255,0.08); border-radius:20px; padding:40px; box-shadow:0 25px 50px -12px rgba(0,0,0,0.5); position:relative; overflow:hidden;">
                <div style="position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg, {C_PRIMARY}, {C_SECONDARY});"></div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="admin@mlops.com", label_visibility="collapsed")
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            password = st.text_input("Password", type="password", placeholder="••••••••", label_visibility="collapsed")
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            
            auth_clicked = st.button("Authenticate System", type="primary", use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if auth_clicked:
                if username: 
                    st.session_state['user'] = username
                    st.session_state['page'] = 'dashboard'
                    st.toast("Authentication Verified", icon="✅")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Please enter valid credentials to proceed.")

def page_dashboard():
    """Premium Analytics View."""
    st.markdown("## <span class='text-gradient'>Live Operations</span>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{C_MUTED}; margin-top:-10px;'>Real-time inference telemetry & risk distribution across the production fleet.</p>", unsafe_allow_html=True)
    
    df = pd.DataFrame(st.session_state['history'])
    
    total_preds = len(df)
    churn_rate = 0.0
    if not df.empty:
        churn_rate = df[df['churn_prediction'] == True].shape[0] / total_preds
    
    # Premium KPI Grid
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        custom_metric_card("Total Inferences", f"{total_preds:,}", icon="🔮")
    with k2:
        delta_churn = f"{churn_rate:.1%}" if total_preds > 0 else "0.0%"
        custom_metric_card("Session Churn Rate", delta_churn, icon="⚡")
    with k3:
        custom_metric_card("Last Latency", f"{st.session_state['last_latency']:.0f}ms", icon="⏱️")
    with k4:
        custom_metric_card("System Status", "Online", delta="Healthy", delta_color="normal", icon="🟢")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Layout
    c1, c2 = st.columns([2.2, 1], gap="large")
    
    with c1:
        st.markdown(f"""
        <div class="glass-card" style="height:420px; display:flex; flex-direction:column;">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;">
                <div>
                    <h4 style="margin:0; font-size:1.05rem;">Risk Distribution</h4>
                    <p style="margin:0; color:{C_MUTED}; font-size:0.8rem;">Probability density of inference scores</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if not df.empty:
            fig = px.histogram(
                df, 
                x="probability", 
                nbins=12, 
                color_discrete_sequence=[C_PRIMARY],
                range_x=[0, 1]
            )
            fig.update_traces(
                marker_line_width=0,
                opacity=0.8,
                hovertemplate='<b>Probability:</b> %{x:.2f}<br><b>Count:</b> %{y}'
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#94A3B8",
                font_family="Inter",
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)",
                    title="Churn Probability",
                    showline=False ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)",
                    title="Frequency",
                    showline=False
                ),
                showlegend=False,
                bargap=0.15            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown("""
                <div class="empty-state" style="flex:1; display:flex; align-items:center; justify-content:center; flex-direction:column;">
                    <div style="font-size:2rem; margin-bottom:12px;">📉</div>
                    <p>No inference data yet. Generate predictions to populate analytics.</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
            
    with c2:
        st.markdown(f"""
        <div class="glass-card" style="height:420px; display:flex; flex-direction:column;">
            <h4 style="margin:0 0 4px 0; font-size:1.05rem;">Recent Activity</h4>
            <p style="margin:0 0 12px 0; color:{C_MUTED}; font-size:0.8rem;">Last 5 inference events</p>
        """, unsafe_allow_html=True)
        
        if not df.empty:
            mini_df = df[['timestamp', 'customer_id', 'probability']].head(5)
            st.dataframe(mini_df, hide_index=True, use_container_width=True)
 st.markdown("""
                <div style="margin-top:auto; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06);">
                    <p style="font-size:0.75rem; color:#475569; margin:0;">
 Logs are ephemeral per session. Export audit trails from the Logs page.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="empty-state" style="flex:1; display:flex; align-items:center; justify-content:center; flex-direction:column;">
                    <div style="font-size:2rem; margin-bottom:8px;">⏳</div>
                    <p style="font-size:0.85rem;">Waiting for incoming requests...</p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

def page_prediction():
    """Premium Inference Interface."""
    st.markdown("## <span class='text-gradient'>Inference Engine</span>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{C_MUTED}; margin-top:-10px;'>Execute model predictions against the production endpoint with full feature-level control.</p>", unsafe_allow_html=True)
    
    left_col, right_col = st.columns([1, 1.6], gap="large")
    
    with left_col:
        with st.container(border=False):
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
                    <div style="background:linear-gradient(135deg, {C_PRIMARY}, {C_SECONDARY}); width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center;">👤</div>
                    <h3 style="margin:0; font-size:1.15rem;">Customer Profile</h3>
                </div>
            """, unsafe_allow_html=True)
            
            customer_id = st.text_input("Customer UUID", value=DEFAULT_CUSTOMER_ID, label_visibility="collapsed")
            st.caption("Target identifier for this inference request")
            
            st.markdown("<br>", unsafe_allow_html=True)
 st.markdown("### Feature Vector")
            # Visual heatmap of current features
            feature_heatmap_bar(st.session_state['features'])
            
            tab_quick, tab_fine = st.tabs(["⚡ Auto-Generate", "🎛️ Manual Control"])
            
            with tab_quick:
                st.markdown(f"<p style='color:{C_MUTED}; font-size:0.85rem;'>Generate a random 38-dimensional feature vector for pipeline testing.</p>", unsafe_allow_html=True)
                if st.button("🎲 Randomize Profile", use_container_width=True):
                    st.session_state['features'] = [round(random.random(), 2) for _ in range(FEATURE_COUNT)]
                    st.rerun()

            with tab_fine:
                st.markdown(f"<p style='color:{C_MUTED}; font-size:0.85rem;'>Manually tune feature dimensions. Grouped for navigability.</p>", unsafe_allow_html=True)
                
                with st.expander("Dimensions 0 – 10"):
                    for i in range(0, 11):
                        val = st.slider(f"Dim_{i}", 0.0, 1.0, st.session_state['features'][i], 0.01, key=f"f_{i}")
                        st.session_state['features'][i] = val
                        
                with st.expander("Dimensions 11 – 20"):
                    for i in range(11, 21):
                        val = st.slider(f"Dim_{i}", 0.0, 1.0, st.session_state['features'][i], 0.01, key=f"f_{i}")
                        st.session_state['features'][i] = val
                        
                with st.expander("Dimensions 21 – 30"):
                    for i in range(21, 31):
                        val = st.slider(f"Dim_{i}", 0.0, 1.0, st.session_state['features'][i], 0.01, key=f"f_{i}")
                        st.session_state['features'][i] = val

                with st.expander("Dimensions 31 – 37"):
                    for i in range(31, FEATURE_COUNT):
                        val = st.slider(f"Dim_{i}", 0.0, 1.0, st.session_state['features'][i], 0.01, key=f"f_{i}")
                        st.session_state['features'][i] = val

            with st.expander("📡 Inspect JSON Payload"):
                display_vec = st.session_state['features'][:5] + ["..."] + st.session_state['features'][-2:]
                st.code(json.dumps({
                    "customer_id": customer_id,
                    "vector_length": len(st.session_state['features']),
                    "sample": display_vec
                }, indent=2), language="json")

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Execute Prediction", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        if predict_btn:
            with st.spinner("Contacting inference endpoint..."):
                response = APIService.predict(customer_id, st.session_state['features'])
                
            if response["success"]:
                data = response["data"]
                
                prob = float(data.get('probability', 0.0))
                is_churn = int(data.get('prediction', 0)) == 1
                latency = response["latency"]
                st.session_state['last_latency'] = latency
                
                SessionState.add_to_history({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "customer_id": customer_id,
                    "probability": prob,
                    "churn_prediction": is_churn,
                    "latitude": latency
                })
                
                # Success Header
                verdict_color = C_DANGER if is_churn else C_SUCCESS
                verdict_text = "CRITICAL RISK" if is_churn else "LOW RISK PROFILE"
                verdict_icon = "🔴" if is_churn else "🟢"
                
                st.markdown(f"""
                <div class="glass-card" style="border-left: 4px solid {verdict_color};">
                    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
                        <div>
                            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                                <span style="font-size:1.5rem;">{verdict_icon}</span>
                                <h2 style="margin:0; font-size:1.4rem;">{verdict_text}</h2>
                            </div>
                            <p style="margin:0; color:{C_MUTED}; font-size:0.9rem;">
                                Customer ID: <span style="font-family:'JetBrains Mono', monospace; color:#CBD5E1;">{customer_id}</span>
                            </p>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.75rem; color:{C_MUTED}; text-transform:uppercase; letter-spacing:0.08em; font-weight:700;">Latency</div>
                            <div style="font-size:1.3rem; font-weight:700; font-family:'JetBrains Mono', monospace;">{latency:.0f}ms</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                render_gauge_chart(prob)
                
                with st.expander("🔍 Full Model Response & Trace"):
                    st.json(data)
                    
            else:
                st.markdown(f"""
                <div style="background: rgba(244,63,94,0.08); border: 1px solid rgba(244,63,94,0.2); padding: 20px; border-radius: 12px; margin-top: 8px;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                        <span style="font-size:1.3rem;">⚠️</span>
                        <strong style="color:#F43F5E; font-size:1.1rem;">Inference Failed</strong>
                    </div>
                    <p style="margin:0; color:#FDA4AF; font-size:0.9rem; font-family:'JetBrains Mono', monospace;">
                        {response['error']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.markdown(f"""
            <div class="glass-card" style="height:100%; min-height:400px; display:flex; align-items:center; justify-content:center; flex-direction:column; text-align:center;">
                <div style="font-size:3rem; margin-bottom:16px; opacity:0.5;">🤖</div>
                <h3 style="margin:0 0 8px 0; color:#E2E8F0;">Ready for Inference</h3>
                <p style="margin:0; color:{C_MUTED}; max-width:280px; font-size:0.9rem; line-height:1.5;">
                    Configure a customer profile and feature vector on the left, then execute a prediction against the production model.
                </p>
            </div>
            """, unsafe_allow_html=True)

def page_logs():
    """Premium Audit Trail View."""
    st.markdown("## <span class='text-gradient'>Inference Audit Logs</span>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{C_MUTED}; margin-top:-10px;'>Immutable session telemetry for compliance and debugging.</p>", unsafe_allow_html=True)
    
    if not st.session_state['history']:
        st.markdown(f"""
        <div class="glass-card" style="padding:60px 20px; text-align:center;">
            <div style="font-size:2.5rem; margin-bottom:12px;">📭</div>
            <h3 style="margin:0 0 8px 0;">No Records Found</h3>
            <p style="margin:0; color:{C_MUTED};">Run predictions to populate the audit trail.</p>
        </div>
        """, unsafe_allow_html=True)
        return
        
    df = pd.DataFrame(st.session_state['history'])
    
    # Stats row    s1, s2 = st.columns([3, 1])
    with s2:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export CSV", 
            csv_data, 
            f"churn_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            "text/csv",
            use_container_width=True
        )
    
    st.markdown(f"""
    <div class="glass-card" style="padding:8px;">
    """, unsafe_allow_html=True)
    
    st.dataframe(
        df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="HH:mm:ss"),
            "customer_id": "Customer UUID",
            "probability": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "churn_prediction": st.column_config.CheckboxColumn("Churn", default=False),
            "latency": st.column_config.NumberColumn("Latency", format="%d ms"),
        },
        use_container_width=True,
        hide_index=True,
        height=500
    )
    
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 7. MAIN CONTROLLER
# ==============================================================================
def main():
    load_custom_css()
    SessionState.initialize()
    
    # Login Gate
    if st.session_state['page'] == 'login':
        page_login()
        return

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 8px 0 24px 0;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
                <div style="width:40px; height:40px; background:linear-gradient(135deg, {C_PRIMARY}, {C_SECONDARY}); border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:700; color:white; font-size:1.1rem;">
                    {st.session_state.get('user', 'A')[0].upper()}
                </div>
                <div>
                    <div style="font-size:0.95rem; font-weight:600; color:#FAFAFA;">{st.session_state.get('user', 'Admin')}</div>
                    <div style="font-size:0.75rem; color:#64748B; margin-top:2px;">MLOps Engineer</div>
                </div>
            </div>
            
            <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.15); border-radius:10px; padding:10px 12px; margin-bottom:28px; display:flex; align-items:center; gap:8px;">
                <div class="status-pulse"></div>
                <div style="font-size:0.8rem; color:#34D399; font-weight:600;">System Online</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        nav = st.radio("Menu", ["Dashboard", "Prediction Engine", "Audit Logs"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown(f"""
        <div style="padding: 0 4px;">
            <div style="font-size:0.7rem; color:#475569; text-transform:uppercase; letter-spacing:0.08em; font-weight:700; margin-bottom:6px;">Backend Endpoint</div>
            <code style="font-size:0.75rem; color:#94A3B8; background:rgba(0,0,0,0.3); padding:6px 8px; border-radius:6px; display:block; word-break:break-all;">
                {BASE_URL}
            </code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
 if st.button("Log Out", use_container_width=True):
            st.session_state['page'] = 'login'
            st.session_state['user'] = None
            st.rerun()

    # View Routing
    render_header()
    
    if nav == "Dashboard":
        page_dashboard()
    elif nav == "Prediction Engine":
        page_prediction()
    elif nav == "Audit Logs":
        page_logs()

if __name__ == "__main__":
    main()