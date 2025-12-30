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
# Use env var for Docker compatibility, default to your Render URL
BASE_URL = os.getenv("API_URL", "https://mlops-api-40lx.onrender.com")
API_URL = f"{BASE_URL.strip('/')}/predict"
TIMEOUT_SECONDS = 60

# Model Configuration
FEATURE_COUNT = 38
DEFAULT_CUSTOMER_ID = "CUST_12345678"

# UI Theme Colors
COLOR_PRIMARY = "#FF4B4B"
COLOR_SUCCESS = "#09AB3B"
COLOR_WARNING = "#FFAA00"
COLOR_DANGER = "#D93025"
COLOR_DARK = "#1E1E1E"
COLOR_BG_CARD = "#262730"

# ==============================================================================
# 2. CUSTOM CSS & STYLING
# ==============================================================================
def load_custom_css():
    st.markdown(f"""
    <style>
        /* Global Main Container */
        .main {{
            background-color: #0E1117;
        }}
        
        /* Styled Metric Cards */
        div[data-testid="stMetric"] {{
            background-color: {COLOR_BG_CARD};
            border: 1px solid #464B5C;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.2s ease-in-out;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            border-color: {COLOR_PRIMARY};
            box-shadow: 0 8px 15px rgba(255, 75, 75, 0.1);
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_BG_CARD};
            border-right: 1px solid #464B5C;
        }}
        
        /* Headers & Typography */
        h1, h2, h3 {{
            font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
            font-weight: 600;
            color: #FAFAFA;
        }}
        
        /* Buttons */
        div.stButton > button {{
            width: 100%;
            border-radius: 6px;
            font-weight: 600;
            height: 3em;
            transition: all 0.2s;
        }}
        
        /* Expander Styling */
        .streamlit-expanderHeader {{
            background-color: {COLOR_BG_CARD};
            border-radius: 6px;
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
        # Navigation
        if 'page' not in st.session_state:
            st.session_state['page'] = 'login'
        
        # Data Inputs
        if 'features' not in st.session_state:
            st.session_state['features'] = [0.0] * FEATURE_COUNT
            
        # App Data
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        if 'user' not in st.session_state:
            st.session_state['user'] = None
        if 'last_latency' not in st.session_state:
            st.session_state['last_latency'] = 0.0

    @staticmethod
    def add_to_history(entry: Dict):
        """Adds a prediction result to the persistent session history."""
        # Insert at top (newest first)
        st.session_state['history'].insert(0, entry)
        # Keep buffer size manageable (last 100 requests)
        if len(st.session_state['history']) > 100:
            st.session_state['history'] = st.session_state['history'][:100]

# ==============================================================================
# 4. NETWORK SERVICE LAYER
# ==============================================================================
class APIService:
    """Encapsulates API communication with robust error handling."""
    
    @staticmethod
    def predict(customer_id: str, features: List[float]) -> Dict[str, Any]:
        """
        Sends prediction payload to FastAPI backend.
        Handles: Timeouts, Cold Starts, and Connection Errors.
        """
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
            # 60s timeout critical for Render Free Tier cold starts
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
# 5. UI COMPONENTS
# ==============================================================================
def render_header():
    c1, c2 = st.columns([0.5, 6])
    with c1:
        st.markdown("# 🛡️")
    with c2:
        st.markdown("## ChurnGuard Enterprise\n*Production MLOps Control Plane*")

def render_gauge_chart(probability: float):
    """Renders a Plotly Gauge Chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)", 'font': {'size': 20, 'color': "white"}},
        delta = {
            'reference': 50, 
            'increasing': {'color': COLOR_DANGER}, 
            'decreasing': {'color': COLOR_SUCCESS}
        },
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Invisible bar, relying on steps
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#444",
            'steps': [
                {'range': [0, 40], 'color': COLOR_SUCCESS},
                {'range': [40, 70], 'color': COLOR_WARNING},
                {'range': [70, 100], 'color': COLOR_DANGER}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 6. APPLICATION PAGES
# ==============================================================================

def page_login():
    """Mock Enterprise Login Screen."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.container(border=True):
            st.markdown("<h1 style='text-align: center;'>🔐 Access Control</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #888;'>Production MLOps Environment</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            username = st.text_input("Username", placeholder="admin@mlops.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            if st.button("Authenticate System", type="primary"):
                if username: 
                    st.session_state['user'] = username
                    st.session_state['page'] = 'dashboard'
                    st.toast("Authentication Successful", icon="✅")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Please enter a username.")

def page_dashboard():
    """Main Analytics View."""
    st.markdown("## 📊 Live Operations Dashboard")
    
    # KPIs
    df = pd.DataFrame(st.session_state['history'])
    
    # Calculate Metrics
    total_preds = len(df)
    churn_rate = 0.0
    if not df.empty:
        churn_rate = df[df['churn_prediction'] == True].shape[0] / total_preds
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Inferences", f"{total_preds}", help="Session total predictions")
    k2.metric("Session Churn Rate", f"{churn_rate:.1%}", help="% of predictions labeled as Churn")
    k3.metric("Avg Latency", f"{st.session_state['last_latency']:.0f}ms", help="Time taken for last API call")
    k4.metric("System Status", "Healthy", delta="Online", delta_color="normal")

    st.divider()

    # Layout: Chart + Recent Logs
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Risk Distribution Analysis")
        if not df.empty:
            fig = px.histogram(
                df, 
                x="probability", 
                nbins=10, 
                title="Churn Probability Density",
                color_discrete_sequence=[COLOR_PRIMARY],
                range_x=[0, 1]
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available yet. Go to 'Prediction Engine' to generate traffic.")
            
    with c2:
        st.subheader("Recent Activity")
        if not df.empty:
            # Show mini-table of last 5
            mini_df = df[['timestamp', 'customer_id', 'probability']].head(5)
            st.dataframe(mini_df, hide_index=True, use_container_width=True)
        else:
            st.markdown("*Waiting for incoming requests...*")

def page_prediction():
    """Inference Interface with Full Feature Control."""
    st.markdown("## 🤖 Inference Engine")
    
    left_col, right_col = st.columns([1, 1.5], gap="large")
    
    with left_col:
        with st.container(border=True):
            st.subheader("Customer Profile")
            customer_id = st.text_input("Customer UUID", value=DEFAULT_CUSTOMER_ID)
            
            # --- FEATURE CONTROLS ---
            st.markdown("### Feature Vector Control")
            
            tab_quick, tab_fine = st.tabs(["⚡ Auto-Generate", "🎛️ Fine Tune (38 Features)"])
            
            with tab_quick:
                st.caption("Generate a random valid feature vector for testing.")
                if st.button("🎲 Randomize Profile", use_container_width=True):
                    # Generate 38 random floats
                    st.session_state['features'] = [round(random.random(), 2) for _ in range(FEATURE_COUNT)]
                    st.rerun()

            with tab_fine:
                st.caption("Manually adjust specific feature dimensions.")
                # FIX: Logic to handle ALL 38 features dynamically
                # We use a scrollable expander concept by breaking them into chunks
                
                with st.expander("Feature Set 1 (0-10)", expanded=False):
                    for i in range(0, 11):
                        val = st.slider(f"Dim_{i}", 0.0, 1.0, st.session_state['features'][i], 0.01, key=f"f_{i}")
                        st.session_state['features'][i] = val
                        
                with st.expander("Feature Set 2 (11-20)", expanded=False):
                    for i in range(11, 21):
                        val = st.slider(f"Dim_{i}", 0.0, 1.0, st.session_state['features'][i], 0.01, key=f"f_{i}")
                        st.session_state['features'][i] = val
                        
                with st.expander("Feature Set 3 (21-30)", expanded=False):
                    for i in range(21, 31):
                        val = st.slider(f"Dim_{i}", 0.0, 1.0, st.session_state['features'][i], 0.01, key=f"f_{i}")
                        st.session_state['features'][i] = val

                with st.expander("Feature Set 4 (31-37)", expanded=False):
                    for i in range(31, FEATURE_COUNT):
                        val = st.slider(f"Dim_{i}", 0.0, 1.0, st.session_state['features'][i], 0.01, key=f"f_{i}")
                        st.session_state['features'][i] = val

            # --- PAYLOAD PREVIEW ---
            with st.expander("📡 Inspect JSON Payload"):
                display_vec = st.session_state['features'][:5] + ["..."] + st.session_state['features'][-2:]
                st.code(json.dumps({
                    "customer_id": customer_id,
                    "vector_length": len(st.session_state['features']),
                    "sample": display_vec
                }, indent=2), language="json")

            st.markdown("---")
            predict_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    with right_col:
        if predict_btn:
            with st.spinner("Processing inference request..."):
                response = APIService.predict(customer_id, st.session_state['features'])
                
            if response["success"]:
                data = response["data"]
                
                # Extract Data
                prob = float(data.get('probability', 0.0))
                is_churn = int(data.get('prediction', 0)) == 1
                latency = response["latency"]
                st.session_state['last_latency'] = latency
                
                # Log History
                SessionState.add_to_history({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "customer_id": customer_id,
                    "probability": prob,
                    "churn_prediction": is_churn,
                    "latency": latency
                })
                
                # --- RESULT DISPLAY ---
                st.success("Inference Successful")
                
                r1, r2 = st.columns(2)
                with r1:
                    st.metric(
                        "Risk Verdict", 
                        "HIGH RISK" if is_churn else "SAFE",
                        delta=f"{prob*100:.1f}% Confidence",
                        delta_color="inverse" if is_churn else "normal"
                    )
                with r2:
                    st.metric("Latency", f"{latency:.0f}ms")
                
                # Gauge
                render_gauge_chart(prob)
                
                # Metadata
                with st.expander("🔍 Model Metadata & Trace"):
                    st.json(data)
                    
            else:
                # Error UI
                st.error("Inference Failed")
                st.markdown(f"""
                <div style='background: #3d1212; padding: 15px; border-radius: 8px; border-left: 4px solid #ff4b4b;'>
                    <strong>System Error:</strong> {response['error']}
                </div>
                """, unsafe_allow_html=True)
                
        else:
            # Empty State
            st.info("👈 Configure the customer profile on the left to begin analysis.")

def page_logs():
    """Audit Trail View."""
    st.markdown("## 📜 Inference Audit Logs")
    
    if not st.session_state['history']:
        st.warning("No records found. Run predictions to populate logs.")
        return
        
    df = pd.DataFrame(st.session_state['history'])
    
    # Advanced Data Table
    st.dataframe(
        df,
        column_config={
            "timestamp": "Time",
            "customer_id": "Customer UUID",
            "probability": st.column_config.ProgressColumn(
                "Risk Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "churn_prediction": st.column_config.CheckboxColumn("Churn?", default=False),
            "latency": st.column_config.NumberColumn("Latency (ms)", format="%d ms"),
        },
        use_container_width=True,
        hide_index=True
    )
    
    # CSV Download
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Export CSV", 
        csv_data, 
        f"churn_logs_{datetime.now().strftime('%Y%m%d')}.csv", 
        "text/csv"
    )

# ==============================================================================
# 7. MAIN CONTROLLER
# ==============================================================================
def main():
    load_custom_css()
    SessionState.initialize()
    
    # 1. Login Gate
    if st.session_state['page'] == 'login':
        page_login()
        return

    # 2. Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🧭 Console Navigation")
        
        # User Widget
        st.markdown(f"""
        <div style='background: #1E1E1E; padding: 10px; border-radius: 8px; display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
            <div style='width: 30px; height: 30px; background: {COLOR_PRIMARY}; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: bold;'>A</div>
            <div>
                <div style='font-size: 13px; font-weight: 600;'>{st.session_state.get('user', 'Admin')}</div>
                <div style='font-size: 11px; color: #888;'>MLOps Engineer</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        nav = st.radio("Menu", ["Dashboard", "Prediction Engine", "Audit Logs"], label_visibility="collapsed")
        
        st.markdown("---")
        st.caption(f"Backend: `{BASE_URL}`")
        if st.button("Log Out"):
            st.session_state['page'] = 'login'
            st.session_state['user'] = None
            st.rerun()

    # 3. View Routing
    render_header()
    
    if nav == "Dashboard":
        page_dashboard()
    elif nav == "Prediction Engine":
        page_prediction()
    elif nav == "Audit Logs":
        page_logs()

if __name__ == "__main__":
    main()