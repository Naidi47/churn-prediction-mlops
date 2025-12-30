import streamlit as st
import requests
import json
import random
import time
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# Dynamic URL that works in both Docker and Localhost
# If running in Docker Compose, this env var should be set to 'http://api:8000'
BASE_URL = os.getenv("API_URL", "https://mlops-api-40lx.onrender.com")
API_URL = f"{BASE_URL.strip('/')}/predict"

st.set_page_config(page_title="Churn Predictor", page_icon="📉")

# ---------------------------------------------------------
# TITLE & DESCRIPTION
# ---------------------------------------------------------
st.title("📉 Customer Churn Prediction System")
st.markdown(f"""
This application connects to a **Production MLOps Pipeline**.
**API Endpoint:** `{API_URL}`
""")

st.divider()

# ---------------------------------------------------------
# SIDEBAR: CONTROLS
# ---------------------------------------------------------
st.sidebar.header("User Input")
customer_id = st.sidebar.text_input("Customer ID", value="CUST_12345678")

# Initialize session state for features if not exists
if 'features' not in st.session_state:
    st.session_state['features'] = [0.0] * 38

# Option to generate random data
if st.sidebar.button("🎲 Generate Random Customer Profile"):
    # Generate 38 random floats between 0 and 1
    random_features = [round(random.random(), 2) for _ in range(38)]
    st.session_state['features'] = random_features
    # Force rerun to update the UI immediately
    st.rerun()

# Get features from state
features = st.session_state['features']

# Show the features being sent
st.write("### 📡 Payload to be sent:")
# Display a truncated version for readability
display_features = features[:5] + ["... (33 more) ..."] if len(features) > 5 else features
st.code(json.dumps({
    "customer_id": customer_id,
    "feature_vector": display_features
}, indent=2), language="json")


# ---------------------------------------------------------
# PREDICTION ACTION
# ---------------------------------------------------------
if st.button("🚀 Predict Churn Risk", type="primary"):
    
    payload = {
        "customer_id": customer_id,
        "feature_vector": features
    }

    try:
        # Spinner message improved for Production MLOps feel
        with st.spinner(f"Transmitting vector to inference engine..."):
            # Record start time
            start_time = time.time()
            
            # Send Request - Timeout increased to 60 for Render reliability
            response = requests.post(API_URL, json=payload, timeout=60)
            
            # Calculate UI latency
            ui_latency = (time.time() - start_time) * 1000

        if response.status_code == 200:
            result = response.json()
            
            # DISPLAY RESULTS
            st.success("Prediction Successful!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Handle cases where prediction might be 0/1 or boolean
                is_churn = int(result.get('prediction', 0)) == 1
                st.metric(
                    label="Prediction", 
                    value="🔴 Churn" if is_churn else "🟢 Retain",
                    delta="-Risk" if not is_churn else "+Risk",
                    delta_color="inverse"
                )
            
            with col2:
                prob = float(result.get('probability', 0.0))
                st.metric(label="Probability", value=f"{prob:.2%}")
                
            with col3:
                latency = result.get('latency_ms', 0.0)
                st.metric(label="API Latency", value=f"{latency:.2f} ms")

            # VISUALIZE PROBABILITY
            st.progress(prob)
            if prob > 0.5:
                st.warning("⚠️ High Risk of Churn! Recommended Action: Send Discount Coupon.")
            else:
                st.info("✅ Customer is Safe.")

            # DEBUG INFO
            with st.expander("🔍 System Debug Info"):
                st.json(result)
                st.write(f"Model Version: `{result.get('model_version', 'N/A')}`")
                st.write(f"Fallback Active: `{result.get('is_fallback', False)}`")
                st.write(f"Feature Source: `{result.get('feature_source', 'N/A')}`")

        elif response.status_code == 502:
            st.error("📉 API Error: 502 (Bad Gateway)")
            st.info("The Render API is currently waking up from sleep mode. Please try again in 15 seconds.")
        else:
            st.error(f"API Error: {response.status_code}")
            st.text(response.text)

    except requests.exceptions.Timeout:
        st.error("⌛ Request Timeout: The API took too long to respond (Cold Start).")
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to API.")
        st.error(f"Tried connecting to: `{API_URL}`")
        st.info("💡 **If running in Docker:** Ensure `API_URL` env var is set to `http://api:8000`.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")