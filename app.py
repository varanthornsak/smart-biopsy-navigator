import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
import uuid

# =====================================================
# 1. INITIAL CONFIG & ENTERPRISE STYLE
# =====================================================
st.set_page_config(page_title="Smart Biopsy Pro | Multi-Organ AI", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 32px; font-weight: 700; color: #1E3A8A; }
    .card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .status-tag { padding: 2px 8px; border-radius: 5px; font-size: 10px; font-weight: bold; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. SESSION STATE MANAGEMENT (Fix for AttributeError)
# =====================================================
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'case_history' not in st.session_state:
    st.session_state.case_history = pd.DataFrame(columns=["Date", "Case_ID", "Patient", "Organ", "Result", "Confidence"])

# =====================================================
# 3. SECURE LOGIN (Password: SNH_SECURE)
# =====================================================
if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<div style='text-align:center; padding: 40px 0;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2864/2864332.png", width=80)
        st.markdown("<h1 class='main-header'>Smart Biopsy Pro</h1>", unsafe_allow_html=True)
        st.write("Enterprise Diagnostic Intelligence Platform")
        
        with st.form("login"):
            hosp = st.selectbox("Institution", ["Srinagarind Hospital (SNH)", "Bangkok Hospital", "Siriraj Hospital"])
            role = st.selectbox("Role", ["Oncologist", "Radiologist", "Medical Director"])
            pwd = st.text_input("Security Key", type="password")
            if st.form_submit_button("Access Secure Hub", use_container_width=True):
                if pwd == "SNH_SECURE":
                    st.session_state.auth = True
                    st.session_state.user = {"hosp": hosp, "role": role}
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Security Key")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# 4. MAIN INTERFACE (Safe Access)
# =====================================================
with st.sidebar:
    # แก้ไข Error บรรทัดนี้ด้วยการเช็ค st.session_state.user
    if st.session_state.user:
        st.markdown(f"### **🏢 {st.session_state.user['hosp']}**")
        st.caption(f"Authenticated as: {st.session_state.user['role']}")
    
    st.divider()
    nav = st.radio("SOLUTIONS", ["Business Dashboard", "Clinical Inference", "Patient Archive", "Manual"])
    
    if st.button("Logout"):
        st.session_state.auth = False
        st.session_state.user = None
        st.rerun()

# =====================================================
# 5. CLINICAL INFERENCE (Multi-Organ)
# =====================================================
if nav == "Clinical Inference":
    st.markdown("<h1 class
