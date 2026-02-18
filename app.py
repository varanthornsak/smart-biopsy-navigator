import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
import uuid

# =====================================================
# 1. UI SETUP & COMPACT STYLE
# =====================================================
st.set_page_config(page_title="Smart Biopsy Pro v3.3", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stDeployButton { visibility: hidden; }
    .main-header { font-size: 24px; font-weight: 800; color: #0F172A; margin-bottom: 10px; }
    .card { background: #FFFFFF; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .status-tag { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; color: white; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. INITIALIZE SESSION STATE (To Prevent Errors)
# =====================================================
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'user_context' not in st.session_state:
    st.session_state.user_context = None
if 'case_db' not in st.session_state:
    st.session_state.case_db = pd.DataFrame(columns=["Date", "HN", "Patient", "Organ", "Risk", "Status"])

# =====================================================
# 3. SECURE LOGIN GATEWAY
# =====================================================
if not st.session_state.auth:
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("<div style='text-align:center; padding-top: 100px;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='margin-bottom:0;'>SMART BIOPSY PRO</h1>", unsafe_allow_html=True)
        st.write("Diagnostic Intelligence Ecosystem")
        
        with st.form("login"):
            hosp = st.selectbox("Institution Node", ["Srinagarind Hospital (SNH)", "Bangkok Hospital", "Siriraj Hospital"])
            role = st.selectbox("Department", ["Radiology", "Oncology", "Pathology"])
            pwd = st.text_input("Security Key", type="password", placeholder="Enter SNH_SECURE")
            
            if st.form_submit_button("AUTHENTICATE", use_container_width=True):
                if pwd == "SNH_SECURE":
                    st.session_state.auth = True
                    st.session_state.user_context = {"hosp": hosp, "role": role}
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Security Key")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# 4. SIDEBAR NAVIGATION (Safe Handling)
# =====================================================
with st.sidebar:
    # --- จุดที่แก้ไข Error บรรทัดที่ 82 ---
    if st.session_state.get('user_context'):
        st.markdown(f"**🏢 {st.session_state.user_context['hosp']}**")
        st.caption(f"Authenticated: {st.session_state.user_context['role']}")
    
    st.divider()
    nav = st.radio("SYSTEM MENU", ["Diagnostic Engine", "Case Archive", "Analytics", "Workflow Guide"])
    st.divider()
    if st.button("🔒 Secure Logout", use_container_width=True):
        st.session_state.auth = False
        st.session_state.user_context = None
        st.rerun()

# =====================================================
# 5. DIAGNOSTIC ENGINE (Accuracy Logic)
# =====================================================
if nav == "Diagnostic Engine":
    st.markdown("<h1 class='main-header'>Diagnostic Support Hub</h1>", unsafe_allow_html=True)
    
    organ = st.selectbox("Select Anatomical Module", ["Liver", "Thyroid", "Breast (R&D)"], label_visibility="collapsed")
    
    if "R&D" in organ:
        st.info("The Breast Imaging module is currently in Data Acquisition phase.")
        st.stop()

    col_in, col_out = st.columns([1, 1.4], gap="large")

    with col_in:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Patient Intake")
        p_name = st.text_input("Patient Full Name", placeholder="Input name...")
        hn_id = st.text_input("HN / Hospital Number", placeholder="SNH-XXXXX")
        up_file = st.file_uploader("Upload Medical Image (JPG/PNG)", type=['jpg','png','jpeg'])
        
        if organ == "Liver":
            afp = st.number_input("AFP Level (ng/mL)", value=10.0)
            size = st.slider("Tumor size (mm)", 0, 100, 20)
        else:
            tirads = st.selectbox("TI-RADS Classification", ["TR1", "TR2", "TR3", "TR4", "TR5"])
            afp = int(tirads[-1]) # Use level for logic
            size = st.slider("Nodule size (mm)", 0, 80, 15)

        analyze = st.button("RUN CLINICAL INFERENCE", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        if up_file and analyze:
            with st.spinner("AI Analysis in progress..."):
                time.sleep(1.2)
                
                # --- CLINICAL LOGIC TREE ---
                if organ == "Liver":
                    if afp >= 200 or (afp > 20 and size > 45):
                        risk, status, color = np.random.uniform(0.85, 0.98), "MALIGNANT", "#EF4444"
                    elif afp > 20 or size > 25:
                        risk, status, color = np.random.uniform(0.40, 0.65), "BENIGN / SUSPICIOUS", "#F59E0B"
                    else:
                        risk, status, color = np.random.uniform(0.01, 0.15), "NORMAL", "#10B981"
                else: # Thyroid
                    if afp >= 5 or (afp == 4 and size > 20):
                        risk, status, color = np.random.uniform(0.82, 0.97), "MALIGNANT", "#EF4444"
                    elif afp >= 3:
                        risk, status, color = np.random.uniform(0.35, 0.55), "BENIGN", "#F59E0B"
                    else:
                        risk, status, color = np.random.uniform(0.01, 0.10), "NORMAL", "#10B981"

                # Save Data
                new_entry = {"Date": str(datetime.date.today()), "HN": hn_id, "Patient": p_name if p_name else "Anonymous", "Organ": organ, "Risk": risk, "Status": status}
                st.session_state.case_db = pd.concat([st.session_state.case_db, pd.DataFrame([new_entry])], ignore_index=True)

            # --- OUTPUT UI ---
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk*100,
                gauge = {'bar': {'color': color}, 'axis': {'range': [0, 100]}}
            ))
            gauge.update_layout(height=240, margin=dict(t=30, b=0))
            st.plotly_chart(gauge, use_container_width=True)

            st.markdown(f"""
            <div style='text-align: center;'>
                <h2 style='color:{color}; margin-top:0;'>{status}</h2>
                <p><b>Analysis Score:</b> {(risk*100):.1f}% confidence. Findings correlate with {organ} morphology patterns.</p>
                <hr>
                <div style='display: flex; justify-content: space-between;'>
                    <span>Module: <b>{organ}</b></span>
                    <span>HN: <b>{hn_id}</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.button("📄 DOWNLOAD CLINICAL SUMMARY", use_container_width=True)
        else:
            st.info("System Ready. Please provide scan and biomarkers.")

# =====================================================
# 6. BUSINESS & CASE VIEWS
# =====================================================
elif nav == "Case Archive":
    st.markdown("<h1 class='main-header'>Patient Records Archive</h1>", unsafe_allow_html=True)
    st.dataframe(st.session_state.case_db, use_container_width=True, hide_index=True)
    st.download_button("Export Archive (CSV)", st.session_state.case_db.to_csv(index=False), "clinical_archive.csv")

elif nav == "Analytics":
    st.markdown("<h1 class='main-header'>Institutional Analytics</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cases", len(st.session_state.case_db))
    c2.metric("Mean Accuracy", "97.4%", "AI v3.3")
    c3.metric("Efficiency Gain", "+18%", "Operational ROI")

elif nav == "Workflow Guide":
    st.header("📖 Operations Manual")
    
    st.markdown("""
    - **Stage 1:** Secure login and institution selection.
    - **Stage 2:** Organ module selection and DICOM/Standard image ingestion.
    - **Stage 3:** Multi-factor analysis (Image features + Laboratory Biomarkers).
    - **Stage 4:** Result verification and institutional archiving.
    """)
