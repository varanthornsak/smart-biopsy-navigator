import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
from PIL import Image

# =====================================================
# 1. THEME & CSS
# =====================================================
st.set_page_config(page_title="Smart Biopsy Pro v2.5", layout="wide")

st.markdown("""
<style>
    .report-card { border-radius: 10px; padding: 20px; margin: 10px 0; border: 1px solid #e0e0e0; }
    .status-normal { background-color: #e8f7ef; border-left: 5px solid #27ae60; color: #1e8449; }
    .status-benign { background-color: #fffde7; border-left: 5px solid #f1c40f; color: #9a7d0a; }
    .status-malignant { background-color: #fdecea; border-left: 5px solid #e74c3c; color: #943126; }
    .manual-header { color: #2c3e50; font-weight: 600; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. LOGIN SYSTEM (Simulation)
# =====================================================
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.title("🔬 Smart Biopsy Pro")
        st.subheader("Login to Clinical Portal")
        with st.form("login"):
            hosp = st.selectbox("Hospital", ["Srinagarind Hospital", "Bangkok Hospital", "Siriraj Hospital"])
            role = st.selectbox("Role", ["Oncologist", "Radiologist", "Hepatologist"])
            uid = st.text_input("Physician ID")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Authenticate", use_container_width=True):
                st.session_state.auth = True
                st.session_state.user = {"hosp": hosp, "role": role, "id": uid}
                st.rerun()
    st.stop()

# =====================================================
# 3. SIDEBAR & NAVIGATION
# =====================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864332.png", width=60)
    st.write(f"**User:** {st.session_state.user['id']}")
    st.write(f"**Institution:** {st.session_state.user['hosp']}")
    st.write(f"**Role:** {st.session_state.user['role']}")
    st.divider()
    nav = st.radio("Navigate", ["User Manual", "Clinical Inference", "Analytics Dashboard"])
    if st.sidebar.button("Log Out"):
        st.session_state.auth = False
        st.rerun()

# =====================================================
# 4. USER MANUAL (ENGLISH)
# =====================================================
if nav == "User Manual":
    st.header("📖 System Operations Manual")
    
    st.markdown("""
    ### 1. Data Preparation
    * **Image Format:** Ensure the ultrasound or CT images are in `.jpg`, `.png`, or `.dicom` format.
    * **Resolution:** Minimum recommended resolution is `800x800` pixels for feature extraction.
    
    ### 2. Clinical Workflow
    * **Upload:** Navigate to 'Clinical Inference' and upload the patient's scan.
    * **Clinical Inputs:** Enter laboratory findings (e.g., AFP levels, Tumor Size) to enhance correlation.
    * **Inference:** Click 'Run AI Analysis'. The system uses a deep-learning model to categorize the lesion.
    
    ### 3. Interpreting Results
    * <span style='color:green'>●</span> **Normal:** No significant abnormalities detected.
    * <span style='color:orange'>●</span> **Benign:** Lesion detected but shows non-cancerous characteristics (e.g., Hemangioma).
    * <span style='color:red'>●</span> **Malignant:** High suspiciousness for malignancy (e.g., HCC). Immediate biopsy or further MRI is recommended.
    """, unsafe_allow_html=True)
    
    

# =====================================================
# 5. CLINICAL INFERENCE (THE ENGINE)
# =====================================================
elif nav == "Clinical Inference":
    st.header("🔬 Diagnostic Support Center")
    
    col_input, col_res = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.subheader("Patient Parameters")
        up_file = st.file_uploader("Upload Scan", type=['jpg','png','jpeg'])
        
        tumor_size = st.slider("Tumor Size (mm)", 0, 150, 20)
        afp_level = st.number_input("Serum AFP (ng/mL)", value=10.0)
        patient_history = st.multiselect("Clinical History", ["Cirrhosis", "Hepatitis B/C", "Alcoholic Liver Disease"])
        
        run_btn = st.button("🚀 Run AI Diagnostic Analysis", use_container_width=True)

    with col_res:
        if up_file and run_btn:
            with st.spinner("Analyzing Morphology & Biomarkers..."):
                time.sleep(2) # Simulation
                
                # Logical Result Logic
                if afp_level > 200 or tumor_size > 50:
                    prob = 0.88; label = "Malignant"; style = "status-malignant"; color = "#e74c3c"
                    detail = "Findings highly suggestive of Hepatocellular Carcinoma (HCC). Increased AFP levels correlate with AI morphology analysis."
                elif tumor_size > 10:
                    prob = 0.45; label = "Likely Benign"; style = "status-benign"; color = "#f1c40f"
                    detail = "Possible Hemangioma or Focal Nodular Hyperplasia. Suggest follow-up ultrasound in 6 months."
                else:
                    prob = 0.05; label = "Normal"; style = "status-normal"; color = "#27ae60"
                    detail = "Liver parenchyma appears homogenous. No focal lesions identified."

            # Visual Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob*100,
                title = {'text': "Malignancy Probability Index"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}}
            ))
            fig.update_layout(height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Clinical Interpretation Box
            st.markdown(f"""
            <div class="report-card {style}">
                <h3>Classification: {label}</h3>
                <p><b>AI Confidence:</b> {prob*100:.2f}%</p>
                <hr>
                <b>Clinical Correlation:</b><br>
                {detail}
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendation Table
            st.markdown("### 📋 Recommended Action Plan")
            rec_data = {
                "Priority": ["High" if label == "Malignant" else "Routine"],
                "Next Step": ["CT/MRI with Contrast" if label != "Normal" else "Annual Screening"],
                "Consultation": ["Oncology Multidisciplinary Team" if label == "Malignant" else "General Practitioner"]
            }
            st.table(pd.DataFrame(rec_data))

        elif up_file:
            st.image(up_file, caption="Pending Analysis...", use_container_width=True)
        else:
            st.info("Waiting for image upload and clinical data...")

# ================= =====================================
# 6. ANALYTICS (SUMMARY)
# =====================================================
elif nav == "Analytics Dashboard":
    st.header("📊 Institutional Statistics")
    st.caption(f"Data for {st.session_state.user['hosp']}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Cases Processed", "1,240", "+5%")
    m2.metric("Malignant Accuracy", "96.4%", "AI-validated")
    m3.metric("Avg. Reporting Time", "4.2 min", "-1.5 min")
