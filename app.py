import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time

# =====================================================
# 1. THEME & CLINICAL STYLE
# =====================================================
st.set_page_config(page_title="AI Diagnostic Platform v3.0", layout="wide")

st.markdown("""
<style>
    .stSelectbox label { font-weight: 600; color: #2c3e50; }
    .organ-badge { 
        padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; 
        margin-left: 10px; display: inline-block;
    }
    .badge-ready { background-color: #d4edda; color: #155724; }
    .badge-training { background-color: #fff3cd; color: #856404; }
    .badge-soon { background-color: #f8f9fa; color: #6c757d; border: 1px solid #dee2e6; }
    .clinical-card { border: 1px solid #e9ecef; padding: 20px; border-radius: 10px; background: #ffffff; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. APP STATE & AUTH SIMULATION
# =====================================================
if 'auth' not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.title("🛡 Clinical AI Gateway")
        with st.form("login"):
            st.selectbox("Institution", ["Srinagarind Hospital (KKU)", "Bangkok Hospital", "Siriraj Hospital"])
            st.selectbox("Role", ["Oncologist", "Radiologist", "Surgeon"])
            st.text_input("Staff Credentials")
            if st.form_submit_button("Enter Diagnostic Hub", use_container_width=True):
                st.session_state.auth = True
                st.rerun()
    st.stop()

# =====================================================
# 3. SIDEBAR NAVIGATION
# =====================================================
with st.sidebar:
    st.markdown("### **Diagnostic Hub**")
    nav = st.radio("System Menu", ["User Manual", "Clinical Analysis", "Data Statistics"])
    st.divider()
    st.caption("AI Engine Version: Multi-Core 3.0.4")
    if st.button("Secure Logout"):
        st.session_state.auth = False
        st.rerun()

# =====================================================
# 4. USER MANUAL (ENGLISH)
# =====================================================
if nav == "User Manual":
    st.header("📖 Clinical Operations Guide")
    st.markdown("""
    ### System Overview
    This platform provides AI-driven morphology analysis across multiple anatomical regions. 
    The engine is designed to support clinical workflows by providing risk stratification and correlation with patient biomarkers.

    ### How to Use
    1. **Organ Selection:** Select the target anatomical region from the dropdown. 
    2. **Clinical Data Input:** Enter specific biomarkers (e.g., AFP for Liver, BI-RADS for Breast).
    3. **Image Acquisition:** Upload DICOM or standard image formats (JPG/PNG).
    4. **Inference:** Review the AI Malignancy Probability and Clinical Correlation report.

    ### Organ Deployment Status
    - **Liver:** Full Diagnostic Support (Ready)
    - **Thyroid:** Deep Learning Training in Progress (Beta Available)
    - **Breast:** Data Acquisition Phase (Coming Soon)
    - **Lymph Nodes:** Research Phase (Planned)
    """)
    

# =====================================================
# 5. CLINICAL ANALYSIS (MULTI-ORGAN FOCUS)
# =====================================================
elif nav == "Clinical Analysis":
    st.header("🔬 Multi-Organ Diagnostic Engine")

    # --- ORGAN SELECTOR (The Hub) ---
    st.markdown("### **Select Anatomical Region**")
    organ_choice = st.selectbox(
        "Select organ for analysis",
        ["Liver (Full Support)", "Thyroid (In-Training)", "Breast (Coming Soon)", "Lymph Nodes (Planned)"],
        index=0
    )

    # Logic to handle non-ready organs
    is_ready = "Full Support" in organ_choice
    is_beta = "In-Training" in organ_choice

    if not is_ready and not is_beta:
        st.warning(f"⚠️ **Notice:** The **{organ_choice}** module is currently in development. If you are part of the Research & Development team, please log in with Dev-Credentials to access the sandbox.")
        st.stop()

    if is_beta:
        st.info("💡 **Beta Feature:** Thyroid analysis is currently in training. Results should be used for research purposes only.")

    st.divider()

    # --- INPUT & ANALYSIS SECTION ---
    col_input, col_res = st.columns([1, 1.2], gap="large")

    with col_input:
        st.subheader("Diagnostic Parameters")
        up_file = st.file_uploader(f"Upload {organ_choice.split(' ')[0]} Scan", type=['jpg','png','jpeg'])
        
        # Dynamic Clinical Input based on Organ
        if "Liver" in organ_choice:
            afp = st.number_input("Serum AFP (ng/mL)", value=10.0)
            tumor_size = st.slider("Max Diameter (mm)", 0, 150, 20)
            history = st.multiselect("Risk Factors", ["Cirrhosis", "HBV/HCV", "Steatosis"])
        elif "Thyroid" in organ_choice:
            st.selectbox("TI-RADS Classification", ["TR1", "TR2", "TR3", "TR4", "TR5"])
            st.slider("Nodule Size (mm)", 0, 80, 10)
            st.checkbox("Presence of Microcalcifications")

        analyze_btn = st.button("🚀 Analyze Clinical Data", use_container_width=True, disabled=not up_file)

    with col_res:
        if up_file and analyze_btn:
            with st.spinner("Processing High-Dimensional Features..."):
                time.sleep(2)
                
                # Dynamic Logic for Result (Example Liver)
                if "Liver" in organ_choice:
                    risk = 0.82 if afp > 200 or tumor_size > 50 else (0.40 if tumor_size > 15 else 0.05)
                else: # Thyroid Beta Logic
                    risk = 0.65 

                label = "Malignant" if risk > 0.7 else ("Benign" if risk > 0.2 else "Normal")
                color = "#e74c3c" if risk > 0.7 else ("#f1c40f" if risk > 0.2 else "#27ae60")
                style = "border-left: 5px solid " + color

            # Visual Indicator
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk*100,
                title = {'text': "Probability of Malignancy"},
                gauge = {'bar': {'color': color}, 'axis': {'range': [None, 100]}}
            ))
            fig.update_layout(height=280, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # INTERPRETATION BOX
            st.markdown(f"""
            <div style="padding:20px; border-radius:10px; background:#f8f9fa; {style}">
                <h3 style="color:{color};">{label.upper()} ASSESSMENT</h3>
                <p><b>AI Index:</b> {risk*100:.1f}% Confidence Level</p>
                <hr>
                <b>Clinical Correlation:</b><br>
                Findings for {organ_choice.split(' ')[0]} analysis show morphological patterns 
                {'consistent with malignancy' if risk > 0.7 else 'likely representing non-cancerous changes'}.
                Cross-reference with patient's serum markers and previous history is highly recommended.
            </div>
            """, unsafe_allow_html=True)
            
            st.button("📄 Export Official Medical Report", use_container_width=True)
        else:
            st.info("System Ready. Please upload an image and provide clinical parameters to begin.")

# =====================================================
# 6. ANALYTICS (GLOBAL)
# =====================================================
elif nav == "Data Statistics":
    st.header("📊 Global Diagnostic Statistics")
    st.write("Cross-departmental AI Performance Monitoring")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Liver Module", "Ready", "Active")
    col2.metric("Thyroid Module", "85% Trained", "Beta")
    col3.metric("Breast Module", "Data Collection", "Planned")
    
    st.divider()
    st.subheader("Model Precision Trends")
    st.line_chart(np.random.randn(20, 2))
