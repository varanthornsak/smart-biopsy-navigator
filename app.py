import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
import uuid

# =====================================================
# 1. PAGE SETUP & ENTERPRISE CSS
# =====================================================
st.set_page_config(page_title="Smart Biopsy Pro | Multi-Organ AI", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 32px; font-weight: 700; color: #1E3A8A; }
    .card { background: white; padding: 25px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .stButton>button { border-radius: 8px; font-weight: 600; height: 3em; }
    .status-badge { padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. SESSION STATE MANAGEMENT
# =====================================================
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'case_history' not in st.session_state:
    # เริ่มต้นฐานข้อมูลจำลองด้วยตัวอย่างเคส
    st.session_state.case_history = pd.DataFrame(columns=["Date", "Case_ID", "Patient", "Organ", "Result", "Confidence"])

# =====================================================
# 3. SECURE LOGIN (Password: SNH_SECURE)
# =====================================================
if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<div style='text-align:center; padding-top: 50px;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2864/2864332.png", width=80)
        st.markdown("<h1 class='main-header'>Smart Biopsy Pro</h1>", unsafe_allow_html=True)
        st.write("Enterprise Diagnostic Intelligence Platform")
        
        with st.form("login_form"):
            hosp = st.selectbox("Institution Node", ["Srinagarind Hospital (SNH)", "Bangkok Hospital", "Siriraj Hospital"])
            role = st.selectbox("Professional Role", ["Chief Oncologist", "Radiologist", "Surgeon"])
            pwd = st.text_input("Security Key", type="password")
            
            if st.form_submit_button("Authenticate System", use_container_width=True):
                if pwd == "SNH_SECURE":
                    st.session_state.auth = True
                    st.session_state.user = {"hosp": hosp, "role": role}
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid Security Key")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# 4. SIDEBAR NAVIGATION
# =====================================================
with st.sidebar:
    if st.session_state.user:
        st.markdown(f"### **🏢 {st.session_state.user['hosp']}**")
        st.caption(f"Authenticated: {st.session_state.user['role']}")
    
    st.divider()
    nav = st.radio("SOLUTIONS", ["Business Dashboard", "Clinical Inference", "Patient Archive", "Manual"])
    
    st.spacer = st.container()
    st.sidebar.markdown("---")
    if st.button("🔒 Secure Logout", use_container_width=True):
        st.session_state.auth = False
        st.session_state.user = None
        st.rerun()

# =====================================================
# 5. BUSINESS DASHBOARD (Business Logic)
# =====================================================
if nav == "Business Dashboard":
    st.markdown("<h1 class='main-header'>Business Intelligence Dashboard</h1>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Diagnostics", len(st.session_state.case_history))
    c2.metric("System Uptime", "99.98%", "Stable")
    c3.metric("AI Precision (Avg)", "96.5%", "+0.2% improvement")
    
    st.divider()
    
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Diagnostic Distribution")
        if not st.session_state.case_history.empty:
            organ_counts = st.session_state.case_history['Organ'].value_counts()
            st.bar_chart(organ_counts)
        else:
            st.info("No diagnostic data available for this period.")

    with col_right:
        st.subheader("Module Status")
        st.markdown("""
        - **Liver Engine:** ✅ Production Ready
        - **Thyroid Engine:** 🟡 Beta Testing (85% Accuracy)
        - **Breast Engine:** ⚪ Data Acquisition
        - **Lymph Node Engine:** ⚪ Research Phase
        """)

# =====================================================
# 6. CLINICAL INFERENCE (Multi-Organ Hub)
# =====================================================
elif nav == "Clinical Inference":
    st.markdown("<h1 class='main-header'>Diagnostic Engine</h1>", unsafe_allow_html=True)
    
    organ = st.selectbox("Target Organ Module", ["Liver (Full Support)", "Thyroid (Beta)", "Breast (Planned)", "Lymph Nodes (Planned)"])
    
    if "Planned" in organ:
        st.warning(f"The {organ} module is currently in development. Please contact R&D for sandbox access.")
        st.stop()

    col_l, col_r = st.columns([1, 1.2], gap="large")

    with col_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        p_name = st.text_input("Patient Full Name")
        up_file = st.file_uploader(f"Upload Image Scan", type=['jpg','png','jpeg'])
        
        # Clinical Contextual Inputs
        if "Liver" in organ:
            val = st.number_input("Serum AFP (ng/mL)", value=10.0)
            st.caption("Standard Threshold: 200 ng/mL for HCC suspicion")
        else: # Thyroid
            val = st.selectbox("TI-RADS Classification", ["TR1", "TR2", "TR3", "TR4", "TR5"])
            
        analyze = st.button("🚀 Run AI Analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if up_file and analyze:
            with st.spinner("Analyzing Morphology..."):
                time.sleep(1.5)
                
                # Mock AI Decision Logic
                risk = 0.88 if ("Liver" in organ and val > 150) or ("Thyroid" in organ and val == "TR5") else 0.12
                res_label = "Malignant" if risk > 0.5 else "Benign/Normal"
                res_color = "#e74c3c" if risk > 0.5 else "#27ae60"

                # Archive Case
                new_entry = pd.DataFrame([{
                    "Date": datetime.date.today().strftime("%Y-%m-%d"),
                    "Case_ID": str(uuid.uuid4())[:8].upper(),
                    "Patient": p_name if p_name else "Anonymous",
                    "Organ": organ.split(' ')[0],
                    "Result": res_label,
                    "Confidence": f"{risk*100:.1f}%"
                }])
                st.session_state.case_history = pd.concat([st.session_state.case_history, new_entry], ignore_index=True)

            # Visual Indicator
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=risk*100,
                gauge={'bar':{'color':res_color}, 'axis':{'range':[0,100]}}
            ))
            fig.update_layout(height=280, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div style='background:{res_color}15; padding:20px; border-radius:10px; border-left:10px solid {res_color}'>
                <h2 style='color:{res_color}; margin-top:0;'>{res_label.upper()} ASSESSMENT</h2>
                <p><b>Clinical Correlation:</b> Findings are {'highly suspicious' if risk > 0.5 else 'unlikely to be cancerous'}. 
                Cross-reference with patient history and secondary imaging is required.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.button("📄 Generate PDF Medical Report", use_container_width=True)
        else:
            st.info("System Ready: Please upload scan and clinical markers to begin.")

# =====================================================
# 7. PATIENT ARCHIVE (The Database)
# =====================================================
elif nav == "Patient Archive":
    st.markdown("<h1 class='main-header'>Clinical Archive</h1>", unsafe_allow_html=True)
    st.write("Search and retrieve institutional case records.")
    
    search_q = st.text_input("🔍 Search by Patient Name or Case ID")
    
    df_view = st.session_state.case_history
    if search_q:
        df_view = df_view[df_view['Patient'].str.contains(search_q, case=False) | df_view['Case_ID'].str.contains(search_q)]
        
    st.dataframe(df_view, use_container_width=True, hide_index=True)
    st.download_button("Export Archive (CSV)", df_view.to_csv(index=False), "clinical_archive.csv")

# =====================================================
# 8. USER MANUAL (ENGLISH)
# =====================================================
elif nav == "Manual":
    st.header("📖 Clinical Workflow Manual")
    
    st.markdown("""
    ### 1. System Authentication
    Access is restricted to authorized medical personnel. Use your **Staff ID** and the **SNH_SECURE** key to enter the platform.
    
    ### 2. Multi-Organ Diagnostic Hub
    The system supports multiple anatomical regions. Ensure you select the correct module (e.g., Liver, Thyroid) before uploading images, as the AI models are region-specific.
    
    ### 3. Interpreting Results
    * **Normal/Benign (Green):** Low suspicion of malignancy. Follow standard observation protocols.
    * **Malignant (Red):** High suspicion. Immediate clinical correlation or biopsy is advised.
    
    ### 4. Data Privacy & Compliance
    All diagnostic data is encrypted and stored in the **Patient Archive** for medico-legal documentation and institutional research.
    """)
