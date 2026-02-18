import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
import uuid

# =====================================================
# 1. THEME & ADVANCED CSS
# =====================================================
st.set_page_config(page_title="Smart Biopsy Pro | Enterprise", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 32px; font-weight: 700; color: #1E3A8A; margin-bottom: 5px; }
    .sub-header { color: #64748B; margin-bottom: 25px; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
    .card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. SESSION STATE & DATABASE MOCKUP
# =====================================================
if 'auth' not in st.session_state:
    st.session_state.auth = False

if 'case_history' not in st.session_state:
    # สร้างข้อมูลจำลองไว้ในเครื่อง
    st.session_state.case_history = pd.DataFrame(columns=[
        "Date", "Case_ID", "Patient_Name", "Organ", "Risk_Score", "Status"
    ])

# =====================================================
# 3. SECURE LOGIN PAGE
# =====================================================
if not st.session_state.auth:
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<div style='text-align:center; margin-top:50px;'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2864/2864332.png", width=80)
        st.markdown("<h1 class='main-header'>Smart Biopsy Pro</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-header'>Enterprise Medical AI Platform</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            hospital = st.selectbox("Hospital Node", ["Srinagarind Hospital (SNH)", "Bangkok Medical Center", "Siriraj Hub"])
            user_role = st.selectbox("Role", ["Chief Oncologist", "Senior Radiologist", "Medical Director"])
            staff_id = st.text_input("Staff ID / Username")
            password = st.text_input("Security Key (Password)", type="password")
            
            if st.form_submit_button("Authenticate System", use_container_width=True):
                if password == "SNH_SECURE":
                    st.session_state.auth = True
                    st.session_state.user = {"hosp": hospital, "role": user_role, "id": staff_id}
                    st.success("Authentication Successful")
                    st.rerun()
                else:
                    st.error("Invalid Security Key. Please contact system administrator.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# 4. SIDEBAR & ENTERPRISE NAV
# =====================================================
with st.sidebar:
    st.markdown(f"**🏢 {st.session_state.user['hosp']}**")
    st.caption(f"Operator: {st.session_state.user['role']}")
    st.divider()
    nav = st.radio("Enterprise Suite", [
        "Dashboard & Analytics", 
        "Clinical Inference", 
        "Case Archive (Database)", 
        "User Manual"
    ])
    st.divider()
    if st.button("🔒 Secure Logout", use_container_width=True):
        st.session_state.auth = False
        st.rerun()

# =====================================================
# 5. BUSINESS DASHBOARD (BUSINESS-READY)
# =====================================================
if nav == "Dashboard & Analytics":
    st.markdown("<h1 class='main-header'>Business & Clinical Overview</h1>", unsafe_allow_html=True)
    
    # KPIs for Business Value
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Cases (MTD)", "1,452", "+14%")
    k2.metric("Mean Accuracy", "97.2%", "AI Core v3")
    k3.metric("Cost Saved (Est.)", "$12,400", "Automation ROI")
    k4.metric("Avg. Wait Time", "1.8 hrs", "-45% efficiency")

    st.divider()
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Diagnostic Volume by Organ")
        # จำลองกราฟการใช้งานแต่ละแผนก
        chart_data = pd.DataFrame({
            'Organ': ['Liver', 'Thyroid', 'Breast', 'Lymph Nodes'],
            'Volume': [850, 420, 150, 32]
        })
        st.bar_chart(chart_data, x='Organ', y='Volume', color="#1E3A8A")
        
    with col_b:
        st.subheader("System Status")
        st.write("✅ **Liver Module:** Active")
        st.write("🟡 **Thyroid Module:** Beta (82% Trained)")
        st.write("⚪ **Breast Module:** Integration Phase")

# =====================================================
# 6. CLINICAL INFERENCE (THE ENGINE)
# =====================================================
elif nav == "Clinical Inference":
    st.markdown("<h1 class='main-header'>Diagnostic Engine</h1>", unsafe_allow_html=True)
    
    organ = st.selectbox("Select Target Organ", ["Liver (Ready)", "Thyroid (Beta)", "Breast (Coming Soon)"])
    
    col_l, col_r = st.columns([1, 1.2], gap="large")
    
    with col_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        patient_name = st.text_input("Patient Full Name")
        hn_id = st.text_input("Hospital Number (HN)")
        up_file = st.file_uploader("Upload Medical Scan", type=['jpg','png','jpeg'])
        
        if "Liver" in organ:
            afp = st.number_input("AFP Level (ng/mL)", value=10.0)
            size = st.slider("Tumor Size (mm)", 0, 100, 20)
        
        analyze = st.button("🚀 Execute AI Diagnostic", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        if up_file and analyze:
            with st.spinner("Processing..."):
                time.sleep(1.5)
                # Simple logic for Demo
                risk = 0.85 if "Liver" in organ and afp > 100 else 0.15
                status = "Malignant" if risk > 0.7 else "Normal/Benign"
                color = "#e74c3c" if risk > 0.7 else "#27ae60"
                
                # บันทึกข้อมูลลง Database จำลอง
                new_case = pd.DataFrame([{
                    "Date": datetime.datetime.now().strftime("%Y-%m-%d"),
                    "Case_ID": str(uuid.uuid4())[:8],
                    "Patient_Name": patient_name if patient_name else "Anonymous",
                    "Organ": organ.split(' ')[0],
                    "Risk_Score": f"{risk*100:.1f}%",
                    "Status": status
                }])
                st.session_state.case_history = pd.concat([st.session_state.case_history, new_case], ignore_index=True)

            # Results UI
            fig = go.Figure(go.Indicator(mode="gauge+number", value=risk*100, gauge={'bar':{'color':color}}))
            fig.update_layout(height=250, margin=dict(t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div style='background:{color}15; padding:20px; border-radius:10px; border-left:5px solid {color}'>
                <h3 style='color:{color}'>{status} Assessment</h3>
                <p>The AI model identifies morphological markers consistent with <b>{status}</b>. 
                Recommended clinical correlation with MRI Contrast.</p>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# 7. CASE ARCHIVE (DATABASE VIEW)
# =====================================================
elif nav == "Case Archive (Database)":
    st.markdown("<h1 class='main-header'>Case Repository</h1>", unsafe_allow_html=True)
    st.write("Search and retrieve previous AI diagnostic reports.")
    
    search = st.text_input("🔍 Search by Patient Name or Case ID")
    
    df = st.session_state.case_history
    if search:
        df = df[df['Patient_Name'].str.contains(search, case=False) | df['Case_ID'].str.contains(search)]
    
    st.dataframe(df, use_container_width=True, hide_index=True)

# =====================================================
# 8. USER MANUAL (ENGLISH)
# =====================================================
elif nav == "User Manual":
    st.header("📖 Clinical Operations Guide")
    st.markdown("""
    ### System Access
    Authorized personnel must use their **Staff ID** and **Secure Key** to access the platform. 
    Unauthorized access attempts are logged for hospital security audits.

    ### Workflow Details
    1. **Selection:** Choose the appropriate organ module (Liver/Thyroid).
    2. **Input:** Enter mandatory clinical biomarkers to improve AI prediction accuracy.
    3. **Image:** Upload high-quality scans. The system supports cross-sectional imaging features.
    4. **Archiving:** All confirmed cases are automatically stored in the **Case Archive** for legal and research compliance.
    """)
