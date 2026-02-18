import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# =====================================================
# 1. INITIAL CONFIG & CSS
# =====================================================
st.set_page_config(page_title="Smart Biopsy Pro", layout="wide", page_icon="🔬")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .login-card {
        background: white;
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #f0f2f6;
    }
    .guide-step {
        padding: 15px;
        border-left: 4px solid #007bff;
        background: #f8f9fa;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    .user-profile {
        font-size: 0.85rem;
        color: #6c757d;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. SESSION STATE (LOGIN CONTROL)
# =====================================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_data = {}

def logout():
    st.session_state.logged_in = False
    st.rerun()

# =====================================================
# 3. LOGIN PAGE
# =====================================================
if not st.session_state.logged_in:
    _, col, _ = st.columns([1, 1.5, 1])
    
    with col:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2864/2864332.png", width=70)
        st.title("Smart Biopsy Navigator")
        st.subheader("Clinical Decision Support System")
        
        with st.form("login_form"):
            hospital = st.selectbox("Select Institution", [
                "Srinagarind Hospital (KKU)",
                "Siriraj Hospital",
                "Bangkok Hospital",
                "Ramathibodi Hospital",
                "King Chulalongkorn Memorial Hospital"
            ])
            
            role = st.selectbox("Professional Role", [
                "Oncologist", 
                "Radiologist", 
                "Pathologist", 
                "Medical Technician"
            ])
            
            user_id = st.text_input("Staff ID / Username", placeholder="EMP-XXXXX")
            password = st.text_input("Password", type="password")
            
            submit = st.form_submit_button("Sign In to System", use_container_width=True)
            
            if submit:
                if user_id and password: # ในระบบจริงต้องเช็ค DB
                    st.session_state.logged_in = True
                    st.session_state.user_data = {
                        "hospital": hospital,
                        "role": role,
                        "user_id": user_id
                    }
                    st.rerun()
                else:
                    st.error("Please enter valid credentials")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# 4. MAIN APP INTERFACE (POST-LOGIN)
# =====================================================

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"""
    <div class='user-profile'>
        Logged in as: <b>{st.session_state.user_data['user_id']}</b><br>
        📍 {st.session_state.user_data['hospital']}<br>
        🩺 {st.session_state.user_data['role']}
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio("MAIN MENU", ["Instruction Guide", "Inference Center", "Analytics", "Settings"])
    st.divider()
    if st.button("Logout", use_container_width=True):
        logout()

# --- Top Header ---
st.markdown(f"### {menu}")
st.caption(f"Network Status: Connected to {st.session_state.user_data['hospital']} Secure Cloud")

# =====================================================
# 5. INSTRUCTION GUIDE (NEW SECTION)
# =====================================================
if menu == "Instruction Guide":
    st.header("📖 Clinical Workflow Manual")
    st.info("โปรดอ่านขั้นตอนการใช้งานเพื่อความแม่นยำสูงสุดในการวินิจฉัย (Recommended for First-time users)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='guide-step'>
            <b>Step 1: Patient Data Acquisition</b><br>
            เตรียมไฟล์ภาพ Ultrasound หรือ CT Scan ในรูปแบบ .JPG หรือ .PNG (รองรับความละเอียดสูงถึง 4K)
        </div>
        <div class='guide-step'>
            <b>Step 2: Analysis Mode Selection</b><br>
            เลือกโหมด <b>Screening</b> (เน้นหาความเสี่ยง) หรือ <b>Diagnostic</b> (เน้นความแม่นยำยืนยันผล)
        </div>
        <div class='guide-step'>
            <b>Step 3: AI Inference</b><br>
            ระบบจะวิเคราะห์ Morphology ของเนื้อเยื่อ และคำนวณค่า Malignancy Probability (%)
        </div>
        <div class='guide-step'>
            <b>Step 4: Clinical Confirmation</b><br>
            แพทย์ตรวจสอบผล AI และทำการยืนยัน (Confirm) เพื่อบันทึกลงในฐานข้อมูลระบบโรงพยาบาล (EMR)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        
        st.image("https://img.freepik.com/free-vector/medical-technology-concept-illustration_114360-7053.jpg", caption="Digital Workflow Concept", use_container_width=True)

# =====================================================
# 6. INFERENCE CENTER (ANALYSIS PAGE)
# =====================================================
elif menu == "Inference Center":
    # (โค้ดส่วนวิเคราะห์เดิมที่ปรับปรุงใหม่)
    c1, c2 = st.columns([1, 1])
    
    with c1:
        uploaded_file = st.file_uploader("Upload Image Scan", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True)
            
    with c2:
        if uploaded_file:
            with st.status("AI Processing...", expanded=True) as status:
                st.write("Extracting features...")
                time.sleep(1)
                st.write("Classifying tissue patterns...")
                time.sleep(1)
                status.update(label="Analysis Complete!", state="complete")
            
            # Mock Result
            risk_score = 78.5
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_score,
                gauge = {'bar': {'color': "#e74c3c" if risk_score > 50 else "#2ecc71"}}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.warning(f"**Potential Finding:** High probability of malignancy detected. Recommended: Biopsy at Segment 4.")
            
            if st.button("Save Result to Hospital Database", use_container_width=True):
                st.success(f"Sent to {st.session_state.user_data['hospital']} PACS System")
        else:
            st.info("Please upload a scan to start analysis.")

# (ส่วนอื่นๆ เช่น Analytics และ Settings สามารถเพิ่มต่อได้ตามความต้องการ)
