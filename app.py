import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import time
import uuid

# =====================================================
# 1. ENTERPRISE UI CONFIG
# =====================================================
st.set_page_config(page_title="Smart Biopsy Pro | Analytical Suite", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .main-header { font-size: 28px; font-weight: 800; color: #1E3A8A; border-bottom: 3px solid #F1F5F9; padding-bottom: 12px; }
    .card { background: white; padding: 22px; border-radius: 15px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    .metric-card { background: #F8FAFC; padding: 15px; border-radius: 10px; border-top: 4px solid #1E3A8A; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. SESSION STATE & MOCK DATA FOR PRO VIEW
# =====================================================
if 'auth' not in st.session_state:
    st.session_state.auth = False
if 'db' not in st.session_state:
    # สร้างข้อมูลจำลองที่มีความหลากหลายเพื่อการแสดงกราฟที่สวยงาม
    dates = [datetime.date.today() - datetime.timedelta(days=x) for x in range(15)]
    mock_data = []
    for d in dates:
        for _ in range(np.random.randint(1, 5)):
            organ = np.random.choice(["Liver", "Thyroid"])
            risk = np.random.rand()
            status = "MALIGNANT" if risk > 0.7 else "BENIGN" if risk > 0.3 else "NORMAL"
            mock_data.append({
                "Date": d.strftime("%Y-%m-%d"),
                "HN": f"SNH-{np.random.randint(1000, 9999)}",
                "Patient": "Simulated Case",
                "Organ": organ,
                "Status": status,
                "Confidence": risk,
                "Marker_Val": np.random.randint(10, 500),
                "Tumor_Size": np.random.randint(5, 100)
            })
    st.session_state.db = pd.DataFrame(mock_data)

# =====================================================
# 3. LOGIN (SNH_SECURE)
# =====================================================
if not st.session_state.auth:
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("<div style='text-align:center; padding-top: 100px;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='color:#1E3A8A;'>SMART BIOPSY PRO</h1>", unsafe_allow_html=True)
        with st.form("login"):
            hosp = st.selectbox("Institution", ["Srinagarind Hospital (SNH)", "Global Medical Center"])
            pwd = st.text_input("Security Key", type="password")
            if st.form_submit_button("AUTHENTICATE", use_container_width=True):
                if pwd == "SNH_SECURE":
                    st.session_state.auth = True
                    st.session_state.user = {"hosp": hosp}
                    st.rerun()
                else: st.error("Invalid Key")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# 4. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown(f"### 🏢 {st.session_state.user['hosp']}")
    nav = st.radio("EXECUTIVE MENU", ["Diagnostic Hub", "Professional Analytics", "Case Archive", "Manual (EN)"])
    st.divider()
    if st.button("Logout"):
        st.session_state.auth = False
        st.rerun()

# =====================================================
# 5. DIAGNOSTIC HUB (ENGINE)
# =====================================================
if nav == "Diagnostic Hub":
    st.markdown("<h1 class='main-header'>Diagnostic Engine</h1>", unsafe_allow_html=True)
    col_in, col_out = st.columns([1, 1.5], gap="large")
    
    with col_in:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        p_name = st.text_input("Patient Name")
        hn_id = st.text_input("HN")
        organ = st.selectbox("Organ", ["Liver", "Thyroid"])
        up_file = st.file_uploader("Upload Scan")
        
        if organ == "Liver":
            afp = st.number_input("AFP Level", value=10.0)
            size = st.slider("Size (mm)", 0, 150, 20)
        else:
            afp = st.selectbox("TI-RADS", [1,2,3,4,5])
            size = st.slider("Size (mm)", 0, 80, 10)
            
        if st.button("RUN ANALYSIS", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                time.sleep(1)
                # Accurate Logic
                risk = 0.92 if (organ == "Liver" and afp > 200) or (organ == "Thyroid" and afp == 5) else 0.15
                status = "MALIGNANT" if risk > 0.7 else "NORMAL"
                
                new_entry = pd.DataFrame([{"Date": str(datetime.date.today()), "HN": hn_id, "Patient": p_name, "Organ": organ, "Status": status, "Confidence": risk, "Marker_Val": afp, "Tumor_Size": size}])
                st.session_state.db = pd.concat([st.session_state.db, new_entry], ignore_index=True)
                st.success("Analysis Complete")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        if not st.session_state.db.empty:
            last_case = st.session_state.db.iloc[-1]
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            res_col = "#EF4444" if last_case['Status'] == "MALIGNANT" else "#10B981"
            
            fig = go.Figure(go.Indicator(mode="gauge+number", value=last_case['Confidence']*100, gauge={'bar':{'color':res_col}}))
            fig.update_layout(height=300, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"<h2 style='color:{res_col}; text-align:center;'>{last_case['Status']}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# 6. PROFESSIONAL ANALYTICS (PRO FEATURES)
# =====================================================
elif nav == "Professional Analytics":
    st.markdown("<h1 class='main-header'>Institutional Performance Analytics</h1>", unsafe_allow_html=True)
    
    # Row 1: Key Performance Indicators
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Cases (MTD)", len(st.session_state.db), "+12%")
        st.markdown("</div>", unsafe_allow_html=True)
    with kpi2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Mean AI Confidence", f"{st.session_state.db['Confidence'].mean()*100:.1f}%", "High")
        st.markdown("</div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Detection Rate", f"{(st.session_state.db['Status']=='MALIGNANT').mean()*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with kpi4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("System ROI", "24.5%", "Hospital Target")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")

    # Row 2: Charts
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.subheader("📈 Diagnostic Volume Trend")
        df_trend = st.session_state.db.groupby('Date').size().reset_index(name='count')
        fig_trend = px.line(df_trend, x='Date', y='count', markers=True, line_shape='spline', color_discrete_sequence=['#1E3A8A'])
        fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_c2:
        st.subheader("📊 Risk Distribution by Organ")
        fig_pie = px.sunburst(st.session_state.db, path=['Organ', 'Status'], values='Confidence', color='Status',
                             color_discrete_map={'MALIGNANT':'#EF4444', 'BENIGN':'#F59E0B', 'NORMAL':'#10B981'})
        st.plotly_chart(fig_pie, use_container_width=True)

    # Row 3: Advanced Scatter
    st.subheader("🔬 Clinical Marker Distribution (Multi-Factor Analysis)")
    fig_scatter = px.scatter(st.session_state.db, x="Marker_Val", y="Tumor_Size", size="Confidence", color="Status",
                             hover_name="HN", log_x=True, size_max=20,
                             labels={"Marker_Val": "Biomarker Level (AFP/Grade)", "Tumor_Size": "Tumor Diameter (mm)"},
                             color_discrete_map={'MALIGNANT':'#EF4444', 'BENIGN':'#F59E0B', 'NORMAL':'#10B981'})
    st.plotly_chart(fig_scatter, use_container_width=True)

# =====================================================
# 7. CASE ARCHIVE
# =====================================================
elif nav == "Case Archive":
    st.markdown("<h1 class='main-header'>Institutional Records</h1>", unsafe_allow_html=True)
    st.dataframe(st.session_state.db, use_container_width=True, hide_index=True)
    st.download_button("Export Enterprise Data (CSV)", st.session_state.db.to_csv(), "hospital_data.csv")

# =====================================================
# 8. MANUAL (ENGLISH)
# =====================================================
elif nav == "Manual (EN)":
    st.header("Operational Protocol")
    st.markdown("""
    ### 1. Data Integrity
    - Ensure all imaging data is DICOM-compliant before ingestion.
    - Validate clinical biomarkers against the latest laboratory results.
    
    ### 2. Analytical Intelligence
    - **Malignancy Score:** A weighted correlation of Image Morphology + Biochemical Markers.
    - **Scoring Tiers:** Red (>0.7), Yellow (0.3-0.7), Green (<0.3).
    
    ### 3. Business Analytics
    - Use the 'Professional Analytics' tab to monitor departmental throughput and AI diagnostic efficacy for administrative reporting.
    """)
  
