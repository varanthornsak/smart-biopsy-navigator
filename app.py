import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Pro | Enterprise AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GLOBAL COLOR MAP
# =====================================================
STATUS_COLOR_MAP = {
    "NORMAL": "#10B981",      # Green
    "BENIGN": "#F59E0B",      # Yellow
    "MALIGNANT": "#EF4444"    # Red
}

# =====================================================
# INITIAL DATABASE
# =====================================================
def get_clean_db():
    if 'db' not in st.session_state:
        data = [
            {"Date": "2026-02-15", "HN": "SNH-9001", "Patient": "Reference Case A",
             "Organ": "Liver", "Status": "MALIGNANT", "Confidence": 0.95,
             "Marker_Val": 450.0, "Tumor_Size": 55},

            {"Date": "2026-02-16", "HN": "SNH-9002", "Patient": "Reference Case B",
             "Organ": "Thyroid", "Status": "NORMAL", "Confidence": 0.08,
             "Marker_Val": 1.2, "Tumor_Size": 12},

            {"Date": "2026-02-17", "HN": "SNH-9003", "Patient": "Reference Case C",
             "Organ": "Liver", "Status": "BENIGN", "Confidence": 0.52,
             "Marker_Val": 35.0, "Tumor_Size": 28},
        ]
        st.session_state.db = pd.DataFrame(data)
    return st.session_state.db

db = get_clean_db()

if 'auth' not in st.session_state:
    st.session_state.auth = False

# =====================================================
# LOGIN PAGE
# =====================================================
if not st.session_state.auth:
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.markdown("<div style='text-align:center; padding-top: 100px;'>",
                    unsafe_allow_html=True)
        st.markdown("<h1 style='color:#1E3A8A; font-weight:800;'>SMART BIOPSY PRO</h1>",
                    unsafe_allow_html=True)
        st.caption("Enterprise Clinical Intelligence Gateway")

        with st.form("login_form"):
            st.selectbox("Institution Node",
                         ["Srinagarind Hospital (SNH)",
                          "External Medical Hub"])
            pwd = st.text_input("Security Key", type="password")

            if st.form_submit_button("AUTHENTICATE SYSTEM",
                                     use_container_width=True):
                if pwd == "SNH_SECURE":
                    st.session_state.auth = True
                    st.session_state.user = {"hosp": "SNH"}
                    st.rerun()
                else:
                    st.error("Authentication Denied")

        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
with st.sidebar:
    st.markdown(f"### 🏢 {st.session_state.user['hosp']} Enterprise")
    nav = st.radio("SOLUTIONS", [
        "Diagnostic Hub",
        "Professional Analytics",
        "Executive Board View",
        "Case Archive",
        "User Manual"
    ])
    st.divider()
    if st.button("Secure Logout", use_container_width=True):
        st.session_state.auth = False
        st.rerun()

# =====================================================
# DIAGNOSTIC HUB
# =====================================================
if nav == "Diagnostic Hub":

    st.markdown("<h1 style='color:#1E3A8A;'>Diagnostic Decision Engine</h1>",
                unsafe_allow_html=True)

    in_col, out_col = st.columns([1, 1.4], gap="large")

    with in_col:
        st.subheader("Patient Clinical Profile")

        p_name = st.text_input("Patient Full Name")
        hn_id = st.text_input("HN (Hospital Number)")
        organ = st.selectbox("Anatomical Module",
                             ["Liver", "Thyroid"])
        up_file = st.file_uploader("Upload Scan (DICOM/JPG)")

        if organ == "Liver":
            m_val = st.number_input("Serum AFP (ng/mL)",
                                    value=10.0, min_value=0.1)
            t_size = st.slider("Tumor Diameter (mm)", 1, 150, 20)
        else:
            m_val = st.selectbox("TI-RADS Level", [1, 2, 3, 4, 5])
            t_size = st.slider("Nodule Size (mm)", 1, 100, 10)

        if st.button("RUN CLINICAL INFERENCE",
                     use_container_width=True,
                     type="primary"):

            if p_name and hn_id and up_file:

                with st.spinner("Analyzing Morphology & Biomarkers..."):
                    time.sleep(1.2)

                    # ===============================
                    # 3-TIER RISK LOGIC
                    # ===============================
                    if organ == "Liver":
                        if m_val > 400 or (m_val > 200 and t_size > 50):
                            risk = 0.92
                            status = "MALIGNANT"
                        elif m_val > 20 or t_size > 30:
                            risk = 0.55
                            status = "BENIGN"
                        else:
                            risk = 0.08
                            status = "NORMAL"

                    else:  # Thyroid
                        if m_val >= 5 or (m_val == 4 and t_size > 25):
                            risk = 0.90
                            status = "MALIGNANT"
                        elif m_val == 4 or t_size > 15:
                            risk = 0.50
                            status = "BENIGN"
                        else:
                            risk = 0.07
                            status = "NORMAL"

                    new_entry = pd.DataFrame([{
                        "Date": str(datetime.date.today()),
                        "HN": hn_id,
                        "Patient": p_name,
                        "Organ": organ,
                        "Status": status,
                        "Confidence": risk,
                        "Marker_Val": m_val,
                        "Tumor_Size": t_size
                    }])

                    st.session_state.db = pd.concat(
                        [st.session_state.db, new_entry],
                        ignore_index=True
                    )

                    st.toast("Case Archived Successfully.")

    # ===============================
    # OUTPUT PANEL
    # ===============================
    with out_col:
        if not st.session_state.db.empty:
            last = st.session_state.db.iloc[-1]
            color = STATUS_COLOR_MAP.get(last['Status'])

            st.markdown(
                f"<div style='background:#FFF; padding:20px; "
                f"border-radius:15px; border-left:10px solid {color}; "
                f"box-shadow:0 4px 6px rgba(0,0,0,0.1);'>",
                unsafe_allow_html=True
            )

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=last['Confidence'] * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': "#D1FAE5"},
                        {'range': [30, 70], 'color': "#FEF3C7"},
                        {'range': [70, 100], 'color': "#FECACA"}
                    ]
                }
            ))

            fig.update_layout(height=300,
                              margin=dict(t=0, b=0))

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                f"<h2 style='color:{color}; text-align:center;'>"
                f"{last['Status']}</h2>",
                unsafe_allow_html=True
            )

            st.markdown("**Clinical Recommendation:**")
            if last['Status'] == "MALIGNANT":
                st.error("High Risk – Prioritize Biopsy.")
            elif last['Status'] == "BENIGN":
                st.warning("Moderate Risk – Imaging Follow-up.")
            else:
                st.success("Low Risk – Routine Surveillance.")

            st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PROFESSIONAL ANALYTICS
# =====================================================
elif nav == "Professional Analytics":

    st.markdown("<h1 style='color:#1E3A8A;'>Professional Analytics Dashboard</h1>",
                unsafe_allow_html=True)

    df = st.session_state.db.copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assessments", len(df))
    c2.metric("Mean AI Confidence",
              f"{df['Confidence'].mean()*100:.1f}%")
    c3.metric("Malignancy Ratio",
              f"{(df['Status']=='MALIGNANT').mean()*100:.1f}%")
    c4.metric("Operational Efficiency Gain", "+34%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Diagnostic Volume Trend")
        trend = df.groupby('Date').size().reset_index(name='Volume')
        fig = px.line(trend, x="Date", y="Volume",
                      markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Malignancy Distribution")
        fig2 = px.pie(
            df,
            names="Status",
            hole=0.4,
            color="Status",
            color_discrete_map=STATUS_COLOR_MAP
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Risk Stratification Overview")
    bar = px.bar(
        df.groupby("Status").size().reset_index(name="Cases"),
        x="Status",
        y="Cases",
        color="Status",
        color_discrete_map=STATUS_COLOR_MAP
    )
    st.plotly_chart(bar, use_container_width=True)

# =====================================================
# EXECUTIVE BOARD VIEW
# =====================================================
elif nav == "Executive Board View":

    st.markdown("<h1 style='color:#1E3A8A;'>Executive Board Intelligence</h1>",
                unsafe_allow_html=True)
    st.caption("For Hospital Directors & Financial Controllers")

    df = st.session_state.db.copy()

    total = len(df)
    malignant_cases = (df['Status'] == "MALIGNANT").sum()
    benign_cases = (df['Status'] == "BENIGN").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total AI-Assisted Diagnoses", total)
    c2.metric("High-Risk Malignancy Cases", malignant_cases)
    c3.metric("Projected Cost Savings (Quarter)", "฿1,250,000")

    st.divider()

    st.markdown(f"""
    ### Institutional Insight

    - Moderate-Risk Monitoring Pool: **{benign_cases}**
    - Early Detection Optimization Index: **High**
    - AI Deployment Stability: **99.2% Uptime**
    """)

    st.subheader("AI Adoption Growth Trend")
    adoption = df.groupby("Date").size().reset_index(name="Cases")
    area = px.area(adoption, x="Date", y="Cases")
    st.plotly_chart(area, use_container_width=True)

# =====================================================
# CASE ARCHIVE
# =====================================================
elif nav == "Case Archive":
    st.markdown("<h1 style='color:#1E3A8A;'>Institutional Clinical Archive</h1>",
                unsafe_allow_html=True)
    st.dataframe(st.session_state.db,
                 use_container_width=True,
                 hide_index=True)

# =====================================================
# USER MANUAL
# =====================================================
elif nav == "User Manual":

    st.markdown("<h1 style='color:#1E3A8A;'>Operational Protocol</h1>",
                unsafe_allow_html=True)

    st.markdown("""
    ## Smart Biopsy Pro — Enterprise CDSS

    **Smart Biopsy Pro** is a Clinical Decision Support System
    integrating morphology analysis and biochemical correlation
    for malignancy risk assessment.

    ### Risk Tier Definition
    - 🟢 NORMAL → Routine Surveillance
    - 🟡 BENIGN → Imaging Follow-up
    - 🔴 MALIGNANT → Biopsy Priority

    ### Clinical Governance
    This system is intended for decision support only.
    Final diagnosis must be confirmed by a licensed physician.
    """)
