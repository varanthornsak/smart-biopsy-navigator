import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Pro",
    layout="wide"
)

# =====================================================
# SESSION INIT
# =====================================================
if "auth" not in st.session_state:
    st.session_state.auth = False

if "role" not in st.session_state:
    st.session_state.role = None

if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        "Patient", "Organ", "Status", "Confidence"
    ])

# =====================================================
# LOGIN
# =====================================================
if not st.session_state.auth:

    st.title("🧬 Smart Biopsy Pro")
    st.subheader("Secure Clinical Access")

    role = st.selectbox(
        "Select Role",
        ["Oncologist", "Radiologist", "Executive", "Admin"]
    )

    password = st.text_input("Enter Security Key", type="password")

    if st.button("Login Securely"):
        if password == "SNH_SECURE":
            st.session_state.auth = True
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid Security Key")

    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.markdown("## 🧬 Smart Biopsy Pro")
    st.markdown(f"**Role:** {st.session_state.role}")

    nav = st.radio(
        "Navigation",
        [
            "Diagnostic Hub",
            "Professional Analytics",
            "Executive Board View"
        ]
    )

    if st.button("Logout"):
        st.session_state.auth = False
        st.rerun()

# =====================================================
# DIAGNOSTIC HUB
# =====================================================
if nav == "Diagnostic Hub":

    st.title("Diagnostic Hub")

    col1, col2 = st.columns(2)

    with col1:
        patient = st.text_input("Patient Name")
        organ = st.selectbox("Organ", ["Liver", "Breast", "Thyroid"])
        marker = st.slider("Marker Level", 0, 500, 50)

        if st.button("Run AI Analysis"):

            if marker > 300:
                status = "MALIGNANT"
                confidence = 0.92
            elif marker > 100:
                status = "BENIGN"
                confidence = 0.55
            else:
                status = "NORMAL"
                confidence = 0.10

            new_row = pd.DataFrame([{
                "Patient": patient,
                "Organ": organ,
                "Status": status,
                "Confidence": confidence
            }])

            st.session_state.db = pd.concat(
                [st.session_state.db, new_row],
                ignore_index=True
            )

    with col2:

        if len(st.session_state.db) > 0:
            last = st.session_state.db.iloc[-1]

            st.metric("AI Status", last["Status"])
            st.metric("Confidence", f"{last['Confidence']*100:.1f}%")

# =====================================================
# PROFESSIONAL ANALYTICS
# =====================================================
elif nav == "Professional Analytics":

    st.title("Professional Clinical Dashboard")

    df = st.session_state.db

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cases", len(df))
    c2.metric("Malignancy %",
              f"{(df['Status']=='MALIGNANT').mean()*100 if len(df)>0 else 0:.1f}%")
    c3.metric("Avg Confidence",
              f"{df['Confidence'].mean()*100 if len(df)>0 else 0:.1f}%")

    if len(df) > 0:

        fig = px.pie(df, names="Status", hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# EXECUTIVE BOARD VIEW
# =====================================================
elif nav == "Executive Board View":

    st.title("Executive Business Intelligence")

    df = st.session_state.db

    total = len(df)
    malignant = (df["Status"] == "MALIGNANT").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Diagnoses", total)
    col2.metric("High Risk Cases", malignant)
    col3.metric("Projected Annual Savings", "฿18,000,000")

    st.markdown("---")

    st.subheader("Business Model")

    st.info("""
    • Hospital SaaS Subscription  
    • Per-case AI scoring  
    • Enterprise API Integration  
    • National Scale Deployment  
    """)

    st.success("AI Triage Layer for Oncology Diagnostics")
