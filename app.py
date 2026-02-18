import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io
import numpy as np
import os
import hashlib

# =====================================================
# PAGE CONFIG (ต้องมีครั้งเดียว)
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Pro | Enterprise Multi-Organ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SAFE SESSION INIT
# =====================================================
if "auth" not in st.session_state:
    st.session_state.auth = False

if "role" not in st.session_state:
    st.session_state.role = None

if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        "Date","HN","Patient","Organ",
        "Status","Confidence","Marker_Val","Tumor_Size"
    ])

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# =====================================================
# CONSTANTS
# =====================================================
STATUS_COLOR = {
    "NORMAL": "#10B981",
    "BENIGN": "#F59E0B",
    "MALIGNANT": "#EF4444"
}

ROLES = ["Admin","Clinician","Radiologist","Executive"]

# =====================================================
# LOGIN
# =====================================================
if not st.session_state.auth:

    st.title("SMART BIOPSY PRO – ENTERPRISE AI")

    role = st.selectbox("Select Role", ROLES, key="login_role")
    pwd = st.text_input("Security Key", type="password", key="login_pwd")

    if st.button("LOGIN", key="login_btn"):
        if pwd == "SNH_SECURE":
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

    st.markdown("## Smart Biopsy Pro")

    if st.session_state.role:
        st.markdown(f"**Role:** {st.session_state.role}")

    nav = st.radio(
        "Navigation",
        [
            "Diagnostic Hub",
            "Professional Analytics",
            "Executive Board View",
            "Case Archive",
            "User Manual",
            "VC AI Module"
        ],
        key="main_nav"
    )

    if st.button("Logout", key="sidebar_logout"):
        st.session_state.auth = False
        st.session_state.role = None
        st.rerun()

# =====================================================
# AI LOGIC
# =====================================================
def run_ai(organ, marker, size):

    if organ == "Liver":
        if marker > 400 or (marker > 200 and size > 50):
            return "MALIGNANT", 0.92
        elif marker > 20 or size > 30:
            return "BENIGN", 0.55
        else:
            return "NORMAL", 0.08

    elif organ == "Thyroid":
        if marker >= 5 or (marker == 4 and size > 25):
            return "MALIGNANT", 0.90
        elif marker == 4 or size > 15:
            return "BENIGN", 0.50
        else:
            return "NORMAL", 0.07

    elif organ == "Breast":
        if marker >= 5:
            return "MALIGNANT", 0.88
        elif marker == 4:
            return "BENIGN", 0.60
        else:
            return "NORMAL", 0.10

    elif organ == "Lymph Nodes":
        if size > 30:
            return "MALIGNANT", 0.85
        elif size > 15:
            return "BENIGN", 0.55
        else:
            return "NORMAL", 0.12

# =====================================================
# PDF GENERATOR
# =====================================================
def generate_pdf(patient, hn, organ, status, confidence):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>SMART BIOPSY PRO REPORT</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Patient: {patient}", styles["Normal"]))
    elements.append(Paragraph(f"HN: {hn}", styles["Normal"]))
    elements.append(Paragraph(f"Organ: {organ}", styles["Normal"]))
    elements.append(Paragraph(f"Status: {status}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence*100:.1f}%", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# DIAGNOSTIC HUB
# =====================================================
if nav == "Diagnostic Hub":

    st.title("Diagnostic Decision Engine – Multi Organ")

    col1, col2 = st.columns([1,1.3])

    with col1:
        patient = st.text_input("Patient Name", key="dh_patient")
        hn = st.text_input("HN", key="dh_hn")
        organ = st.selectbox("Organ",
                             ["Liver","Thyroid","Breast","Lymph Nodes"],
                             key="dh_organ")

        if organ == "Liver":
            marker = st.number_input("AFP",0.0,value=10.0,key="dh_marker")
        elif organ == "Thyroid":
            marker = st.selectbox("TI-RADS",[1,2,3,4,5],key="dh_marker")
        elif organ == "Breast":
            marker = st.selectbox("BI-RADS",[1,2,3,4,5],key="dh_marker")
        else:
            marker = 0

        size = st.slider("Lesion Size (mm)",1,100,10,key="dh_size")

        if st.button("Run AI Analysis", key="dh_run"):

            if patient and hn:
                status, confidence = run_ai(organ, marker, size)

                new = pd.DataFrame([{
                    "Date": str(datetime.date.today()),
                    "HN": hn,
                    "Patient": patient,
                    "Organ": organ,
                    "Status": status,
                    "Confidence": confidence,
                    "Marker_Val": marker,
                    "Tumor_Size": size
                }])

                st.session_state.db = pd.concat(
                    [st.session_state.db,new],
                    ignore_index=True
                )

                st.success("Analysis Complete")

    with col2:
        if len(st.session_state.db)>0:
            last = st.session_state.db.iloc[-1]
            color = STATUS_COLOR[last["Status"]]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=last["Confidence"]*100,
                number={'suffix':"%"},
                gauge={'axis':{'range':[0,100]},
                       'bar':{'color':color}}
            ))

            st.plotly_chart(fig,use_container_width=True)
            st.markdown(f"## {last['Status']}")

            pdf = generate_pdf(
                last["Patient"],
                last["HN"],
                last["Organ"],
                last["Status"],
                last["Confidence"]
            )

            st.download_button(
                "Download PDF Report",
                pdf,
                file_name="Smart_Biopsy_Report.pdf",
                key="dh_pdf"
            )

# =====================================================
# PROFESSIONAL ANALYTICS
# =====================================================
elif nav == "Professional Analytics":

    st.title("Professional Clinical Dashboard")

    df = st.session_state.db

    c1,c2,c3 = st.columns(3)

    c1.metric("Total Cases",len(df))
    c2.metric("Malignancy %",
              f"{(df['Status']=='MALIGNANT').mean()*100 if len(df)>0 else 0:.1f}%")
    c3.metric("Avg Confidence",
              f"{df['Confidence'].mean()*100 if len(df)>0 else 0:.1f}%")

    if len(df)>0:
        st.plotly_chart(
            px.pie(df,names="Status",
                   color="Status",
                   color_discrete_map=STATUS_COLOR),
            use_container_width=True)

# =====================================================
# EXECUTIVE VIEW
# =====================================================
elif nav == "Executive Board View":

    st.title("Executive Business Intelligence")

    df = st.session_state.db
    total = len(df)
    malignant = (df["Status"]=="MALIGNANT").sum()

    st.metric("Total Diagnoses",total)
    st.metric("High Risk Cases",malignant)
    st.metric("Projected Quarterly Savings","฿1,500,000")

# =====================================================
# CASE ARCHIVE
# =====================================================
elif nav == "Case Archive":
    st.dataframe(st.session_state.db,use_container_width=True)

# =====================================================
# USER MANUAL
# =====================================================
elif nav == "User Manual":

    st.title("Smart Biopsy Pro – Detailed Manual")

    st.markdown("""
Multi-Organ AI Decision Support System  
Liver, Thyroid, Breast, Lymph Nodes  

NORMAL → Routine follow-up  
BENIGN → Imaging surveillance  
MALIGNANT → Biopsy priority  

Decision support only.  
Final diagnosis by licensed physician.
""")

# =====================================================
# VC AI MODULE (SECOND SYSTEM FIXED – NO DUPLICATE)
# =====================================================
elif nav == "VC AI Module":

    st.title("VC Production AI Module")

    if st.button("Run VC Demo AI", key="vc_run"):
        prob = np.random.uniform(0.4,0.95)
        pred = "Malignant" if prob>0.6 else "Benign"

        st.session_state.audit_log.append(
            f"{datetime.datetime.now()} VC AI run by {st.session_state.role}"
        )

        if pred=="Malignant":
            st.error(f"Malignant ({prob:.2%})")
        else:
            st.success(f"Benign ({prob:.2%})")
