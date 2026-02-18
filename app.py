import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import Table
import io

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Smart Biopsy Pro Enterprise",
                   layout="wide")

STATUS_COLOR = {
    "NORMAL": "#10B981",
    "BENIGN": "#F59E0B",
    "MALIGNANT": "#EF4444"
}

ROLES = ["Admin", "Clinician", "Radiologist", "Executive"]

# =====================================================
# DATABASE INIT
# =====================================================
def init_db():
    if "db" not in st.session_state:
        st.session_state.db = pd.DataFrame(columns=[
            "Date", "HN", "Patient", "Organ",
            "Status", "Confidence",
            "Marker_Val", "Tumor_Size"
        ])
    return st.session_state.db

db = init_db()

# =====================================================
# AUTH
# =====================================================
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.title("SMART BIOPSY PRO – ENTERPRISE AI")
    role = st.selectbox("Select Role", ROLES)
    pwd = st.text_input("Security Key", type="password")
    if st.button("LOGIN"):
        if pwd == "SNH_SECURE":
            st.session_state.auth = True
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid Key")
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown(f"### Role: {st.session_state.role}")
    nav = st.radio("Navigation", [
        "Diagnostic Hub",
        "Professional Analytics",
        "Executive Board View",
        "Case Archive",
        "User Manual"
    ])
    if st.button("Logout"):
        st.session_state.auth = False
        st.rerun()

# =====================================================
# AI LOGIC FUNCTION
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
# PDF REPORT FUNCTION
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

    col1, col2 = st.columns([1, 1.3])

    with col1:
        patient = st.text_input("Patient Name")
        hn = st.text_input("HN")
        organ = st.selectbox("Organ",
                             ["Liver", "Thyroid", "Breast", "Lymph Nodes"])
        file = st.file_uploader("Upload Image")

        if organ == "Liver":
            marker = st.number_input("AFP")
        elif organ == "Thyroid":
            marker = st.selectbox("TI-RADS", [1, 2, 3, 4, 5])
        elif organ == "Breast":
            marker = st.selectbox("BI-RADS", [1, 2, 3, 4, 5])
        else:
            marker = 0

        size = st.slider("Lesion Size (mm)", 1, 100, 10)

        if st.button("Run AI"):
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
                [st.session_state.db, new],
                ignore_index=True)

            st.success("Analysis Complete")

    with col2:
        if not st.session_state.db.empty:
            last = st.session_state.db.iloc[-1]
            color = STATUS_COLOR[last["Status"]]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=last["Confidence"] * 100,
                number={'suffix': "%"},
                gauge={'bar': {'color': color}}
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"## {last['Status']}")
            pdf = generate_pdf(
                last["Patient"],
                last["HN"],
                last["Organ"],
                last["Status"],
                last["Confidence"]
            )
            st.download_button("Download PDF Report",
                               pdf,
                               file_name="report.pdf")

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
        st.plotly_chart(
            px.pie(df, names="Status",
                   color="Status",
                   color_discrete_map=STATUS_COLOR),
            use_container_width=True)

        st.plotly_chart(
            px.bar(df, x="Organ",
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
    malignant = (df["Status"] == "MALIGNANT").sum()

    st.metric("Total Diagnoses", total)
    st.metric("High Risk Cases", malignant)
    st.metric("Estimated Quarterly Savings", "฿1,500,000")

    if total > 0:
        trend = df.groupby("Date").size().reset_index(name="Cases")
        st.plotly_chart(px.area(trend, x="Date", y="Cases"),
                        use_container_width=True)

# =====================================================
# ARCHIVE
# =====================================================
elif nav == "Case Archive":
    st.dataframe(st.session_state.db,
                 use_container_width=True)

# =====================================================
# USER MANUAL
# =====================================================
elif nav == "User Manual":

    st.title("Smart Biopsy Pro – Complete Operational Guide")

    st.markdown("""
# 1. System Overview
Smart Biopsy Pro is a Multi-Organ Clinical Decision Support System
integrating morphology and biomarker logic.

# 2. Organ Modules
- Liver → AFP logic
- Thyroid → TI-RADS
- Breast → BI-RADS prototype
- Lymph Nodes → Size-based malignancy logic

# 3. Risk Classification
- NORMAL (Green) → Routine follow-up
- BENIGN (Yellow) → Imaging surveillance
- MALIGNANT (Red) → Biopsy priority

# 4. Professional Dashboard
Tracks institutional risk load,
organ distribution,
AI confidence,
operational performance.

# 5. Executive Board View
Financial simulation,
cost saving estimate,
risk forecasting,
AI adoption growth.

# 6. Governance
System is decision support only.
Final diagnosis must be confirmed
by licensed physician.
""")
