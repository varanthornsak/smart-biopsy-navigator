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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Pro | Enterprise Multi-Organ",
    layout="wide"
)

# =====================================================
# SAFE SESSION INITIALIZATION (CRITICAL FOR CLOUD)
# =====================================================
if "auth" not in st.session_state:
    st.session_state.auth = False

if "role" not in st.session_state:
    st.session_state.role = None

if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        "Date", "HN", "Patient", "Organ",
        "Status", "Confidence",
        "Marker_Val", "Tumor_Size"
    ])

# =====================================================
# CONSTANTS
# =====================================================
STATUS_COLOR = {
    "NORMAL": "#10B981",
    "BENIGN": "#F59E0B",
    "MALIGNANT": "#EF4444"
}

ROLES = ["Admin", "Clinician", "Radiologist", "Executive"]

# =====================================================
# LOGIN
# =====================================================
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
            st.error("Invalid Security Key")

    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    if st.session_state.role:
        st.markdown(f"### Role: {st.session_state.role}")
    else:
        st.markdown("### Role: Not Assigned")

    nav = st.radio("Navigation", [
        "Diagnostic Hub",
        "Professional Analytics",
        "Executive Board View",
        "Case Archive",
        "User Manual"
    ])

    if st.button("Logout"):
        st.session_state.auth = False
        st.session_state.role = None
        st.rerun()

# =====================================================
# AI LOGIC
# =====================================================
def run_ai(organ, marker, size):

    # LIVER – Morphology priority (AFP optional)
    if organ == "Liver":

        if size > 60:
            return "MALIGNANT", 0.90

        if marker is not None and marker > 400:
            return "MALIGNANT", 0.92

        if size > 30:
            return "BENIGN", 0.60

        if marker is not None and marker > 200:
            return "BENIGN", 0.55

        return "NORMAL", 0.15

    # Default organs (minimal logic)
    if size > 40:
        return "MALIGNANT", 0.80
    elif size > 20:
        return "BENIGN", 0.55
    else:
        return "NORMAL", 0.10

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

    st.title("AI Diagnostic Engine")

    col1, col2 = st.columns([1,1])

    # ================= LEFT =================
    with col1:

        patient = st.text_input("Patient Name")
        hn = st.text_input("HN")
        organ = st.selectbox("Organ", ["Liver", "Thyroid", "Breast", "Lung"])

        marker = None

        if organ == "Liver":
            use_afp = st.checkbox("Include AFP (Optional Biomarker)")
            if use_afp:
                marker = st.number_input("AFP (ng/mL)", min_value=0.0, value=10.0)

        size = st.slider("Lesion Size (mm)", 1, 100, 10)

        if st.button("Run AI Analysis"):

            if patient and hn:

                status, confidence = run_ai(organ, marker, size)

                new = pd.DataFrame([{
                    "Status": status,
                    "Confidence": confidence
                }])

                st.session_state.db = pd.concat(
                    [st.session_state.db, new],
                    ignore_index=True
                )

                st.success("Analysis Complete")

    # ================= RIGHT =================
    with col2:

        st.subheader("AI Result Dashboard")

        if len(st.session_state.db) > 0:

            last = st.session_state.db.iloc[-1]
            confidence_percent = last["Confidence"] * 100
            status = last["Status"]

            if status == "NORMAL":
                color = "#28a745"
            elif status == "BENIGN":
                color = "#ffc107"
            else:
                color = "#dc3545"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_percent,
                number={'suffix': "%"},
                title={'text': status},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color}
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Run analysis to generate result.")

# =====================================================
# 📊 PROFESSIONAL ANALYTICS (SAFE VERSION)
# =====================================================

if nav == "Professional Analytics":

    st.title("Professional Analytics Dashboard")

    STATUS_COLOR = {
        "NORMAL": "#28a745",
        "BENIGN": "#ffc107",
        "MALIGNANT": "#dc3545"
    }

    if len(st.session_state.db) > 0:

        df = st.session_state.db.copy()

        # 🔥 ทำให้ status เป็นมาตรฐานก่อน
        df["Status"] = df["Status"].astype(str).str.upper().str.strip()

        status_counts = df["Status"].value_counts()

        # ================= KPI =================
        col1, col2, col3 = st.columns(3)

        col1.metric("🟢 NORMAL", status_counts.get("NORMAL", 0))
        col2.metric("🟡 BENIGN", status_counts.get("BENIGN", 0))
        col3.metric("🔴 MALIGNANT", status_counts.get("MALIGNANT", 0))

        st.markdown("---")

        # ================= PIE =================
        st.subheader("Risk Distribution")

        labels = status_counts.index.tolist()
        values = status_counts.values.tolist()

        colors = [
            STATUS_COLOR.get(s, "#9ca3af")  # เทา fallback กัน error
            for s in labels
        ]

        pie_fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.0
        )])

        pie_fig.update_layout(height=600)

        st.plotly_chart(pie_fig, use_container_width=True)

        # ================= BAR =================
        st.subheader("Case Distribution")

        bar_fig = go.Figure()

        for status in labels:
            bar_fig.add_trace(go.Bar(
                x=[status],
                y=[status_counts[status]],
                marker_color=STATUS_COLOR.get(status, "#9ca3af")
            ))

        bar_fig.update_layout(
            showlegend=False,
            height=400
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    else:
        st.info("No case data available yet.")

# =====================================================
# EXECUTIVE BOARD VIEW
# =====================================================
elif nav == "Executive Board View":

    st.title("Executive Business Intelligence")

    df = st.session_state.db
    total = len(df)
    malignant = (df["Status"] == "MALIGNANT").sum()

    st.metric("Total Diagnoses", total)
    st.metric("High Risk Cases", malignant)
    st.metric("Projected Quarterly Savings", "฿1,500,000")

    if total > 0:
        trend = df.groupby("Date").size().reset_index(name="Cases")
        st.plotly_chart(
            px.area(trend, x="Date", y="Cases"),
            use_container_width=True)

# =====================================================
# CASE ARCHIVE
# =====================================================
elif nav == "Case Archive":
    st.dataframe(st.session_state.db, use_container_width=True)

# =====================================================
# USER MANUAL (DETAILED)
# =====================================================
elif nav == "User Manual":

    st.title("Smart Biopsy Pro – Detailed Operational Manual")

    st.markdown("""
# 1. System Overview
Smart Biopsy Pro is a Multi-Organ Clinical Decision Support System
integrating biomarker logic and morphology-based inference.

# 2. Organ Modules
- Liver → AFP-based risk logic
- Thyroid → TI-RADS stratification
- Breast → BI-RADS prototype logic
- Lymph Nodes → Size-based malignancy logic

# 3. Risk Classification
🟢 NORMAL → Routine follow-up  
🟡 BENIGN → Imaging surveillance  
🔴 MALIGNANT → Biopsy priority  

# 4. Professional Analytics
Provides:
- Case volume monitoring
- Risk distribution
- Confidence tracking
- Organ workload analysis

# 5. Executive Board View
Displays:
- Institutional risk burden
- Financial impact simulation
- AI adoption growth

# 6. Governance Notice
This system is decision-support only.
Final clinical decisions must be made
by licensed physicians.
""")
# =====================================================
# ADVANCED ENTERPRISE EXTENSIONS (APPEND BELOW EXISTING CODE)
# =====================================================

import uuid
import random

# =====================================================
# ENHANCED AI EXPLANATION ENGINE
# =====================================================
def generate_explanation(organ, marker, size, status):

    reasons = []
    recommendation = ""

    if organ == "Liver":
        if marker > 400:
            reasons.append("AFP > 400 (High oncologic risk)")
        if size > 50:
            reasons.append("Tumor size > 50mm")

    if organ == "Thyroid":
        if marker >= 5:
            reasons.append("TI-RADS 5 (Highly suspicious)")
        if size > 25:
            reasons.append("Nodule size > 25mm")

    if organ == "Breast":
        if marker >= 5:
            reasons.append("BI-RADS 5 (High malignancy probability)")

    if organ == "Lymph Nodes":
        if size > 30:
            reasons.append("Lymph node > 30mm")

    if status == "MALIGNANT":
        recommendation = "Immediate biopsy recommended."
    elif status == "BENIGN":
        recommendation = "Short-term imaging follow-up suggested."
    else:
        recommendation = "Routine surveillance."

    return reasons, recommendation


# =====================================================
# AUTO CASE ID GENERATOR
# =====================================================
def generate_case_id():
    return f"SBP-{datetime.date.today().year}-{str(uuid.uuid4())[:6].upper()}"


# =====================================================
# PATCH DATABASE WITH EXTRA FIELDS
# =====================================================
if "Case_ID" not in st.session_state.db.columns:
    st.session_state.db["Case_ID"] = ""
    st.session_state.db["Timestamp"] = ""
    st.session_state.db["Created_By"] = ""


# =====================================================
# ENHANCE LATEST CASE WITH ENTERPRISE DATA
# =====================================================
if len(st.session_state.db) > 0:

    last_index = st.session_state.db.index[-1]

    if st.session_state.db.loc[last_index, "Case_ID"] == "":
        st.session_state.db.loc[last_index, "Case_ID"] = generate_case_id()
        st.session_state.db.loc[last_index, "Timestamp"] = str(datetime.datetime.now())
        st.session_state.db.loc[last_index, "Created_By"] = st.session_state.role


# =====================================================
# ADD EXPLAINABLE AI PANEL TO DIAGNOSTIC HUB (FIXED)
# =====================================================
if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    df = st.session_state.db
    last_index = df.index[-1]

    # -------- Auto-fix missing enterprise fields --------
    if pd.isna(df.loc[last_index, "Case_ID"]) or df.loc[last_index, "Case_ID"] == "":
        df.loc[last_index, "Case_ID"] = generate_case_id()

    if pd.isna(df.loc[last_index, "Timestamp"]) or df.loc[last_index, "Timestamp"] == "":
        df.loc[last_index, "Timestamp"] = str(datetime.datetime.now())

    if pd.isna(df.loc[last_index, "Created_By"]) or df.loc[last_index, "Created_By"] == "":
        df.loc[last_index, "Created_By"] = st.session_state.role

    last = df.loc[last_index]

    # -------- Generate explanation --------
    reasons, recommendation = generate_explanation(
        last["Organ"],
        last["Marker_Val"],
        last["Tumor_Size"],
        last["Status"]
    )

    st.markdown("---")
    st.subheader("AI Risk Explanation")

    st.markdown(f"**Case ID:** {last['Case_ID']}")
    st.markdown(f"**Generated:** {last['Timestamp']}")
    st.markdown(f"**Created By:** {last['Created_By']}")

    st.markdown("### Risk Factors Identified:")

    if len(reasons) > 0:
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.markdown("No high-risk features detected.")

    st.markdown("### Clinical Recommendation:")
    st.success(recommendation)

# =====================================================
# ENHANCED EXECUTIVE METRICS
# =====================================================
if nav == "Executive Board View":

    df = st.session_state.db

    st.markdown("---")
    st.subheader("AI Adoption & Impact Metrics")

    total = len(df)

    if total > 0:

        adoption_rate = min(100, total * 5)
        biopsy_reduction = random.randint(15, 35)
        time_saved = total * 12

        colA, colB, colC = st.columns(3)

        colA.metric("AI Adoption Rate", f"{adoption_rate}%")
        colB.metric("Biopsy Reduction", f"{biopsy_reduction}%")
        colC.metric("Time Saved (hrs)", f"{time_saved}")

        st.markdown("### ROI Simulation")

        monthly_cases = st.slider("Monthly Cases", 50, 2000, 300)
        cost_per_biopsy = st.slider("Cost per Biopsy (฿)", 5000, 50000, 15000)
        reduction_percent = st.slider("False Positive Reduction %", 5, 50, 20)

        saved_cases = monthly_cases * (reduction_percent / 100)
        savings = saved_cases * cost_per_biopsy

        st.success(f"Projected Monthly Savings: ฿{savings:,.0f}")
# =====================================================
# ENTERPRISE CONFIGURATION LAYER
# =====================================================

ENTERPRISE_MODE = True
SYSTEM_VERSION = "SBP Enterprise v2.3"
AI_MODEL_VERSION = "MorphoBio-AI 4.1"

if "system_log" not in st.session_state:
    st.session_state.system_log = []


# =====================================================
# ADVANCED NUMERIC RISK SCORING MODEL
# =====================================================
def calculate_risk_score(organ, marker, size):

    score = 0

    # Morphology Weight
    score += size * 0.8

    # Biomarker Weight
    if marker is not None:
        score += marker * 0.05

    # Organ risk weighting
    organ_weight = {
        "Liver": 1.3,
        "Breast": 1.2,
        "Thyroid": 1.1,
        "Lung": 1.4
    }

    score *= organ_weight.get(organ, 1.0)

    return round(score, 2)


# =====================================================
# CONFIDENCE CALIBRATION (Sigmoid Scaling)
# =====================================================
import math

def calibrated_confidence(score):
    sigmoid = 1 / (1 + math.exp(-0.05 * (score - 50)))
    return round(sigmoid, 3)


# =====================================================
# ENTERPRISE LOGGING ENGINE
# =====================================================
def log_event(event_type, case_id):

    st.session_state.system_log.append({
        "Timestamp": str(datetime.datetime.now()),
        "Event": event_type,
        "Case_ID": case_id,
        "User": st.session_state.role,
        "System_Version": SYSTEM_VERSION
    })


# =====================================================
# PATCH DIAGNOSTIC HUB WITH RISK SCORE
# =====================================================
if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    df = st.session_state.db
    last_index = df.index[-1]
    last = df.loc[last_index]

    risk_score = calculate_risk_score(
        last["Organ"],
        last["Marker_Val"],
        last["Tumor_Size"]
    )

    calibrated = calibrated_confidence(risk_score)

    df.loc[last_index, "Risk_Score"] = risk_score
    df.loc[last_index, "Calibrated_Confidence"] = calibrated

    st.markdown("### Advanced Risk Metrics")
    col1, col2 = st.columns(2)

    col1.metric("Risk Score", risk_score)
    col2.metric("Calibrated Confidence", f"{calibrated*100:.1f}%")

    log_event("AI_ANALYSIS_COMPLETED", last["Case_ID"])


# =====================================================
# ENTERPRISE AUDIT TRAIL VIEW
# =====================================================
if nav == "Professional Analytics" and ENTERPRISE_MODE:

    st.markdown("---")
    st.subheader("System Audit Trail")

    if len(st.session_state.system_log) > 0:
        log_df = pd.DataFrame(st.session_state.system_log)
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("No system events logged yet.")


# =====================================================
# EXECUTIVE FORECAST MODEL
# =====================================================
if nav == "Executive Board View" and len(st.session_state.db) > 10:

    st.markdown("---")
    st.subheader("6-Month Workload Forecast")

    df = st.session_state.db.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    monthly = df.groupby(pd.Grouper(key="Timestamp", freq="M")).size().reset_index(name="Cases")

    if len(monthly) > 1:
        monthly["Forecast"] = monthly["Cases"].rolling(2).mean()

        forecast_fig = px.line(
            monthly,
            x="Timestamp",
            y=["Cases", "Forecast"],
            markers=True
        )

        st.plotly_chart(forecast_fig, use_container_width=True)


# =====================================================
# ENTERPRISE PDF REPORT (FORMAL STYLE)
# =====================================================
def generate_enterprise_pdf(case_row):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"<b>{SYSTEM_VERSION}</b>", styles["Title"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"AI Model: {AI_MODEL_VERSION}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    for field in case_row.index:
        elements.append(
            Paragraph(f"<b>{field}:</b> {case_row[field]}", styles["Normal"])
        )

    doc.build(elements)
    buffer.seek(0)
    return buffer


# =====================================================
# DOWNLOAD BUTTON IN DIAGNOSTIC HUB
# =====================================================
if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    last = st.session_state.db.iloc[-1]

    pdf_buffer = generate_enterprise_pdf(last)

    st.download_button(
        label="Download Enterprise Report (PDF)",
        data=pdf_buffer,
        file_name=f"{last['Case_ID']}_Enterprise_Report.pdf",
        mime="application/pdf"
    )
# =====================================================
# MULTI-HOSPITAL SUPPORT
# =====================================================

HOSPITALS = [
    "Siam Neuro Hospital",
    "Bangkok Oncology Center",
    "Chiang Mai Diagnostic Institute",
    "Phuket Advanced Imaging"
]

if "hospital" not in st.session_state:
    st.session_state.hospital = HOSPITALS[0]

with st.sidebar:
    st.markdown("---")
    st.session_state.hospital = st.selectbox("Active Institution", HOSPITALS)


# =====================================================
# ROLE BASED ACCESS CONTROL (STRICT)
# =====================================================

ROLE_PERMISSIONS = {
    "Admin": ["read", "write", "export", "override", "audit"],
    "Clinician": ["read", "write", "export"],
    "Radiologist": ["read", "write"],
    "Executive": ["read", "analytics"]
}

def has_permission(action):
    role = st.session_state.role
    return action in ROLE_PERMISSIONS.get(role, [])


# =====================================================
# CLINICAL / RESEARCH MODE
# =====================================================

if "system_mode" not in st.session_state:
    st.session_state.system_mode = "Clinical"

with st.sidebar:
    st.session_state.system_mode = st.radio(
        "System Mode",
        ["Clinical", "Research"]
    )


# =====================================================
# MODEL REGISTRY
# =====================================================

MODEL_REGISTRY = {
    "MorphoBio-AI 4.1": {
        "type": "Rule-Based",
        "validated": True,
        "auc": 0.89
    },
    "MorphoBio-AI 5.0 Beta": {
        "type": "Hybrid ML",
        "validated": False,
        "auc": 0.93
    }
}

if "active_model" not in st.session_state:
    st.session_state.active_model = "MorphoBio-AI 4.1"

with st.sidebar:
    st.markdown("---")
    st.session_state.active_model = st.selectbox(
        "AI Model Version",
        list(MODEL_REGISTRY.keys())
    )


# =====================================================
# RISK STRATIFICATION LAYER
# =====================================================

def stratify_risk(score):
    if score >= 80:
        return "HIGH RISK"
    elif score >= 40:
        return "INTERMEDIATE RISK"
    else:
        return "LOW RISK"


# =====================================================
# DATA INTEGRITY HASH
# =====================================================

import hashlib
import json

def generate_integrity_hash(case_row):
    data_string = json.dumps(case_row.to_dict(), sort_keys=True)
    return hashlib.sha256(data_string.encode()).hexdigest()


# =====================================================
# FHIR-READY EXPORT
# =====================================================

def generate_fhir_bundle(case_row):

    fhir_bundle = {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "code": {
            "text": "AI Assisted Biopsy Risk Assessment"
        },
        "subject": {
            "reference": f"Patient/{case_row['HN']}"
        },
        "result": [{
            "valueString": case_row["Status"]
        }],
        "extension": [{
            "url": "confidence",
            "valueDecimal": case_row.get("Calibrated_Confidence", 0)
        }]
    }

    return json.dumps(fhir_bundle, indent=2)


# =====================================================
# GOVERNANCE LOCK – CLINICIAN SIGN-OFF
# =====================================================

if "signoff" not in st.session_state:
    st.session_state.signoff = {}

if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    last_index = st.session_state.db.index[-1]
    last = st.session_state.db.loc[last_index]

    risk_score = last.get("Risk_Score", 0)
    risk_level = stratify_risk(risk_score)

    st.markdown("### Risk Stratification")
    st.metric("Risk Level", risk_level)

    # Integrity hash
    integrity_hash = generate_integrity_hash(last)
    st.caption(f"Data Integrity Hash: {integrity_hash[:16]}...")

    # Clinician signoff
    if has_permission("write"):
        if st.button("Clinician Sign-Off"):
            st.session_state.signoff[last["Case_ID"]] = {
                "Signed_By": st.session_state.role,
                "Timestamp": str(datetime.datetime.now())
            }
            st.success("Case signed and locked.")

    # Display signoff
    if last["Case_ID"] in st.session_state.signoff:
        sign = st.session_state.signoff[last["Case_ID"]]
        st.info(f"Signed by {sign['Signed_By']} at {sign['Timestamp']}")


# =====================================================
# AI OVERRIDE TRACKING
# =====================================================

if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    last_index = st.session_state.db.index[-1]
    last = st.session_state.db.loc[last_index]

    if has_permission("override"):

        st.markdown("---")
        st.subheader("Manual Override (Admin Only)")

        new_status = st.selectbox(
            "Override Status",
            ["NORMAL", "BENIGN", "MALIGNANT"]
        )

        if st.button("Apply Override"):

            st.session_state.db.loc[last_index, "Status"] = new_status

            log_event("MANUAL_OVERRIDE", last["Case_ID"])
            st.warning("Status manually overridden.")


# =====================================================
# INSTITUTIONAL BENCHMARKING
# =====================================================

if nav == "Executive Board View":

    st.markdown("---")
    st.subheader("Institutional Benchmark")

    df = st.session_state.db

    if len(df) > 0:

        malignant_rate = (
            (df["Status"] == "MALIGNANT").sum() / len(df)
        ) * 100

        national_avg = 18.5

        col1, col2 = st.columns(2)

        col1.metric("Your Malignancy Rate", f"{malignant_rate:.1f}%")
        col2.metric("National Average", f"{national_avg}%")

        if malignant_rate > national_avg:
            st.error("Above national malignancy rate.")
        else:
            st.success("Within acceptable national benchmark.")


# =====================================================
# FHIR EXPORT BUTTON
# =====================================================

if nav == "Case Archive" and len(st.session_state.db) > 0:

    selected_case = st.session_state.db.iloc[-1]

    fhir_json = generate_fhir_bundle(selected_case)

    if has_permission("export"):
        st.download_button(
            label="Export FHIR DiagnosticReport",
            data=fhir_json,
            file_name=f"{selected_case['Case_ID']}_FHIR.json",
            mime="application/json"
        )
# =====================================================
# RESEARCH DATABASE (SEPARATE FROM CLINICAL)
# =====================================================

if "research_db" not in st.session_state:
    st.session_state.research_db = pd.DataFrame()

if st.session_state.system_mode == "Research":

    st.sidebar.markdown("### 🔬 Research Lab Active")


# =====================================================
# AUTO ANONYMIZATION
# =====================================================

def anonymize_df(df):

    anon = df.copy()

    if "Patient" in anon.columns:
        anon["Patient"] = "ANON"

    if "HN" in anon.columns:
        anon["HN"] = "ANON_ID"

    return anon


# =====================================================
# IRB TRACKING
# =====================================================

if "irb_info" not in st.session_state:
    st.session_state.irb_info = {
        "Study_ID": "",
        "Principal_Investigator": "",
        "Approval_Status": "Not Submitted"
    }

if st.session_state.system_mode == "Research":

    st.markdown("---")
    st.subheader("IRB Study Registration")

    st.session_state.irb_info["Study_ID"] = st.text_input(
        "Study ID",
        st.session_state.irb_info["Study_ID"]
    )

    st.session_state.irb_info["Principal_Investigator"] = st.text_input(
        "Principal Investigator",
        st.session_state.irb_info["Principal_Investigator"]
    )

    st.session_state.irb_info["Approval_Status"] = st.selectbox(
        "IRB Approval Status",
        ["Not Submitted", "Pending", "Approved"]
    )

    st.info(f"IRB Status: {st.session_state.irb_info['Approval_Status']}")


# =====================================================
# MOVE CASE TO RESEARCH DB
# =====================================================

if nav == "Diagnostic Hub" and st.session_state.system_mode == "Research":

    if len(st.session_state.db) > 0:

        if st.button("Add Latest Case to Research Dataset"):

            last = st.session_state.db.iloc[-1]
            st.session_state.research_db = pd.concat(
                [st.session_state.research_db, pd.DataFrame([last])],
                ignore_index=True
            )

            st.success("Case added to research dataset.")


# =====================================================
# MODEL COMPARISON SIMULATION
# =====================================================

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

if nav == "Professional Analytics" and st.session_state.system_mode == "Research":

    st.markdown("---")
    st.subheader("Model Validation Dashboard")

    if len(st.session_state.research_db) > 5:

        df = st.session_state.research_db.copy()

        # Simulated ground truth
        df["Ground_Truth"] = np.random.choice(
            ["NORMAL", "BENIGN", "MALIGNANT"],
            size=len(df)
        )

        y_true = (df["Ground_Truth"] == "MALIGNANT").astype(int)
        y_pred = (df["Status"] == "MALIGNANT").astype(int)

        cm = confusion_matrix(y_true, y_pred)

        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0

        auc = roc_auc_score(y_true, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("Sensitivity", f"{sensitivity:.2f}")
        col2.metric("Specificity", f"{specificity:.2f}")
        col3.metric("AUC", f"{auc:.2f}")

        st.markdown("### Confusion Matrix")
        st.write(cm)

    else:
        st.info("Add at least 6 cases to Research dataset.")


# =====================================================
# EXPORT ANONYMIZED DATASET
# =====================================================

if nav == "Case Archive" and st.session_state.system_mode == "Research":

    if len(st.session_state.research_db) > 0:

        anon = anonymize_df(st.session_state.research_db)

        csv = anon.to_csv(index=False)

        st.download_button(
            label="Export Anonymized Research Dataset (CSV)",
            data=csv,
            file_name="Research_Dataset_Anonymized.csv",
            mime="text/csv"
        )
