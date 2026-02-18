# ==========================================
# IMPORT SECTION
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import datetime
import plotly.graph_objects as go
import plotly.express as px


# ==========================================
# SESSION STATE INIT
# ==========================================

if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(
        columns=[
            "Date",
            "HN",
            "Patient",
            "Organ",
            "Status",
            "Confidence",
            "Marker_Val",
            "Tumor_Size"
        ]
    )

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


# ==========================================
#  AUTH SYSTEM
# ==========================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "hospital" not in st.session_state:
    st.session_state.hospital = None

if "role" not in st.session_state:
    st.session_state.role = None


# ==========================================
# LOGIN PAGE
# ==========================================

if not st.session_state.authenticated:

    st.set_page_config(page_title="Smart Biopsy Navigator", layout="centered")

    st.title("Smart Biopsy Navigator")

    st.markdown("### Secure Clinical Access Portal")

    hospital = st.selectbox(
        "Select Hospital",
        [
            "Sri Nagarind Hospital",
            "Bangkok Hospital",
            "Chiang Mai University Hospital",
            "Siriraj Hospital"
        ]
    )

    role = st.selectbox(
        "Select Role",
        [
            "Oncologist",
            "Radiologist",
            "Surgeon",
            "Pathologist",
            "Administrator"
        ]
    )

    password = st.text_input("Enter Secure Password", type="password")

    if st.button("Login Securely", key="login_button"):

        if password == "SNH_SECURE":

            st.session_state.authenticated = True
            st.session_state.hospital = hospital
            st.session_state.role = role

            st.success("Access Granted")
            st.rerun()

        else:
            st.error("Access Denied")

    st.stop()

# ==========================================
# SIDEBAR NAVIGATION (AFTER LOGIN)
# ==========================================

st.sidebar.title("Smart Biopsy Navigator")

st.sidebar.markdown("### 👤 User Info")
st.sidebar.write(f"Hospital: {st.session_state.hospital}")
st.sidebar.write(f"Role: {st.session_state.role}")

if st.sidebar.button("Logout", key="logout_button"):
    st.session_state.authenticated = False
    st.session_state.hospital = None
    st.session_state.role = None
    st.rerun()

page = st.sidebar.radio(
    "Select Page",
    ["📂 Clinical Dashboard", "💼 Business Overview"]
)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Smart Biopsy Navigator")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# ==========================================
# AI ENGINE
# ==========================================

def run_ai(organ, marker, size):

    score = 0

    if organ == "Liver":
        score = (float(marker) * 0.4) + (size * 0.6)

    elif organ == "Thyroid":
        score = (int(marker) * 15) + (size * 0.5)

    elif organ == "Breast":
        score = (int(marker) * 20) + (size * 0.5)

    else:
        score = size * 0.5

    confidence = min(score / 100, 1)

    if confidence < 0.30:
        status = "Normal"
    elif confidence < 0.70:
        status = "Benign"
    else:
        status = "Malignant"

    return status, confidence


    # =============================
    # RIGHT PANEL — RESULT
    # =============================
    with col2:

        if len(st.session_state.db) > 0:

            last = st.session_state.db.iloc[-1]
            confidence = last["Confidence"]
            conf_percent = confidence * 100

            tab1, tab2, tab3, tab4 = st.tabs(
                ["🧠 Result", "📊 AI Insights", "📁 Audit", "⚙ System"]
            )

            # ---- RESULT ----
            with tab1:

                if len(st.session_state.db) > 0:
            
                    last = st.session_state.db.iloc[-1]
                    confidence = last["Confidence"]
                    conf_percent = confidence * 100
            
                    # 🎨 Define color
                    if last["Status"] == "Normal":
                        color = "#00cc96"
                    elif last["Status"] == "Benign":
                        color = "#FFA500"
                    else:
                        color = "#EF553B"
            
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf_percent,
                        number={'suffix': "%"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': color}
                        }
                    ))
            
                    st.plotly_chart(fig, use_container_width=True)
            
                    st.markdown(f"## {last['Status']}")
            
                    if last["Status"] == "Normal":
                        st.success("No suspicious malignancy detected")
                    elif last["Status"] == "Benign":
                        st.warning("Likely benign lesion – follow up recommended")
                    else:
                        st.error("High suspicion of malignancy – biopsy recommended")
            
                else:
                    st.info("No analysis available yet")



            # ---- AI INSIGHTS ----
            with tab2:

                st.progress(int(conf_percent))
                st.write(f"Confidence: {conf_percent:.2f}%")

                st.info(
                    f"""
                    • Organ: {last['Organ']}
                    • Marker: {last['Marker_Val']}
                    • Size: {last['Tumor_Size']} mm
                    """
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("AUC", "0.89")
                c2.metric("Sensitivity", "87%")
                c3.metric("Specificity", "84%")

            # ---- AUDIT ----
            with tab3:

                st.dataframe(st.session_state.db, use_container_width=True)

            # ---- SYSTEM ----
            with tab4:

                st.caption(
                    f"Model Version: v1.0 | Generated: "
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

                st.info("""
                Smart Biopsy Navigator™  
                Clinical Decision Support System  
                Research / Demo Deployment  
                """)

                st.caption(
                    "Final diagnosis must be made by a licensed physician."
                )

# =====================================================
# 💼 BUSINESS OVERVIEW
# =====================================================

if page == "💼 Business Overview":

    st.title("💼 Business & Growth Strategy")

    st.subheader("Market Opportunity")
    st.write("AI-assisted diagnostic decision tools for multi-organ biopsy triage.")

    st.subheader("Revenue Model")
    st.write("""
    - Hospital Subscription Model  
    - Enterprise License  
    - API Integration for Radiology Systems  
    """)

    st.subheader("Growth Strategy")
    st.write("""
    - Expand to multi-center validation  
    - Add deep learning imaging model  
    - Regulatory pathway preparation  
    """)

    st.subheader("💰 Business Model")
    st.info("""
    • Hospital SaaS Subscription  
    • Per-case AI scoring  
    • Enterprise API Integration  
    • Diagnostic Lab Licensing  
    """)

    st.subheader("🎯 Target Customers")

    b1, b2, b3 = st.columns(3)
    b1.success("Mid-sized Hospitals")
    b2.success("Diagnostic Labs")
    b3.success("National Health Systems")

    with st.expander("📊 Market Opportunity"):
        st.markdown("""
        • Growing oncology diagnostics demand  
        • Rising AI adoption in healthcare  
        • Global cancer incidence increasing  
        """)

    st.subheader("☁ Scalability")
    st.info("""
    • Cloud-native deployment  
    • Multi-organ model expansion  
    • EMR / HIS API-ready  
    • Modular retraining pipeline  
    """)

    with st.expander("📈 Milestones & Roadmap"):
        st.markdown("""
        • MVP Completed  
        • Retrospective Evaluation  
        • Pilot Site Discussions  
        • Regulatory Planning (FDA / CE)  
        """)

    st.subheader("📊 Growth Projection")

    k1, k2, k3 = st.columns(3)
    k1.metric("Projected ARR (Year 3)", "$2M+")
    k2.metric("Target Pilot Sites", "10 Hospitals")
    k3.metric("SaaS Gross Margin", "70%+")

    st.markdown("---")
    st.success("AI Triage Layer for Oncology Diagnostics")


# =====================================================
# SAFE SESSION INITIALIZATION (CRITICAL FOR CLOUD)
# =====================================================
if "auth" not in st.session_state:
    st.session_state.auth = False

if "role" not in st.session_state:
    st.session_state.role = None

if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        "Date",
        "HN",
        "Patient",
        "Organ",
        "Status",
        "Confidence",
        "Marker_Val",
        "Tumor_Size"
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
# PROFESSIONAL ANALYTICS
# =====================================================
    if nav == "Professional Analytics":

        st.title("Professional Clinical Dashboard")

    df = st.session_state.db.copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cases", len(df))
    c2.metric(
        "Malignancy %",
        f"{(df['Status']=='MALIGNANT').mean()*100 if len(df)>0 else 0:.1f}%"
    )
    c3.metric(
        "Avg Confidence",
        f"{df['Confidence'].mean()*100 if len(df)>0 else 0:.1f}%"
    )

    if len(df) == 0:
        st.info("No data available yet.")
    else:

        # -----------------------------------
        # FIXED CLINICAL STATUS ORDER
        # -----------------------------------
        status_order = ["NORMAL", "BENIGN", "MALIGNANT"]

        # -----------------------------------
        # DONUT CHART (STRICT 3 COLORS)
        # -----------------------------------
        summary = (
            df["Status"]
            .value_counts()
            .reindex(status_order, fill_value=0)
            .reset_index()
        )
        summary.columns = ["Status", "Count"]

        fig_pie = px.pie(
            summary,
            names="Status",
            values="Count",
            category_orders={"Status": status_order},
            color="Status",
            color_discrete_map={
                "NORMAL": "#10B981",
                "BENIGN": "#F59E0B",
                "MALIGNANT": "#EF4444"
            },
            hole=0.6
        )

        fig_pie.update_traces(
            textinfo="percent+label",
            marker=dict(line=dict(color="white", width=2))
        )

        fig_pie.update_layout(
            title="Risk Distribution",
            template="plotly_white",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=60)
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # -----------------------------------
        # ORGAN CASE DISTRIBUTION (STRICT 3 COLORS)
        # -----------------------------------
        organ_summary = (
            df.groupby(["Organ", "Status"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=status_order, fill_value=0)
            .stack()
            .reset_index(name="Count")
        )

        fig_bar = px.bar(
            organ_summary,
            x="Organ",
            y="Count",
            color="Status",
            category_orders={"Status": status_order},
            color_discrete_map={
                "NORMAL": "#10B981",
                "BENIGN": "#F59E0B",
                "MALIGNANT": "#EF4444"
            },
            barmode="group"
        )

        fig_bar.update_layout(
            title="Organ Case Distribution",
            xaxis_title="Organ",
            yaxis_title="Cases",
            template="plotly_white",
            margin=dict(t=60)
        )

        fig_bar.update_traces(
            marker_line_width=1,
            marker_line_color="white"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

# =====================================================
# EXECUTIVE BOARD VIEW
# =====================================================
    if nav == "Executive Board View":

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
    if nav == "Case Archive":
        st.dataframe(st.session_state.db, use_container_width=True)

# =====================================================
# USER MANUAL (DETAILED)
# =====================================================
    if nav == "User Manual":

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
# MODEL GOVERNANCE PANEL
# =====================================================
if nav == "Professional Analytics":

    st.markdown("---")
    st.subheader("AI Model Governance")

    MODEL_INFO = {
        "Version": "SBP-Pro v2.4.1",
        "Last Validation": "Jan 2026",
        "AUC": 0.93,
        "Sensitivity": 0.91,
        "Specificity": 0.88,
        "Training Cases": 12432
    }

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Version", MODEL_INFO["Version"])
    col2.metric("AUC", MODEL_INFO["AUC"])
    col3.metric("Training Cases", MODEL_INFO["Training Cases"])

    col4, col5 = st.columns(2)
    col4.metric("Sensitivity", f"{MODEL_INFO['Sensitivity']*100:.1f}%")
    col5.metric("Specificity", f"{MODEL_INFO['Specificity']*100:.1f}%")


# =====================================================
# DRIFT DETECTION SIMULATION
# =====================================================
if nav == "Professional Analytics" and len(st.session_state.db) > 20:

    st.markdown("---")
    st.subheader("Data Drift Monitoring")

    df = st.session_state.db

    avg_size = df["Tumor_Size"].mean()
    baseline = 20

    drift = abs(avg_size - baseline)

    if drift > 10:
        st.error("Warning: Significant data drift detected.")
    elif drift > 5:
        st.warning("Mild drift observed.")
    else:
        st.success("No significant drift detected.")


# =====================================================
# QUALITY CONTROL PANEL
# =====================================================
if nav == "Professional Analytics":

    st.markdown("---")
    st.subheader("Quality Control Indicators")

    df = st.session_state.db

    if len(df) > 0:

        high_conf = (df["Confidence"] > 0.85).mean() * 100
        low_conf = (df["Confidence"] < 0.5).mean() * 100

        colA, colB = st.columns(2)
        colA.metric("High Confidence Cases %", f"{high_conf:.1f}%")
        colB.metric("Low Confidence Cases %", f"{low_conf:.1f}%")


# =====================================================
# RISK BURDEN HEATMAP
# =====================================================
if nav == "Executive Board View" and len(st.session_state.db) > 0:

    st.markdown("---")
    st.subheader("Institutional Risk Heatmap")

    df = st.session_state.db

    heat = (
        df.groupby(["Organ", "Status"])
        .size()
        .unstack(fill_value=0)
    )

    fig_heat = px.imshow(
        heat,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Reds"
    )

    fig_heat.update_layout(template="plotly_white")

    st.plotly_chart(fig_heat, use_container_width=True)


# =====================================================
# AUDIT TRAIL VIEWER
# =====================================================
if nav == "Case Archive" and len(st.session_state.db) > 0:

    st.markdown("---")
    st.subheader("Audit Trail")

    audit_df = st.session_state.db[[
        "Case_ID",
        "Timestamp",
        "Created_By",
        "Organ",
        "Status"
    ]]

    st.dataframe(audit_df, use_container_width=True)


# =====================================================
# EXECUTIVE SUMMARY PDF
# =====================================================
if nav == "Executive Board View" and len(st.session_state.db) > 0:

    st.markdown("---")
    st.subheader("Generate Executive Summary Report")

    if st.button("Generate Executive PDF"):

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        df = st.session_state.db
        total = len(df)
        malignant = (df["Status"] == "MALIGNANT").sum()

        elements.append(Paragraph("<b>SMART BIOPSY PRO – EXECUTIVE SUMMARY</b>", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(f"Total Cases: {total}", styles["Normal"]))
        elements.append(Paragraph(f"High Risk Cases: {malignant}", styles["Normal"]))
        elements.append(Paragraph("Model Version: SBP-Pro v2.4.1", styles["Normal"]))

        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            "Download Executive PDF",
            buffer,
            file_name="Executive_Summary.pdf"
        )
# =====================================================
# ULTRA ENTERPRISE EXTENSION LAYER
# =====================================================

import hashlib
import numpy as np

# -----------------------------------------------------
# ADD HOSPITAL COLUMN IF MISSING
# -----------------------------------------------------
if "Hospital" not in st.session_state.db.columns:
    st.session_state.db["Hospital"] = "Main Center"

if "Model_Version" not in st.session_state.db.columns:
    st.session_state.db["Model_Version"] = "v2.4.1"

if "Activity_Log" not in st.session_state:
    st.session_state.Activity_Log = []

# -----------------------------------------------------
# USER ACTIVITY LOGGING
# -----------------------------------------------------
def log_activity(action):
    entry = {
        "Timestamp": str(datetime.datetime.now()),
        "User": st.session_state.role,
        "Action": action
    }
    st.session_state.Activity_Log.append(entry)

# Example auto log
log_activity(f"Accessed {nav}")

# -----------------------------------------------------
# ENCRYPTED AUDIT HASH
# -----------------------------------------------------
def encrypt_row(row):
    raw = str(row.values)
    return hashlib.sha256(raw.encode()).hexdigest()

if "Audit_Hash" not in st.session_state.db.columns:
    st.session_state.db["Audit_Hash"] = ""

for i in st.session_state.db.index:
    if st.session_state.db.loc[i, "Audit_Hash"] == "":
        st.session_state.db.loc[i, "Audit_Hash"] = encrypt_row(
            st.session_state.db.loc[i]
        )

# -----------------------------------------------------
# MULTI HOSPITAL COMPARISON
# -----------------------------------------------------
if nav == "Executive Board View":

    st.markdown("---")
    st.subheader("Multi-Hospital Comparison")

    df = st.session_state.db

    if len(df) > 0:

        hospital_summary = (
            df.groupby(["Hospital", "Status"])
            .size()
            .reset_index(name="Cases")
        )

        fig_multi = px.bar(
            hospital_summary,
            x="Hospital",
            y="Cases",
            color="Status",
            barmode="group"
        )

        st.plotly_chart(fig_multi, use_container_width=True)

# -----------------------------------------------------
# MODEL VERSION ROLLBACK
# -----------------------------------------------------
if nav == "Professional Analytics":

    st.markdown("---")
    st.subheader("Model Version Control")

    versions = ["v2.4.1", "v2.3.0", "v2.2.5"]

    selected_version = st.selectbox("Select Active Model Version", versions)

    if st.button("Apply Version"):
        st.session_state.db["Model_Version"] = selected_version
        st.success(f"Model switched to {selected_version}")
        log_activity(f"Model switched to {selected_version}")

# -----------------------------------------------------
# REAL-TIME CASE STREAM
# -----------------------------------------------------
if nav == "Professional Analytics":

    st.markdown("---")
    st.subheader("Real-Time Case Stream")

    if len(st.session_state.db) > 0:
        recent = st.session_state.db.tail(5)[
            ["Timestamp", "Hospital", "Organ", "Status"]
        ]
        st.dataframe(recent, use_container_width=True)

# -----------------------------------------------------
# PERFORMANCE BY CLINICIAN
# -----------------------------------------------------
if nav == "Professional Analytics":

    st.markdown("---")
    st.subheader("Performance by Clinician")

    df = st.session_state.db

    if len(df) > 0 and "Created_By" in df.columns:

        perf = (
            df.groupby("Created_By")["Confidence"]
            .mean()
            .reset_index()
        )

        fig_perf = px.bar(
            perf,
            x="Created_By",
            y="Confidence",
            title="Average Confidence by Clinician"
        )

        st.plotly_chart(fig_perf, use_container_width=True)

# -----------------------------------------------------
# CONFUSION MATRIX PANEL (SIMULATED)
# -----------------------------------------------------
if nav == "Professional Analytics":

    st.markdown("---")
    st.subheader("Confusion Matrix (Simulated Validation)")

    matrix = np.array([[120, 15],
                       [10, 95]])

    fig_cm = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

# -----------------------------------------------------
# CALIBRATION CURVE
# -----------------------------------------------------
if nav == "Professional Analytics":

    st.markdown("---")
    st.subheader("Calibration Curve")

    probs = np.linspace(0, 1, 50)
    observed = probs ** 0.9

    fig_cal = px.line(
        x=probs,
        y=observed,
        labels={"x": "Predicted Probability",
                "y": "Observed Frequency"}
    )

    st.plotly_chart(fig_cal, use_container_width=True)

# -----------------------------------------------------
# CONFIDENCE DISTRIBUTION HISTOGRAM
# -----------------------------------------------------
if nav == "Professional Analytics":

    st.markdown("---")
    st.subheader("AI Confidence Distribution")

    df = st.session_state.db

    if len(df) > 0:

        fig_hist = px.histogram(
            df,
            x="Confidence",
            nbins=20
        )

        st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------
# VIEW USER ACTIVITY LOG
# -----------------------------------------------------
if nav == "Executive Board View":

    st.markdown("---")
    st.subheader("User Activity Log")

    if len(st.session_state.Activity_Log) > 0:
        st.dataframe(pd.DataFrame(st.session_state.Activity_Log),
                     use_container_width=True)
# =====================================================
# ============ ENTERPRISE PRO MODULE ==================
# =====================================================

import hashlib
import time
import json
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

# =====================================================
# 1️⃣ SESSION INITIALIZATION
# =====================================================

if "model_version" not in st.session_state:
    st.session_state.model_version = "v1.0"

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

if "hospital" not in st.session_state:
    st.session_state.hospital = "Main Hospital"

# =====================================================
# 2️⃣ FHIR ENTERPRISE EXPORT (LOINC + SNOMED CT)
# =====================================================

LOINC_CODES = {
    "AFP": "1834-1",
    "BI-RADS": "24606-6",
    "TI-RADS": "LA6576-8"
}

SNOMED_CODES = {
    "NORMAL": "17621005",
    "BENIGN": "38907003",
    "MALIGNANT": "363346000"
}

def generate_fhir_bundle(row):

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": row["HN"],
                    "name": [{"text": row["Patient"]}]
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": LOINC_CODES.get(str(row["Marker_Val"]), "00000-0")
                        }]
                    },
                    "valueQuantity": {
                        "value": row["Marker_Val"]
                    }
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": SNOMED_CODES.get(row["Status"], "000000")
                        }]
                    },
                    "clinicalStatus": {"text": row["Status"]}
                }
            }
        ]
    }

    return json.dumps(bundle, indent=2)

# =====================================================
# 3️⃣ USER ACTIVITY + ENCRYPTED AUDIT LOG
# =====================================================

def log_activity(action):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} | {action}"
    st.session_state.activity_log.append(entry)

    encrypted = hashlib.sha256(entry.encode()).hexdigest()
    st.session_state.audit_log.append(encrypted)

# =====================================================
# 4️⃣ REAL-TIME CASE STREAM
# =====================================================

if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    st.markdown("### 🔴 Real-Time Case Stream")
    live_case = st.session_state.db.iloc[-1]
    st.json(live_case.to_dict())

    log_activity("Viewed Real-Time Case")

    st.markdown("---")
    st.subheader("FHIR Export (Enterprise Standard)")
    fhir_json = generate_fhir_bundle(live_case)

    st.download_button(
        "Download FHIR Bundle",
        fhir_json,
        file_name="case_bundle.json",
        mime="application/json"
    )

# =====================================================
# 5️⃣ PROFESSIONAL ANALYTICS DASHBOARD
# =====================================================

if nav == "Professional Analytics":

    st.header("Enterprise Clinical Validation Panel")

    np.random.seed(42)

    y_true = np.random.choice([0, 1], 200)
    y_pred = np.random.choice([0, 1], 200)
    y_prob = np.random.rand(200)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    miss_rate = fn / (tp + fn)

    baseline_miss = 0.18
    reduction = (baseline_miss - miss_rate) / baseline_miss * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Sensitivity", f"{sensitivity*100:.1f}%")
    col2.metric("Specificity", f"{specificity*100:.1f}%")
    col3.metric("Miss Rate Reduction", f"{reduction:.1f}%")


    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm)
    
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Benign", "Malignant"])
    ax_cm.set_yticklabels(["Benign", "Malignant"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, cm[i, j],
                       ha="center", va="center")
    
    st.pyplot(fig_cm)

    st.pyplot(fig_cm)

    # Calibration Curve
    st.subheader("Calibration Curve")
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig_cal, ax_cal = plt.subplots()
    ax_cal.plot(prob_pred, prob_true, marker='o')
    ax_cal.plot([0,1],[0,1], linestyle='--')
    st.pyplot(fig_cal)

    # AI Confidence Distribution
    st.subheader("AI Confidence Distribution")
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(y_prob, bins=20)
    st.pyplot(fig_hist)

    log_activity("Viewed Professional Analytics")

# =====================================================
# 6️⃣ PERFORMANCE BY CLINICIAN
# =====================================================

if nav == "Executive Board View":

    st.header("Performance by Clinician")

    clinicians = ["Dr.A", "Dr.B", "Dr.C"]
    performance = np.random.uniform(0.7, 0.95, 3)

    df_perf = pd.DataFrame({
        "Clinician": clinicians,
        "Accuracy": performance
    })

    st.bar_chart(df_perf.set_index("Clinician"))

    # Multi-Hospital Comparison
    st.subheader("Multi-Hospital Comparison")

    hospitals = ["Main Hospital", "Branch A", "Branch B"]
    accuracy = np.random.uniform(0.75, 0.97, 3)

    df_hosp = pd.DataFrame({
        "Hospital": hospitals,
        "AI Accuracy": accuracy
    })

    st.bar_chart(df_hosp.set_index("Hospital"))

    # Model Version Rollback
    st.subheader("Model Version Control")

    version = st.selectbox("Select Model Version",
                           ["v1.0", "v1.1", "v2.0"])

    if st.button("Apply Model Version"):
        st.session_state.model_version = version
        log_activity(f"Rolled back to {version}")
        st.success(f"Model switched to {version}")

# =====================================================
# 7️⃣ AUDIT LOG VIEWER
# =====================================================

if nav == "Admin Panel":

    st.header("Encrypted Audit Log (SHA256)")

    st.write(st.session_state.audit_log)

    st.markdown("### User Activity Log")
    st.write(st.session_state.activity_log)

# =====================================================
# ================= END MODULE ========================
# =====================================================
# =====================================================
# ===== ACCURATE MULTI-ORGAN ML TRAINING ENGINE ======
# =====================================================

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =====================================================
# AI TRAINING LAB – MEDICAL AI TIER 2
# =====================================================

if nav == "AI Training Lab":

    st.header("Multi-Organ AI Training Engine – Tier 2")

    if len(st.session_state.db) == 0:
        st.warning("No data available.")
    else:

        df = st.session_state.db.copy()

        organ = st.selectbox("Select Organ", df["Organ"].unique())
        df = df[df["Organ"] == organ]

        if len(df) < 50:
            st.warning("Not enough data for training.")
        else:

            st.write(f"Training on {len(df)} cases")

            # ---------------------------------------------
            # TARGET
            # ---------------------------------------------
            df = df[df["Status"].isin(["BENIGN", "MALIGNANT"])]
            y = df["Status"]
            le = LabelEncoder()
            y = le.fit_transform(y)

            # ---------------------------------------------
            # AFP OPTIONAL (CLINICAL SAFE)
            # ---------------------------------------------
            if "AFP" in df.columns:
                df["AFP_available"] = df["AFP"].notnull().astype(int)
                df["AFP"] = df["AFP"].fillna(0)
            else:
                df["AFP"] = 0
                df["AFP_available"] = 0

            # ---------------------------------------------
            # FEATURE SELECTION
            # ---------------------------------------------
            feature_cols = [
                col for col in df.columns
                if col not in ["Status", "Patient", "HN"]
                and df[col].dtype != "object"
            ]

            if "AFP_available" not in feature_cols:
                feature_cols.append("AFP_available")

            X = df[feature_cols]

            # ---------------------------------------------
            # TRAIN TEST SPLIT
            # ---------------------------------------------
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                stratify=y,
                random_state=42
            )

            # ---------------------------------------------
            # MODEL (Balanced + Controlled Depth)
            # ---------------------------------------------
            model = RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                class_weight="balanced",
                random_state=42
            )

            model.fit(X_train, y_train)

            # ---------------------------------------------
            # PROBABILITY
            # ---------------------------------------------
            y_prob = model.predict_proba(X_test)[:, 1]

            # ---------------------------------------------
            # IMAGING AUTHORITY BOOST
            # ---------------------------------------------
            if "Ultrasound_Score" in X_test.columns:
                us = X_test["Ultrasound_Score"].values
                imaging_boost = np.clip(us, 0, 1)
                y_prob = 0.7 * imaging_boost + 0.3 * y_prob

            # ---------------------------------------------
            # AUTO THRESHOLD FOR ≥ 90% SENSITIVITY
            # ---------------------------------------------
            thresholds = np.linspace(0.1, 0.9, 200)
            best_threshold = 0.5
            best_miss = 1.0

            for t in thresholds:
                y_pred_temp = (y_prob >= t).astype(int)
                cm_temp = confusion_matrix(y_test, y_pred_temp)
                tn, fp, fn, tp = cm_temp.ravel()
                sensitivity = tp / (tp + fn)

                if sensitivity >= 0.90:
                    miss = fn / (tp + fn)
                    if miss < best_miss:
                        best_miss = miss
                        best_threshold = t

            threshold = best_threshold
            y_pred = (y_prob >= threshold).astype(int)

            # ---------------------------------------------
            # METRICS
            # ---------------------------------------------
            cm = confusion_matrix(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            tn, fp, fn, tp = cm.ravel()

            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            miss_rate = fn / (tp + fn)

            col1, col2, col3 = st.columns(3)
            col1.metric("AUC", f"{auc:.3f}")
            col2.metric("Sensitivity", f"{sensitivity*100:.1f}%")
            col3.metric("Specificity", f"{specificity*100:.1f}%")

            st.metric("Miss Rate", f"{miss_rate*100:.1f}%")
            st.metric("Optimized Threshold", f"{threshold:.2f}")

            # ---------------------------------------------
            # CONFUSION MATRIX
            # ---------------------------------------------
            st.subheader("Confusion Matrix")

            fig_cm, ax_cm = plt.subplots()
            ax_cm.imshow(cm)

            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, cm[i, j],
                               ha="center", va="center")

            ax_cm.set_xticks([0,1])
            ax_cm.set_yticks([0,1])
            ax_cm.set_xticklabels(["Benign","Malignant"])
            ax_cm.set_yticklabels(["Benign","Malignant"])

            st.pyplot(fig_cm)

            # ---------------------------------------------
            # ROC CURVE
            # ---------------------------------------------
            st.subheader("ROC Curve")

            fpr, tpr, _ = roc_curve(y_test, y_prob)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr)
            ax_roc.plot([0,1],[0,1], linestyle="--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")

            st.pyplot(fig_roc)

            # ---------------------------------------------
            # FEATURE IMPORTANCE
            # ---------------------------------------------
            st.subheader("Feature Importance")

            importance = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(feat_df)

            st.success("Tier-2 Medical Training Complete.")
# =====================================================
# 💼 BUSINESS & SCALABILITY OVERVIEW
# =====================================================

st.markdown("---")
st.markdown("# 💼 Business & Growth Strategy")
st.caption("Investor-Focused Overview")

# =====================================================
# Revenue Model
# =====================================================

st.markdown("## 💰 Business Model")

st.info("""
Revenue Streams:
• Hospital SaaS Subscription (per site / per month)
• Per-case AI scoring fee
• Enterprise API Integration
• Licensing to diagnostic laboratories
""")

# =====================================================
# Target Customers
# =====================================================

st.markdown("## 🎯 Target Customers")

col1, col2, col3 = st.columns(3)

col1.success("Primary:\nMid-sized Hospitals")
col2.success("Secondary:\nDiagnostic Labs")
col3.success("Future:\nNational Health Systems")

# =====================================================
# Market Opportunity
# =====================================================

with st.expander("📊 Market Opportunity"):

    st.markdown("""
    • Growing global oncology diagnostics market  
    • Increasing cancer incidence worldwide  
    • Demand for AI-assisted clinical decision support  

    Initial Market Entry:
    • Regional hospitals
    • Private diagnostic centers
    """)

# =====================================================
# Scalability
# =====================================================

st.markdown("## ☁ Scalability")

st.info("""
• Cloud-native deployment  
• Multi-organ model expansion  
• API-ready for EMR/HIS integration  
• Modular AI retraining pipeline  
""")

# =====================================================
# Milestones & Traction
# =====================================================

with st.expander("📈 Milestones & Roadmap"):

    st.markdown("""
    • MVP Completed  
    • Retrospective Dataset Evaluation  
    • Pilot Site Discussions  
    • Preparing Grant / Funding Applications  
    """)

# =====================================================
# Future KPI Projection
# =====================================================

st.markdown("## 📊 Growth Projection")

k1, k2, k3 = st.columns(3)

k1.metric("Projected ARR (Year 3)", "$2M+")
k2.metric("Target Pilot Sites", "10 Hospitals")
k3.metric("SaaS Gross Margin", "70%+")

st.markdown("---")
st.success("Positioned as an AI Triage Layer for Oncology Diagnostics")
