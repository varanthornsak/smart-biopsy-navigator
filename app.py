import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import uuid
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Smart Biopsy Navigator AI",
                   layout="wide")

# =====================================================
# SESSION INIT
# =====================================================
if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        "Case_ID",
        "Timestamp",
        "Created_By",
        "Role",
        "Organ",
        "Lesion_Size",
        "Risk_Score",
        "Status"
    ])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =====================================================
# SIMPLE AUTH SYSTEM
# =====================================================
USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "doctor": {"password": "doc123", "role": "Radiologist"},
    "exec": {"password": "exec123", "role": "Executive"}
}

def login():
    st.title("Smart Biopsy Navigator AI")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.session_state.role = USERS[username]["role"]
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Menu", [
    "Diagnostic Hub",
    "Case Archive",
    "Executive Dashboard"
])

st.sidebar.markdown("---")
st.sidebar.write("Logged in as:", st.session_state.user)
st.sidebar.write("Role:", st.session_state.role)

# =====================================================
# AI RISK LOGIC
# =====================================================
def calculate_risk(organ, size):
    base = 0.2
    if organ == "Liver":
        base += 0.2
    if organ == "Breast":
        base += 0.15
    if organ == "Thyroid":
        base += 0.1

    risk = base + (size * 0.05)
    return min(risk, 1.0)

def classify(risk):
    if risk < 0.4:
        return "NORMAL"
    elif risk < 0.7:
        return "BENIGN"
    else:
        return "MALIGNANT"

def color_status(status):
    if status == "NORMAL":
        return "green"
    if status == "BENIGN":
        return "orange"
    if status == "MALIGNANT":
        return "red"

def differential_dx(organ, status):
    if status == "MALIGNANT":
        return ["Primary Cancer", "Metastasis", "Aggressive Lesion"]
    if status == "BENIGN":
        return ["Adenoma", "Cyst", "Inflammatory Lesion"]
    return ["No significant pathology"]

# =====================================================
# DIAGNOSTIC HUB
# =====================================================
if nav == "Diagnostic Hub":

    st.title("Diagnostic Hub")

    col1, col2 = st.columns(2)

    organ = col1.selectbox("Select Organ",
                           ["Liver", "Breast", "Thyroid", "Lymph Nodes"])

    size = col2.number_input("Lesion Size (cm)", min_value=0.1, max_value=10.0, step=0.1)

    if st.button("Run AI Assessment"):

        risk = calculate_risk(organ, size)
        status = classify(risk)

        case_id = str(uuid.uuid4())[:8]

        new_case = {
            "Case_ID": case_id,
            "Timestamp": datetime.now(),
            "Created_By": st.session_state.user,
            "Role": st.session_state.role,
            "Organ": organ,
            "Lesion_Size": size,
            "Risk_Score": round(risk, 2),
            "Status": status
        }

        st.session_state.db = pd.concat(
            [st.session_state.db, pd.DataFrame([new_case])],
            ignore_index=True
        )

        st.subheader("AI Risk Explanation")

        st.metric("Risk Score", f"{risk:.2f}")
        st.markdown(
            f"<h2 style='color:{color_status(status)}'>{status}</h2>",
            unsafe_allow_html=True
        )

        st.write("### Differential Diagnosis")
        for dx in differential_dx(organ, status):
            st.write("-", dx)

# =====================================================
# CASE ARCHIVE
# =====================================================
elif nav == "Case Archive":

    st.title("Case Archive")

    if len(st.session_state.db) == 0:
        st.info("No cases yet.")
    else:
        st.dataframe(st.session_state.db, use_container_width=True)

        csv = st.session_state.db.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            "cases.csv",
            "text/csv"
        )

# =====================================================
# EXECUTIVE DASHBOARD
# =====================================================
elif nav == "Executive Dashboard":

    st.title("Executive Dashboard")

    df = st.session_state.db.copy()

    if len(df) == 0:
        st.info("No data available.")
    else:

        total = len(df)
        malignant = len(df[df["Status"] == "MALIGNANT"])
        benign = len(df[df["Status"] == "BENIGN"])
        normal = len(df[df["Status"] == "NORMAL"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Cases", total)
        col2.metric("Malignant", malignant)
        col3.metric("Benign", benign)
        col4.metric("Normal", normal)

        st.markdown("---")

        # Status Distribution (3 colors only)
        fig_status = px.bar(
            df["Status"].value_counts().reset_index(),
            x="index",
            y="Status",
            color="index",
            color_discrete_map={
                "NORMAL": "green",
                "BENIGN": "orange",
                "MALIGNANT": "red"
            },
            labels={"index": "Status", "Status": "Count"},
            title="Status Distribution"
        )

        st.plotly_chart(fig_status, use_container_width=True)

        # Organ Distribution (3-color stacked)
        pivot = df.groupby(["Organ", "Status"]).size().reset_index(name="Count")

        fig_organ = px.bar(
            pivot,
            x="Organ",
            y="Count",
            color="Status",
            color_discrete_map={
                "NORMAL": "green",
                "BENIGN": "orange",
                "MALIGNANT": "red"
            },
            title="Organ Case Distribution"
        )

        st.plotly_chart(fig_organ, use_container_width=True)

# =====================================================
# DARK MODE
# =====================================================
st.sidebar.markdown("---")
if st.sidebar.checkbox("Enable Dark Mode"):
    st.markdown("""
        <style>
        body { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)
# =====================================================
# PROFESSIONAL USER MANUAL
# =====================================================
elif nav == "User Manual":

    st.title("Smart Biopsy Navigator AI - Clinical User Guide")

    st.markdown("## 1. Intended Use")

    st.write("""
    Smart Biopsy Navigator AI is a clinical decision support system
    designed to assist physicians in estimating malignancy risk
    for image-guided biopsy cases.

    This system does NOT replace clinical judgment.
    It provides probabilistic risk stratification only.
    """)

    st.markdown("## 2. Target Users")

    st.write("""
    • Radiologists  
    • Interventional Radiologists  
    • Pathologists  
    • Oncology Specialists  
    • Hospital Executives (Dashboard View Only)
    """)

    st.markdown("## 3. Clinical Workflow")

    st.write("""
    Step 1: Login with assigned credentials  
    Step 2: Go to Diagnostic Hub  
    Step 3: Select Organ  
    Step 4: Enter Lesion Size (cm)  
    Step 5: Click 'Run AI Assessment'  
    Step 6: Review Risk Score + Status  
    Step 7: Review Differential Diagnosis  
    Step 8: Document final clinical decision  
    """)

    st.markdown("## 4. Risk Interpretation")

    st.write("""
    Risk Score Range:

    • 0.00 – 0.39 → NORMAL (Green)  
    • 0.40 – 0.69 → BENIGN (Yellow)  
    • 0.70 – 1.00 → MALIGNANT (Red)

    MALIGNANT classification suggests high suspicion
    and should prompt further diagnostic confirmation.
    """)

    st.markdown("## 5. Color Coding Standard")

    st.write("""
    GREEN  → Low Risk  
    YELLOW → Moderate Risk  
    RED    → High Risk  
    """)

    st.markdown("## 6. Differential Diagnosis Panel")

    st.write("""
    The differential diagnosis panel lists possible
    pathological conditions based on organ + risk class.

    It is intended as a reference only.
    Final diagnosis must rely on histopathology.
    """)

    st.markdown("## 7. Executive Dashboard Usage")

    st.write("""
    Executive Dashboard provides:

    • Total cases
    • Malignancy rate
    • Organ distribution
    • Status distribution

    This view is intended for hospital leadership
    and quality monitoring.
    """)

    st.markdown("## 8. Data Governance")

    st.write("""
    • All case data are stored locally in session memory.
    • Exported CSV files must comply with hospital data policy.
    • Do not upload identifiable patient information.
    """)

    st.markdown("## 9. Model Limitation")

    st.write("""
    • Model is simulation-based.
    • Does not use real imaging data.
    • Not validated in prospective clinical trials.
    • Not approved as a medical device.
    """)

    st.markdown("## 10. Contraindications")

    st.write("""
    This system must NOT be used as the sole basis
    for surgical or oncological decision-making.
    """)

    st.markdown("## 11. Troubleshooting")

    st.write("""
    If system does not respond:

    • Refresh browser
    • Re-login
    • Check internet connection

    If risk score does not generate:
    • Ensure lesion size > 0
    """)

    st.markdown("## 12. Regulatory Status")

    st.write("""
    Current Status:
    Clinical Prototype / Research Use Only

    Not FDA / CE / Thai FDA approved.
    """)

    st.markdown("---")
    st.info("For internal clinical research use only.")
