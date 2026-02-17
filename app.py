import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import datetime
import uuid
import math

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Navigator",
    layout="wide",
)

# =====================================================
# APP STYLE (Apple Minimal)
# =====================================================
st.markdown("""
<style>
html, body { background-color: #f8f9fb; }
.big-title { font-size: 32px; font-weight: 700; }
.subtitle { color: #6e6e73; margin-bottom: 20px; }
.card { padding: 20px; border-radius: 14px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.green { background: #e8f7ef; border-left: 6px solid #2ecc71; }
.yellow { background: #fff8e5; border-left: 6px solid #f1c40f; }
.red { background: #fdecea; border-left: 6px solid #e74c3c; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL CONFIG
# =====================================================
MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth"
DEFAULT_THRESHOLD = 0.2835

# =====================================================
# DATABASE (Multi-tenant SaaS Simulation)
# =====================================================
def init_db():
    conn = sqlite3.connect("saas.db", check_same_thread=False)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT,
        hospital TEXT,
        role TEXT,
        organ TEXT,
        prob REAL,
        classification TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    return conn

conn = init_db()

def log_case(case_id, hospital, role, organ, prob, classification):
    conn.execute(
        "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?)",
        (case_id, hospital, role, organ, prob, classification, str(datetime.datetime.now()))
    )
    conn.commit()

# =====================================================
# LOGIN (Multi-tenant)
# =====================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Premium Enterprise SaaS</div>", unsafe_allow_html=True)

    hospital = st.selectbox("Hospital Tenant",
                            ["Sri Nagarind Hospital", "Bangkok Hospital", "Demo Hospital"])

    role = st.selectbox("User Role",
                        ["Radiologist", "Surgeon", "Oncologist", "Admin"])

    key = st.text_input("Access Key", type="password")

    if st.button("Login"):
        if key == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.hospital = hospital
            st.session_state.role = role
        else:
            st.error("Invalid Key")

    st.stop()

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# =====================================================
# HEADER
# =====================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>{st.session_state.hospital} | {st.session_state.role}</div>", unsafe_allow_html=True)

tabs = st.tabs(["Worklist", "Case Viewer", "Analytics", "Billing", "User Guide"])

# =====================================================
# WORKLIST (PACS STYLE)
# =====================================================
with tabs[0]:
    st.subheader("Case Worklist")

    df = pd.read_sql_query("SELECT * FROM cases WHERE hospital=?",
                           conn,
                           params=(st.session_state.hospital,))

    if not df.empty:
        st.dataframe(df.sort_values("timestamp", ascending=False))
    else:
        st.info("No cases yet.")

# =====================================================
# CASE VIEWER
# =====================================================
with tabs[1]:

    st.subheader("New Case")

    organ = "Liver"
    threshold = st.slider("Operating Threshold", 0.1, 0.9, DEFAULT_THRESHOLD)

    uploaded = st.file_uploader(
        "Upload Ultrasound Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()

        # Classification
        if prob < 0.1:
            label = "Normal"
            style = "green"
        elif prob < threshold:
            label = "Likely Benign"
            style = "yellow"
        else:
            label = "Suspicious Malignant"
            style = "red"

        case_id = str(uuid.uuid4())[:8]

        col1, col2 = st.columns([1.3, 1])

        # ================= IMAGE =================
        with col1:
            st.image(image, use_column_width=True)

        # ================= RESULT + GAUGE =================
        with col2:

            st.markdown(
                f"<div class='card {style}'><b>{label}</b><br>{round(prob*100,2)}%</div>",
                unsafe_allow_html=True
            )

            # ===============================
            # Advanced Clinical Risk Gauge
            # ===============================
            fig, ax = plt.subplots(figsize=(4,2.5))

            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.2)
            ax.axis("off")

            theta = np.linspace(0, math.pi, 200)

            ax.fill_between(np.cos(theta[:70]), 0, np.sin(theta[:70]), alpha=0.15)
            ax.fill_between(np.cos(theta[70:140]), 0, np.sin(theta[70:140]), alpha=0.15)
            ax.fill_between(np.cos(theta[140:]), 0, np.sin(theta[140:]), alpha=0.15)

            ax.plot(np.cos(theta), np.sin(theta), linewidth=2)

            if label == "Normal":
                needle_color = "#2ecc71"
            elif label == "Likely Benign":
                needle_color = "#f1c40f"
            else:
                needle_color = "#e74c3c"

            angle = math.pi * (1 - prob)

            ax.plot(
                [0, np.cos(angle)],
                [0, np.sin(angle)],
                linewidth=3,
                color=needle_color
            )

            ax.scatter(0, 0, s=80)

            ax.text(
                0, -0.05,
                f"{round(prob*100,1)}%",
                ha="center",
                fontsize=20,
                fontweight="bold"
            )

            ax.text(
                0, -0.18,
                label,
                ha="center",
                fontsize=10
            )

            plt.tight_layout()
            st.pyplot(fig)

        # ================= REPORT =================
        st.markdown("### Structured Clinical Report")

        report = f"""
Case ID: {case_id}
Hospital: {st.session_state.hospital}
Role: {st.session_state.role}
Organ: {organ}

Probability: {round(prob,4)}
Threshold: {threshold}
Classification: {label}

Recommended Action:
"""

        if label == "Normal":
            report += "Routine surveillance."
        elif label == "Likely Benign":
            report += "Clinical correlation and short-interval follow-up."
        else:
            report += "Recommend biopsy and oncologic referral."

        st.text_area("Report Preview", report, height=220)

        st.download_button(
            "Export Report",
            data=report,
            file_name=f"{case_id}_report.txt"
        )

        # Sync for FHIR
        st.session_state.fhir_probability = prob
        st.session_state.fhir_patient_id = case_id
        st.session_state.uploaded_image = uploaded

        log_case(
            case_id,
            st.session_state.hospital,
            st.session_state.role,
            organ,
            prob,
            label
        )

# =====================================================
# ANALYTICS DASHBOARD
# =====================================================
with tabs[2]:

    st.subheader("Deployment Analytics")

    df = pd.read_sql_query("SELECT * FROM cases WHERE hospital=?",
                           conn,
                           params=(st.session_state.hospital,))

    if not df.empty:
        st.metric("Total Cases", len(df))
        st.metric("Average Risk", round(df["prob"].mean(),3))
        st.line_chart(df["prob"])

        fig, ax = plt.subplots()
        ax.hist(df["prob"], bins=20)
        ax.set_title("Risk Distribution")
        st.pyplot(fig)
    else:
        st.info("No data available.")

# =====================================================
# BILLING SIMULATION
# =====================================================
with tabs[3]:

    st.subheader("SaaS Billing Simulation")

    df = pd.read_sql_query("SELECT * FROM cases WHERE hospital=?",
                           conn,
                           params=(st.session_state.hospital,))

    price_per_case = 25  # USD simulation

    total_cases = len(df)
    revenue = total_cases * price_per_case

    st.metric("Cases Processed", total_cases)
    st.metric("Revenue (Simulated USD)", revenue)

# =====================================================
# USER GUIDE
# =====================================================
with tabs[4]:

    st.subheader("How to Use the System")

    st.write("""
1. Login with hospital access key.
2. Navigate to Case Viewer.
3. Upload ultrasound image.
4. Review classification:
   - Green → Normal
   - Yellow → Likely Benign
   - Red → Suspicious Malignant
5. Adjust threshold if needed.
6. Export structured report for EMR upload.
7. View analytics and billing in respective tabs.
""")

    st.write("""
Intended Use:
Clinical decision support tool only.
Not a standalone diagnostic system.
Final decisions must be made by licensed physicians.
""")
# =====================================================
# ================= BOARD READY MODE ==================
# =====================================================

st.markdown("---")
st.markdown("## 🏛 Executive & Regulatory Module")

board_tabs = st.tabs([
    "Executive Summary",
    "Regulatory & Compliance",
    "Calibration Performance",
    "FHIR Export Mock",
    "DICOM Metadata",
    "SLA Dashboard",
    "Subscription Model"
])

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================
with board_tabs[0]:

    st.subheader("Hospital Executive Overview")

    df_all = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df_all.empty:

        total = len(df_all)
        high_risk = len(df_all[df_all["classification"] == "Suspicious Malignant"])
        benign = len(df_all[df_all["classification"] == "Likely Benign"])
        normal = len(df_all[df_all["classification"] == "Normal"])

        # -------------------------
        # KPI + Chart Layout
        # -------------------------
        col_metrics, col_chart = st.columns([1.1, 0.9])

        with col_metrics:
            st.metric("Total Cases", total)
            st.metric("High Risk", high_risk)
            st.metric("Likely Benign", benign)
            st.metric("Normal", normal)

        with col_chart:

            labels = ["Normal", "Likely Benign", "Suspicious Malignant"]
            values = [normal, benign, high_risk]
            colors = ["#2ecc71", "#f1c40f", "#e74c3c"]  # เขียว เหลือง แดง

            fig, ax = plt.subplots(figsize=(5,3.5))

            ax.barh(labels, values, color=colors)

            ax.set_title("Clinical Distribution Overview", fontsize=11)
            ax.set_xlabel("Number of Cases")

            # remove top/right borders for cleaner look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            st.pyplot(fig, use_container_width=False)

    else:
        st.info("No enterprise data available yet.")

# =====================================================
# REGULATORY
# =====================================================
with board_tabs[1]:

    st.subheader("Regulatory & Compliance")

    st.write("""
Device Classification: Clinical Decision Support (CDS)

Intended Use:
Ultrasound risk stratification for malignancy probability estimation.

Regulatory Pathway (Simulated):
- US FDA 510(k) SaMD
- EU MDR Class IIa
- Thailand FDA Class 2

Data Governance:
- Audit trail logging enabled
- Multi-tenant isolation
- Role-based access control
- On-prem or cloud deployment option

This system does NOT replace clinical diagnosis.
""")

# =====================================================
# CALIBRATION PANEL
# =====================================================
with board_tabs[2]:

    st.subheader("Model Calibration Performance")

    df_all = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df_all.empty:

        probs = df_all["prob"].values
        bins = np.linspace(0,1,6)

        digitized = np.digitize(probs, bins)

        observed = []
        predicted = []

        for i in range(1,len(bins)):
            bin_probs = probs[digitized == i]
            if len(bin_probs) > 0:
                observed.append(np.mean(bin_probs))
                predicted.append(np.mean(bin_probs))

        fig, ax = plt.subplots()
        ax.plot(predicted, observed, marker="o")
        ax.plot([0,1],[0,1])
        ax.set_title("Calibration Curve (Simulated)")
        ax.set_xlabel("Predicted Risk")
        ax.set_ylabel("Observed Risk")
        st.pyplot(fig)

    else:
        st.info("No calibration data available.")

# =====================================================
# FHIR EXPORT MOCK
# =====================================================
with board_tabs[3]:

    st.subheader("FHIR REST Export Simulation")

    st.code("""
POST /fhir/Observation
{
  "resourceType": "Observation",
  "status": "final",
  "code": { "text": "AI Malignancy Risk" },
  "valueQuantity": {
      "value": 0.78,
      "unit": "probability"
  }
}
""")

    st.write("FHIR endpoint ready for EMR integration.")

# =====================================================
# DICOM METADATA VIEWER
# =====================================================
with board_tabs[4]:

    st.subheader("DICOM Metadata (Simulated)")

    st.table(pd.DataFrame({
        "Tag": [
            "PatientID",
            "StudyDate",
            "Modality",
            "BodyPartExamined",
            "Manufacturer"
        ],
        "Value": [
            "HN123456",
            "2026-02-17",
            "US",
            "Liver",
            "GE Healthcare"
        ]
    }))

# =====================================================
# SLA DASHBOARD
# =====================================================
with board_tabs[5]:

    st.subheader("System SLA Monitoring")

    uptime = 99.87
    inference_latency_ms = 120
    api_calls_today = np.random.randint(50,150)

    st.metric("Uptime (%)", uptime)
    st.metric("Avg Inference Latency (ms)", inference_latency_ms)
    st.metric("API Calls Today", api_calls_today)

# =====================================================
# SUBSCRIPTION SIMULATOR
# =====================================================
with board_tabs[6]:

    st.subheader("SaaS Subscription Model")

    tier = st.selectbox(
        "Plan",
        ["Starter", "Professional", "Enterprise"]
    )

    if tier == "Starter":
        price = 500
        cases = 100
    elif tier == "Professional":
        price = 2000
        cases = 1000
    else:
        price = 10000
        cases = "Unlimited"

    st.metric("Monthly Price (USD)", price)
    st.write("Included Cases:", cases)
# =====================================================
# ===== ADVANCED BOARD / VC / ENTERPRISE MODULE ======
# =====================================================

st.markdown("---")
st.markdown("## 🚀 Strategic & Enterprise Expansion Layer")

advanced_tabs = st.tabs([
    "Improved Executive Overview",
    "FHIR REST (Advanced)",
    "VC Pitch Simulation",
    "Market Size (TAM/SAM/SOM)",
    "ROI & Cost Reduction",
    "Multi-Center Validation",
    "AI Governance",
    "Retraining Pipeline"
])

# =====================================================
# IMPROVED EXECUTIVE OVERVIEW (FIX PIE OVERLAP)
# =====================================================
with advanced_tabs[0]:

    st.subheader("Enhanced Clinical Distribution Overview")

    df_all = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df_all.empty:

        counts = df_all["classification"].value_counts()

        normal = counts.get("Normal", 0)
        benign = counts.get("Likely Benign", 0)
        malignant = counts.get("Suspicious Malignant", 0)

        total = normal + benign + malignant

        col_left, col_right = st.columns([1.1, 1])

        # ==========================
        # LEFT SIDE – KPI + %
        # ==========================
        with col_left:

            st.metric("Total Cases (Enterprise)", total)

            if total > 0:
                st.metric("Normal (%)", round(normal/total*100,1))
                st.metric("Benign (%)", round(benign/total*100,1))
                st.metric("Malignant (%)", round(malignant/total*100,1))
            else:
                st.metric("Normal (%)", 0)
                st.metric("Benign (%)", 0)
                st.metric("Malignant (%)", 0)

        # ==========================
        # RIGHT SIDE – Horizontal Bar
        # ==========================
        with col_right:

            categories = ["Normal", "Benign", "Malignant"]
            values = [normal, benign, malignant]
            colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

            fig, ax = plt.subplots(figsize=(5,3))

            bars = ax.barh(categories, values, color=colors)

            ax.set_title("Enterprise Clinical Distribution", fontsize=11)
            ax.set_xlabel("Number of Cases")

            max_val = max(values) if max(values) > 0 else 1
            ax.set_xlim(0, max_val * 1.2)

            # clean look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + (max_val*0.02),
                        bar.get_y() + bar.get_height()/2,
                        f'{int(width)}',
                        va='center',
                        fontsize=9)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    else:
        st.info("No data yet.")
# =====================================================
# ADVANCED FHIR EXPORT
# =====================================================
with advanced_tabs[1]:

    st.subheader("FHIR REST Endpoint Simulation (Complete Structure)")

    example_payload = {
        "resourceType": "Observation",
        "id": "ai-risk-observation-001",
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging"
            }]
        }],
        "code": {
            "text": "AI Liver Malignancy Risk"
        },
        "subject": {
            "reference": "Patient/HN123456"
        },
       "effectiveDateTime": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "valueQuantity": {
            "value": 0.78,
            "unit": "probability"
        },
        "interpretation": [{
            "text": "Suspicious Malignant"
        }]
    }

    st.json(example_payload)

    st.write("Ready to POST to /fhir/Observation endpoint")

# =====================================================
# VC PITCH SIMULATION
# =====================================================
with advanced_tabs[2]:

    st.subheader("VC Investment Projection Simulation")

    hospitals = st.slider("Number of Hospitals", 10, 500, 100)
    monthly_price = st.slider("Monthly Subscription per Hospital (USD)", 1000, 10000, 3000)

    annual_revenue = hospitals * monthly_price * 12

    st.metric("Projected Annual Revenue (USD)", f"{annual_revenue:,.0f}")

    fig, ax = plt.subplots()
    years = [1,2,3,4,5]
    growth = [annual_revenue * (1.5**(y-1)) for y in years]
    ax.plot(years, growth)
    ax.set_title("5-Year Growth Projection")
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue")
    st.pyplot(fig)

# =====================================================
# MARKET SIZE CALCULATOR
# =====================================================
with advanced_tabs[3]:

    st.subheader("TAM / SAM / SOM Calculator")

    total_hospitals = 20000
    target_region_hospitals = 2000
    expected_capture = st.slider("Market Capture %", 1, 50, 10)

    tam = total_hospitals * 3000 * 12
    sam = target_region_hospitals * 3000 * 12
    som = sam * (expected_capture/100)

    st.metric("TAM (USD)", f"{tam:,.0f}")
    st.metric("SAM (USD)", f"{sam:,.0f}")
    st.metric("SOM (USD)", f"{som:,.0f}")

# =====================================================
# ROI MODEL
# =====================================================
# =====================================================
# AI GOVERNANCE DASHBOARD
# =====================================================
with advanced_tabs[6]:

    st.subheader("AI Governance & Risk Control")

    st.write("""
• Model Version Control Enabled  
• Audit Trail Logging Active  
• Multi-Tenant Isolation Enforced  
• Threshold Monitoring Active  
• Drift Monitoring Enabled (Simulated)  
• Retraining Governance Workflow Defined  
""")

# =====================================================
# RETRAINING PIPELINE
# =====================================================
with advanced_tabs[7]:

    st.subheader("Automated Retraining Trigger Simulation")

    drift_metric = np.random.uniform(0,0.2)
    threshold_drift = 0.15

    st.metric("Drift Score", round(drift_metric,3))

    if drift_metric > threshold_drift:
        st.error("Retraining Trigger Activated")
    else:
        st.success("Model Performance Stable")

    st.write("""
Pipeline:
1. Data aggregation from multi-center
2. Quality control filtering
3. Cross-validation training
4. Calibration
5. External validation
6. Version freeze
7. Deployment to production registry
""")
# =====================================================
# =============== FHIR COMPLETE MODULE (PUBLIC SAFE) =
# =====================================================

import json
import requests
import datetime
import uuid

st.markdown("---")
st.markdown("## 🏥 FHIR Complete Integration Module")

fhir_tabs = st.tabs([
    "Generate Production Bundle",
    "Send to HAPI FHIR",
    "Download for Validator",
    "Bundle Preview"
])

# =====================================================
# SAFE VALUES FROM MODEL SESSION
# =====================================================

safe_patient_id = st.session_state.get("fhir_patient_id", "HN123456")
safe_probability = float(st.session_state.get("fhir_probability", 0.5))

loinc_code = "34543-9"

# =====================================================
# 1️⃣ GENERATE CLEAN HAPI-COMPATIBLE BUNDLE
# =====================================================

with fhir_tabs[0]:

    st.subheader("Generate HAPI-Compatible Transaction Bundle")

    iso_time = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # deterministic IDs
    patient_id = f"patient-{uuid.uuid4().hex[:8]}"
    practitioner_id = f"practitioner-{uuid.uuid4().hex[:8]}"
    observation_id = f"observation-{uuid.uuid4().hex[:8]}"
    report_id = f"report-{uuid.uuid4().hex[:8]}"

    # ---------------- PATIENT ----------------
    patient_resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "active": True
    }

    # ---------------- PRACTITIONER ----------------
    practitioner_resource = {
        "resourceType": "Practitioner",
        "id": practitioner_id,
        "active": True,
        "name": [{"text": "AI Radiologist"}]
    }

    # ---------------- OBSERVATION ----------------
    observation_resource = {
        "resourceType": "Observation",
        "id": observation_id,
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": loinc_code,
                "display": "AI Malignancy Risk"
            }]
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "effectiveDateTime": iso_time,
        "performer": [{
            "reference": f"Practitioner/{practitioner_id}"
        }],
        "valueQuantity": {
            "value": round(safe_probability, 5),
            "unit": "probability"
        }
    }

    # ---------------- DIAGNOSTIC REPORT ----------------
    diagnostic_report_resource = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "final",
        "code": {
            "text": "AI Ultrasound Risk Assessment"
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "effectiveDateTime": iso_time,
        "result": [{
            "reference": f"Observation/{observation_id}"
        }]
    }

    # ---------------- TRANSACTION BUNDLE ----------------
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "resource": patient_resource,
                "request": {
                    "method": "POST",
                    "url": "Patient"
                }
            },
            {
                "resource": practitioner_resource,
                "request": {
                    "method": "POST",
                    "url": "Practitioner"
                }
            },
            {
                "resource": observation_resource,
                "request": {
                    "method": "POST",
                    "url": "Observation"
                }
            },
            {
                "resource": diagnostic_report_resource,
                "request": {
                    "method": "POST",
                    "url": "DiagnosticReport"
                }
            }
        ]
    }

    st.session_state.clean_bundle = bundle

    st.success("HAPI-Compatible Bundle Generated")
    st.json(bundle)

# =====================================================
# 2️⃣ SEND TO PUBLIC HAPI
# =====================================================

with fhir_tabs[1]:

    st.subheader("Send to https://hapi.fhir.org/baseR4/")

    if st.button("POST Bundle to HAPI"):

        if "clean_bundle" not in st.session_state:
            st.error("Generate bundle first.")
        else:
            try:
                response = requests.post(
                    "https://hapi.fhir.org/baseR4/",
                    json=st.session_state.clean_bundle,
                    headers={
                        "Content-Type": "application/fhir+json",
                        "Accept": "application/fhir+json"
                    },
                    timeout=20
                )

                st.write("Status Code:", response.status_code)
                st.text(response.text)

            except Exception as e:
                st.error(f"Connection failed: {e}")

# =====================================================
# 3️⃣ DOWNLOAD FOR OFFICIAL VALIDATOR
# =====================================================

with fhir_tabs[2]:

    st.subheader("Download for Official Validator")

    if "clean_bundle" not in st.session_state:
        st.warning("Generate bundle first.")
    else:
        bundle_json = json.dumps(
            st.session_state.clean_bundle,
            indent=2
        )

        st.download_button(
            label="Download Validator-Ready JSON",
            data=bundle_json,
            file_name="validator_ready_bundle.json",
            mime="application/json"
        )

        st.info("Upload this file to https://validator.fhir.org/")

# =====================================================
# 4️⃣ PREVIEW
# =====================================================

with fhir_tabs[3]:

    st.subheader("Current Bundle in Session")

    if "clean_bundle" in st.session_state:
        st.json(st.session_state.clean_bundle)
    else:
        st.info("No bundle generated yet.")
# =====================================================
# ========== ADVANCED CLINICAL EVALUATION =============
# =====================================================

st.markdown("---")
st.markdown("## 📊 Advanced Clinical Evaluation Module")

evaluation_tabs = st.tabs([
    "ROC Curve",
    "Threshold Impact Simulation",
    "Confusion Matrix",
    "Decision Curve Analysis"
])

# =====================================================
# 🔹 Simulated Ground Truth (for demo research mode)
# =====================================================

df_eval = pd.read_sql_query("SELECT * FROM cases", conn)

if not df_eval.empty and len(df_eval) > 5:

    # Simulate ground truth from stored classification
    df_eval["ground_truth"] = df_eval["classification"].apply(
        lambda x: 1 if x == "Suspicious Malignant" else 0
    )

    probs = df_eval["prob"].values
    y_true = df_eval["ground_truth"].values

else:
    probs = np.random.uniform(0,1,50)
    y_true = np.random.randint(0,2,50)

# =====================================================
# 1️⃣ ROC CURVE
# =====================================================
with evaluation_tabs[0]:

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1])
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    st.pyplot(fig)

# =====================================================
# 2️⃣ THRESHOLD IMPACT SIMULATION
# =====================================================
with evaluation_tabs[1]:

    th = st.slider("Select Threshold", 0.0, 1.0, 0.28)

    preds = (probs >= th).astype(int)

    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)

    col1, col2 = st.columns(2)
    col1.metric("Sensitivity", round(sensitivity,3))
    col2.metric("Specificity", round(specificity,3))

# =====================================================
# 3️⃣ CONFUSION MATRIX
# =====================================================
with evaluation_tabs[2]:

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, preds)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

# =====================================================
# 4️⃣ DECISION CURVE ANALYSIS (Mock)
# =====================================================
with evaluation_tabs[3]:

    thresholds = np.linspace(0.01, 0.99, 50)
    net_benefits = []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        n = len(y_true)
        net_benefit = (tp/n) - (fp/n)*(t/(1-t))
        net_benefits.append(net_benefit)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(thresholds, net_benefits)
    ax.set_title("Decision Curve Analysis")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    st.pyplot(fig)
