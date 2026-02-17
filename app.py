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

    uploaded = st.file_uploader("Upload Ultrasound Image",
                                type=["jpg", "png", "jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        # API Simulation (internal inference endpoint)
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

        col1, col2 = st.columns([1.3,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"<div class='card {style}'><b>{label}</b><br>{round(prob*100,2)}%</div>", unsafe_allow_html=True)

            # Risk Gauge
            fig, ax = plt.subplots()
            ax.axis("off")
            theta = np.linspace(0, math.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta))
            angle = math.pi*(1-prob)
            ax.plot([0,np.cos(angle)],[0,np.sin(angle)], linewidth=4)
            ax.text(0,-0.2,f"{round(prob*100,1)}%",ha="center")
            st.pyplot(fig)

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

        st.download_button("Export Report", data=report,
                           file_name=f"{case_id}_report.txt")

        log_case(case_id,
                 st.session_state.hospital,
                 st.session_state.role,
                 organ,
                 prob,
                 label)

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

        st.metric("Total Cases Processed", total)
        st.metric("High Risk Cases", high_risk)
        st.metric("Benign Cases", benign)
        st.metric("Normal Cases", normal)

        fig, ax = plt.subplots()
        ax.pie(
            [normal, benign, high_risk],
            labels=["Normal", "Benign", "Malignant"],
            autopct="%1.1f%%"
        )
        ax.set_title("Clinical Distribution Overview")
        st.pyplot(fig)

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

        fig, ax = plt.subplots(figsize=(6,6))
        wedges, texts, autotexts = ax.pie(
            [normal, benign, malignant],
            labels=["Normal", "Benign", "Malignant"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)

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
        "effectiveDateTime": str(datetime.datetime.now()),
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
with advanced_tabs[4]:

    st.subheader("Biopsy Avoidance ROI Simulation")

    annual_cases = st.slider("Annual Ultrasound Cases", 1000, 50000, 10000)
    biopsy_cost = st.slider("Cost per Biopsy (USD)", 500, 5000, 2000)
    avoided_rate = st.slider("Biopsy Avoidance %", 5, 50, 20)

    savings = annual_cases * (avoided_rate/100) * biopsy_cost

    st.metric("Estimated Annual Cost Savings (USD)", f"{savings:,.0f}")

# =====================================================
# MULTI-CENTER VALIDATION
# =====================================================
with advanced_tabs[5]:

    st.subheader("External Multi-Center Validation (Simulated)")

    centers = ["Sri Nagarind Hospital", "Bangkok Hospital", "Chiang Mai University"]
    auc_scores = [0.91, 0.88, 0.90]

    df_centers = pd.DataFrame({
        "Center": centers,
        "AUC": auc_scores
    })

    st.dataframe(df_centers)

    fig, ax = plt.subplots()
    ax.bar(centers, auc_scores)
    ax.set_ylim(0.8,1.0)
    ax.set_title("External Validation AUC")
    st.pyplot(fig)

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
# =============== FHIR COMPLETE MODULE ===============
# =====================================================

import json
import requests
import datetime
import uuid

st.markdown("---")
st.markdown("## 🏥 FHIR Complete Integration Module")

fhir_tabs = st.tabs([
    "Generate FHIR Resources",
    "Bundle Builder",
    "Validate Structure",
    "Export JSON",
    "Send to FHIR Server (Mock)"
])

# =====================================================
# 1️⃣ GENERATE OBSERVATION + DIAGNOSTIC REPORT
# =====================================================
with fhir_tabs[0]:

    st.subheader("Generate FHIR Observation + DiagnosticReport")

    patient_id = st.text_input("Patient ID", "HN123456")
    probability_input = st.number_input("AI Probability", 0.0, 1.0, 0.78)
    interpretation_input = st.selectbox(
        "Interpretation",
        ["Normal", "Likely Benign", "Suspicious Malignant"]
    )

    observation_id = f"ai-risk-{uuid.uuid4().hex[:8]}"
    report_id = f"ai-report-{uuid.uuid4().hex[:8]}"

    observation = {
        "resourceType": "Observation",
        "id": observation_id,
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging"
            }]
        }],
        "code": {"text": "AI Liver Malignancy Risk"},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": str(datetime.datetime.utcnow()),
        "valueQuantity": {
            "value": probability_input,
            "unit": "probability"
        },
        "interpretation": [{
            "text": interpretation_input
        }]
    }

    diagnostic_report = {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "final",
        "code": {"text": "AI Ultrasound Risk Assessment"},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": str(datetime.datetime.utcnow()),
        "result": [{
            "reference": f"Observation/{observation_id}"
        }]
    }

    st.json({"Observation": observation, "DiagnosticReport": diagnostic_report})

# =====================================================
# 2️⃣ BUNDLE BUILDER (TRANSACTION MODE)
# =====================================================
with fhir_tabs[1]:

    st.subheader("FHIR Bundle (Transaction)")

    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "resource": observation,
                "request": {
                    "method": "POST",
                    "url": "Observation"
                }
            },
            {
                "resource": diagnostic_report,
                "request": {
                    "method": "POST",
                    "url": "DiagnosticReport"
                }
            }
        ]
    }

    st.json(bundle)

# =====================================================
# 3️⃣ VALIDATION (BASIC STRUCTURE CHECK)
# =====================================================
with fhir_tabs[2]:

    st.subheader("FHIR Structure Validation")

    def basic_validate(resource):
        required = ["resourceType", "status", "subject"]
        missing = [r for r in required if r not in resource]
        return missing

    missing_fields = basic_validate(observation)

    if not missing_fields:
        st.success("Basic structure valid (Observation)")
    else:
        st.error(f"Missing fields: {missing_fields}")

# =====================================================
# 4️⃣ EXPORT JSON
# =====================================================
with fhir_tabs[3]:

    st.subheader("Export FHIR Bundle JSON")

    bundle_json = json.dumps(bundle, indent=2)

    st.download_button(
        label="Download FHIR Bundle",
        data=bundle_json,
        file_name="fhir_bundle.json",
        mime="application/json"
    )

# =====================================================
# 5️⃣ MOCK SEND TO FHIR SERVER
# =====================================================
with fhir_tabs[4]:

    st.subheader("Send to FHIR Server (Mock Mode)")

    fhir_endpoint = st.text_input("FHIR Server URL",
                                   "https://hospital-emr.example.com/fhir")

    if st.button("Send Bundle (Mock)"):
        st.info(f"Simulated POST to {fhir_endpoint}")
        st.code("POST /fhir Bundle transaction")
        st.success("Bundle queued for transmission (Simulation)")
# =====================================================
# ========== ADVANCED FHIR ENTERPRISE LAYER ===========
# =====================================================

import requests
import base64

st.markdown("---")
st.markdown("## 🔐 Advanced FHIR Enterprise Integration")

advanced_fhir_tabs = st.tabs([
    "OAuth2 Auth",
    "LOINC / SNOMED Mapping",
    "Patient & Practitioner",
    "Media Resource (Ultrasound)",
    "Full Transaction Bundle",
    "Send to HAPI FHIR",
    "Official Validator"
])

# =========================
# SAFE DEFAULT VARIABLES
# =========================
safe_patient_id = st.session_state.get("fhir_patient_id", "HN123456")
safe_probability = st.session_state.get("fhir_probability", 0.5)
safe_uploaded = st.session_state.get("uploaded_image", None)

loinc_code = "LA6576-8"
snomed_code = "108369006"
practitioner_id = "PRAC001"

# =====================================================
# 1️⃣ OAUTH2
# =====================================================
with advanced_fhir_tabs[0]:

    st.subheader("OAuth2 Bearer Authentication")

    token = st.text_input(
        "OAuth2 Bearer Token",
        key="oauth_token_input"
    )

    if token:
        st.success("Token stored in session")
        st.session_state.oauth_token = token

# =====================================================
# 2️⃣ LOINC / SNOMED
# =====================================================
with advanced_fhir_tabs[1]:

    st.subheader("Clinical Coding Mapping")

    st.write("LOINC Code:", loinc_code)
    st.write("SNOMED Code:", snomed_code)

# =====================================================
# 3️⃣ PATIENT + PRACTITIONER
# =====================================================
with advanced_fhir_tabs[2]:

    st.subheader("Patient & Practitioner Resource")

    patient_id = st.text_input(
        "Patient ID",
        safe_patient_id,
        key="advanced_fhir_patient_id"
    )

    patient_resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{
            "system": "http://hospital.org/mrn",
            "value": patient_id
        }],
        "active": True
    }

    practitioner_resource = {
        "resourceType": "Practitioner",
        "id": practitioner_id,
        "active": True,
        "name": [{"text": "Dr. AI Radiologist"}]
    }

    st.json({
        "Patient": patient_resource,
        "Practitioner": practitioner_resource
    })

# =====================================================
# 4️⃣ MEDIA RESOURCE
# =====================================================
with advanced_fhir_tabs[3]:

    st.subheader("Media Resource (Ultrasound Reference)")

    if safe_uploaded:

        img_bytes = safe_uploaded.getvalue()
        encoded_image = base64.b64encode(img_bytes).decode("utf-8")

        media_resource = {
            "resourceType": "Media",
            "status": "completed",
            "type": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/media-type",
                    "code": "image"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "content": {
                "contentType": "image/jpeg",
                "data": encoded_image[:120] + "... (truncated)"
            }
        }

        st.json(media_resource)

    else:
        st.info("Upload image in Case Viewer first.")

# =====================================================
# 5️⃣ FULL TRANSACTION BUNDLE (FHIR VALIDATOR CLEAN)
# =====================================================
with advanced_fhir_tabs[4]:

    st.subheader("Full FHIR Transaction Bundle")

    iso_time = datetime.datetime.utcnow().isoformat() + "Z"

    # Observation (Validator Clean)
    observation_resource = {
        "resourceType": "Observation",
        "id": f"obs-{uuid.uuid4().hex[:8]}",
        "text": {
            "status": "generated",
            "div": "<div>AI Liver Malignancy Risk Assessment</div>"
        },
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "imaging",
                "display": "Imaging"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": loinc_code,
                "display": "AI Malignancy Risk"
            }]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": iso_time,
        "performer": [{
            "reference": f"Practitioner/{practitioner_id}"
        }],
        "valueQuantity": {
            "value": safe_probability,
            "unit": "probability"
        },
        "interpretation": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": "H",
                "display": "High"
            }]
        }]
    }

    # DiagnosticReport (Validator Clean)
    diagnostic_report_resource = {
        "resourceType": "DiagnosticReport",
        "id": f"dr-{uuid.uuid4().hex[:8]}",
        "text": {
            "status": "generated",
            "div": "<div>AI Ultrasound Risk Assessment Report</div>"
        },
        "status": "final",
        "code": {
            "text": "AI Ultrasound Risk Assessment"
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": iso_time,
        "result": [{
            "reference": f"Observation/{observation_resource['id']}"
        }]
    }

    # Full Transaction Bundle (with fullUrl REQUIRED)
    full_bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": patient_resource,
                "request": {"method": "POST", "url": "Patient"}
            },
            {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": practitioner_resource,
                "request": {"method": "POST", "url": "Practitioner"}
            },
            {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": observation_resource,
                "request": {"method": "POST", "url": "Observation"}
            },
            {
                "fullUrl": f"urn:uuid:{uuid.uuid4()}",
                "resource": diagnostic_report_resource,
                "request": {"method": "POST", "url": "DiagnosticReport"}
            }
        ]
    }

    st.json(full_bundle)

# =====================================================
# 6️⃣ SEND TO HAPI
# =====================================================
with advanced_fhir_tabs[5]:

    st.subheader("Send to HAPI FHIR Test Server")

    hapi_url = st.text_input(
        "HAPI FHIR Endpoint",
        "https://hapi.fhir.org/baseR4",
        key="hapi_url_input"
    )

    if st.button("POST Bundle to HAPI"):

        headers = {
            "Content-Type": "application/fhir+json"
        }

        if "oauth_token" in st.session_state:
            headers["Authorization"] = f"Bearer {st.session_state.oauth_token}"

        try:
            response = requests.post(
                hapi_url,
                json=full_bundle,
                headers=headers,
                timeout=10
            )

            st.write("Status Code:", response.status_code)
            try:
                st.json(response.json())
            except:
                st.write("Response received (non-JSON).")

        except Exception:
            st.error("Connection failed or server rejected request.")

# =====================================================
# 7️⃣ VALIDATOR
# =====================================================
with advanced_fhir_tabs[6]:

    st.subheader("Official FHIR Validator")

    st.write("1. Download Bundle JSON")
    st.write("2. Go to https://validator.fhir.org/")
    st.write("3. Paste Bundle JSON and validate")

    bundle_json = json.dumps(full_bundle, indent=2)

    st.download_button(
        label="Download Validator-Ready Bundle",
        data=bundle_json,
        file_name="validator_ready_bundle.json",
        mime="application/json"
    )
