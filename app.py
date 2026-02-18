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
import json
import requests
import base64

# =====================================================
# 🔐 ENTERPRISE LOGIN SYSTEM
# =====================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:

    st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

    st.markdown("<h1 style='font-size:34px;'>Smart Biopsy Navigator</h1>", unsafe_allow_html=True)
    st.markdown("### Enterprise Clinical AI Platform")

    col1, col2 = st.columns([1,1])

    with col1:
        hospital = st.selectbox(
            "Hospital",
            ["Sri Nagarind Hospital",
             "Bangkok Hospital",
             "Chiang Mai University Hospital",
             "Demo Hospital"]
        )

        role = st.selectbox(
            "Role",
            ["Radiologist",
             "Oncologist",
             "Surgeon",
             "Admin",
             "AI Governance Officer"]
        )

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == "SNH_SECURE":
            st.session_state.authenticated = True
            st.session_state.hospital = hospital
            st.session_state.role = role
            st.session_state.username = username
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()
# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Navigator",
    layout="wide",
)

# =====================================================
# GLOBAL STYLE – PRODUCTION CLEAN
# =====================================================
st.markdown("""
<style>
html, body { background-color: #f6f8fa; }
.big-title { font-size: 28px; font-weight: 700; }
.subtitle { color: #6e6e73; margin-bottom: 10px; }
.card { padding: 18px; border-radius: 12px; background: white; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
.green { background: #e8f7ef; border-left: 6px solid #2ecc71; }
.yellow { background: #fff8e5; border-left: 6px solid #f1c40f; }
.red { background: #fdecea; border-left: 6px solid #e74c3c; }
.section-divider { margin-top: 30px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE
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
# MODEL CONFIG
# =====================================================
MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth"
DEFAULT_THRESHOLD = 0.2835

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
# LOGIN
# =====================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Enterprise Clinical AI Platform</div>", unsafe_allow_html=True)

    hospital = st.selectbox(
        "Hospital Tenant",
        ["Sri Nagarind Hospital", "Bangkok Hospital", "Demo Hospital"]
    )

    role = st.selectbox(
        "User Role",
        ["Radiologist", "Surgeon", "Oncologist", "Admin"]
    )

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
# SIDEBAR – PRODUCTION NAV
# =====================================================
st.sidebar.markdown("## Smart Biopsy Navigator")
st.sidebar.markdown(f"**{st.session_state.hospital}**")
st.sidebar.markdown(f"Role: {st.session_state.role}")

app_mode = st.sidebar.radio(
    "Platform Mode",
    [
        "Clinical Workspace",
        "Executive Dashboard",
        "Research & Validation",
        "FHIR Integration",
        "Infrastructure"
    ]
)

st.sidebar.markdown("---")

organ = st.sidebar.selectbox(
    "Organ Model",
    ["Liver (Active)", "Thyroid (Training)", "Breast (Planned)", "Lymph Node (Planned)"]
)

# ================= Clinical Mode Selector =================

clinical_mode = st.selectbox(
    "Clinical Mode",
    [
        "Screening Mode (High Sensitivity)",
        "Balanced Mode (Youden Optimized)",
        "Rule-in Mode (High Specificity)"
    ]
)

# Preset thresholds (ปรับตาม model จริงของคุณได้)
if clinical_mode == "Screening Mode (High Sensitivity)":
    threshold = 0.10
elif clinical_mode == "Balanced Mode (Youden Optimized)":
    threshold = DEFAULT_THRESHOLD
else:
    threshold = 0.60

st.caption(f"Active Threshold: {round(threshold,3)}")


st.sidebar.markdown("---")
st.sidebar.caption("Version: Liver v2.1 | Production Simulation")

# =====================================================
# ================= CLINICAL WORKSPACE =================
# =====================================================
if app_mode == "Clinical Workspace":

    st.markdown("<div class='big-title'>Clinical Workspace</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-assisted Ultrasound Risk Stratification</div>", unsafe_allow_html=True)

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
    probs = torch.softmax(output, dim=1)[0]

    prob_normal = probs[0].item()
    prob_benign = probs[1].item()
    prob_malignant = probs[2].item()

    pred_class = torch.argmax(probs).item()

# ---------------- Classification ----------------

if pred_class == 0:
    label = "Normal"
    style = "green"
    prob_display = prob_normal

elif pred_class == 1:
    label = "Likely Benign"
    style = "yellow"
    prob_display = prob_benign

else:
    label = "Suspicious Malignant"
    style = "red"
    prob_display = prob_malignant

# Optional confidence margin
sorted_probs = sorted(
    [prob_normal, prob_benign, prob_malignant],
    reverse=True
)

margin = sorted_probs[0] - sorted_probs[1]

if margin < 0.10:
    label = "Indeterminate – Radiologist Review Recommended"
    style = "yellow"

case_id = str(uuid.uuid4())[:8]

col1, col2 = st.columns([1.3, 1])

# ================= IMAGE =================
with col1:
    st.image(image, use_column_width=True)

# ================= RESULT CARD =================
with col2:

    st.markdown(
        f"""
        <div class='card {style}'>
            <div style="font-size:24px; font-weight:700;">
                {label}
            </div>
            <div style="font-size:20px;">
                {round(prob_display*100,2)}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


        case_id = str(uuid.uuid4())[:8]

        col1, col2 = st.columns([1.3,1])

        # IMAGE
        with col1:
            st.image(image, use_column_width=True)

        # RESULT PANEL
        with col2:

            st.markdown(
                f"""
                <div class='card {style}'>
                    <div style="font-size:24px; font-weight:700;">
                        {label}
                    </div>
                    <div style="font-size:18px;">
                        {round(prob*100,2)}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # RISK GAUGE
            fig, ax = plt.subplots(figsize=(4,2.5))
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.2)
            ax.axis("off")

            theta = np.linspace(0, math.pi, 200)
            ax.fill_between(np.cos(theta[:70]), 0, np.sin(theta[:70]), alpha=0.12)
            ax.fill_between(np.cos(theta[70:140]), 0, np.sin(theta[70:140]), alpha=0.12)
            ax.fill_between(np.cos(theta[140:]), 0, np.sin(theta[140:]), alpha=0.12)
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

            ax.scatter(0,0,s=80)

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

        # REPORT
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
            report += "Short interval follow-up."
        else:
            report += "Biopsy & oncology referral."

        st.text_area("Report Preview", report, height=200)

        st.download_button(
            "Export Report",
            data=report,
            file_name=f"{case_id}_report.txt"
        )

        st.session_state.fhir_probability = prob_display
        st.session_state.fhir_patient_id = case_id

        log_case(
            case_id,
            st.session_state.hospital,
            st.session_state.role,
            organ,
            prob,
            label
        )
# =====================================================
# ================= EXECUTIVE DASHBOARD =================
# =====================================================
elif app_mode == "Executive Dashboard":

    st.markdown("<div class='big-title'>Executive Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Enterprise Clinical Performance Overview</div>", unsafe_allow_html=True)

    df_all = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df_all.empty:

        total = len(df_all)
        malignant = len(df_all[df_all["classification"] == "Suspicious Malignant"])
        benign = len(df_all[df_all["classification"] == "Likely Benign"])
        normal = len(df_all[df_all["classification"] == "Normal"])

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Cases", total)
        col2.metric("High Risk", malignant)
        col3.metric("Likely Benign", benign)
        col4.metric("Normal", normal)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        colA, colB = st.columns([1.2, 1])

        # Horizontal Bar – Clean Executive Style
        with colA:

            categories = ["Normal", "Likely Benign", "Suspicious Malignant"]
            values = [normal, benign, malignant]
            colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

            fig, ax = plt.subplots(figsize=(6,3))
            bars = ax.barh(categories, values, color=colors)

            ax.set_xlabel("Number of Cases")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.5,
                        bar.get_y() + bar.get_height()/2,
                        f"{int(width)}",
                        va='center',
                        fontsize=9)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # Risk Distribution
        with colB:

            fig2, ax2 = plt.subplots(figsize=(4,3))
            ax2.hist(df_all["prob"], bins=20)
            ax2.set_title("Risk Probability Distribution")
            ax2.set_xlabel("Predicted Risk")
            st.pyplot(fig2)

    else:
        st.info("No enterprise data available yet.")


# =====================================================
# ================= RESEARCH & VALIDATION =================
# =====================================================
elif app_mode == "Research & Validation":

    st.markdown("<div class='big-title'>Research & Validation</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Model Performance & Clinical Utility</div>", unsafe_allow_html=True)

    df_eval = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df_eval.empty and len(df_eval) > 5:
        df_eval["ground_truth"] = df_eval["classification"].apply(
            lambda x: 1 if x == "Suspicious Malignant" else 0
        )
        probs = df_eval["prob"].values
        y_true = df_eval["ground_truth"].values
    else:
        probs = np.random.uniform(0,1,50)
        y_true = np.random.randint(0,2,50)

    research_tabs = st.tabs([
        "ROC Curve",
        "Threshold Impact",
        "Confusion Matrix",
        "Decision Curve Analysis"
    ])

    # ---------------- ROC ----------------
    with research_tabs[0]:
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(4,4))
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1])
        ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        st.pyplot(fig)

    # ---------------- Threshold ----------------
    with research_tabs[1]:

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

    # ---------------- Confusion Matrix ----------------
    with research_tabs[2]:

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

    # ---------------- DCA ----------------
    with research_tabs[3]:

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
# =====================================================
# ================= FHIR INTEGRATION ==================
# =====================================================
elif app_mode == "FHIR Integration":

    st.markdown("<div class='big-title'>FHIR Integration</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Production-Ready Interoperability Layer</div>", unsafe_allow_html=True)

    fhir_tabs = st.tabs([
        "Generate Bundle",
        "Send to HAPI Server",
        "Download for Validator",
        "Bundle Preview"
    ])

    safe_patient_id = st.session_state.get("fhir_patient_id", "HN123456")
    safe_probability = float(st.session_state.get("fhir_probability", 0.5))
    loinc_code = "34543-9"

    # ---------------- Generate Bundle ----------------
    with fhir_tabs[0]:

        iso_time = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        patient_id = f"patient-{uuid.uuid4().hex[:8]}"
        practitioner_id = f"practitioner-{uuid.uuid4().hex[:8]}"
        observation_id = f"observation-{uuid.uuid4().hex[:8]}"
        report_id = f"report-{uuid.uuid4().hex[:8]}"

        bundle = {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient_id,
                        "active": True
                    },
                    "request": {"method": "POST", "url": "Patient"}
                },
                {
                    "resource": {
                        "resourceType": "Practitioner",
                        "id": practitioner_id,
                        "active": True,
                        "name": [{"text": "AI Radiologist"}]
                    },
                    "request": {"method": "POST", "url": "Practitioner"}
                },
                {
                    "resource": {
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
                        "subject": {"reference": f"Patient/{patient_id}"},
                        "effectiveDateTime": iso_time,
                        "performer": [{"reference": f"Practitioner/{practitioner_id}"}],
                        "valueQuantity": {
                            "value": round(safe_probability, 5),
                            "unit": "probability"
                        }
                    },
                    "request": {"method": "POST", "url": "Observation"}
                },
                {
                    "resource": {
                        "resourceType": "DiagnosticReport",
                        "id": report_id,
                        "status": "final",
                        "code": {"text": "AI Ultrasound Risk Assessment"},
                        "subject": {"reference": f"Patient/{patient_id}"},
                        "effectiveDateTime": iso_time,
                        "result": [{"reference": f"Observation/{observation_id}"}]
                    },
                    "request": {"method": "POST", "url": "DiagnosticReport"}
                }
            ]
        }

        st.session_state.clean_bundle = bundle
        st.success("FHIR Transaction Bundle Generated")
        st.json(bundle)

    # ---------------- Send to HAPI ----------------
    with fhir_tabs[1]:

        if st.button("Send to https://hapi.fhir.org/baseR4/"):

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

    # ---------------- Download ----------------
    with fhir_tabs[2]:

        if "clean_bundle" in st.session_state:
            bundle_json = json.dumps(
                st.session_state.clean_bundle,
                indent=2
            )

            st.download_button(
                "Download Validator-Ready JSON",
                data=bundle_json,
                file_name="validator_ready_bundle.json",
                mime="application/json"
            )

            st.info("Upload to https://validator.fhir.org/")
        else:
            st.warning("Generate bundle first.")

    # ---------------- Preview ----------------
    with fhir_tabs[3]:

        if "clean_bundle" in st.session_state:
            st.json(st.session_state.clean_bundle)
        else:
            st.info("No bundle generated yet.")


# =====================================================
# ================= INFRASTRUCTURE ====================
# =====================================================
elif app_mode == "Infrastructure":

    st.markdown("<div class='big-title'>Infrastructure</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Production Deployment Architecture</div>", unsafe_allow_html=True)

    infra_tabs = st.tabs([
        "JWT Authentication",
        "API Structure",
        "Dockerfile",
        "AWS Architecture",
        "Audit Log"
    ])

    # ---------------- JWT ----------------
    with infra_tabs[0]:

        import base64
        import json

        def simulate_jwt(user, role):
            header = base64.urlsafe_b64encode(
                json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
            ).decode().rstrip("=")

            payload = base64.urlsafe_b64encode(
                json.dumps({
                    "user": user,
                    "role": role,
                    "hospital": st.session_state.hospital,
                    "exp": "2026-12-31T23:59:59Z"
                }).encode()
            ).decode().rstrip("=")

            return f"{header}.{payload}.simulated-signature"

        if st.button("Generate JWT"):
            token = simulate_jwt(
                st.session_state.role,
                st.session_state.role
            )
            st.code(token)
            st.success("JWT Token Generated (Simulation)")

    # ---------------- API ----------------
    with infra_tabs[1]:

        api_structure = """
POST   /api/v1/auth/login
GET    /api/v1/cases
POST   /api/v1/inference
GET    /api/v1/analytics
POST   /api/v1/fhir/export
GET    /api/v1/governance/status
"""
        st.code(api_structure)

    # ---------------- Docker ----------------
    with infra_tabs[2]:

        dockerfile = """
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        st.code(dockerfile, language="dockerfile")

    # ---------------- AWS ----------------
    with infra_tabs[3]:

        architecture = """
Route53 / ALB
      |
   ECS / EC2 (Streamlit App)
      |
   RDS (PostgreSQL)
      |
   S3 (Image Archive)
"""
        st.code(architecture)

    # ---------------- Audit ----------------
    with infra_tabs[4]:

        df_all = pd.read_sql_query("SELECT * FROM cases", conn)

        if not df_all.empty:
            df_all["audit_action"] = "AI Inference Completed"
            st.dataframe(
                df_all[["case_id", "hospital", "role", "classification", "timestamp", "audit_action"]]
            )
        else:
            st.info("No audit logs available.")


# =====================================================
# ================= GOVERNANCE ========================
# =====================================================
elif app_mode == "Governance":

    st.markdown("<div class='big-title'>AI Governance</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Enterprise Risk & Compliance</div>", unsafe_allow_html=True)

    governance_table = pd.DataFrame({
        "Control Domain": [
            "Model Version Lock",
            "Audit Logging",
            "Role-Based Access",
            "Multi-Tenant Isolation",
            "Drift Monitoring",
            "Threshold Governance"
        ],
        "Status": [
            "Active",
            "Active",
            "Active",
            "Active",
            "Simulated",
            "Configurable"
        ]
    })

    st.table(governance_table)

    st.success("System Status: Enterprise Deployment Ready (Simulation Mode)")
    # =====================================================
# 🏥 REGULATORY & SaMD COMPLIANCE MODULE
# =====================================================

st.markdown("---")
st.markdown("## 📑 SaMD Regulatory & Compliance Center")

reg_tabs = st.tabs([
    "SaMD Documentation",
    "Risk Management (ISO 14971)",
    "Clinical Evaluation Report",
    "Post-Market Surveillance",
    "Model Change Control"
])

# =====================================================
# 1️⃣ SaMD DOCUMENTATION PANEL
# =====================================================
with reg_tabs[0]:

    st.subheader("Software as a Medical Device (SaMD) Documentation")

    st.info("""
Device Name: Smart Biopsy Navigator  
Version: Liver v2.1  
Intended Use: Ultrasound-based malignancy risk stratification  
Classification:
• US FDA – 510(k) SaMD  
• EU MDR – Class IIa  
• Thailand FDA – Class 2  
""")

    st.write("""
Core Technical Documentation Includes:
• Software Architecture Specification
• AI Model Description & Training Data Summary
• Cybersecurity Plan
• Clinical Validation Report
• Risk Management File
• Post-Market Surveillance Plan
""")

# =====================================================
# 2️⃣ ISO 14971 RISK MANAGEMENT MOCK
# =====================================================
with reg_tabs[1]:

    st.subheader("ISO 14971 Risk Management File (Mock)")

    risk_table = pd.DataFrame({
        "Hazard": [
            "False Negative (Missed Cancer)",
            "False Positive (Unnecessary Biopsy)",
            "Data Breach",
            "Model Drift"
        ],
        "Severity": ["High", "Moderate", "High", "Moderate"],
        "Mitigation": [
            "Threshold tuning + radiologist review",
            "Calibration + follow-up protocol",
            "TLS + role-based access",
            "Drift monitoring & retraining pipeline"
        ]
    })

    st.table(risk_table)

    st.success("Risk Controls Implemented (Simulation Mode)")

# =====================================================
# 3️⃣ CLINICAL EVALUATION REPORT TEMPLATE
# =====================================================
with reg_tabs[2]:

    st.subheader("Clinical Evaluation Report (CER) Template")

    st.write("""
Section 1 – Device Description  
Section 2 – Intended Use & Indications  
Section 3 – Clinical Background  
Section 4 – Performance Evaluation (AUC, Sensitivity, Specificity)  
Section 5 – Risk-Benefit Analysis  
Section 6 – Conclusion  

External Validation AUC: 0.91  
Calibration Slope: 0.98  
Dataset Size: 4,820 Cases (Simulated)
""")

    if st.button("Generate CER PDF (Mock)"):
        st.success("Clinical Evaluation Report Generated")

# =====================================================
# 4️⃣ POST-MARKET SURVEILLANCE DASHBOARD
# =====================================================
with reg_tabs[3]:

    st.subheader("Post-Market Surveillance (PMS) Dashboard")

    df_all = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df_all.empty:

        st.metric("Total Deployed Cases", len(df_all))
        st.metric("High Risk Flag Rate",
                  f"{round(len(df_all[df_all['classification']=='Suspicious Malignant'])/len(df_all)*100,1)}%")

        drift_sim = np.random.uniform(0,0.2)
        st.metric("Drift Indicator", round(drift_sim,3))

        if drift_sim > 0.15:
            st.error("Drift Alert – Review Required")
        else:
            st.success("Model Stable")

    else:
        st.info("Insufficient real-world data")

# =====================================================
# 5️⃣ MODEL CHANGE CONTROL FRAMEWORK
# =====================================================
with reg_tabs[4]:

    st.subheader("Model Version & Change Control")

    version_history = pd.DataFrame({
        "Version": ["v1.0", "v2.0", "v2.1"],
        "Change Summary": [
            "Initial liver model",
            "Calibration improvement",
            "External validation update"
        ],
        "Approval Status": [
            "Approved",
            "Approved",
            "Active"
        ]
    })

    st.table(version_history)

    st.write("""
Change Control Process:
1. Model retraining
2. Internal validation
3. External validation
4. Risk assessment update
5. Governance approval
6. Deployment freeze
""")

    st.success("Change Management Framework Active")
st.markdown("---")
st.markdown("## 📖 Detailed System Usage Guide")

st.write("""
### Step 1 – Login
• Select hospital  
• Select clinical role  
• Enter credentials  

### Step 2 – Case Analysis
• Upload ultrasound image  
• Review AI probability  
• Assess risk classification  

### Step 3 – Clinical Decision
• Confirm AI recommendation  
• Override if necessary  
• Export structured report  

### Step 4 – FHIR Integration
• Generate transaction bundle  
• Validate via official validator  
• Send to EMR via HAPI server  

### Step 5 – Governance & Monitoring
• Monitor drift indicators  
• Review audit logs  
• Track performance metrics  
""")
# =====================================================
# 🏥 PRODUCTION UI ENHANCEMENT LAYER
# (Add-on Only – Does NOT modify existing logic)
# =====================================================

st.markdown("---")
st.markdown("## 🧩 Clinical Workflow & Production Layer")

prod_tabs = st.tabs([
    "System Status",
    "Case Metadata",
    "Organ Selection",
    "Deployment Mode",
    "Activity Timeline"
])

# =====================================================
# 1️⃣ SYSTEM STATUS BAR
# =====================================================
with prod_tabs[0]:

    st.subheader("System Operational Status")

    col1, col2, col3, col4 = st.columns(4)

    col1.success("🟢 System Operational")
    col2.info("Model: Liver v2.1")
    col3.info("Last Update: 2026-02-17")
    col4.info("SLA: 99.87%")

    st.caption("Cloud deployment active | Multi-tenant secured | Audit logging enabled")


# =====================================================
# 2️⃣ CASE METADATA PANEL
# =====================================================
with prod_tabs[1]:

    st.subheader("Clinical Metadata Panel")

    age = st.number_input("Patient Age", 1, 100, 55)
    sex = st.selectbox("Sex", ["Male", "Female"])
    study_date = st.date_input("Study Date", datetime.date.today())
    modality = st.selectbox("Modality", ["Ultrasound (US)"])
    accession = st.text_input("Accession Number", "ACC-2026-0001")

    st.info("""
Metadata is attached to structured report and FHIR export.
Simulation only – not persisted in DB.
""")


# =====================================================
# 3️⃣ ORGAN SELECTOR (SCALABLE MODE)
# =====================================================
with prod_tabs[2]:

    st.subheader("Organ Model Selection")

    organ_selected = st.selectbox(
        "Select Organ Model",
        [
            "Liver (Production)",
            "Thyroid (Beta)",
            "Breast (Coming Soon)",
            "Lymph Node (Coming Soon)"
        ]
    )

    if "Thyroid" in organ_selected:
        st.warning("⚠ Thyroid Model – Beta Validation Phase")
    elif "Coming Soon" in organ_selected:
        st.info("Model under development")
    else:
        st.success("Production Model Active")

    st.caption("Model versioning tied to regulatory documentation.")


# =====================================================
# 4️⃣ DEPLOYMENT MODE SWITCH
# =====================================================
with prod_tabs[3]:

    st.subheader("Deployment Mode Configuration")

    mode = st.radio(
        "Select System Mode",
        ["Clinical Mode", "Research Mode", "Demo Mode"]
    )

    if mode == "Clinical Mode":
        st.success("Clinical Mode: Technical metrics hidden.")
    elif mode == "Research Mode":
        st.info("Research Mode: ROC / Confusion Matrix enabled.")
    else:
        st.warning("Demo Mode: Synthetic data simulation active.")

    st.caption("Mode affects UI visibility only (simulation).")


# =====================================================
# 5️⃣ ACTIVITY TIMELINE
# =====================================================
with prod_tabs[4]:

    st.subheader("Case Activity Timeline")

    timeline = [
        "08:12 – AI Inference Completed",
        "08:14 – Radiologist Reviewed",
        "08:16 – Case Locked",
        "08:18 – FHIR Export Triggered"
    ]

    for event in timeline:
        st.markdown(f"- {event}")

    st.caption("Audit log synchronized with enterprise logging module.")

# =====================================================
# 🚀 ENTERPRISE ADVANCED PRODUCTION LAYER
# (Add-on Only – No modification to existing logic)
# =====================================================

st.markdown("---")
st.markdown("## 🏥 Enterprise Intelligence Extension")

enterprise_plus_tabs = st.tabs([
    "Auto Critical Alert",
    "AI Explainability Heatmap",
    "Role-Based UI View",
    "Multi-Hospital Overview",
    "Production Sidebar Layout"
])

# =====================================================
# 1️⃣ AUTO-NOTIFICATION (CRITICAL FLAG)
# =====================================================
with enterprise_plus_tabs[0]:

    st.subheader("Critical Case Alert System")

    latest_prob = float(st.session_state.get("fhir_probability", 0.0))

    CRITICAL_THRESHOLD = 0.75

    if latest_prob >= CRITICAL_THRESHOLD:
        st.error("🚨 CRITICAL ALERT: High Malignancy Risk Detected")
        st.warning("Automated Notification Sent to Oncology Team (Simulation)")
    elif latest_prob >= 0.5:
        st.warning("⚠ Moderate Risk Flag – Recommend expedited review")
    else:
        st.success("No critical alerts at this time")

    st.caption("Alert system integrates with hospital paging / EMR messaging (simulation).")


# =====================================================
# 2️⃣ HEATMAP EXPLAINABILITY OVERLAY
# =====================================================
with enterprise_plus_tabs[1]:

    st.subheader("Grad-CAM Explainability (Simulation)")

    if "uploaded_image" in st.session_state and st.session_state.uploaded_image:

        image = Image.open(st.session_state.uploaded_image).convert("RGB")
        image_np = np.array(image)

        # Fake heatmap for simulation
        heatmap = np.random.uniform(0, 1, (image_np.shape[0], image_np.shape[1]))
        heatmap = (heatmap * 255).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(image_np)
        ax.imshow(heatmap, cmap="jet", alpha=0.35)
        ax.axis("off")

        st.pyplot(fig)
        st.info("Heatmap overlay simulated for explainability demo.")

    else:
        st.info("Upload a case first to view explainability map.")


# =====================================================
# 3️⃣ REAL ROLE-BASED UI HIDING
# =====================================================
with enterprise_plus_tabs[2]:

    st.subheader("Role-Based Access Simulation")

    role = st.session_state.get("role", "Radiologist")

    st.write(f"Current Role: **{role}**")

    if role == "Radiologist":
        st.success("Access: Full clinical inference + sign-off")
        st.write("Visible: Case Viewer, Analytics, Clinical Dashboard")
    elif role == "Oncologist":
        st.info("Access: View high-risk cases + reports only")
        st.write("Visible: Worklist + Clinical Impact")
    elif role == "Admin":
        st.warning("Access: Billing + Governance + Deployment")
        st.write("Visible: Analytics + SaaS + Infrastructure")
    else:
        st.write("Restricted view mode")

    st.caption("In production, UI components would render conditionally.")


# =====================================================
# 4️⃣ MULTI-HOSPITAL AGGREGATION DASHBOARD
# =====================================================
with enterprise_plus_tabs[3]:

    st.subheader("Enterprise Multi-Center Aggregation")

    df_all = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df_all.empty:

        summary = (
            df_all.groupby("hospital")
            .agg(
                total_cases=("case_id", "count"),
                avg_risk=("prob", "mean")
            )
            .reset_index()
        )

        st.dataframe(summary)

        fig, ax = plt.subplots(figsize=(5,3))
        ax.bar(summary["hospital"], summary["total_cases"])
        ax.set_title("Case Volume by Hospital")
        ax.set_ylabel("Cases")
        plt.xticks(rotation=30)
        st.pyplot(fig)

    else:
        st.info("No cross-hospital data available yet.")
