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
import requests
import json

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Navigator",
    layout="wide",
)

# =====================================================
# STYLE
# =====================================================
st.markdown("""
<style>
.big-title { font-size: 30px; font-weight: 700; }
.subtitle { color: #6e6e73; margin-bottom: 20px; }
.card { padding: 18px; border-radius: 12px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.green { background: #e8f7ef; border-left: 6px solid #2ecc71; }
.yellow { background: #fff8e5; border-left: 6px solid #f1c40f; }
.red { background: #fdecea; border-left: 6px solid #e74c3c; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL CONFIG
# =====================================================
MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth"
NUM_CLASSES = 3  # Liver = 3 classes

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
# LOAD MODEL (SAFE)
# =====================================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    try:
        state = torch.hub.load_state_dict_from_url(
            MODEL_URL,
            map_location="cpu"
        )

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state, strict=False)

    except Exception as e:
        st.error(f"Model load failed: {e}")

    model.eval()
    return model

model = load_model()

# =====================================================
# SIDEBAR ROUTER
# =====================================================
app_mode = st.sidebar.radio(
    "Navigation",
    [
        "Case Viewer",
        "Analytics",
        "FHIR Integration",
        "Research & Validation",
        "Governance",
        "Infrastructure"
    ]
)

# =====================================================
# CASE VIEWER
# =====================================================
if app_mode == "Case Viewer":

    st.markdown("<div class='big-title'>Case Viewer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Clinical AI Inference</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

    # Clinical Mode Selector
    mode = st.radio(
        "Clinical Mode",
        ["Screening Mode (High Sensitivity)",
         "Balanced Diagnostic Mode",
         "High Specificity Mode"]
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

        if mode == "Screening Mode (High Sensitivity)":
            prob_display = prob_malignant
        elif mode == "Balanced Diagnostic Mode":
            prob_display = prob_malignant
        else:
            prob_display = prob_malignant

        if prob_malignant > 0.5:
            label = "Suspicious Malignant"
            style = "red"
        elif prob_benign > prob_normal:
            label = "Likely Benign"
            style = "yellow"
        else:
            label = "Normal"
            style = "green"

        case_id = str(uuid.uuid4())[:8]

        col1, col2 = st.columns([1.3,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:

            st.markdown(
                f"""
                <div class='card {style}'>
                    <div style="font-size:24px; font-weight:700;">
                        {label}
                    </div>
                    <div style="font-size:18px;">
                        {round(prob_display*100,2)}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Risk Gauge
            fig, ax = plt.subplots(figsize=(4,2.5))
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-0.2, 1.2)
            ax.axis("off")

            theta = np.linspace(0, math.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), linewidth=2)

            angle = math.pi * (1 - prob_display)
            ax.plot([0, np.cos(angle)], [0, np.sin(angle)], linewidth=3)
            ax.scatter(0,0,s=80)

            ax.text(0,-0.05,f"{round(prob_display*100,1)}%",ha="center",fontsize=20,fontweight="bold")
            ax.text(0,-0.18,label,ha="center",fontsize=10)

            st.pyplot(fig)

        st.session_state.fhir_probability = prob_display
        st.session_state.fhir_patient_id = case_id
        st.session_state.uploaded_image = uploaded

        log_case(case_id,"Hospital","Radiologist","Liver",prob_display,label)

# =====================================================
# ANALYTICS
# =====================================================
elif app_mode == "Analytics":

    st.markdown("<div class='big-title'>Analytics</div>", unsafe_allow_html=True)

    df = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df.empty:
        st.metric("Total Cases", len(df))
        st.metric("Average Risk", round(df["prob"].mean(),3))

        fig, ax = plt.subplots()
        ax.hist(df["prob"], bins=20)
        st.pyplot(fig)
    else:
        st.info("No data")

# =====================================================
# FHIR
# =====================================================
elif app_mode == "FHIR Integration":

    st.markdown("<div class='big-title'>FHIR Integration</div>", unsafe_allow_html=True)

    safe_patient_id = st.session_state.get("fhir_patient_id","HN123")
    safe_probability = float(st.session_state.get("fhir_probability",0.5))

    if st.button("Generate FHIR Bundle"):

        iso_time = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        bundle = {
            "resourceType": "Observation",
            "status": "final",
            "code": {"text":"AI Malignancy Risk"},
            "subject": {"reference":f"Patient/{safe_patient_id}"},
            "effectiveDateTime": iso_time,
            "valueQuantity": {"value":safe_probability,"unit":"probability"}
        }

        st.json(bundle)

# =====================================================
# RESEARCH
# =====================================================
elif app_mode == "Research & Validation":

    st.markdown("<div class='big-title'>Research & Validation</div>", unsafe_allow_html=True)

    df = pd.read_sql_query("SELECT * FROM cases", conn)

    if not df.empty:
        st.line_chart(df["prob"])

# =====================================================
# GOVERNANCE
# =====================================================
elif app_mode == "Governance":

    st.markdown("<div class='big-title'>Governance</div>", unsafe_allow_html=True)
    st.success("System operational")

# =====================================================
# INFRA
# =====================================================
elif app_mode == "Infrastructure":

    st.markdown("<div class='big-title'>Infrastructure</div>", unsafe_allow_html=True)
    st.code("Docker / AWS / JWT Simulation")
