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

st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# ======================================================
# DATABASE SAFE INIT
# ======================================================
conn = sqlite3.connect("audit.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS audit (
case_id TEXT,
organ TEXT,
prob REAL,
timestamp TEXT
)
""")
conn.commit()

# ======================================================
# MODEL REGISTRY
# ======================================================
MODEL_REGISTRY = {
    "Liver": {
        "status": "Deployed (v2.1 Binary)",
        "url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold_screening": 0.2835,
        "threshold_balanced": 0.5,
        "threshold_specificity": 0.7,
        "auc": "0.899 ± 0.03",
        "n": 735
    },
    "Thyroid": {
        "status": "Cross-validation complete (Mean CV AUC = 0.851)"
    },
    "Breast": {
        "status": "Dataset curation phase"
    },
    "Prostate": {
        "status": "Planned expansion"
    }
}

# ======================================================
# STYLE
# ======================================================
st.markdown("""
<style>
.big-title {font-size:28px;font-weight:700;}
.card {padding:20px;border-radius:12px;color:white;font-weight:600;text-align:center;}
.green {background:#27ae60;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;}
.gray {background:#2c3e50;}
.section {font-size:18px;font-weight:600;margin-top:20px;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL
# ======================================================
@st.cache_resource
def load_model(url):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model(MODEL_REGISTRY["Liver"]["url"])

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)

tabs = st.tabs(["Clinical AI","Performance","Monitoring","Platform Roadmap"])

# ======================================================
# 1️⃣ CLINICAL AI
# ======================================================
with tabs[0]:

    organ = st.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))

    if organ != "Liver":
        st.info(MODEL_REGISTRY[organ]["status"])
        st.stop()

    mode = st.selectbox("Decision Mode", 
                        ["Screening (High Sensitivity)",
                         "Balanced",
                         "High Specificity"])

    if mode.startswith("Screening"):
        threshold = MODEL_REGISTRY["Liver"]["threshold_screening"]
    elif mode == "Balanced":
        threshold = MODEL_REGISTRY["Liver"]["threshold_balanced"]
    else:
        threshold = MODEL_REGISTRY["Liver"]["threshold_specificity"]

    uploaded = st.file_uploader("Upload Liver Ultrasound Image", type=["jpg","png","jpeg"])

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

        if prob < 0.1:
            label="Likely Normal"
            color="green"
        elif prob < threshold:
            label="Likely Benign"
            color="yellow"
        else:
            label="Suspicious Malignant"
            color="red"

        case_id = str(uuid.uuid4())[:8]

        col1, col2 = st.columns([1.2,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"<div class='card {color}'>{label}</div>", unsafe_allow_html=True)
            st.write(f"Malignancy Probability: {round(prob*100,2)}%")
            st.write(f"Applied Threshold: {threshold}")

            st.markdown("### Clinical Recommendation")
            if label == "Likely Normal":
                st.write("Routine surveillance recommended.")
            elif label == "Likely Benign":
                st.write("Short-term imaging follow-up suggested.")
            else:
                st.write("Further diagnostic work-up recommended.")

            st.markdown("### Model Metadata")
            st.write("Model Version: Liver v2.1")
            st.write(f"Validation AUC: {MODEL_REGISTRY['Liver']['auc']}")
            st.write(f"Validation N: {MODEL_REGISTRY['Liver']['n']}")

        c.execute("""
        INSERT INTO audit (case_id, organ, prob, timestamp)
        VALUES (?, ?, ?, ?)
        """, (case_id,
              organ,
              float(prob),
              str(datetime.datetime.now())))
        conn.commit()

# ======================================================
# 2️⃣ PERFORMANCE
# ======================================================
with tabs[1]:

    st.markdown("### Model Performance Summary")
    st.write("Cross-Validated AUC:", MODEL_REGISTRY["Liver"]["auc"])
    st.write("Screening Sensitivity ≥95% (threshold optimized)")
    st.write("Binary classifier (Malignant vs Non-Malignant)")

# ======================================================
# 3️⃣ MONITORING
# ======================================================
with tabs[2]:

    df = pd.read_sql_query("SELECT * FROM audit", conn)

    st.write("Total Cases Logged:", len(df))

    if len(df) > 10:
        recent = df.tail(30)
        st.write("Recent Mean Risk:", round(recent["prob"].mean(),3))

# ======================================================
# 4️⃣ PLATFORM ROADMAP
# ======================================================
with tabs[3]:

    roadmap = pd.DataFrame({
        "Organ": ["Liver","Thyroid","Breast","Prostate"],
        "Status": [
            MODEL_REGISTRY["Liver"]["status"],
            MODEL_REGISTRY["Thyroid"]["status"],
            MODEL_REGISTRY["Breast"]["status"],
            MODEL_REGISTRY["Prostate"]["status"]
        ]
    })

    st.table(roadmap)
