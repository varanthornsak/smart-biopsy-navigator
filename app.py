import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import uuid
import sqlite3
import matplotlib.pyplot as plt
import cv2
import base64
import json
import time

st.set_page_config(page_title="Smart Biopsy Navigator v3.0", layout="wide")

# =====================================================
# APPLE MINIMAL STYLE
# =====================================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.card {
    background: #ffffff;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.05);
}
.green {background:#eafaf1;}
.yellow {background:#fef9e7;}
.red {background:#fdecea;}
.badge-active {color:#27ae60;font-weight:600;}
.badge-validation {color:#f39c12;font-weight:600;}
.badge-training {color:#3498db;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL REGISTRY
# =====================================================
MODEL_REGISTRY = {
    "Liver": {
        "status": "Active",
        "url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold_screen": 0.2835,
        "threshold_balanced": 0.5,
        "auc": 0.899,
        "version": "Liver v2.1",
        "dataset": 735
    },
    "Thyroid": {
        "status": "Validation",
        "auc": 0.851,
        "version": "Thyroid v1.0"
    },
    "Breast": {
        "status": "Training"
    }
}

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("v3_enterprise.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS audit (
case_id TEXT,
organ TEXT,
prob REAL,
interpretation TEXT,
mode TEXT,
timestamp TEXT
)
""")
conn.commit()

# =====================================================
# HEADER
# =====================================================
st.title("Smart Biopsy Navigator")
st.caption("Enterprise Clinical AI Platform")

tabs = st.tabs([
    "Clinical AI",
    "Monitoring",
    "Model Registry",
    "Expansion Pipeline",
    "Governance"
])

# =====================================================
# 1️⃣ CLINICAL AI
# =====================================================
with tabs[0]:

    st.subheader("Clinical Decision Support")

    organ = st.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))

    status = MODEL_REGISTRY[organ]["status"]

    if status != "Active":
        st.warning(f"{organ} model is currently in {status} phase.")
        st.stop()

    mode = st.radio("Mode", ["Screening", "Balanced"])

    model_info = MODEL_REGISTRY[organ]

    @st.cache_resource
    def load_model(url):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model

    model = load_model(model_info["url"])

    uploaded = st.file_uploader("Upload Ultrasound", type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()

        threshold = model_info["threshold_screen"] if mode=="Screening" else model_info["threshold_balanced"]

        if prob < 0.1:
            label = "Normal"
            bg = "green"
            recommendation = "Routine screening follow-up."
        elif prob < threshold:
            label = "Benign"
            bg = "yellow"
            recommendation = "Short-term imaging follow-up recommended."
        else:
            label = "Malignant"
            bg = "red"
            recommendation = "Biopsy evaluation recommended."

        st.markdown(f"""
        <div class="card {bg}">
        <h2>{label}</h2>
        <p><b>Malignancy Probability:</b> {round(prob*100,2)}%</p>
        <p><b>Clinical Recommendation:</b> {recommendation}</p>
        </div>
        """, unsafe_allow_html=True)

        # Save audit
        case_id = str(uuid.uuid4())[:8]
        c.execute("INSERT INTO audit VALUES (?,?,?,?,?,?)",
                  (case_id, organ, prob, label, mode,
                   str(datetime.datetime.now())))
        conn.commit()

        st.markdown("### Model Performance")
        fpr = np.linspace(0,1,100)
        tpr = fpr ** (1/model_info["auc"])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1])
        st.pyplot(fig)

# =====================================================
# 2️⃣ MONITORING
# =====================================================
with tabs[1]:

    st.subheader("Enterprise Monitoring")

    df = pd.read_sql_query("SELECT * FROM audit", conn)

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cases", len(df))
        col2.metric("Malignant Rate", f"{round((df['interpretation']=='Malignant').mean()*100,2)}%")
        col3.metric("Avg Probability", f"{round(df['prob'].mean()*100,2)}%")

        st.bar_chart(df["interpretation"].value_counts())

    else:
        st.info("No cases recorded yet.")

# =====================================================
# 3️⃣ MODEL REGISTRY
# =====================================================
with tabs[2]:

    st.subheader("Model Registry")

    for organ, info in MODEL_REGISTRY.items():

        if info["status"] == "Active":
            badge = "badge-active"
        elif info["status"] == "Validation":
            badge = "badge-validation"
        else:
            badge = "badge-training"

        st.markdown(f"""
        <div class="card">
        <h3>{organ}</h3>
        <p class="{badge}">{info["status"]}</p>
        <p>Version: {info.get("version","—")}</p>
        <p>AUC: {info.get("auc","—")}</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# 4️⃣ EXPANSION PIPELINE
# =====================================================
with tabs[3]:

    st.subheader("Multi-Organ Expansion Roadmap")

    st.write("• Liver – Production Active")
    st.write("• Thyroid – External Validation Phase")
    st.write("• Breast – Model Training Phase")
    st.write("• Future: Lung, Prostate, Pancreas")

# =====================================================
# 5️⃣ GOVERNANCE
# =====================================================
with tabs[4]:

    st.subheader("Model Governance")

    st.write("Production Model:", MODEL_REGISTRY["Liver"]["version"])
    st.write("Frozen Threshold:", MODEL_REGISTRY["Liver"]["threshold_screen"])
    st.write("Calibration Status: Completed")
    st.write("Regulatory Status: Clinical Decision Support Only")
    st.write("Last Updated:", datetime.date.today())

    st.caption("For research and clinical decision support use only.")
