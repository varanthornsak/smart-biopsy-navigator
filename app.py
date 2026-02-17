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

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Biopsy Navigator – Enterprise",
    layout="wide"
)

# =========================================================
# MODEL REGISTRY
# =========================================================
MODEL_REGISTRY = {
    "Liver": {
        "url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold": 0.2835,
        "version": "v2.1 Binary"
    }
}

# =========================================================
# DATABASE INIT (CLOUD SAFE)
# =========================================================
def init_db():
    try:
        conn = sqlite3.connect("enterprise.db", check_same_thread=False)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT,
            organ TEXT,
            role TEXT,
            prob REAL,
            classification TEXT,
            timestamp TEXT
        )
        """)
        conn.commit()
        return conn
    except:
        return None

conn = init_db()

def log_case(case_id, organ, role, prob, classification):
    if conn:
        try:
            conn.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, organ, role, float(prob), classification, str(datetime.datetime.now()))
            )
            conn.commit()
        except:
            pass

# =========================================================
# LOGIN
# =========================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.title("Smart Biopsy Navigator")
    st.caption("Enterprise Hospital Deployment")

    role = st.selectbox(
        "User Role",
        ["Radiologist", "Surgeon", "Oncologist", "Admin"]
    )

    key = st.text_input("Hospital Access Key", type="password")

    if st.button("Login"):
        if key == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.role = role
        else:
            st.error("Invalid access key")

    st.stop()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(url):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# =========================================================
# HEADER
# =========================================================
st.title("Smart Biopsy Navigator – Enterprise")
st.caption(f"Logged in as: {st.session_state.role}")

tabs = st.tabs(["Worklist", "Case Viewer", "Monitoring", "System Info"])

# =========================================================
# WORKLIST
# =========================================================
with tabs[0]:

    st.subheader("Case Worklist")

    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM cases", conn)
            if not df.empty:
                st.dataframe(df.sort_values("timestamp", ascending=False))
            else:
                st.info("No cases processed yet.")
        except:
            st.info("Database unavailable.")

# =========================================================
# CASE VIEWER
# =========================================================
with tabs[1]:

    st.subheader("New Case")

    organ = st.selectbox("Organ", list(MODEL_REGISTRY.keys()))
    info = MODEL_REGISTRY[organ]

    model = load_model(info["url"])
    threshold = st.slider(
        "Operating Threshold",
        0.1,
        0.9,
        info["threshold"]
    )

    uploaded = st.file_uploader(
        "Upload Ultrasound Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()

        if prob < 0.1:
            label = "Normal"
        elif prob < threshold:
            label = "Likely Benign"
        else:
            label = "Suspicious Malignant"

        case_id = str(uuid.uuid4())[:8]

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            st.metric("Malignancy Probability", f"{round(prob*100,2)}%")
            st.write("Classification:", label)

            fig, ax = plt.subplots()
            ax.axis("off")
            theta = np.linspace(0, math.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta))
            angle = math.pi * (1 - prob)
            ax.plot([0, np.cos(angle)], [0, np.sin(angle)], linewidth=4)
            ax.text(0, -0.2, f"{round(prob*100,1)}%", ha="center")
            st.pyplot(fig)

        report = f"""
Case ID: {case_id}
Organ: {organ}
Model Version: {info['version']}

Predicted Probability: {round(prob,4)}
Threshold: {threshold}
Classification: {label}

Recommendation:
"""

        if label == "Normal":
            report += "Routine follow-up."
        elif label == "Likely Benign":
            report += "Consider interval imaging or biopsy based on clinical risk."
        else:
            report += "Recommend biopsy and oncologic referral."

        st.text_area("Report Preview", report, height=200)

        st.download_button(
            "Export Report",
            data=report,
            file_name=f"{case_id}_report.txt"
        )

        log_case(case_id, organ, st.session_state.role, prob, label)

# =========================================================
# MONITORING
# =========================================================
with tabs[2]:

    st.subheader("Deployment Monitoring")

    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM cases", conn)
            if not df.empty:
                st.metric("Total Cases", len(df))
                st.metric("Average Risk Score", round(df["prob"].mean(), 3))
                st.line_chart(df["prob"])
            else:
                st.info("No data yet.")
        except:
            st.info("Monitoring unavailable.")

# =========================================================
# SYSTEM INFO
# =========================================================
with tabs[3]:

    st.subheader("System Information")

    st.write("""
Smart Biopsy Navigator – Enterprise Mode

Intended Use:
Clinical decision-support tool for ultrasound risk stratification.

Not intended to replace physician judgment.
""")
