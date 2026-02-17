importimport streamlit as st
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
import io

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Smart Biopsy Navigator – Enterprise Pro", layout="wide")

MODEL_REGISTRY = {
    "Liver": {
        "url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold": 0.2835,
        "version": "v2.1 Binary"
    },
    "Thyroid": {"url": None, "threshold": 0.5, "version": "Coming Soon"},
    "Breast": {"url": None, "threshold": 0.5, "version": "Planned"},
}

# =========================================================
# SAFE DATABASE
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
                (case_id, organ, role, prob, classification, str(datetime.datetime.now()))
            )
            conn.commit()
        except:
            pass

# =========================================================
# LOGIN SYSTEM
# =========================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.title("Smart Biopsy Navigator – Enterprise")
    st.caption("Hospital Deployment System")

    role = st.selectbox("User Role",
                        ["Radiologist", "Surgeon", "Oncologist", "Admin"])

    key = st.text_input("Access Key", type="password")

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
st.title("Smart Biopsy Navigator")
st.caption(f"Logged in as: {st.session_state.role}")

tabs = st.tabs(["Worklist", "Case Viewer", "Monitoring", "System Info"])

# =========================================================
# WORKLIST (PACS-style)
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

    if info["url"] is None:
        st.warning("Model not yet deployed.")
        st.stop()

    model = load_model(info["url"])
    threshold = st.slider("Operating Threshold", 0.1, 0.9, info["threshold"])

    uploaded = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            prob = torch.softmax(model(tensor), dim=1)[0][1].item()

        # Classification bands
        if prob < 0.1:
            label = "Normal"
            color = "green"
        elif prob < threshold:
            label = "Likely Benign"
            color = "orange"
        else:
            label = "Suspicious Malignant"
            color = "red"

        case_id = str(uuid.uuid4())[:8]

        col1, col2 = st.columns([1.2,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"### {label}")
            st.metric("Malignancy Probability", f"{round(prob*100,2)}%")

            # Risk Gauge
            fig, ax = plt.subplots()
            ax.axis("off")
            theta = np.linspace(0, math.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta))
            angle = math.pi*(1-prob)
            ax.plot([0,np.cos(angle)],[0,np.sin(angle)], linewidth=4)
            ax.text(0,-0.2,f"{round(prob*100,1)}%",ha="center")
            st.pyplot(fig)

        st.subheader("Structured Clinical Report")

        report = f"""
        Case ID: {case_id}
        Organ: {organ}
        Model Version: {info['version']}

        Predicted Malignancy Probability: {round(prob,4)}
        Applied Threshold: {threshold}

        Classification: {label}

        Recommended Action:
        """

        if label == "Normal":
            report += "Routine surveillance.\n"
        elif label == "Likely Benign":
            report += "Correlate clinically. Consider short interval follow-up or biopsy based on risk.\n"
        else:
            report += "High suspicion. Recommend biopsy and oncologic referral.\n"

        st.text_area("Report Preview", report, height=250)

        # Export as text file
        st.download_button(
            label="Export Report (TXT)",
            data=report,
            file_name=f"{case_id}_report.txt"
        )

        log_case(case_id, organ, st.session_state.role, prob, label)

# =========================================================
# MONITORING DASHBOARD
# =========================================================
with tabs[2]:

    st.subheader("Deployment Monitoring")

    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM cases", conn)
            if not df.empty:

                st.metric("Total Cases", len(df))
                st.metric("Mean Risk Score", round(df["prob"].mean(),3))

                st.line_chart(df["prob"])

                # Distribution
                fig, ax = plt.subplots()
                ax.hist(df["prob"], bins=20)
                ax.set_title("Risk Distribution")
                st.pyplot(fig)

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
    Smart Biopsy Navigator – Enterprise Pro Mode

    Intended Use:
    Clinical decision-support tool for ultrasound risk stratification.

    Not intended to replace physician judgment.
    """

    )

    st.write("Current Deployment Organ Models:")
    for organ in MODEL_REGISTRY:
        st.write(f"- {organ}: {MODEL_REGISTRY[organ]['version']}")
