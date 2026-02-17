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
# CONFIG
# =========================================================
st.set_page_config(page_title="Smart Biopsy Navigator – Hospital Mode", layout="wide")

MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth"
DEFAULT_THRESHOLD = 0.2835

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.big-title {font-size:34px;font-weight:700;}
.subtitle {color:#6b7280;margin-bottom:20px;}
.section {font-size:22px;font-weight:600;margin-top:30px;}
.card {padding:22px;border-radius:16px;font-weight:600;text-align:center;}
.green {background:#27ae60;color:white;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;color:white;}
.metric-box {background:#f4f6f9;padding:15px;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SAFE AUDIT DB
# =========================================================
def init_db():
    try:
        conn = sqlite3.connect("audit.db", check_same_thread=False)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS audit (
            case_id TEXT,
            role TEXT,
            organ TEXT,
            prob REAL,
            timestamp TEXT
        )
        """)
        conn.commit()
        return conn
    except:
        return None

conn = init_db()

def safe_log(case_id, role, organ, prob):
    if conn:
        try:
            conn.execute("INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
                         (case_id, role, organ, prob, str(datetime.datetime.now())))
            conn.commit()
        except:
            pass

# =========================================================
# LOGIN PAGE
# =========================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Clinical Decision Support System</div>", unsafe_allow_html=True)

    role = st.selectbox("Select Role",
                        ["Radiologist", "Surgeon", "Oncologist", "Admin"])

    hospital_key = st.text_input("Hospital Access Key", type="password")

    if st.button("Login"):
        if hospital_key == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.role = role
        else:
            st.error("Invalid access key.")

    st.stop()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>Logged in as: {st.session_state.role}</div>", unsafe_allow_html=True)

tabs = st.tabs(["Clinical Assessment", "Monitoring", "User Guide"])

# =========================================================
# CLINICAL ASSESSMENT
# =========================================================
with tabs[0]:

    st.markdown("<div class='section'>Step 1 – Upload Ultrasound</div>", unsafe_allow_html=True)

    organ = st.selectbox("Organ", ["Liver"])
    threshold = st.slider("Operating Threshold", 0.1, 0.9, DEFAULT_THRESHOLD)

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1)[0][1].item()

        # 3-band logic
        if prob < 0.1:
            label = "Normal"
            color = "green"
        elif prob < threshold:
            label = "Likely Benign"
            color = "yellow"
        else:
            label = "Suspicious Malignant"
            color = "red"

        case_id = str(uuid.uuid4())[:8]

        col1, col2 = st.columns([1.2,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"<div class='card {color}'>{label}<br>{round(prob*100,2)}%</div>", unsafe_allow_html=True)

            # Risk Gauge
            fig, ax = plt.subplots()
            ax.axis("off")
            theta = np.linspace(0, math.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta))
            angle = math.pi*(1-prob)
            ax.plot([0,np.cos(angle)],[0,np.sin(angle)], linewidth=4)
            ax.text(0,-0.2,f"{round(prob*100,1)}%",ha="center")
            st.pyplot(fig)

        st.markdown("<div class='section'>Clinical Interpretation</div>", unsafe_allow_html=True)

        interpretation_table = pd.DataFrame({
            "Parameter": [
                "Predicted Probability",
                "Applied Threshold",
                "Classification"
            ],
            "Value": [
                round(prob,4),
                threshold,
                label
            ]
        })

        st.table(interpretation_table)

        # Suggested Action
        st.markdown("### Recommended Next Step")

        if label == "Normal":
            st.success("Routine follow-up recommended.")
        elif label == "Likely Benign":
            st.warning("Correlate with ultrasound morphology. Consider interval imaging or biopsy based on clinical risk.")
        else:
            st.error("High suspicion. Recommend biopsy and oncologic referral.")

        safe_log(case_id, st.session_state.role, organ, prob)

# =========================================================
# MONITORING
# =========================================================
with tabs[1]:

    st.markdown("<div class='section'>System Monitoring</div>", unsafe_allow_html=True)

    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM audit", conn)
            if not df.empty:
                st.metric("Total Cases Processed", len(df))
                st.line_chart(df["prob"])
            else:
                st.info("No cases logged yet.")
        except:
            st.info("Monitoring unavailable.")

# =========================================================
# USER GUIDE
# =========================================================
with tabs[2]:

    st.markdown("<div class='section'>How to Use</div>", unsafe_allow_html=True)

    st.write("""
    1. Login using your hospital access key.
    2. Select organ (currently Liver active).
    3. Upload a high-quality grayscale ultrasound image.
    4. Adjust operating threshold if required:
        - Lower threshold → higher sensitivity
        - Higher threshold → higher specificity
    5. Review probability and classification band.
    6. Follow recommended clinical action.
    """)

    st.markdown("<div class='section'>Threshold Logic</div>", unsafe_allow_html=True)

    st.write("""
    Probability < 0.1 → Normal  
    0.1 – Threshold → Likely Benign  
    ≥ Threshold → Suspicious Malignant  
    """)

    st.markdown("<div class='section'>Intended Use</div>", unsafe_allow_html=True)

    st.write("""
    This system is a clinical decision-support tool.
    It does not replace physician judgment.
    Final management decisions must integrate imaging morphology,
    laboratory data, and clinical context.
    """)

    st.markdown("<div class='section'>Data Privacy</div>", unsafe_allow_html=True)

    st.write("""
    No patient identifiers are stored.
    Audit logs store only probability and timestamp.
    Suitable for internal hospital deployment.
    """)
