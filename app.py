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

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# =========================================
# STYLE (Apple Minimal)
# =========================================
st.markdown("""
<style>
.big-title {font-size:32px;font-weight:700;}
.subtitle {color:#6b7280;}
.card {
    background:white;
    padding:24px;
    border-radius:20px;
    box-shadow:0 8px 30px rgba(0,0,0,0.05);
}
.green {background:#eafaf1;}
.yellow {background:#fef9e7;}
.red {background:#fdecea;}
.section {font-size:20px;font-weight:600;margin-top:25px;}
</style>
""", unsafe_allow_html=True)

# =========================================
# DATABASE (Audit Log)
# =========================================
conn = sqlite3.connect("audit.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS audit (
case_id TEXT,
hospital TEXT,
prob REAL,
timestamp TEXT
)
""")
conn.commit()

# =========================================
# LOGIN SYSTEM
# =========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Clinical AI Platform</div>", unsafe_allow_html=True)

    hospital = st.selectbox("Hospital", ["Sri Nagarind Hospital", "Demo Hospital"])
    role = st.selectbox("Role", ["Viewer", "Clinician", "Admin"])
    password = st.text_input("Access Key", type="password")

    if st.button("Login"):
        if password == "SNH_SECURE":
            st.session_state.logged_in = True
            st.session_state.hospital = hospital
            st.session_state.role = role
            st.experimental_rerun()
        else:
            st.error("Invalid access key")

    st.stop()

# =========================================
# HEADER
# =========================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>{st.session_state.hospital} | Role: {st.session_state.role}</div>", unsafe_allow_html=True)

# =========================================
# MODEL CONFIG
# =========================================
MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth"
SCREENING_THRESHOLD = 0.2835

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features,2)
    state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# =========================================
# TABS
# =========================================
tab1, tab2, tab3, tab4 = st.tabs(["Clinical AI","Monitoring","Model Registry","Regulatory"])

# =========================================
# 1️⃣ CLINICAL AI
# =========================================
with tab1:

    st.markdown("<div class='section'>Liver Risk Stratification</div>", unsafe_allow_html=True)

    colA, colB = st.columns([1.2,1])

    mode = st.radio("Mode", ["Screening (High Sensitivity)", "Balanced"])

    threshold = SCREENING_THRESHOLD if "Screening" in mode else 0.5
    temperature = st.slider("Calibration Temperature", 0.5, 3.0, 1.0)

    uploaded = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            logits = logits / temperature
            prob = torch.softmax(logits,dim=1)[0][1].item()

        # Interpretation Logic
        if prob < 0.1:
            label = "Likely Normal"
            bg = "green"
        elif prob < threshold:
            label = "Likely Benign"
            bg = "yellow"
        else:
            label = "Suspicious Malignant"
            bg = "red"

        with colA:
            st.image(image, use_column_width=True)

        with colB:
            st.markdown(f"""
            <div class='card {bg}'>
            <h2>{label}</h2>
            <p>Probability: {round(prob*100,2)}%</p>
            <p>Threshold: {round(threshold,3)}</p>
            </div>
            """, unsafe_allow_html=True)

            # Risk Gauge
            fig, ax = plt.subplots()
            ax.axis("off")
            theta = np.linspace(0, math.pi,100)
            ax.plot(np.cos(theta),np.sin(theta))
            angle = math.pi*(1-prob)
            ax.plot([0,np.cos(angle)],[0,np.sin(angle)],linewidth=4)
            ax.text(0,-0.2,f"{round(prob*100,1)}%",ha="center",fontsize=14)
            st.pyplot(fig)

        # Save audit
        c.execute("INSERT INTO audit VALUES (?,?,?,?)",
                  (str(uuid.uuid4())[:8],
                   st.session_state.hospital,
                   prob,
                   str(datetime.datetime.now())))
        conn.commit()

# =========================================
# 2️⃣ MONITORING
# =========================================
with tab2:

    df = pd.read_sql_query("SELECT * FROM audit", conn)

    if df.empty:
        st.info("No cases yet.")
    else:
        st.metric("Total Cases", len(df))
        st.line_chart(df["prob"])

        # Drift Detection
        if len(df) > 30:
            rolling = df["prob"].rolling(30).mean()
            if rolling.iloc[-1] > 0.6:
                st.error("Model Drift Alert – Retraining Recommended")

        # Multi-hospital comparison
        hospital_group = df.groupby("hospital")["prob"].mean()
        st.bar_chart(hospital_group)

# =========================================
# 3️⃣ MODEL REGISTRY
# =========================================
with tab3:

    st.markdown("<div class='section'>Production Model</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <b>Model:</b> Liver v2.1 (Binary)<br>
    <b>Status:</b> Production<br>
    <b>AUC:</b> 0.899 ± 0.03<br>
    <b>Calibration:</b> Temperature Scaling<br>
    <b>Screening Threshold:</b> 0.2835<br>
    <b>Dataset:</b> 735 cases<br>
    <b>Validation:</b> 5-fold CV<br>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# 4️⃣ REGULATORY PANEL
# =========================================
with tab4:

    st.markdown("<div class='section'>Regulatory & Intended Use</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
    <b>Intended Use:</b> Ultrasound-based liver malignancy risk stratification tool.<br><br>
    <b>Clinical Role:</b> Screening support system (not standalone diagnostic device).<br><br>
    <b>Human-in-the-loop:</b> Required.<br><br>
    <b>SaMD Classification:</b> Likely Class II (to be determined by regulator).<br><br>
    <b>Post-Market Monitoring:</b> Drift detection enabled.<br>
    </div>
    """, unsafe_allow_html=True)
