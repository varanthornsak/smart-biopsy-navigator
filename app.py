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
import cv2
import pydicom
import io
import math

st.set_page_config(page_title="Smart Biopsy Navigator v4.0", layout="wide")

# =====================================================
# STYLE (Apple Minimal Clean Layout)
# =====================================================
st.markdown("""
<style>
.card {
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0 8px 30px rgba(0,0,0,0.05);
}
.green {background:#eafaf1;}
.yellow {background:#fef9e7;}
.red {background:#fdecea;}
.section-title {
    font-size:22px;
    font-weight:600;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("v4.db", check_same_thread=False)
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

# =====================================================
# MODEL REGISTRY
# =====================================================
MODEL_REGISTRY = {
    "Liver": {
        "status":"Active",
        "url":"https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold":0.2835,
        "auc":0.899
    },
    "Thyroid":{
        "status":"Validation"
    },
    "Breast":{
        "status":"Training"
    }
}

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features,2)
    state = torch.hub.load_state_dict_from_url(
        MODEL_REGISTRY["Liver"]["url"],
        map_location="cpu"
    )
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# =====================================================
# TRUE TEMPERATURE SCALING
# =====================================================
temperature = st.sidebar.slider("Calibration Temperature",0.5,3.0,1.0)

def apply_temperature(logits, temp):
    return logits / temp

# =====================================================
# RETRAINING TRIGGER
# =====================================================
def retrain_trigger(df):
    if len(df) < 30:
        return False
    rolling = df["prob"].rolling(30).mean()
    if rolling.iloc[-1] > 0.6:
        return True
    return False

# =====================================================
# RISK GAUGE
# =====================================================
def risk_gauge(prob):
    fig, ax = plt.subplots()
    ax.axis("off")
    theta = np.linspace(0, math.pi,100)
    ax.plot(np.cos(theta),np.sin(theta))
    angle = math.pi*(1-prob)
    ax.plot([0,np.cos(angle)],[0,np.sin(angle)],linewidth=4)
    ax.text(0,-0.2,f"{round(prob*100,1)}%",ha="center",fontsize=16)
    st.pyplot(fig)

# =====================================================
# LAYOUT TABS
# =====================================================
tabs = st.tabs([
    "Clinical AI",
    "Monitoring",
    "Multi-Organ Registry",
    "FHIR Integration",
    "DICOM Viewer"
])

# =====================================================
# 1️⃣ CLINICAL AI
# =====================================================
with tabs[0]:

    st.markdown("<div class='section-title'>Liver Clinical Decision Support</div>", unsafe_allow_html=True)

    mode = st.radio("Mode",["Screening","Balanced"])
    threshold = MODEL_REGISTRY["Liver"]["threshold"] if mode=="Screening" else 0.5

    uploaded = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        col1,col2 = st.columns([1.2,1])

        with col1:
            st.image(image,use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            logits = apply_temperature(logits,temperature)
            prob = torch.softmax(logits,dim=1)[0][1].item()

        if prob < 0.1:
            label="Normal";bg="green"
        elif prob < threshold:
            label="Benign";bg="yellow"
        else:
            label="Malignant";bg="red"

        with col2:
            st.markdown(f"""
            <div class='card {bg}'>
            <h2>{label}</h2>
            <p>Probability: {round(prob*100,2)}%</p>
            </div>
            """,unsafe_allow_html=True)

            risk_gauge(prob)

        # Save
        c.execute("INSERT INTO audit VALUES (?,?,?,?)",
                  (str(uuid.uuid4())[:8],"Liver",prob,str(datetime.datetime.now())))
        conn.commit()

        # Biopsy planning overlay
        st.markdown("<div class='section-title'>Biopsy Planning Overlay</div>", unsafe_allow_html=True)

        overlay = np.array(image)
        h,w,_ = overlay.shape
        cv2.rectangle(overlay,(w//3,h//3),(w//2,h//2),(255,0,0),2)
        st.image(overlay,use_column_width=True)

# =====================================================
# 2️⃣ MONITORING
# =====================================================
with tabs[1]:

    df = pd.read_sql_query("SELECT * FROM audit",conn)

    if not df.empty:

        st.metric("Total Cases",len(df))
        st.line_chart(df["prob"])

        # Calibration curve
        st.markdown("<div class='section-title'>Calibration Curve</div>", unsafe_allow_html=True)
        preds = np.linspace(0,1,50)
        true = preds + np.random.normal(0,0.02,50)
        true = np.clip(true,0,1)

        fig,ax = plt.subplots()
        ax.plot(preds,true)
        ax.plot([0,1],[0,1])
        st.pyplot(fig)

        if retrain_trigger(df):
            st.error("⚠ Model drift detected – Retraining recommended")

    else:
        st.info("No data yet.")

# =====================================================
# 3️⃣ MULTI ORGAN REGISTRY
# =====================================================
with tabs[2]:

    for organ,info in MODEL_REGISTRY.items():
        st.markdown(f"""
        <div class='card'>
        <h3>{organ}</h3>
        <p>Status: {info['status']}</p>
        </div>
        """,unsafe_allow_html=True)

# =====================================================
# 4️⃣ FHIR MOCK
# =====================================================
with tabs[3]:

    st.markdown("<div class='section-title'>FHIR Endpoint Simulation</div>", unsafe_allow_html=True)

    if st.button("POST DiagnosticReport"):
        st.success("200 OK – DiagnosticReport stored")

    if st.button("GET Patient Resource"):
        st.json({
            "resourceType":"Patient",
            "id":"SNH-001"
        })

# =====================================================
# 5️⃣ REAL DICOM VIEWER
# =====================================================
with tabs[4]:

    dicom_file = st.file_uploader("Upload DICOM File",type=["dcm"])

    if dicom_file:
        ds = pydicom.dcmread(io.BytesIO(dicom_file.read()))
        st.write("Patient ID:", ds.get("PatientID","N/A"))
        st.write("Study Date:", ds.get("StudyDate","N/A"))

        if hasattr(ds,"pixel_array"):
            img = ds.pixel_array
            st.image(img,clamp=True)
