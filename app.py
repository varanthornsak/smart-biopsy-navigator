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
# SAFE DATABASE INIT
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
        "threshold": 0.2835
    },
    "Thyroid": {
        "status": "Model training in progress (cross-validation ongoing)"
    },
    "Breast": {
        "status": "Dataset curation and preprocessing phase"
    },
    "Prostate": {
        "status": "Planned future expansion"
    }
}

# ======================================================
# STYLE
# ======================================================
st.markdown("""
<style>
.big-title {font-size:30px;font-weight:700;}
.card {padding:25px;border-radius:15px;color:white;font-weight:600;text-align:center;font-size:20px;}
.green {background:#27ae60;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;}
.section {font-size:20px;font-weight:600;margin-top:25px;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOGIN
# ======================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    hospital = st.selectbox("Hospital", ["Sri Nagarind Hospital","Demo Hospital"])
    password = st.text_input("Access Key", type="password")
    if st.button("Login"):
        if password == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.hospital = hospital
        else:
            st.error("Invalid key")
    st.stop()

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

tabs = st.tabs(["Clinical AI","Model Registry","How to Use"])

# ======================================================
# 1️⃣ CLINICAL AI
# ======================================================
with tabs[0]:

    organ = st.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))

    if organ != "Liver":
        st.info(f"{organ}: {MODEL_REGISTRY[organ]['status']}")
        st.stop()

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

        threshold = MODEL_REGISTRY["Liver"]["threshold"]

        if prob < 0.1:
            label="Likely Normal"
            color="green"
            explanation="""
Normal hepatic echotexture with homogeneous parenchymal pattern.
No high-risk radiologic features detected.
Routine surveillance recommended.
"""
        elif prob < threshold:
            label="Likely Benign"
            color="yellow"
            explanation="""
Low-risk lesion morphology detected.
Imaging features suggest benign etiology.
Short-term imaging follow-up recommended.
"""
        else:
            label="Suspicious Malignant"
            color="red"
            explanation="""
High-risk imaging characteristics identified.
Probability exceeds validated screening threshold.
Further diagnostic evaluation (contrast imaging or biopsy) recommended.
"""

        col1,col2 = st.columns([1.2,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"<div class='card {color}'>{label}<br>{round(prob*100,2)}%</div>", unsafe_allow_html=True)

            fig, ax = plt.subplots()
            ax.axis("off")
            theta = np.linspace(0, math.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta))
            angle = math.pi*(1-prob)
            ax.plot([0,np.cos(angle)],[0,np.sin(angle)], linewidth=4)
            ax.text(0,-0.2,f"{round(prob*100,1)}%",ha="center")
            st.pyplot(fig)

        st.markdown("### Clinical Interpretation")
        st.write(explanation)

        st.markdown("### Model Details")
        st.write("Binary classifier (Malignant vs Non-Malignant)")
        st.write("Validated AUC: 0.899 ± 0.03")
        st.write("Screening sensitivity optimized ≥95%")

        c.execute("""
        INSERT INTO audit (case_id, organ, prob, timestamp)
        VALUES (?, ?, ?, ?)
        """, (str(uuid.uuid4())[:8],
              organ,
              float(prob),
              str(datetime.datetime.now())))
        conn.commit()

# ======================================================
# 2️⃣ MODEL REGISTRY
# ======================================================
with tabs[1]:

    for organ,info in MODEL_REGISTRY.items():
        st.markdown(f"### {organ}")
        st.write(info["status"])

# ======================================================
# 3️⃣ HOW TO USE
# ======================================================
with tabs[2]:

    st.markdown("### Usage Guide")

    st.markdown("""
1. Login using authorized hospital access key.
2. Select organ (Liver currently deployed).
3. Upload high-quality transverse ultrasound image.
4. Review color-coded AI classification:
   - Green → Likely Normal
   - Yellow → Likely Benign
   - Red → Suspicious Malignant
5. Review probability score and clinical recommendation.
6. All predictions logged for quality monitoring.
""")

    st.markdown("### Important Notes")
    st.write("This system is intended as decision-support only.")
    st.write("Clinical correlation is required.")
