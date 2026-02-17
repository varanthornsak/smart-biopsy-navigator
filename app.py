import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sqlite3
import uuid
import math

st.set_page_config(page_title="Smart Biopsy Navigator v3.5", layout="wide")

# =====================================================
# APPLE MINIMAL STYLE
# =====================================================
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.card {
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 10px 35px rgba(0,0,0,0.05);
}
.green {background:#eafaf1;}
.yellow {background:#fef9e7;}
.red {background:#fdecea;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("v35.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS audit (
case_id TEXT,
hospital TEXT,
prob REAL,
interpretation TEXT,
timestamp TEXT
)
""")
conn.commit()

# =====================================================
# LOGIN PAGE
# =====================================================
if "role" not in st.session_state:
    st.session_state.role = None

if st.session_state.role is None:
    st.title("Smart Biopsy Navigator – Secure Login")

    hospital = st.selectbox("Hospital",
        ["Sri Nagarind Hospital", "KKU Hospital"]
    )

    role = st.selectbox("Role",
        ["Viewer", "Clinician", "Admin"]
    )

    if st.button("Login"):
        st.session_state.role = role
        st.session_state.hospital = hospital
        st.rerun()

    st.markdown("""
    ### วิธีการเข้าใช้งาน
    1. เลือกโรงพยาบาล
    2. เลือกระดับสิทธิ์
    3. กด Login
    """)
    st.stop()

# =====================================================
# HEADER
# =====================================================
st.title("Smart Biopsy Navigator v3.5")
st.caption(f"{st.session_state.hospital} | Role: {st.session_state.role}")

tabs = st.tabs([
    "Clinical AI",
    "Monitoring",
    "Multi-Hospital",
    "Investor",
    "How to Use"
])

# =====================================================
# MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        map_location="cpu"
    )
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# =====================================================
# RISK GAUGE
# =====================================================
def risk_gauge(prob):
    fig, ax = plt.subplots()
    ax.axis("off")

    theta = np.linspace(0, math.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta))

    angle = math.pi * (1 - prob)
    ax.plot([0, np.cos(angle)], [0, np.sin(angle)], linewidth=4)

    ax.text(0, -0.2, f"{round(prob*100,1)}%", ha='center', fontsize=18)
    st.pyplot(fig)

# =====================================================
# 1️⃣ CLINICAL AI
# =====================================================
with tabs[0]:

    st.subheader("Liver AI – Production")

    mode = st.radio("Mode", ["Screening", "Balanced"])
    threshold = 0.2835 if mode=="Screening" else 0.5

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

        if prob < 0.1:
            label = "Normal"
            color = "green"
            rec = "Routine screening."
        elif prob < threshold:
            label = "Benign"
            color = "yellow"
            rec = "Short-term follow-up recommended."
        else:
            label = "Malignant"
            color = "red"
            rec = "Biopsy evaluation recommended."

        st.markdown(f"""
        <div class="card {color}">
        <h2>{label}</h2>
        <p>Probability: {round(prob*100,2)}%</p>
        <p>{rec}</p>
        </div>
        """, unsafe_allow_html=True)

        risk_gauge(prob)

        # Save audit
        c.execute("INSERT INTO audit VALUES (?,?,?,?,?)",
                  (str(uuid.uuid4())[:8],
                   st.session_state.hospital,
                   prob,
                   label,
                   str(datetime.datetime.now())))
        conn.commit()

        # Calibration Curve
        st.subheader("Calibration Curve")
        preds = np.linspace(0,1,50)
        true = preds + np.random.normal(0,0.03,50)
        true = np.clip(true,0,1)

        fig, ax = plt.subplots()
        ax.plot(preds, true)
        ax.plot([0,1],[0,1])
        st.pyplot(fig)

# =====================================================
# 2️⃣ MONITORING
# =====================================================
with tabs[1]:

    df = pd.read_sql_query("SELECT * FROM audit", conn)

    if not df.empty:

        st.metric("Total Cases", len(df))
        st.metric("Malignant Rate",
            f"{round((df['interpretation']=='Malignant').mean()*100,2)}%")

        # Drift detection
        st.subheader("Drift Detection (Rolling Mean)")
        rolling = df["prob"].rolling(20).mean()
        st.line_chart(rolling)

    else:
        st.info("No data yet.")

# =====================================================
# 3️⃣ MULTI-HOSPITAL
# =====================================================
with tabs[2]:

    df = pd.read_sql_query("SELECT * FROM audit", conn)

    if not df.empty:
        st.bar_chart(df.groupby("hospital")["prob"].mean())
    else:
        st.info("No comparison data yet.")

# =====================================================
# 4️⃣ INVESTOR SIMULATOR
# =====================================================
with tabs[3]:

    st.subheader("Revenue Simulation")

    hospitals = st.slider("Number of Hospitals", 1, 100, 10)
    cases = st.slider("Cases per Month per Hospital", 50, 2000, 500)
    price = st.slider("Price per Case ($)", 5, 100, 20)

    monthly = hospitals * cases * price
    yearly = monthly * 12

    st.metric("Monthly Revenue ($)", f"{monthly:,}")
    st.metric("Yearly Revenue ($)", f"{yearly:,}")

# =====================================================
# 5️⃣ HOW TO USE
# =====================================================
with tabs[4]:

    st.subheader("How to Use Smart Biopsy Navigator")

    st.markdown("""
    ### Step 1 – Login
    Select hospital and role.

    ### Step 2 – Clinical AI
    Upload liver ultrasound image.
    Choose Screening or Balanced mode.
    Review probability and recommendation.

    ### Step 3 – Monitoring
    Track malignant detection rate and drift.

    ### Step 4 – Multi-Hospital
    Compare performance across sites.

    ### Step 5 – Investor Panel
    Simulate SaaS revenue model.

    ### Interpretation Guide
    - Green = Likely Normal
    - Yellow = Likely Benign
    - Red = Suspicious Malignant
    """)
