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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Navigator",
    layout="wide",
)

# =====================================================
# APP STYLE (Apple Minimal)
# =====================================================
st.markdown("""
<style>
html, body { background-color: #f8f9fb; }
.big-title { font-size: 32px; font-weight: 700; }
.subtitle { color: #6e6e73; margin-bottom: 20px; }
.card { padding: 20px; border-radius: 14px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.green { background: #e8f7ef; border-left: 6px solid #2ecc71; }
.yellow { background: #fff8e5; border-left: 6px solid #f1c40f; }
.red { background: #fdecea; border-left: 6px solid #e74c3c; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL CONFIG
# =====================================================
MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth"
DEFAULT_THRESHOLD = 0.2835

# =====================================================
# DATABASE (Multi-tenant SaaS Simulation)
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
# LOGIN (Multi-tenant)
# =====================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Premium Enterprise SaaS</div>", unsafe_allow_html=True)

    hospital = st.selectbox("Hospital Tenant",
                            ["Sri Nagarind Hospital", "Bangkok Hospital", "Demo Hospital"])

    role = st.selectbox("User Role",
                        ["Radiologist", "Surgeon", "Oncologist", "Admin"])

    key = st.text_input("Access Key", type="password")

    if st.button("Login"):
        if key == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.hospital = hospital
            st.session_state.role = role
        else:
            st.error("Invalid Key")

    st.stop()

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# =====================================================
# HEADER
# =====================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>{st.session_state.hospital} | {st.session_state.role}</div>", unsafe_allow_html=True)

tabs = st.tabs(["Worklist", "Case Viewer", "Analytics", "Billing", "User Guide"])

# =====================================================
# WORKLIST (PACS STYLE)
# =====================================================
with tabs[0]:
    st.subheader("Case Worklist")

    df = pd.read_sql_query("SELECT * FROM cases WHERE hospital=?",
                           conn,
                           params=(st.session_state.hospital,))

    if not df.empty:
        st.dataframe(df.sort_values("timestamp", ascending=False))
    else:
        st.info("No cases yet.")

# =====================================================
# CASE VIEWER
# =====================================================
with tabs[1]:

    st.subheader("New Case")

    organ = "Liver"
    threshold = st.slider("Operating Threshold", 0.1, 0.9, DEFAULT_THRESHOLD)

    uploaded = st.file_uploader("Upload Ultrasound Image",
                                type=["jpg", "png", "jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        # API Simulation (internal inference endpoint)
        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()

        # Classification
        if prob < 0.1:
            label = "Normal"
            style = "green"
        elif prob < threshold:
            label = "Likely Benign"
            style = "yellow"
        else:
            label = "Suspicious Malignant"
            style = "red"

        case_id = str(uuid.uuid4())[:8]

        col1, col2 = st.columns([1.3,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"<div class='card {style}'><b>{label}</b><br>{round(prob*100,2)}%</div>", unsafe_allow_html=True)

            # Risk Gauge
            fig, ax = plt.subplots()
            ax.axis("off")
            theta = np.linspace(0, math.pi, 100)
            ax.plot(np.cos(theta), np.sin(theta))
            angle = math.pi*(1-prob)
            ax.plot([0,np.cos(angle)],[0,np.sin(angle)], linewidth=4)
            ax.text(0,-0.2,f"{round(prob*100,1)}%",ha="center")
            st.pyplot(fig)

        st.markdown("### Structured Clinical Report")

        report = f"""
Case ID: {case_id}
Hospital: {st.session_state.hospital}
Role: {st.session_state.role}
Organ: {organ}

Probability: {round(prob,4)}
Threshold: {threshold}
Classification: {label}

Recommended Action:
"""

        if label == "Normal":
            report += "Routine surveillance."
        elif label == "Likely Benign":
            report += "Clinical correlation and short-interval follow-up."
        else:
            report += "Recommend biopsy and oncologic referral."

        st.text_area("Report Preview", report, height=220)

        st.download_button("Export Report", data=report,
                           file_name=f"{case_id}_report.txt")

        log_case(case_id,
                 st.session_state.hospital,
                 st.session_state.role,
                 organ,
                 prob,
                 label)

# =====================================================
# ANALYTICS DASHBOARD
# =====================================================
with tabs[2]:

    st.subheader("Deployment Analytics")

    df = pd.read_sql_query("SELECT * FROM cases WHERE hospital=?",
                           conn,
                           params=(st.session_state.hospital,))

    if not df.empty:
        st.metric("Total Cases", len(df))
        st.metric("Average Risk", round(df["prob"].mean(),3))
        st.line_chart(df["prob"])

        fig, ax = plt.subplots()
        ax.hist(df["prob"], bins=20)
        ax.set_title("Risk Distribution")
        st.pyplot(fig)
    else:
        st.info("No data available.")

# =====================================================
# BILLING SIMULATION
# =====================================================
with tabs[3]:

    st.subheader("SaaS Billing Simulation")

    df = pd.read_sql_query("SELECT * FROM cases WHERE hospital=?",
                           conn,
                           params=(st.session_state.hospital,))

    price_per_case = 25  # USD simulation

    total_cases = len(df)
    revenue = total_cases * price_per_case

    st.metric("Cases Processed", total_cases)
    st.metric("Revenue (Simulated USD)", revenue)

# =====================================================
# USER GUIDE
# =====================================================
with tabs[4]:

    st.subheader("How to Use the System")

    st.write("""
1. Login with hospital access key.
2. Navigate to Case Viewer.
3. Upload ultrasound image.
4. Review classification:
   - Green → Normal
   - Yellow → Likely Benign
   - Red → Suspicious Malignant
5. Adjust threshold if needed.
6. Export structured report for EMR upload.
7. View analytics and billing in respective tabs.
""")

    st.write("""
Intended Use:
Clinical decision support tool only.
Not a standalone diagnostic system.
Final decisions must be made by licensed physicians.
""")
