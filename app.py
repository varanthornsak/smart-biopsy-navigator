import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import uuid
import datetime
import time
import matplotlib.pyplot as plt

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="Smart Biopsy Navigator™", layout="wide")

# =====================================
# ENTERPRISE STYLE
# =====================================
st.markdown("""
<style>
body { background-color: #f4f6fa; }

.hero {
    background: linear-gradient(90deg,#0f172a,#1e3a8a);
    padding: 24px;
    border-radius: 14px;
    color: white;
    margin-bottom: 18px;
}

.badge {
    background: rgba(255,255,255,0.15);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    margin-right: 8px;
}

.kpi {
    background: white;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

.card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

.viewer {
    background: #0f172a;
    padding: 12px;
    border-radius: 10px;
}

.section-title {
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 10px;
}

.footer {
    font-size: 12px;
    color: #6b7280;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HERO BAR
# =====================================
st.markdown("""
<div class="hero">
<h2>Smart Biopsy Navigator™</h2>
<div>
<span class="badge">Enterprise Deployment</span>
<span class="badge">Model v1.0.0</span>
<span class="badge">🟢 System Operational</span>
<span class="badge">Bangkok Advanced Medical Center (Mock)</span>
</div>
</div>
""", unsafe_allow_html=True)

# =====================================
# SESSION STATE
# =====================================
if "registry" not in st.session_state:
    st.session_state.registry = []

# =====================================
# KPI STRIP
# =====================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("AI-Assisted Cases", len(st.session_state.registry))
col2.metric("High-Risk Flags",
            sum(1 for r in st.session_state.registry if r["risk"]=="High"))
col3.metric("Avg Inference Time", "0.24 sec")
col4.metric("Deployment Mode", "On-Prem CPU")

st.markdown("---")

# =====================================
# SIDEBAR
# =====================================
page = st.sidebar.radio(
    "Platform",
    ["Clinical Console", "Enterprise Impact", "Governance"]
)

# =====================================
# MODEL LOAD
# =====================================
MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/best_liver_model.pth"

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.hub.load_state_dict_from_url(
        MODEL_URL,
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()
classes = ['benign','malignant','normal']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =====================================
# RISK HALF-GAUGE
# =====================================
def half_gauge(score):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.axis('off')

    theta = np.linspace(np.pi, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)

    ax.plot(x, y)

    pointer = np.pi + (score/100)*np.pi
    ax.plot([0, np.cos(pointer)], [0, np.sin(pointer)])

    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-0.2,1.2)

    ax.text(0, -0.1, f"{int(score)}%", ha='center', fontsize=14)
    st.pyplot(fig)

# =====================================
# MINI TREND
# =====================================
def mini_trend():
    if len(st.session_state.registry) > 0:
        values = [90 if r["risk"]=="High" else
                  50 if r["risk"]=="Moderate" else 10
                  for r in st.session_state.registry]

        fig, ax = plt.subplots()
        ax.plot(values)
        ax.set_ylim(0,100)
        ax.set_ylabel("Risk Index")
        st.pyplot(fig)

# =====================================
# CLINICAL CONSOLE
# =====================================
if page == "Clinical Console":

    colA, colB, colC = st.columns([1,1.3,1])

    # Registry
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient Registry</div>', unsafe_allow_html=True)

        if len(st.session_state.registry)==0:
            st.write("No cases logged.")
        else:
            st.dataframe(pd.DataFrame(st.session_state.registry), use_container_width=True)
            st.markdown("### Risk Trend")
            mini_trend()

        st.markdown('</div>', unsafe_allow_html=True)

    # Viewer
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Imaging Viewer</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Liver Ultrasound", type=["jpg","png","jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown('<div class="viewer">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # AI Panel
    with colC:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AI Risk Intelligence</div>', unsafe_allow_html=True)

        if uploaded_file:
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            malignant_prob = probs[classes.index("malignant")]
            risk_score = malignant_prob*100

            half_gauge(risk_score)

            if risk_score < 20:
                risk_label="Low"
            elif risk_score < 60:
                risk_label="Moderate"
            else:
                risk_label="High"

            st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")

            if st.button("Log Case"):
                st.session_state.registry.append({
                    "Case ID": str(uuid.uuid4())[:8],
                    "Risk": risk_label,
                    "Time": datetime.datetime.now().strftime("%H:%M:%S")
                })
                st.success("Case Logged")

        st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# ENTERPRISE IMPACT
# =====================================
elif page == "Enterprise Impact":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    scans = st.slider("Monthly Scan Volume", 500, 5000, 2000)
    fee = st.slider("Per-Scan Fee ($)", 5, 20, 10)

    st.metric("Projected Annual Revenue", f"${scans*fee*12:,}")
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# GOVERNANCE
# =====================================
elif page == "Governance":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
    • Architecture: ResNet18  
    • Risk Computation: Malignant posterior probability  
    • Deployment: On-Prem Hospital Infrastructure  
    • Intended Use: Clinical Decision Support  
    • Non-autonomous system  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Smart Biopsy Navigator™ — MedTech Enterprise Platform</div>', unsafe_allow_html=True)
