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

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Smart Biopsy Navigator™", layout="wide")

# =========================================
# ENTERPRISE STYLE
# =========================================
st.markdown("""
<style>
body { background-color: #eef2f7; }
.header {
    background: linear-gradient(90deg, #1f3c88, #2563eb);
    padding: 20px;
    border-radius: 8px;
    color: white;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}
.section-title {
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 10px;
}
.low { color: #047857; font-weight:600; }
.mid { color: #b45309; font-weight:600; }
.high { color: #b91c1c; font-weight:600; }
.footer { font-size: 12px; color: #6b7280; margin-top: 25px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
<h2>Smart Biopsy Navigator™</h2>
AI-Powered Clinical Risk Intelligence Platform — HIS Integrated
</div>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
page = st.sidebar.radio(
    "Platform Modules",
    [
        "Executive Overview",
        "Clinical Workflow",
        "Enterprise Impact",
        "AI Governance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Deployment: Hospital Internal CPU")
st.sidebar.markdown("Model Version: v1.0.0")
st.sidebar.markdown("Regulatory Position: Clinical Decision Support")

# =========================================
# MODEL LOAD
# =========================================
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
classes = ['benign', 'malignant', 'normal']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================================
# EXECUTIVE OVERVIEW
# =========================================
if page == "Executive Overview":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Enterprise AI Performance Snapshot (Simulated)</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AI-Assisted Cases", "2,184")
    col2.metric("High-Risk Detection Rate", "14.2%")
    col3.metric("Avoided Biopsies", "148")
    col4.metric("Estimated Annual Cost Impact", "$520,000")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# CLINICAL WORKFLOW
# =========================================
elif page == "Clinical Workflow":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Active Patient Registry (Simulated)</div>', unsafe_allow_html=True)

    data = pd.DataFrame({
        "Patient ID": ["P-1001", "P-1002", "P-1003"],
        "Age": [58, 64, 45],
        "AI Risk": ["Low", "High", "Moderate"],
        "Last Updated": ["2026-02-15", "2026-02-15", "2026-02-14"]
    })

    st.dataframe(data, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Case-Level AI Risk Assessment</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Liver Ultrasound", type=["jpg","png","jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        with st.spinner("Running AI inference..."):
            start = time.time()
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            inference_time = round(time.time() - start, 3)

        malignant_prob = probs[classes.index("malignant")]
        risk_score = malignant_prob * 100

        st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
        st.write(f"Inference Time: {inference_time} sec")

        if risk_score < 20:
            st.markdown('<span class="low">LOW RISK</span>', unsafe_allow_html=True)
        elif risk_score < 60:
            st.markdown('<span class="mid">MODERATE RISK</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="high">HIGH RISK</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# ENTERPRISE IMPACT
# =========================================
elif page == "Enterprise Impact":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Revenue & Cost Simulation</div>', unsafe_allow_html=True)

    scans = st.slider("Monthly Ultrasound Volume", 500, 5000, 2000)
    price = st.slider("Per-Scan AI Fee ($)", 5, 20, 10)

    monthly = scans * price
    annual = monthly * 12

    st.metric("Projected Monthly Revenue", f"${monthly:,}")
    st.metric("Projected Annual Revenue", f"${annual:,}")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# GOVERNANCE
# =========================================
elif page == "AI Governance":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Transparency & Compliance</div>', unsafe_allow_html=True)

    st.write("""
    • Model Architecture: ResNet18  
    • Risk Basis: Malignant posterior probability  
    • Intended Use: Clinical Decision Support  
    • Deployment: On-Premise Hospital Infrastructure  
    • Non-autonomous system  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Smart Biopsy Navigator™ — Enterprise Clinical AI Infrastructure Platform</div>', unsafe_allow_html=True)
