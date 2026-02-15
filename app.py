import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import uuid
import datetime
import time

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Smart Biopsy Navigator™", layout="wide")

# =========================================
# ENTERPRISE STYLE (Hospital Theme)
# =========================================
st.markdown("""
<style>
body {
    background-color: #eef2f7;
}
.header {
    background: #1f3c88;
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 18px;
}
.section-title {
    font-weight: 600;
    font-size: 16px;
    margin-bottom: 10px;
}
.footer {
    font-size: 12px;
    color: #6b7280;
    margin-top: 30px;
}
.low { color: #047857; font-weight:600; }
.mid { color: #b45309; font-weight:600; }
.high { color: #b91c1c; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown("""
<div class="header">
<h2>Smart Biopsy Navigator™</h2>
Clinical Risk Stratification Module — Radiology Enterprise System
</div>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR (Hospital Style Navigation)
# =========================================
st.sidebar.title("Radiology AI Module")

page = st.sidebar.radio(
    "Navigation",
    [
        "Clinical Dashboard",
        "Hospital Analytics",
        "Economic Impact",
        "Compliance & Governance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("System Status: Operational")
st.sidebar.markdown("Deployment Mode: On-Premise / CPU")
st.sidebar.markdown("Model Version: v1.0.0")
st.sidebar.markdown("Regulatory: Clinical Decision Support")

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
# CLINICAL DASHBOARD
# =========================================
if page == "Clinical Dashboard":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Patient Case Input</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        uploaded_file = st.file_uploader("Upload Liver Ultrasound Image", type=["jpg","png","jpeg"])
        age = st.slider("Patient Age", 18, 90, 55)
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        with left:
            st.image(image, use_column_width=True)

        with right:
            with st.spinner("Running AI inference..."):
                start = time.time()
                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0].numpy()

                inference_time = round(time.time() - start, 3)

            pred_idx = np.argmax(probs)
            pred_class = classes[pred_idx]
            malignant_prob = probs[classes.index("malignant")]
            risk_score = malignant_prob * 100

            case_id = str(uuid.uuid4())[:8]

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">AI Risk Assessment</div>', unsafe_allow_html=True)

            st.write(f"Case ID: {case_id}")
            st.write(f"Inference Time: {inference_time} sec")

            st.metric("Predicted Classification", pred_class.upper())
            st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")

            if risk_score < 20:
                st.markdown('<span class="low">LOW RISK</span>', unsafe_allow_html=True)
            elif risk_score < 60:
                st.markdown('<span class="mid">MODERATE RISK</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="high">HIGH RISK</span>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# HOSPITAL ANALYTICS
# =========================================
elif page == "Hospital Analytics":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI-Assisted Case Summary (Simulated)</div>', unsafe_allow_html=True)

    st.metric("Total AI-Assisted Cases", "2,184")
    st.metric("High-Risk Cases Identified", "312")
    st.metric("Flagged for Biopsy Review", "198")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# ECONOMIC IMPACT
# =========================================
elif page == "Economic Impact":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Cost Impact Simulation</div>', unsafe_allow_html=True)

    scans = st.slider("Monthly Ultrasound Volume", 500, 5000, 2000)
    price = st.slider("AI Per-Scan Fee ($)", 5, 20, 10)

    monthly = scans * price
    annual = monthly * 12

    st.metric("Projected Monthly Revenue", f"${monthly:,}")
    st.metric("Projected Annual Revenue", f"${annual:,}")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# COMPLIANCE
# =========================================
elif page == "Compliance & Governance":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Governance Overview</div>', unsafe_allow_html=True)

    st.write("""
    • Architecture: ResNet18  
    • Risk Basis: Malignant posterior probability  
    • Intended Use: Clinical Decision Support  
    • Deployment: Hospital internal infrastructure  
    • Regulatory Pathway: Non-autonomous AI support system  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Smart Biopsy Navigator™ — Enterprise Radiology AI Platform</div>', unsafe_allow_html=True)
