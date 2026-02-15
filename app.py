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
# CONFIG
# =========================================
st.set_page_config(page_title="Smart Biopsy Navigator™", layout="wide")

# =========================================
# STYLE
# =========================================
st.markdown("""
<style>
body { background-color: #eef2f7; }

.header {
    background: linear-gradient(90deg, #1f3c88, #2563eb);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 15px;
}

.kpi-strip {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

.card {
    background: white;
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

.section-title {
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 10px;
}

.low { color: #047857; font-weight:600; }
.mid { color: #b45309; font-weight:600; }
.high { color: #b91c1c; font-weight:600; }

.footer {
    font-size: 12px;
    color: #6b7280;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown("""
<div class="header">
<h2>Smart Biopsy Navigator™</h2>
Hybrid Enterprise AI Platform — HIS Integrated Clinical Intelligence
</div>
""", unsafe_allow_html=True)

# =========================================
# SESSION STATE REGISTRY
# =========================================
if "registry" not in st.session_state:
    st.session_state.registry = []

# =========================================
# TOP KPI STRIP
# =========================================
st.markdown('<div class="kpi-strip">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("AI-Assisted Cases", len(st.session_state.registry))
col2.metric("High-Risk Cases", 
            sum(1 for r in st.session_state.registry if r["risk"]=="High"))
col3.metric("System Uptime", "99.9%")
col4.metric("Deployment Mode", "Hospital CPU")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================
page = st.sidebar.radio(
    "Platform Modules",
    ["Clinical Console", "Enterprise Impact", "Governance"]
)

st.sidebar.markdown("---")
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
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =========================================
# CLINICAL CONSOLE
# =========================================
if page == "Clinical Console":

    col_left, col_mid, col_right = st.columns([1,1.2,1])

    # Patient Registry
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Patient Registry</div>', unsafe_allow_html=True)

        if len(st.session_state.registry) == 0:
            st.write("No cases yet.")
        else:
            df = pd.DataFrame(st.session_state.registry)
            st.dataframe(df, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Imaging Viewer
    with col_mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Imaging Viewer</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Liver Ultrasound", type=["jpg","png","jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # AI Risk Panel
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AI Risk Panel</div>', unsafe_allow_html=True)

        if uploaded_file:

            with st.spinner("Running AI inference..."):
                start = time.time()
                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0].numpy()

                inference_time = round(time.time() - start, 3)

            malignant_prob = probs[classes.index("malignant")]
            risk_score = malignant_prob * 100

            if risk_score < 20:
                risk_label = "Low"
                st.markdown('<span class="low">LOW RISK</span>', unsafe_allow_html=True)
            elif risk_score < 60:
                risk_label = "Moderate"
                st.markdown('<span class="mid">MODERATE RISK</span>', unsafe_allow_html=True)
            else:
                risk_label = "High"
                st.markdown('<span class="high">HIGH RISK</span>', unsafe_allow_html=True)

            st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
            st.write(f"Inference Time: {inference_time} sec")

            if st.button("Add to Registry"):
                case_id = str(uuid.uuid4())[:8]
                st.session_state.registry.append({
                    "Case ID": case_id,
                    "Risk": risk_label,
                    "Timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                })
                st.success("Case Added")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# ENTERPRISE IMPACT
# =========================================
elif page == "Enterprise Impact":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Revenue Simulation</div>', unsafe_allow_html=True)

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
elif page == "Governance":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Governance</div>', unsafe_allow_html=True)

    st.write("""
    • Architecture: ResNet18  
    • Risk Basis: Malignant posterior probability  
    • Intended Use: Clinical Decision Support  
    • Deployment: Hospital Internal Infrastructure  
    • Non-autonomous system  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Smart Biopsy Navigator™ — Hybrid Hospital + Startup Enterprise AI</div>', unsafe_allow_html=True)

