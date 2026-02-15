import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# -------------------------
# Apple-style UI
# -------------------------
st.markdown("""
<style>
body {
    background-color: #f5f5f7;
}
.big-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #1d1d1f;
}
.subtitle {
    font-size: 1.1rem;
    color: #6e6e73;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Liver Ultrasound Clinical Decision Support</div>", unsafe_allow_html=True)

# -------------------------
# Model Load
# -------------------------
MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/best_liver_model.pth"

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

classes = ['benign', 'malignant', 'normal']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------
# Risk Label Logic
# -------------------------
def get_risk_label(score):
    if score < 50:
        return "🟢 Low Risk"
    elif score < 75:
        return "🟡 Moderate Risk"
    else:
        return "🔴 High Risk"

# -------------------------
# Layout
# -------------------------
left, right = st.columns([1.2, 1])

with left:
    uploaded_file = st.file_uploader("Upload Liver Ultrasound Image", type=["jpg","png","jpeg"])
    age = st.slider("Patient Age", 18, 90, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        with st.spinner("🩺 AI Scanning..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)[0].numpy()

        pred
