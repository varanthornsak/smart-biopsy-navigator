import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import uuid
import datetime
import time

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Smart Biopsy Navigator™", layout="wide")

# =====================================
# ENTERPRISE UI STYLE
# =====================================
st.markdown("""
<style>
body {
    background-color: #f3f4f6;
}
.hero {
    padding: 25px;
    background: white;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.hero-title {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
}
.hero-sub {
    font-size: 15px;
    color: #6b7280;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.badge-low {
    background: #d1fae5;
    color: #065f46;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
}
.badge-mid {
    background: #fef3c7;
    color: #92400e;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
}
.badge-high {
    background: #fee2e2;
    color: #991b1b;
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: 600;
}
.footer {
    font-size: 12px;
    color: #9ca3af;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HERO SECTION
# =====================================
st.markdown("""
<div class="hero">
<div class="hero-title">Smart Biopsy Navigator™</div>
<div class="hero-sub">
Enterprise Clinical Risk Intelligence Platform for Liver Ultrasound
</div>
</div>
""", unsafe_allow_html=True)

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
classes = ['benign', 'malignant', 'normal']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================
# INPUT PANEL
# =====================================
left, right = st.columns([1.2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Liver Ultrasound", type=["jpg","png","jpeg"])
    age = st.slider("Patient Age", 18, 90, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# INFERENCE PANEL
# =====================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        with st.spinner("AI inference running..."):
            start = time.time()
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            inference_time = round(time.time() - start, 3)

        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])
        malignant_prob = probs[classes.index("malignant")]
        risk_score = malignant_prob * 100

        case_id = str(uuid.uuid4())[:8]

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Case Summary")
        colA, colB = st.columns(2)
        colA.write(f"Case ID: {case_id}")
        colA.write(f"Age: {age}")
        colB.write(f"Gender: {gender}")
        colB.write(f"Inference Time: {inference_time} sec")

        st.subheader("AI Risk Intelligence")
        st.metric("Predicted Class", pred_class.upper())
        st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")

        if risk_score < 20:
            st.markdown('<span class="badge-low">LOW RISK</span>', unsafe_allow_html=True)
        elif risk_score < 60:
            st.markdown('<span class="badge-mid">MODERATE RISK</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-high">HIGH RISK</span>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Clinical Decision Support System • Not for standalone diagnosis</div>', unsafe_allow_html=True)
