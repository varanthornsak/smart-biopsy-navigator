import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# =============================
# FUTURISTIC STYLE
# =============================
st.markdown("""
<style>
body { background-color: #0f172a; }
.title { font-size: 2.6rem; font-weight: 700; color: #22d3ee; }
.subtitle { font-size: 1rem; color: #94a3b8; margin-bottom: 25px; }
.card {
    background: #111827;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 0 25px rgba(34,211,238,0.15);
    margin-bottom: 20px;
    border: 1px solid rgba(34,211,238,0.2);
}
.footer { font-size: 0.8rem; color: #475569; margin-top: 40px; }
.pulse {
  width: 12px;
  height: 12px;
  background: #22d3ee;
  border-radius: 50%;
  box-shadow: 0 0 0 rgba(34,211,238, 0.7);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(34,211,238, 0.7); }
  70% { box-shadow: 0 0 0 10px rgba(34,211,238, 0); }
  100% { box-shadow: 0 0 0 0 rgba(34,211,238, 0); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>SMART BIOPSY NAVIGATOR</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Risk Intelligence Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='pulse'></div>", unsafe_allow_html=True)

# =============================
# MODEL LOAD
# =============================
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

# =============================
# CONFIDENCE CALIBRATION
# =============================
def temperature_scaling(logits, temperature=1.5):
    return torch.softmax(logits / temperature, dim=1)

# =============================
# CLINICAL RECOMMENDATION
# =============================
def clinical_recommendation(risk):
    if risk < 20:
        return "Routine follow-up recommended."
    elif risk < 60:
        return "Consider additional imaging and clinical correlation."
    else:
        return "High suspicion. Recommend biopsy and urgent specialist referral."

# =============================
# CIRCULAR GAUGE
# =============================
def draw_gauge(score):
    fig, ax = plt.subplots()
    ax.axis('off')
    circle = plt.Circle((0.5,0.5), 0.4)
    ax.add_artist(circle)
    ax.text(0.5,0.5, f"{int(score)}%", ha='center', va='center', fontsize=24)
    st.pyplot(fig)

# =============================
# LAYOUT
# =============================
col1, col2 = st.columns([1.2,1])

with col1:
    uploaded_file = st.file_uploader("Upload Ultrasound", type=["jpg","png","jpeg"])
    age = st.slider("Patient Age", 18, 90, 55)
    gender = st.selectbox("Gender", ["Male","Female"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)/255.0

    st.image(image, use_column_width=True)

    with st.spinner("⚡ AI Core Processing..."):

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = temperature_scaling(logits)[0].numpy()

    pred_idx = np.argmax(probs)
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    malignant_index = classes.index("malignant")
    malignant_prob = probs[malignant_index]
    risk_score = malignant_prob * 100

    st.subheader("AI Risk Gauge")
    draw_gauge(risk_score)

    st.metric("Prediction", pred_class.upper())
    st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
    st.metric("Calibrated Confidence", f"{round(confidence*100,2)}%")

    st.subheader("Clinical Recommendation")
    st.write(clinical_recommendation(risk_score))

    # =============================
    # GRAD-CAM
    # =============================
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    st.subheader("AI Heatmap (Explainability)")
    st.image(cam_image, use_column_width=True)

# =============================
# FOOTER
# =============================
st.markdown("<div class='footer'>AI Clinical Decision Support • Research Use Only</div>", unsafe_allow_html=True)
