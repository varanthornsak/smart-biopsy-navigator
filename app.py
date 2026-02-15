import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import uuid
import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

st.title("SMART BIOPSY NAVIGATOR")
st.caption("AI-Driven Liver Lesion Risk Intelligence System")

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

try:
    model = load_model()
    st.success("AI Core Initialized")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

classes = ['benign', 'malignant', 'normal']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =============================
# INPUT SECTION
# =============================
uploaded_file = st.file_uploader("Upload Ultrasound Scan", type=["jpg","png","jpeg"])
age = st.slider("Patient Age", 18, 90, 55)
gender = st.selectbox("Gender", ["Male","Female"])

# =============================
# INFERENCE
# =============================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    with st.spinner("AI Core Processing..."):

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()

    pred_idx = np.argmax(probs)
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    malignant_index = classes.index("malignant")
    malignant_prob = probs[malignant_index]
    risk_score = malignant_prob * 100

    case_id = str(uuid.uuid4())[:8]

    st.subheader("Case Information")
    st.write(f"Case ID: {case_id}")
    st.write(f"Age: {age} | Gender: {gender}")
    st.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    st.subheader("AI Results")
    st.metric("Prediction", pred_class.upper())
    st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
    st.metric("Confidence", f"{round(confidence*100,2)}%")
    st.metric("Risk Score", f"{round(risk_score,2)} / 100")

    if risk_score < 20:
        st.success("LOW RISK")
    elif risk_score < 60:
        st.warning("MODERATE RISK")
    else:
        st.error("HIGH RISK")

st.markdown("AI Clinical Decision Support Prototype • Research Use Only")
