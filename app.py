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
# Page Config
# -------------------------
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

st.title("Smart Biopsy Navigator")
st.caption("AI-Powered Liver Ultrasound Clinical Decision Support")

# -------------------------
# Model URL (HuggingFace)
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

# -------------------------
# Load model safely
# -------------------------
try:
    model = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

classes = ['benign', 'malignant', 'normal']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------
# Upload Section
# -------------------------
uploaded_file = st.file_uploader("Upload Liver Ultrasound Image", type=["jpg","png","jpeg"])

age = st.slider("Patient Age", 18, 90, 55)
gender = st.selectbox("Gender", ["Male", "Female"])

# -------------------------
# Prediction Section
# -------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    with st.spinner("AI is analyzing..."):
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

    pred_idx = np.argmax(probs)
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    risk_score = confidence * 100

    case_id = str(uuid.uuid4())[:8]

    st.subheader("Case Information")
    st.write(f"Case ID: {case_id}")
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")
    st.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.subheader("AI Prediction")
    st.metric("Prediction", pred_class.upper())
    st.metric("Confidence", f"{round(confidence*100,2)}%")
    st.metric("Risk Score", f"{round(risk_score,2)}/100")

    # Probability Chart
    st.subheader("Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(classes, probs)
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
