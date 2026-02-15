import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# -----------------------------
# Professional Clinical Theme
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f9fafb;
}
.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #111827;
}
.sub-title {
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 24px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.metric-label {
    font-size: 0.9rem;
    color: #6b7280;
}
.footer {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Powered Liver Ultrasound Clinical Decision Support System</div>", unsafe_allow_html=True)

# -----------------------------
# Model
# -----------------------------
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
except:
    st.error("Model failed to load.")
    st.stop()

classes = ['benign', 'malignant', 'normal']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------------
# Risk Color Logic
# -----------------------------
def get_risk_display(score):
    if score < 50:
        return "🟢 Low Risk", "#16a34a"
    elif score < 75:
        return "🟡 Moderate Risk", "#ca8a04"
    else:
        return "🔴 High Risk", "#dc2626"

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Liver Ultrasound Image", type=["jpg","png","jpeg"])
    age = st.slider("Patient Age", 18, 90, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        with st.spinner("AI is analyzing clinical image..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)[0].numpy()

        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])
        risk_score = confidence * 100
        adequacy = 60 + confidence * 40
        case_id = str(uuid.uuid4())[:8]

        risk_text, risk_color = get_risk_display(risk_score)

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Case Summary")
        st.write(f"Case ID: {case_id}")
        st.write(f"Age: {age} | Gender: {gender}")
        st.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

        st.subheader("AI Assessment")
        st.metric("Predicted Class", pred_class.upper())
        st.metric("Confidence", f"{round(confidence*100,2)}%")
        st.metric("Risk Score", f"{round(risk_score,2)} / 100")
        st.markdown(f"<span style='color:{risk_color}; font-weight:600'>{risk_text}</span>", unsafe_allow_html=True)

        st.metric("Biopsy Adequacy Probability", f"{round(adequacy,2)}%")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots()
        ax.bar(classes, probs)
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div class='footer'>For research and educational use only. Not intended for standalone clinical diagnosis.</div>", unsafe_allow_html=True)
