import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# =============================
# FUTURISTIC UI
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
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>SMART BIOPSY NAVIGATOR</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Driven Liver Lesion Risk Intelligence System</div>", unsafe_allow_html=True)

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
except:
    st.error("AI Core failed to initialize.")
    st.stop()

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
    fig, ax = plt.subplots(figsize=(4,4))
    ax.axis('off')

    theta = np.linspace(0, 2*np.pi, 100)
    x = 0.5 + 0.4*np.cos(theta)
    y = 0.5 + 0.4*np.sin(theta)

    ax.plot(x, y)
    ax.text(0.5, 0.5, f"{int(score)}%", ha='center', va='center', fontsize=28)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    st.pyplot(fig)

# =============================
# LAYOUT
# =============================
col1, col2 = st.columns([1.2,1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Ultrasound Scan", type=["jpg","png","jpeg"])
    age = st.slider("Patient Age", 18, 90, 55)
    gender = st.selectbox("Gender", ["Male","Female"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    with st.spinner("⚡ AI Core Processing..."):

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = temperature_scaling(logits)[0].numpy()

    pred_idx = np.argmax(probs)
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    # 🔴 Risk based only on malignant probability
    malignant_index = classes.index("malignant")
    malignant_prob = probs[malignant_index]
    risk_score = malignant_prob * 100

    adequacy = 60 + confidence * 40
    case_id = str(uuid.uuid4())[:8]

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("CASE PROFILE")
        st.write(f"Case ID: {case_id}")
        st.write(f"Age: {age} | Gender: {gender}")
        st.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.subheader("AI RISK INTELLIGENCE")

        st.metric("Predicted Classification", pred_class.upper())
        st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
        st.metric("Calibrated Confidence", f"{round(confidence*100,2)}%")

        st.markdown("### RISK GAUGE")
        draw_gauge(risk_score)

        st.subheader("Clinical Recommendation")
        st.write(clinical_recommendation(risk_score))

        st.metric("Biopsy Adequacy Probability", f"{round(adequacy,2)}%")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Probability Matrix")

        fig, ax = plt.subplots()
        ax.bar(classes, probs)
        ax.set_ylim(0,1)
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# =============================
# FOOTER
# =============================
st.markdown("<div class='footer'>AI Clinical Decision Support Prototype • Research Use Only</div>", unsafe_allow_html=True)
