import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# -------------------------
# Page Setup
# -------------------------
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# -------------------------
# Apple-style Minimal CSS
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
# Model
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
# Risk Color Logic
# -------------------------
def get_risk_label(score):
    if score < 50:
        return "🟢 Low Risk"
    elif score < 75:
        return "🟡 Moderate Risk"
    else:
        return "🔴 High Risk"

# -------------------------
# PDF Generator
# -------------------------
def generate_pdf(case_id, pred_class, confidence, risk_score, adequacy):
    file_path = f"{case_id}_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Smart Biopsy Navigator Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Case ID: {case_id}", styles["Normal"]))
    elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph(f"Prediction: {pred_class.upper()}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {round(confidence*100,2)}%", styles["Normal"]))
    elements.append(Paragraph(f"Risk Score: {round(risk_score,2)} / 100", styles["Normal"]))
    elements.append(Paragraph(f"Biopsy Adequacy Probability: {round(adequacy,2)}%", styles["Normal"]))

    doc.build(elements)
    return file_path

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

        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])

        risk_score = confidence * 100
        adequacy = 60 + confidence * 40

        case_id = str(uuid.uuid4())[:8]

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("Case Information")
        st.write(f"Case ID: {case_id}")
        st.write(f"Age: {age}")
        st.write(f"Gender: {gender}")

        st.subheader("AI Prediction")
        st.metric("Prediction", pred_class.upper())
        st.metric("Confidence", f"{round(confidence*100,2)}%")

        st.metric("Risk Score", f"{round(risk_score,2)}/100")
        st.markdown(get_risk_label(risk_score))

        st.metric("Biopsy Adequacy Probability", f"{round(adequacy,2)}%")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots()
        ax.bar(classes, probs)
        ax.set_ylim(0,1)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        # PDF Export
        if st.button("📄 Download Clinical Report"):
            pdf_file = generate_pdf(case_id, pred_class, confidence, risk_score, adequacy)
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_file)
