import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import uuid
import datetime
import time

# ===================================
# PAGE CONFIG
# ===================================
st.set_page_config(
    page_title="Smart Biopsy Navigator",
    layout="wide"
)

# ===================================
# CLEAN CLINICAL UI
# ===================================
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1f2937;
}
.subtitle {
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 20px;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.badge {
    background: #e5e7eb;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    color: #374151;
}
.footer {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Liver Ultrasound Clinical Decision Support</div>", unsafe_allow_html=True)
st.markdown("<span class='badge'>Model v1.0.0 • CPU Deployment</span>", unsafe_allow_html=True)

# ===================================
# MODEL LOADING
# ===================================
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

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

classes = ['benign', 'malignant', 'normal']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===================================
# CLINICAL LOGIC
# ===================================
def clinical_recommendation(risk):
    if risk < 20:
        return "Routine follow-up suggested."
    elif risk < 60:
        return "Recommend further imaging (CT/MRI) and clinical correlation."
    else:
        return "High suspicion. Recommend biopsy and specialist referral."

# ===================================
# INPUT SECTION
# ===================================
left, right = st.columns([1.2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Liver Ultrasound Image",
        type=["jpg", "jpeg", "png"]
    )
    age = st.slider("Patient Age", 18, 90, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])
    st.markdown("</div>", unsafe_allow_html=True)

# ===================================
# INFERENCE
# ===================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        with st.spinner("AI analyzing image..."):
            start_time = time.time()

            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            inference_time = round(time.time() - start_time, 3)

        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        confidence = float(probs[pred_idx])

        # Risk based ONLY on malignant probability
        malignant_index = classes.index("malignant")
        malignant_prob = probs[malignant_index]
        risk_score = malignant_prob * 100

        case_id = str(uuid.uuid4())[:8]

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # ---------------- Case Summary ----------------
        st.subheader("Case Summary")
        st.write(f"Case ID: {case_id}")
        st.write(f"Age: {age} | Gender: {gender}")
        st.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Inference Time: {inference_time} sec")

        # ---------------- AI Assessment ----------------
        st.subheader("AI Risk Intelligence")
        st.metric("Predicted Class", pred_class.upper())
        st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
        st.metric("Prediction Confidence", f"{round(confidence*100,2)}%")
        st.metric("Computed Risk Score", f"{round(risk_score,2)} / 100")

        # Risk Interpretation
        if risk_score < 20:
            st.success("LOW RISK")
        elif risk_score < 60:
            st.warning("MODERATE RISK")
        else:
            st.error("HIGH RISK")

        # Confidence Interpretation
        if confidence > 0.8:
            st.info("High model confidence.")
        elif confidence > 0.5:
            st.info("Moderate model confidence.")
        else:
            st.warning("Low model confidence. Interpret with caution.")

        # Clinical Recommendation
        st.subheader("Clinical Recommendation")
        st.write(clinical_recommendation(risk_score))

        st.markdown("</div>", unsafe_allow_html=True)

# ===================================
# FOOTER
# ===================================
st.markdown(
    "<div class='footer'>This AI system is intended for decision support only and should not replace clinical judgment.</div>",
    unsafe_allow_html=True
)
