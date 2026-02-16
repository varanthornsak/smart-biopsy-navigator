import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import uuid
import datetime

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# =====================================
# ENTERPRISE UI STYLE
# =====================================
st.markdown("""
<style>
body { background-color: #0f172a; }
.main-title { font-size: 2.4rem; font-weight: 700; color: white; }
.sub-title { color: #94a3b8; margin-bottom: 25px; }
.card { background-color: #1e293b; padding: 20px; border-radius: 12px; margin-bottom: 15px; }
.metric-label { color: #94a3b8; font-size: 0.9rem; }
.metric-value { font-size: 1.7rem; font-weight: 600; }
.high-risk { color: #ef4444; }
.medium-risk { color: #f59e0b; }
.low-risk { color: #22c55e; }
.sidebar .sidebar-content { background-color: #0f172a; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enterprise Multi-Organ Clinical AI Platform</div>", unsafe_allow_html=True)

# =====================================
# MODEL REGISTRY
# =====================================
MODEL_REGISTRY = {
    "Liver": {
        "model_url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "num_classes": 2,
        "malignant_index": 1,
        "auc": 0.899,
        "screening_threshold": 0.2835,
        "balanced_threshold": 0.5,
        "version": "Liver v2.1"
    },
    "Thyroid": {
        "model_url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/thyroid_v1_final.pth",
        "num_classes": 2,
        "malignant_index": 1,
        "auc": 0.851,
        "screening_threshold": 0.40,
        "balanced_threshold": 0.5,
        "version": "Thyroid v1.0"
    }
}

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("System Configuration")

selected_organ = st.sidebar.selectbox(
    "Select Organ",
    list(MODEL_REGISTRY.keys())
)

mode = st.sidebar.radio(
    "Deployment Mode",
    ["Screening", "Balanced"]
)

model_info = MODEL_REGISTRY[selected_organ]

# =====================================
# MODEL LOADER
# =====================================
@st.cache_resource
def load_model(model_url, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.hub.load_state_dict_from_url(
        model_url,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model(model_info["model_url"], model_info["num_classes"])

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =====================================
# MAIN LAYOUT
# =====================================
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("### Ultrasound Input")
    uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

with right_col:
    st.markdown("### AI Clinical Assessment")

    if uploaded_file:

        case_id = str(uuid.uuid4())[:8]

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

        malignant_prob = probs[model_info["malignant_index"]].item()
        risk_score = malignant_prob * 100

        # Select threshold based on mode
        if mode == "Screening":
            threshold = model_info["screening_threshold"]
        else:
            threshold = model_info["balanced_threshold"]

        risk_flag = 1 if malignant_prob >= threshold else 0

        # Risk color classification
        if risk_score >= 70:
            risk_class = "high-risk"
        elif risk_score >= 40:
            risk_class = "medium-risk"
        else:
            risk_class = "low-risk"

        # Case Info Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Case ID</div><div class='metric-value'>{case_id}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Date</div><div class='metric-value'>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Risk Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Malignancy Probability</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value {risk_class}'>{round(risk_score,2)}%</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Clinical Recommendation
        st.markdown("### Clinical Recommendation")

        if risk_flag == 1:
            if mode == "Screening":
                st.error("High-risk lesion detected. Recommend urgent specialist referral and biopsy evaluation.")
            else:
                st.warning("Elevated malignancy risk. Consider biopsy or further diagnostic imaging.")
        else:
            st.success("Low-risk finding. Recommend routine follow-up according to clinical guideline.")

        # Model Info
        st.markdown("---")
        st.markdown("### Model Information")
        st.write("Organ:", selected_organ)
        st.write("Model Version:", model_info["version"])
        st.write("Validated AUC:", model_info["auc"])
        st.write("Deployment Mode:", mode)
        st.write("Threshold Applied:", threshold)
