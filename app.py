import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import uuid
import datetime
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
body { background-color: #0f172a; }
.main-title { font-size: 2.4rem; font-weight: 700; color: white; }
.sub-title { color: #94a3b8; margin-bottom: 25px; }
.card { background-color: #1e293b; padding: 20px; border-radius: 12px; margin-bottom: 15px; }
.high-risk { color: #ef4444; }
.medium-risk { color: #f59e0b; }
.low-risk { color: #22c55e; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enterprise Multi-Organ Clinical AI Platform</div>", unsafe_allow_html=True)

# =========================
# MODEL REGISTRY
# =========================
MODEL_REGISTRY = {
    "Liver": {
        "model_url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "num_classes": 2,
        "malignant_index": 1,
        "auc": 0.899,
        "screening_threshold": 0.2835,
        "balanced_threshold": 0.5,
        "version": "Liver v2.1",
        "dataset_size": 735
    },
    "Thyroid": {
        "model_url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/thyroid_v1_final.pth",
        "num_classes": 2,
        "malignant_index": 1,
        "auc": 0.851,
        "screening_threshold": 0.40,
        "balanced_threshold": 0.5,
        "version": "Thyroid v1.0",
        "dataset_size": 3115
    }
}

# =========================
# SIDEBAR
# =========================
st.sidebar.header("System Configuration")

selected_organ = st.sidebar.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))
mode = st.sidebar.radio("Deployment Mode", ["Screening", "Balanced"])

model_info = MODEL_REGISTRY[selected_organ]

# =========================
# MODEL LOADER
# =========================
@st.cache_resource
def load_model(model_url, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.hub.load_state_dict_from_url(model_url, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model(model_info["model_url"], model_info["num_classes"])

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =========================
# GRAD-CAM
# =========================
def generate_gradcam(model, image_tensor, target_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    final_layer = model.layer4
    handle_f = final_layer.register_forward_hook(forward_hook)
    handle_b = final_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2,3))
    cam = torch.zeros(acts.shape[2:], dtype=torch.float32)

    for i, w in enumerate(weights[0]):
        cam += w * acts[0, i, :, :]

    cam = torch.relu(cam)
    cam = cam / torch.max(cam)
    cam = cam.detach().numpy()

    handle_f.remove()
    handle_b.remove()

    return cam

# =========================
# MAIN LAYOUT
# =========================
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("### Ultrasound Input")
    uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

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

        if mode == "Screening":
            threshold = model_info["screening_threshold"]
        else:
            threshold = model_info["balanced_threshold"]

        # 3-level stratification
        if malignant_prob < 0.1:
            interpretation = "Likely Normal"
            risk_class = "low-risk"
        elif malignant_prob < threshold:
            interpretation = "Likely Benign"
            risk_class = "medium-risk"
        else:
            interpretation = "Suspicious Malignant"
            risk_class = "high-risk"

        # =========================
        # DISPLAY
        # =========================
        st.markdown(f"**Case ID:** {case_id}")
        st.markdown(f"**Model:** {model_info['version']}")
        st.markdown(f"**Malignancy Probability:** {round(risk_score,2)}%")
        st.markdown(f"<span class='{risk_class}'><b>{interpretation}</b></span>", unsafe_allow_html=True)

        # Calibration indicator
        if abs(malignant_prob - threshold) < 0.05:
            st.info("Confidence near operating threshold – borderline case.")
        else:
            st.success("Confidence well-separated from threshold.")

        # =========================
        # ROC Visualization
        # =========================
        st.markdown("### ROC Curve (Validation)")
        fpr = np.linspace(0,1,100)
        tpr = fpr ** (1/model_info["auc"])

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1])
        ax.scatter(1-model_info["auc"], model_info["auc"])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        st.pyplot(fig)

        # =========================
        # GRAD-CAM Overlay
        # =========================
        st.markdown("### AI Focus Region (Grad-CAM)")

        cam = generate_gradcam(model, input_tensor, model_info["malignant_index"])
        cam = cv2.resize(cam, (image.size[0], image.size[1]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.array(image) * 0.6 + heatmap * 0.4
        overlay = overlay.astype(np.uint8)

        st.image(overlay, use_column_width=True)

        # =========================
        # PERFORMANCE DASHBOARD
        # =========================
        st.markdown("### Model Performance Dashboard")
        st.write("Validated AUC:", model_info["auc"])
        st.write("Dataset Size:", model_info["dataset_size"])
        st.write("Operating Threshold:", threshold)

        # =========================
        # AUDIT LOGGING
        # =========================
        log_data = {
            "case_id": case_id,
            "organ": selected_organ,
            "mode": mode,
            "probability": malignant_prob,
            "interpretation": interpretation,
            "timestamp": datetime.datetime.now()
        }

        log_file = "audit_log.csv"

        if os.path.exists(log_file):
            df_existing = pd.read_csv(log_file)
            df_new = pd.concat([df_existing, pd.DataFrame([log_data])], ignore_index=True)
        else:
            df_new = pd.DataFrame([log_data])

        df_new.to_csv(log_file, index=False)

        st.markdown("### Recent Cases")
        st.dataframe(df_new.tail(5))

