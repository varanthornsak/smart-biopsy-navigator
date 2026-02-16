import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime
import pandas as pd
import platform
import os

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# =============================
# MODEL REGISTRY
# =============================
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

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model(model_info):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, model_info["num_classes"])
    state_dict = torch.hub.load_state_dict_from_url(
        model_info["model_url"], map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

# =============================
# NAVIGATION
# =============================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Clinical AI",
        "Hospital Analytics",
        "Model Registry",
        "Regulatory & Compliance",
        "Deployment Dashboard",
        "Organ Expansion"
    ]
)

# =============================
# 1️⃣ CLINICAL AI
# =============================
if page == "Clinical AI":

    st.title("Clinical AI Module")

    organ = st.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))
    mode = st.radio("Deployment Mode", ["Screening", "Balanced"])

    model_info = MODEL_REGISTRY[organ]
    model = load_model(model_info)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    uploaded_file = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

        malignant_prob = probs[model_info["malignant_index"]].item()
        threshold = model_info["screening_threshold"] if mode=="Screening" else model_info["balanced_threshold"]

        # 3-level stratification
        if malignant_prob < 0.1:
            interpretation = "Likely Normal"
        elif malignant_prob < threshold:
            interpretation = "Likely Benign"
        else:
            interpretation = "Suspicious Malignant"

        st.metric("Malignancy Probability", f"{round(malignant_prob*100,2)}%")
        st.write("Interpretation:", interpretation)
        st.write("Model Version:", model_info["version"])

        # Calibration indicator
        if abs(malignant_prob - threshold) < 0.05:
            st.warning("Borderline case near threshold.")
        else:
            st.success("Confidence separated from decision boundary.")

        # Audit Logging
        case_id = str(uuid.uuid4())[:8]
        log_data = {
            "case_id": case_id,
            "organ": organ,
            "mode": mode,
            "probability": malignant_prob,
            "interpretation": interpretation,
            "timestamp": datetime.datetime.now()
        }

        log_file = "audit_log.csv"

        if os.path.exists(log_file):
            df_old = pd.read_csv(log_file)
            df_new = pd.concat([df_old, pd.DataFrame([log_data])], ignore_index=True)
        else:
            df_new = pd.DataFrame([log_data])

        df_new.to_csv(log_file, index=False)

        st.subheader("Recent Cases")
        st.dataframe(df_new.tail(5))


# =============================
# 2️⃣ HOSPITAL ANALYTICS
# =============================
elif page == "Hospital Analytics":

    st.title("Multi-Hospital Analytics")

    hospitals = ["Sri Nagarind Hospital", "KKU Community Clinic", "Bangkok General"]
    cases = [320, 180, 450]

    fig, ax = plt.subplots()
    ax.bar(hospitals, cases)
    ax.set_ylabel("Total AI Cases")
    st.pyplot(fig)


# =============================
# 3️⃣ MODEL REGISTRY
# =============================
elif page == "Model Registry":

    st.title("Model Registry")

    for organ, info in MODEL_REGISTRY.items():
        st.subheader(organ)
        st.write("Version:", info["version"])
        st.write("AUC:", info["auc"])
        st.write("Dataset Size:", info["dataset_size"])
        st.write("Screening Threshold:", info["screening_threshold"])
        st.write("---")


# =============================
# 4️⃣ REGULATORY
# =============================
elif page == "Regulatory & Compliance":

    st.title("Regulatory & Compliance")

    st.markdown("""
    **Intended Use:**  
    AI-assisted ultrasound malignancy risk stratification.

    **Disclaimer:**  
    Not intended as a standalone diagnostic system.

    **Data Privacy:**  
    No patient-identifiable information is stored.

    **Audit Trace:**  
    All cases are logged for traceability.
    """)


# =============================
# 5️⃣ DEPLOYMENT DASHBOARD
# =============================
elif page == "Deployment Dashboard":

    st.title("Deployment Metadata")

    st.write("Environment:", platform.platform())
    st.write("PyTorch Version:", torch.__version__)
    st.write("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
    st.write("Deployment: Streamlit Cloud")
    st.write("Total Models:", len(MODEL_REGISTRY))


# =============================
# 6️⃣ ORGAN EXPANSION
# =============================
elif page == "Organ Expansion":

    st.title("Organ Expansion Framework")

    roadmap = {
        "Liver": "Production",
        "Thyroid": "Production",
        "Breast": "Validation",
        "Prostate": "Development",
        "Lung": "Planned"
    }

    for organ, status in roadmap.items():
        st.write(f"{organ}: {status}")


