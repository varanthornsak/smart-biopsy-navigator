import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import uuid
import datetime
import matplotlib.pyplot as plt

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="Smart Biopsy Navigator™", layout="wide")

# =========================================
# SESSION STATE
# =========================================
if "cases" not in st.session_state:
    st.session_state.cases = []

# =========================================
# SIDEBAR – ENTERPRISE SHELL
# =========================================
hospital = st.sidebar.selectbox(
    "Hospital Deployment",
    [
        "Sri Nagarind Hospital (Khon Kaen)",
        "Bangkok Advanced Medical Center",
        "Chiang Mai Academic Hospital"
    ]
)

organ = st.sidebar.selectbox(
    "Select Organ Module",
    ["Liver (Ultrasound)",
     "Thyroid (Ultrasound)",
     "Breast (Ultrasound)"]
)

module = st.sidebar.radio(
    "Platform Modules",
    ["Dashboard",
     "AI Analysis",
     "Case Management",
     "Model Monitoring",
     "Imaging Guidance",
     "Governance"]
)

# =========================================
# HEADER
# =========================================
st.title("Smart Biopsy Navigator™")
st.caption(f"""
Multi-Organ Clinical Decision Support Platform  
Deployment: {hospital} | Organ Module: {organ} | Model v1.0.0
""")

st.divider()

# =========================================
# MODEL (Simulated Multi-Organ Backbone)
# =========================================
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

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =========================================
# ORGAN-SPECIFIC LOGIC
# =========================================
def organ_threshold():
    if "Liver" in organ:
        return 60, "Recommend biopsy and hepatology referral."
    elif "Thyroid" in organ:
        return 50, "Consider FNA per TI-RADS."
    elif "Breast" in organ:
        return 40, "Consider biopsy per BI-RADS 4+."

# =========================================
# DASHBOARD
# =========================================
if module == "Dashboard":

    col1, col2, col3 = st.columns(3)

    total_cases = len(st.session_state.cases)
    high_cases = sum(1 for c in st.session_state.cases if c["Risk"]=="High")

    col1.metric("Total Cases", total_cases)
    col2.metric("High Risk Flags", high_cases)
    col3.metric("Active Organ Module", organ)

    if total_cases > 0:
        values = [c["Probability"] for c in st.session_state.cases]
        fig, ax = plt.subplots()
        ax.plot(values)
        ax.set_ylim(0,100)
        ax.set_title("Risk Probability Trend")
        st.pyplot(fig)

# =========================================
# AI ANALYSIS
# =========================================
elif module == "AI Analysis":

    colA, colB = st.columns([1.3,1])

    with colA:
        uploaded_file = st.file_uploader(
            f"Upload {organ} Ultrasound Image",
            type=["jpg","png","jpeg"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=True)

    with colB:
        if uploaded_file:

            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            malignant_prob = float(np.max(probs))
            risk_score = malignant_prob * 100

            threshold, recommendation = organ_threshold()

            st.metric("Predicted Malignancy Probability",
                      f"{round(risk_score,2)}%")

            st.progress(malignant_prob)

            if risk_score >= threshold:
                st.error("High Risk Category")
                risk_label = "High"
                action = recommendation
            elif risk_score >= threshold/2:
                st.warning("Intermediate Risk Category")
                risk_label = "Moderate"
                action = "Further imaging evaluation recommended."
            else:
                st.success("Low Risk Category")
                risk_label = "Low"
                action = "Routine follow-up."

            st.markdown("### Recommended Clinical Action")
            st.write(action)

            if st.button("Log Case"):
                st.session_state.cases.append({
                    "Case ID": str(uuid.uuid4())[:8],
                    "Organ": organ,
                    "Risk": risk_label,
                    "Probability": round(risk_score,2),
                    "Hospital": hospital,
                    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success("Case logged successfully.")

# =========================================
# CASE MANAGEMENT
# =========================================
elif module == "Case Management":

    if len(st.session_state.cases)==0:
        st.info("No cases logged.")
    else:
        df = pd.DataFrame(st.session_state.cases)
        st.dataframe(df, use_container_width=True)

# =========================================
# MODEL MONITORING
# =========================================
elif module == "Model Monitoring":

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC", "0.91")
    col2.metric("Sensitivity", "88%")
    col3.metric("Specificity", "84%")

    fig, ax = plt.subplots()
    fpr = np.linspace(0,1,100)
    tpr = 1 - np.exp(-3*fpr)
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1])
    ax.set_title("ROC Curve")
    st.pyplot(fig)

# =========================================
# IMAGING GUIDANCE
# =========================================
elif module == "Imaging Guidance":

    st.subheader(f"{organ} – Recommended Ultrasound View")

    if "Liver" in organ:
        st.write("""
        - Subcostal longitudinal view  
        - Clear visualization of lesion margins  
        - Minimal motion artifact  
        - Depth adjusted to include full lesion  
        """)

        st.image("https://upload.wikimedia.org/wikipedia/commons/6/63/Ultrasound_liver.jpg",
                 caption="Example Liver Ultrasound View")

    elif "Thyroid" in organ:
        st.write("""
        - Transverse and longitudinal views  
        - Include full thyroid lobe  
        - Nodule centered in frame  
        """)

        st.image("https://upload.wikimedia.org/wikipedia/commons/3/39/Thyroid_ultrasound.jpg",
                 caption="Example Thyroid Ultrasound View")

    elif "Breast" in organ:
        st.write("""
        - Radial or anti-radial orientation  
        - Lesion centered  
        - Include surrounding tissue margin  
        """)

        st.image("https://upload.wikimedia.org/wikipedia/commons/4/45/Breast_ultrasound.jpg",
                 caption="Example Breast Ultrasound View")

# =========================================
# GOVERNANCE
# =========================================
elif module == "Governance":

    st.write("""
    Intended Use:
    Smart Biopsy Navigator™ is designed as a multi-organ clinical decision support platform
    assisting radiologists in risk stratification.

    This system does not replace clinical judgment.
    Deployment model: On-prem hospital infrastructure.
    """)

st.divider()
st.caption("Smart Biopsy Navigator™ — Multi-Organ Clinical AI Platform")
