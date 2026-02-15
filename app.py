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
# CONFIG
# ===================================
st.set_page_config(
    page_title="Smart Biopsy Navigator™",
    layout="wide"
)

# ===================================
# SIDEBAR NAVIGATION
# ===================================
st.sidebar.title("Smart Biopsy Navigator™")
page = st.sidebar.radio(
    "Platform Modules",
    [
        "Clinical Dashboard",
        "Clinical Insights",
        "Business Impact",
        "AI Governance",
        "Integration"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Deployment: Enterprise (CPU)")
st.sidebar.markdown("Model Version: v1.0.0")
st.sidebar.markdown("Status: Operational")

# ===================================
# MODEL LOAD
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

model = load_model()
classes = ['benign', 'malignant', 'normal']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===================================
# CLINICAL DASHBOARD
# ===================================
if page == "Clinical Dashboard":

    st.title("Enterprise Clinical Risk Dashboard")
    st.caption("AI-Driven Liver Lesion Risk Stratification")

    left, right = st.columns([1.2, 1])

    with left:
        uploaded_file = st.file_uploader(
            "Upload Liver Ultrasound Image",
            type=["jpg", "jpeg", "png"]
        )
        age = st.slider("Patient Age", 18, 90, 55)
        gender = st.selectbox("Gender", ["Male", "Female"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        with left:
            st.image(image, use_column_width=True)

        with right:
            with st.spinner("AI inference in progress..."):
                start_time = time.time()

                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0].numpy()

                inference_time = round(time.time() - start_time, 3)

            pred_idx = np.argmax(probs)
            pred_class = classes[pred_idx]
            confidence = float(probs[pred_idx])

            malignant_index = classes.index("malignant")
            malignant_prob = probs[malignant_index]
            risk_score = malignant_prob * 100

            case_id = str(uuid.uuid4())[:8]

            st.subheader("Case Summary")
            st.write(f"Case ID: {case_id}")
            st.write(f"Age: {age} | Gender: {gender}")
            st.write(f"Inference Time: {inference_time} sec")

            st.subheader("AI Risk Intelligence")
            st.metric("Predicted Class", pred_class.upper())
            st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
            st.metric("Confidence", f"{round(confidence*100,2)}%")

            if risk_score < 20:
                st.success("LOW RISK")
            elif risk_score < 60:
                st.warning("MODERATE RISK")
            else:
                st.error("HIGH RISK")

# ===================================
# CLINICAL INSIGHTS
# ===================================
elif page == "Clinical Insights":
    st.title("Clinical Impact Analysis")

    st.write("""
    • Standardized malignant risk scoring  
    • Reduced inter-observer variability  
    • Improved triage prioritization  
    • Supports radiology workflow integration  
    """)

# ===================================
# BUSINESS IMPACT
# ===================================
elif page == "Business Impact":
    st.title("Operational & Economic Impact")

    st.write("""
    Estimated Value Propositions:
    - Reduce unnecessary biopsies
    - Shorten diagnostic turnaround time
    - Lower procedural costs
    - Improve resource allocation efficiency
    """)

    st.metric("Estimated Cost Avoidance per Case", "$1,200 (Simulated)")
    st.metric("Estimated Annual Impact (Mid-size Hospital)", "$450,000 (Simulated)")

# ===================================
# AI GOVERNANCE
# ===================================
elif page == "AI Governance":
    st.title("Model Transparency & Governance")

    st.write("""
    Model Architecture: ResNet18  
    Risk Calculation: Malignant posterior probability  
    Deployment Mode: CPU inference  
    Intended Use: Clinical decision support  
    Not for standalone diagnostic use  
    """)

# ===================================
# INTEGRATION
# ===================================
elif page == "Integration":
    st.title("Enterprise Integration")

    st.write("""
    Planned Integration Capabilities:
    - PACS connectivity
    - HL7/FHIR compatibility
    - API-based hospital integration
    - Edge deployment readiness
    """)

st.markdown("---")
st.markdown("Smart Biopsy Navigator™ • Enterprise Clinical AI Platform")
