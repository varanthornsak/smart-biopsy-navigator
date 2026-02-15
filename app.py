import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import uuid
import datetime
import time

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Smart Biopsy Navigator™",
    layout="wide"
)

# =====================================
# SIDEBAR NAVIGATION
# =====================================
st.sidebar.title("Smart Biopsy Navigator™")

page = st.sidebar.radio(
    "Platform Modules",
    [
        "Clinical Dashboard",
        "Enterprise Analytics",
        "Business Model",
        "AI Governance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Deployment: Enterprise CPU")
st.sidebar.markdown("Model Version: v1.0.0")
st.sidebar.markdown("Regulatory Position: Clinical Decision Support")

# =====================================
# MODEL LOAD
# =====================================
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

# =====================================
# CLINICAL DASHBOARD
# =====================================
if page == "Clinical Dashboard":

    st.title("Clinical Risk Stratification Dashboard")
    st.caption("Real-Time AI Decision Support for Liver Ultrasound")

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
                start = time.time()
                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0].numpy()

                inference_time = round(time.time() - start, 3)

            pred_idx = np.argmax(probs)
            pred_class = classes[pred_idx]
            confidence = float(probs[pred_idx])

            malignant_prob = probs[classes.index("malignant")]
            risk_score = malignant_prob * 100

            case_id = str(uuid.uuid4())[:8]

            st.subheader("Case Summary")
            st.write(f"Case ID: {case_id}")
            st.write(f"Age: {age} | Gender: {gender}")
            st.write(f"Inference Time: {inference_time} sec")

            st.subheader("AI Risk Intelligence")
            st.metric("Predicted Class", pred_class.upper())
            st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
            st.metric("Prediction Confidence", f"{round(confidence*100,2)}%")

            if risk_score < 20:
                st.success("LOW RISK")
            elif risk_score < 60:
                st.warning("MODERATE RISK")
            else:
                st.error("HIGH RISK")

            st.subheader("Clinical Recommendation")

            if risk_score < 20:
                st.write("Routine follow-up suggested.")
            elif risk_score < 60:
                st.write("Recommend further imaging and clinical correlation.")
            else:
                st.write("High suspicion. Recommend biopsy and specialist referral.")

# =====================================
# ENTERPRISE ANALYTICS
# =====================================
elif page == "Enterprise Analytics":

    st.title("Hospital-Level Performance Analytics")

    st.metric("Total AI-Assisted Cases (Simulated)", "1,245")
    st.metric("High-Risk Cases Flagged", "186")
    st.metric("Estimated Avoided Biopsies", "92")

    st.write("""
    Platform-Level Value:
    - Standardized malignant risk scoring
    - Reduced unnecessary invasive procedures
    - Improved triage prioritization
    - Operational efficiency gain
    """)

# =====================================
# BUSINESS MODEL
# =====================================
elif page == "Business Model":

    st.title("Enterprise Revenue Model Simulation")

    st.write("Example Revenue Scenarios")

    scans_per_month = st.slider("Estimated Scans per Month (Hospital)", 500, 5000, 1500)
    price_per_scan = st.slider("Per-Scan SaaS Fee ($)", 3, 20, 8)

    monthly_revenue = scans_per_month * price_per_scan
    annual_revenue = monthly_revenue * 12

    st.metric("Projected Monthly Revenue", f"${monthly_revenue:,}")
    st.metric("Projected Annual Revenue", f"${annual_revenue:,}")

    st.write("""
    Target Customers:
    - Tertiary Hospitals
    - Academic Medical Centers
    - Imaging Centers
    """)

# =====================================
# AI GOVERNANCE
# =====================================
elif page == "AI Governance":

    st.title("AI Transparency & Governance")

    st.write("""
    Model Architecture: ResNet18  
    Risk Computation: Malignant posterior probability  
    Deployment Mode: CPU inference  
    Regulatory Strategy: Clinical Decision Support (CDS)  
    Intended Use: Assistive decision support  
    """)

    st.warning("Not intended for standalone diagnostic use.")

st.markdown("---")
st.markdown("Smart Biopsy Navigator™ • Enterprise Clinical Risk Intelligence Platform")
