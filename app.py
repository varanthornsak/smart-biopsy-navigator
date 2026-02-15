import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import uuid
import datetime
import time
import matplotlib.pyplot as plt

# ===================================
# PAGE CONFIG
# ===================================
st.set_page_config(
    page_title="Smart Biopsy Navigator",
    layout="wide"
)

# ===================================
# SIDEBAR
# ===================================
st.sidebar.title("Smart Biopsy Navigator")
page = st.sidebar.radio("Navigation", ["Dashboard", "About"])

st.sidebar.markdown("---")
st.sidebar.markdown("Model Version: v1.0.0")
st.sidebar.markdown("Deployment: CPU")
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
# ABOUT PAGE
# ===================================
if page == "About":
    st.title("About Smart Biopsy Navigator")
    st.write("""
    Smart Biopsy Navigator is an AI-powered clinical decision support system 
    designed to assist in liver ultrasound lesion risk stratification.
    
    • Deep learning architecture: ResNet18  
    • Risk computed from malignant posterior probability  
    • Designed for research and educational purposes  
    • Not intended for standalone clinical diagnosis
    """)
    st.stop()

# ===================================
# DASHBOARD PAGE
# ===================================
st.title("AI Liver Lesion Risk Dashboard")
st.caption("Clinical Decision Support System")

# ===================================
# INPUT SECTION
# ===================================
left, right = st.columns([1.2, 1])

with left:
    uploaded_file = st.file_uploader(
        "Upload Liver Ultrasound Image",
        type=["jpg", "jpeg", "png"]
    )
    age = st.slider("Patient Age", 18, 90, 55)
    gender = st.selectbox("Gender", ["Male", "Female"])

# ===================================
# GAUGE FUNCTION
# ===================================
def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.axis("off")

    circle = plt.Circle((0.5, 0.5), 0.4)
    ax.add_artist(circle)

    ax.text(0.5, 0.5, f"{int(score)}%",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=24)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    st.pyplot(fig)

# ===================================
# INFERENCE
# ===================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with left:
        st.image(image, use_column_width=True)

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

        malignant_index = classes.index("malignant")
        malignant_prob = probs[malignant_index]
        risk_score = malignant_prob * 100

        case_id = str(uuid.uuid4())[:8]

        st.subheader("Case Summary")
        st.write(f"Case ID: {case_id}")
        st.write(f"Age: {age} | Gender: {gender}")
        st.write(f"Inference Time: {inference_time} sec")
        st.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        st.subheader("AI Risk Intelligence")

        st.metric("Predicted Class", pred_class.upper())
        st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")
        st.metric("Confidence", f"{round(confidence*100,2)}%")

        st.subheader("Risk Gauge")
        draw_gauge(risk_score)

        if risk_score < 20:
            st.success("LOW RISK")
        elif risk_score < 60:
            st.warning("MODERATE RISK")
        else:
            st.error("HIGH RISK")

        st.subheader("Probability Distribution")
        fig, ax = plt.subplots()
        ax.bar(classes, probs)
        ax.set_ylim(0,1)
        st.pyplot(fig)

        st.subheader("Clinical Recommendation")

        if risk_score < 20:
            st.write("Routine follow-up suggested.")
        elif risk_score < 60:
            st.write("Recommend further imaging and clinical correlation.")
        else:
            st.write("High suspicion. Recommend biopsy and specialist referral.")

st.markdown("---")
st.markdown("For research and educational purposes only. Not for standalone clinical diagnosis.")
