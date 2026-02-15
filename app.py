import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import uuid
import datetime
import time
import matplotlib.pyplot as plt

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="Smart Biopsy Navigator™", layout="wide")

# =====================================
# SESSION STATE
# =====================================
if "registry" not in st.session_state:
    st.session_state.registry = []

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# =====================================
# HOSPITAL TOGGLE
# =====================================
hospital = st.sidebar.selectbox(
    "Hospital Deployment",
    [
        "Bangkok Advanced Medical Center",
        "Chiang Mai Academic Hospital",
        "Singapore Liver Institute"
    ]
)

page = st.sidebar.radio(
    "Platform",
    [
        "Clinical Console",
        "Model Performance",
        "Audit Log",
        "How It Works"
    ]
)

# =====================================
# HERO
# =====================================
st.markdown(f"""
## Smart Biopsy Navigator™  
**Deployment:** {hospital}  
Model v1.0.0 | Clinical Decision Support | 🟢 Operational
""")

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
classes = ['benign','malignant','normal']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# =====================================
# CALIBRATION BAR
# =====================================
def confidence_bar(conf):
    st.markdown("### Model Confidence Calibration")
    st.progress(conf)
    if conf < 0.4:
        st.warning("Low Confidence – Interpret cautiously.")
    elif conf < 0.75:
        st.info("Moderate Confidence.")
    else:
        st.success("High Confidence – Stable prediction.")

# =====================================
# CLINICAL CONSOLE
# =====================================
if page == "Clinical Console":

    col1, col2 = st.columns([1.3,1])

    with col1:
        uploaded_file = st.file_uploader("Upload Liver Ultrasound", type=["jpg","png","jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, use_column_width=True)

    with col2:
        if uploaded_file:
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].numpy()

            pred_class = classes[np.argmax(probs)]
            confidence = float(np.max(probs))
            malignant_prob = probs[classes.index("malignant")]
            risk_score = malignant_prob*100

            st.metric("Predicted Classification", pred_class.upper())
            st.metric("Malignant Probability", f"{round(malignant_prob*100,2)}%")

            confidence_bar(confidence)

            # Detailed Interpretation
            st.markdown("### Clinical Interpretation")
            st.write(f"""
            The AI model estimates a **{round(malignant_prob*100,2)}% probability**
            of malignant liver lesion based on learned imaging features.
            
            Risk classification is derived from malignant posterior probability.
            
            Suggested Action:
            """)
            if risk_score < 20:
                st.success("Low risk – Routine follow-up imaging.")
            elif risk_score < 60:
                st.warning("Moderate risk – Consider further imaging (CT/MRI).")
            else:
                st.error("High risk – Recommend biopsy and specialist referral.")

            # Log case
            if st.button("Log Case to Registry"):
                case_id = str(uuid.uuid4())[:8]
                entry = {
                    "Case ID": case_id,
                    "Hospital": hospital,
                    "Risk": pred_class,
                    "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.registry.append(entry)
                st.session_state.audit_log.append(entry)
                st.success("Case Logged")

# =====================================
# MODEL PERFORMANCE PAGE
# =====================================
elif page == "Model Performance":

    st.markdown("### Validation Performance Summary (Mock Data)")

    colA, colB, colC, colD = st.columns(4)
    colA.metric("AUC", "0.91")
    colB.metric("Sensitivity", "88%")
    colC.metric("Specificity", "84%")
    colD.metric("Validation Cohort", "735 Cases")

    st.write("""
    The model was trained and internally validated on curated liver ultrasound images.
    External validation pending multi-center collaboration.
    """)

# =====================================
# AUDIT LOG
# =====================================
elif page == "Audit Log":

    st.markdown("### Audit Trail")
    if len(st.session_state.audit_log)==0:
        st.write("No cases logged yet.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)

# =====================================
# HOW IT WORKS
# =====================================
elif page == "How It Works":

    st.markdown("### System Overview")
    st.write("""
    1. Upload ultrasound image.
    2. AI computes malignant posterior probability.
    3. Risk classification generated.
    4. Clinician reviews recommendation.
    5. Case optionally logged for audit tracking.
    """)

    st.markdown("### Intended Use")
    st.write("""
    Smart Biopsy Navigator™ is intended as a clinical decision support tool
    to assist radiologists in liver lesion risk stratification.
    It does not replace clinical judgment.
    """)

st.markdown("---")
st.caption("Smart Biopsy Navigator™ — Enterprise MedTech AI Platform")
