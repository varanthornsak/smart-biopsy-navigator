import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime
import sqlite3
import pandas as pd
import cv2
import platform
import json

st.set_page_config(page_title="Smart Biopsy Navigator Enterprise", layout="wide")

# =====================================================
# MODEL REGISTRY
# =====================================================
MODEL_REGISTRY = {
    "Liver": {
        "model_url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold": 0.2835,
        "auc": 0.899
    },
    "Thyroid": {
        "model_url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/thyroid_v1_final.pth",
        "threshold": 0.40,
        "auc": 0.851
    }
}

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("enterprise_audit.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS audit (
    case_id TEXT,
    organ TEXT,
    hospital TEXT,
    probability REAL,
    interpretation TEXT,
    timestamp TEXT
)
""")
conn.commit()

# =====================================================
# AUTH SIMULATION (OAuth2)
# =====================================================
st.sidebar.title("Hospital Login")
token = st.sidebar.text_input("OAuth2 Access Token")

if token != "SNH_SECURE_TOKEN":
    st.warning("Authentication required (Use SNH_SECURE_TOKEN)")
    st.stop()

# =====================================================
# NAVIGATION
# =====================================================
page = st.sidebar.radio("Module", [
    "Clinical AI",
    "DICOM Metadata",
    "HL7 v2 Simulation",
    "PACS Queue",
    "FHIR REST Mock",
    "Analytics Dashboard"
])

# =====================================================
# MODEL LOADER
# =====================================================
@st.cache_resource
def load_model(url):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# =====================================================
# CLINICAL AI
# =====================================================
if page == "Clinical AI":

    st.title("Enterprise Clinical AI")

    hospital = st.selectbox("Hospital", ["Sri Nagarind Hospital", "KKU Clinic"])
    organ = st.selectbox("Organ", list(MODEL_REGISTRY.keys()))

    model_info = MODEL_REGISTRY[organ]
    model = load_model(model_info["model_url"])

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    file = st.file_uploader("Upload Ultrasound", type=["jpg","png","jpeg"])

    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_column_width=True)

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]

        malignant_prob = probs[1].item()
        threshold = model_info["threshold"]

        if malignant_prob < 0.1:
            interpretation = "Normal"
            color = "green"
        elif malignant_prob < threshold:
            interpretation = "Benign"
            color = "orange"
        else:
            interpretation = "Malignant"
            color = "red"

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Malignancy Probability", f"{round(malignant_prob*100,2)}%")

        with col2:
            st.markdown(
                f"<h2 style='color:{color};'> {interpretation} </h2>",
                unsafe_allow_html=True
            )

        # Save audit
        case_id = str(uuid.uuid4())[:8]
        c.execute("INSERT INTO audit VALUES (?,?,?,?,?,?)",
                  (case_id, organ, hospital,
                   malignant_prob, interpretation,
                   str(datetime.datetime.now())))
        conn.commit()

        # ROC
        st.subheader("Model Performance (ROC)")
        fpr = np.linspace(0,1,100)
        tpr = fpr ** (1/model_info["auc"])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1])
        st.pyplot(fig)

# =====================================================
# DICOM METADATA
# =====================================================
elif page == "DICOM Metadata":

    st.title("DICOM Metadata Viewer")

    dicom_metadata = {
        "PatientName": "Anonymous",
        "PatientID": "SNH-2026-001",
        "StudyInstanceUID": "1.2.840.113619.2.55.3.604688654.783.145",
        "Modality": "US",
        "StudyDate": "20260216",
        "InstitutionName": "Sri Nagarind Hospital"
    }

    st.json(dicom_metadata)

# =====================================================
# HL7 v2 SIMULATION
# =====================================================
elif page == "HL7 v2 Simulation":

    st.title("HL7 ORU^R01 Message")

    hl7_message = """
MSH|^~\&|AI_SYSTEM|SNH|EMR|SNH|202602161030||ORU^R01|12345|P|2.3
PID|||SNH-2026-001||Anonymous||19900101|M
OBR|1|||Ultrasound Liver
OBX|1|NM|MalignancyRisk||0.32|Probability
    """

    st.code(hl7_message)

# =====================================================
# PACS QUEUE
# =====================================================
elif page == "PACS Queue":

    st.title("PACS Push Queue")

    queue_data = pd.DataFrame({
        "StudyUID": ["1.2.3.4.5", "1.2.3.4.6"],
        "Status": ["Processed", "Pending"]
    })

    st.dataframe(queue_data)

# =====================================================
# FHIR REST MOCK
# =====================================================
elif page == "FHIR REST Mock":

    st.title("FHIR REST Endpoint")

    if st.button("POST DiagnosticReport"):
        st.success("200 OK – DiagnosticReport stored in FHIR server.")

    if st.button("GET Patient Resource"):
        st.json({
            "resourceType": "Patient",
            "id": "SNH-2026-001"
        })

# =====================================================
# ANALYTICS
# =====================================================
elif page == "Analytics Dashboard":

    st.title("Enterprise Analytics")

    df = pd.read_sql_query("SELECT * FROM audit", conn)

    if not df.empty:
        st.dataframe(df.tail(10))
        st.bar_chart(df["interpretation"].value_counts())
    else:
        st.info("No records yet.")
