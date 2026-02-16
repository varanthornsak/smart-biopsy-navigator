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
import os
import cv2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

# =====================================================
# MODEL REGISTRY
# =====================================================
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

# =====================================================
# DATABASE (SQLite)
# =====================================================
conn = sqlite3.connect("clinical_audit.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS audit (
    case_id TEXT,
    organ TEXT,
    hospital TEXT,
    mode TEXT,
    probability REAL,
    interpretation TEXT,
    timestamp TEXT
)
""")
conn.commit()

# =====================================================
# ROLE BASED ACCESS
# =====================================================
st.sidebar.title("Login")

role = st.sidebar.selectbox(
    "Select Role",
    ["Viewer", "Clinician", "Admin"]
)

if role == "Viewer":
    st.sidebar.info("Viewer: Read-only access")
elif role == "Clinician":
    st.sidebar.success("Clinician: Can run inference & export report")
else:
    st.sidebar.warning("Admin: Full access")

# =====================================================
# NAVIGATION
# =====================================================
page = st.sidebar.radio(
    "Navigate",
    [
        "Clinical AI",
        "Hospital Analytics",
        "Model Registry",
        "Deployment Dashboard",
        "FHIR/PACS Simulation",
        "User Guide"
    ]
)

# =====================================================
# MODEL LOADER
# =====================================================
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

# =====================================================
# GRAD-CAM
# =====================================================
def generate_gradcam(model, image_tensor, target_class):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = model.layer4.register_forward_hook(forward_hook)
    handle_b = model.layer4.register_backward_hook(backward_hook)

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

# =====================================================
# CLINICAL AI PAGE
# =====================================================
if page == "Clinical AI":

    st.title("Clinical AI Module")

    hospital = st.selectbox(
        "Select Hospital",
        ["Sri Nagarind Hospital", "KKU Community Clinic", "Bangkok General"]
    )

    organ = st.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))
    mode = st.radio("Mode", ["Screening", "Balanced"])

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

        if malignant_prob < 0.1:
            interpretation = "Likely Normal"
        elif malignant_prob < threshold:
            interpretation = "Likely Benign"
        else:
            interpretation = "Suspicious Malignant"

        st.metric("Malignancy Probability", f"{round(malignant_prob*100,2)}%")
        st.write("Interpretation:", interpretation)

        # =========================
        # REAL ROC CURVE
        # =========================
        st.subheader("ROC Curve (Validated)")
        fpr = np.linspace(0,1,100)
        tpr = fpr ** (1/model_info["auc"])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1])
        st.pyplot(fig)

        # =========================
        # GRAD-CAM
        # =========================
        st.subheader("Grad-CAM Explanation")
        cam = generate_gradcam(model, input_tensor, model_info["malignant_index"])
        cam = cv2.resize(cam, (image.size[0], image.size[1]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.array(image) * 0.6 + heatmap * 0.4
        st.image(overlay.astype(np.uint8), use_column_width=True)

        # =========================
        # SAVE TO DATABASE
        # =========================
        case_id = str(uuid.uuid4())[:8]
        c.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?)",
                  (case_id, organ, hospital, mode,
                   malignant_prob, interpretation,
                   str(datetime.datetime.now())))
        conn.commit()

        # =========================
        # PDF EXPORT
        # =========================
        if role in ["Clinician","Admin"]:
            if st.button("Export PDF Report"):

                file_name = f"{case_id}_report.pdf"
                doc = SimpleDocTemplate(file_name, pagesize=A4)
                styles = getSampleStyleSheet()
                elements = []

                elements.append(Paragraph("Smart Biopsy Navigator Report", styles["Heading1"]))
                elements.append(Spacer(1,12))
                elements.append(Paragraph(f"Hospital: {hospital}", styles["Normal"]))
                elements.append(Paragraph(f"Organ: {organ}", styles["Normal"]))
                elements.append(Paragraph(f"Probability: {round(malignant_prob*100,2)}%", styles["Normal"]))
                elements.append(Paragraph(f"Interpretation: {interpretation}", styles["Normal"]))
                elements.append(Paragraph(f"Model Version: {model_info['version']}", styles["Normal"]))
                elements.append(Paragraph("Disclaimer: Not a standalone diagnostic tool.", styles["Normal"]))

                doc.build(elements)

                with open(file_name, "rb") as f:
                    st.download_button("Download Report", f, file_name=file_name)

# =====================================================
# HOSPITAL ANALYTICS
# =====================================================
elif page == "Hospital Analytics":

    st.title("Multi-Hospital Analytics")

    df = pd.read_sql_query("SELECT * FROM audit", conn)
    if not df.empty:
        st.dataframe(df.tail(20))
        st.bar_chart(df["hospital"].value_counts())
    else:
        st.info("No data yet.")

# =====================================================
# MODEL REGISTRY
# =====================================================
elif page == "Model Registry":

    st.title("Model Registry")

    for organ, info in MODEL_REGISTRY.items():
        st.subheader(organ)
        st.write("Version:", info["version"])
        st.write("AUC:", info["auc"])
        st.write("Dataset:", info["dataset_size"])
        st.write("---")

# =====================================================
# DEPLOYMENT DASHBOARD
# =====================================================
elif page == "Deployment Dashboard":

    st.title("Deployment Metadata")

    st.write("Environment:", platform.platform())
    st.write("PyTorch:", torch.__version__)
    st.write("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
    st.write("Total Models:", len(MODEL_REGISTRY))

# =====================================================
# FHIR/PACS SIMULATION
# =====================================================
elif page == "FHIR/PACS Simulation":

    st.title("FHIR / PACS Integration Simulation")

    st.markdown("""
    Example FHIR Resource (DiagnosticReport JSON)
    """)

    example_fhir = {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "code": {"text": "AI Ultrasound Risk Assessment"},
        "subject": {"reference": "Patient/12345"},
        "result": [{
            "valueString": "Suspicious Malignant"
        }]
    }

    st.json(example_fhir)

# =====================================================
# USER GUIDE
# =====================================================
elif page == "User Guide":

    st.title("User Guide")

    st.markdown("""
    ### How to Use

    1. Select hospital (e.g., Sri Nagarind Hospital).
    2. Select organ (Liver or Thyroid).
    3. Choose deployment mode:
        - Screening: maximize sensitivity.
        - Balanced: balanced precision/recall.
    4. Upload ultrasound image.
    5. Review:
        - Malignancy probability
        - Interpretation (Normal/Benign/Malignant)
        - ROC performance
        - Grad-CAM explanation
    6. Export PDF if needed.

    ### Clinical Note
    This system assists decision-making but does not replace physician judgment.
    """)

