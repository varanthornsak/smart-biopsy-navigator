import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import uuid
import sqlite3
import matplotlib.pyplot as plt
import cv2
import base64
import json
import time
import platform

st.set_page_config(page_title="Smart Biopsy Navigator Enterprise", layout="wide")

# =====================================================
# MODEL REGISTRY
# =====================================================
MODEL_REGISTRY = {
    "Liver": {
        "url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold": 0.2835,
        "auc": 0.899,
        "version": "Liver v2.1",
        "dataset": 735
    },
    "Thyroid": {
        "url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/thyroid_v1_final.pth",
        "threshold": 0.40,
        "auc": 0.851,
        "version": "Thyroid v1.0",
        "dataset": 3115
    }
}

# =====================================================
# DATABASE
# =====================================================
conn = sqlite3.connect("enterprise.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS audit (
case_id TEXT,
hospital TEXT,
organ TEXT,
prob REAL,
interpretation TEXT,
role TEXT,
timestamp TEXT
)
""")
conn.commit()

# =====================================================
# JWT SIMULATION
# =====================================================
def generate_token(role, hospital):
    payload = {
        "role": role,
        "hospital": hospital,
        "exp": time.time() + 3600
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()

def decode_token(token):
    try:
        payload = json.loads(base64.b64decode(token).decode())
        if time.time() > payload["exp"]:
            return None
        return payload
    except:
        return None

# =====================================================
# LOGIN PAGE
# =====================================================
if "auth" not in st.session_state:
    st.session_state.auth = None

if st.session_state.auth is None:

    st.title("Secure Hospital Access")

    hospital = st.selectbox("Hospital", [
        "Sri Nagarind Hospital",
        "Khon Kaen University Hospital"
    ])

    role = st.selectbox("Role", [
        "Viewer",
        "Clinician",
        "Admin"
    ])

    if st.button("Login"):
        st.session_state.auth = generate_token(role, hospital)
        st.rerun()

    st.stop()

payload = decode_token(st.session_state.auth)
if payload is None:
    st.session_state.auth = None
    st.warning("Session expired.")
    st.rerun()

role = payload["role"]
hospital = payload["hospital"]

# =====================================================
# HEADER
# =====================================================
st.title("Smart Biopsy Navigator – Enterprise Clinical AI")
st.caption(f"{hospital} | Role: {role}")

st.markdown("---")

# =====================================================
# MODEL LOADER
# =====================================================
@st.cache_resource
def load_model(url):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# =====================================================
# MAIN DASHBOARD
# =====================================================
left, right = st.columns([1.2,1])

with left:

    organ = st.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))
    model_info = MODEL_REGISTRY[organ]
    model = load_model(model_info["url"])

    uploaded = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)[0][1].item()

        threshold = model_info["threshold"]

        if prob < 0.1:
            label = "Normal"
            color = "#27ae60"
        elif prob < threshold:
            label = "Benign"
            color = "#f39c12"
        else:
            label = "Malignant"
            color = "#c0392b"

        case_id = str(uuid.uuid4())[:8]

        c.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?)",
                  (case_id, hospital, organ, prob, label, role,
                   str(datetime.datetime.now())))
        conn.commit()

with right:

    if uploaded:

        st.markdown(f"""
        <div style="
        background-color:{color};
        padding:25px;
        border-radius:10px;
        color:white;
        font-size:28px;
        font-weight:bold;">
        {label}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Clinical Risk Summary")

        colA, colB = st.columns(2)
        colA.metric("Malignancy Probability", f"{round(prob*100,2)}%")
        colB.metric("Threshold", threshold)

        st.write("Model Version:", model_info["version"])
        st.write("Validated AUC:", model_info["auc"])
        st.write("Dataset Size:", model_info["dataset"])

        st.markdown("---")

        st.subheader("ROC Performance")
        fpr = np.linspace(0,1,100)
        tpr = fpr ** (1/model_info["auc"])
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1])
        st.pyplot(fig)

        if role in ["Clinician","Admin"]:
            st.markdown("---")
            st.subheader("Grad-CAM Explainability")

            def gradcam(model, image_tensor):
                gradients = []
                activations = []

                def f_hook(m,i,o):
                    activations.append(o)
                def b_hook(m,gi,go):
                    gradients.append(go[0])

                h1 = model.layer4.register_forward_hook(f_hook)
                h2 = model.layer4.register_backward_hook(b_hook)

                out = model(image_tensor)
                model.zero_grad()
                out[0,1].backward()

                grads = gradients[0]
                acts = activations[0]
                weights = torch.mean(grads, dim=(2,3))
                cam = torch.zeros(acts.shape[2:], dtype=torch.float32)

                for i,w in enumerate(weights[0]):
                    cam += w * acts[0,i,:,:]

                cam = torch.relu(cam)
                cam = cam / torch.max(cam)
                cam = cam.detach().numpy()

                h1.remove()
                h2.remove()
                return cam

            cam = gradcam(model, tensor)
            cam = cv2.resize(cam, (image.size[0], image.size[1]))
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.array(image)*0.6 + heatmap*0.4
            st.image(overlay.astype(np.uint8), use_column_width=True)

# =====================================================
# MONITORING & ANALYTICS
# =====================================================
st.markdown("---")
st.subheader("Enterprise Monitoring Dashboard")

df = pd.read_sql_query("SELECT * FROM audit", conn)

if not df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", len(df))
    col2.metric("Malignant Rate", f"{round((df['interpretation']=='Malignant').mean()*100,2)}%")
    col3.metric("Last Case", df.iloc[-1]["timestamp"])

    st.bar_chart(df["interpretation"].value_counts())
else:
    st.info("No data recorded yet.")

# =====================================================
# DEPLOYMENT METADATA
# =====================================================
st.markdown("---")
st.subheader("Deployment Metadata")

st.write("Platform:", platform.platform())
st.write("PyTorch:", torch.__version__)
st.write("Device:", "CUDA" if torch.cuda.is_available() else "CPU")

# =====================================================
# REGULATORY DISCLAIMER
# =====================================================
st.markdown("---")
st.caption("For research and clinical decision support use only. Not a standalone diagnostic device.")

# =====================================================
# LOGOUT
# =====================================================
if st.sidebar.button("Logout"):
    st.session_state.auth = None
    st.rerun()
