import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import datetime
import uuid
import math
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

MODEL_REGISTRY = {
    "Liver": {
        "status": "Active",
        "url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold": 0.2835
    },
    "Thyroid": {"status": "Training"},
    "Breast": {"status": "Planned"},
    "Prostate": {"status": "Planned"}
}

# ======================================================
# STYLE
# ======================================================
st.markdown("""
<style>
.big-title {font-size:32px;font-weight:700;}
.card {padding:20px;border-radius:15px;color:white;font-weight:600;text-align:center;}
.green {background:#27ae60;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;}
.section {font-size:20px;font-weight:600;margin-top:25px;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# DATABASE
# ======================================================
conn = sqlite3.connect("audit.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS audit (
case_id TEXT,
organ TEXT,
prob REAL,
timestamp TEXT
)
""")
conn.commit()

# ======================================================
# LOGIN
# ======================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    hospital = st.selectbox("Hospital", ["Sri Nagarind Hospital","Demo Hospital"])
    password = st.text_input("Access Key", type="password")
    if st.button("Login"):
        if password == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.hospital = hospital
        else:
            st.error("Invalid key")
    st.stop()

# ======================================================
# LOAD MODEL (Liver only active)
# ======================================================
@st.cache_resource
def load_model(url):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features,2)
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model(MODEL_REGISTRY["Liver"]["url"])

# ======================================================
# GRAD-CAM (Stable version)
# ======================================================
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = torch.argmax(output)
    model.zero_grad()
    output[0,class_idx].backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()

    weights = torch.mean(grads, dim=(2,3))
    cam = torch.zeros(acts.shape[2:])

    for i,w in enumerate(weights[0]):
        cam += w * acts[0,i]

    cam = torch.relu(cam)
    cam = cam / torch.max(cam)
    return cam.numpy()

# ======================================================
# HEADER
# ======================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)

tabs = st.tabs(["Clinical AI","Calibration","External Validation","Multi-Organ Registry"])

# ======================================================
# 1️⃣ CLINICAL AI
# ======================================================
with tabs[0]:

    st.markdown("<div class='section'>Organ Selection</div>", unsafe_allow_html=True)

    organ = st.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))

    if MODEL_REGISTRY[organ]["status"] != "Active":
        st.warning(f"{organ} model not yet deployed. Status: {MODEL_REGISTRY[organ]['status']}")
        st.stop()

    uploaded = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output,dim=1)[0][1].item()

        threshold = MODEL_REGISTRY[organ]["threshold"]

        if prob < 0.1:
            label="Normal";color="green"
        elif prob < threshold:
            label="Benign";color="yellow"
        else:
            label="Malignant";color="red"

        col1,col2 = st.columns([1.2,1])

        with col1:
            st.image(image,use_column_width=True)

            cam = generate_gradcam(model, tensor)
            st.image(cam, clamp=True, caption="Grad-CAM Heatmap")

        with col2:
            st.markdown(f"<div class='card {color}'>{label}<br>Prob: {round(prob*100,2)}%</div>",unsafe_allow_html=True)

            fig, ax = plt.subplots()
            ax.axis("off")
            theta = np.linspace(0, math.pi,100)
            ax.plot(np.cos(theta),np.sin(theta))
            angle = math.pi*(1-prob)
            ax.plot([0,np.cos(angle)],[0,np.sin(angle)],linewidth=4)
            ax.text(0,-0.2,f"{round(prob*100,1)}%",ha="center")
            st.pyplot(fig)

        c.execute("INSERT INTO audit VALUES (?,?,?,?)",
                  (str(uuid.uuid4())[:8],organ,prob,str(datetime.datetime.now())))
        conn.commit()

# ======================================================
# 2️⃣ CALIBRATION
# ======================================================
with tabs[1]:

    st.markdown("<div class='section'>Calibration Curve</div>", unsafe_allow_html=True)

    df = pd.read_sql_query("SELECT * FROM audit WHERE organ='Liver'",conn)

    if len(df) > 20:
        probs = df["prob"].values
        labels = np.random.randint(0,2,len(probs))  # simulated true labels

        frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10)

        fig, ax = plt.subplots()
        ax.plot(mean_pred, frac_pos)
        ax.plot([0,1],[0,1])
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        st.pyplot(fig)
    else:
        st.info("Not enough data for calibration analysis.")

# ======================================================
# 3️⃣ EXTERNAL VALIDATION
# ======================================================
with tabs[2]:

    st.markdown("<div class='section'>External Validation Upload</div>", unsafe_allow_html=True)

    csv = st.file_uploader("Upload CSV (columns: prob,label)", type=["csv"])

    if csv:
        data = pd.read_csv(csv)

        auc = roc_auc_score(data["label"], data["prob"])
        fpr, tpr, _ = roc_curve(data["label"], data["prob"])

        st.write("External AUC:", round(auc,4))

        fig, ax = plt.subplots()
        ax.plot(fpr,tpr)
        ax.plot([0,1],[0,1])
        ax.set_title("External ROC")
        st.pyplot(fig)

# ======================================================
# 4️⃣ MULTI-ORGAN REGISTRY
# ======================================================
with tabs[3]:

    for organ,info in MODEL_REGISTRY.items():
        st.write(f"{organ} — Status: {info['status']}")
