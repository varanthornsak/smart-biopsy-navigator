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

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth"
SCREENING_THRESHOLD = 0.2835

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.big-title {font-size:34px;font-weight:700;}
.subtitle {color:#6b7280;margin-bottom:20px;}
.card {padding:25px;border-radius:18px;font-weight:600;text-align:center;}
.green {background:#27ae60;color:white;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;color:white;}
.section {font-size:22px;font-weight:600;margin-top:30px;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SAFE DB INIT (Cloud-Proof)
# =========================================================
def init_db():
    try:
        conn = sqlite3.connect("audit.db", check_same_thread=False)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS audit (
            case_id TEXT,
            hospital TEXT,
            organ TEXT,
            prob REAL,
            timestamp TEXT
        )
        """)
        conn.commit()
        return conn
    except:
        return None

conn = init_db()

def safe_log(case_id, hospital, organ, prob):
    if conn is None:
        return
    try:
        conn.execute(
            "INSERT INTO audit (case_id, hospital, organ, prob, timestamp) VALUES (?, ?, ?, ?, ?)",
            (case_id, hospital, organ, float(prob), str(datetime.datetime.now()))
        )
        conn.commit()
    except:
        pass

# =========================================================
# LOGIN
# =========================================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Enterprise Clinical AI Platform</div>", unsafe_allow_html=True)

    hospital = st.selectbox("Hospital", ["Sri Nagarind Hospital", "Demo Hospital"])
    access = st.text_input("Access Key", type="password")

    if st.button("Login"):
        if access == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.hospital = hospital
        else:
            st.error("Invalid Access Key")
    st.stop()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features,2)
    state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# =========================================================
# STABLE GRAD-CAM
# =========================================================
def generate_gradcam(model, image_tensor):
    target_layer = model.layer4[-1]
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    score = output[:,1]
    model.zero_grad()
    score.backward()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2,3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam.detach().numpy()
    cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)

    handle_f.remove()
    handle_b.remove()

    return cam

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>{st.session_state.hospital}</div>", unsafe_allow_html=True)

tabs = st.tabs(["Clinical AI", "External Validation", "Monitoring"])

# =========================================================
# CLINICAL AI
# =========================================================
with tabs[0]:

    mode = st.radio("Mode", ["Screening", "Balanced"])
    threshold = SCREENING_THRESHOLD if mode=="Screening" else 0.5
    temperature = st.slider("Calibration Temperature", 0.5, 3.0, 1.0)

    uploaded = st.file_uploader("Upload Liver Ultrasound", type=["jpg","png","jpeg"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)/temperature
            prob = torch.softmax(logits,dim=1)[0][1].item()

        if prob < 0.1:
            label="Normal";color="green"
        elif prob < threshold:
            label="Benign";color="yellow"
        else:
            label="Malignant";color="red"

        col1,col2 = st.columns([1.2,1])

        with col1:
            st.image(image,use_column_width=True)

            cam = generate_gradcam(model,tensor)
            fig,ax=plt.subplots()
            ax.imshow(image)
            ax.imshow(cam,cmap="jet",alpha=0.4)
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.markdown(f"<div class='card {color}'>{label}<br>{round(prob*100,2)}%</div>",unsafe_allow_html=True)

            fig2,ax2=plt.subplots()
            ax2.axis("off")
            theta=np.linspace(0,math.pi,100)
            ax2.plot(np.cos(theta),np.sin(theta))
            angle=math.pi*(1-prob)
            ax2.plot([0,np.cos(angle)],[0,np.sin(angle)],linewidth=4)
            ax2.text(0,-0.2,f"{round(prob*100,1)}%",ha="center")
            st.pyplot(fig2)

        safe_log(str(uuid.uuid4())[:8], st.session_state.hospital, "Liver", prob)

# =========================================================
# EXTERNAL VALIDATION
# =========================================================
with tabs[1]:

    val_file = st.file_uploader("Upload CSV (prob,label)", type=["csv"])

    if val_file:
        df = pd.read_csv(val_file)
        probs=df["prob"].values
        labels=df["label"].values

        thresholds=np.linspace(0,1,100)
        tpr_list=[];fpr_list=[]

        for t in thresholds:
            preds=(probs>=t).astype(int)
            tp=np.sum((preds==1)&(labels==1))
            fp=np.sum((preds==1)&(labels==0))
            fn=np.sum((preds==0)&(labels==1))
            tn=np.sum((preds==0)&(labels==0))
            tpr=tp/(tp+fn+1e-8)
            fpr=fp/(fp+tn+1e-8)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        fig,ax=plt.subplots()
        ax.plot(fpr_list,tpr_list)
        ax.plot([0,1],[0,1])
        ax.set_title("External ROC")
        st.pyplot(fig)

# =========================================================
# MONITORING
# =========================================================
with tabs[2]:

    if conn:
        try:
            df=pd.read_sql_query("SELECT * FROM audit",conn)
            if not df.empty:
                st.metric("Total Cases",len(df))
                st.line_chart(df["prob"])
        except:
            st.info("Monitoring disabled.")
