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
import random

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

MODEL_REGISTRY = {
    "Liver": {
        "model_url": "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth",
        "threshold": 0.2835,
        "version": "v2.1 Binary Calibrated"
    },
    "Thyroid": {
        "model_url": None,
        "threshold": 0.5,
        "version": "Training"
    },
    "Breast": {"model_url": None, "threshold": 0.5, "version": "Planned"},
}

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.big-title {font-size:34px;font-weight:700;}
.subtitle {color:#6b7280;margin-bottom:15px;}
.card {padding:22px;border-radius:18px;font-weight:600;text-align:center;}
.green {background:#27ae60;color:white;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;color:white;}
.section {font-size:22px;font-weight:600;margin-top:30px;}
.metric-box {padding:15px;border-radius:12px;background:#f4f6f9;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SAFE DB
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
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?)",
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
    key = st.text_input("Access Key", type="password")

    if st.button("Login"):
        if key == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.hospital = hospital
        else:
            st.error("Invalid Key")
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>{st.session_state.hospital}</div>", unsafe_allow_html=True)

tabs = st.tabs(["Clinical AI", "Publication Metrics", "Monitoring", "Help"])

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(url):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# =========================================================
# MANUAL AUC
# =========================================================
def compute_auc(labels, probs):
    sorted_idx = np.argsort(probs)
    labels = labels[sorted_idx]
    probs = probs[sorted_idx]

    tprs = []
    fprs = []

    thresholds = np.unique(probs)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((preds==1)&(labels==1))
        fp = np.sum((preds==1)&(labels==0))
        fn = np.sum((preds==0)&(labels==1))
        tn = np.sum((preds==0)&(labels==0))

        tpr = tp/(tp+fn+1e-8)
        fpr = fp/(fp+tn+1e-8)
        tprs.append(tpr)
        fprs.append(fpr)

    # sort by fpr
    fprs, tprs = zip(*sorted(zip(fprs, tprs)))
    auc = np.trapz(tprs, fprs)
    return auc

# =========================================================
# BOOTSTRAP CI
# =========================================================
def bootstrap_auc_ci(labels, probs, n=500):
    aucs = []
    size = len(labels)
    for _ in range(n):
        idx = np.random.choice(range(size), size, replace=True)
        auc = compute_auc(labels[idx], probs[idx])
        aucs.append(auc)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

# =========================================================
# BRIER SCORE
# =========================================================
def brier_score(labels, probs):
    return np.mean((probs - labels)**2)

# =========================================================
# CLINICAL AI
# =========================================================
with tabs[0]:

    st.markdown("<div class='section'>Organ Selection</div>", unsafe_allow_html=True)

    organ = st.selectbox("Select Organ", list(MODEL_REGISTRY.keys()))
    info = MODEL_REGISTRY[organ]

    if info["model_url"] is None:
        st.warning("Model under development.")
        st.stop()

    model = load_model(info["model_url"])

    mode = st.radio("Mode", ["Screening", "Balanced"])
    threshold = info["threshold"] if mode=="Screening" else 0.5

    uploaded = st.file_uploader("Upload Ultrasound Image", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            prob = torch.softmax(logits,dim=1)[0][1].item()

        if prob < 0.1:
            label="Normal";color="green"
        elif prob < threshold:
            label="Benign";color="yellow"
        else:
            label="Malignant";color="red"

        col1,col2 = st.columns([1.3,1])

        with col1:
            st.image(image, use_column_width=True)

        with col2:
            st.markdown(f"<div class='card {color}'>{label}<br>{round(prob*100,2)}%</div>", unsafe_allow_html=True)

        # Interpretation table
        st.markdown("### Clinical Interpretation")
        df_result = pd.DataFrame({
            "Metric": ["Probability of Malignancy", "Applied Threshold", "Operating Mode"],
            "Value": [round(prob,4), threshold, mode]
        })
        st.table(df_result)

        safe_log(str(uuid.uuid4())[:8], st.session_state.hospital, organ, prob)

# =========================================================
# PUBLICATION METRICS
# =========================================================
with tabs[1]:

    st.markdown("<div class='section'>External Validation Module</div>", unsafe_allow_html=True)

    file = st.file_uploader("Upload CSV with columns: prob,label", type=["csv"])

    if file:
        df = pd.read_csv(file)
        probs = df["prob"].values
        labels = df["label"].values

        auc = compute_auc(labels, probs)
        ci_low, ci_high = bootstrap_auc_ci(labels, probs)
        brier = brier_score(labels, probs)

        st.markdown("### Performance Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["AUC", "95% CI Lower", "95% CI Upper", "Brier Score"],
            "Value": [round(auc,4), round(ci_low,4), round(ci_high,4), round(brier,4)]
        })
        st.table(metrics_df)

        # ROC Curve
        thresholds = np.linspace(0,1,100)
        tpr_list=[]
        fpr_list=[]

        for t in thresholds:
            preds = (probs>=t).astype(int)
            tp = np.sum((preds==1)&(labels==1))
            fp = np.sum((preds==1)&(labels==0))
            fn = np.sum((preds==0)&(labels==1))
            tn = np.sum((preds==0)&(labels==0))
            tpr = tp/(tp+fn+1e-8)
            fpr = fp/(fp+tn+1e-8)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        fig,ax = plt.subplots()
        ax.plot(fpr_list,tpr_list,label="ROC")
        ax.plot([0,1],[0,1],'--')
        ax.set_title("ROC Curve")
        st.pyplot(fig)

        # Retraining trigger
        if auc < 0.80:
            st.error("Performance degraded. Retraining recommended.")
        else:
            st.success("Model performance acceptable.")

# =========================================================
# MONITORING
# =========================================================
with tabs[2]:

    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM audit", conn)
            if not df.empty:
                st.metric("Total Cases", len(df))
                st.line_chart(df["prob"])
        except:
            st.info("Monitoring disabled.")

# =========================================================
# HELP
# =========================================================
with tabs[3]:

    st.markdown("### How to Use")
    st.write("""
1. Login with hospital key.
2. Select organ.
3. Upload ultrasound image.
4. Choose mode:
   - Screening = high sensitivity
   - Balanced = standard threshold
5. Review probability and interpretation table.
6. For publication metrics, upload validation CSV.
""")

    st.markdown("### Threshold Logic")
    st.write("""
Probability < 0.1 → Likely Normal  
0.1 – Threshold → Likely Benign  
≥ Threshold → Suspicious Malignant  
""")

    st.markdown("### Intended Use")
    st.write("""
Decision-support tool only. Not a standalone diagnostic system.
Final decision must be made by a licensed physician.
""")
