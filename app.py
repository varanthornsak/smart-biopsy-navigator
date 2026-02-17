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
DEFAULT_THRESHOLD = 0.2835

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.big-title {font-size:34px;font-weight:700;}
.subtitle {color:#6b7280;margin-bottom:15px;}
.card {padding:20px;border-radius:16px;font-weight:600;text-align:center;}
.green {background:#27ae60;color:white;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;color:white;}
.section {font-size:22px;font-weight:600;margin-top:30px;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# =========================================================
# METRIC FUNCTIONS
# =========================================================

def confusion_metrics(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tp = np.sum((preds==1)&(labels==1))
    fp = np.sum((preds==1)&(labels==0))
    fn = np.sum((preds==0)&(labels==1))
    tn = np.sum((preds==0)&(labels==0))

    sens = tp/(tp+fn+1e-8)
    spec = tn/(tn+fp+1e-8)
    ppv = tp/(tp+fp+1e-8)
    npv = tn/(tn+fn+1e-8)

    return tp,fp,fn,tn,sens,spec,ppv,npv

def compute_auc(labels, probs):
    sorted_idx = np.argsort(probs)
    labels = labels[sorted_idx]
    probs = probs[sorted_idx]
    tprs=[]; fprs=[]
    thresholds=np.unique(probs)
    for t in thresholds:
        tp,fp,fn,tn,_,_,_,_ = confusion_metrics(labels,probs,t)
        tpr = tp/(tp+fn+1e-8)
        fpr = fp/(fp+tn+1e-8)
        tprs.append(tpr)
        fprs.append(fpr)
    fprs,tprs = zip(*sorted(zip(fprs,tprs)))
    return np.trapz(tprs,fprs)

def calibration_slope_intercept(labels, probs):
    logit = np.log(probs/(1-probs+1e-8)+1e-8)
    X = np.vstack([np.ones(len(logit)), logit]).T
    y = labels
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    intercept = beta[0]
    slope = beta[1]
    return intercept, slope

def decision_curve(labels, probs):
    thresholds = np.linspace(0.01,0.99,50)
    net_benefits=[]
    N=len(labels)
    for pt in thresholds:
        preds=(probs>=pt).astype(int)
        tp=np.sum((preds==1)&(labels==1))
        fp=np.sum((preds==1)&(labels==0))
        nb=(tp/N)-(fp/N)*(pt/(1-pt))
        net_benefits.append(nb)
    return thresholds, net_benefits

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enterprise + Academic Mode</div>", unsafe_allow_html=True)

tabs = st.tabs(["Clinical AI", "Publication Analytics", "Prospective Simulation"])

# =========================================================
# CLINICAL AI
# =========================================================
with tabs[0]:

    threshold = st.slider("Operating Threshold",0.05,0.95,DEFAULT_THRESHOLD)

    uploaded = st.file_uploader("Upload Liver Ultrasound", type=["jpg","png","jpeg"])

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

        if prob<0.1:
            label="Normal";color="green"
        elif prob<threshold:
            label="Benign";color="yellow"
        else:
            label="Malignant";color="red"

        col1,col2=st.columns([1.2,1])
        with col1:
            st.image(image,use_column_width=True)
        with col2:
            st.markdown(f"<div class='card {color}'>{label}<br>{round(prob*100,2)}%</div>",unsafe_allow_html=True)

        st.markdown("### Clinical Commentary (CC)")
        st.write(f"""
        The predicted probability of malignancy is {round(prob,3)}.
        Under the current threshold ({round(threshold,2)}), this case is classified as {label}.
        Clinical decision should integrate imaging morphology, patient risk profile, and histopathological correlation.
        """)

# =========================================================
# PUBLICATION ANALYTICS
# =========================================================
with tabs[1]:

    file = st.file_uploader("Upload Validation CSV (prob,label,center)", type=["csv"])

    if file:
        df = pd.read_csv(file)
        probs = df["prob"].values
        labels = df["label"].values

        auc = compute_auc(labels,probs)
        intercept,slope = calibration_slope_intercept(labels,probs)
        tp,fp,fn,tn,sens,spec,ppv,npv = confusion_metrics(labels,probs,DEFAULT_THRESHOLD)

        st.markdown("### Performance Table")
        perf = pd.DataFrame({
            "Metric":["AUC","Sensitivity","Specificity","PPV","NPV",
                      "Calibration Intercept","Calibration Slope"],
            "Value":[round(auc,4),round(sens,3),round(spec,3),
                     round(ppv,3),round(npv,3),
                     round(intercept,4),round(slope,4)]
        })
        st.table(perf)

        # ROC
        thresholds=np.linspace(0,1,100)
        tpr=[];fpr=[]
        for t in thresholds:
            tp,fp,fn,tn,_,_,_,_=confusion_metrics(labels,probs,t)
            tpr.append(tp/(tp+fn+1e-8))
            fpr.append(fp/(fp+tn+1e-8))
        fig,ax=plt.subplots()
        ax.plot(fpr,tpr)
        ax.plot([0,1],[0,1],'--')
        ax.set_title("ROC Curve")
        st.pyplot(fig)

        # Decision Curve
        th,nb=decision_curve(labels,probs)
        fig2,ax2=plt.subplots()
        ax2.plot(th,nb,label="Model")
        ax2.set_title("Decision Curve Analysis")
        ax2.set_xlabel("Threshold Probability")
        ax2.set_ylabel("Net Benefit")
        st.pyplot(fig2)

        # Multi-center
        if "center" in df.columns:
            st.markdown("### External Multi-Center Comparison")
            centers=df["center"].unique()
            center_auc=[]
            for c in centers:
                sub=df[df["center"]==c]
                center_auc.append(compute_auc(sub["label"].values,sub["prob"].values))
            center_df=pd.DataFrame({"Center":centers,"AUC":center_auc})
            st.bar_chart(center_df.set_index("Center"))

# =========================================================
# PROSPECTIVE SIMULATION
# =========================================================
with tabs[2]:

    st.markdown("### Prevalence Shift Simulation")
    base_prev = st.slider("Simulated Prevalence",0.01,0.5,0.2)

    N=1000
    simulated_labels=np.random.binomial(1,base_prev,N)
    simulated_probs=simulated_labels*0.7+(1-simulated_labels)*0.2
    simulated_probs+=np.random.normal(0,0.1,N)
    simulated_probs=np.clip(simulated_probs,0,1)

    tp,fp,fn,tn,sens,spec,ppv,npv = confusion_metrics(simulated_labels,simulated_probs,DEFAULT_THRESHOLD)

    sim_df=pd.DataFrame({
        "Metric":["Sensitivity","Specificity","PPV","NPV"],
        "Value":[round(sens,3),round(spec,3),round(ppv,3),round(npv,3)]
    })

    st.table(sim_df)

    st.write("""
    This module simulates prospective deployment under varying prevalence.
    It demonstrates how PPV and NPV change under real-world epidemiological shifts.
    """)
