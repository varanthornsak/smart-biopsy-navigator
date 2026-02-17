import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Smart Biopsy Navigator – Academic Mode", layout="wide")

MODEL_URL = "https://huggingface.co/Varanthorn/smart-biopsy-model/resolve/main/liver_v2_1_final.pth"
DEFAULT_THRESHOLD = 0.2835

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.big-title {font-size:36px;font-weight:700;}
.subtitle {color:#6b7280;margin-bottom:20px;}
.section {font-size:22px;font-weight:600;margin-top:30px;}
.card {padding:20px;border-radius:14px;font-weight:600;text-align:center;}
.green {background:#27ae60;color:white;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;color:white;}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
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
        tprs.append(tp/(tp+fn+1e-8))
        fprs.append(fp/(fp+tn+1e-8))
    fprs,tprs = zip(*sorted(zip(fprs,tprs)))
    return np.trapz(tprs,fprs)

def bootstrap_auc_ci(labels, probs, n=500):
    aucs=[]
    N=len(labels)
    for _ in range(n):
        idx=np.random.choice(range(N),N,replace=True)
        aucs.append(compute_auc(labels[idx],probs[idx]))
    return np.percentile(aucs,2.5),np.percentile(aucs,97.5)

def calibration_slope_intercept(labels, probs):
    logit = np.log(probs/(1-probs+1e-8)+1e-8)
    X = np.vstack([np.ones(len(logit)), logit]).T
    beta = np.linalg.inv(X.T@X)@X.T@labels
    return beta[0],beta[1]

def brier_score(labels, probs):
    return np.mean((probs-labels)**2)

def decision_curve(labels, probs):
    thresholds=np.linspace(0.01,0.99,50)
    nb_model=[]
    nb_all=[]
    N=len(labels)
    prevalence=np.mean(labels)
    for pt in thresholds:
        tp,fp,fn,tn,_,_,_,_=confusion_metrics(labels,probs,pt)
        nb=(tp/N)-(fp/N)*(pt/(1-pt))
        nb_model.append(nb)
        nb_all.append(prevalence-(1-prevalence)*(pt/(1-pt)))
    return thresholds,nb_model,nb_all

def nri_idi(labels, probs, threshold):
    preds=(probs>=threshold).astype(int)
    risk_old=np.full(len(labels),0.5)
    reclass_up=((preds==1)&(risk_old==0))
    reclass_down=((preds==0)&(risk_old==1))
    nri=(np.mean(reclass_up[labels==1])-np.mean(reclass_down[labels==1])) + \
        (np.mean(reclass_down[labels==0])-np.mean(reclass_up[labels==0]))
    idi=np.mean(probs[labels==1])-np.mean(probs[labels==0])
    return nri,idi

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Absolute Final Academic Mode</div>", unsafe_allow_html=True)

tabs=st.tabs(["Clinical AI","Validation & Publication","Prospective & Health Economics"])

# =========================================================
# CLINICAL AI
# =========================================================
with tabs[0]:
    threshold=st.slider("Operating Threshold",0.05,0.95,DEFAULT_THRESHOLD)

    uploaded=st.file_uploader("Upload Liver Ultrasound",type=["jpg","png","jpeg"])

    if uploaded:
        image=Image.open(uploaded).convert("RGB")
        transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        tensor=transform(image).unsqueeze(0)

        with torch.no_grad():
            prob=torch.softmax(model(tensor),dim=1)[0][1].item()

        if prob<0.1:
            label="Normal";color="green"
        elif prob<threshold:
            label="Benign";color="yellow"
        else:
            label="Malignant";color="red"

        col1,col2=st.columns([1.3,1])
        with col1:
            st.image(image,use_column_width=True)
        with col2:
            st.markdown(f"<div class='card {color}'>{label}<br>{round(prob*100,2)}%</div>",unsafe_allow_html=True)

        st.markdown("### Detailed Clinical Interpretation")
        st.write(f"""
        Predicted malignancy probability: {round(prob,3)}  
        Applied threshold: {round(threshold,2)}  
        Classification: {label}

        Clinical recommendation:
        - Normal: routine follow-up.
        - Benign: correlate with ultrasound morphology and consider interval imaging.
        - Malignant: recommend biopsy or oncologic referral.
        """)

# =========================================================
# VALIDATION & PUBLICATION
# =========================================================
with tabs[1]:
    file=st.file_uploader("Upload Validation CSV (prob,label,center optional)",type=["csv"])

    if file:
        df=pd.read_csv(file)
        probs=df["prob"].values
        labels=df["label"].values

        auc=compute_auc(labels,probs)
        ci_low,ci_high=bootstrap_auc_ci(labels,probs)
        brier=brier_score(labels,probs)
        intercept,slope=calibration_slope_intercept(labels,probs)
        tp,fp,fn,tn,sens,spec,ppv,npv=confusion_metrics(labels,probs,DEFAULT_THRESHOLD)
        nri,idi=nri_idi(labels,probs,DEFAULT_THRESHOLD)

        st.markdown("### Publication Metrics Table")
        table=pd.DataFrame({
            "Metric":["AUC","95% CI Lower","95% CI Upper","Sensitivity",
                      "Specificity","PPV","NPV",
                      "Calibration Intercept","Calibration Slope",
                      "Brier Score","NRI","IDI"],
            "Value":[round(auc,4),round(ci_low,4),round(ci_high,4),
                     round(sens,3),round(spec,3),round(ppv,3),round(npv,3),
                     round(intercept,4),round(slope,4),
                     round(brier,4),round(nri,4),round(idi,4)]
        })
        st.table(table)

        # ROC
        thresholds=np.linspace(0,1,100)
        tpr=[];fpr=[]
        for t in thresholds:
            tp,fp,fn,tn,_,_,_,_=confusion_metrics(labels,probs,t)
            tpr.append(tp/(tp+fn+1e-8))
            fpr.append(fp/(fp+tn+1e-8))
        fig,ax=plt.subplots()
        ax.plot(fpr,tpr,label="Model")
        ax.plot([0,1],[0,1],'--')
        ax.set_title("ROC Curve")
        st.pyplot(fig)

        # Decision Curve
        th,nb_model,nb_all=decision_curve(labels,probs)
        fig2,ax2=plt.subplots()
        ax2.plot(th,nb_model,label="Model")
        ax2.plot(th,nb_all,label="Treat All")
        ax2.axhline(0,color='black',linestyle='--',label="Treat None")
        ax2.legend()
        ax2.set_title("Decision Curve Analysis")
        st.pyplot(fig2)

        # Multi-center
        if "center" in df.columns:
            st.markdown("### Multi-Center AUC")
            centers=df["center"].unique()
            center_auc=[]
            for c in centers:
                sub=df[df["center"]==c]
                center_auc.append(compute_auc(sub["label"].values,sub["prob"].values))
            center_df=pd.DataFrame({"Center":centers,"AUC":center_auc})
            st.bar_chart(center_df.set_index("Center"))

# =========================================================
# PROSPECTIVE & HEALTH ECONOMICS
# =========================================================
with tabs[2]:
    st.markdown("### Prevalence Shift Simulation")

    prev=st.slider("Simulated Prevalence",0.01,0.5,0.2)
    cost_fp=st.slider("Cost of False Positive",100,5000,1000)
    cost_fn=st.slider("Cost of False Negative",1000,50000,10000)

    N=1000
    labels=np.random.binomial(1,prev,N)
    probs=labels*0.7+(1-labels)*0.2
    probs+=np.random.normal(0,0.1,N)
    probs=np.clip(probs,0,1)

    tp,fp,fn,tn,sens,spec,ppv,npv=confusion_metrics(labels,probs,DEFAULT_THRESHOLD)

    total_cost=fp*cost_fp+fn*cost_fn

    sim_table=pd.DataFrame({
        "Metric":["Sensitivity","Specificity","PPV","NPV","Total Cost"],
        "Value":[round(sens,3),round(spec,3),
                 round(ppv,3),round(npv,3),
                 total_cost]
    })
    st.table(sim_table)

    st.write("""
    This simulation demonstrates economic impact under prospective deployment.
    Health-economic outputs depend strongly on disease prevalence and operating threshold.
    """)
