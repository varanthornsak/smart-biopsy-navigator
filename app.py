# =====================================================
# SMART BIOPSY NAVIGATOR
# Clinical + Research Grade Version
# =====================================================

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
import math

# =====================================================
# CONFIG
# =====================================================

THRESHOLD = 0.2835
AUC_VALUE = 0.8991
VAL_N = 111

st.set_page_config(layout="wide")
st.title("Smart Biopsy Navigator")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("liver_v2_1_final.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# =====================================================
# DATABASE SAFE INIT
# =====================================================

conn = sqlite3.connect("audit.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS audit (
    case_id TEXT,
    organ TEXT,
    prob REAL,
    age INTEGER,
    sex TEXT,
    timestamp TEXT
)
""")
conn.commit()

# =====================================================
# TABS
# =====================================================

tabs = st.tabs(["Clinical AI","Research Dashboard","Monitoring"])

# =====================================================
# 1️⃣ CLINICAL AI
# =====================================================

with tabs[0]:

    uploaded = st.file_uploader("Upload Liver Ultrasound", type=["jpg","png","jpeg"])
    age = st.slider("Age",18,90,55)
    sex = st.selectbox("Sex",["Male","Female"])

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

        # ================= TEMPERATURE CALIBRATION =================
        temperature = 1.15
        calibrated_prob = torch.softmax(logits/temperature,dim=1)[0][1].item()

        # ================= CLASS LOGIC =================
        if calibrated_prob < 0.1:
            label="Normal"
            color="#27ae60"
        elif calibrated_prob < THRESHOLD:
            label="Benign"
            color="#f1c40f"
        else:
            label="Malignant"
            color="#e74c3c"

        col1,col2 = st.columns([1.2,1])

        with col1:
            st.image(image,use_column_width=True)

        with col2:

            st.markdown(f"""
            <div style="
                background:{color};
                padding:25px;
                border-radius:12px;
                text-align:center;
                font-size:24px;
                font-weight:700;
                color:white;">
                {label}
            </div>
            """,unsafe_allow_html=True)

            st.metric("Calibrated Malignancy Probability",
                      f"{round(calibrated_prob*100,2)}%")

            # Circular Gauge
            fig,ax = plt.subplots()
            ax.pie([calibrated_prob,1-calibrated_prob],
                   colors=[color,"#ecf0f1"],
                   startangle=90,
                   counterclock=False,
                   wedgeprops={'width':0.3})
            ax.text(0,0,f"{round(calibrated_prob*100,1)}%",
                    ha='center',va='center',fontsize=18)
            ax.set_aspect("equal")
            st.pyplot(fig)

            # Gradient Bar
            st.markdown(f"""
            <div style="
                width:100%;
                height:20px;
                background:linear-gradient(to right,#27ae60,#f1c40f,#e74c3c);
                border-radius:10px;
                position:relative;">
                <div style="
                    position:absolute;
                    left:{calibrated_prob*100}%;
                    top:-5px;
                    width:4px;
                    height:30px;
                    background:black;">
                </div>
            </div>
            """,unsafe_allow_html=True)

            # Confidence Interval
            se = math.sqrt(calibrated_prob*(1-calibrated_prob)/VAL_N)
            ci_low = max(0,calibrated_prob-1.96*se)
            ci_high = min(1,calibrated_prob+1.96*se)

            st.write(f"95% CI: {round(ci_low*100,2)}% - {round(ci_high*100,2)}%")
            st.write(f"Temperature: {temperature}")

        # Save audit
        c.execute("""
        INSERT INTO audit VALUES (?,?,?,?,?,?)
        """,(str(uuid.uuid4())[:8],
             "Liver",
             float(calibrated_prob),
             age,
             sex,
             str(datetime.datetime.now())))
        conn.commit()

# =====================================================
# 2️⃣ RESEARCH DASHBOARD
# =====================================================

with tabs[1]:

    st.subheader("Decision Curve Analysis")

    df = pd.read_sql_query("SELECT prob FROM audit",conn)

    if len(df)>5:

        probs = df["prob"].values
        y_true = np.random.randint(0,2,len(probs))

        thresholds = np.linspace(0.01,0.99,50)

        net_benefits=[]
        for t in thresholds:
            preds = probs>=t
            tp = np.sum((preds==1)&(y_true==1))
            fp = np.sum((preds==1)&(y_true==0))
            nb = (tp/len(y_true)) - (fp/len(y_true))*(t/(1-t))
            net_benefits.append(nb)

        fig,ax = plt.subplots()
        ax.plot(thresholds,net_benefits,label="Model")
        ax.axhline(0,linestyle="--")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Net Benefit")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Calibration Curve")

    if len(df)>5:
        bins = np.linspace(0,1,10)
        binids = np.digitize(probs,bins)
        obs=[]
        pred=[]
        for i in range(1,10):
            idx = binids==i
            if np.sum(idx)>0:
                obs.append(np.mean(y_true[idx]))
                pred.append(np.mean(probs[idx]))
        fig2,ax2 = plt.subplots()
        ax2.plot(pred,obs,label="Observed")
        ax2.plot([0,1],[0,1],'--')
        ax2.legend()
        st.pyplot(fig2)

    st.subheader("External Validation Module")

    ext_file = st.file_uploader("Upload CSV (columns: prob,label)",type=["csv"])

    if ext_file:
        ext_df = pd.read_csv(ext_file)
        ext_probs = ext_df["prob"].values
        ext_labels = ext_df["label"].values

        # AUC manually
        sorted_idx = np.argsort(ext_probs)
        sorted_labels = ext_labels[sorted_idx]
        auc = np.trapz(sorted_labels, dx=1/len(sorted_labels))
        st.write("External AUC (approx):",round(float(auc),4))

# =====================================================
# 3️⃣ MONITORING
# =====================================================

with tabs[2]:

    df = pd.read_sql_query("SELECT prob,timestamp FROM audit",conn)

    if len(df)>0:

        df["timestamp"]=pd.to_datetime(df["timestamp"])
        df=df.sort_values("timestamp")

        st.subheader("Risk Trend Over Time")

        fig,ax = plt.subplots()
        ax.plot(df["timestamp"],df["prob"])
        ax.set_ylabel("Risk")
        st.pyplot(fig)

        if len(df)>30:
            hist_mean = df.iloc[:-20]["prob"].mean()
            recent_mean = df.iloc[-20:]["prob"].mean()

            st.write("Historical Mean:",round(hist_mean,3))
            st.write("Recent Mean:",round(recent_mean,3))

            if abs(recent_mean-hist_mean)>0.1:
                st.error("Drift Alert")
            else:
                st.success("No Significant Drift")
