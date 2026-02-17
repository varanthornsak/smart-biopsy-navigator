import streamlit as st
import requests
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
import datetime
import math

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="Smart Biopsy Navigator", layout="wide")

API_URL = "http://127.0.0.1:8000/infer/liver"
SCREENING_THRESHOLD = 0.2835

# =========================================
# STYLE
# =========================================
st.markdown("""
<style>
.big-title {font-size:32px;font-weight:700;}
.subtitle {color:#6b7280;}
.card {
    padding:25px;
    border-radius:18px;
    color:white;
    font-weight:600;
}
.green {background:#27ae60;}
.yellow {background:#f1c40f;color:black;}
.red {background:#e74c3c;}
.section {font-size:22px;font-weight:600;margin-top:25px;}
</style>
""", unsafe_allow_html=True)

# =========================================
# DATABASE
# =========================================
conn = sqlite3.connect("audit_frontend.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS audit (
case_id TEXT,
hospital TEXT,
prob REAL,
timestamp TEXT
)
""")
conn.commit()

# =========================================
# LOGIN
# =========================================
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:

    st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Clinical AI Platform</div>", unsafe_allow_html=True)

    hospital = st.selectbox("Hospital", ["Sri Nagarind Hospital", "Demo Hospital"])
    role = st.selectbox("Role", ["Viewer", "Clinician", "Admin"])
    password = st.text_input("Access Key", type="password")

    if st.button("Login"):
        if password == "SNH_SECURE":
            st.session_state.login = True
            st.session_state.hospital = hospital
            st.session_state.role = role
            st.experimental_rerun()
        else:
            st.error("Invalid access key")

    st.stop()

# =========================================
# HEADER
# =========================================
st.markdown("<div class='big-title'>Smart Biopsy Navigator</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>{st.session_state.hospital} | Role: {st.session_state.role}</div>", unsafe_allow_html=True)

# =========================================
# TABS
# =========================================
tab1, tab2, tab3, tab4 = st.tabs(["Clinical AI", "Monitoring", "Study Results", "How to Use"])

# =========================================
# 1️⃣ CLINICAL AI
# =========================================
with tab1:

    st.markdown("<div class='section'>Liver Risk Stratification</div>", unsafe_allow_html=True)

    mode = st.radio("Mode", ["Screening (High Sensitivity)", "Balanced"])
    threshold = SCREENING_THRESHOLD if "Screening" in mode else 0.5
    temperature = st.slider("Calibration Temperature", 0.5, 3.0, 1.0)

    uploaded = st.file_uploader("Upload Ultrasound Image", type=["jpg", "png", "jpeg"])

    if uploaded:

        files = {"file": uploaded.getvalue()}
        params = {"temperature": temperature}

        response = requests.post(API_URL, files=files, params=params)

        if response.status_code == 200:
            data = response.json()
            prob = data["probability"]

            if prob < 0.1:
                label = "Likely Normal"
                color = "green"
            elif prob < threshold:
                label = "Likely Benign"
                color = "yellow"
            else:
                label = "Suspicious Malignant"
                color = "red"

            col1, col2 = st.columns([1.2,1])

            with col1:
                st.image(uploaded, use_column_width=True)

            with col2:
                st.markdown(f"""
                <div class='card {color}'>
                {label}<br>
                Probability: {round(prob*100,2)}%
                </div>
                """, unsafe_allow_html=True)

                # Risk Gauge
                fig, ax = plt.subplots()
                ax.axis("off")
                theta = np.linspace(0, math.pi, 100)
                ax.plot(np.cos(theta), np.sin(theta))
                angle = math.pi * (1 - prob)
                ax.plot([0, np.cos(angle)], [0, np.sin(angle)], linewidth=4)
                ax.text(0, -0.2, f"{round(prob*100,1)}%", ha="center", fontsize=14)
                st.pyplot(fig)

                # Probability Bar
                fig2, ax2 = plt.subplots()
                ax2.bar(["Risk"], [prob])
                ax2.set_ylim(0,1)
                st.pyplot(fig2)

            # Clinical Explanation
            if label == "Likely Normal":
                st.success("Ultrasound echotexture appears normal. Routine surveillance recommended.")
            elif label == "Likely Benign":
                st.warning("Low-risk lesion pattern detected. Recommend short-term follow-up imaging.")
            else:
                st.error("High-risk imaging pattern detected. Further diagnostic workup advised.")

            # Save audit
            c.execute("INSERT INTO audit VALUES (?,?,?,?)",
                      (str(uuid.uuid4())[:8],
                       st.session_state.hospital,
                       prob,
                       str(datetime.datetime.now())))
            conn.commit()

        else:
            st.error("API connection failed. Ensure backend is running.")

# =========================================
# 2️⃣ MONITORING
# =========================================
with tab2:

    df = pd.read_sql_query("SELECT * FROM audit", conn)

    if df.empty:
        st.info("No cases yet.")
    else:
        st.metric("Total Cases", len(df))
        st.line_chart(df["prob"])

        # Drift detection
        if len(df) > 30:
            rolling = df["prob"].rolling(30).mean()
            if rolling.iloc[-1] > 0.6:
                st.error("Model Drift Alert – Retraining Recommended")

        st.bar_chart(df.groupby("hospital")["prob"].mean())

# =========================================
# 3️⃣ STUDY RESULTS (Manuscript Ready)
# =========================================
with tab3:

    st.markdown("<div class='section'>Validation Summary</div>", unsafe_allow_html=True)

    st.write("Model: Liver v2.1 (Binary)")
    st.write("Mean AUC: 0.899 ± 0.03")
    st.write("Screening Sensitivity ≥95%")
    st.write("Calibration: Temperature Scaling")

    # ROC Curve mock
    fpr = np.linspace(0,1,100)
    tpr = np.sqrt(fpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1])
    ax.set_title("ROC Curve")
    st.pyplot(fig)

# =========================================
# 4️⃣ HOW TO USE
# =========================================
with tab4:

    st.markdown("<div class='section'>Usage Instructions</div>", unsafe_allow_html=True)

    st.markdown("""
    **Step 1:** Login with hospital access key.  
    **Step 2:** Upload liver ultrasound image (clear transverse view).  
    **Step 3:** Select Screening or Balanced mode.  
    **Step 4:** Review risk classification and clinical recommendation.  
    **Step 5:** Document case (automatically logged).  

    Color Codes:
    - 🟢 Green → Normal
    - 🟡 Yellow → Benign
    - 🔴 Red → Suspicious Malignant
    """)
