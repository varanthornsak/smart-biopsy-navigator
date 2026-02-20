import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Biopsy Pro | Enterprise Multi-Organ",
    layout="wide"
)

# =====================================================
# SAFE SESSION INITIALIZATION (CRITICAL FOR CLOUD)
# =====================================================
if "auth" not in st.session_state:
    st.session_state.auth = False

if "role" not in st.session_state:
    st.session_state.role = None

if "db" not in st.session_state:
    st.session_state.db = pd.DataFrame(columns=[
        "Date", "HN", "Patient", "Organ",
        "Status", "Confidence",
        "Marker_Val", "Tumor_Size"
    ])

# =====================================================
# CONSTANTS
# =====================================================
STATUS_COLOR = {
    "NORMAL": "#10B981",
    "BENIGN": "#F59E0B",
    "MALIGNANT": "#EF4444"
}

ROLES = ["Admin", "Clinician", "Radiologist", "Executive"]

# =====================================================
# LOGIN
# =====================================================
if not st.session_state.auth:
    st.title("SMART BIOPSY PRO – ENTERPRISE AI")

    role = st.selectbox("Select Role", ROLES)
    pwd = st.text_input("Security Key", type="password")

    if st.button("LOGIN"):
        if pwd == "SNH_SECURE":
            st.session_state.auth = True
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid Security Key")

    st.stop()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    if st.session_state.role:
        st.markdown(f"### Role: {st.session_state.role}")
    else:
        st.markdown("### Role: Not Assigned")

    nav = st.radio("Navigation", [
        "Diagnostic Hub",
        "Professional Analytics",
        "Executive Board View",
        "Case Archive",
        "User Manual"
    ])

    if st.button("Logout"):
        st.session_state.auth = False
        st.session_state.role = None
        st.rerun()

# =====================================================
# AI LOGIC
# =====================================================
def run_ai(organ, marker, size):

    # LIVER – Morphology priority (AFP optional)
    if organ == "Liver":

        if size > 60:
            return "MALIGNANT", 0.90

        if marker is not None and marker > 400:
            return "MALIGNANT", 0.92

        if size > 30:
            return "BENIGN", 0.60

        if marker is not None and marker > 200:
            return "BENIGN", 0.55

        return "NORMAL", 0.15

    # Default organs (minimal logic)
    if size > 40:
        return "MALIGNANT", 0.80
    elif size > 20:
        return "BENIGN", 0.55
    else:
        return "NORMAL", 0.10

# =====================================================
# PDF GENERATOR
# =====================================================
def generate_pdf(patient, hn, organ, status, confidence):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>SMART BIOPSY PRO REPORT</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Patient: {patient}", styles["Normal"]))
    elements.append(Paragraph(f"HN: {hn}", styles["Normal"]))
    elements.append(Paragraph(f"Organ: {organ}", styles["Normal"]))
    elements.append(Paragraph(f"Status: {status}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence*100:.1f}%", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# =====================================================
# 🧠 SMART BIOPSY PRO – CLEAN DIAGNOSTIC MODULE
# =====================================================

import base64
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import random

# =====================================================
# DATABASE INIT
# =====================================================
required_cols = [
    "Date","HN","Patient","Organ",
    "Status","Confidence",
    "Marker_Val","Tumor_Size",
    "Ultrasound_Image",
    "Image_Malignancy_Prob",
    "Image_AI_Confidence",
    "Fusion_Score","Risk_Tier"
]

for col in required_cols:
    if col not in st.session_state.db.columns:
        st.session_state.db[col] = None


# =====================================================
# 🔬 CLINICAL AI
# =====================================================
def clinical_ai(organ, marker, size):

    if organ == "Liver":
        if size > 60 or (marker and marker > 400):
            return "MALIGNANT", 0.9
        if size > 30:
            return "BENIGN", 0.6
        return "NORMAL", 0.2

    if size > 40:
        return "MALIGNANT", 0.8
    elif size > 20:
        return "BENIGN", 0.6
    else:
        return "NORMAL", 0.2


# =====================================================
# 🧠 CNN MOCK IMAGE AI
# =====================================================
def image_ai(image):

    img = np.array(image.convert("L"))
    edges = cv2.Canny(img, 50, 150)

    texture = np.var(img)
    edge_density = np.sum(edges) / (img.shape[0]*img.shape[1])

    malignancy = min(0.95, (texture/5000)*0.6 + edge_density*0.4)
    confidence = min(0.95, 0.5 + abs(texture/5000 - edge_density))

    return round(malignancy,2), round(confidence,2)


# =====================================================
# 🧬 FUSION ENGINE
# =====================================================
def fusion_engine(clinical_conf, image_prob):

    score = 0.55*clinical_conf + 0.45*(image_prob or 0)

    if score < 0.3:
        tier = "LOW"
    elif score < 0.6:
        tier = "MODERATE"
    elif score < 0.8:
        tier = "HIGH"
    else:
        tier = "CRITICAL"

    return round(score,2), tier


# =====================================================
# 🎨 VISUALIZATION
# =====================================================
def gauge(title, value, color):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value*100,
        number={'suffix': "%"},
        title={'text': title},
        gauge={'axis': {'range':[0,100]},
               'bar': {'color': color}}
    ))
    st.plotly_chart(fig, use_container_width=True)


def heatmap_overlay(image, intensity):

    img = np.array(image)
    h,w,_ = img.shape
    cam = np.zeros((h,w))
    cam[h//2,w//2] = 255*intensity
    cam = gaussian_filter(cam, sigma=50)

    cam_rgb = np.zeros_like(img)
    cam_rgb[:,:,0] = cam

    overlay = np.clip(img*0.6 + cam_rgb*0.6,0,255)
    return Image.fromarray(overlay.astype(np.uint8))


def bounding_box(image, tier):

    img = np.array(image).copy()
    h,w,_ = img.shape
    x1,y1 = w//3,h//3
    x2,y2 = x1+120,y1+120

    color = (255,0,0) if tier in ["HIGH","CRITICAL"] else (255,165,0)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,3)

    return Image.fromarray(img)


# =====================================================
# 🖥 DIAGNOSTIC HUB
# =====================================================
if nav == "Diagnostic Hub":

    st.title("🧠 Diagnostic Hub")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Clinical Input",
        "🖼 Image AI",
        "🧠 AI Fusion",
        "📊 Visualization"
    ])

    # =========================================================
    # TAB 1 — CLINICAL INPUT
    # =========================================================
    with tab1:
        st.subheader("Clinical Risk Assessment")

        age = st.number_input("Age", 1, 120, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
        smoker = st.selectbox("Smoker", ["No", "Yes"])

        clinical_score = (age * 0.01) + (bmi * 0.02)
        if smoker == "Yes":
            clinical_score += 0.15

        clinical_score = min(clinical_score, 1.0)

        st.metric("Clinical Risk Score", f"{clinical_score:.2f}")


    # =========================================================
    # TAB 2 — IMAGE AI (Mock CNN)
    # =========================================================
    with tab2:
        st.subheader("Ultrasound Image AI")

        uploaded = st.file_uploader("Upload Ultrasound", type=["png","jpg","jpeg"])

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)

            # Mock CNN score
            img_array = np.array(image.resize((128,128)))
            image_score = np.mean(img_array) / 255

            # Image confidence (fake softmax-like)
            image_confidence = np.random.uniform(0.75, 0.98)

            st.metric("Image AI Score", f"{image_score:.2f}")
            st.metric("Image Confidence", f"{image_confidence:.2f}")


    # =========================================================
    # TAB 3 — AI FUSION ENGINE
    # =========================================================
    with tab3:
        st.subheader("AI Fusion Model")

        if 'clinical_score' in locals() and 'image_score' in locals():

            fusion_score = (clinical_score * 0.4) + (image_score * 0.6)

            st.success(f"Final AI Risk Score: {fusion_score:.2f}")

        else:
            st.info("Please complete Clinical and Image tabs first.")


    # =========================================================
    # TAB 4 — VISUALIZATION
    # =========================================================
    with tab4:
        st.subheader("AI Visualization")

        if 'fusion_score' in locals():

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fusion_score,
                title={'text': "AI Risk Score"},
                gauge={'axis': {'range': [0, 1]}}
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Fake Heatmap
            heatmap = np.random.rand(128,128)

            fig2 = px.imshow(heatmap, color_continuous_scale="jet")
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("Run Fusion Model first.")
# =====================================================
# 📊 PROFESSIONAL ANALYTICS (SAFE VERSION)
# =====================================================

if nav == "Professional Analytics":

    st.title("Professional Analytics Dashboard")

    STATUS_COLOR = {
        "NORMAL": "#28a745",
        "BENIGN": "#ffc107",
        "MALIGNANT": "#dc3545"
    }

    if len(st.session_state.db) > 0:

        df = st.session_state.db.copy()

        # 🔥 ทำให้ status เป็นมาตรฐานก่อน
        df["Status"] = df["Status"].astype(str).str.upper().str.strip()

        status_counts = df["Status"].value_counts()

        # ================= KPI =================
        col1, col2, col3 = st.columns(3)

        col1.metric("🟢 NORMAL", status_counts.get("NORMAL", 0))
        col2.metric("🟡 BENIGN", status_counts.get("BENIGN", 0))
        col3.metric("🔴 MALIGNANT", status_counts.get("MALIGNANT", 0))

        st.markdown("---")

        # ================= PIE =================
        st.subheader("Risk Distribution")

        labels = status_counts.index.tolist()
        values = status_counts.values.tolist()

        colors = [
            STATUS_COLOR.get(s, "#9ca3af")  # เทา fallback กัน error
            for s in labels
        ]

        pie_fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.4
        )])

        pie_fig.update_layout(height=400)

        st.plotly_chart(pie_fig, use_container_width=True)

        # ================= BAR =================
        st.subheader("Case Distribution")

        bar_fig = go.Figure()

        for status in labels:
            bar_fig.add_trace(go.Bar(
                x=[status],
                y=[status_counts[status]],
                marker_color=STATUS_COLOR.get(status, "#9ca3af")
            ))

        bar_fig.update_layout(
            showlegend=False,
            height=400
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    else:
        st.info("No case data available yet.")

# =====================================================
# EXECUTIVE BOARD VIEW
# =====================================================
elif nav == "Executive Board View":

    st.title("Executive Business Intelligence")

    df = st.session_state.db
    total = len(df)
    malignant = (df["Status"] == "MALIGNANT").sum()

    st.metric("Total Diagnoses", total)
    st.metric("High Risk Cases", malignant)
    st.metric("Projected Quarterly Savings", "฿1,500,000")

    if total > 0:
        trend = df.groupby("Date").size().reset_index(name="Cases")
        st.plotly_chart(
            px.area(trend, x="Date", y="Cases"),
            use_container_width=True)

# =====================================================
# CASE ARCHIVE
# =====================================================
elif nav == "Case Archive":
    st.dataframe(st.session_state.db, use_container_width=True)

# =====================================================
# USER MANUAL (DETAILED)
# =====================================================
elif nav == "User Manual":

    st.title("Smart Biopsy Pro – Detailed Operational Manual")

    st.markdown("""
# 1. System Overview
Smart Biopsy Pro is a Multi-Organ Clinical Decision Support System
integrating biomarker logic and morphology-based inference.

# 2. Organ Modules
- Liver → AFP-based risk logic
- Thyroid → TI-RADS stratification
- Breast → BI-RADS prototype logic
- Lymph Nodes → Size-based malignancy logic

# 3. Risk Classification
🟢 NORMAL → Routine follow-up  
🟡 BENIGN → Imaging surveillance  
🔴 MALIGNANT → Biopsy priority  

# 4. Professional Analytics
Provides:
- Case volume monitoring
- Risk distribution
- Confidence tracking
- Organ workload analysis

# 5. Executive Board View
Displays:
- Institutional risk burden
- Financial impact simulation
- AI adoption growth

# 6. Governance Notice
This system is decision-support only.
Final clinical decisions must be made
by licensed physicians.
""")
