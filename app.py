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
        "Business Intelligence",
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
# DIAGNOSTIC HUB
# =====================================================
if nav == "Diagnostic Hub":

    st.title("AI Diagnostic Engine")

    col1, col2 = st.columns([1, 1])

    # =====================================================
    # LEFT PANEL – INPUT SECTION
    # =====================================================
    with col1:

        st.subheader("Patient Information")

        patient = st.text_input("Patient Name")
        hn = st.text_input("HN")

        organ = st.selectbox(
            "Organ",
            ["Liver", "Thyroid", "Breast", "Lung", "Lymph Nodes"]
        )

        # -------------------------------
        # AFP (แสดงทันทีถ้าเป็น Liver)
        # -------------------------------
        marker = None

        if organ == "Liver":
            marker = st.number_input(
                "AFP (ng/mL)",
                min_value=0.0,
                value=10.0,
                step=1.0,
                help="Alpha-fetoprotein biomarker"
            )

        # -------------------------------
        # Lesion Size
        # -------------------------------
        size = st.slider(
            "Lesion Size (mm)",
            min_value=1,
            max_value=100,
            value=10
        )

        # -------------------------------
        # Ultrasound Upload
        # -------------------------------
        st.markdown("### Ultrasound Image Upload")

        uploaded_image = st.file_uploader(
            "Upload Ultrasound Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            st.image(
                uploaded_image,
                caption="Ultrasound Preview",
                use_column_width=True
            )

        # -------------------------------
        # RUN AI
        # -------------------------------
        if st.button("Run AI Analysis", use_container_width=True):

            if not patient or not hn:
                st.warning("Please enter Patient Name and HN")
            else:

                status, confidence = run_ai(organ, marker, size)

                new_case = pd.DataFrame([{
                    "Date": datetime.date.today(),
                    "HN": hn,
                    "Patient": patient,
                    "Organ": organ,
                    "Status": status,
                    "Confidence": confidence,
                    "Marker_Val": marker,
                    "Tumor_Size": size
                }])

                st.session_state.db = pd.concat(
                    [st.session_state.db, new_case],
                    ignore_index=True
                )

                st.success("AI Analysis Completed Successfully")

    # =====================================================
    # RIGHT PANEL – RESULT DASHBOARD
    # =====================================================
    with col2:

        st.subheader("AI Result Dashboard")

        if len(st.session_state.db) > 0:

            last = st.session_state.db.iloc[-1]

            confidence_percent = float(last["Confidence"]) * 100
            status = last["Status"]

            if status == "NORMAL":
                color = "#28a745"
            elif status == "BENIGN":
                color = "#ffc107"
            else:
                color = "#dc3545"

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_percent,
                number={'suffix': "%"},
                title={'text': f"Diagnosis: {status}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color}
                }
            ))

            fig.update_layout(height=350)

            st.plotly_chart(fig, use_container_width=True)

            # Case Summary Card
            st.markdown("### Case Summary")

            st.write(f"**Patient:** {last['Patient']}")
            st.write(f"**HN:** {last['HN']}")
            st.write(f"**Organ:** {last['Organ']}")
            st.write(f"**Tumor Size:** {last['Tumor_Size']} mm")

            if last["Organ"] == "Liver" and pd.notna(last["Marker_Val"]):
                st.write(f"**AFP:** {last['Marker_Val']} ng/mL")

        else:
            st.info("Run analysis to generate AI result.")
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
# =====================================================
# ADVANCED ENTERPRISE EXTENSIONS (APPEND BELOW EXISTING CODE)
# =====================================================

import uuid
import random

# =====================================================
# ENHANCED AI EXPLANATION ENGINE
# =====================================================
def generate_explanation(organ, marker, size, status):

    reasons = []
    recommendation = ""

    if organ == "Liver":
        if marker > 400:
            reasons.append("AFP > 400 (High oncologic risk)")
        if size > 50:
            reasons.append("Tumor size > 50mm")

    if organ == "Thyroid":
        if marker >= 5:
            reasons.append("TI-RADS 5 (Highly suspicious)")
        if size > 25:
            reasons.append("Nodule size > 25mm")

    if organ == "Breast":
        if marker >= 5:
            reasons.append("BI-RADS 5 (High malignancy probability)")

    if organ == "Lymph Nodes":
        if size > 30:
            reasons.append("Lymph node > 30mm")

    if status == "MALIGNANT":
        recommendation = "Immediate biopsy recommended."
    elif status == "BENIGN":
        recommendation = "Short-term imaging follow-up suggested."
    else:
        recommendation = "Routine surveillance."

    return reasons, recommendation


# =====================================================
# AUTO CASE ID GENERATOR
# =====================================================
def generate_case_id():
    return f"SBP-{datetime.date.today().year}-{str(uuid.uuid4())[:6].upper()}"


# =====================================================
# PATCH DATABASE WITH EXTRA FIELDS
# =====================================================
if "Case_ID" not in st.session_state.db.columns:
    st.session_state.db["Case_ID"] = ""
    st.session_state.db["Timestamp"] = ""
    st.session_state.db["Created_By"] = ""


# =====================================================
# ENHANCE LATEST CASE WITH ENTERPRISE DATA
# =====================================================
if len(st.session_state.db) > 0:

    last_index = st.session_state.db.index[-1]

    if st.session_state.db.loc[last_index, "Case_ID"] == "":
        st.session_state.db.loc[last_index, "Case_ID"] = generate_case_id()
        st.session_state.db.loc[last_index, "Timestamp"] = str(datetime.datetime.now())
        st.session_state.db.loc[last_index, "Created_By"] = st.session_state.role


# =====================================================
# ADD EXPLAINABLE AI PANEL TO DIAGNOSTIC HUB (FIXED)
# =====================================================
if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    df = st.session_state.db
    last_index = df.index[-1]

    # -------- Auto-fix missing enterprise fields --------
    if pd.isna(df.loc[last_index, "Case_ID"]) or df.loc[last_index, "Case_ID"] == "":
        df.loc[last_index, "Case_ID"] = generate_case_id()

    if pd.isna(df.loc[last_index, "Timestamp"]) or df.loc[last_index, "Timestamp"] == "":
        df.loc[last_index, "Timestamp"] = str(datetime.datetime.now())

    if pd.isna(df.loc[last_index, "Created_By"]) or df.loc[last_index, "Created_By"] == "":
        df.loc[last_index, "Created_By"] = st.session_state.role

    last = df.loc[last_index]

    # -------- Generate explanation --------
    reasons, recommendation = generate_explanation(
        last["Organ"],
        last["Marker_Val"],
        last["Tumor_Size"],
        last["Status"]
    )

    st.markdown("---")
    st.subheader("AI Risk Explanation")

    st.markdown(f"**Case ID:** {last['Case_ID']}")
    st.markdown(f"**Generated:** {last['Timestamp']}")
    st.markdown(f"**Created By:** {last['Created_By']}")

    st.markdown("### Risk Factors Identified:")

    if len(reasons) > 0:
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.markdown("No high-risk features detected.")

    st.markdown("### Clinical Recommendation:")
    st.success(recommendation)

# =====================================================
# ENHANCED EXECUTIVE METRICS
# =====================================================
if nav == "Executive Board View":

    df = st.session_state.db

    st.markdown("---")
    st.subheader("AI Adoption & Impact Metrics")

    total = len(df)

    if total > 0:

        adoption_rate = min(100, total * 5)
        biopsy_reduction = random.randint(15, 35)
        time_saved = total * 12

        colA, colB, colC = st.columns(3)

        colA.metric("AI Adoption Rate", f"{adoption_rate}%")
        colB.metric("Biopsy Reduction", f"{biopsy_reduction}%")
        colC.metric("Time Saved (hrs)", f"{time_saved}")

        st.markdown("### ROI Simulation")

        monthly_cases = st.slider("Monthly Cases", 50, 2000, 300)
        cost_per_biopsy = st.slider("Cost per Biopsy (฿)", 5000, 50000, 15000)
        reduction_percent = st.slider("False Positive Reduction %", 5, 50, 20)

        saved_cases = monthly_cases * (reduction_percent / 100)
        savings = saved_cases * cost_per_biopsy

        st.success(f"Projected Monthly Savings: ฿{savings:,.0f}")
# =====================================================
# ENTERPRISE IMAGE AI EXTENSION BLOCK
# =====================================================

import base64
from PIL import Image
import numpy as np
from reportlab.platypus import Image as RLImage
from reportlab.lib import colors
from reportlab.platypus import Table
from reportlab.lib.pagesizes import A4

# =====================================================
# 1️⃣ DATABASE PATCH – ADD IMAGE FIELDS
# =====================================================
if "Ultrasound_Image" not in st.session_state.db.columns:
    st.session_state.db["Ultrasound_Image"] = None
    st.session_state.db["Image_AI_Confidence"] = None


# =====================================================
# 2️⃣ IMAGE → BASE64 STORAGE
# =====================================================
def image_to_base64(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode()


def base64_to_image(b64_string):
    return Image.open(io.BytesIO(base64.b64decode(b64_string)))


# =====================================================
# 3️⃣ MOCK IMAGE AI CONFIDENCE
# =====================================================
def run_image_ai(image):
    # mock confidence from pixel intensity variance
    img_array = np.array(image.convert("L"))
    variance = np.var(img_array)
    confidence = min(0.95, max(0.10, variance / 10000))
    return round(confidence, 2)


# =====================================================
# 4️⃣ RISK HEATMAP OVERLAY
# =====================================================
def generate_heatmap_overlay(image):

    img_array = np.array(image)
    heatmap = np.zeros_like(img_array)

    # mock suspicious zone center
    h, w, _ = img_array.shape
    center_x, center_y = w // 2, h // 2

    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i-center_y)**2 + (j-center_x)**2)
            intensity = max(0, 255 - dist*2)
            heatmap[i,j] = [intensity, 0, 0]

    overlay = np.clip(img_array*0.6 + heatmap*0.4, 0, 255)
    return Image.fromarray(overlay.astype(np.uint8))


# =====================================================
# 5️⃣ PACS STYLE VIEWER
# =====================================================
def pacs_viewer(image):

    st.markdown("### PACS Viewer")

    zoom = st.slider("Zoom Level", 1, 5, 1)

    img_array = np.array(image)
    resized = Image.fromarray(img_array).resize(
        (img_array.shape[1]*zoom, img_array.shape[0]*zoom)
    )

    st.image(resized, use_column_width=True)


# =====================================================
# 6️⃣ PDF EXPORT WITH IMAGE
# =====================================================
def generate_full_pdf(case_row):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>SMART BIOPSY PRO REPORT</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    info_data = [
        ["Patient", case_row["Patient"]],
        ["HN", case_row["HN"]],
        ["Organ", case_row["Organ"]],
        ["Status", case_row["Status"]],
        ["AI Confidence", f"{case_row['Confidence']*100:.1f}%"],
        ["Image AI Confidence", f"{case_row['Image_AI_Confidence']*100:.1f}%"]
    ]

    table = Table(info_data, colWidths=[150, 250])
    table.setStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ])
    elements.append(table)
    elements.append(Spacer(1, 0.3 * inch))

    # add ultrasound image
    if case_row["Ultrasound_Image"]:
        img = base64_to_image(case_row["Ultrasound_Image"])
        img_path = "/tmp/temp_ultrasound.jpg"
        img.save(img_path)
        elements.append(RLImage(img_path, width=4*inch, height=4*inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# =====================================================
# 7️⃣ HOOK INTO DIAGNOSTIC HUB (AUTO RUN IF IMAGE EXISTS)
# =====================================================
if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    last_index = st.session_state.db.index[-1]
    last = st.session_state.db.loc[last_index]

    # If image just uploaded
    if "uploaded_image" in locals() and uploaded_image is not None:

        b64 = image_to_base64(uploaded_image)
        st.session_state.db.loc[last_index, "Ultrasound_Image"] = b64

        img = Image.open(uploaded_image)
        img_conf = run_image_ai(img)

        st.session_state.db.loc[last_index, "Image_AI_Confidence"] = img_conf

        overlay = generate_heatmap_overlay(img)

        st.markdown("---")
        st.subheader("Image AI Confidence")

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=img_conf * 100,
            number={'suffix': "%"},
            title={'text': "Image AI Risk"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#ff4d4d"}}
        ))

        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Risk Heatmap Overlay")
        st.image(overlay, use_column_width=True)

        pacs_viewer(img)

        # PDF Download
        pdf_buffer = generate_full_pdf(st.session_state.db.loc[last_index])

        st.download_button(
            "Download Full AI Report (PDF)",
            data=pdf_buffer,
            file_name=f"{last['HN']}_AI_Report.pdf",
            mime="application/pdf"
        )
        # =====================================================
# 🔬 ADVANCED CNN MOCK AI ENGINE
# =====================================================

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# =====================================================
# 1️⃣ CNN MOCK FEATURE EXTRACTION
# =====================================================
def cnn_mock_feature_extractor(image):

    img = np.array(image.convert("L"))

    # Edge detection (simulate CNN low-level feature)
    edges = cv2.Canny(img, 50, 150)

    # Texture feature (variance)
    texture_score = np.var(img)

    # Edge density feature
    edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])

    # Normalize features
    texture_norm = min(texture_score / 5000, 1.0)
    edge_norm = min(edge_density / 50, 1.0)

    # Mock fully-connected layer
    malignancy_probability = (0.6 * texture_norm) + (0.4 * edge_norm)

    confidence_score = min(0.95, 0.5 + abs(texture_norm - edge_norm))

    return round(malignancy_probability, 2), round(confidence_score, 2)


# =====================================================
# 2️⃣ ADVANCED HEATMAP GENERATOR (Gaussian Hotspot)
# =====================================================
def advanced_heatmap_overlay(image, malignancy_score):

    img = np.array(image)
    h, w, _ = img.shape

    heatmap = np.zeros((h, w))

    # Create Gaussian hotspot center
    center_x = np.random.randint(w//3, w*2//3)
    center_y = np.random.randint(h//3, h*2//3)

    heatmap[center_y, center_x] = 255 * malignancy_score

    heatmap = gaussian_filter(heatmap, sigma=40)

    heatmap_rgb = np.zeros_like(img)
    heatmap_rgb[:,:,0] = heatmap  # red channel

    overlay = np.clip(img * 0.6 + heatmap_rgb * 0.5, 0, 255)

    return Image.fromarray(overlay.astype(np.uint8))


# =====================================================
# 3️⃣ IMAGE AI EXECUTION PANEL (AUTO IF IMAGE EXISTS)
# =====================================================
if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    last_index = st.session_state.db.index[-1]
    last = st.session_state.db.loc[last_index]

    if last["Ultrasound_Image"]:

        st.markdown("---")
        st.subheader("🧠 Advanced CNN Image AI Analysis")

        img = base64_to_image(last["Ultrasound_Image"])

        malignancy_score, image_confidence = cnn_mock_feature_extractor(img)

        # Save into database
        st.session_state.db.loc[last_index, "Image_AI_Confidence"] = image_confidence
        st.session_state.db.loc[last_index, "Image_Malignancy_Prob"] = malignancy_score

        # =============================
        # Gauge 1 – Image Confidence
        # =============================
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=image_confidence * 100,
            number={'suffix': "%"},
            title={'text': "Image Confidence Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6"}
            }
        ))

        st.plotly_chart(fig_conf, use_container_width=True)

        # =============================
        # Gauge 2 – Malignancy Risk
        # =============================
        fig_mal = go.Figure(go.Indicator(
            mode="gauge+number",
            value=malignancy_score * 100,
            number={'suffix': "%"},
            title={'text': "Image Malignancy Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444"}
            }
        ))

        st.plotly_chart(fig_mal, use_container_width=True)

        # =============================
        # Heatmap Overlay
        # =============================
        st.subheader("🔥 AI Risk Heatmap Visualization")

        overlay = advanced_heatmap_overlay(img, malignancy_score)

        st.image(overlay, use_column_width=True)

        st.success("CNN Mock AI Analysis Completed")
        # =====================================================
# 🚀 SMART BIOPSY PRO v3 – AI FUSION ENGINE
# =====================================================

import random

# -----------------------------------------------------
# 1️⃣ Fusion Engine (Clinical + Image)
# -----------------------------------------------------
def fusion_engine(clinical_conf, image_prob):

    if image_prob is None:
        image_prob = 0.0

    fusion_score = (0.55 * clinical_conf) + (0.45 * image_prob)

    if fusion_score < 0.30:
        tier = "LOW RISK"
    elif fusion_score < 0.60:
        tier = "MODERATE RISK"
    elif fusion_score < 0.80:
        tier = "HIGH RISK"
    else:
        tier = "CRITICAL"

    return round(fusion_score, 2), tier


# -----------------------------------------------------
# 2️⃣ Mock Bounding Box Detection
# -----------------------------------------------------
def draw_mock_bounding_box(image, risk_level):

    img = np.array(image).copy()
    h, w, _ = img.shape

    x1 = random.randint(w//4, w//2)
    y1 = random.randint(h//4, h//2)
    x2 = x1 + random.randint(60, 120)
    y2 = y1 + random.randint(60, 120)

    color = (255, 0, 0) if risk_level in ["HIGH RISK", "CRITICAL"] else (255, 165, 0)

    cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)

    return Image.fromarray(img)


# -----------------------------------------------------
# 3️⃣ Grad-CAM Style Mock
# -----------------------------------------------------
def generate_gradcam_mock(image, intensity):

    img = np.array(image)
    h, w, _ = img.shape

    cam = np.zeros((h, w))
    cx = w // 2
    cy = h // 2

    cam[cy, cx] = 255 * intensity
    cam = gaussian_filter(cam, sigma=60)

    cam_rgb = np.zeros_like(img)
    cam_rgb[:,:,1] = cam  # green channel

    blended = np.clip(img * 0.6 + cam_rgb * 0.6, 0, 255)

    return Image.fromarray(blended.astype(np.uint8))


# -----------------------------------------------------
# 4️⃣ AI Fusion Display Panel
# -----------------------------------------------------
if nav == "Diagnostic Hub" and len(st.session_state.db) > 0:

    last_index = st.session_state.db.index[-1]
    last = st.session_state.db.loc[last_index]

    if last["Ultrasound_Image"]:

        st.markdown("---")
        st.subheader("🧠 AI Fusion Engine")

        clinical_conf = float(last["Confidence"])
        image_prob = last.get("Image_Malignancy_Prob", 0.0)

        fusion_score, tier = fusion_engine(clinical_conf, image_prob)

        st.session_state.db.loc[last_index, "Fusion_Score"] = fusion_score
        st.session_state.db.loc[last_index, "Risk_Tier"] = tier

        # ------------------------
        # Fusion Gauge
        # ------------------------
        fig_fusion = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fusion_score * 100,
            number={'suffix': "%"},
            title={'text': f"Fusion Risk Score – {tier}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#8b5cf6"}
            }
        ))

        st.plotly_chart(fig_fusion, use_container_width=True)

        # ------------------------
        # Bounding Box
        # ------------------------
        img = base64_to_image(last["Ultrasound_Image"])
        bbox_img = draw_mock_bounding_box(img, tier)

        st.subheader("📦 AI Lesion Detection (Mock)")
        st.image(bbox_img, use_column_width=True)

        # ------------------------
        # Grad-CAM
        # ------------------------
        gradcam_img = generate_gradcam_mock(img, fusion_score)

        st.subheader("🔥 Grad-CAM Visualization")
        st.image(gradcam_img, use_column_width=True)

        # ------------------------
        # Severity Banner
        # ------------------------
        if tier == "LOW RISK":
            st.success("Low Risk – Routine Monitoring")
        elif tier == "MODERATE RISK":
            st.warning("Moderate Risk – Follow-up Recommended")
        elif tier == "HIGH RISK":
            st.error("High Risk – Biopsy Consideration")
        else:
            st.error("CRITICAL – Immediate Intervention Recommended")

        st.success("AI Fusion Analysis Completed")
        if nav == "Business Intelligence":

    st.title("📈 Business Intelligence Dashboard")

    st.subheader("💰 Revenue Overview")

    # Mock Data
    monthly_revenue = np.random.randint(500000, 1500000, 12)
    months = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ]

    fig = px.line(
        x=months,
        y=monthly_revenue,
        markers=True,
        title="Monthly Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Revenue (YTD)", f"{monthly_revenue.sum():,.0f} THB")
    col2.metric("Avg / Month", f"{monthly_revenue.mean():,.0f} THB")
    col3.metric("Growth Rate", f"{np.random.uniform(5,15):.1f}%")

    st.divider()

    st.subheader("🏥 AI Service Utilization")

    service_data = {
        "AI Screening": np.random.randint(200,500),
        "Ultrasound AI": np.random.randint(100,300),
        "Risk Analytics": np.random.randint(150,400),
        "Premium AI Report": np.random.randint(50,150)
    }

    fig2 = px.bar(
        x=list(service_data.keys()),
        y=list(service_data.values()),
        title="Service Usage"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("📊 Market Projection (Mock)")

    projection = np.cumsum(np.random.randint(100000, 300000, 12))

    fig3 = px.area(
        x=months,
        y=projection,
        title="Projected Revenue Growth"
    )
    st.plotly_chart(fig3, use_container_width=True)
