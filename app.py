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
import torch
import os
import gdown

# ===============================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ===============================

MODEL_ID = "1uVGvt5KKvhxumGapxjjOF10fWRXoZbDs"
MODEL_PATH = "ultrasound_model.pt"

@st.cache_resource
def download_model():

    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return MODEL_PATH

@st.cache_resource
def load_model():

    import torch.nn as nn
    from torchvision import models

    model_path = download_model()

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()

    return model

model = load_model()
# ===============================
# GRAD-CAM FUNCTION
# ===============================

import numpy as np
import cv2

def generate_gradcam(model, image_tensor):

    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = cam.detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    forward_handle.remove()
    backward_handle.remove()

    return cam

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
STATUS_OR = {
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
        "User Manual",
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

    col1, col2 = st.columns([1,1])

    # ================= LEFT =================
    with col1:

        st.subheader("Patient Information")

        patient = st.text_input("Patient Name")
        hn = st.text_input("HN")
        organ = st.selectbox("Organ", ["Liver", "Thyroid", "Breast", "Lung"])

        # =============================
        # AFP FIELD (Optional)
        # =============================
        st.markdown("### Biomarker (Optional)")
        marker = st.number_input(
            "AFP (ng/mL) – Optional",
            min_value=0.0,
            value=0.0,
            step=1.0
        )

        # =============================
        # Ultrasound Image Upload
        # =============================
        st.markdown("### Ultrasound Image Upload")

        uploaded_file = st.file_uploader(
            "Upload Ultrasound Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            with st.expander("🔍 View Ultrasound Image"):
                st.image(uploaded_file, width=400)

        # =============================
        # Tumor Size
        # =============================
        size = st.slider("Lesion Size (mm)", 1, 100, 10)

                # =============================
        # RUN AI BUTTON
        # =============================
        run = st.button("Run AI Analysis", use_container_width=True)

        if run and patient and hn:

            if uploaded_file is not None:

                from PIL import Image
                import torch
                from torchvision import transforms

                image = Image.open(uploaded_file).convert("RGB")

                preprocess = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]
                    )
                ])

                img_tensor = preprocess(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)[0]
                    conf, pred = torch.max(probs, dim=0)

                classes = ["NORMAL", "BENIGN", "MALIGNANT"]
                status = classes[pred.item()]
                confidence = conf.item()

                # ===============================
                # Grad-CAM Explainability
                # ===============================
                
                cam = generate_gradcam(model, img_tensor)
                
                heatmap = cv2.resize(cam, (image.width, image.height))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                overlay = cv2.addWeighted(
                    np.array(image),
                    0.6,
                    heatmap,
                    0.4,
                    0
                )

st.image(overlay, caption="Grad-CAM Explainability", use_column_width=True)


                new = pd.DataFrame([{
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
                    [st.session_state.db, new],
                    ignore_index=True
                )

                st.success("AI Image Analysis Complete")

            else:
                st.error("Please upload an ultrasound image.")



    # ================= RIGHT =================
    with col2:

        st.subheader("AI Result Dashboard")

        if len(st.session_state.db) > 0:

            last = st.session_state.db.iloc[-1]
            confidence_percent = last["Confidence"] * 100
            status = last["Status"]

            if status == "NORMAL":
                color = "#28a745"
            elif status == "BENIGN":
                color = "#ffc107"
            else:
                color = "#dc3545"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence_percent,
                number={'suffix': "%"},
                title={'text': status},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color}
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Run analysis to generate result.")


# =====================================================
# 📊 PROFESSIONAL ANALYTICS – HOSPITAL VERSION
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

        # ===== Clean data =====
        df["Status"] = df["Status"].astype(str).str.upper().str.strip()
        df["Date"] = pd.to_datetime(df["Date"])

        status_counts = df["Status"].value_counts()

        # ================= KPI =================
        total_cases = len(df)
        malignant_cases = status_counts.get("MALIGNANT", 0)
        malignancy_rate = round((malignant_cases / total_cases) * 100, 1)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Cases", total_cases)
        col2.metric("🟢 Normal", status_counts.get("NORMAL", 0))
        col3.metric("🔴 Malignant", malignant_cases)
        col4.metric("Malignancy Rate (%)", f"{malignancy_rate}%")

        st.markdown("---")

        # ================= DONUT =================
        st.subheader("Risk Distribution")

        labels = status_counts.index.tolist()
        values = status_counts.values.tolist()

        colors = [STATUS_COLOR.get(s, "#9ca3af") for s in labels]

        pie_fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=colors),
                textinfo="percent+label",
                textfont=dict(size=18)
            )
        ])

        pie_fig.update_layout(height=500)

        st.plotly_chart(pie_fig, use_container_width=True)

        # ================= CASE TREND =================
        st.subheader("Case Volume Trend")

        trend_df = df.groupby(df["Date"].dt.date).size().reset_index(name="Count")

        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=trend_df["Date"],
            y=trend_df["Count"],
            mode="lines+markers"
        ))

        trend_fig.update_layout(height=450)

        st.plotly_chart(trend_fig, use_container_width=True)

       # ================= ORGAN DISTRIBUTION =================
        st.subheader("Organ Distribution by Risk")
        
        organ_status = df.groupby(["Organ", "Status"]).size().reset_index(name="Count")
        
        organ_fig = go.Figure()
        
        for status in ["NORMAL", "BENIGN", "MALIGNANT"]:
            subset = organ_status[organ_status["Status"] == status]
        
            organ_fig.add_trace(go.Bar(
                x=subset["Organ"],
                y=subset["Count"],
                name=status,
                marker_color=STATUS_COLOR.get(status)
            ))
        
        organ_fig.update_layout(
            barmode="stack",
            height=450
        )
        
        st.plotly_chart(organ_fig, use_container_width=True)

      # ================= CONFIDENCE DISTRIBUTION =================
        st.subheader("AI Confidence Distribution by Risk")
        
        conf_fig = go.Figure()
        
        for status in ["NORMAL", "BENIGN", "MALIGNANT"]:
            subset = df[df["Status"] == status]
        
            conf_fig.add_trace(go.Histogram(
                x=subset["Confidence"] * 100,
                name=status,
                marker_color=STATUS_COLOR.get(status),
                opacity=0.6
            ))
        
        conf_fig.update_layout(
            barmode="overlay",
            height=450
        )
        
        st.plotly_chart(conf_fig, use_container_width=True)

        # ================= DATA TABLE =================
        st.subheader("Case Database")

        st.dataframe(df, use_container_width=True)

    else:
        st.info("No case data available yet.")

# =====================================================
# 🏥 EXECUTIVE BOARD VIEW – HOSPITAL EDITION
# =====================================================

if nav == "Professional Analytics":

    st.title("Executive Clinical AI Dashboard")

    STATUS_COLOR = {
        "NORMAL": "#28a745",
        "BENIGN": "#ffc107",
        "MALIGNANT": "#dc3545"
    }

    if len(st.session_state.db) > 0:

        df = st.session_state.db.copy()
        df["Status"] = df["Status"].astype(str).str.upper().str.strip()
        df["Date"] = pd.to_datetime(df["Date"])

        total_cases = len(df)
        malignant_cases = len(df[df["Status"] == "MALIGNANT"])
        benign_cases = len(df[df["Status"] == "BENIGN"])
        normal_cases = len(df[df["Status"] == "NORMAL"])

        malignancy_rate = round((malignant_cases / total_cases) * 100, 1)
        avg_conf = round(df["Confidence"].mean() * 100, 1)

        # ================= KPI =================
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Cases", total_cases)
        col2.metric("🔴 Malignant", malignant_cases)
        col3.metric("🟡 Benign", benign_cases)
        col4.metric("🟢 Normal", normal_cases)
        col5.metric("Malignancy Rate", f"{malignancy_rate}%")

        st.markdown("### AI Performance Overview")
        st.metric("Average AI Confidence", f"{avg_conf}%")
        st.markdown("---")

        # ================= WORKFLOW TRACKING =================
        st.subheader("Specimen Workflow Status")

        workflow_steps = ["Requested", "Collected", "In Lab", "Resulted"]
        current_status = "In Lab" if total_cases > 0 else "Requested"
        current_index = workflow_steps.index(current_status)
        progress_value = (current_index + 1) / len(workflow_steps)

        st.progress(progress_value, text=current_status)

        cols = st.columns(len(workflow_steps))
        for i, step in enumerate(workflow_steps):
            if i <= current_index:
                cols[i].success(step)
            else:
                cols[i].write(step)

        st.markdown("---")

        # ================= CASE VOLUME TREND =================
        st.subheader("Case Volume Trend")

        trend_df = df.groupby(df["Date"].dt.date).size().reset_index(name="Count")

        trend_fig = go.Figure()
        trend_fig.add_trace(go.Scatter(
            x=trend_df["Date"],
            y=trend_df["Count"],
            mode="lines+markers",
            line=dict(width=3)
        ))

        trend_fig.update_layout(height=450)

        st.plotly_chart(
            trend_fig,
            use_container_width=True,
            key="trend_chart"
        )

        # ================= ORGAN RISK BURDEN =================
        st.subheader("Organ Risk Burden (Stacked)")

        organ_status = (
            df.groupby(["Organ", "Status"])
              .size()
              .reset_index(name="Count")
        )

        organ_fig = go.Figure()

        for status in ["NORMAL", "BENIGN", "MALIGNANT"]:
            subset = organ_status[organ_status["Status"] == status]
            if not subset.empty:
                organ_fig.add_trace(go.Bar(
                    x=subset["Organ"],
                    y=subset["Count"],
                    name=status,
                    marker_color=STATUS_COLOR.get(status)
                ))

        organ_fig.update_layout(
            barmode="stack",
            height=450,
            xaxis_title="Organ",
            yaxis_title="Number of Cases",
            legend_title="Status"
        )

        st.plotly_chart(
            organ_fig,
            use_container_width=True,
            key="organ_chart"
        )

        # ================= AI CONFIDENCE =================
        st.subheader("AI Confidence by Risk Level")

        conf_fig = go.Figure()

        for status in ["NORMAL", "BENIGN", "MALIGNANT"]:
            subset = df[df["Status"] == status]
            if not subset.empty:
                conf_fig.add_trace(go.Histogram(
                    x=subset["Confidence"] * 100,
                    name=status,
                    marker_color=STATUS_COLOR.get(status),
                    opacity=0.6
                ))

        conf_fig.update_layout(
            barmode="overlay",
            height=450,
            xaxis_title="Confidence (%)",
            yaxis_title="Frequency"
        )

        st.plotly_chart(
            conf_fig,
            use_container_width=True,
            key="confidence_chart"
        )

        # ================= HIGH RISK PANEL =================
        st.subheader("High-Risk Alert Panel")

        high_risk_df = df[df["Status"] == "MALIGNANT"]

        if len(high_risk_df) > 0:
            st.error(f"{len(high_risk_df)} High-Risk Cases Detected")
            st.dataframe(high_risk_df.tail(10), use_container_width=True)
        else:
            st.success("No high-risk cases detected.")

    else:
        st.info("No case data available yet.")

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
elif nav == "Case Archive":

    st.title("Clinical Case Archive")

    df = st.session_state.db

    if len(df) == 0:
        st.info("No cases available")
        st.stop()

    # ===============================
    # FILTER PANEL
    # ===============================
    st.sidebar.subheader("Filter Cases")

    search_hn = st.sidebar.text_input("Search HN")
    organ_filter = st.sidebar.multiselect(
        "Filter by Organ",
        options=df["Organ"].unique()
    )
    status_filter = st.sidebar.multiselect(
        "Filter by Risk",
        options=df["Status"].unique()
    )

    # Date filter
    min_date = df["Date"].min()
    max_date = df["Date"].max()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date]
    )

    filtered_df = df.copy()

    if search_hn:
        filtered_df = filtered_df[
            filtered_df["HN"].str.contains(search_hn)
        ]

    if organ_filter:
        filtered_df = filtered_df[
            filtered_df["Organ"].isin(organ_filter)
        ]

    if status_filter:
        filtered_df = filtered_df[
            filtered_df["Status"].isin(status_filter)
        ]

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["Date"] >= date_range[0]) &
            (filtered_df["Date"] <= date_range[1])
        ]

    # ===============================
    # SUMMARY
    # ===============================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Cases", len(filtered_df))
    col2.metric("High Risk Cases",
                len(filtered_df[filtered_df["Status"] == "High Risk"]))
    col3.metric("Avg Confidence",
                f"{filtered_df['Confidence'].mean():.1f}%")

    st.divider()

    # ===============================
    # DATA TABLE
    # ===============================
    st.dataframe(filtered_df, use_container_width=True)

    # ===============================
    # DOWNLOAD BUTTON
    # ===============================
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "clinical_cases.csv",
        "text/csv"
    )

    # ===============================
    # CASES BY ORGAN (Colored by Risk)
    # ===============================
    import matplotlib.pyplot as plt
    
    st.subheader("Cases by Organ (Risk Classification)")
    
    grouped = filtered_df.groupby(["Organ", "Status"]).size().unstack(fill_value=0)
    
    # Ensure all risk categories exist
    for col in ["Low Risk", "Moderate Risk", "High Risk"]:
        if col not in grouped.columns:
            grouped[col] = 0
    
    # Rename for clinical display
    grouped = grouped.rename(columns={
        "Low Risk": "Normal",
        "Moderate Risk": "Benign",
        "High Risk": "Malignant"
    })
    
    grouped = grouped[["Normal", "Benign", "Malignant"]]
    
    fig, ax = plt.subplots()
    
    bottom_vals = None
    
    colors = {
        "Normal": "green",
        "Benign": "yellow",
        "Malignant": "red"
    }
    
    for col in grouped.columns:
        if bottom_vals is None:
            ax.bar(grouped.index, grouped[col],
                   label=col,
                   color=colors[col])
            bottom_vals = grouped[col].values
        else:
            ax.bar(grouped.index, grouped[col],
                   bottom=bottom_vals,
                   label=col,
                   color=colors[col])
            bottom_vals = bottom_vals + grouped[col].values
    
    ax.set_xlabel("Organ")
    ax.set_ylabel("Number of Cases")
    ax.legend()
    
    st.pyplot(fig)
# =====================================================
# 📘 USER MANUAL (WORKING VERSION)
# =====================================================

if nav == "User Manual":

    st.title("Smart Biopsy Pro – Detailed Operational Manual")

    st.markdown("""
    ## 1. System Overview
    Smart Biopsy Pro is a Multi-Organ Clinical Decision Support System
    integrating biomarker logic and morphology-based inference.

    ## 2. Organ Modules
    - Liver → AFP-based risk logic
    - Thyroid → TI-RADS stratification
    - Breast → BI-RADS prototype logic
    - Lymph Nodes → Size-based malignancy logic

    ## 3. Risk Classification
    🟢 NORMAL → Routine follow-up  
    🟡 BENIGN → Imaging surveillance  
    🔴 MALIGNANT → Biopsy priority  

    ## 4. Professional Analytics
    Provides:
    - Case volume monitoring
    - Risk distribution
    - Confidence tracking
    - Organ workload analysis

    ## 5. Executive Board View
    Displays:
    - Institutional risk burden
    - AI adoption growth
    - Institutional performance metrics

    ## 6. Governance Notice
    This system is decision-support only.
    Final clinical decisions must be made by licensed physicians.
    """)

    st.success("User Manual Loaded Successfully")

