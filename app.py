import json
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support,
)

st.set_page_config(page_title="AnemiaSense Dashboard", layout="wide", page_icon="\U0001FA78")

# =========================================================
# THEME: animated gradient background + custom styled cards
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0f3d3e, #1b1b3a);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Floating "blood cell" particles - live, multi-color, drifting background layer */
.cell-particle {
    position: fixed;
    border-radius: 50%;
    filter: blur(1px);
    opacity: 0.35;
    z-index: 0;
    pointer-events: none;
    animation: floatDrift linear infinite;
}
@keyframes floatDrift {
    0%   { transform: translateY(0) translateX(0) scale(1); }
    25%  { transform: translateY(-40px) translateX(20px) scale(1.05); }
    50%  { transform: translateY(-10px) translateX(-30px) scale(0.95); }
    75%  { transform: translateY(30px) translateX(15px) scale(1.03); }
    100% { transform: translateY(0) translateX(0) scale(1); }
}
.block-container {
    position: relative; z-index: 1; max-width: 1500px;
    padding-top: 1.5rem; padding-bottom: 3rem;
}

.hero-card {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, rgba(12,30,48,0.96), rgba(20,68,78,0.92));
    border: 1px solid rgba(94,234,212,0.22); border-radius: 24px;
    padding: 28px 30px; margin-bottom: 22px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.28);
}
.hero-card::after {
    content: ''; position: absolute; width: 360px; height: 360px;
    right: -110px; top: -190px; border-radius: 50%;
    background: radial-gradient(circle, rgba(38,224,201,0.28), transparent 68%);
}
.hero-content { position: relative; z-index: 1; }
.hero-eyebrow { color: #5eead4; font-size: 11px; font-weight: 700; letter-spacing: 1.8px; }
.hero-title { color: #f8fafc; font-size: 38px; font-weight: 700; margin: 6px 0; }
.hero-title span { color: #7dd3fc; }
.hero-subtitle { color: #cbd5e1; max-width: 760px; line-height: 1.6; font-size: 14px; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.status-chip {
    display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px;
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.10);
    color: #dbeafe; font-size: 11px; font-weight: 600;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5, #ff6a88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

.subtitle {
    color: #b8c6db;
    font-size: 15px;
    margin-top: 0px;
    margin-bottom: 20px;
}

.kpi-card {
    border-radius: 14px;
    padding: 18px 18px; min-height: 126px;
    text-align: left;
    color: white;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
}
.kpi-icon { font-size: 20px; margin-bottom: 10px; }
.kpi-label {
    font-size: 12px;
    opacity: 0.9;
    margin-top: 4px;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.kpi-tip { color: rgba(255,255,255,0.72); font-size: 11px; margin-top: 7px; line-height: 1.35; }

.kpi-card-muted { filter: grayscale(0.55) brightness(0.85); border: 1px dashed rgba(255,255,255,0.35); }
.kpi-badge {
    display: inline-block;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.3px;
    padding: 3px 9px;
    border-radius: 999px;
    margin-top: 8px;
    background: rgba(0,0,0,0.28);
}
.kpi-badge.ok { color: #baffd6; }
.kpi-badge.warn { color: #ffe4ae; }
.kpi-margin { font-size: 13px; font-weight: 500; opacity: 0.85; margin-left: 6px; }

.section-card {
    background: rgba(8,18,30,0.62);
    border: 1px solid rgba(148,163,184,0.13);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 20px;
    backdrop-filter: blur(6px);
}

.metric-guide {
    background: linear-gradient(145deg, rgba(17,40,58,0.96), rgba(21,56,62,0.92));
    border: 1px solid rgba(94,234,212,0.20); border-radius: 16px;
    padding: 18px; margin: 8px 0 12px 0; min-height: 208px;
}
.metric-guide-top { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }
.metric-guide-icon { font-size: 30px; }
.metric-guide-name { color: #f8fafc; font-size: 21px; font-weight: 700; }
.metric-guide-value { color: #5eead4; font-size: 28px; font-weight: 700; }
.metric-guide-copy { color: #cbd5e1; font-size: 13px; line-height: 1.55; margin-top: 10px; }
.formula-pill {
    display: inline-block; color: #bae6fd; background: rgba(56,189,248,0.10);
    border: 1px solid rgba(56,189,248,0.18); border-radius: 9px;
    padding: 7px 10px; margin-top: 12px; font-family: monospace; font-size: 12px;
}

div[data-testid="stMetric"] {
    background: rgba(15,23,42,0.52); border: 1px solid rgba(148,163,184,0.12);
    padding: 11px 13px; border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(15,23,42,0.55); padding: 6px; border-radius: 13px;
}
.stTabs [data-baseweb="tab"] { border-radius: 9px; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: rgba(38,224,201,0.14); }

.result-badge {
    display: inline-block;
    padding: 10px 22px;
    border-radius: 30px;
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.rec-list li {
    margin-bottom: 6px;
    color: #dcdcdc;
}

[data-testid="stFileUploader"] {
    border: 2px dashed rgba(255,255,255,0.25);
    border-radius: 14px;
    padding: 10px;
    background: rgba(255,255,255,0.03);
}

.info-card {
    border-radius: 16px;
    padding: 20px;
    color: white;
    height: 100%;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    transition: transform 0.2s ease;
}
.info-card:hover { transform: translateY(-4px); }
.info-icon { font-size: 32px; margin-bottom: 8px; }
.info-title { font-size: 16px; font-weight: 700; margin-bottom: 6px; }
.info-text { font-size: 13px; opacity: 0.92; line-height: 1.5; }

.story-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 18px;
    height: 100%;
}
.story-avatar {
    font-size: 40px;
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
    width: 64px; height: 64px;
    line-height: 64px;
    text-align: center;
    margin-bottom: 10px;
}
.story-name { font-weight: 700; font-size: 16px; margin-bottom: 2px; }
.story-tag {
    display: inline-block;
    background: rgba(38, 224, 201, 0.15);
    color: #26e0c9;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 10px;
}
.story-text { font-size: 13px; color: #dcdcdc; line-height: 1.6; }
.illustrative-note {
    font-size: 12px;
    color: #9aa5b1;
    font-style: italic;
    margin-top: 6px;
}

@media (max-width: 800px) {
    .hero-title { font-size: 29px; }
    .hero-card { padding: 22px 20px; }
    .kpi-card { min-height: 112px; }
}
</style>
""", unsafe_allow_html=True)

# Render the floating particle layer - varied colors/sizes/speeds for a "live" feel
import random as _bg_random
_bg_random.seed(7)
_particle_colors = ["#ff5c7a", "#ff9a5a", "#3ab0ff", "#26e0c9", "#a86bff", "#ff6ab3"]
_particles_html = ""
for i in range(14):
    size = _bg_random.randint(30, 90)
    top = _bg_random.randint(0, 95)
    left = _bg_random.randint(0, 95)
    color = _bg_random.choice(_particle_colors)
    duration = _bg_random.uniform(10, 22)
    delay = _bg_random.uniform(0, 8)
    _particles_html += f"""<div class="cell-particle" style="
        width:{size}px; height:{size}px; top:{top}%; left:{left}%;
        background:{color}; animation-duration:{duration:.1f}s; animation-delay:{delay:.1f}s;
    "></div>"""
st.markdown(_particles_html, unsafe_allow_html=True)

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dashboard_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load saved artifacts ----------
@st.cache_resource
def load_everything():
    with open(f"{DATA_DIR}/metrics.json") as f:
        metrics = json.load(f)
    with open(f"{DATA_DIR}/class_avg_color.json") as f:
        class_avg_color = {k: np.array(v) for k, v in json.load(f).items()}

    class_names = metrics["class_names"]
    model = models.resnet34(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(class_names))
    )
    model.load_state_dict(torch.load(f"{DATA_DIR}/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # CBC regression model is optional - only load it if it was trained and saved
    reg_model, reg_meta = None, None
    reg_model_path = f"{DATA_DIR}/best_reg_model.pth"
    reg_meta_path = f"{DATA_DIR}/reg_meta.json"
    if os.path.exists(reg_model_path) and os.path.exists(reg_meta_path):
        with open(reg_meta_path) as f:
            reg_meta = json.load(f)
        reg_model = models.resnet34(weights=None)
        reg_model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(reg_model.fc.in_features, len(reg_meta["target_fields"]))
        )
        reg_model.load_state_dict(torch.load(reg_model_path, map_location=device))
        reg_model.to(device)
        reg_model.eval()

    return metrics, class_avg_color, model, class_names, reg_model, reg_meta

metrics, class_avg_color, model, class_names, reg_model, reg_meta = load_everything()

all_labels = np.array(metrics["all_labels"])
all_preds = np.array(metrics["all_preds"])
all_probs = np.array(metrics["all_probs"])
val_labels_for_threshold = np.array(metrics.get("val_labels_for_threshold", []))
val_probs_for_threshold = np.array(metrics.get("val_probs_for_threshold", []))
best_threshold = metrics["best_threshold"]
temperature = metrics.get("temperature", 1.0)
train_losses = metrics["train_losses"]
val_losses = metrics["val_losses"]
val_accuracies = metrics["val_accuracies"]

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_with_tta(model, images):
    # Match notebook validation/test inference exactly: original, horizontal,
    # vertical, and 180-degree views, averaged as raw logits.
    views = (
        images,
        torch.flip(images, dims=[3]),
        torch.flip(images, dims=[2]),
        torch.flip(images, dims=[2, 3]),
    )
    avg_logits = torch.stack([model(view) for view in views], dim=0).mean(dim=0)
    return torch.softmax(avg_logits / temperature, dim=1)

def metrics_at_threshold(labels, probs, threshold):
    """Educational validation-only threshold snapshot for the dashboard."""
    predictions = (probs[:, 1] >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float((predictions == labels).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

# ---------- Grad-CAM (sharper: percentile-thresholded to cut visual noise) ----------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()

gradcam = GradCAM(model, model.layer4)

def make_gradcam_overlay(pil_img, input_tensor, class_idx, keep_top_pct=25):
    cam = gradcam.generate(input_tensor, class_idx)
    cam_resized = cv2.resize(cam, pil_img.size)

    thresh = np.percentile(cam_resized, 100 - keep_top_pct)
    cam_sharp = np.where(cam_resized >= thresh, cam_resized, 0)
    cam_sharp = cv2.GaussianBlur(cam_sharp, (9, 9), 0)
    if cam_sharp.max() > 0:
        cam_sharp = cam_sharp / cam_sharp.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_sharp), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    base = np.array(pil_img).astype(np.uint8)
    overlay = cv2.addWeighted(base, 0.55, heatmap, 0.45, 0)
    hot_area_pct = float((cam_sharp > 0.3).mean() * 100)
    return overlay, hot_area_pct

def mean_cell_color(pil_img, dark_thresh=20):
    arr = np.array(pil_img.convert("RGB"))
    mask = arr.sum(axis=2) > dark_thresh * 3
    if mask.sum() == 0:
        return arr.reshape(-1, 3).mean(axis=0)
    return arr[mask].mean(axis=0)

RECOMMENDATIONS = {
    "Anemic_individuals": [
        "This is a model prediction from a research project, not a medical diagnosis.",
        "See a doctor for a confirmatory blood test (a full CBC panel) before drawing any conclusions.",
        "General info only: iron-rich foods (leafy greens, legumes, lean meat) and vitamin C (aids iron "
        "absorption) are commonly discussed around anemia - a doctor can advise what's right for you.",
        "Don't start iron supplements on your own without medical guidance - excess iron can itself be harmful.",
    ],
    "Healthy_individuals": [
        "This is a model prediction from a research project, not a medical diagnosis.",
        "No signs consistent with anemia were found in this image, but routine checkups are still worthwhile.",
        "If you have symptoms like fatigue, paleness, or shortness of breath despite this result, see a doctor "
        "anyway - a single image classifier is not a substitute for a blood test.",
    ],
}


# ================= TABS =================
tab1, tab2 = st.tabs(["\U0001F4CA Model Dashboard", "\U0001F4DA Learn About Anemia"])

with tab1:
    # ================= HEADER =================
    st.markdown(f"""
        <div class="hero-card">
            <div class="hero-content">
                <div class="hero-eyebrow">AI-ASSISTED RBC IMAGE ANALYSIS</div>
                <div class="hero-title">\U0001FA78 AnemiaSense <span>Model Dashboard</span></div>
                <div class="hero-subtitle">
                    Explore held-out model performance, understand every metric, and inspect a new
                    RGB-segmented blood-cell image. Educational research interface - not a diagnosis.
                </div>
                <div class="chip-row">
                    <span class="status-chip">\u2713 Held-out test set</span>
                    <span class="status-chip">{len(all_labels)} test images</span>
                    <span class="status-chip">{len(class_names)} balanced classes</span>
                    <span class="status-chip">Threshold {best_threshold:.2f}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    display_class_names = [name.replace("_individuals", "") for name in class_names]
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    overall_acc = (all_preds == all_labels).mean()
    auc_score = roc_auc_score(all_labels, all_probs[:, 1]) if len(class_names) == 2 else None

    # ================= KPI CARDS =================
    kpi_defs = [
        ("\U0001F3AF", "Test Accuracy", f"{overall_acc:.1%}", "linear-gradient(135deg,#0f9b8e,#0c6e64)",
         "Correct predictions out of all test images."),
        ("\U0001F50E", "Precision (macro)", f"{report_dict['macro avg']['precision']:.1%}", "linear-gradient(135deg,#1f77b4,#154a73)",
         "When the model predicts a class, how often it is right."),
        ("\U0001F4E1", "Recall (macro)", f"{report_dict['macro avg']['recall']:.1%}", "linear-gradient(135deg,#d97732,#8f431b)",
         "How many real cases of each class the model finds."),
        ("\u2696\ufe0f", "F1-score (macro)", f"{report_dict['macro avg']['f1-score']:.1%}", "linear-gradient(135deg,#8e44ad,#5b2c6f)",
         "The balance between precision and recall."),
        ("\U0001F4C8", "ROC-AUC", f"{auc_score:.3f}" if auc_score is not None else "-", "linear-gradient(135deg,#334155,#172033)",
         "Ranking quality across all possible thresholds."),
        ("\U0001F39A\ufe0f", "Decision Threshold", f"{best_threshold:.2f}", "linear-gradient(135deg,#b7354a,#712335)",
         "Probability cutoff selected on validation data."),
    ]
    for row_start in range(0, len(kpi_defs), 3):
        kpi_cols = st.columns(3)
        for col, (icon, label, value, grad, tip) in zip(kpi_cols, kpi_defs[row_start:row_start + 3]):
            col.markdown(f"""
                <div class="kpi-card" style="background:{grad};" title="{tip}">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-tip">{tip}</div>
                </div>
            """, unsafe_allow_html=True)
        st.write("")

    # ================= CHARTS =================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        epochs_list = list(range(1, len(train_losses) + 1))
        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(x=epochs_list, y=train_losses, mode="lines+markers",
                                        name="Train Loss", line=dict(color="#ff9a5a")))
        fig_train.add_trace(go.Scatter(x=epochs_list, y=val_losses, mode="lines+markers",
                                        name="Val Loss", line=dict(color="#3ab0ff")))
        fig_train.add_trace(go.Scatter(x=epochs_list, y=[a / 100 for a in val_accuracies], mode="lines+markers",
                                        name="Val Accuracy (scaled)", line=dict(color="#26e0c9", dash="dot"),
                                        yaxis="y2"))
        fig_train.update_layout(
            title="Training Loss & Validation Accuracy",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Epoch"), yaxis=dict(title="Loss"),
            yaxis2=dict(title="Val Accuracy", overlaying="y", side="right", range=[0, 1]),
            legend=dict(orientation="h", y=-0.25), height=360,
        )
        st.plotly_chart(fig_train, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        metrics_rows = []
        for cls, display_cls in zip(class_names, display_class_names):
            metrics_rows.append({"class": display_cls, "metric": "Precision", "value": report_dict[cls]["precision"]})
            metrics_rows.append({"class": display_cls, "metric": "Recall", "value": report_dict[cls]["recall"]})
            metrics_rows.append({"class": display_cls, "metric": "F1-score", "value": report_dict[cls]["f1-score"]})
        metrics_df = pd.DataFrame(metrics_rows)
        fig_prf = px.bar(
            metrics_df, x="class", y="value", color="metric", barmode="group", text_auto=".2%",
            template="plotly_dark", title="Precision / Recall / F1-score per class",
            color_discrete_map={"Precision": "#3ab0ff", "Recall": "#ff9a5a", "F1-score": "#26e0c9"},
        )
        fig_prf.update_layout(
            yaxis=dict(range=[0, 1], title="Score"), xaxis_title="", height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_prf, use_container_width=True)
        st.caption("Precision = few false alarms. Recall = few missed cases. F1 = balance of both. Higher bars = better.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= CONFUSION MATRIX + INTERACTIVE METRIC LAB =================
    cm_col, guide_col = st.columns([1.08, 0.92], gap="large")

    with cm_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### \U0001F9ED Confusion Matrix")
        cm_view = st.radio(
            "Confusion matrix view", ("Counts", "Row percentages"),
            horizontal=True, label_visibility="collapsed", key="cm_view",
        )
        cm_counts = confusion_matrix(
            all_labels, all_preds, labels=list(range(len(class_names)))
        )
        if cm_view == "Row percentages":
            row_totals = cm_counts.sum(axis=1, keepdims=True)
            cm_display = np.divide(
                cm_counts, row_totals, out=np.zeros_like(cm_counts, dtype=float), where=row_totals != 0
            )
            text_format = ".1%"
            color_title = "Row %"
            hover_template = "Actual: %{y}<br>Predicted: %{x}<br>Share: %{z:.1%}<extra></extra>"
        else:
            cm_display = cm_counts
            text_format = True
            color_title = "Images"
            hover_template = "Actual: %{y}<br>Predicted: %{x}<br>Images: %{z}<extra></extra>"

        fig_cm = px.imshow(
            cm_display, text_auto=text_format, x=display_class_names, y=display_class_names,
            color_continuous_scale="Teal", template="plotly_dark",
            labels=dict(x="Predicted", y="Actual", color=color_title),
            aspect="auto",
        )
        fig_cm.update_traces(hovertemplate=hover_template)
        fig_cm.update_layout(
            height=355, margin=dict(l=5, r=5, t=15, b=5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(thickness=12),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        correct_total = int(np.trace(cm_counts))
        error_total = int(cm_counts.sum() - correct_total)
        cm_stat1, cm_stat2 = st.columns(2)
        cm_stat1.metric("Correct", f"{correct_total} / {int(cm_counts.sum())}")
        cm_stat2.metric("Misclassified", error_total)
        st.caption(
            "Read across each row: the diagonal cells are correct predictions; off-diagonal cells are mistakes."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with guide_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### \U0001F9E0 Interactive Metric Guide")
        st.caption("Pick a metric to see what it means and how to read the current value.")

        metric_info = {
            "Accuracy": {
                "icon": "\U0001F3AF", "score": overall_acc, "display": f"{overall_acc:.1%}",
                "formula": "Correct predictions / All predictions",
                "definition": "The overall share of test images classified correctly.",
                "reading": "Best as a quick summary when the two classes are balanced.",
            },
            "Precision": {
                "icon": "\U0001F50E", "score": report_dict["macro avg"]["precision"],
                "display": f"{report_dict['macro avg']['precision']:.1%}",
                "formula": "TP / (TP + FP)",
                "definition": "When the model predicts a class, precision asks how often that prediction is right.",
                "reading": "Higher precision means fewer false alarms. This dashboard shows the macro average.",
            },
            "Recall": {
                "icon": "\U0001F4E1", "score": report_dict["macro avg"]["recall"],
                "display": f"{report_dict['macro avg']['recall']:.1%}",
                "formula": "TP / (TP + FN)",
                "definition": "Recall asks how many of the real cases the model successfully finds.",
                "reading": "Higher recall means fewer missed cases. This dashboard shows the macro average.",
            },
            "F1-score": {
                "icon": "\u2696\ufe0f", "score": report_dict["macro avg"]["f1-score"],
                "display": f"{report_dict['macro avg']['f1-score']:.1%}",
                "formula": "2 x (Precision x Recall) / (Precision + Recall)",
                "definition": "F1 combines precision and recall into one balanced score.",
                "reading": "Useful when both false alarms and missed cases matter.",
            },
            "ROC-AUC": {
                "icon": "\U0001F4C8", "score": auc_score if auc_score is not None else 0.0,
                "display": f"{auc_score:.3f}" if auc_score is not None else "-",
                "formula": "Ranking quality across every cutoff",
                "definition": "ROC-AUC measures whether higher probabilities usually go to the correct class.",
                "reading": "1.0 is perfect ranking; 0.5 is roughly random ranking.",
            },
            "Threshold": {
                "icon": "\U0001F39A\ufe0f", "score": best_threshold, "display": f"{best_threshold:.3f}",
                "formula": f"Predict {display_class_names[1]} when probability >= threshold",
                "definition": "The threshold converts a probability into the final class decision.",
                "reading": f"Lower values predict {display_class_names[1]} more often; higher values favor {display_class_names[0]}. It does not retrain the model.",
            },
        }
        selected_metric = st.selectbox("Metric", list(metric_info), key="metric_guide_choice")
        selected_info = metric_info[selected_metric]
        st.markdown(f"""
            <div class="metric-guide">
                <div class="metric-guide-top">
                    <div>
                        <div class="metric-guide-icon">{selected_info['icon']}</div>
                        <div class="metric-guide-name">{selected_metric}</div>
                    </div>
                    <div class="metric-guide-value">{selected_info['display']}</div>
                </div>
                <div class="metric-guide-copy">{selected_info['definition']}<br><br>{selected_info['reading']}</div>
                <div class="formula-pill">{selected_info['formula']}</div>
            </div>
        """, unsafe_allow_html=True)
        st.progress(
            min(max(float(selected_info["score"]), 0.0), 1.0),
            text=f"Current official value: {selected_info['display']}",
        )

        st.markdown("#### \U0001F39B\ufe0f Threshold playground")
        st.caption(
            "Validation-data demo only. Moving this slider never changes the official test result or saved threshold."
        )
        if len(val_labels_for_threshold) and len(val_probs_for_threshold):
            positive_class = display_class_names[1] if len(display_class_names) > 1 else "Class 1"
            demo_threshold = st.slider(
                f"{positive_class} probability cutoff", 0.05, 0.95,
                value=float(best_threshold), step=0.005, format="%.3f", key="threshold_playground",
            )
            official_snapshot = metrics_at_threshold(
                val_labels_for_threshold, val_probs_for_threshold, best_threshold
            )
            demo_snapshot = metrics_at_threshold(
                val_labels_for_threshold, val_probs_for_threshold, demo_threshold
            )
            demo_names = (("Accuracy", "accuracy"), ("Precision", "precision"),
                          ("Recall", "recall"), ("F1", "f1"))
            for start in range(0, len(demo_names), 2):
                demo_cols = st.columns(2)
                for col, (label, key) in zip(demo_cols, demo_names[start:start + 2]):
                    delta_pp = (demo_snapshot[key] - official_snapshot[key]) * 100
                    col.metric(label, f"{demo_snapshot[key]:.1%}", delta=f"{delta_pp:+.1f} pp")

            with st.expander("Show metric curves across thresholds"):
                curve_rows = []
                for threshold_value in np.linspace(0.05, 0.95, 91):
                    snapshot = metrics_at_threshold(
                        val_labels_for_threshold, val_probs_for_threshold, threshold_value
                    )
                    for metric_key, metric_label in (("accuracy", "Accuracy"), ("precision", "Precision"),
                                                     ("recall", "Recall"), ("f1", "F1")):
                        curve_rows.append({
                            "Threshold": threshold_value, "Metric": metric_label,
                            "Score": snapshot[metric_key],
                        })
                curve_df = pd.DataFrame(curve_rows)
                fig_curve = px.line(
                    curve_df, x="Threshold", y="Score", color="Metric",
                    template="plotly_dark", color_discrete_sequence=["#5eead4", "#60a5fa", "#fb923c", "#c084fc"],
                )
                fig_curve.add_vline(x=best_threshold, line_dash="dash", line_color="#f8fafc")
                fig_curve.add_vline(x=demo_threshold, line_dash="dot", line_color="#fb7185")
                fig_curve.update_layout(
                    height=280, yaxis=dict(range=[0, 1]), margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_curve, use_container_width=True)
                st.caption("Dashed = official threshold; dotted pink = slider threshold.")
        else:
            st.info("Run the calibration/threshold cell, then re-save dashboard artifacts to enable this playground.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= UPLOAD + PREDICT =================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### \U0001F4E4 Prediction Studio")
    st.caption("Upload one RGB_segmented-style image - color preserved, background removed - to inspect the model's result.")
    st.markdown("""
        <div class="chip-row">
            <span class="status-chip">1 &middot; Upload image</span>
            <span class="status-chip">2 &middot; Review prediction</span>
            <span class="status-chip">3 &middot; Inspect Grad-CAM</span>
        </div>
    """, unsafe_allow_html=True)
    st.write("")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_224 = img.resize((224, 224))
        input_tensor = transform_val(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = predict_with_tta(model, input_tensor)
        prob_class1 = probs[0][1].item()
        pred_idx = int(prob_class1 >= best_threshold)
        predicted_label = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

        overlay, hot_area_pct = make_gradcam_overlay(img_224, input_tensor, pred_idx)

        this_color = mean_cell_color(img)
        dist_to_anemic = float(np.linalg.norm(this_color - class_avg_color["Anemic_individuals"]))
        dist_to_healthy = float(np.linalg.norm(this_color - class_avg_color["Healthy_individuals"]))
        closer_to = "Anemic_individuals" if dist_to_anemic < dist_to_healthy else "Healthy_individuals"

        label_display = predicted_label.replace("_individuals", "")
        color_class_display = closer_to.replace("_individuals", "")

        r1, r2, r3 = st.columns([1, 1, 1.3])
        with r1:
            st.image(img_224, caption="Uploaded image")
        with r2:
            st.image(overlay, caption="Grad-CAM: what the model focused on")
        with r3:
            badge_bg = "linear-gradient(135deg,#c0392b,#7b1e14)" if predicted_label == "Anemic_individuals" else "linear-gradient(135deg,#0f9b8e,#0c6e64)"
            st.markdown(f'<div class="result-badge" style="background:{badge_bg};">{label_display}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.1%}")
            if abs(prob_class1 - best_threshold) < 0.05:
                st.warning("This prediction is close to the decision boundary - treat as uncertain.")

        st.subheader("Why the model called it this way")
        total_dist = dist_to_anemic + dist_to_healthy
        color_lean_pct = (1 - dist_to_anemic / total_dist) * 100 if total_dist > 0 else 50
        explanation = (
            f"Grad-CAM shows the model concentrated roughly **{hot_area_pct:.0f}%** of its strongest attention "
            f"on a focused region of the image (ideally the red blood cells themselves, not background) - "
            f"see the red/yellow zones in the heatmap above.\n\n"
            f"Separately, this image's average cell color (R={this_color[0]:.0f}, G={this_color[1]:.0f}, "
            f"B={this_color[2]:.0f}) sits **{dist_to_anemic:.1f}** units from the training set's typical "
            f"*Anemic* color profile and **{dist_to_healthy:.1f}** units from the typical *Healthy* profile - "
            f"placing it closer to the **{color_class_display}** group ({color_lean_pct:.0f}% lean toward Anemic "
            f"on this color scale). Paler, less saturated cells score closer to the Anemic average, since "
            f"pallor is the actual biological signal being picked up on.\n\n"
            f"Combined with a model confidence of **{confidence:.1%}**, both the visual attention (Grad-CAM) "
            f"and the color statistics point toward **{label_display}**."
        )
        st.markdown(explanation)

        # ---- Approximate CBC panel (only if the regression model was trained) ----
        if reg_model is not None and reg_meta is not None:
            st.subheader("\U0001F9EA Estimated CBC panel (approximate)")
            st.caption(
                "Estimated from this image by a separate model trained on the dataset's real CBC reports. "
                "These are NOT real lab values - always shown with the model's actual measured error margin. "
                "A real blood test is the only way to get true values."
            )
            with torch.no_grad():
                reg_out = reg_model(input_tensor)[0].cpu().numpy()

            target_fields = reg_meta["target_fields"]
            field_mean = reg_meta["field_mean"]
            field_std = reg_meta["field_std"]
            field_mae = reg_meta["field_mae"]
            field_r2 = reg_meta["field_r2"]

            # Same colorful card language as the KPI section up top - one
            # signature gradient + icon per CBC field, so this panel reads
            # as part of the same visual system instead of a flat gray afterthought.
            CBC_STYLE = {
                "Hemoglobin": ("\U0001FA78", "linear-gradient(135deg,#0f9b8e,#0c6e64)", "g/dL \u00b7 oxygen-carrying protein in red cells"),
                "RBC":        ("\U0001F534", "linear-gradient(135deg,#1f77b4,#154a73)", "million/\u00b5L \u00b7 red blood cell count"),
                "WBC":        ("\U0001F6E1\ufe0f", "linear-gradient(135deg,#d97732,#8f431b)", "thousand/\u00b5L \u00b7 white blood cell (immune) count"),
                "Hematocrit": ("\U0001F4A7", "linear-gradient(135deg,#8e44ad,#5b2c6f)", "% \u00b7 blood volume made up of red cells"),
                "Platelets":  ("\U0001F9EB", "linear-gradient(135deg,#c2185b,#6a1b3d)", "thousand/\u00b5L \u00b7 cell fragments that help clotting"),
                "MCV":        ("\U0001F4CF", "linear-gradient(135deg,#2563eb,#1e3a8a)", "fL \u00b7 average size of a single red blood cell"),
                "MCH":        ("\U0001F9EC", "linear-gradient(135deg,#334155,#172033)", "pg \u00b7 average hemoglobin per red blood cell"),
                "MCHC":       ("\u2696\ufe0f", "linear-gradient(135deg,#b7354a,#712335)", "g/dL \u00b7 hemoglobin concentration in red cells"),
                "RDW-CV":     ("\U0001F4CA", "linear-gradient(135deg,#16a34a,#166534)", "% \u00b7 variation in red blood cell size"),
                "MPV":        ("\U0001F52C", "linear-gradient(135deg,#ca8a04,#78350f)", "fL \u00b7 average platelet size"),
            }
            DEFAULT_STYLE = ("\U0001F9EA", "linear-gradient(135deg,#475569,#1e293b)", "")

            for row_start in range(0, len(target_fields), 3):
                cbc_cols = st.columns(3)
                for col, f in zip(cbc_cols, target_fields[row_start:row_start + 3]):
                    value = reg_out[target_fields.index(f)] * field_std[f] + field_mean[f]
                    mae = field_mae.get(f, None)
                    r2 = field_r2.get(f, None)
                    reliable = r2 is not None and r2 > 0.3
                    icon, grad, unit_note = CBC_STYLE.get(f, DEFAULT_STYLE)
                    mae_text = f"\u00b1{mae:.2f}" if mae is not None else "?"
                    badge_html = (
                        '<span class="kpi-badge ok">\u2705 usable</span>' if reliable
                        else '<span class="kpi-badge warn">\u26a0\ufe0f low confidence</span>'
                    )
                    card_class = "kpi-card" + ("" if reliable else " kpi-card-muted")
                    tip = f"{unit_note}. Typical error \u00b1{mae:.2f} on held-out test data (R\u00b2={r2:.2f})." if mae is not None and r2 is not None else unit_note
                    col.markdown(f"""
                        <div class="{card_class}" style="background:{grad};" title="{tip}">
                            <div class="kpi-icon">{icon}</div>
                            <div class="kpi-value">{value:.1f}<span class="kpi-margin">{mae_text}</span></div>
                            <div class="kpi-label">{f}</div>
                            {badge_html}
                        </div>
                    """, unsafe_allow_html=True)
                st.write("")
            st.markdown(
                "<div class='illustrative-note'>Values with 'low confidence' mean the model doesn't actually "
                "track that parameter well (measured on real held-out test data) - treat those as unreliable "
                "rather than a real estimate.</div>", unsafe_allow_html=True
            )

        st.subheader("General notes (not a diagnosis)")
        rec_html = "".join(f"<li>{r}</li>" for r in RECOMMENDATIONS[predicted_label])
        st.markdown(f'<ul class="rec-list">{rec_html}</ul>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### What is Anemia?")
    ic1, ic2, ic3, ic4 = st.columns(4)
    info_cards = [
        ("\U0001FA78", "The Basics", "Anemia means your blood has fewer healthy red blood cells "
         "or less hemoglobin than normal, so less oxygen reaches your body's tissues.",
         "linear-gradient(135deg,#3a7bd5,#1b3d6d)"),
        ("\U0001F914", "Common Symptoms", "Fatigue, pale skin, shortness of breath, dizziness, "
         "cold hands/feet, and a fast or irregular heartbeat.",
         "linear-gradient(135deg,#e07b39,#a85423)"),
        ("\U0001F50D", "Common Causes", "Iron deficiency, vitamin B12/folate deficiency, chronic disease, "
         "blood loss, or inherited conditions like thalassemia.",
         "linear-gradient(135deg,#8e44ad,#5b2c6f)"),
        ("\U0001FA7A", "How It's Diagnosed", "A simple blood test (CBC) checks hemoglobin and red blood "
         "cell counts - the gold standard, not an image alone.",
         "linear-gradient(135deg,#0f9b8e,#0c6e64)"),
    ]
    for col, (icon, title, text, grad) in zip([ic1, ic2, ic3, ic4], info_cards):
        col.markdown(f"""
            <div class="info-card" style="background:{grad};">
                <div class="info-icon">{icon}</div>
                <div class="info-title">{title}</div>
                <div class="info-text">{text}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Recovery Journeys")
    st.markdown('<div class="illustrative-note">Illustrative, composite stories for educational purposes - not real documented case histories.</div>', unsafe_allow_html=True)
    st.write("")
    s1, s2, s3 = st.columns(3)
    stories = [
        ("\U0001F469", "Maya, 29", "Iron-deficiency anemia",
         "Constant fatigue and hair loss led Maya to get tested. A CBC confirmed low iron. "
         "With a doctor-guided iron regimen and dietary changes, her levels normalized within a few months, "
         "and her energy returned."),
        ("\U0001F468", "Daniyal, 41", "B12-deficiency anemia",
         "Frequent dizziness and numbness in his hands prompted Daniyal to see a doctor. Blood work showed "
         "low B12. Regular supplementation under medical supervision resolved his symptoms over several weeks."),
        ("\U0001F9D1", "Sara, 34", "Anemia from chronic blood loss",
         "Sara noticed persistent paleness and breathlessness during light exercise. Investigation found "
         "an underlying cause of blood loss, which was treated directly - her hemoglobin recovered fully afterward."),
    ]
    for col, (avatar, name, tag, text) in zip([s1, s2, s3], stories):
        col.markdown(f"""
            <div class="story-card">
                <div class="story-avatar">{avatar}</div>
                <div class="story-name">{name}</div>
                <div class="story-tag">{tag}</div>
                <div class="story-text">{text}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Frequently Asked Questions")

    faqs = [
        ("Is anemia serious?", "It ranges from mild to severe. Mild anemia is often manageable, but "
         "untreated or severe anemia can strain the heart and other organs, so it's worth taking seriously "
         "and getting properly diagnosed rather than guessing."),
        ("Can anemia be cured?", "Many common forms (like iron or B12 deficiency) are very treatable, "
         "often fully reversible once the underlying cause is addressed. Some inherited forms are managed "
         "long-term rather than cured. A doctor can tell you which situation applies to you."),
        ("What test actually confirms anemia?", "A Complete Blood Count (CBC) is the standard test - it "
         "measures hemoglobin, hematocrit, and red blood cell counts directly from a blood sample. "
         "An image-based prediction like this dashboard is not a substitute for that test."),
        ("Can diet alone fix anemia?", "For mild iron-deficiency anemia, diet can help, but it depends on "
         "the cause and severity. Some cases need supplements or medical treatment. Don't self-treat without "
         "a diagnosis, since the wrong approach (like unsupervised iron supplementation) can cause harm."),
        ("Does pale skin always mean anemia?", "No - paleness has many causes (cold, low blood pressure, "
         "skin tone, lighting). It's a signal worth checking, not a diagnosis on its own."),
        ("How long does treatment usually take?", "It varies by cause and severity - some people feel "
         "better within weeks of starting treatment, others take longer. Your doctor can give you a realistic "
         "timeline based on your specific blood work and cause."),
        ("Is this dashboard a diagnosis?", "No. This is a research/portfolio project analyzing a single "
         "image. A real diagnosis requires a blood test and a doctor's evaluation - please don't treat any "
         "prediction here as medical fact."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.write(a)
    st.markdown('</div>', unsafe_allow_html=True)
