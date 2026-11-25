import json
import os
from pathlib import Path
from threading import Lock

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

from workspace_home import select_module

st.set_page_config(page_title="AnemiaSense Dashboard", layout="wide", page_icon="\U0001FA78")

from ui import apply_theme, render_hero, metric_grid, section_heading

apply_theme()

if select_module() == "pbs":
    from pbs_page import render_pbs

    render_pbs()
    st.stop()

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

    return metrics, class_avg_color, model, class_names, reg_model, reg_meta, Lock()

metrics, class_avg_color, model, class_names, reg_model, reg_meta, model_lock = load_everything()

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
        self.hooks = (
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        )

    def close(self):
        for hook in self.hooks:
            hook.remove()

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

def make_gradcam_overlay(pil_img, input_tensor, class_idx, keep_top_pct=25):
    gradcam = GradCAM(model, model.layer4)
    try:
        cam = gradcam.generate(input_tensor, class_idx)
    finally:
        gradcam.close()
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
tab1, tab2 = st.tabs(["Model dashboard", "Learn about anaemia"])

with tab1:
    # ================= HEADER =================
    render_hero(
        "Read the cells.\nExplore the evidence.",
        "Explore red blood cell images, understand model performance, and inspect "
        "every prediction in one research workspace.",
        eyebrow="ANAEMIA · IMAGE ANALYSIS",
        tags=["Held-out evaluation", f"{len(all_labels):,} test images", "ResNet-34"],
        anchor="model-performance", accent="cells",
    )

    display_class_names = [name.replace("_individuals", "") for name in class_names]
    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )
    overall_acc = (all_preds == all_labels).mean()
    auc_score = roc_auc_score(all_labels, all_probs[:, 1]) if len(class_names) == 2 else None

    # ================= KPI CARDS =================
    section_heading("01 / MODEL PERFORMANCE", "The numbers behind the model.",
                    "Held-out test results. Scroll to explore each measure.", anchor="model-performance")
    metric_grid([
        {"label": "Test accuracy", "value": overall_acc * 100, "decimals": 1, "suffix": "%",
         "note": "Correct predictions across all test images.", "featured": True},
        {"label": "Macro precision", "value": report_dict['macro avg']['precision'] * 100,
         "decimals": 1, "suffix": "%", "note": "How often a predicted class is correct."},
        {"label": "Macro recall", "value": report_dict['macro avg']['recall'] * 100,
         "decimals": 1, "suffix": "%", "note": "How many cases of each class are found."},
        {"label": "Macro F1", "value": report_dict['macro avg']['f1-score'] * 100,
         "decimals": 1, "suffix": "%", "note": "The balance of precision and recall."},
        {"label": "ROC–AUC", "value": auc_score, "decimals": 3,
         "note": "Ranking quality across probability cutoffs."},
        {"label": "Decision threshold", "value": best_threshold, "decimals": 2,
         "note": "Probability cutoff selected on validation data."},
    ], key="anemia_performance")
    section_heading("02 / TRAINING & EVALUATION", "Go beyond a single score.",
                    "Explore learning curves, class performance, and prediction errors.")

    # ================= CHARTS =================
    col1, col2 = st.columns(2)

    with col1:
        epochs_list = list(range(1, len(train_losses) + 1))
        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(x=epochs_list, y=train_losses, mode="lines+markers",
                                        name="Train Loss", line=dict(color="#b92842")))
        fig_train.add_trace(go.Scatter(x=epochs_list, y=val_losses, mode="lines+markers",
                                        name="Val Loss", line=dict(color="#d5827b")))
        fig_train.add_trace(go.Scatter(x=epochs_list, y=[a / 100 for a in val_accuracies], mode="lines+markers",
                                        name="Val Accuracy (scaled)", line=dict(color="#603943", dash="dot"),
                                        yaxis="y2"))
        fig_train.update_layout(
            title="Training Loss & Validation Accuracy",
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Epoch"), yaxis=dict(title="Loss"),
            yaxis2=dict(title="Val Accuracy", overlaying="y", side="right", range=[0, 1]),
            legend=dict(orientation="h", y=-0.25), height=360,
        )
        st.plotly_chart(fig_train, width="stretch")

    with col2:
        metrics_rows = []
        for cls, display_cls in zip(class_names, display_class_names):
            metrics_rows.append({"class": display_cls, "metric": "Precision", "value": report_dict[cls]["precision"]})
            metrics_rows.append({"class": display_cls, "metric": "Recall", "value": report_dict[cls]["recall"]})
            metrics_rows.append({"class": display_cls, "metric": "F1-score", "value": report_dict[cls]["f1-score"]})
        metrics_df = pd.DataFrame(metrics_rows)
        fig_prf = px.bar(
            metrics_df, x="class", y="value", color="metric", barmode="group", text_auto=".2%",
            template="plotly_white", title="Precision / Recall / F1-score per class",
            color_discrete_map={"Precision": "#d5827b", "Recall": "#b92842", "F1-score": "#603943"},
        )
        fig_prf.update_layout(
            yaxis=dict(range=[0, 1], title="Score"), xaxis_title="", height=360,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_prf, width="stretch")
        st.caption("Precision = few false alarms. Recall = few missed cases. F1 = balance of both. Higher bars = better.")

    # ================= CONFUSION MATRIX + INTERACTIVE METRIC LAB =================
    cm_col, guide_col = st.columns([1.08, 0.92], gap="large")

    with cm_col:
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
            color_continuous_scale="Reds", template="plotly_white",
            labels=dict(x="Predicted", y="Actual", color=color_title),
            aspect="auto",
        )
        fig_cm.update_traces(hovertemplate=hover_template)
        fig_cm.update_layout(
            height=355, margin=dict(l=5, r=5, t=15, b=5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(thickness=12),
        )
        st.plotly_chart(fig_cm, width="stretch")
        correct_total = int(np.trace(cm_counts))
        error_total = int(cm_counts.sum() - correct_total)
        cm_stat1, cm_stat2 = st.columns(2)
        cm_stat1.metric("Correct", f"{correct_total} / {int(cm_counts.sum())}")
        cm_stat2.metric("Misclassified", error_total)
        st.caption(
            "Read across each row: the diagonal cells are correct predictions; off-diagonal cells are mistakes."
        )

    with guide_col:
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
                    template="plotly_white", color_discrete_sequence=["#bb2943", "#d5827b", "#93626b", "#5e3943"],
                )
                fig_curve.add_vline(x=best_threshold, line_dash="dash", line_color="#40282f")
                fig_curve.add_vline(x=demo_threshold, line_dash="dot", line_color="#db5268")
                fig_curve.update_layout(
                    height=280, yaxis=dict(range=[0, 1]), margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_curve, width="stretch")
                st.caption("Dashed = official threshold; dotted pink = slider threshold.")
        else:
            st.info("Run the calibration/threshold cell, then re-save dashboard artifacts to enable this playground.")

    # ================= UPLOAD + PREDICT =================
    section_heading("03 / PREDICTION STUDIO", "From image to insight.", "Upload a segmented blood cell image to explore the model’s prediction.")
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

        with model_lock, torch.no_grad():
            probs = predict_with_tta(model, input_tensor)
        prob_class1 = probs[0][1].item()
        pred_idx = int(prob_class1 >= best_threshold)
        predicted_label = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

        with model_lock:
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

        st.markdown('<div class="gradient-heading">Why the model called it this way</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="gradient-heading">\U0001F9EA Estimated CBC panel (approximate)</div>', unsafe_allow_html=True)
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

            units = {
                "Hemoglobin": "g/dL", "RBC": "million/µL", "WBC": "thousand/µL",
                "Hematocrit": "%", "Platelets": "thousand/µL", "MCV": "fL",
                "MCH": "pg", "MCHC": "g/dL", "RDW-CV": "%", "MPV": "fL",
            }
            cbc_metrics = []
            for index, field in enumerate(target_fields):
                value = reg_out[index] * field_std[field] + field_mean[field]
                mae = field_mae.get(field)
                r2 = field_r2.get(field)
                confidence_note = "" if r2 is not None and r2 > 0.3 else "Low confidence. "
                error_note = f"Typical error ±{mae:.2f}. " if mae is not None else ""
                r2_note = f"Held-out R² {r2:.2f}." if r2 is not None else ""
                cbc_metrics.append({
                    "label": f"{field} · {units.get(field, '')}", "value": float(value), "decimals": 1,
                    "note": f"{confidence_note}{error_note}{r2_note}",
                })
            metric_grid(cbc_metrics, key="estimated_cbc")
            st.markdown(
                "<div class='illustrative-note'>Values with 'low confidence' mean the model doesn't actually "
                "track that parameter well (measured on real held-out test data) - treat those as unreliable "
                "rather than a real estimate.</div>", unsafe_allow_html=True
            )

        st.markdown('<div class="gradient-heading">General notes (not a diagnosis)</div>', unsafe_allow_html=True)
        rec_html = "".join(f"<li>{r}</li>" for r in RECOMMENDATIONS[predicted_label])
        st.markdown(f'<ul class="rec-list">{rec_html}</ul>', unsafe_allow_html=True)


with tab2:

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
