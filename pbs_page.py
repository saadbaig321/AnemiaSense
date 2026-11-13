"""Streamlit interface for the ThalassemiaPBS research workflow."""

from collections import Counter
from hashlib import sha256
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from pbs import CELL_TYPES, COLORS, classify_cells, decode_image, load_morphology_model, segment_smear

SAMPLE = Path(__file__).resolve().parent / "dashboard_data" / "pbs_example.jpg"
LOW_SCORE = 0.55


@st.cache_resource
def get_classifier():
    return load_morphology_model()


def render_pbs():
    st.markdown("""
        <div class="hero-card">
            <h1 class="hero-title">Thalassemia <span>PBS</span></h1>
            <p class="hero-subtitle">Analyse a peripheral blood smear and review the
            predicted morphology of individual red blood cells.</p>
        </div>
    """, unsafe_allow_html=True)
    st.info("Research use only. This workflow reports red blood cell morphology. "
            "Thalassemia diagnosis requires clinical assessment and laboratory testing.")

    source = st.radio(
        "Image source", ("Published example", "Upload a smear"), horizontal=True, key="pbs_source",
    )
    if source == "Published example":
        data = SAMPLE.read_bytes()
        st.caption("Published example T1-001 · Tyas et al., ThalassemiaPBS · CC BY 4.0")
    else:
        upload = st.file_uploader(
            "Upload a peripheral blood smear", type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="pbs_upload", max_upload_size=20, help="Single microscope field at 1000×.",
        )
        if upload is None:
            st.session_state.pop("pbs_result", None)
            return
        data = upload.getvalue()

    digest = sha256(data).hexdigest()
    result = st.session_state.get("pbs_result")
    if result is not None and result[0] != digest:
        st.session_state.pop("pbs_result")
        result = None

    if st.button("Analyse smear", type="primary"):
        try:
            with st.spinner("Segmenting the smear and classifying candidate cells…"):
                image = decode_image(data)
                working, green, cells = segment_smear(image)
                if not cells:
                    st.session_state.pop("pbs_result", None)
                    st.warning("No candidate cells were segmented. Try a sharp, well-lit 1000× field.")
                    return
                model, evaluation = get_classifier()
                predictions = classify_cells(model, [cell.crop for cell in cells])
                result = (digest, working, green, cells, predictions, evaluation)
                st.session_state["pbs_result"] = result
        except ValueError as error:
            st.session_state.pop("pbs_result", None)
            st.error(str(error))
            return
        except FileNotFoundError:
            st.error("The PBS classifier checkpoint is missing. See the README for setup.")
            return

    if result is not None:
        render_results(*result[1:])


def render_results(working, green, cells, predictions, evaluation):
    overlay = working.copy()
    rows = []
    for number, (cell, (label, score)) in enumerate(zip(cells, predictions), start=1):
        uncertain = score < LOW_SCORE
        color = (220, 220, 220) if uncertain else COLORS[label]
        cv2.drawContours(overlay, [cell.contour], -1, (25, 25, 25), 3)
        cv2.drawContours(overlay, [cell.contour], -1, color, 1)
        x, y, width, height = cell.box
        position = (x, max(y - 2, 11))
        cv2.putText(overlay, str(number), position, cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 3)
        cv2.putText(overlay, str(number), position, cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)
        rows.append({
            "Cell": number, "Predicted morphology": CELL_TYPES[label],
            "Model score": round(score, 3), "Review": "Low score" if uncertain else "",
            "X": x, "Y": y, "Width": width, "Height": height,
        })

    st.subheader("Smear analysis")
    original, annotated = st.columns(2)
    original.image(working, caption="Smear at analysis resolution", width="stretch")
    annotated.image(overlay, caption="Candidate cells numbered for review", width="stretch")
    with st.expander("View green-channel preprocessing"):
        st.image(green, caption="Green channel used for segmentation")

    counts = Counter(row["Predicted morphology"] for row in rows if not row["Review"])
    uncertain_count = sum(bool(row["Review"]) for row in rows)
    left, right = st.columns(2)
    left.metric("Candidate cells", len(rows))
    right.metric("Low-score candidates", uncertain_count)
    st.caption("Low-score candidates are flagged for review and counted separately. "
               "Model scores are uncalibrated. Counts describe this image field only.")
    distribution = pd.DataFrame({
        "Morphology": [*CELL_TYPES, "Low score"],
        "Candidate cells": [*(counts[label] for label in CELL_TYPES), uncertain_count],
    })
    st.bar_chart(distribution, x="Morphology", y="Candidate cells", horizontal=True, sort=False)

    table = pd.DataFrame(rows)
    with st.expander("Review individual candidate cells"):
        st.dataframe(table, hide_index=True)
        cell_number = st.selectbox("Cell number", table["Cell"].tolist(), key="pbs_cell_number")
        row = rows[cell_number - 1]
        st.image(cells[cell_number - 1].crop, width=160,
                 caption=f"Cell {cell_number} · {row['Predicted morphology']}")
        st.download_button("Download candidate cell CSV", table.to_csv(index=False),
                           file_name="pbs-morphology.csv", mime="text/csv", on_click="ignore")

    with st.expander("Model evaluation and dataset"):
        st.write(f"Evaluation on {evaluation['test_crops']:,} labelled crops from "
                 f"{evaluation['test_fields']} held-out microscope fields. "
                 f"Macro F1 {evaluation['macro_f1']:.3f}. "
                 f"Balanced accuracy {evaluation['balanced_accuracy']:.1%}.")
        st.caption("These scores evaluate isolated crops. Full-smear segmentation accuracy "
                   "and performance on unseen patients have yet to be established.")
        st.dataframe(pd.DataFrame({"Morphology": CELL_TYPES,
                                   "Held-out crop count": evaluation["test_support"]}),
                     hide_index=True)
