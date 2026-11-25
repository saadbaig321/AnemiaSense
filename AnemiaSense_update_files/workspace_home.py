"""Sign-in and module selection for the Streamlit entry point."""

import os

from html import escape

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from ui import render_hero, render_brand, section_heading, metric_grid, render_overview, render_workflow



def select_module():
    render_brand()
    try:
        auth = st.secrets.get("auth")
    except StreamlitSecretNotFoundError:
        auth = None
    demo = auth is None and os.environ.get("ANEMIASENSE_DEMO") == "1"

    if not demo and not getattr(st.user, "is_logged_in", False):
        render_hero("Read the cells.\nReveal the pattern.",
                    "A considered workspace for anaemia and peripheral blood smear research. Sign in to explore your analysis modules.",
                    tags=["Image analysis", "Explainable predictions", "Educational research"],
                    accent="cells", anchor="access-control", cta="Access your workspace")
        st.html('<div id="access-control"></div>')
        if auth is None:
            st.info("Configure sign-in in .streamlit/secrets.toml. See the README for setup.")
        elif st.button("Sign in", type="primary"):
            st.login()
        st.stop()

    top, account = st.columns([4, 1])
    current = st.session_state.get("analysis_module")
    if current and top.button("← Choose another module"):
        st.session_state.pop("analysis_module", None)
        st.session_state.pop("pbs_result", None)
        st.session_state.pop("pbs_source", None)
        st.session_state.pop("pbs_cell_number", None)
        st.rerun()
    if demo:
        account.caption("Local demo")
    elif account.button("Sign out"):
        st.session_state.clear()
        st.logout()
        st.stop()

    if current:
        return current

    render_hero("Read the cells.\nReveal the pattern.",
                "Discover a clearer way to explore blood cell images. Two focused modules bring model performance, visual explanations, and morphology into view.",
                tags=["Image analysis", "Explainable predictions", "Educational research"],
                accent="cells", anchor="modules", cta="Explore the modules")
    metric_grid([
        {"label": "Analysis modules", "value": 2, "note": "Segmented cells and whole microscope fields."},
        {"label": "PBS morphology classes", "value": 9, "note": "Cell-by-cell morphology exploration."},
        {"label": "Research steps", "value": 3, "note": "Choose, analyse, and review the evidence."},
    ], key="workspace_overview", compact=True)
    render_overview()
    section_heading("WHAT YOU CAN EXPLORE", "Analysis built around\nyour research.",
                    "Choose the workflow that matches your image. Each module brings its own saved evaluation and analysis tools.", anchor="modules")
    anemia, pbs = st.columns(2, gap="large")
    modules = [
        (anemia, "anemia", "01 / SEGMENTED RBC", "Anaemia", "A single cell. A closer look.",
         "Explore classification, Grad-CAM visual explanations, and approximate CBC estimates from a segmented red blood cell image.",
         "Colour image · PNG / JPG", "Open anaemia dashboard"),
        (pbs, "pbs", "02 / PERIPHERAL BLOOD SMEAR", "Thalassemia PBS", "A whole field of possibilities.",
         "Segment a microscope field, explore nine red blood cell morphology classes, and review individual candidate cells.",
         "Full microscope field · 1000×", "Open Thalassemia PBS"),
    ]
    for col, module, eyebrow, title, subtitle, copy, note, button in modules:
        with col:
            st.html(f'<article class="as-module-card {module}"><div class="as-module-top">{escape(eyebrow)}<span>↗</span></div>'
                    f'<div class="as-module-symbol" aria-hidden="true">{"◉" if module == "anemia" else "⠿"}</div><h3>{escape(title)}</h3><p><strong>{escape(subtitle)}</strong><br>{escape(copy)}</p><small>{escape(note)}</small></article>')
            if st.button(button, type="primary", key=f"open_{module}", width="stretch"):
                st.session_state["analysis_module"] = module
                st.rerun()
    render_workflow()
    st.html('<footer class="as-footer"><span>AnemiaSense · A clearer perspective on blood cell research.</span>'
            '<span>Educational research only · Not a medical diagnosis</span></footer>')
    st.stop()
