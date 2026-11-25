"""Locus-inspired interface with accessible, viewport-aware research metrics."""
from base64 import b64encode
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

ROOT = Path(__file__).resolve().parent
_metric_component = components.component(
    "anemiasense.metric_grid",
    html='<div class="as-metrics" role="list" aria-label="Research metrics"></div>',
    css=(ROOT / "metrics.css").read_text(),
    js=(ROOT / "metrics.js").read_text(),
)
_motion_component = components.component(
    "anemiasense.scroll_motion", js=(ROOT / "motion.js").read_text(),
)


@lru_cache(maxsize=4)
def _image_data(path):
    return b64encode(Path(path).read_bytes()).decode()


def apply_theme():
    st.html(ROOT / "styles.css")
    _motion_component(key="page_motion", height=0)


def metric_grid(items, key, compact=False):
    """Animate actual model or analysis values; no scores are fabricated."""
    _metric_component(data={"items": items, "compact": compact}, key=key,
                      width="stretch", height="content")


def section_heading(eyebrow, title, description="", anchor=None):
    ident = f' id="{escape(anchor, quote=True)}"' if anchor else ""
    st.html(f'<div class="as-section"{ident}><div><span class="as-eyebrow">{escape(eyebrow)}</span>'
            f'<h2>{escape(title)}</h2></div><p>{escape(description)}</p></div>')


def render_hero(title, description, eyebrow="BLOOD CELL RESEARCH", tags=(), anchor=None, accent="", cta="Explore the metrics"):
    title_html = "<br>".join(escape(title).split("\n"))
    if accent:
        safe_accent = escape(accent)
        title_html = title_html.replace(safe_accent, f'<em>{safe_accent}</em>', 1)
    chips = "".join(f'<span>{escape(tag)}</span>' for tag in tags)
    image = _image_data(ROOT / "assets" / "blood-cells-hero.webp")
    link = (f'<a class="as-hero-link" href="#{escape(anchor, quote=True)}">{escape(cta)} <span>↗</span></a>'
            if anchor else '')
    st.html(f'''<section class="as-hero">
      <img class="as-hero-art" alt="" aria-hidden="true" src="data:image/webp;base64,{image}">
      <div class="as-hero-copy"><span class="as-eyebrow">{escape(eyebrow)}</span>
        <h1>{title_html}</h1><p>{escape(description)}</p>{link}
        <div class="as-tags">{chips}</div>
      </div>
      <div id="research-note" class="as-research-note"><span class="as-note-dot"></span> Educational research. Not a clinical diagnosis.</div>
    </section>''')


def render_brand():
    logo = ROOT.parent / "duhs_logo.png"
    institution = ""
    if logo.exists():
        image = _image_data(logo)
        institution = f'<div class="as-institution"><img alt="DUHS" src="data:image/png;base64,{image}"><span>Supervised by<br><strong>Professors of D.U.H.S</strong></span></div>'
    st.html(f'<header id="workspace-top" class="as-brand-row"><a class="as-brand" href="#workspace-top" aria-label="AnemiaSense overview"><span class="as-brand-mark">✳</span>AnemiaSense<span class="as-brand-period">.</span></a>'
            f'<nav class="as-nav" aria-label="Page navigation"><a href="#workspace-top">Overview</a><a href="#research-note">Research notes</a></nav>{institution}</header>')


def render_overview():
    """Editorial research section and workflow, adapted to the app's capabilities."""
    image = _image_data(ROOT / "assets" / "blood-cells-hero.webp")
    st.html(f'''<section id="research-context" class="as-about">
      <div class="as-about-art"><img alt="Decorative red blood cell artwork" src="data:image/webp;base64,{image}"><span>ANEMIASENSE / A CLOSER LOOK</span></div>
      <div class="as-about-copy"><span class="as-eyebrow">ABOUT THE WORKSPACE</span>
        <h2>Read the image.<br>Understand the model.<br><span>Keep the evidence in view.</span></h2>
        <p>Move from a single segmented cell to a full microscope field. Explore saved model performance, visual explanations, and individual cell predictions in one workspace.</p>
        <a href="#modules">Find your analysis module <span>↗</span></a>
      </div></section>''')


def render_workflow():
    st.html('''<section id="workflow" class="as-workflow"><div class="as-workflow-heading"><span class="as-eyebrow">HOW IT WORKS</span><h2>From pixels<br>to perspective.</h2><p>Three steps to explore your image.</p></div>
      <ol><li><span>01</span><h3>Choose your input</h3><p>A segmented red blood cell or a full peripheral blood smear.</p></li>
      <li><span>02</span><h3>Explore the analysis</h3><p>Review model predictions, explanations, and field-level results.</p></li>
      <li><span>03</span><h3>Review the evidence</h3><p>Inspect saved evaluation, uncertainty, and individual candidate cells.</p></li></ol></section>''')
