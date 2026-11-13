"""Sign-in and module selection for the Streamlit entry point."""

import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def hero(title, description):
    st.markdown(f"""
        <div class="hero-card">
            <div class="hero-eyebrow">ANEMIASENSE</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{description}</p>
        </div>
    """, unsafe_allow_html=True)


def select_module():
    try:
        auth = st.secrets.get("auth")
    except StreamlitSecretNotFoundError:
        auth = None
    demo = auth is None and os.environ.get("ANEMIASENSE_DEMO") == "1"

    if not demo and not getattr(st.user, "is_logged_in", False):
        hero("Sign in to AnemiaSense",
             "Access the anaemia and peripheral blood smear research workflows.")
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

    hero("Choose an analysis", "Select a module based on the image you have.")
    anemia, pbs = st.columns(2)
    with anemia:
        with st.container(border=True):
            st.subheader("Anaemia")
            st.write("Use the aneRBC model with a colour image of a segmented red blood cell.")
            if st.button("Open anaemia dashboard", type="primary"):
                st.session_state["analysis_module"] = "anemia"
                st.rerun()
    with pbs:
        with st.container(border=True):
            st.subheader("Thalassemia PBS")
            st.write("Use a full microscope field to explore nine red blood cell morphology classes.")
            if st.button("Open Thalassemia PBS", type="primary"):
                st.session_state["analysis_module"] = "pbs"
                st.rerun()
    st.stop()
