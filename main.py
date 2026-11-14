"""Legacy Streamlit entry point for the sodium-ion digital twin."""

import streamlit as st

st.set_page_config(
    page_title="Sodium Twin · Operations Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Sodium-Ion Battery Digital Twin · FMI + AI Hybrid",
    },
)

from config.settings import apply_custom_css
from utils.db_manager import init_db, session_valid
from views.dashboard_view import render_dashboard
from views.login_view import render_login


init_db()
apply_custom_css()

st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("username", "")
st.session_state.setdefault("workspace_page", "运行总览")

if st.session_state["logged_in"] and not session_valid(
    st.session_state["username"], st.session_state.get("credential_revision")
):
    st.session_state["logged_in"] = False
    st.session_state.pop("sim_result", None)

if st.session_state["logged_in"]:
    render_dashboard()
else:
    render_login()
