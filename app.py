import streamlit as st

from src.ui.workspace import main


st.set_page_config(
    page_title="Observatório de anomalias",
    page_icon=":material/query_stats:",
    layout="wide",
)

main()