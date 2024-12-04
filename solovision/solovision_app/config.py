import streamlit as st

def initialize_session_state():
    # Initializes the session state with default values for tracking and placeholders
    if 'tracking_active' not in st.session_state:
        st.session_state.tracking_active = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'stop_tracking' not in st.session_state:
        st.session_state.stop_tracking = False
    if 'placeholders' not in st.session_state:
        st.session_state.placeholders = {
            "description": st.empty(),
            "tracking": st.empty(),
            "video": st.empty() 
            }        
