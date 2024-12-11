import time
from pathlib import Path
import streamlit as st

def cleanup_temp_file(temp_file_path: str):
    try:
        time.sleep(1)  # Allow time for file release
        temp_file = Path(temp_file_path)
        if temp_file.exists():
            temp_file.unlink()
    except Exception as e:
        st.error(f"Error cleaning up temporary file: {str(e)}")

def display_temporary_message(message, placeholder=None, message_type="status", duration=3):
    if not placeholder:
        placeholder = st.empty()
    if message_type == "success":
        placeholder.success(message)
    elif message_type == "warning":
        placeholder.warning(message)
    else:
        placeholder.status(message)
    time.sleep(duration)
    placeholder.empty()

def reset_tracking_state():
    st.session_state.stop_tracking = True
    st.session_state.tracking_active = False
    st.session_state.show_results = False

def get_default_tracking_args():
    """Returns default tracking arguments"""
    return {
        'imgsz': [640],
        'classes': None,
        'project': 'runs/track',
        'name': 'exp',
        'exist_ok': False,
        'half': False,
        'vid_stride': 1,
        'show': False,
        'show_conf': True,
        'show_trajectories': False,
        'save_txt': False,
        'line_width': None,
        'per_class': False,
        'verbose': True,
        'agnostic_nms': False,
        'ext_track': True,
    }