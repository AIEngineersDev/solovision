import streamlit as st
from pathlib import Path
from solovision.solovision_app.styles import get_custom_css


def initialize_app_layout():
    # Initializes the app layout with custom CSS and title section
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    st.markdown("""
        <div class="title-section">
            <h1>Solovision</h1>
            <h3>Advanced Multi-Object Tracking System</h3>
        </div>
    """, unsafe_allow_html=True)


def read_description():
    # Reads and displays the description and key features of Solovision
    description = st.session_state.placeholders["description"]
    description.markdown("""
            <div class="description-section">
                <h2>About Solovision</h2>
                <p>Solovision is a powerful multi-object tracking system that combines state-of-the-art 
                YOLO object detection with robust tracking algorithms. It offers real-time tracking 
                capabilities with support for multiple object classes and ReID models.</p>
            </div>
            <div class="features-section">
                <h2>Key Features</h2>
                <div class="features-grid">
                    <div class="feature-item">
                        <h6>Support for Multiple YOLO versions</h6>
                    </div>
                    <div class="feature-item">
                        <h6>Real-time Tracking</h6>
                    </div>
                    <div class="feature-item">
                        <h6>Multi-class and Multi Camera Detection</h6>
                    </div>
                    <div class="feature-item">
                        <h6>ReID Integration</h6>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


def tracking_display(model_path):
    # Displays the tracking results and the model used for tracking
    model_path = model_path.replace('.pt', '').capitalize()
    tracking_placeholder =  st.session_state.placeholders["tracking"]
    tracking_results, model_markdown = tracking_placeholder.columns(2)
    tracking_results.markdown('<div style="text-align: center; margin-bottom: 1rem;">'
        '<p style="font-size: 18px; font-weight: normal; color: white;">Tracking Results</p></div>', unsafe_allow_html=True)
    model_markdown.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <p style="font-size: 18px; font-weight: normal; color: white;">Model: {model_path}</p>
    </div>
    """, unsafe_allow_html=True)