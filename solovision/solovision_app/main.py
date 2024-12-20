import cv2
import torch
import base64
import argparse
import streamlit as st
from pathlib import Path
from solovision.solovision_app.utils import *
from solovision.solovision_app.yolo_classes import YOLO_CLASSES
from solovision.inference import run
from solovision.utils import WEIGHTS
from solovision.solovision_app.model_selection import (
    get_model_sizes,
    get_model_variants,
    construct_model_path,
    get_available_yolo_versions,
    get_reid_models
)
from solovision.solovision_app.layout import (
    initialize_app_layout, 
    read_description, 
    tracking_display )

def main():
    st.set_page_config(page_title="Solovision")
    # Initializes the Solovision app layout, session state, and handles the tracking process.
    initialize_app_layout()
    initialize_session_state()
    
    # Show description when no tracking results are displayed
    if not st.session_state.show_results:
        read_description()
    else:
        st.session_state.placeholders["description"].empty()
    
    # Sidebar: Source selection
    st.logo(image="https://raw.githubusercontent.com/AIEngineersDev/solovision/multiple-trackers/assets/logo/logo.png",
            size= 'large', link = "http://localhost:8501/?page=home")
    
    source_type = st.sidebar.selectbox(
        "Select Source",
        ["Video File", "Webcam", "Stream"]
    )
    source = None
    if source_type == "Video File":
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            # Create a temporary file to store the uploaded video
            temp_file = Path("solovision.mp4")
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            source = str(temp_file)
    elif source_type == "Webcam":
        source = "0"  # Pass as string for webcam
    elif source_type == "Stream":
        source = st.sidebar.text_input("http:// or rtsp://")

    st.session_state.placeholders.update({
    "track_button": st.sidebar.empty(),
    "spinner": st.sidebar.empty()
    })

    track_button = st.session_state.placeholders["track_button"]
    spinner = st.session_state.placeholders["spinner"]

    # Sidebar: Model selection and configurations
    st.sidebar.markdown("### YOLO Model Selection")
    yolo_version = st.sidebar.selectbox(
        "YOLO Version",
        get_available_yolo_versions(),
        index=get_available_yolo_versions().index("YOLOv8")
    )
    model_path = None
    selected_indices = []  # Detect all classes by default
    
    if yolo_version == "Custom":
        model_path = st.sidebar.file_uploader("Upload Custom Model", type=['pt'])
    else:
        model_sizes, display_sizes = get_model_sizes(yolo_version)
        size_index = st.sidebar.selectbox(
            "Model Size", 
            display_sizes,
            index=display_sizes.index("Small")
        )
        size = model_sizes[display_sizes.index(size_index)]
        variants, variant_names = get_model_variants(yolo_version, size)
        variant = ""
        if len(variants) > 1:
            variant_index = st.sidebar.selectbox(
                "Task Mode", 
                variant_names,
                index=variant_names.index("Detection")
            )
            variant = variants[variant_names.index(variant_index)]
        model_path = construct_model_path(yolo_version, size, variant)
        
        # Only show class selection for non-custom models
        class_names = list(YOLO_CLASSES.values())
        class_names = ["All"] + class_names
        selected_classes = st.sidebar.multiselect(
            "Objects to Detect",
            options=class_names,
            default=["All"] 
        )
        if "All" in selected_classes:
            selected_indices = None  # YOLO defaults to all classes if this is None
        else:
            selected_indices = [k for k, v in YOLO_CLASSES.items() if v in selected_classes]

    # Confidence and IOU thresholds
    conf_thres = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
    iou_thres = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.7)
    
    command = st.sidebar.toggle('Tracking', value=False, 
                           help='Enables Object Tracking across video frames using unqiue ids')
    if command:
        # ReID tracking model selection
        st.sidebar.markdown("### ReID Model Selection")
        with_reid = st.sidebar.toggle('ReID Tracking', value=False, 
                            help='Enable ReID features for better tracking association')
        if with_reid:
            reid_model = st.sidebar.selectbox(
                "Select ReID Model", 
                get_reid_models(),
                index=get_reid_models().index("osnet_x1_0_msmt17.pt")
        )
    
    reid_model_path = WEIGHTS / reid_model if command and with_reid else None
    
    # Post Detection Settings
    st.sidebar.markdown("### Display and Save Options")
    save_results = st.sidebar.checkbox("Save Results", False)
    show_labels = st.sidebar.checkbox("Show Labels", True)
    
    # Model configuration
    model_config = {
        'yolo_model': str(model_path),
        'reid_model': reid_model_path,
        'source': str(source),
        'conf': conf_thres,
        'iou': iou_thres,
        'command': 'track' if command else 'detect',
        'classes': selected_indices,
        'show_labels': show_labels,
        'save': save_results,
        'device': '' if torch.cuda.is_available() else 'cpu',
        'with_reid': with_reid if command else False 
    }

    # Start or stop tracking
    if source and model_path:
        if not st.session_state.tracking_active:
            if track_button.button("Start Inference"):
                st.session_state.tracking_active = True
                st.session_state.show_results = True
                st.session_state.stop_tracking = False
                display_temporary_message("Starting", spinner, message_type= "status", duration=2)
                st.rerun()
        else:
            if track_button.button("Stop Inference"):
                reset_tracking_state()
                if source_type == "Video File":
                    cleanup_temp_file("solovision.mp4")
                display_temporary_message("Stopping", spinner, message_type= "status", duration=2)
                st.session_state.placeholders["tracking"].empty()
                st.session_state.placeholders["video"].empty()
                st.rerun()

            if not st.session_state.stop_tracking:
                st.session_state.placeholders["description"].empty()
                tracking_display(model_path)
                video_placeholder = st.session_state.placeholders["video"]
                # Get default args and update with user config
                default_args = get_default_tracking_args()
                default_args.update(model_config)
                model_args = argparse.Namespace(**default_args)
                
                try:
                    def frame_callback(frame):
                        if st.session_state.stop_tracking:
                            return False
                        # Convert frame to Base64 to embed within HTML 
                        _, buffer = cv2.imencode('.jpg', frame)
                        encoded_frame = base64.b64encode(buffer).decode('utf-8')
                        video_placeholder.markdown(
                            f"""
                            <div style="
                                margin: 0px auto; 
                                width: 80%; 
                                max-width: 100%; 
                                padding: 10px;
                                border: none;
                                border-radius: 15px;
                                background-color: black; 
                                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
                                text-align: center;">
                                <img src="data:image/jpeg;base64,{encoded_frame}" 
                                    alt="Video Frame" 
                                    style="width: 100%; border-radius: 15px;">
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        return True
                    
                    model_args.stream_display = frame_callback
                    run(model_args)
                except Exception as e:
                    st.error(f"Error during tracking: {str(e)}")
                finally:
                    reset_tracking_state()
                    if source_type == "Video File":
                        cleanup_temp_file("solovision.mp4")
                    display_temporary_message("Tracking Complete", spinner, message_type= "success", duration=2)
                    st.session_state.placeholders["tracking"].empty()
                    st.session_state.placeholders["video"].empty()
                    torch.cuda.empty_cache()
                    st.rerun()
                    
if __name__ == "__main__":
    main()