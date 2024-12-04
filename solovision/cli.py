import argparse
import subprocess
from pathlib import Path
from solovision.utils import WEIGHTS, ROOT
from solovision.track import run


def track_command(args):
    """Runs the tracking using cli."""
    run(args)
    

def inference_command(args):
    """Run the Solovision app for inference."""
    # Use subprocess to launch the app with the required arguments
    streamlit_path = ROOT / "solovision_app"/ "main.py"
    subprocess.run(["streamlit", "run", str(streamlit_path)])

def main():
    # Main parser for the CLI
    parser = argparse.ArgumentParser(
        description="Solo Vision CLI: Command-line tool for object tracking and inference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Sub-command: Tracking
    parser_track = subparsers.add_parser("track", help="Run object tracking using cli")
    
    # Required/Important arguments
    parser_track.add_argument('--yolo-model', type=str, default=str(WEIGHTS / "yolov8n.pt"), help="Path to YOLO model")
    parser_track.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x1_0_msmt17.pt', help="Path to ReID model")
    parser_track.add_argument('--source', type=str, default='0', help="Input source (video file, webcam ID, or stream URL)")

    # Tracking configuration
    parser_track.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (0.0 to 1.0)")
    parser_track.add_argument('--iou', type=float, default=0.7, help="IoU threshold for Non-Max Suppression (NMS)")
    parser_track.add_argument('--classes', nargs='+', type=int, help="Filter by class IDs (e.g., --classes 0 1 2)")

    # Optional features
    parser_track.add_argument(
    '--imgsz', '--img', '--img-size', nargs='+', type=int, 
    default=[640], help="Inference size as (height, width). Default is [640].")
    parser_track.add_argument('--show', action='store_true', help="Display tracking video results during inference")
    parser_track.add_argument('--save', action='store_true', help="Save tracking video results")
    parser_track.add_argument('--save-txt', action='store_true', help="Save tracking results in a text file")
    parser_track.add_argument('--save-crops', action='store_true', help="Save detected object crops to individual folders")
    parser_track.add_argument('--project', default='runs/track', help="Project directory for saving results")
    parser_track.add_argument('--name', default='exp', help="Experiment name for saving results")
    parser_track.add_argument('--show-labels', action='store_false', help="Display labels on tracked objects")
    parser_track.add_argument('--show-conf', action='store_false', help="Display confidence scores on tracked objects")
    parser_track.add_argument('--show-trajectories', action='store_true', help="Display object trajectories during tracking")
    
    # Hardware options
    parser_track.add_argument('--device', default='', help="Specify device for inference (e.g., 'cuda:0', 'cpu')")
    parser_track.add_argument('--half', action='store_true', help="Use FP16 half-precision for inference")
    parser_track.add_argument('--vid-stride', type=int, default=1, help="Frame stride for video input")

    # Advanced options
    parser_track.add_argument('--with-reid', action='store_true', default=False, help="Use ReID features for tracking association")
    parser_track.add_argument('--line-width', type=int, default=None, help="Line width for bounding boxes (auto-scaled if None)")
    parser_track.add_argument('--agnostic-nms', action='store_true', help="Perform class-agnostic NMS")
    parser_track.add_argument('--per-class', action='store_true', help="Do not mix up classes when tracking")
    parser_track.add_argument('--verbose', action='store_true', default=True, help="Print results for each frame")
    parser_track.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # Sub-command: Inference
    parser_inference = subparsers.add_parser("inference", help="Run the Solovision web app for object detection and tracking inference")
    # (Additional Streamlit-specific arguments can be added if required in the future)

    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "track":
        track_command(args)
    elif args.command == "inference":
        inference_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
