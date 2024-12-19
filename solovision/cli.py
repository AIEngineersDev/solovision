import argparse
import subprocess
from pathlib import Path
from solovision.utils import WEIGHTS, ROOT
from solovision.inference import run
from solovision.utils import logger as LOGGER

def inference_cli(args):
    """Runs the detection and tracking inference using cli."""
    print("🚀 Starting Solovision Inference ...")
    run(args)
    
def inference_app():
    """Run the Solovision app for inference."""
    # Use subprocess to launch the app with the required arguments
    print("🚀 Starting Solovision Web Inference ...")
    streamlit_path = ROOT / "solovision_app"/ "main.py"
    subprocess.run(["streamlit", "run", str(streamlit_path)])

def add_common_arguments(parser):
    """Add common arguments for track and detect commands."""
    parser.add_argument('--yolo-model', type=str, default=str(WEIGHTS / "yolov8n.pt"), help="Path to YOLO model")
    parser.add_argument('--source', type=str, default='0', help="Input source (video file, webcam ID, or stream URL)")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument('--iou', type=float, default=0.7, help="IoU threshold for Non-Max Suppression (NMS)")
    parser.add_argument('--classes', nargs='+', type=int, help="Filter by class IDs (e.g., --classes 0 1 2)")
    parser.add_argument(
    '--imgsz', '--img', '--img-size', nargs='+', type=int, 
    default=[640], help="Inference size as (height, width). Default is [640].")
    parser.add_argument('--show', action='store_true', help="Display tracking video results during inference")
    parser.add_argument('--save', action='store_true', help="Save tracking video results")
    parser.add_argument('--save-txt', action='store_true', help="Save tracking results in a text file")
    parser.add_argument('--save-crops', action='store_true', help="Save detected object crops to individual folders")
    parser.add_argument('--project', default='runs/track', help="Project directory for saving results")
    parser.add_argument('--name', default='exp', help="Experiment name for saving results")
    parser.add_argument('--show-labels', action='store_false', help="Display labels on tracked objects")
    parser.add_argument('--show-conf', action='store_false', help="Display confidence scores on tracked objects")
    parser.add_argument('--show-trajectories', action='store_true', help="Display object trajectories during tracking") 
    parser.add_argument('--plot', action='store_true', default=False, help="Plot a line graph showing the Track_Id Count per frame")
    
    # Hardware ptions
    parser.add_argument('--device', default='', help="Specify device for inference (e.g., 'cuda:0', 'cpu')")
    parser.add_argument('--vid-stride', type=int, default=1, help="Frame stride for video in")
    
    # Advanced aptions
    parser.add_argument('--line-width', type=int, default=None, help="Line width for bounding boxes (auto-scaled if None)")
    parser.add_argument('--agnostic-nms', action='store_true', help="Perform class-agnostic NMS")
    parser.add_argument('--verbose', action='store_true', default=True, help="Print results for each frame")
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    

def main():
    # Main parser for the CLI
    parser = argparse.ArgumentParser(
        description="Solo Vision CLI: Command-line tool for object detection, tracking and web app inference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Sub-command: Web Inference
    parser_inference = subparsers.add_parser("run_app", help="Run the Solovision web app for object detection and tracking inference")

    # Sub-command: Detection
    parser_detect = subparsers.add_parser("detect", help="Run object detection using cli")
    add_common_arguments(parser_detect)  # Add common arguments

    # Sub-command: Tracking
    parser_track = subparsers.add_parser("track", help="Run object tracking using cli")
    # Tracking arguments
    parser_track.add_argument('--with-reid', action='store_true', default=False, help="Use ReID features for tracking association")
    parser_track.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x1_0_msmt17.pt', help="Path to ReID model")
    parser_track.add_argument('--half', action='store_true', help="Use FP16 half-precision for inference")
    parser_track.add_argument('--per-class', action='store_true', help="Do not mix up classes when tracking")
    parser_track.add_argument('--save-tracks', action='store_true', default=False, help="Save detected track_ids to the project folder")
    add_common_arguments(parser_track)  # Add common arguments
    # Parse arguments
    args = parser.parse_args()

     
    # Execute the appropriate command
    if args.command == "track" or args.command == "detect":
        # Validation: Ensure --show and --plot are not both true
        if args.show and args.plot:
            parser.error("The arguments --show and --plot cannot be used together. Choose one.")
        if args.command == "detect" and args.plot:
            parser.error("The plot feature is only supported with track command, Trying using solovision track.")
        inference_cli(args)
    elif args.command == "run_app":
        inference_app()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
