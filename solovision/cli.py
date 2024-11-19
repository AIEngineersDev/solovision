import argparse
import subprocess
from pathlib import Path
from solovision.utils import ROOT,  WEIGHTS
from solovision.track import run


def track_command(args):
    """Run the tracking functionality using track.py."""
    run(args)
    

# def inference_command(args):
#     """Run the Streamlit app for inference."""
#     # Use subprocess to launch the Streamlit app with the required arguments
#     subprocess.run(["streamlit", "run", "solovision_app.py"])

def main():
    parser = argparse.ArgumentParser(description="Solo Vision CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Sub-command for tracking
    parser_track = subparsers.add_parser("track", help="Run object tracking")
    parser_track.add_argument('--yolo-model', type=str, default="yolov8n.pt", help='YOLO model path')
    parser_track.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x1_0_msmt17.pt', help='reid model path')
    parser_track.add_argument('--source', type=str, default='0', help='Source for video input')
    parser_track.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser_track.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser_track.add_argument('--iou', type=float, default=0.7, help='IoU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser_track.add_argument('--show', action='store_true', help='display tracking video results')
    parser_track.add_argument('--save', action='store_true', help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser_track.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser_track.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser_track.add_argument('--name', default='exp', help='save results to project/name')
    parser_track.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser_track.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser_track.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser_track.add_argument('--show-labels', action='store_false', help='either show all or only bboxes')
    parser_track.add_argument('--show-conf', action='store_false', help='hide confidences when show')
    parser_track.add_argument('--show-trajectories', action='store_true', help='show confidences')
    parser_track.add_argument('--save-txt', action='store_true', help='save tracking results in a txt file')
    parser_track.add_argument('--save-crops', action='store_true', help='save each crop to its respective id folder')
    parser_track.add_argument('--line-width', default=None, type=int, 
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser_track.add_argument('--per-class', default=False, action='store_true', help='not mix up classes when tracking')
    parser_track.add_argument('--verbose', default=True, action='store_true', help='print results per frame')
    parser_track.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')

    # Sub-command for inference
    parser_inference = subparsers.add_parser("run_inference", help="Run Streamlit app for inference")
    # Additional arguments can be added if needed for Streamlit

    args = parser.parse_args()

    # Decide which command to run
    if args.command == "track":
        track_command(args)
    # elif args.command == "run_inference":
    #     inference_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
