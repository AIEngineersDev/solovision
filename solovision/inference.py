
import cv2
from functools import partial
from pathlib import Path

import torch
from ultralytics.cfg import get_save_dir
from solovision.tracker_zoo import create_tracker
from solovision.utils import ROOT, WEIGHTS, TRACKER_CONFIGS
from solovision.utils.checks import RequirementsChecker
from solovision.detectors import get_yolo_inferer
from ultralytics import YOLO
from solovision.post_processing import tracking_plot

# Check requirements
checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/AIEngineersDev/solo-ultralytics.git', )) 

def ultralytics_model(model, ul_models):
    """Check if the given model belongs to the list of supported ultralytics models."""
    return any(yolo in str(model) for yolo in ul_models)

def on_predict_start(predictor, persist=False):
    """Initialize trackers for object tracking during prediction."""
    tracking_config = TRACKER_CONFIGS / 'bytetrack.yaml'
    trackers = []

    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            tracking_config,
            predictor.custom_args.with_reid,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers

def display_frame(frame, window_name="Solovision Analytics"):
    """Display a single frame using OpenCV."""
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF in (ord(' '), ord('q'))

@torch.no_grad()
def run(args):
    ul_models = ['yolov3', 'yolov5', 'yolov8', 'yolov9', 'yolov10', 'yolo11', 'rtdetr', 'sam']

    # Initialize YOLO model
    yolo = YOLO(args.yolo_model if ultralytics_model(args.yolo_model, ul_models) else 'yolov8n.pt')

    # Common parameters for prediction and tracking
    params = {
        'source': args.source,
        'conf': args.conf,
        'iou': args.iou,
        'agnostic_nms': args.agnostic_nms,
        'show': args.show,
        'stream': True,
        'save_crop': args.save_crops,
        'device': args.device,
        'show_conf': args.show_conf,
        'save_txt': args.save_txt,
        'show_labels': args.show_labels,
        'save': args.save,
        'verbose': args.verbose,
        'exist_ok': args.exist_ok,
        'project': args.project,
        'name': args.name,
        'classes': args.classes,
        'imgsz': args.imgsz,
        'vid_stride': args.vid_stride,
        'line_width': args.line_width,
    }

    if args.command == 'track':
        results = yolo.track(**params, ext_track=True, save_tracks=args.save_tracks)
        yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))
    elif args.command == 'detect':
        results = yolo.predict(**params)

    if not ultralytics_model(args.yolo_model, ul_models):
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    yolo.predictor.custom_args = args

    # Initialize video writer if plotting is enabled
    if args.plot:
        save_dir = get_save_dir(args)
        video_writer = tracking_plot(output_path=save_dir, init_only=True)
        x_data, y_data = [], []
        frame_count = 0

    for result in results:

        if args.command == "track" and args.plot:
            frame_count += 1
            graph_image, x_data, y_data = tracking_plot(
                result=result, x_data=x_data, y_data=y_data,
                video_writer=video_writer, frame=frame_count
            )
            cv2.imshow("Solovision Analytics", graph_image)
            if cv2.waitKey(1) & 0xFF in (ord(' '), ord('q')):
                break

        if hasattr(args, 'stream_display') and args.stream_display is not None:
            plotted_frame = result.plot()
            args.stream_display(plotted_frame)

    if args.plot and video_writer:
        video_writer.release()