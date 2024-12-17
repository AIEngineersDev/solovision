import argparse
import cv2
import numpy as np
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


checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/zeeshaan28/solo-ultralytics.git', ))  # install


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

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
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    
    ul_models = ['yolov3', 'yolov5', 'yolov8', 'yolov9', 'yolov10', 'yolo11', 'rtdetr', 'sam']

    yolo = YOLO(
        args.yolo_model if any(yolo in str(args.yolo_model) for yolo in ul_models) else 'yolov8n.pt',
    )

    if args.command == 'track':

        results = yolo.track(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            ext_track = True,
            agnostic_nms=args.agnostic_nms,
            show=args.show,
            stream=True,
            save_tracks =args.save_tracks,
            save_crop= args.save_crops,
            device=args.device,
            show_conf=args.show_conf,
            save_txt=args.save_txt,
            show_labels=args.show_labels,
            save=args.save,
            verbose=args.verbose,
            exist_ok=args.exist_ok,
            project=args.project,
            name=args.name,
            classes=args.classes,
            imgsz=args.imgsz,
            vid_stride=args.vid_stride,
            line_width=args.line_width
        )

        
        yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))
        
    elif args.command =='detect':
         results = yolo.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            ext_track = False,
            agnostic_nms=args.agnostic_nms,
            show=args.show,
            stream=True,
            save_tracks =False,
            save_crop= args.save_crops,
            device=args.device,
            show_conf=args.show_conf,
            save_txt=args.save_txt,
            show_labels=args.show_labels,
            save=args.save,
            verbose=args.verbose,
            exist_ok=args.exist_ok,
            project=args.project,
            name=args.name,
            classes=args.classes,
            imgsz=args.imgsz,
            vid_stride=args.vid_stride,
            line_width=args.line_width
        )

    if not any(yolo in str(args.yolo_model) for yolo in ul_models):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args

    if args.command == "track" and args.plot:
        save_dir = get_save_dir(args)
        video_writer = tracking_plot(output_path=save_dir, init_only=True)
        x_data, y_data = [], []  # Initialize data lists
        frame_count = 0

    for result in results:
        frame_count += 1
        plotted_frame = result.plot()

        if args.command == "track" and args.plot:
            graph_image, x_data, y_data = tracking_plot(
                result=result, x_data=x_data, y_data=y_data,
                video_writer=video_writer, frame=frame_count
            )
            if args.show is False:
                cv2.imshow("Solovision Analytics", graph_image)
                if cv2.waitKey(1) & 0xFF in (ord(' '), ord('q')):
                    break

        # If stream_display callback is provided, use it
        if hasattr(args, 'stream_display') and args.stream_display is not None:
            args.stream_display(plotted_frame)
        
        # if args.show is False and :
        #     cv2.imshow("Solovision Analytics", graph_image)
        #     if cv2.waitKey(1) & 0xFF in (ord(' '), ord('q')):
        #         break
    
    # Release video writer after processing all frames
    if video_writer:
        video_writer.release()


