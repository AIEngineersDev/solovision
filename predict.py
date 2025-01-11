# Prediction interface for Cog ⚙️
# https://cog.run/python

import cv2
import os
import torch
import tempfile
import numpy as np

from PIL import Image
from typing import Iterator, Any, List
from ultralytics import YOLO
from solovision import ByteTracker
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    MODEL_WEIGHTS = "checkpoints/yolov8m.pt"
    REID_WEIGHTS = "checkpoints/osnet_x1_0_msmt17.pt"

    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("Starting model setup...")
        device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
        print(f"Using device: {device}")
        self.model = YOLO(self.MODEL_WEIGHTS)
        self.tracker = ByteTracker(
            with_reid=True,
            reid_weights=Path(self.REID_WEIGHTS),
            device=device,
            half=False
        )

    def plot_detections(self, image: np.ndarray, tracks) -> np.ndarray:
        input_is_pil = isinstance(image, Image.Image)
        line_width = max(round(sum(image.size if input_is_pil else image.shape) / 2 * 0.001), 2)
        font_th = max(line_width - 1, 2)
        font_scale = line_width / 2
        label_padding = 7
        color = (37, 4, 11)  # BGR format

        for track in tracks:
            bbox = track[:4].astype(int)
            track_id = int(track[4])
            # Convert bbox to integers
            x_min, y_min, x_max, y_max = map(int, bbox)

            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=font_th, lineType=cv2.LINE_AA)

            # Prepare label with confidence score and ID
            label = f"{track_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=font_th)
            label_x = max(x_min, 0)
            label_y = max(y_min - label_size[1] - label_padding, 0)

            # Draw label background with padding
            cv2.rectangle(
                image,
                (label_x, label_y),
                (label_x + label_size[0] + 2 * label_padding, label_y + label_size[1] + 2 * label_padding),
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            # Draw label text
            cv2.putText(
                image,
                label,
                (label_x + label_padding, label_y + label_size[1] + label_padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness=font_th,
                lineType=cv2.LINE_AA,
            )

        return image

    def predict(
        self,
        video: Path = Input(description="Video Input"),
        conf: float = Input(
            description="Confidence Threshold", default=0.2
        ),
        iou: float = Input(
            description="NMS Threshold", default=0.75
        ),
    ) -> Path:

        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        temp_dir = tempfile.mkdtemp()

        """Run inference on the model"""
        print(f"- Confidence threshold: {conf}")
        print(f"- IOU threshold: {iou}")

        params = {
        'source': str(video),
        'conf': conf,
        'iou': iou,
        'stream': True,
        'classes': 0
    }
        # Get video properties for output
        cap = cv2.VideoCapture(str(video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        output_video_path = output_dir / "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path), 
            fourcc, 
            fps, 
            (width, height)
        )

        # model predictions
        results = self.model.predict(**params)
        frame_idx = 0
        print("[*] Processing frames...")
        for result in results:
            frame_idx += 1
            dets = result.boxes.data.cpu().numpy()
            frame = result.orig_img
            tracks = self.tracker.update(dets, frame)
            
            # Draw tracks on frame if they exist
            if len(tracks) > 0:
                frame = self.plot_detections(frame, tracks)
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
            cv2.imwrite(frame_path, frame)
            
        print("[*] Generating outputs...")
        # Generate output video
        video_name = os.path.basename(str(video))
        output_path = f"outputs/{video_name}"
        frames_pattern = os.path.join(temp_dir, "frame_%05d.png")
        ffmpeg_cmd = f"ffmpeg -y -r {fps} -i {frames_pattern} -c:v libx264 -pix_fmt yuv420p {output_path}"
        os.system(ffmpeg_cmd)

        return Path(output_path)