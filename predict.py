# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from typing import Iterator
from ultralytics import YOLO
from solovision import ByteTracker


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = YOLO("./yolov8m.pt")
        self.tracker = ByteTracker()

    def predict(
        self,
        video: Path = Input(description="Video Input),
        conf: float = Input(
            description="Confidence Threshold", default=0.2
        ),
        iou: float = Input(
            description="NMS Threshold", default=0.75
        ),
    ) -> Iterator[Path]:
        """Run inference on the model"""
        params = {
        'source': str(video),
        'conf': conf,
        'iou': iou,
        'stream': True,
    }
        results = self.model.detect(**params)
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
