# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from typing import Iterator
from ultralytics import YOLO
from solovision import ByteTracker
import torch


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = YOLO("./yolov8m.pt")
        self.tracker = ByteTracker(
        with_reid= True,
        reid_weights=Path("./osnet_x1_0_msmt17.pt"),
        device=torch.device,  
        half=False,
    )


    def predict(
        self,
        video: Path = Input(description="Video Input"),
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

        for result in results:
            dets = result.boxes.data.cpu().numpy()
            frame = result.orig_img

            tracks = self.tracker.update(dets, frame)
            if len(tracks) == 0:
                continue
            idx = tracks[:, -1].astype(int)
            predictor.results[i] = predictor.results[i][idx]

            update_args = {"obb" if is_obb else "boxes": torch.as_tensor(tracks[:, :-1])}
            predictor.results[i].update(**update_args)
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
