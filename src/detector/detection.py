import os
from pathlib import Path

import cv2
from ultralytics import YOLO

# custom modules
import src.conf as conf


class Detector:
    def __init__(self, model_type: str = conf.DETECTOR_PARAMS["model_type"]):
        self.model = YOLO(model_type)
        
    def train(self, data: str, params: dict = conf.DETECTOR_PARAMS) -> None:      
        self.model.train(
            data=data,
            project=params["project"],
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            freeze=params["freeze"],
            batch=params["batch"],
            save=params["save"],
            plots=params["plots"],
            optimizer=params["optimizer"],
            save_period=params["save_period"],
            val=params["val"],
            patience=params["patience"],
            warmup_epochs=params["warmup_epochs"],
            degrees=params["degrees"],
            multi_scale=params["multi_scale"],
            mosaic=params["mosaic"],
            flipud=params["flipud"],
            fliplr=params["flipdir"],
            device=params["device"],
        )
        # model.save("last.pt")


    def predict(self, source: str, save_dir: str, params: dict = conf.DETECTOR_PARAMS) -> None:
        results = self.model.predict(source, imgsz=params["imgsz"], conf=params["conf"], iou=params["iou"])

        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, r in enumerate(results):
            im_bgr = r.plot()
            image_path = output_path / f"test_result_{i}.jpg"
            cv2.imwrite(str(image_path), im_bgr)
