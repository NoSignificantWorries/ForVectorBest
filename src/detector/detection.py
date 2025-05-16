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


    # def predict(self, source: str, save_dir: str, params: dict = conf.DETECTOR_PARAMS) -> None:
    #     results = self.model.predict(source, imgsz=params["imgsz"], conf=params["conf"], iou=params["iou"])

    #     output_path = Path(save_dir)
    #     output_path.mkdir(parents=True, exist_ok=True)

    #     for i, r in enumerate(results):
    #         im_bgr = r.plot()
    #         image_path = output_path / f"test_result_{i}.jpg"
    #         cv2.imwrite(str(image_path), im_bgr)

    def predict(self, source: str, save_dir: str, params: dict = conf.DETECTOR_PARAMS, allowed_classes: list = None):
        results = self.model.predict(source, imgsz=params["imgsz"], conf=params["conf"], iou=params["iou"])

        output_path = Path(save_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        predictions = []

        for i, r in enumerate(results):
            im_bgr = r.plot()
            image_path = output_path / f"test_result_{i}.jpg"
            cv2.imwrite(str(image_path), im_bgr)

            image_results = {
                "image_path": r.path,
                "text_boxes": [],
                "object_boxes": []
            }

            for box in r.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf_score, class_id = box[0], box[1], box[2], box[3], box[4], int(box[5])
                if allowed_classes is not None and class_id not in allowed_classes:
                    continue

                box_info = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(conf_score),
                    "class": class_id
                }

                if class_id == 1:
                    image_results["text_boxes"].append(box_info)
                else:
                    image_results["object_boxes"].append(box_info)
                    pass

            predictions.append(image_results)

        return predictions