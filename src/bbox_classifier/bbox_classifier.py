import joblib

import ultralytics as ult
import numpy as np
import pandas as pd

import src.conf as conf
from src.base_worker import BaseWorker


class BBOXClassifier(BaseWorker):
    def __init__(self, model_path: str = conf.BBOX_CLASSIFIER_PATH):
        self.model = joblib.load(model_path)
        self.boxes = None
        self.data = None
        self.height = 0
        self.width = 0
    
    def __call__(self, detection: ult.engine.results.Results) -> pd.DataFrame:
        self.data = None
        self.height, self.width = detection.orig_shape
        self.boxes = detection.boxes.data.cpu().numpy()
        
        self.data = {
            "x1": [],
            "y1": [],
            "x2": [],
            "y2": [],
            "crop_width": [],
            "crop_width_%": [],
            "crop_height": [],
            "crop_height_%": [],
            "x": [],
            "y": [],
            "area": [],
            "width/height": [],
            "height/width": []
        }
        for box in self.boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            self.expand_boxes_data(x1, y1, x2, y2)
        
        self.data = pd.DataFrame(self.data)
        
        predict = self.model.predict(self.data)

        self.data["class"] = predict
        
        return self.data
    
    def visualize(self) -> None:
        return None
    
    def expand_boxes_data(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self.data is None:
            return

        center_x = self.width / 2
        center_y = self.height / 2
        # left-up point
        self.data["x1"].append(x1 - center_x)
        self.data["y1"].append(y1 - center_y)
        # right-down point
        self.data["x2"].append(x2 - center_x)
        self.data["y2"].append(y2 - center_y)
        # width
        self.data["crop_width"].append(x2 - x1)
        # width %
        self.data["crop_width_%"].append(self.data["crop_width"][-1] / self.width)
        # height
        self.data["crop_height"].append(y2 - y1)
        # height %
        self.data["crop_height_%"].append(self.data["crop_height"][-1] / self.height)
        # center point
        self.data["x"].append((self.data["x1"][-1] + self.data["x2"][-1]) / 2)
        self.data["y"].append((self.data["y1"][-1] + self.data["y2"][-1]) / 2)
        # area
        self.data["area"].append(self.data["crop_height"][-1] * self.data["crop_width"][-1])
        # width/height
        self.data["width/height"].append(self.data["crop_width"][-1] / self.data["crop_height"][-1])
        # height/width
        self.data["height/width"].append(self.data["crop_height"][-1] / self.data["crop_width"][-1])
