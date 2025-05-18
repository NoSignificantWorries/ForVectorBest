import os
import joblib

import numpy as np
import pandas as pd
import cv2

import src.conf as conf
from src.base_worker import BaseWorker


class PATTERNClassifier(BaseWorker):
    def __init__(self, model_path: str = conf.PATTERN_CLASSIFIER_PATH):
        self.model = joblib.load(model_path)
        self.index = -1
        self.image = None
        self.predict = None
        self.data = None

        self.save_dir = os.path.join(conf.SAVE_DIR, conf.PATTERN_SAVE_DIR)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def __call__(self, data: tuple[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        self.index += 1
        if conf.DEBUG_OUTPUT:
            print("Pattern classificator called")
        df, self.image = data
        self.data = self.expand_params(df)

        predict = self.model.predict(self.data)
        self.predict = predict[0]
        self.data["good/bad"] = predict
        
        return self.data
    
    def save_call(self) -> None:
        if self.image is None:
            return
        
        height, width, _ = self.image.shape
        pad_rect = 8

        color = (0, 255, 0) if self.predict == 1 else (0, 0, 255)
        label = "Good" if self.predict == 1 else "Bad"

        cv2.rectangle(self.image, (pad_rect, pad_rect), (width - pad_rect, height - pad_rect), color, 6)

        (text_width, text_height) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 5, 8)[0]
        text_x = (width - text_width) // 2
        text_y = pad_rect + text_height + 12

        cv2.rectangle(self.image, (text_x, text_y - text_height - 15), (text_x + text_width, text_y + 15), color, -1)
        cv2.putText(self.image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 8)
        
        cv2.imwrite(os.path.join(self.save_dir, f"pattern_{self.index}.png"), self.image)
    
    def verify(self) -> bool:
        return int(self.predict) == 1
    
    def expand_params(self, df: pd.DataFrame) -> pd.DataFrame:
        importances = ["x", "y", "x1", "x2", "y1", "y2", "area", "height/width", "width/height", "crop_height", "crop_height_%", "crop_width", "crop_width_%"]
        df2 = {}
        for i in range(conf.NUM_CLUSSES):
            df2[f"class_{i}"] = []
            for elem in importances:
                df2[f"{elem}_{i}"] = []
            for j in range(conf.NUM_CLUSSES):
                if i == j:
                    continue
                df2[f"{i}_to_{j}_x"] = []
                df2[f"{i}_to_{j}_y"] = []

        for i in range(conf.NUM_CLUSSES):
            if df[df["class"] == i].shape[0] == 0:
                df2[f"class_{i}"].append(0)
                for elem in importances:
                    df2[f"{elem}_{i}"].append(0)
            else:
                df2[f"class_{i}"].append(1)
                for elem in importances:
                    df2[f"{elem}_{i}"].append(float(df[df["class"] == i][elem].iloc[0]))
            for j in range(conf.NUM_CLUSSES):
                if i == j:
                    continue
                if df[df["class"] == i].shape[0] == 0 or df[df["class"] == j].shape[0] == 0:
                    df2[f"{i}_to_{j}_x"].append(0)
                    df2[f"{i}_to_{j}_y"].append(0)
                else:
                    vec_x = float(df[df["class"] == j]["x"].iloc[0]) - float(df[df["class"] == i]["x"].iloc[0])
                    vec_y = float(df[df["class"] == j]["y"].iloc[0]) - float(df[df["class"] == i]["y"].iloc[0])
                    df2[f"{i}_to_{j}_x"].append(vec_x)
                    df2[f"{i}_to_{j}_y"].append(vec_y)

        df2 = pd.DataFrame(df2)
        return df2
