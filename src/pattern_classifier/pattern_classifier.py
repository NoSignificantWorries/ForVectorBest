import joblib

import pandas as pd

import src.conf as conf
from src.base_worker import BaseWorker


class PATTERNClassifier(BaseWorker):
    def __init__(self, model_path: str = conf.PATTERN_CLASSIFIER_PATH):
        self.model = joblib.load(model_path)
    
    def __call__(self, df: pd.DataFrame) -> None:
        self.data = self.expand_params(df)

        predict = self.model.predict(self.data)
        self.data["good/bad"] = predict
        
        return self.data
    
    def visualize(self) -> None:
        return None
    
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
