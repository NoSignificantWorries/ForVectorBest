import joblib

import src.conf as conf
from src.base_worker import BaseWorker


class PATTERNClassifier(BaseWorker):
    def __init__(self, model_path: str = conf.PATTERN_CLASSIFIER_PATH):
        self.model = joblib.load(model_path)
    
    def __call__(self) -> None:
        return None
    
    def visualize(self) -> None:
        return None
