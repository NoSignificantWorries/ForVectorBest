import joblib

import src.conf as conf
from src.base_worker import BaseWorker


class BBOXClassifier(BaseWorker):
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
