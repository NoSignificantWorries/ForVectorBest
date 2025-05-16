
import numpy as np

import src.detector.detector as detect
import src.image_processor.image_processor as imgproc
from src.base_worker import BaseWorker
import src.conf as conf


class Pipeline:
    def __init__(self, pipeline: list[BaseWorker]):
        self.pipeline = pipeline

    def __call__(self, image: np.ndarray) -> bool:
        pred = image
        for worker in self.pipeline:
            pred = worker(pred)
            if conf.SAVE_MODE:
                worker.visualize()
        
        return True
