import time

import numpy as np

import src.detector.detector as detect
import src.image_processor.image_processor as imgproc
from src.base_worker import BaseWorker
import src.conf as conf


class Pipeline:
    def __init__(self, pipeline: list[BaseWorker]):
        self.pipeline = pipeline

    def __call__(self, image: np.ndarray) -> bool:
        total_time = 0.0
        pred = image
        for i, worker in enumerate(self.pipeline):
            start = time.time()
            pred = worker(pred)
            end = time.time()
            if conf.DEBUG_OUTPUT:
                print(i, end - start, sep=" | ")
                total_time += end - start
            if conf.SAVE_MODE:
                worker.visualize()
        if conf.DEBUG_OUTPUT:
            print("Total time:", total_time)
        
        return True
