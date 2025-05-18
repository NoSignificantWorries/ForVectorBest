import os
import time

import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import src.pipeline.pipeline as pipeline
import src.detector.detector as detector
import src.image_processor.image_processor as img_processor
import src.bbox_classifier.bbox_classifier as bbox_classifier
import src.pattern_classifier.pattern_classifier as pattern_classifier
import src.conf as conf


workflow = [
    img_processor.Preprocessor(),
    detector.Detector(),
    bbox_classifier.BBOXClassifier(),
    pattern_classifier.PATTERNClassifier()
]


def main() -> None:
    work_process = pipeline.Pipeline(workflow)

    image = cv2.imread("local_data/stickers2/Bad/1_1.png")
    print(work_process(image))
    

if __name__ == "__main__":
    conf.DEBUG_OUTPUT = True
    # conf.SAVE_MODE = True

    main()
