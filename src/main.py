import os

import cv2

import src.pipeline.pipeline as pipeline
import src.detector.detector as detector
import src.image_processor.image_processor as img_processor
import src.conf as conf


workflow = [
    img_processor.Preprocessor(),
    detector.Detector()
]


def main() -> None:
    work_process = pipeline.Pipeline(workflow)
    
    image = cv2.imread("data/stickers2/Bad/1_1.png")

    work_process(image)


if __name__ == "__main__":
    conf.DEBUG_OUTPUT = True
    conf.SAVE_MODE = True

    main()
