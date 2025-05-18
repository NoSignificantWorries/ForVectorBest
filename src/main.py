import os
import time
import argparse

import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import src.pipeline.pipeline as pipeline
import src.detector.detector as detector
import src.image_processor.image_processor as img_processor
import src.bbox_classifier.bbox_classifier as bbox_classifier
import src.pattern_classifier.pattern_classifier as pattern_classifier
import src.video.video as video
import src.conf as conf


workflow = [
    img_processor.Preprocessor(),
    detector.Detector(),
    bbox_classifier.BBOXClassifier(),
    pattern_classifier.PATTERNClassifier()
]


def main(params: dict) -> None:
    work_process = pipeline.Pipeline(workflow)

    match params["mode"]:
        case "image":
            image = cv2.imread(params["input"])
            res = work_process(image)
            match params["output_signal"]:
                case "GPIO":
                    pass
                case "terminal":
                    print(f"Result for {params["input"]}:", "Good" if res else "Bad")
        case "video":
            cam = video.Video(params["input"])
            frame = cam.get()
            i = 0
            while frame is not None:
                if (i + 1) % params["step"] == 0:
                    res = work_process(frame)
                    frame = cam.get()
                    match params["output_signal"]:
                        case "GPIO":
                            pass
                        case "terminal":
                            print(f"Result for frame {i + 1}:", "Good" if res else "Bad")
                i += 1
        case "cap":
            cam = video.Video(params["input"])
            frame = cam.get()
            while frame is not None:
                match params["input_signal"]:
                    case "GPIO":
                        pass
                    case "terminal":
                        while True:
                            code = input()
                            if code == "1":
                                break
                res = work_process(frame)
                frame = cam.get()
                match params["output_signal"]:
                    case "GPIO":
                        pass
                    case "terminal":
                        print(f"Result for frame {i + 1}:", "Good" if res else "Bad")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO: обучение и предсказание")
    parser.add_argument("-m", "--mode", type=str, required=True, choices=["video", "cam", "image"],
                        help="Working mode: reading from camera, image or video stream")

    parser.add_argument("-i", "--input", type=str, help="Path to input image or video stream", default="/dev/video0")
    parser.add_argument("--input_signal_type", type=str, choices=["GPIO", "terminal"], help="Signal type, only for 'cam' mode")
    parser.add_argument("--output_signal_type", type=str, choices=["GPIO", "terminal"], help="Signal type, only for 'cam' mode")
    parser.add_argument("-p", "--step", type=int, default=24, help="Frame step for video stream, only for 'video' mode")
    parser.add_argument("-d", "--debug_output", type=bool, default=False, help="Enable debug output")
    parser.add_argument("-s", "--save_mode", type=bool, default=False, help="Enable save mode")
    parser.add_argument("-o", "--output_dir", type=str, default="results", help="Path to save results of save mode")

    args = parser.parse_args()

    conf.SAVE_DIR = args.output_dir
    conf.DEBUG_OUTPUT = args.debug_output
    conf.SAVE_MODE = args.save_mode
    
    params = {
        "mode": args.mode,
        "input": args.input,
        "output_signal": args.output_signal_type
    }
    match args.mode:
        case "video":
            params["step"] = args.step
        case "cam":
            params["input_signal"] = args.input_signal_type

    main(params)
