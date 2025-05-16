
DETECTOR_WEIGHTS_PATH = "resources/weights/detector.pt"

BASE_DATASET_PATH = "data/stickers2"
DETECTOR_DATASET_PATH = "data/YOLO_bbox_dataset"

DEBUG_OUTPUT = False
SAVE_MODE = False
SAVE_DIR = "results"
DETECTION_SAVE_DIR = "detection"
PREPROCESS_SAVE_DIR = "preprocess"

DETECTION_PROJECT = "runs/detection"


DETECTOR_PARAMS = {
    "model_type": "yolov8n.pt",
    "project": DETECTION_PROJECT,
    "epochs": 150,
    "batch": 2,
    "imgsz": 640,
    "freeze": 10,
    "save": True,
    "plots": True,
    "optimizer": "SGD",
    "save_period": 5,
    "val": True,
    "patience": 150,
    "warmup_epochs": 5,
    "degrees": 10,
    "multi_scale": True,
    "mosaic": 1.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "device": 0,
    "conf": 0.5,
    "iou": 0.5
}
