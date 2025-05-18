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

BBOX_CLASSES = [
    "Product", 
    "Exclamation mark",
    "IVD",
    "LOT",
    "Temperature",
    "Hourglass",
    "Production date",
    "Serial number",
    "Company title",
    "Volume"
]
CLASS_COLORS = [
    (255, 0, 0),
    (200, 50, 0),
    (150, 100, 0),
    (100, 150, 0),
    (50, 200, 0),
    (0, 255, 0),
    (0, 200, 50),
    (0, 150, 100),
    (0, 100, 150),
    (0, 0, 255)
]
NUM_CLUSSES = len(BBOX_CLASSES)

BBOX_CLASSIFIER_PATH = "resources/weights/bbox_classifier.pkl"
PATTERN_CLASSIFIER_PATH = "resources/weights/pattern_classifier.pkl"

BBOX_SAVE_DIR = "bbox_classifier"
PATTERN_SAVE_DIR = "pattern_classifier"
