
FILTERS_SAVE_DIR = "filters"

SEGMENTATION_WEIGHTS_PATH = "resources/weights/segmentation.pt"
SEGMENTATION_DATASET_PATH = "data/YOLO_seg_dataset"
SEGMENTATION_SAVE_DIR = "segmentation"
SEGMENTATION_COLOR = (0, 255, 0)
SEGMENTATION_PROJECT = "runs/segmentation"
SEGMENT_PARAMS = {
    "epochs": 30,
    "imgsz": 640,
    "project": SEGMENTATION_PROJECT,
    "name": "train_segmentation_custom",
    "save": True,
    "show": False,
    "hide_labels": True,
    "hide_conf": True,
    "line_thickness": 0
}

