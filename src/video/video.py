import numpy as np
import cv2

class Video:
    def __init__(self, video: str):
        self.cap = cv2.VideoCapture(video)

        if not self.cap.isOpened():
            raise RuntimeError("Can't open video flow.")
    
    def get(self) -> np.ndarray | None:
        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame
    
    def __del__(self) -> None:
        self.cap.release()
