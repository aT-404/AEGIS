import cv2
from typing import Generator, Tuple


class StreamReader:
    """Video stream reader supporting files, webcam devices, and RTSP streams."""

    def __init__(self, source: str):
        self.source = source
        if isinstance(source, str) and source.isdigit():
            self._cap = cv2.VideoCapture(int(source))
        else:
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

    @property
    def fps(self) -> float:
        val = self._cap.get(cv2.CAP_PROP_FPS)
        return float(val) if val and val > 0 else 30.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    @property
    def total_frames(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self) -> Generator:
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self):
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
