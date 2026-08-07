from pathlib import Path
import cv2

def inspect_video(path: Path, sample_every_sec: float = 2.0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Unable to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 else 0

    samples = []
    if duration > 0:
        t = 0.0
        while t <= duration:
            samples.append(round(t, 3))
            t += sample_every_sec

    cap.release()
    return round(duration, 3), samples
