"""Main entry point that orchestrates the navigation assistance pipeline."""

from __future__ import annotations

import logging
import sys
import time
import threading
from typing import List, Tuple

import cv2
import numpy as np

from .audio_feedback import AudioFeedback
from .config import CAMERA_CONFIG
from .detector import Detection, ObjectDetector
from .distance_estimator import DistanceEstimator
from .visualizer import draw_detections

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def _open_camera() -> cv2.VideoCapture:
    capture = cv2.VideoCapture(CAMERA_CONFIG.source_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {CAMERA_CONFIG.source_index}"
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG.frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG.frame_height)
    # Kameranın buffer'ını küçük tut → her zaman en güncel kareyi al
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _select_priority_detection(
    detections: List[Detection],
    distances: List[float],
) -> Tuple[Detection, float] | Tuple[None, None]:
    if not detections:
        return None, None
    closest_index = min(range(len(detections)), key=lambda i: distances[i])
    return detections[closest_index], distances[closest_index]


def run() -> None:
    """Run the real-time detection and feedback loop."""
    _configure_logging()

    detector = ObjectDetector()
    estimator = DistanceEstimator()
    audio = AudioFeedback()
    capture = _open_camera()

    # Inference thread ile ana thread arasında paylaşılan durum
    latest_frame: list = [None]          # kamera karesi
    latest_result: list = [None, None]   # (detections, annotated_frame)
    frame_lock = threading.Lock()
    result_lock = threading.Lock()
    stop_event = threading.Event()

    def inference_worker():
        while not stop_event.is_set():
            with frame_lock:
                frame = latest_frame[0]
            if frame is None:
                time.sleep(0.001)
                continue

            detections = detector.detect(frame)
            distances = [estimator.estimate_distance(d) for d in detections]

            priority_det, priority_dist = _select_priority_detection(detections, distances)
            if priority_det is not None:
                direction = estimator.estimate_direction(priority_det, frame.shape[1])
                audio.announce(priority_det.class_name, direction, priority_dist)

            annotated = draw_detections(frame, detections, distances)
            with result_lock:
                latest_result[0] = annotated
                latest_result[1] = detections

    worker = threading.Thread(target=inference_worker, daemon=True)
    worker.start()

    fps_timer = time.time()
    frame_count = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                logger.warning("Failed to grab a frame; exiting loop")
                break

            with frame_lock:
                latest_frame[0] = frame.copy()

            with result_lock:
                display = latest_result[0]

            # Sonuç hazır değilse ham kareyi göster
            if display is None:
                display = frame

            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame_count = 0
                fps_timer = time.time()

            cv2.imshow("Navigation Assistance - press q to quit", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        capture.release()
        cv2.destroyAllWindows()
        audio.shutdown()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(0)
