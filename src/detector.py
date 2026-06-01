"""YOLO26-based object detection module.

This module wraps the Ultralytics YOLO26 inference API behind a small,
domain-specific interface.  YOLO26 features an NMS-free end-to-end
architecture that eliminates the non-maximum suppression bottleneck,
resulting in lower latency compared to YOLOv8.  The rest of the system
interacts with plain Python data classes instead of raw framework
tensors, which keeps the coupling to the deep learning library
contained in this single file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from .config import MODEL_CONFIG, ModelConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Structured representation of a single detected object.

    Attributes:
        class_name: Human-readable class label predicted by YOLO26.
        confidence: Confidence score in the range ``[0.0, 1.0]``.
        x1: Left coordinate of the bounding box, in pixels.
        y1: Top coordinate of the bounding box, in pixels.
        x2: Right coordinate of the bounding box, in pixels.
        y2: Bottom coordinate of the bounding box, in pixels.
    """

    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def box_height(self) -> int:
        """Return the bounding-box height in pixels."""
        return self.y2 - self.y1

    @property
    def box_width(self) -> int:
        """Return the bounding-box width in pixels."""
        return self.x2 - self.x1

    @property
    def center_x(self) -> float:
        """Return the horizontal center of the bounding box."""
        return (self.x1 + self.x2) / 2.0


class ObjectDetector:
    """Thin wrapper around a pretrained YOLO26 model.

    The class is responsible for loading the weights once, running inference on
    individual frames, and converting the framework output into a list of
    :class:`Detection` instances.  YOLO26 eliminates the NMS post-processing
    step, yielding faster inference on both CPU and GPU.
    """

    def __init__(self, config: ModelConfig = MODEL_CONFIG) -> None:
        """Load the YOLO26 model and prepare it for inference.

        Args:
            config: Model configuration. Defaults to the singleton defined in
                :mod:`config`.
        """
        self._config = config
        logger.info("Loading YOLO26 weights from %s", config.weights_path)
        self._model = YOLO(config.weights_path)
        self._class_names = self._model.names

        # Warm-up: one dummy forward pass eliminates first-frame latency spike.
        dummy = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
        self._model.predict(source=dummy, device=config.device,
                            imgsz=config.imgsz, half=config.half,
                            verbose=False)
        logger.info("Detector ready on device '%s' (half=%s, imgsz=%d)",
                    config.device, config.half, config.imgsz)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run object detection on a single BGR frame.

        Args:
            frame: Image array as produced by OpenCV ``VideoCapture``. Expected
                shape is ``(H, W, 3)`` with BGR channel ordering.

        Returns:
            A list of :class:`Detection` objects that survived confidence and
            IoU filtering. The list is empty when nothing is detected.
        """
        results = self._model.predict(
            source=frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            device=self._config.device,
            imgsz=self._config.imgsz,
            half=self._config.half,
            verbose=False,
        )

        detections: List[Detection] = []
        # Ultralytics returns a list even for a single input image, so index 0.
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Move tensors to CPU once to avoid repeated host/device transfers.
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confidences, class_ids):
            detections.append(
                Detection(
                    class_name=self._class_names[cls_id],
                    confidence=float(conf),
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                )
            )

        return detections
