"""Debug visualization helpers.

During development it is useful to confirm visually that the detector and
the distance estimator produce sensible outputs. This module overlays bounding
boxes, class labels, and distance annotations on a copy of the input frame.
The final product is intended for visually impaired users and therefore does
not rely on the rendered frame, but the overlay is invaluable while iterating
on calibration and thresholds.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from .detector import Detection


# Colors are specified in BGR to match the OpenCV convention.
_BOX_COLOR = (0, 255, 0)
_TEXT_COLOR = (0, 255, 0)
_TEXT_BACKGROUND_COLOR = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    distances_meters: List[float],
) -> np.ndarray:
    """Return a copy of ``frame`` annotated with detection metadata.

    Args:
        frame: Source BGR image.
        detections: Detections produced for the frame.
        distances_meters: Distance estimates aligned with ``detections``.

    Returns:
        A new image array containing the original frame overlaid with a
        bounding box, class name, confidence score, and distance label for
        each detection.
    """
    annotated = frame.copy()

    for detection, distance in zip(detections, distances_meters):
        cv2.rectangle(
            annotated,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            _BOX_COLOR,
            _BOX_THICKNESS,
        )

        label = f"{detection.class_name} {detection.confidence:.2f} | {distance:.1f}m"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, _FONT, _FONT_SCALE, _FONT_THICKNESS
        )
        # Place the label above the box. When the box is near the top of the
        # frame, fall back to placing the label inside the box.
        label_y = detection.y1 - 6 if detection.y1 - 6 > text_height else detection.y1 + text_height + 6

        cv2.rectangle(
            annotated,
            (detection.x1, label_y - text_height - baseline),
            (detection.x1 + text_width, label_y + baseline),
            _TEXT_BACKGROUND_COLOR,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (detection.x1, label_y),
            _FONT,
            _FONT_SCALE,
            _TEXT_COLOR,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated
