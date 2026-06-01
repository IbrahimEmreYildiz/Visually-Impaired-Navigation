"""Monocular distance estimation module.

A single RGB camera does not directly provide depth information. This module
recovers an approximate distance by combining the pixel height of a detected
object with a prior on its real-world height and the calibrated focal length
of the camera.

The underlying relation is the pinhole camera model:

    distance = (real_height * focal_length) / pixel_height

where every quantity is expressed in consistent units (meters for real height,
meters for the resulting distance, and pixels for both the focal length and
the measured box height).
"""

from __future__ import annotations

import logging

from .config import DIRECTION_CONFIG, DISTANCE_CONFIG, DirectionConfig, DistanceConfig
from .detector import Detection

logger = logging.getLogger(__name__)


class DistanceEstimator:
    """Estimate object distance and horizontal direction from a bounding box.

    The class intentionally exposes two separate concerns behind a single
    interface because both the distance value and the left/center/right
    qualifier are required together to produce a meaningful audio message.
    """

    def __init__(
        self,
        distance_config: DistanceConfig = DISTANCE_CONFIG,
        direction_config: DirectionConfig = DIRECTION_CONFIG,
    ) -> None:
        """Store calibration parameters and direction thresholds.

        Args:
            distance_config: Parameters controlling the pinhole model.
            direction_config: Thresholds used for horizontal zone partitioning.
        """
        self._distance_config = distance_config
        self._direction_config = direction_config

    def estimate_distance(self, detection: Detection) -> float:
        """Return the distance in meters between the camera and the object.

        Args:
            detection: Detection for which the distance should be computed.

        Returns:
            Distance in meters. A sentinel value of ``float('inf')`` is
            returned when the bounding-box height is non-positive, which
            would otherwise cause a division by zero.
        """
        pixel_height = detection.box_height
        if pixel_height <= 0:
            logger.warning(
                "Non-positive bounding-box height for class '%s'",
                detection.class_name,
            )
            return float("inf")

        real_height = self._distance_config.known_heights_meters.get(
            detection.class_name,
            self._distance_config.default_height_meters,
        )

        distance = (real_height * self._distance_config.focal_length_pixels) / pixel_height
        return distance

    def estimate_direction(self, detection: Detection, frame_width: int) -> str:
        """Return the horizontal direction label of an object.

        The frame is partitioned into three vertical zones controlled by the
        boundaries in :class:`DirectionConfig`. The zone containing the
        bounding-box center determines the label.

        Args:
            detection: Detection whose direction should be labeled.
            frame_width: Width of the source image in pixels. Required to
                normalize the bounding-box center position.

        Returns:
            One of ``"left"``, ``"ahead"``, or ``"right"``.
        """
        if frame_width <= 0:
            return "ahead"

        normalized_center = detection.center_x / frame_width
        if normalized_center < self._direction_config.left_boundary:
            return "left"
        if normalized_center > self._direction_config.right_boundary:
            return "right"
        return "ahead"
