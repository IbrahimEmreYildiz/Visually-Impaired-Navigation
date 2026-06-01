"""Centralized configuration module for the navigation assistance system.

This module stores all tunable parameters and constants used across the
detection, distance estimation, and audio feedback components. Keeping these
values in a single location avoids magic numbers scattered throughout the code
base and makes experimentation safer.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the YOLO26 detector.

    Attributes:
        weights_path: Path or identifier of the pretrained YOLO26 weights.
        confidence_threshold: Minimum confidence score for a detection to be
            accepted. Predictions below this value are discarded as unreliable.
        iou_threshold: Intersection-over-Union threshold. YOLO26 uses an
            end-to-end (NMS-free) architecture by default, but this value is
            retained for fallback compatibility with non-e2e exports.
        device: Compute device identifier. ``"cuda"`` selects the GPU when
            available, otherwise ``"cpu"`` is used as a safe fallback.
    """

    weights_path: str = "yolo26n.pt"
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.45
    # "cuda" → GPU (önerilir), "cpu" → işlemci, "mps" → Apple Silicon
    device: str = "cuda"
    # 640: en yüksek doğruluk | 416: denge | 320: en hızlı
    imgsz: int = 416
    # CUDA'da float16 kullanır → ~2x hız, CPU'da otomatik devre dışı
    half: bool = True


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for the video capture source.

    Attributes:
        source_index: Index of the webcam exposed by the operating system.
            ``0`` corresponds to the default built-in camera.
        frame_width: Requested capture width in pixels.
        frame_height: Requested capture height in pixels.
    """

    source_index: int = 0
    frame_width: int = 640
    frame_height: int = 480


@dataclass(frozen=True)
class DistanceConfig:
    """Configuration for the monocular distance estimation module.

    Attributes:
        focal_length_pixels: Focal length of the camera expressed in pixels.
            This value is obtained via a one-time calibration step using a
            reference object of known real-world size placed at a known
            distance from the camera.
        known_heights_meters: Mapping between object class names and their
            average real-world heights in meters. These priors are required
            because a single image does not contain absolute scale.
        default_height_meters: Fallback height used when a detected class is
            not present in ``known_heights_meters``.
    """

    focal_length_pixels: float = 600.0
    known_heights_meters: dict = field(
        default_factory=lambda: {
            # İnsanlar ve hayvanlar
            "person": 1.70,
            "cat": 0.30,
            "dog": 0.50,
            "horse": 1.60,
            "sheep": 0.90,
            "cow": 1.50,
            "elephant": 3.00,
            "bear": 1.20,
            "zebra": 1.50,
            "giraffe": 5.50,
            # Araçlar
            "bicycle": 1.10,
            "car": 1.50,
            "motorcycle": 1.10,
            "airplane": 5.00,
            "bus": 3.20,
            "train": 4.00,
            "truck": 3.50,
            "boat": 2.50,
            # Sokak eşyaları
            "traffic light": 0.75,
            "fire hydrant": 0.60,
            "stop sign": 0.75,
            "parking meter": 1.20,
            "bench": 0.90,
            # Ev eşyaları
            "chair": 0.90,
            "couch": 0.85,
            "dining table": 0.75,
            "bed": 0.60,
            "toilet": 0.70,
            "tv": 0.60,
            "refrigerator": 1.80,
            "oven": 0.90,
            "microwave": 0.30,
            "sink": 0.85,
            "door": 2.00,
            # Elektronik & küçük eşya
            "laptop": 0.25,
            "mouse": 0.05,
            "remote": 0.05,
            "keyboard": 0.05,
            "cell phone": 0.15,
            "clock": 0.30,
            "vase": 0.30,
            "scissors": 0.15,
            "toothbrush": 0.18,
            "hair drier": 0.20,
            # Mutfak
            "bottle": 0.25,
            "wine glass": 0.20,
            "cup": 0.10,
            "fork": 0.18,
            "knife": 0.18,
            "spoon": 0.16,
            "bowl": 0.10,
            "banana": 0.20,
            "apple": 0.08,
            "sandwich": 0.08,
            "orange": 0.08,
            "broccoli": 0.20,
            "carrot": 0.20,
            "hot dog": 0.12,
            "pizza": 0.05,
            "donut": 0.05,
            "cake": 0.12,
            # Diğer
            "backpack": 0.45,
            "umbrella": 1.00,
            "handbag": 0.30,
            "tie": 0.60,
            "suitcase": 0.70,
            "frisbee": 0.03,
            "skis": 1.50,
            "snowboard": 1.50,
            "sports ball": 0.22,
            "kite": 0.60,
            "baseball bat": 1.00,
            "baseball glove": 0.25,
            "skateboard": 0.15,
            "surfboard": 2.00,
            "tennis racket": 0.70,
            "book": 0.25,
            "potted plant": 0.50,
            "teddy bear": 0.35,
            "toaster": 0.20,
            "bird": 0.30,
        }
    )
    default_height_meters: float = 0.50


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for the Text-to-Speech feedback module.

    Attributes:
        speech_rate: Speaking rate in words per minute passed to the TTS engine.
        volume: Output volume in the range ``[0.0, 1.0]``.
        message_cooldown_seconds: Minimum interval between two consecutive
            announcements for the same object class. This prevents the engine
            from flooding the user with repetitive speech.
    """

    speech_rate: int = 175
    volume: float = 1.0
    message_cooldown_seconds: float = 2.5


@dataclass(frozen=True)
class DirectionConfig:
    """Thresholds that partition the image horizontally into zones.

    The image width is normalized to the interval ``[0, 1]``. An object whose
    bounding-box center falls below ``left_boundary`` is reported as being on
    the left, above ``right_boundary`` on the right, and otherwise in front.

    Attributes:
        left_boundary: Upper limit of the left zone.
        right_boundary: Lower limit of the right zone.
    """

    left_boundary: float = 0.35
    right_boundary: float = 0.65


# Public singletons consumed by the rest of the package.
MODEL_CONFIG = ModelConfig()
CAMERA_CONFIG = CameraConfig()
DISTANCE_CONFIG = DistanceConfig()
AUDIO_CONFIG = AudioConfig()
DIRECTION_CONFIG = DirectionConfig()
