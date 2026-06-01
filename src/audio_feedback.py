"""Text-to-Speech audio feedback module.

The speech synthesis call is inherently blocking: while the engine is reading a
sentence, the thread that initiated the call cannot proceed. Coupling this
behaviour directly to the main detection loop would cause the live video to
freeze every time an announcement is made. To avoid that, this module runs a
dedicated worker thread that consumes messages from a thread-safe queue.

A per-class cooldown is also enforced so that the same object does not trigger
a new announcement on every single frame.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, Optional

import pyttsx3

from .config import AUDIO_CONFIG, AudioConfig

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Non-blocking Text-to-Speech announcer.

    The class spawns a background worker thread on construction. Client code
    only calls :meth:`announce`, which returns immediately and delegates the
    actual speech synthesis to the worker.
    """

    def __init__(self, config: AudioConfig = AUDIO_CONFIG) -> None:
        """Initialize the TTS engine and start the background worker.

        Args:
            config: Audio configuration parameters.
        """
        self._config = config

        # Initialize the TTS engine with the configured voice properties.
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", config.speech_rate)
        self._engine.setProperty("volume", config.volume)

        # Thread-safe message queue. ``None`` is reserved as a shutdown signal.
        self._message_queue: "queue.Queue[Optional[str]]" = queue.Queue()

        # Tracks the last announcement timestamp for each object class so the
        # cooldown policy can be enforced without a global lock on the queue.
        self._last_announcement: Dict[str, float] = {}
        self._cooldown_lock = threading.Lock()

        # Start the worker thread. ``daemon=True`` makes sure the thread does
        # not prevent the interpreter from exiting when the main program ends.
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        logger.info("Audio feedback worker started")

    def announce(self, class_name: str, direction: str, distance_meters: float) -> None:
        """Queue an announcement for the given detection.

        The call is non-blocking: the method returns as soon as the message is
        enqueued. If the same class was recently announced, the new message is
        silently dropped to respect the cooldown policy.

        Args:
            class_name: Name of the detected object class.
            direction: Horizontal direction label (``"left"``, ``"ahead"``,
                ``"right"``).
            distance_meters: Estimated distance to the object, in meters.
        """
        if not self._is_cooldown_elapsed(class_name):
            return

        message = self._build_message(class_name, direction, distance_meters)
        self._message_queue.put(message)

    def shutdown(self) -> None:
        """Signal the worker to exit and wait for it to terminate."""
        self._message_queue.put(None)
        self._worker.join(timeout=3.0)
        logger.info("Audio feedback worker stopped")

    def _is_cooldown_elapsed(self, class_name: str) -> bool:
        """Check whether the cooldown for a class has expired and update it.

        Args:
            class_name: Class name whose cooldown should be checked.

        Returns:
            ``True`` when the class may be announced again, ``False`` otherwise.
        """
        now = time.monotonic()
        with self._cooldown_lock:
            last_time = self._last_announcement.get(class_name, 0.0)
            if now - last_time < self._config.message_cooldown_seconds:
                return False
            self._last_announcement[class_name] = now
        return True

    @staticmethod
    def _build_message(class_name: str, direction: str, distance_meters: float) -> str:
        """Compose the sentence that will be passed to the TTS engine.

        Args:
            class_name: Class of the detected object.
            direction: Horizontal direction label.
            distance_meters: Estimated distance in meters.

        Returns:
            A short, imperative sentence suitable for spoken delivery.
        """
        rounded_distance = max(0.1, round(distance_meters, 1))
        return f"{class_name} {direction}, {rounded_distance} meters"

    def _run_worker(self) -> None:
        """Background loop that consumes the message queue and calls the engine."""
        while True:
            message = self._message_queue.get()
            # ``None`` is the agreed shutdown sentinel.
            if message is None:
                break
            try:
                self._engine.say(message)
                self._engine.runAndWait()
            except Exception:  # noqa: BLE001 - engine exceptions are not typed
                logger.exception("TTS engine failed while speaking '%s'", message)
