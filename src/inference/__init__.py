"""Inference: detector wrapper, image prediction, and alert logic."""

from .detector import AlertState, Detection, WildfireDetector

__all__ = ["WildfireDetector", "Detection", "AlertState"]
