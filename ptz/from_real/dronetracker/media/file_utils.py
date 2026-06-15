"""File-type detection helpers for video and image files."""

__all__ = ["VIDEO_EXTENSIONS", "IMAGE_EXTENSIONS", "is_video_file", "is_image_file"]

from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_video_file(path) -> bool:
    """Return True if ``path`` has a recognised video file extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path) -> bool:
    """Return True if ``path`` has a recognised image file extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS
