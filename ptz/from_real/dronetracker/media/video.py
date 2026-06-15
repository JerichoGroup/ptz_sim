"""Video file helpers: writer factory with explicit codec constants.

Codec note from CLAUDE.md:
  - Use XVID + .avi for offline scripts — AVI is readable mid-write.
  - Use mp4v + .mp4 only when a clean writer.release() is guaranteed —
    mp4v requires a finalised moov atom; an interrupted write is corrupt.
"""

__all__ = ["AVI_FOURCC", "MP4_FOURCC", "make_writer"]

import cv2

AVI_FOURCC = cv2.VideoWriter_fourcc(*"XVID")
MP4_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def make_writer(
    path: str,
    fps: float,
    width: int,
    height: int,
    fourcc: int = AVI_FOURCC,
) -> cv2.VideoWriter:
    """Create and return a ``cv2.VideoWriter``.

    Args:
        path:   Output file path (.avi recommended; see module docstring).
        fps:    Frames per second.
        width:  Frame width in pixels.
        height: Frame height in pixels.
        fourcc: Codec FourCC code.  Defaults to ``AVI_FOURCC`` (XVID).

    Returns:
        An opened ``cv2.VideoWriter``.  Caller is responsible for calling
        ``writer.release()`` when done.
    """
    return cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
