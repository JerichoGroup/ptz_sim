"""Vectorised Non-Maximum Suppression (class-aware, IoM overlap metric)."""

__all__ = ["global_nms"]

import numpy as np


def global_nms(boxes, scores, classes, iou_threshold):
    """Class-aware NMS using IoM (inter/min-area), keeping the larger box.

    Standard IoU fails when one box is fully inside another (IoU ≈ 0 for a
    tiny box inside a huge one).  IoM = inter / min(a1,a2) gives 1.0 for any
    fully-contained box, reliably suppressing redundant detections.

    Boxes are processed in decreasing area order so the *larger* (outer) box
    is always kept and the smaller (inner) box is suppressed — regardless of
    which has the higher confidence score.

    Args:
        boxes:         [[x1,y1,x2,y2], ...] in absolute pixel coordinates.
        scores:        Confidence scores (same length as boxes).
        classes:       Class indices (same length as boxes).
        iou_threshold: Boxes with overlap (IoM) > this are suppressed.

    Returns:
        (kept_boxes, kept_scores, kept_classes) as plain Python lists.
    """
    if len(boxes) == 0:
        return [], [], []

    boxes   = np.array(boxes)
    scores  = np.array(scores)
    classes = np.array(classes)
    n       = len(boxes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order     = areas.argsort()[::-1]  # largest area first — keeps the outer/parent box, suppresses smaller inner boxes
    suppressed = np.zeros(n, dtype=bool)
    keep      = []

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)

        same_cls = classes == classes[i]
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)
        w    = np.maximum(0.0, xx2 - xx1 + 1)
        h    = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        min_area = np.minimum(areas[i], areas)
        iom      = inter / np.maximum(min_area, 1e-12)
        iom[i]   = 0.0  # don't self-suppress

        suppressed |= same_cls & (iom > iou_threshold)

    return (
        boxes[keep].tolist(),
        scores[keep].tolist(),
        classes[keep].tolist(),
    )
