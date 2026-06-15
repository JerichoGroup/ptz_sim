"""Vectorised Non-Maximum Suppression across merged tile detections."""

__all__ = ["global_nms"]

import numpy as np


def global_nms(boxes, scores, classes, iou_threshold):
    """Suppress overlapping boxes by IoU, keeping the highest-scoring one per group.

    Args:
        boxes:         [[x1,y1,x2,y2], ...] in absolute pixel coordinates.
        scores:        Confidence scores (same length as boxes).
        classes:       Class indices (same length as boxes).
        iou_threshold: Boxes with IoU > this against a kept box are suppressed.

    Returns:
        (kept_boxes, kept_scores, kept_classes) as plain Python lists.
    """
    if len(boxes) == 0:
        return [], [], []

    boxes   = np.array(boxes)
    scores  = np.array(scores)
    classes = np.array(classes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w    = np.maximum(0.0, xx2 - xx1 + 1)
        h    = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou  = inter / (areas[i] + areas[order[1:]] - inter)

        inds  = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return (
        boxes[keep].tolist(),
        scores[keep].tolist(),
        classes[keep].tolist(),
    )
