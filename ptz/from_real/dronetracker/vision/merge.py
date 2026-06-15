"""Box-merging helpers (alternative to NMS for overlapping tile detections)."""

__all__ = ["iou", "iou_min", "merge_group", "global_merge"]

import numpy as np


def iou(box1, box2):
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def iou_min(box1, box2):
    """IoU relative to the *smaller* of the two boxes (useful for contained boxes)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter    = max(0, x2 - x1) * max(0, y2 - y1)
    area1    = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2    = (box2[2] - box2[0]) * (box2[3] - box2[1])
    min_area = min(area1, area2)
    return inter / min_area if min_area > 0 else 0


def merge_group(group):
    """Merge a list of [x1,y1,x2,y2,score,class] rows into one consensus box.

    Coordinates and score: mean / max over the group.
    Class: taken from the first (highest-scoring) member.
    """
    g = np.array(group)
    return [
        np.mean(g[:, 0]),
        np.mean(g[:, 1]),
        np.mean(g[:, 2]),
        np.mean(g[:, 3]),
        np.max(g[:, 4]),
        int(g[0, 5]),
    ]


def global_merge(all_boxes, iou_thresh=0.5, iou_func=iou):
    """Greedily cluster boxes by IoU and merge each cluster into one box.

    Args:
        all_boxes:  List of [x1,y1,x2,y2,score,class] rows.
        iou_thresh: Minimum IoU to include a box in a cluster.
        iou_func:   IoU function — pass ``iou_min`` for contained-box clustering.

    Returns:
        List of merged [x1,y1,x2,y2,score,class] rows.
    """
    if len(all_boxes) == 0:
        return []

    all_boxes = np.array(all_boxes)
    used      = np.zeros(len(all_boxes), dtype=bool)
    merged    = []

    for i in range(len(all_boxes)):
        if used[i]:
            continue
        group  = [all_boxes[i]]
        used[i] = True
        for j in range(i + 1, len(all_boxes)):
            if used[j]:
                continue
            if all_boxes[i, 5] != all_boxes[j, 5]:   # different class → skip
                continue
            if iou_func(all_boxes[i], all_boxes[j]) > iou_thresh:
                group.append(all_boxes[j])
                used[j] = True
        merged.append(merge_group(group))

    return merged
