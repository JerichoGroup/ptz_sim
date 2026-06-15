"""YOLO label format conversions and file I/O helpers."""

__all__ = ["xywh_to_yolo", "xywh_to_xyxy", "save_yolo_label"]

from random import randint


def xywh_to_yolo(rect, img_w, img_h):
    """Convert an (x, y, w, h) pixel rect to YOLO normalised centre format.

    If ``rect`` is empty, returns a random small box (used for data augmentation
    when no real annotation is available).
    """
    if len(rect) == 0:
        return (
            img_w // 2,
            img_h // 2,
            img_w // (2 ** randint(3, 12)),
            img_h // (2 ** randint(0, 12)),
        )

    x, y, w, h = rect
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h


def xywh_to_xyxy(box):
    """Convert (x, y, w, h) pixel box to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [int(x), int(y), int(x + w), int(y + h)]


def save_yolo_label(label_path, bbox, img_w, img_h, class_id=0):
    """Write a single-annotation YOLO label file for the given bbox.

    Args:
        label_path: Destination ``.txt`` path.
        bbox:       (x, y, w, h) in pixel coordinates.
        img_w:      Frame width in pixels.
        img_h:      Frame height in pixels.
        class_id:   YOLO class index (default 0).
    """
    x, y, w, h  = bbox
    x_center    = (x + w / 2) / img_w
    y_center    = (y + h / 2) / img_h
    w_norm      = w / img_w
    h_norm      = h / img_h
    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
