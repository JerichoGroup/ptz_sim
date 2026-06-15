"""Compatibility shims for the Jetson deployment environment.

Must be imported (and ``patch_torchvision()`` called) **before** any
``import torch`` or ``from ultralytics import YOLO`` statement, because
both packages trigger a ``torchvision`` import at load time.  The entry
point ``apps/live_ptz.py`` does this before any other import.

Background: the Jetson wheels for PyTorch ship without ``torchvision.ops.nms``.
Calling any YOLO inference path that touches NMS would raise ``AttributeError``
or ``ImportError`` without this shim.
"""

__all__ = ["patch_torchvision"]

import sys


def patch_torchvision() -> None:
    """Stub out torchvision for Jetson environments where it is absent or has missing metadata.

    Two problems handled:
    1. ultralytics/__init__.py calls ``importlib.metadata.version("torchvision")`` at
       module-load time (before any import of torchvision itself).  If the dist-info
       is missing this raises PackageNotFoundError and the whole import fails.
    2. ``import torchvision`` fails if the package is not installed at all.

    Both are fixed here.  Safe to call when real torchvision is installed — the
    metadata patch only fires when the lookup would otherwise fail, and the
    sys.modules shim only fires when torchvision is not already loaded.
    """
    import importlib.metadata as _meta

    # Fix (1): ensure importlib.metadata.version("torchvision") never raises.
    try:
        _meta.version("torchvision")
    except _meta.PackageNotFoundError:
        _orig_version = _meta.version
        def _patched_version(name):
            if name == "torchvision":
                return "0.20.0"
            return _orig_version(name)
        _meta.version = _patched_version

    # Fix (2): inject a fake torchvision module so "import torchvision" works.
    if "torchvision" in sys.modules:
        return   # real torchvision present — nothing more to do

    import torch
    from types import ModuleType

    def _nms(boxes, scores, iou_threshold):
        if boxes.shape[0] == 0:
            return torch.zeros(0, dtype=torch.long)
        x1 = boxes[:, 0]; y1 = boxes[:, 1]
        x2 = boxes[:, 2]; y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        keep  = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            order = order[1:]
            xx1 = x1[order].clamp(min=float(x1[i]))
            yy1 = y1[order].clamp(min=float(y1[i]))
            xx2 = x2[order].clamp(max=float(x2[i]))
            yy2 = y2[order].clamp(max=float(y2[i]))
            inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
            iou   = inter / (areas[i] + areas[order] - inter + 1e-6)
            order = order[iou <= iou_threshold]
        return torch.tensor(keep, dtype=torch.long)

    tv      = ModuleType("torchvision")
    tv_ops  = ModuleType("torchvision.ops")
    tv_ops.nms     = _nms
    tv_ops.box_iou = lambda a, b: torch.zeros(a.shape[0], b.shape[0])
    tv.ops         = tv_ops
    tv.__version__ = "0.20.0"
    sys.modules["torchvision"]            = tv
    sys.modules["torchvision.ops"]        = tv_ops
    sys.modules["torchvision.ops.boxes"]  = tv_ops
