#!/usr/bin/env python3
"""Entry point for the live PTZ tracking pipeline.

patch_torchvision() MUST be called before any torch/ultralytics import.
This module does it at the top level so the import order is always correct,
even if someone imports this file rather than running it as a script.
"""

from dronetracker.compat import patch_torchvision
patch_torchvision()   # must run before torch / ultralytics are imported

from dronetracker.config.loader import load_pipeline_config  # noqa: E402
from dronetracker.pipeline.live_ptz import LivePtzPipeline   # noqa: E402


if __name__ == "__main__":
    cfg = load_pipeline_config()
    LivePtzPipeline(cfg).run()
