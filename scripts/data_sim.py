#!/usr/bin/env python3
"""Launch the simulator in DATA-CAPTURE mode.  Run this instead of ptz/ptz_sim.py.

    ISAACSIM_PYTHON ./scripts/data_sim.py

Identical to ptz/ptz_sim.py except for three constructor flags — ptz_sim.py itself
is untouched, so the normal sim stand keeps behaving exactly as before:

  sat=True            loads /Environment/SAT + sat_node.py, which is what actually
                      writes PNG files when we publish to /isaac_core/sat.
                      ptz_sim.py's __main__ never passes this, which is the only
                      reason data capture was unavailable.

  bbox_publisher=True the source of the labels (/isaac_core/bbox).

  image_rtp=False     no RTSP server, no lib_manager subprocess, no birds/blur.
                      None of that is needed to capture frames, and it competes
                      for GPU with the renderer. NOTE: this also means the
                      realism effects are NOT in the captured images — the bird
                      compositing in gen_dataset.py is done afterwards, on the
                      saved PNG, precisely so we know exactly where it landed.

Leave this running, then run scripts/gen_dataset.py in a second terminal.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ptz"))

from ptz_sim import PTZSim                                      # noqa: E402


CORE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
USD_PATH = "./usd/maps/earth/earth.usda"


def main():
    sim = PTZSim(
        listen_port=5005,
        jetson_ip=os.environ.get("PTZ_JETSON_IP"),   # None -> reply-to-sender
        jetson_port=int(os.environ.get("PTZ_JETSON_PORT", "5006")),
        core_path=CORE_PATH,
        usd_path=USD_PATH,
        show_isaac_logs=False,
        image_rtp=False,        # no RTSP/effects — see module docstring
        bbox_publisher=True,    # labels
        sat=True,              # image capture
    )
    print("[data_sim] data-capture mode: sat=ON  bbox=ON  rtsp=OFF")
    print("[data_sim] now run:  ISAACSIM_PYTHON ./scripts/gen_dataset.py")
    sim.run()


if __name__ == "__main__":
    main()
