"""This file configs the simulation and starts it"""

# ==================== imports ====================
import atexit
import ctypes
import os
import json
import signal
import subprocess

import sim_utils


# ==================== lib_manager child cleanup ====================
# lib_manager owns the RTSP server port. If it outlives this process it keeps that
# port bound, and the NEXT run's RTSP server cannot bind — clients then get served
# by the stale orphan, whose Isaac Sim is gone, so they see 503 with no video and
# nothing in the logs explains it. Isaac Sim is frequently stopped with Ctrl-C or
# killed, in which case a plain terminate() at the end of main() never runs, so
# belt-and-braces: ask the kernel to SIGKILL the child when we die (PDEATHSIG),
# and also terminate it from atexit / signal handlers.
_PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """preexec_fn: kernel sends SIGKILL to this child when the parent exits."""
    try:
        ctypes.CDLL("libc.so.6").prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass   # best effort — the atexit/signal handlers below still apply


def _install_child_cleanup(proc: subprocess.Popen) -> None:
    """Terminate *proc* on normal exit and on SIGINT/SIGTERM."""
    def _cleanup(*_args):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(_cleanup)
    for sig in (signal.SIGINT, signal.SIGTERM):
        prev = signal.getsignal(sig)

        def _handler(signum, frame, _prev=prev):
            _cleanup()
            if callable(_prev):
                _prev(signum, frame)
            else:
                raise KeyboardInterrupt

        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass   # not on the main thread — atexit + PDEATHSIG still cover us


# ==================== the main simulation func ====================
def main():
    args = sim_utils.parse_arguments()

    cur_launch_config = sim_utils.DEFAULT_LAUNCH_CONFIG.copy()
    cur_launch_config["headless"] = args.headless
    os.environ["LAUNCH_CONFIG"] = json.dumps(cur_launch_config)

    from sim_app import Simulation          # import sim app after setting the launch config

    usds_to_add = sim_utils.get_usds_to_add(args, sim_utils.PROJECT_HOME_DIR)

    simulation = Simulation(args.usd_path, usds_to_add)

    libs_to_run_json = json.dumps(sim_utils.get_libs_to_add(args))
    print(f"[main_sim] Starting LibManager with libs: {libs_to_run_json}", flush=True)
    lib_proc = subprocess.Popen(["/usr/bin/python3",    # running on system python instead of isaac python to use unsupported packages
                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib_manager.py"),
                                 "--libs",
                                 libs_to_run_json],
                                preexec_fn=_die_with_parent)
    _install_child_cleanup(lib_proc)

    try:
        simulation.run_simulation()
    finally:
        if lib_proc.poll() is None:
            lib_proc.terminate()
            try:
                lib_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                lib_proc.kill()


# ==================== run the main ====================
if __name__ == "__main__":
    main()
