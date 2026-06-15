"""Shared reporting utilities for diagnostic probe scripts.

Provides a ``Tee`` output multiplexer and small helper functions that mirror the
``ok``/``fail``/``hr``/``dump_obj`` style used in ``ptz_probe.py``.

Typical usage in a probe script::

    from dronetracker.probes.report import Tee, hr, ok, fail, dump_obj
    import sys, io

    report_file = open("probe_report.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, report_file)
    try:
        hr("Section title")
        ok("Everything worked")
        fail("Something broke")
    finally:
        sys.stdout = sys.__stdout__
        report_file.close()
"""

__all__ = ["Tee", "hr", "ok", "fail", "dump_obj"]

import sys


class Tee:
    """Write to multiple streams simultaneously.

    Wraps ``stdout`` (or any other stream) so every ``print()`` goes both to the
    terminal and to a report file.  Implements the minimal stream interface needed
    by Python's ``sys.stdout`` and ONVIF/zeep internals.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        """Return the real stdout file descriptor (required by some zeep/urllib3 internals)."""
        return sys.__stdout__.fileno()

    def isatty(self):
        """Return False so that logging/urllib3 doesn't try to use ANSI escape codes."""
        return False


def hr(label: str = "") -> None:
    """Print a horizontal rule with an optional section label."""
    print(f"\n{'─' * 55}  {label}")


def ok(msg: str) -> None:
    """Print a success line (✓  msg)."""
    print(f"  ✓  {msg}")


def fail(msg: str) -> None:
    """Print a failure line (✗  msg)."""
    print(f"  ✗  {msg}")


def dump_obj(obj, indent: int = 4, _depth: int = 0) -> None:
    """Recursively print a zeep/suds response object.

    Walks the object graph by calling ``dict()`` on each node (which works for
    zeep ``ComplexType`` objects) and falls back to ``str()`` for leaf values.
    Prints to ``stdout`` (which may be a ``Tee`` to also write to a file).

    Args:
        obj:     The object to dump (zeep response, dict, or any printable).
        indent:  Current indentation in spaces.
        _depth:  Recursion depth guard (stops at 12 to prevent infinite loops
                 in circular zeep object graphs).
    """
    pad = " " * indent

    if obj is None:
        print(f"{pad}None")
        return

    if _depth > 12:
        print(f"{pad}(max depth reached)")
        return

    try:
        d = dict(obj)
        for k, v in d.items():
            if hasattr(v, "__iter__") and not isinstance(v, str):
                print(f"{pad}{k}:")
                try:
                    for item in v:
                        dump_obj(item, indent + 4, _depth + 1)
                except Exception:
                    print(f"{pad}    {v}")
            elif hasattr(v, "__dict__") or hasattr(v, "_raw_elements"):
                print(f"{pad}{k}:")
                dump_obj(v, indent + 4, _depth + 1)
            else:
                print(f"{pad}{k}: {v}")
    except Exception:
        try:
            print(f"{pad}{obj}")
        except Exception:
            print(f"{pad}(undumpable: {type(obj)})")
