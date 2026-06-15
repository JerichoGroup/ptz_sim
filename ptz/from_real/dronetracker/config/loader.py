"""Config loader — merges defaults → config.yaml → .env overrides.

Usage::

    from dronetracker.config.loader import load_pipeline_config
    cfg = load_pipeline_config()           # reads config.yaml + .env if present
    cfg = load_pipeline_config("my.yaml")  # explicit path

All file paths in the loaded config are resolved relative to the current
working directory (preserve the "run from repo root" convention).

Priority (highest wins): .env > config.yaml > dataclass defaults.
"""

import os
from pathlib import Path

from dronetracker.config.schema import PipelineConfig, CameraConfig


def _load_yaml(path: str) -> dict:
    """Load a YAML file if it exists; return empty dict otherwise."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # optional dependency
    except ImportError:
        raise ImportError(
            "PyYAML is required to load config.yaml.  "
            "Install it with:  pip install pyyaml"
        )
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _load_dotenv(path: str = ".env") -> dict:
    """Parse a .env file (KEY=VALUE lines) if it exists; return empty dict otherwise."""
    p = Path(path)
    if not p.exists():
        return {}
    result = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def _apply_camera_overrides(cam: CameraConfig, yaml_data: dict, env: dict) -> CameraConfig:
    """Merge camera credentials from YAML and .env into a CameraConfig."""
    # YAML section: camera: {ip: ..., port: ..., user: ..., password: ...}
    section = yaml_data.get("camera", {})
    if section.get("ip"):
        cam.ip = section["ip"]
    if section.get("port"):
        cam.port = int(section["port"])
    if section.get("user"):
        cam.user = section["user"]
    if section.get("password"):
        cam.password = section["password"]

    # .env overrides (highest priority): CAM_IP, CAM_PORT, CAM_USER, CAM_PASS
    if "CAM_IP"   in env: cam.ip       = env["CAM_IP"]
    if "CAM_PORT" in env: cam.port     = int(env["CAM_PORT"])
    if "CAM_USER" in env: cam.user     = env["CAM_USER"]
    if "CAM_PASS" in env: cam.password = env["CAM_PASS"]

    return cam


def load_pipeline_config(
    yaml_path: str = "config.yaml",
    env_path:  str = ".env",
) -> PipelineConfig:
    """Build a PipelineConfig by overlaying config.yaml and .env onto defaults.

    Neither file is required — if both are absent the dataclass defaults are
    returned unchanged (same behaviour as the old hard-coded CONFIG block).

    Args:
        yaml_path: Path to the YAML config file (relative to CWD).
        env_path:  Path to the .env credentials file (relative to CWD).

    Returns:
        A fully-populated PipelineConfig ready to pass to LivePtzPipeline.
    """
    cfg  = PipelineConfig()
    yaml = _load_yaml(yaml_path)
    env  = _load_dotenv(env_path)

    # Merge camera section (credentials come from .env by priority).
    cfg.camera = _apply_camera_overrides(cfg.camera, yaml, env)

    # Merge remaining scalar sections from YAML.
    # Each section maps to the matching dataclass field on PipelineConfig.
    # Only keys that already exist in the dataclass are applied (unknown keys ignored).
    section_map = {
        "zoom":    cfg.zoom,
        "pid":     cfg.pid,
        "motion":  cfg.motion,
        "tracker": cfg.tracker,
        "yolo":    cfg.yolo,
        "track":   cfg.track,
        "frozen":  cfg.frozen,
        "home":    cfg.home,
        "cmd":     cfg.cmd,
        "display": cfg.display,
    }
    for section_name, obj in section_map.items():
        overrides = yaml.get(section_name, {})
        for key, val in overrides.items():
            if hasattr(obj, key):
                setattr(obj, key, val)

    return cfg
