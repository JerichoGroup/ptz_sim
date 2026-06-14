"""
OgnSimGlobalPositionToLocalPosition Node
========================================
Converts global LLA positions and orientations to local ENU coordinates using pyproj and transforms3d.
"""

# ==================== imports ==================== #
import carb
import math
import numpy as np
from pyproj import Transformer
from typing import Any, Tuple, Dict
from transforms3d.quaternions import qmult
from transforms3d.euler import euler2quat, quat2euler

from omni.sim.math.ogn.OgnSimGlobalPositionToLocalPositionDatabase import OgnSimGlobalPositionToLocalPositionDatabase


# ==================== Constants ==================== #
EULER_AXES = 'rxyz' # Roll-Pitch-Yaw convention


# ==================== Internal State ====================
class OgnSimGlobalPositionToLocalPositionInternalState:
    """Internal state for the global-to-local position node. Stores geo reference, converter, and warning flags."""

    def __init__(self) -> None:
        carb.log_info("SIM | GPTLP | Initializing internal state")
        self.geo_reference: Tuple[float, float, float] | None = None
        self.converter: 'ENUConverter' | None = None
        self.warned_no_data: bool = False

    def define_converter(self, geo_reference: Tuple[float, float, float]) -> None:
        """Define the ENU converter for a reference point.

        Args:
            geo_reference: Tuple of (latitude, longitude, altitude) in degrees/meters.
        """
        self.geo_reference = geo_reference
        self.converter = ENUConverter(*geo_reference)


# ==================== Helpers ====================
def is_invalid_global_position(global_position: Tuple[float, float, float], state: OgnSimGlobalPositionToLocalPositionInternalState) -> bool:
    """Validate global position. Logs a warning once if invalid.

    Args:
        global_position: Global LLA tuple (lat, lon, alt)
        state: Internal node state

    Returns:
        True if position is invalid, else False
    """
    if global_position is None or np.allclose(global_position, [0.0, 0.0, 0.0]):
        if not state.warned_no_data:
            carb.log_warn("SIM | GPTLP | Skipping compute — no valid global position yet")
            state.warned_no_data = True
        return True

    state.warned_no_data = False
    return False


# ==================== ENU Converter ====================
class ENUConverter:
    """Converts LLA coordinates to ENU coordinates based on a reference point."""
    def __init__(self, ref_lat: float, ref_lon: float, ref_alt: float) -> None:
        self.lla_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
        self.x0, self.y0, self.z0 = self.lla_to_ecef.transform(ref_lon, ref_lat, ref_alt)

        lat_rad = math.radians(ref_lat)
        lon_rad = math.radians(ref_lon)
        self.rotation_matrix = np.array([
            [-np.sin(lon_rad),                np.cos(lon_rad),               0],
            [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)],
            [ np.cos(lat_rad) * np.cos(lon_rad),  np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)]
        ])


    def convert_lla_enu(self, lat: float, lon: float, alt: float) -> Dict[str, float]:
        """Convert LLA to ENU"""
        x, y, z = self.lla_to_ecef.transform(lon, lat, alt)
        return self.convert_ecef_enu(x, y, z)


    def convert_ecef_enu(self, x: float, y: float, z: float) -> Dict[str, float]:
        """Convert ECEF to ENU"""
        dx, dy, dz = x - self.x0, y - self.y0, z - self.z0
        enu = self.rotation_matrix @ np.array([dx, dy, dz])

        return {"east": float(enu[0]), "north": float(enu[1]), "up": float(enu[2])}


# ==================== Node Class ====================
class OgnSimGlobalPositionToLocalPosition:
    """OmniGraph node: converts global positions/orientations to local ENU coordinates."""

    @staticmethod
    def internal_state() -> OgnSimGlobalPositionToLocalPositionInternalState:
        """Returns the persistent internal node state."""
        return OgnSimGlobalPositionToLocalPositionInternalState()


    @staticmethod
    def compute(db: OgnSimGlobalPositionToLocalPositionDatabase) -> bool:
        """Compute ENU position and transformed orientation."""
        carb.log_info("SIM | GPTLP | GlobalPositionToLocalPosition compute triggered")

        reference = db.inputs.enu_reference
        global_position = db.inputs.global_position
        global_orientation = db.inputs.global_orientation
        state = db.internal_state

        if is_invalid_global_position(global_position, state):
            return True

        if state.converter is None or not np.allclose(state.geo_reference, reference):
            state.define_converter(reference)

        enu = state.converter.convert_lla_enu(*global_position)
        x, y, z = enu["east"], enu["north"], enu["up"]

        q_drone = euler2quat(*global_orientation, axes=EULER_AXES)

        offset_roll = math.radians(db.inputs.offset_roll)
        offset_pitch = math.radians(db.inputs.offset_pitch)
        offset_yaw = math.radians(db.inputs.offset_yaw)
        # PTZ rotation order: yaw around world Z first, then pitch around the resulting
        # local horizontal axis, then roll.  Intrinsic ZYX achieves this: each rotation
        # is around the body axis *after* the previous rotation, so pan does not tilt the
        # tilt axis and tilt does not tilt the pan axis.
        q_offset = euler2quat(offset_yaw, offset_pitch, offset_roll, axes='rzyx')

        qw, qx, qy, qz = qmult(q_drone, q_offset)
        roll, pitch, yaw = quat2euler((qw, qx, qy, qz), axes=EULER_AXES)

        db.outputs.global_position = global_position
        db.outputs.global_orientation = [roll, pitch, yaw]
        db.outputs.local_position = [x, y, z]

        # Crazy Isaac sim bug:
        # Isaac sim uses (w, x, y, z) conventions. And this is how it is shown in gui.
        # But isaac sim is mixing the order of the components, so we need to mix them back:
        db.outputs.local_orientation = [qx, qy, qz, qw]

        return True


    @staticmethod
    def release(node: Any) -> None:
        """Release internal node state when OmniGraph destroys the node."""
        carb.log_info("SIM | GPTLP | Node release triggered")
        state = None
        
        try:
            state = OgnSimGlobalPositionToLocalPositionDatabase.per_node_internal_state(node)
        except Exception as e:
            carb.log_error(f"SIM | GPTLP | Node release error: {e}")

        if state is not None:
            carb.log_info("SIM | GPTLP | Node resources released")
