"""This file defines a class to control and command a UDP robot in Isaac Sim."""

# ==================== Imports ====================
import math
import time
import numpy as np
from typing import Tuple
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat

from .base_udp_sender import BaseUDPSender
from .udp_utils import lla_distance_to_meters, meters_to_latlon_offset, LLAPoint


# ==================== Consts ====================
DOT_THRESHOLD = 0.9995  # for slerp interpolation fallback to lerp
EULER_AXES = 'rxyz'


# ==================== the UdpBot class ====================
class UdpBot(BaseUDPSender):
    """controls a UDP robot in Isaac Sim."""

    def __init__(self,
                 start_lat: float, start_lon: float, start_alt: float,
                 start_roll_d: float, start_pitch_d: float, start_yaw_d: float,
                 udp_port: float = 33333, send_rate_hz: float = 30,
                 broadcast: bool = True, target_ip: str = "127.0.0.1"
                ) -> None:
        """Initialize the UdpBot with a starting point and a UDP address"""

        super().__init__(udp_port=udp_port, send_rate_hz=send_rate_hz,
                         broadcast=broadcast, target_ip=target_ip)
        
        self._current_lat = start_lat
        self._current_lon = start_lon
        self._current_alt = start_alt

        self._world_rotation_matrix = euler2mat(
            math.radians(start_roll_d),
            math.radians(start_pitch_d),
            math.radians(start_yaw_d),
            axes=EULER_AXES
        )

        self._update_cur_rotation_from_world_rotation_matrix()
        self._publish_current_pose()

    def get_next_point(self, step: int = 0) -> Tuple[float, float, float, float, float, float]:
        """Returns the current point (lat, lon, alt, body roll/pitch/yaw).

        ``step`` is supplied by ``BaseUDPSender._run_loop`` and ignored here: the
        background loop simply re-publishes whatever pose the move methods
        (``move_to_point``/``steer``/etc.) have set on this bot."""

        return (self._current_lat,
                self._current_lon,
                self._current_alt,
                self._current_body_roll_r,
                self._current_body_pitch_r,
                self._current_body_yaw_r)

    def _update_cur_rotation_from_world_rotation_matrix(self) -> None:
        """Update the current roll, pitch, yaw angles from the world rotation matrix."""

        roll_r, pitch_r, yaw_r = mat2euler(self._world_rotation_matrix, axes=EULER_AXES)

        norm_roll_r = self._normalize_angle_r(roll_r)
        norm_pitch_r = self._normalize_angle_r(pitch_r)
        norm_yaw_r = self._normalize_angle_r(yaw_r)

        self._current_world_roll_r = norm_roll_r
        self._current_world_pitch_r = norm_pitch_r
        self._current_world_yaw_r = norm_yaw_r

        self._current_body_roll_r = norm_roll_r
        self._current_body_pitch_r = norm_pitch_r
        self._current_body_yaw_r = norm_yaw_r

    def _apply_world_axis_rotation(self, delta_r: float, axis: str, duration_s: float = 1.0):
        """Applies a angle delta rotation around a world axis ('roll', 'pitch', 'yaw')."""

        if axis == "roll":
            delta_R = euler2mat(delta_r, 0.0, 0.0, axes=EULER_AXES)
        elif axis == "pitch":
            delta_R = euler2mat(0.0, delta_r, 0.0, axes=EULER_AXES)
        elif axis == "yaw":
            delta_R = euler2mat(0.0, 0.0, delta_r, axes=EULER_AXES)
        else:
            raise ValueError(f"Invalid axis '{axis}'")

        target_R = delta_R @ self._world_rotation_matrix

        roll_r, pitch_r, yaw_r = mat2euler(target_R, axes=EULER_AXES)

        self._turn_to(
            math.degrees(roll_r),
            math.degrees(pitch_r),
            math.degrees(yaw_r),
            duration_s
        )

    def _publish_current_pose(self) -> None:
        """Send the current pose (lat, lon, alt, body roll/pitch/yaw) once."""

        self.send_once(self._current_lat,
                       self._current_lon,
                       self._current_alt,
                       self._current_body_roll_r,
                       self._current_body_pitch_r,
                       self._current_body_yaw_r)

    def _apply_world_rotation_offset(self, rotation_matrix_delta):
        """Update world-frame orientation and recompute body-frame orientation"""

        self._world_rotation_matrix = rotation_matrix_delta @ self._world_rotation_matrix
        self._update_cur_rotation_from_world_rotation_matrix()

    def _smooth_rotation_matrixs_interpolation(self, rotation_matrix_1: np.ndarray, rotation_matrix_2: np.ndarray, rotation_progress: float) -> np.ndarray:
        """Interpolates smoothly between two rotation matrices using quaternion SLERP."""

        # Convert rotation matrices to quaternions
        quaternion_1 = mat2quat(rotation_matrix_1)
        quaternion_2 = mat2quat(rotation_matrix_2)

        # Normalize quaternions and ensure the shortest path is taken
        quaternion_1 = quaternion_1 / np.linalg.norm(quaternion_1)
        quaternion_2 = quaternion_2 / np.linalg.norm(quaternion_2)

        # Compute scalar product to check if quaternions are close enough for linear interpolation
        scalar_product = float(np.dot(quaternion_1, quaternion_2))

        # If scalar product is negative, flip quaternion_2 to ensure shortest path
        if scalar_product < 0.0:
            quaternion_2 = -quaternion_2
            scalar_product = -scalar_product

        # If the quaternions are very close, use linear interpolation
        if scalar_product > DOT_THRESHOLD:
            result_quaternion = quaternion_1 + rotation_progress * (quaternion_2 - quaternion_1)
            result_quaternion = result_quaternion / np.linalg.norm(result_quaternion)

            return quat2mat(result_quaternion)

        # Compute the angle between the quaternions and perform spherical interpolation
        theta_0 = math.acos(scalar_product)
        sin_theta_0 = math.sin(theta_0)

        theta = theta_0 * rotation_progress
        sin_theta = math.sin(theta)

        sin_0 = math.sin(theta_0 - theta) / sin_theta_0
        sin_1 = sin_theta / sin_theta_0

        # Compute the interpolated quaternion and convert back to rotation matrix
        result_quaternion = sin_0 * quaternion_1 + sin_1 * quaternion_2
        result_quaternion = result_quaternion / np.linalg.norm(result_quaternion)

        return quat2mat(result_quaternion)

    def _turn_to(self, target_roll_d: float, target_pitch_d: float, target_yaw_d: float, duration_s: float = 1.0) -> None:
        """Turns the bot to a new orientation by smoothly transitioning from the current to the target point, in world frame"""

        steps = max(1, int(self.send_rate_hz * duration_s))
        dt = 1.0 / self.send_rate_hz
        start_time = time.perf_counter()

        start_world_rotation_matrix = self._world_rotation_matrix.copy()
        target_world_rotation_matrix = euler2mat(
            math.radians(target_roll_d),
            math.radians(target_pitch_d),
            math.radians(target_yaw_d),
            axes=EULER_AXES
        )

        for i in range(1, steps + 1):
            alpha = i / steps

            self._world_rotation_matrix = self._smooth_rotation_matrixs_interpolation(start_world_rotation_matrix, target_world_rotation_matrix, alpha)
            self._update_cur_rotation_from_world_rotation_matrix()
            self._publish_current_pose()

            next_tick = start_time + i * dt
            now = time.perf_counter()
            if next_tick > now:
                time.sleep(next_tick - now)

    def turn_roll(self, delta_roll_d: float, duration_s: float = 1.0) -> None:
        """Turns the bot by a certain angle on the roll axis in world frame"""

        self._apply_world_axis_rotation(math.radians(delta_roll_d), "roll", duration_s)

    def turn_pitch(self, delta_pitch_d: float, duration_s: float = 1.0) -> None:
        """Turns the bot by a certain angle on the pitch axis in world frame"""

        self._apply_world_axis_rotation(math.radians(delta_pitch_d), "pitch", duration_s)

    def turn_yaw(self, delta_yaw_d: float, duration_s: float = 1.0) -> None:
        """Turns the bot by a certain angle on the yaw axis in world frame"""

        self._apply_world_axis_rotation(math.radians(delta_yaw_d), "yaw", duration_s)

    def turn_to_point(self, target_lat: float, target_lon: float, target_alt: float, duration_s: float = 1.0,) -> None:
        """Smoothly rotate in world frame to look at the given target LLA."""

        start_lat = self._current_lat
        start_lon = self._current_lon
        start_alt = self._current_alt

        start_point = LLAPoint(start_lat, start_lon, start_alt)

        dy = lla_distance_to_meters(start_point, LLAPoint(target_lat, start_lon, start_alt))
        if target_lat < start_lat:
            dy = -dy

        dx = lla_distance_to_meters(start_point, LLAPoint(start_lat, target_lon, start_alt))
        if target_lon < start_lon:
            dx = -dx

        dz = target_alt - start_alt

        yaw_r = math.atan2(dx, dy)

        aerial_dist = math.sqrt(dx * dx + dy * dy)
        pitch_r = math.atan2(dz, aerial_dist)

        self._turn_to(
            math.degrees(self._current_world_roll_r),
            math.degrees(pitch_r),
            math.degrees(yaw_r),
            duration_s
        )

    def move_to_point(self, target_lat: float, target_lon: float, target_alt: float,
                      target_roll_d: float, target_pitch_d: float, target_yaw_d: float,
                      duration_s: float = 1.0, look_at_target: bool = True, turn_duration_s: float = 0.5) -> None:
        """Moves the bot to a new point by smoothly transitioning from the current to the target point"""

        if look_at_target:
            self.turn_to_point(target_lat, target_lon, target_alt, duration_s=turn_duration_s)

        steps = max(1, int(self.send_rate_hz * duration_s))
        dt = 1.0 / self.send_rate_hz
        start_time = time.perf_counter()

        start_lat = self._current_lat
        start_lon = self._current_lon
        start_alt = self._current_alt

        start_point = LLAPoint(start_lat, start_lon, start_alt)

        dy = lla_distance_to_meters(start_point, LLAPoint(target_lat, start_lon, start_alt))
        if target_lat < start_lat:
            dy = -dy

        dx = lla_distance_to_meters(start_point, LLAPoint(start_lat, target_lon, start_alt))
        if target_lon < start_lon:
            dx = -dx

        dz = target_alt - start_alt

        for i in range(1, steps + 1):
            alpha = i / steps

            x = alpha * dx
            y = alpha * dy
            z = start_alt + alpha * dz

            d_lat, d_lon = meters_to_latlon_offset(y, x, start_lat)
            self._current_lat = start_lat + d_lat
            self._current_lon = start_lon + d_lon
            self._current_alt = z

            self._publish_current_pose()

            next_tick = start_time + i * dt
            now = time.perf_counter()
            if next_tick > now:
                time.sleep(next_tick - now)

        if look_at_target:
            self._turn_to(target_roll_d, target_pitch_d, target_yaw_d, duration_s=turn_duration_s)

    def move_forward_backward(self, distance_m: float, duration_s: float = 1.0) -> None:
        """Moves the bot by a certain distance in the direction it is currently facing, positive for forward, negative for backward"""

        yaw_r = self._current_world_yaw_r

        dx = distance_m * math.sin(yaw_r)
        dy = distance_m * math.cos(yaw_r)

        d_lat, d_lon = meters_to_latlon_offset(dy, dx, self._current_lat)
        target_lat = self._current_lat + d_lat
        target_lon = self._current_lon + d_lon

        self.move_to_point(target_lat=target_lat,
                           target_lon=target_lon,
                           target_alt=self._current_alt,
                           target_roll_d=math.degrees(self._current_world_roll_r),
                           target_pitch_d=math.degrees(self._current_world_pitch_r),
                           target_yaw_d=math.degrees(self._current_world_yaw_r),
                           duration_s=duration_s,
                           look_at_target=False)

    def move_right_left(self, distance_m: float, duration_s: float = 1.0) -> None:
        """Moves the bot by a certain distance with respect to the direction it is currently facing, positive for right, negative for left"""

        yaw_r = self._current_world_yaw_r

        dx = distance_m * math.cos(yaw_r)
        dy = -distance_m * math.sin(yaw_r)

        d_lat, d_lon = meters_to_latlon_offset(dy, dx, self._current_lat)
        target_lat = self._current_lat + d_lat
        target_lon = self._current_lon + d_lon

        self.move_to_point(target_lat=target_lat,
                        target_lon=target_lon,
                        target_alt=self._current_alt,
                        target_roll_d=math.degrees(self._current_world_roll_r),
                        target_pitch_d=math.degrees(self._current_world_pitch_r),
                        target_yaw_d=math.degrees(self._current_world_yaw_r),
                        duration_s=duration_s,
                        look_at_target=False)

    def move_up_down(self, distance_m: float, duration_s: float = 1.0) -> None:
        """Moves the bot by a certain distance with respect to the direction it is currently facing, positive for up, negative for down"""

        target_alt = self._current_alt + distance_m

        self.move_to_point(self._current_lat,
                           self._current_lon,
                           target_alt,
                           math.degrees(self._current_world_roll_r),
                           math.degrees(self._current_world_pitch_r),
                           math.degrees(self._current_world_yaw_r),
                           duration_s,
                           look_at_target=False)        

    def steer(self, turn_radius_m: float, speed_ms: float, duration_s: float = 1.0) -> None:
        """Drives the bot in a circular path with given turn radius and speed, positive radius = right, negative radius = left, radius = 0 straight."""

        steps = max(1, int(self.send_rate_hz * duration_s))
        dt = 1.0 / self.send_rate_hz
        start_time = time.perf_counter()

        if turn_radius_m == 0.0:
            yaw_rate = 0.0
        else:
            yaw_rate = speed_ms / turn_radius_m

        for i in range(steps):
            distance = speed_ms * dt

            dx = distance * math.sin(self._current_world_yaw_r)
            dy = distance * math.cos(self._current_world_yaw_r)

            d_lat, d_lon = meters_to_latlon_offset(dy, dx, self._current_lat)
            self._current_lat += d_lat
            self._current_lon += d_lon

            if yaw_rate != 0.0:
                delta_yaw_r = yaw_rate * dt
                delta_world_rotation_matrix = euler2mat(0.0, 0.0, delta_yaw_r, axes=EULER_AXES)
                self._world_rotation_matrix = delta_world_rotation_matrix @ self._world_rotation_matrix
                self._update_cur_rotation_from_world_rotation_matrix()

            self._publish_current_pose()

            next_tick = start_time + (i + 1) * dt
            now = time.perf_counter()
            if next_tick > now:
                time.sleep(next_tick - now)

    @staticmethod
    def _normalize_angle_r(angle_r: float) -> float:
        """Normalize angle in radians to [-pi, pi]"""

        return (angle_r + math.pi) % (2.0 * math.pi) - math.pi
