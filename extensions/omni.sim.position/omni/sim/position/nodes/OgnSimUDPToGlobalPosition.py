"""This file contains the implementation of a node that receives global position data via UDP packets"""

# ==================== Imports ====================
import carb
import socket
import struct
from typing import Any
import math
from dataclasses import dataclass
from omni.sim.position.ogn.OgnSimUDPToGlobalPositionDatabase import OgnSimUDPToGlobalPositionDatabase

# ==================== Constants ====================
HEADER1 = 0xAC
HEADER2 = 0xDC
HEADER1_INDEX = 0
HEADER2_INDEX = 1
PACKET_SIZE = 51
PAYLOAD_START_INDEX = 2
CHECKSUM_INDEX = 50
LITTLE_ENDIAN_STRING = "<6d"
BROADCAST_ADDRESS = '0.0.0.0'


# ==================== Helper Functions ====================
def convert_ned_to_enu(roll_ned: float, pitch_ned: float, yaw_ned: float) -> tuple[float, float, float]:
    """Convert intrinsic xyz Euler angles from NED (aerospace convention: +X forward, +Y right, +Z down)
    to ENU (+X east, +Y north, +Z up).

    Args:
        roll_ned (float): between -pi and pi.
        pitch_ned (float): between -pi/2 and pi/2.
        yaw_ned (float): between -pi and pi.

    Returns:
        Angles in ENU.
    """
    roll_enu = pitch_ned
    pitch_enu = roll_ned
    yaw_enu = -yaw_ned + math.pi / 2

    # Normalise to [-pi, pi]
    yaw_enu = (yaw_enu + math.pi) % (2 * math.pi) - math.pi

    return roll_enu, pitch_enu, yaw_enu


# ==================== Internal State ====================
class OgnSimUDPToGlobalPositionInternalState:
    """Internal state for the OgnSimUDPToGlobalPosition node"""

    def __init__(self):
        """Initialize the internal state of the node"""
        carb.log_info("SIM | UTGP | Initializing UDPToGlobalPosition internal state")

        self.parser = None

    def create_parser(self, port: int) -> None:
        """Create a new PacketGetter instance with the given port"""
        if self.parser is None:
            carb.log_info(f"SIM | UTGP | Creating PacketGetter with port {port}")
            self.parser = PacketGetter(HEADER1, HEADER2, PACKET_SIZE, port)


# ==================== The PoseData dataclass ====================
@dataclass
class PoseData:
    """Data class to hold a position data"""
    lat: float
    lon: float
    alt: float
    roll: float
    pitch: float
    yaw: float


# ==================== The PacketParser class ==================
class PacketGetter:

    def __init__(self, header1: int, header2: int, packet_size: int, port: int) -> None:
        """initialize the PacketParser with the given headers and port"""
        self._header1 = header1
        self._header2 = header2
        self._packet_size = packet_size
        self._last_good_packet = None
        self._sock = self.define_socket(port)

    def define_socket(self, port: int) -> socket.socket:
        """Define the UDP socket with the given port"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((BROADCAST_ADDRESS, port))
        sock.setblocking(False)

        return sock

    def get_cur_data(self) -> PoseData | None:
        """Drain the socket and return the NEWEST valid pose.

        This MUST drain rather than read a single packet.  ``compute`` runs once
        per rendered frame (~20/s), while a sender publishes faster than that —
        ``BaseUDPSender``'s background loop defaults to 30 Hz, and callers may
        publish on top of it.  Reading one packet per frame leaves the surplus
        queued in the kernel, so the rendered position falls steadily further
        behind real time the longer a sender runs, and then jumps forward when the
        receive buffer overflows and packets are discarded.

        Measured with the real packet size and rates (60 Hz in, 20 Hz out), four
        seconds of continuous motion left the render 8 s behind reality with 162 of
        241 positions never displayed.  Draining to the newest packet brought that
        to 0.1 s and 3.

        Every packet carries an absolute pose, so discarding the older ones in the
        queue loses nothing — the newest is the whole truth.
        """
        newest = None
        while True:
            try:
                raw_data, sender = self._sock.recvfrom(self._packet_size)
            except BlockingIOError:
                break
            carb.log_verbose(
                f"SIM | UTGP | Received packet from {sender}: {raw_data.hex()}")
            parsed = self._parse_packet(raw_data)
            if parsed is not None:
                newest = parsed

        if newest is not None:
            self._last_good_packet = newest
        return self._last_good_packet

    def _parse_packet(self, raw_data: bytes) -> PoseData | None:
        """Validate and unpack one packet; None (and a log line) if it is bad."""
        if len(raw_data) != self._packet_size:
            carb.log_error(
                f"SIM | UTGP | Received packet of incorrect size: {len(raw_data)} bytes, expected {self._packet_size} bytes.")
            return None

        if raw_data[HEADER1_INDEX] != self._header1 or raw_data[HEADER2_INDEX] != self._header2:
            carb.log_error(
                f"SIM | UTGP | Received packet with incorrect headers: {raw_data[HEADER1_INDEX]}, {raw_data[HEADER2_INDEX]}. Expected: {self._header1}, {self._header2}.")
            return None

        payload = raw_data[PAYLOAD_START_INDEX:CHECKSUM_INDEX]
        expected_checksum = raw_data[CHECKSUM_INDEX]
        actual_checksum = 0
        for byte in payload:
            actual_checksum ^= byte

        if actual_checksum != expected_checksum:
            carb.log_error(f"SIM | UTGP | Checksum mismatch: expected {expected_checksum}, got {actual_checksum}.")
            return None

        try:
            fields = struct.unpack(LITTLE_ENDIAN_STRING, payload)
            return PoseData(
                lat=fields[0],
                lon=fields[1],
                alt=fields[2],
                roll=fields[3],
                pitch=fields[4],
                yaw=fields[5]
            )
        except struct.error as e:
            carb.log_error(f"SIM | UTGP | Failed to unpack payload: {e}. Payload: {payload.hex()}")
            return None


# ==================== the OgnSimUDPToGlobalPosition class ====================
class OgnSimUDPToGlobalPosition:
    """this class implements a node that receives global position data via UDP packets"""

    @staticmethod
    def internal_state() -> OgnSimUDPToGlobalPositionInternalState:
        """Create and return the internal state for the node"""
        return OgnSimUDPToGlobalPositionInternalState()

    @staticmethod
    def compute(db: OgnSimUDPToGlobalPositionDatabase) -> bool:
        """Compute the output global position and orientation from the received UDP packets"""
        carb.log_info("SIM | UTGP | UDPToGlobalPosition compute triggered")

        port = db.inputs.udp_port
        state = db.internal_state
        state.create_parser(port)

        ned_pose = state.parser.get_cur_data()

        if ned_pose is not None:
            lat, lon, alt = ned_pose.lat, ned_pose.lon, ned_pose.alt

            enu_roll, enu_pitch, enu_yaw = convert_ned_to_enu(ned_pose.roll, ned_pose.pitch, ned_pose.yaw)
            db.outputs.global_position = [lat, lon, alt]
            db.outputs.global_orientation = [enu_roll, enu_pitch, enu_yaw]

        return True

    @staticmethod
    def release(node: Any) -> None:
        """Release the resources used by the subscriber node"""
        carb.log_info("SIM | UTGP | Node release triggered")
        state = None

        try:
            state = OgnSimUDPToGlobalPositionDatabase.per_node_internal_state(node)
        except Exception as e:
            carb.log_error(f"SIM | UTGP | Node release error: {e}")

        if state is not None:
            carb.log_info("SIM | UTGP | Node resources released")
