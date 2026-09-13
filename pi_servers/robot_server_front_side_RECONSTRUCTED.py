# RECONSTRUCTED compatibility server — NOT the original historical file.
#
# We do not have a preserved copy of the exact Pi server that was used for the front+side
# (2-camera) dataset/model. This file is rebuilt from the verified side+claw server
# (robot_server_side_claw_VERIFIED.py) by swapping the camera roles:
#   side + claw   ->   front + side
#
# Must be documented in the writeup as a recreated compatibility server, not claimed to be
# the original source used during that experiment.
#
# Camera hardware, updated 2026-09-11: the claw camera (5MP USB) is now disconnected entirely
# and not used by this server. Two Logitech HD Pro Webcam C920 units are connected instead —
# genuinely two side cameras (left/right). Confirmed via v4l2-ctl --list-devices AND
# lerobot-find-cameras opencv (both individually verified to open and capture a real frame):
#   HD Pro Webcam C920 (usb-xhci-hcd.0-2) -> /dev/video2 (capture node)
#   HD Pro Webcam C920 (usb-xhci-hcd.1-2) -> /dev/video0 (capture node, on a SEPARATE USB
#     controller than the other camera — moved there deliberately to rule out USB bandwidth
#     contention as a cause of the connection failures below)
# If either feed looks wrong/swapped once you check recorded video, swap CAMERA_INDEX_FRONT
# and CAMERA_INDEX_SIDE below.
#
# The variable/dict names stay "front"/"side" — that's not a description of camera placement,
# it's the literal feature-key naming the trained checkpoint expects as input and can't be
# renamed without breaking the model. Physically both are side-mounted cameras now (left/right).
# This is still not the original 2-camera hardware setup used to train this model (unknown
# original camera placement) — a best-available reconstruction, not a faithful reproduction.
#
# 2026-09-11: root-caused and fixed a real connection failure, not a bandwidth issue after all.
# robot.connect() kept failing on /dev/video2 with "failed to set capture_width=640
# (actual_width=640, width_success=False)" plus a "Bad file descriptor" on VIDIOC_QBUF.
# Isolated testing (check_video2_alone.py) proved this camera's driver ALWAYS returns False
# from cv2 .set() for width/height/fourcc — even when the actual resulting value already
# matches what was requested (640x480) — a known quirk with some UVC drivers, not a real
# failure; the camera opens and captures real 640x480 frames fine. LeRobot's OpenCVCamera
# (_validate_width_and_height / _validate_fps in camera_opencv.py) treats any False from
# .set() as fatal regardless of the actual value, which turned this harmless quirk into a hard
# crash. Tried leaving width/height/fps unset in the config to sidestep the check entirely —
# blocked: SO101FollowerConfig.__post_init__ (robots/config.py) requires width/height/fps to
# be set on every camera used by a robot, no way around that at the config level. Real fix:
# monkey-patch just those two validation methods (below, before connecting) to check the
# ACTUAL resulting value instead of trusting the driver's unreliable success flag — the
# installed lerobot package itself is untouched, this only affects this process's copy of the
# class, and only changes what counts as a genuine failure (a real value mismatch still raises).
#
# 2026-09-02: added motor-load HOLD/E-STOP safety, ported from robot_server_1cam_EVAL.py
# (originally from Smoth VLA/robot_server_new.py), after a real overheating incident on this
# exact server — it previously had NO load/current protection at all, only the robot's
# per-step angle clamp (max_relative_target), which does nothing to stop a motor sustaining
# high current against a mechanical limit or obstruction over many steps. Now identical
# protection to the 1cam server: HOLD skips an action if load crosses 68% of rated torque,
# E-STOP disables torque entirely at 88% and requires an explicit reset_estop command.
#
# Also ported 1cam's SAFE_LIMITS absolute per-joint angle clipping, same day, same reasoning:
# max_relative_target alone only bounds how big ONE step is — it does nothing to stop the arm
# reaching an extreme/unsafe absolute position gradually over many small steps, which could
# push a joint against its own mechanical hard-stop. Every action is now clipped to the same
# absolute-angle ranges as the 1cam server (same physical arm, same calibration) BEFORE being
# handed to robot.send_action(), which still applies its own max_relative_target on top — two
# independent layers instead of one.

import math
import pickle

import cv2
import numpy as np
import zmq

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError


@check_if_not_connected
def _lenient_validate_width_and_height(self) -> None:
    """Drop-in replacement for OpenCVCamera._validate_width_and_height that checks the actual
    resulting value instead of trusting cv2's .set() return flag, which this camera's driver
    always reports as False even when the value is already correct. Still raises if the real
    value genuinely doesn't match — this only removes the false-positive failure, not the
    real check."""
    if self.videocapture is None:
        raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

    self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
    self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

    actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    if self.capture_width != actual_width:
        raise RuntimeError(
            f"{self} failed to set capture_width={self.capture_width} ({actual_width=})."
        )

    actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if self.capture_height != actual_height:
        raise RuntimeError(
            f"{self} failed to set capture_height={self.capture_height} ({actual_height=})."
        )


@check_if_not_connected
def _lenient_validate_fps(self) -> None:
    """Same reasoning as _lenient_validate_width_and_height, for fps. Untested in isolation
    whether this camera's fps actually lands on the requested value, so this one only warns
    (doesn't raise) on a mismatch — fps accuracy isn't safety-critical for this harness, which
    just processes frames as they arrive rather than depending on an exact reported rate."""
    if self.videocapture is None:
        raise DeviceNotConnectedError(f"{self} videocapture is not initialized")
    if self.fps is None:
        raise ValueError(f"{self} FPS is not set")

    self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
    actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
    if not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
        print(f"  [WARN] {self} requested fps={self.fps} but camera reports {actual_fps} — "
              f"continuing anyway (not treated as fatal).")


OpenCVCamera._validate_width_and_height = _lenient_validate_width_and_height
OpenCVCamera._validate_fps = _lenient_validate_fps

ROBOT_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5AE6080769-if00"
)
ROBOT_ID = "my_follower_arm"
PORT = 5555

CAMERA_INDEX_FRONT = "/dev/video2"   # Logitech C920 (usb-xhci-hcd.0-2)
CAMERA_INDEX_SIDE = "/dev/video0"    # Logitech C920 (usb-xhci-hcd.0-1)
# Using explicit device paths, not bare integers: OpenCV's own V4L2 index enumeration doesn't
# map 1:1 to /dev/videoN. Device numbers shifted after unplugging the 5MP claw camera — both
# confirmed as the only two real capture devices via `lerobot-find-cameras opencv` and
# `v4l2-ctl --list-devices` on 2026-09-11. Swap these two if the feeds turn out reversed once
# you check the recorded video.

CAMERAS = {
    "front": OpenCVCameraConfig(
        index_or_path=CAMERA_INDEX_FRONT, width=640, height=480, fps=30
    ),
    "side": OpenCVCameraConfig(
        index_or_path=CAMERA_INDEX_SIDE, width=640, height=480, fps=30
    ),
}
# width/height/fps ARE required here (SO101FollowerConfig.__post_init__ enforces it for every
# camera used by a robot) — the monkey-patch above is what makes these safe to request despite
# the driver's false-failure quirk on this camera. Both cameras' real default already is
# 640x480 anyway (confirmed by direct testing), so these values match reality; the patch exists
# for the case where cv2 .set() reports failure even though the actual value is already right.

config = SO101FollowerConfig(
    port=ROBOT_PORT,
    id=ROBOT_ID,
    cameras=CAMERAS,
    use_degrees=True,
    max_relative_target=15.0,
    disable_torque_on_disconnect=False,
)

robot = SO101Follower(config)
robot.connect(calibrate=False)
# Auto-apply the saved calibration file to the servos if it doesn't already match what's on
# the motors (same effect as manually pressing Enter at the interactive "use provided
# calibration file?" prompt that robot.connect() would otherwise show) — non-interactive.
if not robot.is_calibrated and robot.calibration:
    print("Calibration file doesn't match servo state — re-applying it automatically...")
    robot.bus.write_calibration(robot.calibration)
    print("Calibration re-applied.")

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Hard angle limits derived from motor calibration with a 5° / 3% safety buffer. Same values
# as robot_server_1cam_EVAL.py — same physical arm, same calibration.
SAFE_LIMITS = {
    "shoulder_pan.pos":  (-115.26, 115.26),
    "shoulder_lift.pos": ( -99.84,  99.84),
    "elbow_flex.pos":    ( -92.98,  92.98),
    "wrist_flex.pos":    ( -99.62,  99.62),
    "wrist_roll.pos":    (-175.00, 175.00),
    "gripper.pos":       (   3.00,  97.00),
}

LOAD_HOLD_THRESHOLD  = 700   # 68% of rated torque
LOAD_ESTOP_THRESHOLD = 900   # 88% of rated torque

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def clip_to_safe(joint_name: str, value: float) -> float:
    lo, hi = SAFE_LIMITS[joint_name]
    clipped = float(np.clip(value, lo, hi))
    if clipped != value:
        print(f"  [LIMIT] {joint_name}: {value:.2f} -> {clipped:.2f}")
    return clipped


def read_loads(robot) -> dict[str, int]:
    try:
        return robot.bus.sync_read("Present_Load")
    except Exception as e:
        print(f"  [WARN] Could not read Present_Load: {e}")
        return {}


def check_loads(loads: dict[str, int]) -> tuple[bool, bool]:
    hold = estop = False
    for motor, val in loads.items():
        magnitude = abs(val)
        if magnitude >= LOAD_ESTOP_THRESHOLD:
            print(f"  [ESTOP] {motor} load={val} (>={LOAD_ESTOP_THRESHOLD})")
            estop = True
        elif magnitude >= LOAD_HOLD_THRESHOLD:
            print(f"  [HOLD]  {motor} load={val} (>={LOAD_HOLD_THRESHOLD})")
            hold = True
    return hold, estop


estop_active = False

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.setsockopt(zmq.LINGER, 0)
socket.bind(f"tcp://*:{PORT}")

print("SO-101 FRONT + SIDE robot server (eval variant — supports torque release)")
print(f"Listening on port {PORT}")

try:
    while True:
        msg_raw = socket.recv()
        try:
            msg = pickle.loads(msg_raw)
        except Exception as exc:
            socket.send(pickle.dumps({
                "ok": False,
                "error": f"Invalid message: {exc}",
            }))
            continue

        cmd = msg.get("cmd")

        if cmd == "get_obs":
            try:
                obs = robot.get_observation()
                if "front" not in obs:
                    raise RuntimeError("FRONT camera missing.")
                if "side" not in obs:
                    raise RuntimeError("SIDE camera missing.")

                state = np.array(
                    [obs[joint] for joint in JOINTS],
                    dtype=np.float32,
                )
                front_frame = obs["front"]
                side_frame = obs["side"]

                socket.send(pickle.dumps({
                    "ok": True,
                    "state": state,
                    "front_frame": front_frame,
                    "side_frame": side_frame,
                }))
            except Exception as exc:
                socket.send(pickle.dumps({
                    "ok": False,
                    "error": str(exc),
                }))

        elif cmd == "action":
            try:
                requested = np.asarray(
                    msg["action"], dtype=np.float32
                )
                if requested.shape != (6,):
                    raise ValueError(
                        f"Expected action shape (6,), "
                        f"received {requested.shape}"
                    )
                if not np.all(np.isfinite(requested)):
                    raise ValueError(
                        "Action contains NaN or infinity."
                    )

                loads = read_loads(robot)
                hold, estop = check_loads(loads)

                if estop:
                    estop_active = True
                    robot.bus.disable_torque()
                    print("  [ESTOP] Torque disabled. Reset required.")
                    socket.send(pickle.dumps({"ok": False, "estop": True}))
                    continue

                if estop_active:
                    socket.send(pickle.dumps({"ok": False, "estop": True}))
                    continue

                if hold:
                    print("  [HOLD] Skipping action this step.")
                    socket.send(pickle.dumps({
                        "ok": True,
                        "action_sent": {
                            joint: float(requested[i]) for i, joint in enumerate(JOINTS)
                        },
                    }))
                    continue

                action_dict = {
                    joint: clip_to_safe(joint, float(requested[i]))
                    for i, joint in enumerate(JOINTS)
                }
                action_sent = robot.send_action(action_dict)
                socket.send(pickle.dumps({
                    "ok": True,
                    "action_sent": action_sent,
                }))
            except Exception as exc:
                socket.send(pickle.dumps({
                    "ok": False,
                    "error": str(exc),
                }))

        elif cmd == "reset_estop":
            if estop_active:
                try:
                    robot.bus.enable_torque()
                    estop_active = False
                    print("  [ESTOP] Cleared. Torque re-enabled.")
                    socket.send(pickle.dumps({"ok": True}))
                except Exception as exc:
                    socket.send(pickle.dumps({"ok": False, "error": str(exc)}))
            else:
                socket.send(pickle.dumps({"ok": True}))

        elif cmd == "torque":
            try:
                if msg.get("enable", True):
                    robot.bus.enable_torque()
                else:
                    robot.bus.disable_torque()
                socket.send(pickle.dumps({"ok": True}))
            except Exception as exc:
                socket.send(pickle.dumps({
                    "ok": False,
                    "error": str(exc),
                }))

        elif cmd == "ping":
            socket.send(pickle.dumps({
                "ok": True,
                "message": "pong",
            }))

        else:
            socket.send(pickle.dumps({
                "ok": False,
                "error": f"Unknown command: {cmd}",
            }))

except KeyboardInterrupt:
    print("Stopping robot server...")

finally:
    try:
        robot.bus.disable_torque()
    except Exception:
        pass
    try:
        robot.disconnect()
    except Exception:
        pass
    socket.close(0)
    context.term()
    print("Robot server stopped. Torque released.")
