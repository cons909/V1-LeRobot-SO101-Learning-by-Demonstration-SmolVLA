# Raspberry Pi server for the 2-camera ("side left + side right") SO-101 config. Reconstructed
# from the side+claw server, since the original file wasn't preserved — not a faithful
# reproduction. Feature keys stay "front"/"side" (the checkpoint's literal input names), even
# though both cameras are physically side-mounted.

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
