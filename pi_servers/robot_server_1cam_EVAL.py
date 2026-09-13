# Raspberry Pi server for the 1-camera (left side) SO-101 config. Serves get_obs/action over
# ZMQ, converts camera frames from BGR to RGB, and reports action_sent for safety-clamp
# detection. Includes SAFE_LIMITS angle clipping and load-based HOLD/E-STOP.

import pickle
import time
import cv2
import numpy as np
import zmq

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROBOT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6080769-if00"
ROBOT_ID = "my_follower_arm"
CAMERA_INDEX = 2   # Logitech HD Pro Webcam C920

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Hard angle limits derived from motor calibration with a 5° / 3% safety buffer.
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
        raw = robot.bus.sync_read("Present_Load")
        return raw
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


config = SO101FollowerConfig(
    port=ROBOT_PORT,
    use_degrees=True,
    disable_torque_on_disconnect=False,
)
config.id = ROBOT_ID

robot = SO101Follower(config)
robot.connect()

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"  [WARN] Camera at index {CAMERA_INDEX} did not open. If another USB camera was "
          f"just unplugged/plugged in, indices may have shifted — check with "
          f"`ls /dev/video*` or `v4l2-ctl --list-devices` and update CAMERA_INDEX.")

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

estop_active = False

print("Robot server ready on port 5555 (1-camera eval variant)")
print(f"Safety limits active. Hold >{LOAD_HOLD_THRESHOLD}, E-stop >{LOAD_ESTOP_THRESHOLD}")

try:
    while True:
        try:
            msg = pickle.loads(socket.recv())
        except Exception as exc:
            socket.send(pickle.dumps({"ok": False, "error": f"Invalid message: {exc}"}))
            continue

        if msg["cmd"] == "get_obs":
            try:
                obs = robot.get_observation()
                ret, frame = cap.read()

                if not ret:
                    raise RuntimeError(
                        f"cap.read() failed (cap.isOpened()={cap.isOpened()}); "
                        f"camera index {CAMERA_INDEX} may be wrong or disconnected."
                    )

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                state = np.array([obs[j] for j in JOINTS], dtype=np.float32)
                loads = read_loads(robot)
                load_array = np.array(
                    [loads.get(m, 0) for m in MOTOR_NAMES], dtype=np.int32
                )

                socket.send(pickle.dumps({
                    "ok": True,
                    "front_frame": frame,
                    "state": state,
                    "loads": load_array,
                }))
            except Exception as exc:
                socket.send(pickle.dumps({"ok": False, "error": str(exc)}))

        elif msg["cmd"] == "action":
            try:
                requested = np.array(msg["action"], dtype=np.float32)

                loads = read_loads(robot)
                hold, estop = check_loads(loads)
                load_array = np.array(
                    [loads.get(m, 0) for m in MOTOR_NAMES], dtype=np.int32
                )

                if estop:
                    estop_active = True
                    robot.bus.disable_torque()
                    print("  [ESTOP] Torque disabled. Reset required.")
                    socket.send(pickle.dumps({"ok": False, "estop": True, "loads": load_array}))
                    continue

                if estop_active:
                    socket.send(pickle.dumps({"ok": False, "estop": True, "loads": load_array}))
                    continue

                if hold:
                    print("  [HOLD] Skipping action this step.")
                    socket.send(pickle.dumps({
                        "ok": True, "held": True, "loads": load_array,
                        "action_sent": requested,
                    }))
                    continue

                action_dict = {}
                for i, joint in enumerate(JOINTS):
                    action_dict[joint] = clip_to_safe(joint, float(requested[i]))

                clipped = np.array([action_dict[j] for j in JOINTS], dtype=np.float32)
                print(f"ACTION: {np.round(clipped, 2)}")

                robot.send_action(action_dict)

                socket.send(pickle.dumps({
                    "ok": True, "held": False, "loads": load_array,
                    "action_sent": clipped,
                }))
            except Exception as exc:
                socket.send(pickle.dumps({"ok": False, "error": str(exc)}))

        elif msg["cmd"] == "reset_estop":
            if estop_active:
                try:
                    robot.bus.enable_torque()
                    estop_active = False
                    print("  [ESTOP] Cleared. Torque re-enabled.")
                    socket.send(pickle.dumps({"ok": True}))
                except Exception as e:
                    socket.send(pickle.dumps({"ok": False, "error": str(e)}))
            else:
                socket.send(pickle.dumps({"ok": True}))

        elif msg["cmd"] == "torque":
            try:
                if msg.get("enable", True):
                    robot.bus.enable_torque()
                else:
                    robot.bus.disable_torque()
                socket.send(pickle.dumps({"ok": True}))
            except Exception as e:
                socket.send(pickle.dumps({"ok": False, "error": str(e)}))

        else:
            socket.send(pickle.dumps({"ok": False}))

except KeyboardInterrupt:
    print("Stopping robot server")

finally:
    try:
        robot.bus.disable_torque()
    except Exception:
        pass
    cap.release()
    robot.disconnect()
    print("Robot server stopped. Torque released.")
