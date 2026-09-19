# Raspberry Pi server for the side+claw SO-101 config, with a "torque" command so the Mac
# harness can release/re-engage the servos between trials. Ran all 50 published side+claw trials.

import pickle
import numpy as np
import zmq

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROBOT_PORT = (
    "/dev/serial/by-id/"
    "usb-1a86_USB_Single_Serial_5AE6080769-if00"
)
ROBOT_ID = "my_follower_arm"
PORT = 5555

CAMERAS = {
    "side": OpenCVCameraConfig(
        index_or_path=2, width=640, height=480, fps=30
    ),
    "claw": OpenCVCameraConfig(
        index_or_path=0, width=640, height=480, fps=30
    ),
}

config = SO101FollowerConfig(
    port=ROBOT_PORT,
    id=ROBOT_ID,
    cameras=CAMERAS,
    use_degrees=True,
    max_relative_target=15.0,
    disable_torque_on_disconnect=False,
)

robot = SO101Follower(config)
robot.connect()

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.setsockopt(zmq.LINGER, 0)
socket.bind(f"tcp://*:{PORT}")

print("SO-101 SIDE + CLAW robot server (eval variant — supports torque release)")
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
                if "side" not in obs:
                    raise RuntimeError("SIDE camera missing.")
                if "claw" not in obs:
                    raise RuntimeError("CLAW camera missing.")

                state = np.array(
                    [obs[joint] for joint in JOINTS],
                    dtype=np.float32,
                )
                side_frame = obs["side"]
                claw_frame = obs["claw"]

                socket.send(pickle.dumps({
                    "ok": True,
                    "state": state,
                    "side_frame": side_frame,
                    "claw_frame": claw_frame,
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

                action_dict = {
                    joint: float(requested[i])
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
