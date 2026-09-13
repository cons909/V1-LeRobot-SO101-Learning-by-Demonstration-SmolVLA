"""
Automated evaluation trial harness — 1 CAMERA (front) model only.

Standalone copy of eval_harness.py's logic, dedicated to this one model — does not import
from or modify eval_harness.py. Behaves identically to it (same trial flow, same fixes):
policy resets each trial, torque syncs to current position before each trial starts, 'n' ends
a trial, 'r' redoes it (not logged), 'q' quits, overtime at TRIAL_TIMEOUT_S gets the reduced
3-choice question, live preview throttled to every PREVIEW_EVERY_N loop.

Start the Pi server first:
    ssh <user>@pi1.local
    source ~/servoenv/bin/activate
    python ~/robot_server_1cam_eval.py     (copy of copies/robot_server_1cam_EVAL.py)

Then on the Mac:
    conda activate smolvla
    python eval_harness_1cam.py [--trials N] [--start-trial N]
"""

import argparse
import csv
import pickle
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import zmq

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

PROJECT_ROOT = Path(
    "/path/to/your/project/root"  # update to your local checkpoint/data root
)
RESULTS_ROOT = Path(__file__).parent.parent / "results"

MODEL_KEY = "1cam"
MODEL_LABEL = "1 Camera (left side)"
CHECKPOINT = PROJECT_ROOT / "robot_datasets" / "smolvla_screwdriver_full" \
    / "checkpoints" / "020000" / "pretrained_model"
CHECKPOINT_STEP = 20000
CAMERAS = ["front"]                          # dataset feature: observation.images.front
RESPONSE_FRAME_KEY = {"front": "front_frame"}

TASK = "Pick up the blue screwdriver and place it into the gray bin."
PI_IP = "192.168.1.42"   # set this to your Pi's current IP address
PORT = 5555

N_TRIALS = 50
TRIAL_TIMEOUT_S = 180      # 3 minutes — trials that hit this are logged as overtime
RESET_COUNTDOWN_S = 8

CLAMP_TOLERANCE = 0.5     # deg/percent difference between requested vs. executed action
PREVIEW_EVERY_N = 3       # only render the live preview every Nth loop, keeps control loop fast

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

OUTCOME_CODES = {
    "1": "full_success",
    "2": "grasp_only",
    "3": "reach_only",
    "4": "no_engagement",
}


class NonBlockingKey:
    """Peek for a single keypress on the Mac's local terminal without blocking the loop."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def poll(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def load_policy(checkpoint_path: Path, device):
    required_files = ["config.json", "model.safetensors",
                       "policy_preprocessor.json", "policy_postprocessor.json"]
    for filename in required_files:
        path = checkpoint_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required checkpoint file: {path}")

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy = policy.to(device)
    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


class PiClient:
    """Owns the ZMQ REQ socket to the Pi and reconnects transparently on timeout.

    A REQ socket enforces strict send/recv alternation. If a recv() times out, the socket is
    left waiting to receive and will reject the next send() until it's recreated.
    """

    def __init__(self, context, pi_ip, port):
        self.context = context
        self.pi_ip = pi_ip
        self.port = port
        self.socket = self._connect()

    def _connect(self):
        s = self.context.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, 5000)
        s.setsockopt(zmq.SNDTIMEO, 5000)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://{self.pi_ip}:{self.port}")
        return s

    def reconnect(self):
        try:
            self.socket.close(0)
        except Exception:
            pass
        self.socket = self._connect()

    def get_observation(self, cameras, response_key):
        """Returns (state, {camera_name: frame}, error)."""
        try:
            self.socket.send(pickle.dumps({"cmd": "get_obs"}))
            response = pickle.loads(self.socket.recv())
        except zmq.error.Again:
            self.reconnect()
            return None, None, "timeout"

        if not response.get("ok", False):
            return None, None, response.get("error", "not_ok, no error given")

        frames = {}
        for cam in cameras:
            key = response_key[cam]
            if key not in response:
                return None, None, f"missing_{key}"
            frame = cv2.resize(response[key], (640, 480))
            frames[cam] = frame

        state = np.asarray(response["state"], dtype=np.float32)
        if state.shape != (6,):
            return None, None, f"bad_state_shape_{state.shape}"

        return state, frames, None

    def send_action(self, action_np):
        """Returns (action_sent, error) — action_sent is what the Pi actually executed, as a
        (6,)-shaped array in JOINTS order. robot.send_action() returns a dict keyed by joint
        name for the reconstructed/verified servers; this 1cam server sends a plain array
        built from its own SAFE_LIMITS clipping. Handle both shapes."""
        try:
            self.socket.send(pickle.dumps({"cmd": "action", "action": action_np}))
            ack = pickle.loads(self.socket.recv())
        except zmq.error.Again:
            self.reconnect()
            return None, "timeout"

        if not ack.get("ok", False):
            if ack.get("estop"):
                # The 1cam server's own load-based safety (HOLD/E-STOP) tripped and disabled
                # torque — every action gets silently rejected after this until cleared. There
                # was previously no visibility into this at all (the arm would just stop moving
                # with nothing printed) and no way to recover without restarting the whole Pi
                # process. Now: surface it loudly and try to self-heal automatically.
                print("  [ESTOP] Motor load safety tripped — arm is not moving. "
                      "Attempting automatic reset...")
                if self.reset_estop():
                    print("  [ESTOP] Cleared automatically. Retrying this action once...")
                    return self._send_action_once(action_np)
                else:
                    print("  [ESTOP] Could not clear automatically — restart the Pi server.")
            return None, ack.get("error", "rejected")

        return self._parse_action_sent(ack)

    def _send_action_once(self, action_np):
        try:
            self.socket.send(pickle.dumps({"cmd": "action", "action": action_np}))
            ack = pickle.loads(self.socket.recv())
        except zmq.error.Again:
            self.reconnect()
            return None, "timeout"
        if not ack.get("ok", False):
            return None, ack.get("error", "rejected")
        return self._parse_action_sent(ack)

    def _parse_action_sent(self, ack):
        raw = ack.get("action_sent")
        if raw is None:
            return None, "no_action_sent_in_ack"

        if isinstance(raw, dict):
            try:
                action_sent = np.array([raw[j] for j in JOINTS], dtype=np.float32)
            except KeyError as exc:
                return None, f"action_sent dict missing key: {exc}"
        else:
            action_sent = np.asarray(raw, dtype=np.float32)

        if action_sent.shape != (6,):
            return None, f"bad_action_sent_shape_{action_sent.shape}"

        return action_sent, None

    def reset_estop(self):
        """Returns True if the E-STOP is now cleared (or wasn't tripped in the first place)."""
        try:
            self.socket.send(pickle.dumps({"cmd": "reset_estop"}))
            ack = pickle.loads(self.socket.recv())
        except zmq.error.Again:
            self.reconnect()
            return False
        return ack.get("ok", False)

    def set_torque(self, enable: bool):
        """Returns True on success. Best-effort: prints a warning and returns False on failure
        rather than raising, so a torque-toggle hiccup doesn't crash a session mid-trial."""
        try:
            self.socket.send(pickle.dumps({"cmd": "torque", "enable": enable}))
            ack = pickle.loads(self.socket.recv())
        except zmq.error.Again:
            self.reconnect()
            print("  [WARN] torque command timed out, reconnected.")
            return False
        if not ack.get("ok", False):
            print(f"  [WARN] torque command failed: {ack.get('error')}")
            return False
        return True

    def close(self):
        try:
            self.socket.close(0)
        except Exception:
            pass


def run_trial(trial_id, policy, preprocessor, postprocessor, device, pi, video_dir):
    if hasattr(policy, "reset"):
        policy.reset()  # clear the action-chunk queue so this trial isn't chasing stale state

    buffered_frames = {cam: [] for cam in CAMERAS}
    video_filenames = {cam: f"trial_{trial_id:03d}_{cam}.mp4" for cam in CAMERAS}

    latencies = []
    clamp_events = 0
    stop_reason = None
    loop_count = 0

    trial_start = time.time()

    with NonBlockingKey() as keys:
        while True:
            elapsed = time.time() - trial_start
            if elapsed >= TRIAL_TIMEOUT_S:
                stop_reason = "timeout"
                break

            t0 = time.time()
            state, frames, err = pi.get_observation(CAMERAS, RESPONSE_FRAME_KEY)
            if state is None:
                print(f"  [WARN] get_obs failed ({err}), retrying...")
                key = keys.poll()
                if key == "n":
                    stop_reason = "manual_end"
                    break
                if key == "r":
                    stop_reason = "redo"
                    break
                if key == "q":
                    stop_reason = "quit"
                    break
                time.sleep(0.2)
                continue

            loop_count += 1
            for cam, frame in frames.items():
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                buffered_frames[cam].append(bgr)
                if loop_count % PREVIEW_EVERY_N == 0:
                    cv2.imshow(f"Live — {cam}", bgr)
            if loop_count % PREVIEW_EVERY_N == 0:
                cv2.waitKey(1)

            observation = {f"observation.images.{cam}": frames[cam] for cam in CAMERAS}
            observation["observation.state"] = state
            observation = prepare_observation_for_inference(observation, device, task=TASK)
            observation = preprocessor(observation)

            with torch.inference_mode():
                action = policy.select_action(observation)
            action = postprocessor(action)
            if isinstance(action, dict):
                action = action["action"]
            action_np = action.squeeze().detach().cpu().numpy().astype(np.float32)

            action_sent, err = pi.send_action(action_np)
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)

            if action_sent is not None:
                if np.max(np.abs(action_np - action_sent)) > CLAMP_TOLERANCE:
                    clamp_events += 1
            else:
                # Previously silent — the arm would just stop moving with nothing printed.
                print(f"  [WARN] action rejected ({err}) — arm did not move this step.")

            key = keys.poll()
            if key == "n":
                stop_reason = "manual_end"
                break
            if key == "r":
                stop_reason = "redo"
                break
            if key == "q":
                stop_reason = "quit"
                break

    duration = time.time() - trial_start

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for cam in CAMERAS:
        frame_list = buffered_frames[cam]
        actual_fps = (len(frame_list) / duration) if duration > 0 and frame_list else 10.0
        actual_fps = max(actual_fps, 1.0)
        out_path = video_dir / video_filenames[cam]
        writer = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (640, 480))
        for f in frame_list:
            writer.write(f)
        writer.release()

    avg_latency = float(np.mean(latencies)) if latencies else float("nan")
    return {
        "duration": duration,
        "clamp_events": clamp_events,
        "avg_latency_ms": avg_latency,
        "stop_reason": stop_reason,
        "video_filenames": video_filenames,
    }


def prompt_outcome():
    print("\nGrade this trial:")
    print("  1 = Full success — grasped and placed in the bin")
    print("  2 = Grasp only — grasped but dropped/misplaced before the bin")
    print("  3 = Reach only — moved toward object, no stable grasp")
    print("  4 = No meaningful engagement — stayed near average/home pose")
    while True:
        choice = input("Outcome (1-4): ").strip()
        if choice in OUTCOME_CODES:
            return OUTCOME_CODES[choice]
        print("Please enter 1, 2, 3, or 4.")


def prompt_timeout_outcome():
    print("\nThis trial hit the time limit (overtime). Grade it:")
    print("  1 = It grasped the object and placed it in the bin, just took too long")
    print("  2 = It grasped/touched the object at some point but didn't finish")
    print("  3 = It never touched the object")
    print("  4 = The object fell out of bounds — it never got a chance to grab it")
    while True:
        choice = input("Outcome (1-4): ").strip()
        if choice == "1":
            return "timeout_success_slow"
        if choice == "2":
            return "timeout_grasped"
        if choice == "3":
            return "timeout_no_touch"
        if choice == "4":
            return "timeout_object_oob"
        print("Please enter 1, 2, 3, or 4.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--start-trial", type=int, default=1,
                         help="Resume at this trial number instead of 1 (won't overwrite "
                              "earlier trials' rows/videos already in trial_log.csv).")
    args = parser.parse_args()

    print(f"Model: {MODEL_LABEL}  (checkpoint step {CHECKPOINT_STEP})")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy, preprocessor, postprocessor = load_policy(CHECKPOINT, device)

    context = zmq.Context()
    pi = PiClient(context, PI_IP, PORT)

    model_dir = RESULTS_ROOT / MODEL_KEY
    video_dir = model_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_path = model_dir / "trial_log.csv"

    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    if write_header:
        writer.writerow([
            "trial_id", "date", "model_config", "checkpoint_step", "outcome",
            "time_to_completion_sec", "safety_clamp_events", "avg_loop_latency_ms",
            "stop_reason", "notes", "video_filename",
        ])
        csv_file.flush()

    pi.set_torque(False)
    print("Torque released — arm is free to move by hand.")

    trial_id = args.start_trial
    while trial_id <= args.trials:
        print(f"\n=== Trial {trial_id}/{args.trials} — reset the environment ===")
        for s in range(RESET_COUNTDOWN_S, 0, -1):
            print(f"  starting in {s}...", end="\r")
            time.sleep(1)
        pi.set_torque(True)
        # Sync the servo's target to wherever it actually is right now before anything else
        # moves it — otherwise it snaps toward its last commanded target from before torque
        # was released, since enable_torque() doesn't do this on its own.
        state, _, err = pi.get_observation(CAMERAS, RESPONSE_FRAME_KEY)
        if state is not None:
            pi.send_action(state)
        else:
            print(f"  [WARN] could not sync position before starting ({err}).")
        print("  torque re-engaged, recording started.        ")
        print("  press 'n' when the attempt is over, 'r' to redo this trial, or 'q' to quit.")

        result = run_trial(trial_id, policy, preprocessor, postprocessor, device, pi, video_dir)

        pi.set_torque(False)
        print("Torque released — arm is free to move by hand.")

        if result["stop_reason"] == "quit":
            print("Quitting early.")
            break

        if result["stop_reason"] == "redo":
            print(f"Redoing trial {trial_id} — not logged.")
            continue

        if result["stop_reason"] == "timeout":
            outcome = prompt_timeout_outcome()
        else:
            outcome = prompt_outcome()

        video_names = ";".join(result["video_filenames"].values())
        writer.writerow([
            trial_id, datetime.now().isoformat(timespec="seconds"), MODEL_KEY,
            CHECKPOINT_STEP, outcome, round(result["duration"], 2),
            result["clamp_events"], round(result["avg_latency_ms"], 1),
            result["stop_reason"], "", video_names,
        ])
        csv_file.flush()

        print(f"Logged trial {trial_id}: outcome={outcome}, "
              f"duration={result['duration']:.1f}s, clamps={result['clamp_events']}, "
              f"latency={result['avg_latency_ms']:.0f}ms")

        trial_id += 1

    csv_file.close()
    pi.close()
    context.term()
    cv2.destroyAllWindows()

    print(f"\nDone. Log: {csv_path}")
    print("Run summarize_eval.py to generate the results table.")


if __name__ == "__main__":
    main()
