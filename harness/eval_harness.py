"""
Automated evaluation trial harness (runs entirely on the Mac).

One run = one model. Start the matching Pi-side server first (see copies/), then run:

    conda activate smolvla
    python eval_harness.py --model 1cam
    python eval_harness.py --model 2cam
    python eval_harness.py --model side_claw

For each of N_TRIALS (default 50) trials:
  1. Torque is released the moment a trial ends (see below), through the whole
     RESET_COUNTDOWN_S (default 8s) reset countdown, and re-engaged right before the next
     trial starts. (Requires the Pi server to support the "torque" command — use the
     *_EVAL / RECONSTRUCTED variants in copies/, not robot_server_side_claw_VERIFIED.py.)
  2. Recording + timer start automatically once torque is back on.
  3. A trial ends one of two ways:
       - you press 'n' the moment you judge the attempt is over (success, failure, whatever —
         you decide by watching it), or
       - TRIAL_TIMEOUT_S (default 120s / 2 min) elapses with no 'n' press — logged as an
         overtime trial. The arm is NOT auto-returned anywhere; torque is released and you
         reposition it by hand, same as any other trial.
     ('q' quits the whole session instead of ending just this trial.)
  4. Ended by 'n' → the full 1-4 rubric. Overtime → a reduced 3-choice question (completed but
     too slow / grasped or touched it but didn't finish / never touched it at all) → logged as
     "timeout_success_slow" / "timeout_grasped" / "timeout_no_touch", duration logged at the
     max time.
  5. Everything else (video, timing, latency, clamp events) is logged automatically.
  6. The next trial's reset countdown starts right after grading — no separate confirmation
     step in between.

After all trials for a model, a summary table is generated (see summarize_eval.py).

Clamp-event detection needs no Pi-side clamp counter: every Pi server variant in copies/
already returns `action_sent` (the value actually executed, possibly clamped) in its action
ack. Comparing that to the action we requested is enough to detect and count clamp events.
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

MODEL_CONFIGS = {
    "1cam": {
        "label": "1 Camera (left side)",
        "checkpoint": PROJECT_ROOT / "robot_datasets" / "smolvla_screwdriver_full"
        / "checkpoints" / "020000" / "pretrained_model",
        "checkpoint_step": 20000,
        "cameras": ["front"],                       # dataset feature: observation.images.front
        "response_frame_key": {"front": "front_frame"},
    },
    "2cam": {
        "label": "2 Cameras (side left + side right)",
        "checkpoint": PROJECT_ROOT / "robot_datasets" / "2 side cameras"
        / "smolvla_screwdriver_2cam_50k" / "checkpoints" / "050000" / "pretrained_model",
        "checkpoint_step": 50000,
        "cameras": ["front", "side"],
        "response_frame_key": {"front": "front_frame", "side": "side_frame"},
    },
    "side_claw": {
        "label": "Side + Claw",
        "checkpoint": PROJECT_ROOT / "robot_datasets" / "Claw and 1 side camera"
        / "smolvla_screwdriver_side_claw_20k" / "checkpoints" / "020000" / "pretrained_model",
        "checkpoint_step": 20000,
        "cameras": ["side", "claw"],
        "response_frame_key": {"side": "side_frame", "claw": "claw_frame"},
    },
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
    left waiting to receive and will reject the next send() until it's recreated — this is why
    the reference run_smolvla_live.py has its own reconnect_socket() helper. Without this, one
    dropped response would permanently stall the rest of a 50-trial session.
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
        (6,)-shaped array in JOINTS order.

        `robot.send_action()` (lerobot/robots/so_follower/so_follower.py) returns a dict keyed
        by joint name, e.g. {"shoulder_pan.pos": ...}, and the side+claw and front+side Pi
        servers forward that dict as-is in `action_sent`. The 1-camera eval server instead
        sends a plain array it built from its own SAFE_LIMITS clipping. Handle both shapes.
        """
        try:
            self.socket.send(pickle.dumps({"cmd": "action", "action": action_np}))
            ack = pickle.loads(self.socket.recv())
        except zmq.error.Again:
            self.reconnect()
            return None, "timeout"

        if not ack.get("ok", False):
            if ack.get("estop"):
                # Motor-load safety tripped and disabled torque — every action gets silently
                # rejected after this until cleared. Surface it loudly and try to self-heal.
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


def run_trial(trial_id, model_key, cfg, policy, preprocessor, postprocessor,
              device, pi, video_dir):
    if hasattr(policy, "reset"):
        policy.reset()  # clear the action-chunk queue so this trial isn't chasing stale state

    cameras = cfg["cameras"]
    response_key = cfg["response_frame_key"]

    # Frames are buffered in memory and encoded to video *after* the trial, once the true
    # elapsed time is known, so playback speed matches real time rather than a guessed fps.
    # Worst case (~90s at a generous 10Hz, 2 cameras) is roughly 1.5GB of RAM, released
    # immediately after each trial — fine on this machine, but noted here deliberately.
    buffered_frames = {cam: [] for cam in cameras}
    video_filenames = {cam: f"trial_{trial_id:03d}_{cam}.mp4" for cam in cameras}

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
            state, frames, err = pi.get_observation(cameras, response_key)
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

            observation = {f"observation.images.{cam}": frames[cam] for cam in cameras}
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
    for cam in cameras:
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
    parser.add_argument("--model", required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--start-trial", type=int, default=1,
                         help="Resume at this trial number instead of 1 (won't overwrite "
                              "earlier trials' rows/videos already in trial_log.csv).")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    print(f"Model: {cfg['label']}  (checkpoint step {cfg['checkpoint_step']})")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy, preprocessor, postprocessor = load_policy(cfg["checkpoint"], device)

    context = zmq.Context()
    pi = PiClient(context, PI_IP, PORT)

    model_dir = RESULTS_ROOT / args.model
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
        state, _, err = pi.get_observation(cfg["cameras"], cfg["response_frame_key"])
        if state is not None:
            pi.send_action(state)
        else:
            print(f"  [WARN] could not sync position before starting ({err}).")
        print("  torque re-engaged, recording started.        ")
        print("  press 'n' when the attempt is over, 'r' to redo this trial, or 'q' to quit.")

        result = run_trial(
            trial_id, args.model, cfg, policy, preprocessor, postprocessor,
            device, pi, video_dir,
        )

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
            trial_id, datetime.now().isoformat(timespec="seconds"), args.model,
            cfg["checkpoint_step"], outcome, round(result["duration"], 2),
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
