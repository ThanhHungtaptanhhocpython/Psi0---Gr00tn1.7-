"""Closed-loop evaluation of a finetuned Psi0 checkpoint in the box_pnp MuJoCo scene.

The policy runs on a GPU (colab_psi0_server.ipynb serves it over HTTP through a tunnel); this
script runs the simulation locally, optionally in the MuJoCo viewer, and asks the server for an
action chunk from the ego image + 43-D state every --exec-steps frames:

  # model on Colab (host:port printed by colab_psi0_server.ipynb), 10 unseen scenes, live viewer
  .venv-datagen/Scripts/python scripts/datagen/box_pnp/closed_loop_eval.py \
      --server http://HOST:PORT --episodes 10 --viewer --out .data/eval/ckpt8000

  # the simulation side alone, no model: replay a generated episode's recorded actions in its scene
  .venv-datagen/Scripts/python scripts/datagen/box_pnp/closed_loop_eval.py \
      --replay .data/box_pnp_tray_140 --replay-episode 0 --viewer

Everything the policy sees or drives matches the data generation (generate.py / episode.py):
the scene from sample_params(seed), the same settle before control starts, the 640x480
head_camera image downscaled to the 320x240 the dataset was exported at, the 43-D state in the
0810 joint order (export_psi0.py), the SONIC 1.0 decoder at 50 Hz with the token latched at
30 Hz, PD hand targets, and the same grasp assist: a weld engages once the right hand is >= 30 %
closed with a finger/thumb on the box in the pinch pocket, and drops when it opens below 30 %.
The simulation pauses while a chunk is requested, so network latency does not change the result.

A scene counts as solved once the box rests in the tray (not held, < 3 cm/s) for 1 s; it fails
on a fall (pelvis below 0.28 m), on the box hitting the floor, or after --max-seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from base64 import b64decode, b64encode
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from episode import INSTRUCTION, Config, Episode, EpisodeFailed  # noqa: E402
from export_psi0 import state_permutation  # noqa: E402
from generate import SONIC_DIR  # noqa: E402
from render import open_video  # noqa: E402
from scene import CTRL_DT, IMG_H, IMG_W, RECORD_FPS, sample_params  # noqa: E402
from sonic_policy import (DEFAULT_ANGLES_MJ, HAND_KD, HAND_KP, HAND_OPEN, RIGHT_HAND_CLOSED,  # noqa: E402
                          SonicPolicy, pd_torque)

SUBSTEPS = int(round(CTRL_DT / 0.005))
POLICY_W, POLICY_H = 320, 240            # export_psi0.py --video-size
IMAGE_KEY = "observation.images.egocentric"
TOKEN, HANDS = 64, 14                    # action row: [SONIC token 64 | left hand 7 | right hand 7]
GRASP_CLOSURE = 0.3                      # episode.py: the weld engages from 30 % closed
FIRST_UNSEEN_SEED = 1_000_000            # box_pnp_tray_140 used seeds 10 000-510 010


# ------------------------------------------------------------------ policies
def _np_encode(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"__numpy__": b64encode(a.tobytes()).decode(), "dtype": dtype_to_descr(a.dtype), "shape": list(a.shape)}


def _np_decode(d: dict) -> np.ndarray:
    a = np.frombuffer(b64decode(d["__numpy__"]), descr_to_dtype(d["dtype"]))
    return a.reshape(d["shape"]) if d["shape"] else a[0]


class HttpPolicy:
    """POST /act of serve_psi0_sonic_http (psi.deploy.helpers.RequestMessage wire format)."""

    def __init__(self, url: str, instruction: str, timeout: float):
        self.url, self.instruction, self.timeout = url.rstrip("/"), instruction, timeout
        with urllib.request.urlopen(f"{self.url}/health", timeout=timeout) as r:
            if json.loads(r.read()).get("status") != "ok":
                raise RuntimeError(f"{self.url}/health is not ok")

    def chunk(self, frame: int, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        payload = {"image": {IMAGE_KEY: _np_encode(image)}, "instruction": self.instruction, "history": {},
                   "state": {"states": _np_encode(state.astype(np.float32))}, "condition": {}, "gt_action": [],
                   "dataset_name": "box_pnp", "timestamp": str(time.time())}
        req = urllib.request.Request(f"{self.url}/act", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        if isinstance(body, str):  # the server reports a failed request as a JSON string '{"status": ...}'
            body = json.loads(body)
        if "action" not in body:
            raise RuntimeError(f"server error: {body}")
        a = np.asarray(_np_decode(body["action"]), dtype=np.float32)
        return a.reshape(-1, a.shape[-1])


class ReplayPolicy:
    """The recorded actions of a generated episode, served chunk by chunk like the model would."""

    def __init__(self, actions: np.ndarray, chunk_size: int = 30):
        self.actions, self.n = actions, chunk_size

    def chunk(self, frame: int, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        idx = np.minimum(np.arange(frame, frame + self.n), len(self.actions) - 1)
        return self.actions[idx]


# ------------------------------------------------------------------ closed loop
class ClosedLoopEpisode(Episode):
    """Episode's scene, sensing and grasp weld; the actions come from a policy instead of the expert."""

    def __init__(self, params, sonic: SonicPolicy, policy, args, viewer_ok: bool):
        super().__init__(params, sonic, None, Config(render=False, debug_render=False))
        self.policy, self.args = policy, args
        self.mode = "settle"
        self.perm = state_permutation()
        self.lhand = HAND_OPEN.copy()
        self.queue: list[np.ndarray] = []
        self.since_request = 0
        self.frame = 0
        self.latency: list[float] = []
        self.cam = mujoco.Renderer(self.m, IMG_H, IMG_W)
        self.videos = None
        if args.out and args.video:
            stem = Path(args.out) / f"seed_{params.seed:06d}"
            self.videos = (open_video(Path(f"{stem}_ego.mp4")), open_video(Path(f"{stem}_third.mp4")))
        self.viewer = None
        if viewer_ok:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.m.camera("debug_cam").id
        self.wall0 = None

    # -- observation and actions
    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        self.cam.update_scene(self.d, camera="head_camera")
        full = self.cam.render()
        image = np.asarray(Image.fromarray(full).resize((POLICY_W, POLICY_H), Image.BOX))
        return image, self.d.qpos[7:50][self.perm].astype(np.float32)

    def next_action(self) -> np.ndarray:
        if not self.queue or self.since_request >= self.args.exec_steps:
            image, state = self.observe()
            t0 = time.time()
            chunk = self.policy.chunk(self.frame, image, state)
            self.latency.append(time.time() - t0)
            if chunk.shape[1] != TOKEN + HANDS:
                raise RuntimeError(f"expected {TOKEN + HANDS}-D actions, got {chunk.shape}")
            self.queue, self.since_request = list(chunk), 0
        if self.videos:
            image, _ = self.observe()
            self.videos[0].append_data(image)
            self.cam.update_scene(self.d, camera="debug_cam")
            self.videos[1].append_data(self.cam.render())
        self.since_request += 1
        self.frame += 1
        return self.queue.pop(0)

    def _step_tick(self):
        d, ix = self.d, self.ix
        self.pol.log_state(d.qpos[ix.body_qadr].copy(), d.qvel[ix.body_vadr].copy(), d.qpos[3:7].copy(),
                           d.qvel[3:6].copy())
        if self.t + 1e-9 >= self.next_token_t:
            if self.mode == "settle":   # the expert's reference, as before recording starts in episode.py
                self.token = self.pol.encode(self.ref.frames, min(self.tick, len(self.ref) - 1), d.qpos[3:7].copy())
            else:
                a = self.next_action()
                token = a[:TOKEN]
                if self.args.clip_token:   # what the robot-side client does before publishing
                    token = np.clip(token, -0.625, 0.625)
                self.token = (np.round(token * 16) / 16).astype(np.float32)   # SONIC's 1/16 FSQ grid
                self.lhand, self.hand_target = a[TOKEN:TOKEN + 7].copy(), a[TOKEN + 7:].copy()
            self.next_token_t += 1.0 / self.cfg.token_rate_hz
        q_des = self.pol.decode(self.token)
        for _ in range(SUBSTEPS):
            tau = pd_torque(q_des, d.qpos[ix.body_qadr], d.qvel[ix.body_vadr])
            d.ctrl[ix.body_act] = np.clip(tau, ix.ctrl_range[ix.body_act, 0], ix.ctrl_range[ix.body_act, 1])
            for act, qa, va, tgt in ((ix.rhand_act, ix.rhand_qadr, ix.rhand_vadr, self.hand_target),
                                     (ix.lhand_act, ix.lhand_qadr, ix.lhand_vadr, self.lhand)):
                tq = HAND_KP * (tgt - d.qpos[qa]) - HAND_KD * d.qvel[va]
                d.ctrl[act] = np.clip(tq, ix.ctrl_range[act, 0], ix.ctrl_range[act, 1])
            mujoco.mj_step(self.m, d)
        self.tick += 1
        self.last_contacts = self._contacts()
        if self.mode == "policy":
            self.grasp_assist()
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt("viewer closed")
            self.viewer.sync()
            if self.args.realtime:
                self.wall0 = self.wall0 or (time.time() - self.t)
                time.sleep(max(0.0, self.wall0 + self.t - time.time()))

    def grasp_assist(self):
        closure = float(np.mean(self.hand_target[2:] / RIGHT_HAND_CLOSED[2:]))
        c = self.last_contacts
        if (not self.grasped and closure >= GRASP_CLOSURE and (c["thumb_box"] or c["finger_box"])
                and self._box_in_pocket(self.cfg.grasp_pocket_tol)):
            self._activate_weld()
        elif self.grasped and closure < GRASP_CLOSURE:
            self._deactivate_weld()

    # -- episode
    def settle(self):
        """episode.py's start: HOME pose with joint noise, 1.5 s hold; recording (= control) starts after."""
        m, d, ix, p = self.m, self.d, self.ix, self.p
        d.qpos[2] = 0.793
        d.qpos[3:7] = [np.cos(p.init_yaw / 2), 0, 0, np.sin(p.init_yaw / 2)]
        d.qpos[ix.body_qadr] = DEFAULT_ANGLES_MJ + self.rng.normal(0, p.init_joint_noise, 29)
        mujoco.mj_forward(m, d)
        q_home = d.qpos.copy(); q_home[ix.body_qadr] = DEFAULT_ANGLES_MJ; q_home[2] = 0.762
        self.ref.start(q_home)
        self.ref.add_hold(1.5)
        self.run_until_ref_end()
        if d.qpos[2] < 0.72:
            raise EpisodeFailed("home_settle_failed")

    def evaluate(self) -> dict:
        args, d = self.args, self.d
        t_wall = time.time()
        solved_since = None
        diag = {"grasp_at_s": None, "release_at_s": None, "box_max_lift_m": 0.0}   # where a failure got stuck
        box_z0, t0 = float(self.box_pose()[0][2]), self.t
        try:
            self.settle()
            self.mode, self.next_token_t, t0 = "policy", self.t, self.t
            while True:
                was_grasped = self.grasped
                self._step_tick()
                bp, _ = self.box_pose()
                if self.grasped and not was_grasped and diag["grasp_at_s"] is None:
                    diag["grasp_at_s"] = round(self.t - t0, 2)
                if was_grasped and not self.grasped:
                    diag["release_at_s"] = round(self.t - t0, 2)
                diag["box_max_lift_m"] = max(diag["box_max_lift_m"], float(bp[2]) - box_z0)
                bv = float(np.linalg.norm(d.qvel[self.ix.box_vadr:self.ix.box_vadr + 3]))
                if d.qpos[2] < 0.28:
                    raise EpisodeFailed("fall")
                if bp[2] < 0.1 and self.last_contacts["box_floor"]:
                    raise EpisodeFailed("box_on_floor")
                resting = self.box_in_tray(resting=True) and bv < 0.03 and not self.grasped
                solved_since = (solved_since if solved_since is not None else self.t) if resting else None
                if solved_since is not None and self.t - solved_since >= 1.0:
                    ok, reason = True, None
                    break
                if self.t - t0 > args.max_seconds:
                    raise EpisodeFailed("timeout")
        except EpisodeFailed as e:
            ok, reason = False, str(e)
        finally:
            if self.videos:
                for v in self.videos:
                    v.close()
            if self.viewer is not None:
                self.viewer.close()
        lat = np.asarray(self.latency) if self.latency else np.zeros(1)
        bp = self.box_pose()[0]
        diag.update(box_max_lift_m=round(diag["box_max_lift_m"], 3), end_grasped=bool(self.grasped),
                    end_box_in_tray=self.box_in_tray(), end_box_to_target_m=round(self.box_place_xy_err(), 3),
                    end_box_z_m=round(float(bp[2]), 3), end_pelvis_xy=[round(float(v), 3) for v in d.qpos[:2]],
                    end_rhand_closure=round(float(np.mean(self.hand_target[2:] / RIGHT_HAND_CLOSED[2:])), 2))
        return {"seed": self.p.seed, "success": ok, "fail_reason": reason, "policy_seconds": round(self.frame / RECORD_FPS, 2),
                "requests": len(self.latency), "latency_mean_s": round(float(lat.mean()), 3),
                "latency_max_s": round(float(lat.max()), 3), "wall_seconds": round(time.time() - t_wall, 1), **diag}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--server", help="policy server URL, e.g. http://r1.a.pinggy.link:41234")
    src.add_argument("--replay", type=Path, help="generated dataset whose recorded actions are replayed")
    ap.add_argument("--replay-episode", type=int, default=0, help="--replay: episode index")
    ap.add_argument("--episodes", type=int, default=10, help="--server: number of scenes")
    ap.add_argument("--seed-start", type=int, default=FIRST_UNSEEN_SEED, help="--server: first scene seed")
    ap.add_argument("--level", type=int, default=0, help="randomization level of sample_params")
    ap.add_argument("--exec-steps", type=int, default=15,
                    help="frames (30 Hz) executed from each chunk before the next request")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="policy time limit per scene")
    ap.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per request (first one loads kernels)")
    ap.add_argument("--clip-token", action="store_true", help="clip tokens to +-0.625 like the robot client")
    ap.add_argument("--viewer", action="store_true", help="show the scene in the MuJoCo viewer")
    ap.add_argument("--realtime", action="store_true", help="--viewer: pace the simulation to wall-clock time")
    ap.add_argument("--out", type=Path, default=None, help="results.jsonl (+ videos with --video) go here")
    ap.add_argument("--video", action="store_true", help="--out: write ego + third-person mp4 per scene")
    ap.add_argument("--threads", type=int, default=4, help="onnxruntime threads for SONIC")
    args = ap.parse_args()
    if args.video and not args.out:
        ap.error("--video needs --out")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    sonic = SonicPolicy(SONIC_DIR, threads=args.threads)
    if args.replay:
        meta = [json.loads(l) for l in (args.replay / "meta/episode_meta.jsonl").read_text().splitlines() if l.strip()]
        import pandas as pd
        df = pd.read_parquet(args.replay / f"data/chunk-000/episode_{args.replay_episode:06d}.parquet", columns=["action"])
        policy = ReplayPolicy(np.stack(df["action"].values).astype(np.float32))
        seeds = [meta[args.replay_episode]["seed"]]
        print(f"replaying episode {args.replay_episode} (seed {seeds[0]}, {len(df)} frames)")
    else:
        policy = HttpPolicy(args.server, INSTRUCTION, args.timeout)
        seeds = list(range(args.seed_start, args.seed_start + args.episodes))
        print(f"server {args.server} is up; {len(seeds)} scenes from seed {seeds[0]}")

    results = []
    for seed in seeds:
        sonic.reset()
        ep = ClosedLoopEpisode(sample_params(seed, level=args.level), sonic, policy, args, args.viewer)
        try:
            r = ep.evaluate()
        except KeyboardInterrupt:
            print("stopped")
            break
        results.append(r)
        print(("SUCCESS" if r["success"] else f"FAIL ({r['fail_reason']})") + f"  seed {seed}: {r['policy_seconds']} s, "
              f"{r['requests']} requests, latency {r['latency_mean_s']} s (max {r['latency_max_s']}) | "
              f"grasp {r['grasp_at_s']} s, release {r['release_at_s']} s, lift {r['box_max_lift_m']} m, "
              f"end: in tray {r['end_box_in_tray']}, held {r['end_grasped']}, box->target {r['end_box_to_target_m']} m, "
              f"hand closure {r['end_rhand_closure']}", flush=True)
        if args.out:
            with open(args.out / "results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
    if results:
        n_ok = sum(r["success"] for r in results)
        print(f"\n{n_ok}/{len(results)} solved ({100 * n_ok / len(results):.0f} %)")


if __name__ == "__main__":
    main()
