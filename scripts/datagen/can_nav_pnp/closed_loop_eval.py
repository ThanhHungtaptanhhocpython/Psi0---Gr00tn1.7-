"""Closed-loop evaluation of a finetuned Psi0 or GR00T N1.7 checkpoint in the can_nav_pnp MuJoCo scene: from
START, walk around the obstacle, pick the can off the pedestal, put it in the tray.

The policy runs on a GPU (Psi0: colab_psi0_server.ipynb serves it over HTTP; GR00T: Isaac-GR00T
gr00t/eval/run_gr00t_server.py over ZMQ, both through a tunnel); this script runs the simulation locally,
optionally in the MuJoCo viewer, and asks the server for an action chunk every --exec-steps frames:

  # Psi0 on Colab (host:port printed by colab_psi0_server.ipynb), 10 unseen scenes, live viewer
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py \
      --server http://HOST:PORT --episodes 10 --viewer --out .data/eval/psi0_ckpt

  # GR00T N1.7 (embodiment unitree_g1_sonic; once: uv pip install --python .venv-datagen/bin/python pyzmq msgpack)
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py \
      --gr00t HOST:PORT --episodes 10 --out .data/eval/gr00t_ckpt

  # the simulation side alone, no model: replay a generated episode's recorded actions in its scene
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py \
      --replay .data/can_nav_160 --replay-episode 0 --viewer

Everything the policy sees or drives matches the data generation (generate.py / episode.py):
the scene from sample_params(seed) with the pick/place layout drawn from the range generated episodes put
it in (sample_eval_layout, or --layout-from a dataset), the robot settled at START as before recording
starts, the 640x480 head_camera image downscaled to the 320x240 the datasets were exported at, the state in
the 0810 joint order (export_psi0.py; GR00T: + projected gravity, export_gr00t.py), the SONIC decoder (SONIC_VARIANT, default 1.1) at 50 Hz with the token latched at
30 Hz, PD hand targets, and the same grasp assist: a weld engages once the right hand is >= 30 %
closed with a finger/thumb on the can in the pinch pocket, and drops when it opens below 30 %.
The simulation pauses while a chunk is requested, so network latency does not change the result.

A scene counts as solved once the can rests in the tray (not held, < 3 cm/s) for 1 s, standing or lying (as in the
generated data); it fails on a fall (pelvis below 0.28 m), on touching the obstacle, on the can tipping over (> 45 deg)
outside the tray or landing on the floor / pedestal again after a lift, or after --max-seconds. The SONIC controller must be the one the data was generated with
(SONIC_VARIANT, default sonic_v1_1, see generate.py).

Evidence is ALWAYS kept, whatever the outcome (default --out: .data/eval/<policy>_<date>_<time>):
  <out>/seed_<seed>_<RESULT>/     RESULT = SUCCESS, FAIL_<reason>, ERROR (server/network) or STOPPED (Ctrl+C / viewer
                                  closed); the folder is named once the scene ends
      ego.mp4                     what the policy saw (head camera, 320x240), from the first policy step
      third.mp4                   third-person view of the whole scene (640x480)
      overview.png                12 moments of the scene, third-person + head camera, with the result
      result.json                 outcome + diagnostics (grasp / release time, can lift, where the can ended, latency)
      trace.npz                   per policy frame (30 Hz): time, robot qpos, can pose, executed action, contacts
  <out>/results.jsonl             one line per scene (also for errors / stops)
  <out>/summary.md, summary.json  running totals + a table linking every scene's folder, rewritten after each scene
  <out>/run_config.json           command-line arguments, scene seeds, git commit
--no-video skips the two mp4s (overview.png, result.json and trace.npz are still written). A server or network error
(e.g. the free tunnel expired) records the scene as ERROR and stops; resume with --seed-start <next seed> and the same
--out.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from base64 import b64decode, b64encode
from pathlib import Path

# The MuJoCo viewer (GLFW) tears down GLFW when it closes after a scene, which corrupted every image the default
# (GLFW) offscreen renderer produced afterwards: from the 2nd scene on, the policy saw pure noise and the videos were
# noise (cannav160-s11_ckpt8000, 2026-10-13). With --viewer the policy / video images come from EGL instead, which
# is independent of the viewer's GLFW (Mesa EGL works on the Intel iGPU too). MUJOCO_GL set by the user wins.
if "--viewer" in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")

import mujoco
import mujoco.viewer
import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from episode import INSTRUCTION, Config, Episode, EpisodeFailed  # noqa: E402
from export_psi0 import state_permutation  # noqa: E402
from generate import SONIC_DIR  # noqa: E402
from render import open_video  # noqa: E402
from scene import CTRL_DT, IMG_H, IMG_W, RECORD_FPS, SceneParams, sample_eval_layout, sample_params  # noqa: E402
from sonic_policy import (DEFAULT_ANGLES_MJ, HAND_KD, HAND_KP, HAND_OPEN, RIGHT_HAND_CLOSED,  # noqa: E402
                          SonicPolicy, pd_torque, quat_rotate)

SUBSTEPS = int(round(CTRL_DT / 0.005))
POLICY_W, POLICY_H = 320, 240            # export_psi0.py --video-size
IMAGE_KEY = "observation.images.egocentric"
TOKEN, HANDS = 64, 14                    # action row: [SONIC token 64 | left hand 7 | right hand 7]
GRASP_CLOSURE = 0.3                      # episode.py: the weld engages from 30 % closed
NOISE_LIMIT = 20.0                       # mean |horizontal pixel step| of a rendered image: scenes 1-4, the corrupted
                                         # (GL context lost) renders 38-47 -> abort instead of feeding noise to the policy
FIRST_UNSEEN_SEED = 1_000_000            # keep the generation seed ranges below this


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
                   "dataset_name": "can_pnp", "timestamp": str(time.time())}
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


class Gr00tPolicy:
    """get_action of Isaac-GR00T's PolicyServer (gr00t/policy/server_client.py) for the unitree_g1_sonic
    embodiment: msgpack over a ZMQ REQ socket, ndarrays sent as .npy bytes (the server's allow_pickle=False
    format) and received in msgpack_numpy's layout. Returns (horizon, 78) = [motion_token | left | right hand]."""

    STATE_KEYS = (("left_leg", 0, 6), ("right_leg", 6, 12), ("waist", 12, 15), ("left_arm", 15, 22),
                  ("right_arm", 22, 29), ("left_hand", 29, 36), ("right_hand", 36, 43), ("projected_gravity", 43, 46))
    ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")
    LANG_KEY = "annotation.human.task_description"

    def __init__(self, address: str, instruction: str, timeout: float):
        import msgpack
        import zmq
        self.msgpack, self.zmq, self.instruction = msgpack, zmq, instruction
        host, port = address.rsplit(":", 1)
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self.sock.setsockopt(zmq.SNDTIMEO, int(timeout * 1000))
        self.sock.connect(f"tcp://{host}:{port}")
        self._call("ping", None)

    @staticmethod
    def _enc(o):
        if isinstance(o, np.ndarray):
            buf = io.BytesIO(); np.save(buf, o, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(type(o))

    @staticmethod
    def _dec(o):
        g = lambda k: o.get(k, o.get(k.encode()))
        if g("__ndarray_class__"):
            return np.load(io.BytesIO(g("as_npy")), allow_pickle=False)
        if g("nd") is True:      # msgpack_numpy
            t = g("type"); t = t.decode() if isinstance(t, bytes) else t
            return np.frombuffer(g("data"), dtype=np.dtype(t)).reshape(g("shape"))
        return o

    def _call(self, endpoint: str, data):
        req = {"endpoint": endpoint, **({"data": data} if data is not None else {})}
        self.sock.send(self.msgpack.packb(req, default=self._enc))
        msg = self.sock.recv()
        if msg == b"ERROR":
            raise RuntimeError("GR00T server error")
        r = self.msgpack.unpackb(msg, object_hook=self._dec, raw=False)
        if isinstance(r, dict) and "error" in r:
            raise RuntimeError(f"GR00T server error: {r['error']}")
        return r

    def chunk(self, frame: int, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        obs = {"video": {"ego_view": image[None, None].astype(np.uint8)},
               "state": {k: state[None, None, a:b].astype(np.float32) for k, a, b in self.STATE_KEYS},
               "language": {self.LANG_KEY: [[self.instruction]]}}
        action, _ = self._call("get_action", {"observation": obs, "options": None})
        return np.concatenate([np.asarray(action[k], dtype=np.float32)[0] for k in self.ACTION_KEYS], axis=-1)


class ReplayPolicy:
    """The recorded actions of a generated episode, served chunk by chunk like the model would."""

    def __init__(self, actions: np.ndarray, chunk_size: int = 30):
        self.actions, self.n = actions, chunk_size

    def chunk(self, frame: int, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        idx = np.minimum(np.arange(frame, frame + self.n), len(self.actions) - 1)
        return self.actions[idx]


# ------------------------------------------------------------------ closed loop
class PolicyError(RuntimeError):
    """The policy server / network failed during a scene (its evidence is saved); carries the scene's result."""

    def __init__(self, result: dict):
        super().__init__(result.get("fail_reason"))
        self.result = result


class EvalStopped(KeyboardInterrupt):
    """Ctrl+C / viewer closed during a scene (its evidence is saved); carries the scene's result."""

    def __init__(self, result: dict):
        super().__init__(result.get("fail_reason"))
        self.result = result


class ClosedLoopEpisode(Episode):
    """Episode's scene, sensing and grasp weld; the actions come from a policy instead of the expert."""

    def __init__(self, params, sonic: SonicPolicy, policy, args, viewer_ok: bool):
        super().__init__(params, sonic, None, Config(render=False, debug_render=False))
        self.policy, self.args = policy, args
        self.with_gravity = isinstance(policy, Gr00tPolicy)
        self.mode = "settle"
        self.perm = state_permutation()
        self.lhand = HAND_OPEN.copy()
        self.queue: list[np.ndarray] = []
        self.since_request = 0
        self.low_requests = 0   # requests made under --exec-steps-low
        self.frame = 0
        self.latency: list[float] = []
        self.cam = mujoco.Renderer(self.m, IMG_H, IMG_W)
        # evidence of this scene, kept whatever the outcome (renamed to seed_<seed>_<RESULT> at the end)
        self.ev_dir = Path(args.out) / f"seed_{params.seed:06d}_RUNNING"
        if self.ev_dir.exists():
            import shutil
            shutil.rmtree(self.ev_dir)
        self.ev_dir.mkdir(parents=True)
        self.videos = None
        if args.video:
            self.videos = (open_video(self.ev_dir / "ego.mp4"), open_video(self.ev_dir / "third.mp4"))
        self.trace: dict[str, list] = {k: [] for k in ("t", "qpos", "box_pose", "action", "new_chunk", "contacts")}
        self.thumbs: list[tuple[float, np.ndarray, np.ndarray]] = []   # (t, third 320x240, ego 320x240) every 0.5 s
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
        state = self.d.qpos[7:50][self.perm]
        if self.with_gravity:   # export_gr00t.py: 0810 joints + projected gravity (episode.py _record)
            q = self.d.qpos[3:7]
            state = np.r_[state, quat_rotate(np.array([q[0], -q[1], -q[2], -q[3]]), np.array([0, 0, -1.0]))]
        return image, state.astype(np.float32)

    @staticmethod
    def _noise(img: np.ndarray) -> float:
        return float(np.abs(np.diff(img.astype(np.int16), axis=1)).mean())

    def next_action(self) -> np.ndarray:
        new_chunk = False
        low = self.args.exec_steps_low is not None and bool(self.d.qpos[2] < self.args.low_pelvis_z)
        if not self.queue or self.since_request >= (self.args.exec_steps_low if low else self.args.exec_steps):
            image, state = self.observe()
            if self._noise(image) > NOISE_LIMIT:
                raise RuntimeError(f"rendered head-camera image is noise (pixel-step {self._noise(image):.1f} > "
                                   f"{NOISE_LIMIT}): the offscreen OpenGL context is broken; run with MUJOCO_GL=egl "
                                   f"or without --viewer")
            t0 = time.time()
            chunk = self.policy.chunk(self.frame, image, state)
            self.latency.append(time.time() - t0)
            # 78 = neckless layout; the SONIC 1.1 post-train layout pads it to 80 (2 neck dims, unused here)
            if chunk.shape[1] < TOKEN + HANDS:
                raise RuntimeError(f"expected >= {TOKEN + HANDS}-D actions, got {chunk.shape}")
            chunk = chunk[:, :TOKEN + HANDS]
            self.queue, self.since_request, new_chunk = list(chunk), 0, True
            self.low_requests += low
        thumb = self.frame % (RECORD_FPS // 2) == 0
        if self.videos or thumb:
            ego, _ = self.observe()
            self.cam.update_scene(self.d, camera="debug_cam")
            third = self.cam.render()
            if self.videos:
                self.videos[0].append_data(ego)
                self.videos[1].append_data(third)
            if thumb:
                self.thumbs.append((self.frame / RECORD_FPS, np.asarray(Image.fromarray(third).resize((320, 240))), ego.copy()))
                self._live(third, ego)
        a = self.queue.pop(0)
        c = self.last_contacts
        self.trace["t"].append(self.frame / RECORD_FPS)
        self.trace["qpos"].append(self.d.qpos[:50].astype(np.float32))
        self.trace["box_pose"].append(np.r_[self.box_pose()].astype(np.float32))
        self.trace["action"].append(np.asarray(a, np.float32))
        self.trace["new_chunk"].append(new_chunk)
        self.trace["contacts"].append([bool(self.grasped), bool(c["hand_box"]), bool(c["box_tray"]), bool(c["box_floor"]),
                                       bool(c["robot_obstacle"]), bool(all(c["feet"]))])
        self.since_request += 1
        self.frame += 1
        return a

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
            tau = pd_torque(q_des, d.qpos[ix.body_qadr], d.qvel[ix.body_vadr], self.pol.kp, self.pol.kd)
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
        """episode.py's start: HOME pose at START with joint noise, 1.5 s hold; recording (= control) starts after."""
        m, d, ix, p = self.m, self.d, self.ix, self.p
        d.qpos[:2] = p.start_xy
        d.qpos[2] = 0.793
        d.qpos[3:7] = [np.cos(p.start_yaw / 2), 0, 0, np.sin(p.start_yaw / 2)]
        d.qpos[ix.body_qadr] = DEFAULT_ANGLES_MJ + self.rng.normal(0, p.init_joint_noise, 29)
        mujoco.mj_forward(m, d)
        q_home = d.qpos.copy(); q_home[ix.body_qadr] = DEFAULT_ANGLES_MJ; q_home[2] = 0.762
        self.ref.start(q_home)
        self.ref.add_hold(1.5)
        self.run_until_ref_end()
        if d.qpos[2] < 0.72:
            raise EpisodeFailed("home_settle_failed")

    def evaluate(self) -> dict:
        """Run the scene; ALWAYS writes its evidence folder (see the module docstring). status: success / fail /
        error (exception from the policy or the network: re-raised as PolicyError after saving) / stopped
        (KeyboardInterrupt: re-raised after saving)."""
        args, d = self.args, self.d
        t_wall = time.time()
        status, error, interrupt = None, None, None
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
                if self.last_contacts["robot_obstacle"]:
                    raise EpisodeFailed("hit_obstacle")
                # it fails once the can is knocked over (outside the tray: lying in the tray counts as placed, as in
                # the generated data), or on the floor / pedestal again after a lift
                if self.box_tilt() > 45.0 and not self.box_in_tray():
                    raise EpisodeFailed("can_tipped_over")
                if diag["box_max_lift_m"] > 0.05 and self.last_contacts["box_floor"]:
                    raise EpisodeFailed("can_dropped")
                resting = self.box_in_tray(resting=True) and bv < 0.03 and not self.grasped
                solved_since = (solved_since if solved_since is not None else self.t) if resting else None
                if solved_since is not None and self.t - solved_since >= 1.0:
                    ok, reason = True, None
                    break
                if self.t - t0 > args.max_seconds:
                    raise EpisodeFailed("timeout")
            status = "success"
        except EpisodeFailed as e:
            ok, reason, status = False, str(e), "fail"
        except KeyboardInterrupt as e:
            ok, reason, status, interrupt = False, f"stopped ({e or 'Ctrl+C'})", "stopped", e
        except Exception as e:   # the policy server / network (tunnel expired, timeout, server error)
            ok, reason, status = False, f"{type(e).__name__}: {e}", "error"
            error = traceback.format_exc()
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
        r = {"seed": self.p.seed, "status": status, "success": ok, "fail_reason": reason,
             "policy_seconds": round(self.frame / RECORD_FPS, 2),
             "requests": len(self.latency), "low_requests": self.low_requests,
             "latency_mean_s": round(float(lat.mean()), 3),
             "latency_max_s": round(float(lat.max()), 3), "wall_seconds": round(time.time() - t_wall, 1), **diag,
             "nav_side": self.p.nav_side, "obstacle_xy": [round(float(v), 3) for v in self.p.obstacle_xy],
             "layout_xy": [round(float(v), 3) for v in self.p.layout_xy]}
        r["evidence"] = self.save_evidence(r, error)
        if interrupt is not None:
            raise EvalStopped(r)
        if status == "error":
            raise PolicyError(r)
        return r

    def _live(self, third: np.ndarray, ego: np.ndarray):
        """<out>/live.jpg (third-person | head camera, every 0.5 s of simulation) + live.json, overwritten in place, so
        a notebook can show the running scene (colab_*_server.ipynb cell 6c / Bước 5c)."""
        out = Path(self.args.out)
        im = Image.new("RGB", (640 + 320, 480), (20, 20, 20))
        im.paste(Image.fromarray(third), (0, 0))
        im.paste(Image.fromarray(ego), (640, 0))
        t = self.frame / RECORD_FPS
        ImageDraw.Draw(im).text((8, 8), f"seed {self.p.seed}  t = {t:.1f} s  grasped {self.grasped}", fill=(255, 255, 0))
        tmp = out / "live.tmp.jpg"
        im.save(tmp, quality=80)
        tmp.replace(out / "live.jpg")
        (out / "live.json").write_text(json.dumps({"seed": self.p.seed, "t": round(t, 1), "grasped": bool(self.grasped),
                                                   "requests": len(self.latency),
                                                   "latency_mean_s": round(float(np.mean(self.latency)), 3) if self.latency else None,
                                                   "updated": time.time()}))

    def save_evidence(self, r: dict, error: str | None) -> str:
        """result.json, trace.npz and overview.png into the scene's folder, then name the folder after the result."""
        d = self.ev_dir
        tag = {"success": "SUCCESS", "fail": f"FAIL_{(r['fail_reason'] or '').split('(')[0]}", "error": "ERROR",
               "stopped": "STOPPED"}[r["status"]]
        final = d.parent / f"seed_{self.p.seed:06d}_{tag}"
        k = 2
        while final.exists():
            final = d.parent / f"seed_{self.p.seed:06d}_{tag}_{k}"; k += 1
        r["evidence"] = final.name
        (d / "result.json").write_text(json.dumps(r, indent=2))
        if error:
            (d / "error.txt").write_text(error)
        if self.trace["t"]:
            np.savez_compressed(d / "trace.npz", **{k: np.asarray(v) for k, v in self.trace.items()},
                                contact_names=np.array(["grasped", "hand_can", "can_tray", "can_floor_or_pedestal",
                                                        "robot_obstacle", "both_feet_down"]),
                                action_layout=np.array("SONIC token 64 | left hand 7 | right hand 7"))
        # the last moment is always on the sheet, also when the scene ended between two thumbnails
        if self.frame:
            ego, _ = self.observe()
            self.cam.update_scene(self.d, camera="debug_cam")
            self.thumbs.append((self.frame / RECORD_FPS, np.asarray(Image.fromarray(self.cam.render()).resize((320, 240))), ego))
        self._overview(r, d / "overview.png")
        self.cam.close()
        d.rename(final)
        return final.name

    def _overview(self, r: dict, path: Path):
        """12 moments of the scene (third-person over head camera) under a banner with the result."""
        W, H, n = 320, 240, 12
        th = self.thumbs
        pick = [th[i] for i in sorted({round(j * (len(th) - 1) / (n - 1)) for j in range(n)})] if th else []
        cols = 6
        rows = max(1, -(-len(pick) // cols))
        im = Image.new("RGB", (W * cols, 40 + rows * 2 * H), (20, 20, 20))
        dr = ImageDraw.Draw(im)
        color = {"success": (60, 200, 90), "fail": (230, 70, 60), "error": (240, 160, 40), "stopped": (160, 160, 160)}
        dr.rectangle([0, 0, W * cols, 39], fill=color.get(r["status"], (120, 120, 120)))
        dr.text((10, 12), f"seed {r['seed']}  |  {(r['status'] or '').upper()}"
                + (f": {r['fail_reason']}" if r["fail_reason"] else "")
                + f"  |  policy {r['policy_seconds']} s  |  grasp {r.get('grasp_at_s')} s, release {r.get('release_at_s')} s,"
                  f" can lift {r.get('box_max_lift_m')} m, in tray {r.get('end_box_in_tray')}", fill=(0, 0, 0))
        for k, (t, third, ego) in enumerate(pick):
            x, y = (k % cols) * W, 40 + (k // cols) * 2 * H
            im.paste(Image.fromarray(third), (x, y))
            im.paste(Image.fromarray(ego), (x, y + H))
            ImageDraw.Draw(im).text((x + 6, y + 6), f"t = {t:.1f} s", fill=(255, 255, 0))
        im.save(path)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--server", help="policy server URL, e.g. http://r1.a.pinggy.link:41234")
    src.add_argument("--gr00t", metavar="HOST:PORT", help="GR00T N1.7 policy server (ZMQ, unitree_g1_sonic)")
    src.add_argument("--replay", type=Path, help="generated dataset whose recorded actions are replayed")
    ap.add_argument("--replay-episode", default="0", help="--replay: episode index, or a comma-separated list")
    ap.add_argument("--episodes", type=int, default=10, help="--server: number of scenes")
    ap.add_argument("--seed-start", type=int, default=FIRST_UNSEEN_SEED, help="--server: first scene seed")
    ap.add_argument("--level", type=int, default=0, help="randomization level of sample_params")
    ap.add_argument("--exec-steps", type=int, default=15,
                    help="frames (30 Hz) executed from each chunk before the next request")
    ap.add_argument("--exec-steps-low", type=int, default=None,
                    help="--exec-steps while the pelvis is below --low-pelvis-z (squatting to pick the can); "
                         "default: --exec-steps throughout")
    ap.add_argument("--low-pelvis-z", type=float, default=0.6,
                    help="pelvis height (m) under which --exec-steps-low applies (standing ~0.77, squatting ~0.5)")
    ap.add_argument("--layout-from", type=Path, default=None,
                    help="draw each scene's pick/place layout from this generated dataset's episodes "
                         "(default: sample_eval_layout)")
    ap.add_argument("--obstacle", choices=("random", "center"), default="random",
                    help="random: as sample_params (shifted sideways, passed on its wider side); center: the layout of "
                         "the first 160 episodes (can_nav_160: centred +-3 cm, 40 x 40 x 50 cm, yellow)")
    ap.add_argument("--max-seconds", type=float, default=100.0, help="policy time limit per scene")
    ap.add_argument("--timeout", type=float, default=120.0, help="timeout per request (first one loads kernels)")
    ap.add_argument("--clip-token", action="store_true", help="clip tokens to +-0.625 like the robot client")
    ap.add_argument("--viewer", action="store_true", help="show the scene in the MuJoCo viewer")
    ap.add_argument("--realtime", action="store_true", help="--viewer: pace the simulation to wall-clock time")
    ap.add_argument("--out", type=Path, default=None,
                    help="evidence + results go here (default: .data/eval/<policy>_<date>_<time>)")
    ap.add_argument("--video", action="store_true", default=True, help="(default) ego + third-person mp4 per scene")
    ap.add_argument("--no-video", dest="video", action="store_false", help="no mp4s (overview.png etc. are still saved)")
    ap.add_argument("--threads", type=int, default=4, help="onnxruntime threads for SONIC")
    args = ap.parse_args()
    if args.out is None:
        kind = "replay" if args.replay else ("gr00t" if args.gr00t else "psi0")
        args.out = HERE.parents[2] / ".data" / "eval" / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"evidence -> {args.out}")

    sonic = SonicPolicy(SONIC_DIR, threads=args.threads)
    if args.replay:
        meta = {m["episode_index"]: m for m in (json.loads(l) for l in
                (args.replay / "meta/episode_meta.jsonl").read_text().splitlines() if l.strip())}
        import pandas as pd
        names = {f.name for f in dataclasses.fields(SceneParams)}
        replays = {}                      # seed -> (scene params, recorded actions)
        for e in (int(x) for x in str(args.replay_episode).split(",")):
            df = pd.read_parquet(args.replay / f"data/chunk-000/episode_{e:06d}.parquet", columns=["action"])
            replays[meta[e]["seed"]] = (SceneParams(**{k: tuple(v) if isinstance(v, list) else v
                                                       for k, v in meta[e]["scene"].items() if k in names}),
                                        np.stack(df["action"].values).astype(np.float32))
            print(f"replaying episode {e} (seed {meta[e]['seed']}, {len(df)} frames)")
        seeds = list(replays)
    else:
        if args.gr00t:
            policy = Gr00tPolicy(args.gr00t, INSTRUCTION, args.timeout)
        else:
            policy = HttpPolicy(args.server, INSTRUCTION, args.timeout)
        seeds = list(range(args.seed_start, args.seed_start + args.episodes))
        print(f"server {args.server or args.gr00t} is up; {len(seeds)} scenes from seed {seeds[0]}")
    layouts = None
    if args.layout_from:
        layouts = [(tuple(m["scene"]["layout_xy"]), m["scene"]["layout_yaw"]) for m in
                   (json.loads(l) for l in (args.layout_from / "meta/episode_meta.jsonl").read_text().splitlines() if l.strip())]

    try:
        commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or None
    except OSError:
        commit = None
    (args.out / "run_config.json").write_text(json.dumps(
        {"argv": sys.argv, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "seeds": seeds, "git_commit": commit, "sonic_dir": str(SONIC_DIR), "started": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=2))

    results = []
    for seed in seeds:
        sonic.reset()
        if args.replay:
            params, actions = replays[seed]
            policy = ReplayPolicy(actions)
        else:
            if layouts:
                xy, yaw = layouts[np.random.default_rng(seed).integers(len(layouts))]
            else:
                xy, yaw = sample_eval_layout(seed)
            params = dataclasses.replace(sample_params(seed, level=args.level), layout_xy=xy, layout_yaw=yaw)
            if args.obstacle == "center":
                r = np.random.default_rng(seed + 777)
                params = dataclasses.replace(params, obstacle_xy=(-0.80 + r.uniform(-0.03, 0.03), r.uniform(-0.03, 0.03)),
                                             obstacle_half=(0.20, 0.20, 0.25), obstacle_rgb=(0.85, 0.65, 0.15))
        ep = ClosedLoopEpisode(params, sonic, policy, args, args.viewer)
        stop = None
        try:
            r = ep.evaluate()
        except EvalStopped as e:
            r, stop = e.result, "stopped"
        except PolicyError as e:
            r, stop = e.result, "error"
        results.append(r)
        with open(args.out / "results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        write_summary(args.out)
        print(f"{r['status'].upper()}" + (f" ({r['fail_reason']})" if r["fail_reason"] else "")
              + f"  seed {seed}: {r['policy_seconds']} s, "
              f"{r['requests']} requests, latency {r['latency_mean_s']} s (max {r['latency_max_s']}) | "
              f"grasp {r['grasp_at_s']} s, release {r['release_at_s']} s, lift {r['box_max_lift_m']} m, "
              f"end: in tray {r['end_box_in_tray']}, held {r['end_grasped']}, box->target {r['end_box_to_target_m']} m"
              f" | evidence {args.out / r['evidence']}", flush=True)
        if stop:
            nxt = seed if stop == "error" else seed + 1
            print(f"\n{'server/network error' if stop == 'error' else 'stopped'}: resume with --seed-start {nxt} "
                  f"--episodes {seeds[-1] - nxt + 1} --out {args.out}")
            break
    if results:
        n_ok = sum(r["success"] for r in results)
        print(f"\n{n_ok}/{len(results)} solved ({100 * n_ok / len(results):.0f} %) -> {args.out / 'summary.md'}")


def write_summary(out: Path):
    """summary.json + summary.md over every scene in results.jsonl (also earlier, resumed runs into the same --out)."""
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l.strip()]
    last = {}
    for r in rows:                       # a resumed scene (same seed) keeps its latest result
        last[r["seed"]] = r
    rows = [last[k] for k in sorted(last)]
    done = [r for r in rows if r.get("status", "success" if r["success"] else "fail") in ("success", "fail")]
    n_ok = sum(r["success"] for r in done)
    reasons: dict[str, int] = {}
    for r in done:
        if not r["success"]:
            k = (r["fail_reason"] or "?").split("(")[0]
            reasons[k] = reasons.get(k, 0) + 1
    summary = {"scenes": len(done), "solved": n_ok, "success_rate": round(n_ok / len(done), 3) if done else None,
               "fail_reasons": reasons, "not_finished": [r["seed"] for r in rows if r not in done],
               "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [f"# Closed-loop eval: {out.name}", "",
             f"**{n_ok}/{len(done)} solved**" + (f" ({100 * n_ok / len(done):.0f} %)" if done else "")
             + f", updated {summary['updated']}", ""]
    if reasons:
        lines += ["Fail reasons: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])), ""]
    lines += ["| seed | result | reason | policy s | grasp s | release s | can lift m | in tray | evidence |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        ev = r.get("evidence", "")
        lines.append(f"| {r['seed']} | {(r.get('status') or '').upper()} | {r['fail_reason'] or ''} | {r['policy_seconds']} | "
                     f"{r.get('grasp_at_s')} | {r.get('release_at_s')} | {r.get('box_max_lift_m')} | "
                     f"{r.get('end_box_in_tray')} | [{ev}]({ev}/overview.png) |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
