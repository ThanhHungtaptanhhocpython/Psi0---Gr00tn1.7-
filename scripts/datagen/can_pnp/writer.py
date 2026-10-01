"""Raw dataset writer: LeRobot v2.1 layout, SIMPLE-source compatible.

The layout matches what scripts/data/simple_to_sonic_lerobot.py consumes
(`action` 78 = [token64 | lhand7 | rhand7], `observation.state` 43, video key
`observation.images.ego_view`, `task_index` + meta/tasks.jsonl), plus every raw signal
needed to debug, regenerate SONIC latents or finetune SONIC later.

  <root>/
    data/chunk-000/episode_000000.parquet          one row per 30 Hz frame
    videos/chunk-000/observation.images.ego_view/episode_000000.mp4
    debug/third_person/episode_000000.mp4           review only, NOT a policy input
    meta/info.json  tasks.jsonl  episodes.jsonl  episodes_stats.jsonl  modality.json
    meta/episode_meta.jsonl                          scene params, events, checks, SONIC hashes
    meta/generation_log.jsonl                        every attempt, incl. rejected ones + reason
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from episode import EVENTS, INSTRUCTION, PHASES
from render import open_video, render_states, render_to_videos
from scene import IMG_H, IMG_W, RECORD_FPS

CHUNK = "chunk-000"
EGO_KEY = "observation.images.ego_view"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
            "std": a.std(0).tolist(), "count": [int(a.shape[0])]}


def write_video(path: Path, frames: list[np.ndarray], fps: int = RECORD_FPS):
    w = open_video(path, fps)
    for f in frames:
        w.append_data(f)
    w.close()


class DatasetWriter:
    def __init__(self, root: str | Path, sonic_dir: Path, planner_path: Path):
        self.root = Path(root)
        from sonic_policy import sonic_version
        self.sonic_variant, self.sonic_version = Path(sonic_dir).name, sonic_version(sonic_dir)
        (self.root / "data" / CHUNK).mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        self.sonic_hashes = {
            "model_encoder.onnx": sha256(sonic_dir / "model_encoder.onnx"),
            "model_decoder.onnx": sha256(sonic_dir / "model_decoder.onnx"),
            "planner_sonic.onnx": sha256(planner_path),
        }
        self.episodes = self._read_jsonl("episodes.jsonl")
        self.total_frames = sum(e["length"] for e in self.episodes)

    def _read_jsonl(self, name):
        p = self.root / "meta" / name
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    def _append_jsonl(self, name, obj):
        with open(self.root / "meta" / name, "a") as f:
            f.write(json.dumps(obj) + "\n")

    @property
    def next_index(self):
        return len(self.episodes)

    def log_attempt(self, seed, result, params):
        self._append_jsonl("generation_log.jsonl", {
            "seed": seed, "success": result.success, "fail_reason": result.fail_reason,
            "scene": {k: params.to_dict()[k] for k in ("box_xy", "box_yaw", "table_edge_x", "table_height", "place_xy", "tray_xy", "tray_yaw")},
            "events": result.events, "info": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                              for k, v in result.info.items()},
            "episode_index": self.next_index if result.success else None,
        })

    def add_episode(self, result, params, cfg) -> int:
        ep = self.next_index
        cols = result.frames
        n = len(cols["frame_index"])
        df = pd.DataFrame({k: (list(np.stack(v)) if np.ndim(v[0]) > 0 else v) for k, v in cols.items()})
        df["episode_index"] = ep
        df["index"] = np.arange(self.total_frames, self.total_frames + n)
        df["task_index"] = 0
        df["next.done"] = [False] * (n - 1) + [True]
        df.to_parquet(self.root / "data" / CHUNK / f"episode_{ep:06d}.parquet", index=False)

        ego_p = self.root / "videos" / CHUNK / EGO_KEY / f"episode_{ep:06d}.mp4"
        dbg_p = self.root / "debug" / "third_person" / f"episode_{ep:06d}.mp4"
        if result.ego_frames:       # rendered during the simulation
            write_video(ego_p, result.ego_frames)
            if result.debug_frames:
                write_video(dbg_p, result.debug_frames)
        else:                       # headless simulation: render from the recorded states
            render_to_videos(params, np.stack(cols["observation.qpos"]), np.stack(cols["object.box_pose"]),
                             ego_p, dbg_p if cfg.debug_render else None)

        self.episodes.append({"episode_index": ep, "tasks": [INSTRUCTION], "length": n})
        self._append_jsonl("episodes.jsonl", self.episodes[-1])
        numeric = {k: np.stack(v) for k, v in cols.items()
                   if np.asarray(v[0]).dtype.kind in "fiub" and k not in ("frame_index",)}
        self._append_jsonl("episodes_stats.jsonl", {"episode_index": ep,
                                                    "stats": {k: _stats(v) for k, v in numeric.items()}})
        self._append_jsonl("episode_meta.jsonl", {
            "episode_index": ep, "seed": params.seed, "success": result.success,
            "fail_reason": result.fail_reason, "scene": params.to_dict(),
            "events": result.events, "info": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                              for k, v in result.info.items()},
            "phase_names": PHASES, "sonic_onnx_sha256": self.sonic_hashes,
            "token_rate_hz": cfg.token_rate_hz, "grasp_assist": "soft weld, engaged on thumb+finger contact",
            "sonic_variant": self.sonic_variant, "sonic_version": self.sonic_version,
        })
        self.total_frames += n
        self._write_info(cols)
        return ep

    def save_failed(self, result, seed: int, params) -> Path:
        """Review material for a rejected attempt (never part of the dataset): ego + third-person
        videos and a per-phase contact sheet under <root>/failed/."""
        from review_episode import contact_sheet
        out = self.root / "failed"
        stem = f"seed_{seed:06d}"
        cols = result.frames
        ego, dbg = result.ego_frames, result.debug_frames
        if not ego:                 # headless simulation: render from the recorded states
            ego, dbg = render_states(params, np.stack(cols["observation.qpos"]), np.stack(cols["object.box_pose"]))
        write_video(out / f"{stem}_ego.mp4", ego)
        if dbg:
            write_video(out / f"{stem}_third.mp4", dbg)
        contact_sheet(ego, dbg or None, np.asarray(cols["phase"]),
                      np.asarray(cols["timestamp"]), PHASES, out / f"{stem}_phases.png",
                      f"seed {seed}: FAILED in {result.info.get('fail_phase')}: {result.fail_reason}")
        return out

    def _write_info(self, cols):
        def feat(k, v):
            a = np.asarray(v[0])
            dt = {"f": "float32", "i": "int64", "u": "int64", "b": "bool"}[a.dtype.kind]
            return {"dtype": dt, "shape": list(a.shape) or [1], "names": None}
        feats = {k: feat(k, v) for k, v in cols.items()}
        feats.update({
            EGO_KEY: {"dtype": "video", "shape": [IMG_H, IMG_W, 3], "names": ["height", "width", "channel"],
                      "info": {"video.fps": RECORD_FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                               "video.is_depth_map": False, "has_audio": False}},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
        })
        info = {
            "codebase_version": "v2.1", "robot_type": "g1_29dof_dex3_sonic", "sonic_version": self.sonic_version,
            "total_episodes": len(self.episodes), "total_frames": self.total_frames, "total_tasks": 1,
            "total_videos": len(self.episodes), "total_chunks": 1, "chunks_size": 1000, "fps": RECORD_FPS,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feats,
        }
        (self.root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        (self.root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": INSTRUCTION}) + "\n")
        (self.root / "meta" / "modality.json").write_text(json.dumps({
            "state": {"left_leg": {"start": 0, "end": 6}, "right_leg": {"start": 6, "end": 12},
                      "waist": {"start": 12, "end": 15}, "left_arm": {"start": 15, "end": 22},
                      "left_hand": {"start": 22, "end": 29}, "right_arm": {"start": 29, "end": 36},
                      "right_hand": {"start": 36, "end": 43}},
            "action": {"sonic_token": {"start": 0, "end": 64}, "left_hand": {"start": 64, "end": 71},
                       "right_hand": {"start": 71, "end": 78}},
            "video": {"ego_view": {"original_key": EGO_KEY}},
            "annotation": {"task": {"original_key": "task_index"}},
            "debug": {"phase": {"names": PHASES}, "events": EVENTS},
        }, indent=2))
