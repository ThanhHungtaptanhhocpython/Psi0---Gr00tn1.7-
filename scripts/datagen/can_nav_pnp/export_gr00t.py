"""Export a generated can_nav_pnp dataset (writer.py / merge_shards.py layout) as train/val LeRobot v2.1 packs
for GR00T N1.7 finetuning with the pre-registered `unitree_g1_sonic` embodiment (Isaac-GR00T
gr00t/configs/data/embodiment_configs.py): the model outputs the SONIC token + Dex3 hand targets, run
through SONIC 1.1 exactly like the Psi0 policy.

  python scripts/datagen/can_nav_pnp/export_gr00t.py .data/can_nav_160 --out .data/gr00t --zip

writes .data/gr00t/can_nav_160_train, .data/gr00t/can_nav_160_val and (with --zip)
.data/gr00t/can_nav_160_gr00t.zip holding both.

Layout (meta/modality.json keys = the unitree_g1_sonic modality keys):
  observation.state (46)  0810 joint order + projected gravity
                          left_leg 0:6, right_leg 6:12, waist 12:15, left_arm 15:22, right_arm 22:29,
                          left_hand 29:36, right_hand 36:43, projected_gravity 43:46
  action (78)             motion_token 0:64, left_hand_joints 64:71, right_hand_joints 71:78 (absolute)
  video                   observation.images.ego_view (modality key ego_view), re-encoded like export_psi0
  language                annotation.human.task_description -> task_index
  meta/stats.json         per-column mean/std/min/max/q01/q99 over every exported frame (both splits), the
                          format gr00t/data/stats.py writes

The split (same --seed / --val-frac / --val-episodes as export_psi0.py) picks the same val episodes as the
Psi0 export, so both models are compared on the same held-out episodes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from export_psi0 import (HAND, STATE_0810, _vectors, count_frames, encode_video, feature,  # noqa: E402
                         parse_size, read_jsonl, state_permutation, write_jsonl)
from writer import CHUNK, EGO_KEY  # noqa: E402

DST_VIDEO_KEY = "observation.images.ego_view"
TOKEN_DIM, HAND_DIM = 64, 7
GRAVITY = ["projected_gravity_x", "projected_gravity_y", "projected_gravity_z"]
STATE_NAMES = STATE_0810 + GRAVITY
ACTION_NAMES = [f"motion_token_{i}" for i in range(TOKEN_DIM)] + HAND
STATE_GROUPS = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22),
                "right_arm": (22, 29), "left_hand": (29, 36), "right_hand": (36, 43), "projected_gravity": (43, 46)}
ACTION_GROUPS = {"motion_token": (0, 64), "left_hand_joints": (64, 71), "right_hand_joints": (71, 78)}
LANG_KEY = "human.task_description"


def gr00t_stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float32)
    return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "min": a.min(0).tolist(),
            "max": a.max(0).tolist(), "q01": np.quantile(a, 0.01, axis=0).tolist(),
            "q99": np.quantile(a, 0.99, axis=0).tolist()}


def convert_episode(df: pd.DataFrame, perm: list[int], new_ep: int, start_index: int, fps: int):
    act = np.stack(df["action"].values).astype(np.float32)                         # (T, 78) token | lhand | rhand
    joints = np.stack(df["observation.state"].values).astype(np.float32)           # (T, 43) MJCF order
    grav = np.stack(df["observation.projected_gravity"].values).astype(np.float32)  # (T, 3)
    assert act.shape[1] == TOKEN_DIM + 2 * HAND_DIM and joints.shape[1] == len(STATE_0810)
    state = np.ascontiguousarray(np.c_[joints[:, perm], grav])
    assert np.isfinite(act).all() and np.isfinite(state).all(), "NaN/Inf in action or state"
    n = len(df)
    frame_index = df["frame_index"].to_numpy(dtype=np.int64)
    assert (frame_index == np.arange(n)).all(), "frame_index is not 0..n-1"
    arrays = {"observation.state": state, "action": np.ascontiguousarray(act)}
    table = pa.table({
        **{k: _vectors(v) for k, v in arrays.items()},
        "timestamp": pa.array((frame_index / fps).astype(np.float32), pa.float32()),
        "frame_index": pa.array(frame_index, pa.int64()),
        "episode_index": pa.array(np.full(n, new_ep), pa.int64()),
        "index": pa.array(np.arange(start_index, start_index + n), pa.int64()),
        "task_index": pa.array(np.zeros(n), pa.int64()),
        "next.done": pa.array(np.arange(n) == n - 1, pa.bool_()),
    })
    return table, arrays


def write_split(src: Path, dst: Path, src_eps: list[int], perm: list[int], info_src: dict, instruction: str,
                video_size, gop: int, jobs: int, provenance: dict) -> dict[str, np.ndarray]:
    fps = info_src["fps"]
    (dst / "data" / CHUNK).mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    episodes, ep_stats, cols, total, video_jobs = [], [], {"action": [], "observation.state": []}, 0, []
    for new_ep, old_ep in enumerate(src_eps):
        df = pd.read_parquet(src / "data" / CHUNK / f"episode_{old_ep:06d}.parquet",
                             columns=["action", "observation.state", "observation.projected_gravity", "frame_index"])
        table, arrays = convert_episode(df, perm, new_ep, total, fps)
        pq.write_table(table, dst / "data" / CHUNK / f"episode_{new_ep:06d}.parquet")
        n = table.num_rows
        video_jobs.append((src / "videos" / CHUNK / EGO_KEY / f"episode_{old_ep:06d}.mp4",
                           dst / "videos" / CHUNK / DST_VIDEO_KEY / f"episode_{new_ep:06d}.mp4", n))
        episodes.append({"episode_index": new_ep, "tasks": [instruction], "length": n, "source_episode_index": old_ep})
        ep_stats.append({"episode_index": new_ep, "stats": {k: gr00t_stats(v) for k, v in arrays.items()}})
        for k, v in arrays.items():
            cols[k].append(v)
        total += n

    def do_video(job):
        s, d, n = job
        if count_frames(s) != n:
            raise SystemExit(f"{s}: {count_frames(s)} video frames vs {n} parquet rows")
        encode_video(s, d, video_size, gop)
        if count_frames(d) != n:
            raise SystemExit(f"{d}: re-encoded video has {count_frames(d)} frames, expected {n}")

    with ThreadPoolExecutor(jobs) as ex:
        for i, _ in enumerate(ex.map(do_video, video_jobs), 1):
            print(f"\r  {dst.name}: video {i}/{len(video_jobs)}", end="", flush=True)
    print()

    h, w = (video_size[1], video_size[0]) if video_size else info_src["features"][EGO_KEY]["shape"][:2]
    n_eps = len(episodes)
    info = {
        "codebase_version": "v2.1", "robot_type": info_src.get("robot_type", "g1_29dof_dex3_sonic"),
        "total_episodes": n_eps, "total_frames": total, "total_tasks": 1, "total_videos": n_eps,
        "total_chunks": 1, "chunks_size": 1000, "fps": fps, "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            DST_VIDEO_KEY: {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"],
                            "info": {"video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                     "video.is_depth_map": False, "has_audio": False}},
            "observation.state": feature("float32", [len(STATE_NAMES)], STATE_NAMES),
            "action": feature("float32", [len(ACTION_NAMES)], ACTION_NAMES),
            "timestamp": feature("float32", [1]),
            "frame_index": feature("int64", [1]),
            "episode_index": feature("int64", [1]),
            "index": feature("int64", [1]),
            "task_index": feature("int64", [1]),
            "next.done": feature("bool", [1]),
        },
        "embodiment_tag": "unitree_g1_sonic",
        "sonic_version": info_src.get("sonic_version"),
        "export": provenance,
    }
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (dst / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": instruction}) + "\n")
    write_jsonl(dst / "meta" / "episodes.jsonl", episodes)
    write_jsonl(dst / "meta" / "episodes_stats.jsonl", ep_stats)
    (dst / "meta" / "modality.json").write_text(json.dumps({
        "state": {k: {"start": a, "end": b} for k, (a, b) in STATE_GROUPS.items()},
        "action": {k: {"start": a, "end": b} for k, (a, b) in ACTION_GROUPS.items()},
        "video": {"ego_view": {"original_key": DST_VIDEO_KEY}},
        "annotation": {LANG_KEY: {"original_key": "task_index"}},
    }, indent=2))
    return {k: np.concatenate(v) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="raw dataset from generate.py / merge_shards.py")
    ap.add_argument("--out", type=Path, required=True, help="parent dir of <name>_train / <name>_val")
    ap.add_argument("--name", default=None, help="pack name (default: the source dir name)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of episodes held out for val")
    ap.add_argument("--val-episodes", default=None, help="comma-separated source episode indices (overrides --val-frac)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random val episode pick (as export_psi0.py)")
    ap.add_argument("--video-size", default="320x240", help="WxH of the exported video, or 'source'")
    ap.add_argument("--gop", type=int, default=10, help="keyframe interval of the exported video")
    ap.add_argument("--jobs", type=int, default=4, help="parallel ffmpeg processes")
    ap.add_argument("--zip", action="store_true", help="also write <out>/<name>_gr00t.zip with both splits")
    ap.add_argument("--force", action="store_true", help="overwrite existing output packs")
    a = ap.parse_args()

    src = a.src
    name = a.name or src.resolve().name
    info_src = json.loads((src / "meta" / "info.json").read_text())
    tasks = read_jsonl(src / "meta" / "tasks.jsonl")
    assert len(tasks) == 1, f"expected one task in {src}/meta/tasks.jsonl, got {len(tasks)}"
    instruction = tasks[0]["task"]
    all_eps = [e["episode_index"] for e in read_jsonl(src / "meta" / "episodes.jsonl")]

    # same val pick as export_psi0.py
    if a.val_episodes:
        val = sorted(int(x) for x in a.val_episodes.split(","))
        assert set(val) <= set(all_eps), f"unknown val episodes {sorted(set(val) - set(all_eps))}"
    else:
        n_val = max(1, round(a.val_frac * len(all_eps)))
        val = sorted(np.random.default_rng(a.seed).choice(all_eps, n_val, replace=False).tolist())
    train = [e for e in all_eps if e not in set(val)]
    if not train or not val:
        raise SystemExit(f"need >= 1 episode in each split, got train {len(train)} / val {len(val)}")

    perm = state_permutation()
    video_size = parse_size(a.video_size)
    dst = {s: a.out / f"{name}_{s}" for s in ("train", "val")}
    for d in dst.values():
        if d.exists():
            if not a.force:
                raise SystemExit(f"{d} exists (pass --force to overwrite)")
            shutil.rmtree(d)
    print(f"{src}: {len(all_eps)} episodes -> train {len(train)} / val {len(val)} (val = {val})")

    provenance = {"source": str(src), "val_source_episode_indices": val, "seed": a.seed,
                  "video_size": a.video_size, "gop": a.gop}
    arrays = {s: write_split(src, dst[s], eps, perm, info_src, instruction, video_size, a.gop, a.jobs,
                             {**provenance, "split": s})
              for s, eps in (("train", train), ("val", val))}

    stats = {k: gr00t_stats(np.concatenate([arrays["train"][k], arrays["val"][k]]))
             for k in ("observation.state", "action")}
    for d in dst.values():
        (d / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    const = [STATE_NAMES[i] for i, (lo, hi) in enumerate(zip(stats["observation.state"]["min"],
                                                             stats["observation.state"]["max"])) if hi - lo < 1e-4]
    if const:
        print(f"note: constant state dims: {const}")
    if a.zip:
        zpath = a.out / f"{name}_gr00t.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for d in dst.values():
                for f in sorted(p for p in d.rglob("*") if p.is_file()):
                    z.write(f, f.relative_to(a.out).as_posix())
        print(f"-> {zpath} ({zpath.stat().st_size / 2**20:.0f} MiB)")
    for s, d in dst.items():
        print(f"-> {d}: {len(train if s == 'train' else val)} episodes, {len(arrays[s]['action'])} frames")


if __name__ == "__main__":
    main()
