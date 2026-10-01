"""Export a generated box_pnp dataset (writer.py / merge_shards.py layout) as the train/val
LeRobot packs that finetune_sonic_psi0_config loads: the neckless 78/43 layout of the
sonic-wbcbox checkpoint (action = SONIC token 64 + hands 14, state 43).

  python scripts/datagen/box_pnp/export_psi0.py .data/box_pnp_tray_140 --out .data/psi0 --zip

writes .data/psi0/box_pnp_tray_140_train, .data/psi0/box_pnp_tray_140_val and (with --zip)
.data/psi0/box_pnp_tray_140_psi0.zip holding both, ready to upload for the Colab notebook
(colab_finetune_psi0.ipynb).

What changes relative to the raw dataset:
  observation.state (43)  MJCF qpos order  [lleg rleg waist larm lhand rarm rhand]  ->
                          0810 order       [lleg rleg waist larm rarm lhand rhand].
                          sonic-wbcbox was trained in 0810 order (the per-joint ranges in its
                          dataset_statistics.json put the right arm at 22:29 and the left hand
                          at 29:36), as are the sonic1.x UniFoLM post-train packs. The
                          permutation is derived by joint name from the generator's MJCF.
  action.body_token (64)  = action[:, 0:64], the SONIC token
  action (36)             = [lhand7 rhand7 | zeros22]; the loader reads action[:14]
  task_description        per-frame instruction string (--data.transform.repack.instruction-key)
  video                   observation.images.ego_view -> observation.images.egocentric,
                          re-encoded with a keyframe every --gop frames (default 10) at
                          --video-size (default 320x240, the training resolution). The loader
                          seeks to the previous keyframe and decodes forward, so the
                          generator's 250-frame GOP costs ~125 decoded frames per sample.
  timestamp               rewritten as frame_index / fps (LeRobot checks the 1/fps spacing)
  everything else         dropped (qpos/qvel/contacts/debug columns, third-person video)

Episodes are split by episode (never by frame) and renumbered 0..N-1 per split with a
contiguous global `index`. Both splits carry the SAME meta/stats_psi0.json (== stats.json),
computed exactly over every exported frame, so train and val normalize identically.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scene import G1_SCENE  # noqa: E402
from writer import CHUNK, EGO_KEY  # noqa: E402

DST_VIDEO_KEY = "observation.images.egocentric"
TOKEN_DIM, HAND_DIM, ACTION_DIM = 64, 14, 36

_SIDES = ("left", "right")
LEG = [f"{s}_{j}_joint" for s in _SIDES
       for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM = [f"{s}_{j}_joint" for s in _SIDES
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
# Dex3 order used by both the recorded hand actions (scene.py) and the 0810 packs
HAND = [f"{s}_hand_{j}_joint" for s in _SIDES
        for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
STATE_0810 = LEG + WAIST + ARM + HAND
ACTION_NAMES = HAND + [f"unused_{i}" for i in range(ACTION_DIM - HAND_DIM)]


def mjcf_state_names() -> list[str]:
    """Joint names of observation.state (= qpos[7:50]) in the generator's MJCF order."""
    m = mujoco.MjModel.from_xml_path(str(G1_SCENE))
    names = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    assert [m.jnt_qposadr[m.joint(n).id] for n in names] == list(range(7, 7 + len(names))), \
        "MJCF hinge joints are not laid out contiguously after the free joint"
    return names


def state_permutation() -> list[int]:
    src = mjcf_state_names()
    if sorted(src) != sorted(STATE_0810):
        raise SystemExit(f"MJCF joints differ from the 0810 layout: "
                         f"{sorted(set(src) ^ set(STATE_0810))}")
    return [src.index(n) for n in STATE_0810]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, rows: list[dict]):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def stats_of(a: np.ndarray, quantiles: bool = False) -> dict:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    s = {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
         "std": a.std(0).tolist(), "count": [int(a.shape[0])]}
    if quantiles:
        s["q01"] = np.quantile(a, 0.01, axis=0).tolist()
        s["q99"] = np.quantile(a, 0.99, axis=0).tolist()
    return s


def count_frames(path: Path) -> int:
    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


def encode_video(src: Path, dst: Path, size: tuple[int, int] | None, gop: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = ["-vf", f"scale={size[0]}:{size[1]}:flags=area"] if size else []
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), *vf,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", str(gop),
                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)


def _vectors(a: np.ndarray) -> pa.ListArray:
    """(T, D) float32 -> list<float32> column, the layout LeRobot's parquet loader expects."""
    return pa.ListArray.from_arrays(np.arange(0, a.size + 1, a.shape[1], dtype=np.int32),
                                    pa.array(a.reshape(-1), type=pa.float32()))


def convert_episode(df: pd.DataFrame, perm: list[int], new_ep: int, start_index: int, fps: int,
                    instruction: str) -> tuple[pa.Table, dict[str, np.ndarray]]:
    """One episode in the loader schema, plus its action/token/state arrays for the stats.

    Built as an explicit arrow table rather than through pandas: the datagen venv runs
    pandas 3, whose schema metadata and large_string text columns the training venv's
    datasets 3.6 / pandas 2 should never have to interpret."""
    act = np.stack(df["action"].values).astype(np.float32)             # (T, 78)
    state = np.stack(df["observation.state"].values).astype(np.float32)  # (T, 43)
    assert act.shape[1] == TOKEN_DIM + HAND_DIM and state.shape[1] == len(STATE_0810)
    assert np.isfinite(act).all() and np.isfinite(state).all(), "NaN/Inf in action or state"
    n = len(df)
    frame_index = df["frame_index"].to_numpy(dtype=np.int64)
    assert (frame_index == np.arange(n)).all(), "frame_index is not 0..n-1"
    action = np.zeros((n, ACTION_DIM), dtype=np.float32)
    action[:, :HAND_DIM] = act[:, TOKEN_DIM:]
    arrays = {"observation.state": np.ascontiguousarray(state[:, perm]), "action": action,
              "action.body_token": np.ascontiguousarray(act[:, :TOKEN_DIM])}
    table = pa.table({
        **{k: _vectors(v) for k, v in arrays.items()},
        "timestamp": pa.array((frame_index / fps).astype(np.float32), pa.float32()),
        "frame_index": pa.array(frame_index, pa.int64()),
        "episode_index": pa.array(np.full(n, new_ep), pa.int64()),
        "index": pa.array(np.arange(start_index, start_index + n), pa.int64()),
        "task_index": pa.array(np.zeros(n), pa.int64()),
        "task_description": pa.array([instruction] * n, pa.string()),
        "next.done": pa.array(np.arange(n) == n - 1, pa.bool_()),
    })
    return table, arrays


def feature(dtype: str, shape: list[int], names=None) -> dict:
    return {"dtype": dtype, "shape": shape, "names": names}


def write_split(src: Path, dst: Path, src_eps: list[int], perm: list[int], info_src: dict,
                instruction: str, video_size: tuple[int, int] | None, gop: int, jobs: int,
                provenance: dict) -> dict[str, np.ndarray]:
    """Write one split; return its concatenated action/token/state arrays for the global stats."""
    fps = info_src["fps"]
    (dst / "data" / CHUNK).mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    episodes, ep_stats, cols, total = [], [], {"action": [], "action.body_token": [], "observation.state": []}, 0
    video_jobs = []
    for new_ep, old_ep in enumerate(src_eps):
        df = pd.read_parquet(src / "data" / CHUNK / f"episode_{old_ep:06d}.parquet",
                             columns=["action", "observation.state", "frame_index"])
        table, arrays = convert_episode(df, perm, new_ep, total, fps, instruction)
        pq.write_table(table, dst / "data" / CHUNK / f"episode_{new_ep:06d}.parquet")
        n = table.num_rows
        video_jobs.append((src / "videos" / CHUNK / EGO_KEY / f"episode_{old_ep:06d}.mp4",
                           dst / "videos" / CHUNK / DST_VIDEO_KEY / f"episode_{new_ep:06d}.mp4", n))
        episodes.append({"episode_index": new_ep, "tasks": [instruction], "length": n,
                         "source_episode_index": old_ep})
        ep_stats.append({"episode_index": new_ep, "stats": {k: stats_of(v) for k, v in arrays.items()}})
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
            "observation.state": feature("float32", [len(STATE_0810)], STATE_0810),
            "action": feature("float32", [ACTION_DIM], ACTION_NAMES),
            "action.body_token": feature("float32", [TOKEN_DIM], [f"t{i}" for i in range(TOKEN_DIM)]),
            "timestamp": feature("float32", [1]),
            "frame_index": feature("int64", [1]),
            "episode_index": feature("int64", [1]),
            "index": feature("int64", [1]),
            "task_index": feature("int64", [1]),
            "task_description": feature("string", [1]),
            "next.done": feature("bool", [1]),
        },
        "joint_order": "observation.state in g1_sonic_lerobot_0810 order (leg waist arm hand), "
                       "action[0:14] = Dex3 hands (left 7, right 7: thumb 0-2, middle 0-1, index 0-1)",
        "export": provenance,
    }
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (dst / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": instruction}) + "\n")
    write_jsonl(dst / "meta" / "episodes.jsonl", episodes)
    write_jsonl(dst / "meta" / "episodes_stats.jsonl", ep_stats)
    (dst / "meta" / "modality.json").write_text(json.dumps({
        "state": {"left_leg": {"start": 0, "end": 6}, "right_leg": {"start": 6, "end": 12},
                  "waist": {"start": 12, "end": 15}, "left_arm": {"start": 15, "end": 22},
                  "right_arm": {"start": 22, "end": 29}, "left_hand": {"start": 29, "end": 36},
                  "right_hand": {"start": 36, "end": 43}},
        "action": {"hand_joints": {"start": 0, "end": HAND_DIM},
                   "body_token": {"start": 0, "end": TOKEN_DIM, "original_key": "action.body_token"}},
        "video": {"egocentric": {"original_key": DST_VIDEO_KEY}},
        "annotation": {"task_description": {}},
    }, indent=2))
    return {k: np.concatenate(v) for k, v in cols.items()}


def parse_size(s: str) -> tuple[int, int] | None:
    if s == "source":
        return None
    w, h = (int(x) for x in s.lower().split("x"))
    return w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="raw dataset from generate.py / merge_shards.py")
    ap.add_argument("--out", type=Path, required=True, help="parent dir of <name>_train / <name>_val")
    ap.add_argument("--name", default=None, help="pack name (default: the source dir name)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of episodes held out for val")
    ap.add_argument("--val-episodes", default=None, help="comma-separated source episode indices (overrides --val-frac)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random val episode pick")
    ap.add_argument("--video-size", default="320x240", help="WxH of the exported video, or 'source'")
    ap.add_argument("--gop", type=int, default=10, help="keyframe interval of the exported video")
    ap.add_argument("--jobs", type=int, default=4, help="parallel ffmpeg processes")
    ap.add_argument("--zip", action="store_true", help="also write <out>/<name>_psi0.zip with both splits")
    ap.add_argument("--force", action="store_true", help="overwrite existing output packs")
    a = ap.parse_args()

    src = a.src
    name = a.name or src.resolve().name
    info_src = json.loads((src / "meta" / "info.json").read_text())
    tasks = read_jsonl(src / "meta" / "tasks.jsonl")
    assert len(tasks) == 1, f"expected one task in {src}/meta/tasks.jsonl, got {len(tasks)}"
    instruction = tasks[0]["task"]
    all_eps = [e["episode_index"] for e in read_jsonl(src / "meta" / "episodes.jsonl")]

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

    # one global normalization, identical in both splits
    stats = {k: stats_of(np.concatenate([arrays["train"][k], arrays["val"][k]]), quantiles=True)
             for k in ("action", "action.body_token", "observation.state")}
    for d in dst.values():
        for fname in ("stats.json", "stats_psi0.json"):
            (d / "meta" / fname).write_text(json.dumps(stats, indent=2))

    tok = stats["action.body_token"]
    print(f"token range [{min(tok['min']):.4f}, {max(tok['max']):.4f}] "
          f"(FSQ grid [-0.625, 0.625]; the deploy client clips to it)")
    const = [i for i, (lo, hi) in enumerate(zip(stats["observation.state"]["min"], stats["observation.state"]["max"]))
             if hi - lo < 1e-4]
    if const:
        print(f"note: constant state dims (skipped by normalization): {[STATE_0810[i] for i in const]}")

    if a.zip:
        zpath = a.out / f"{name}_psi0.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:   # parquet + mp4 are already compressed
            for d in dst.values():
                for f in sorted(p for p in d.rglob("*") if p.is_file()):
                    z.write(f, f.relative_to(a.out).as_posix())
        print(f"-> {zpath} ({zpath.stat().st_size / 2**20:.0f} MiB)")
    for s, d in dst.items():
        print(f"-> {d}: {len(train if s == 'train' else val)} episodes, "
              f"{len(arrays[s]['action'])} frames")


if __name__ == "__main__":
    main()
