"""Render missing videos of a generated dataset from its recorded states, without re-simulating.

  # episodes simulated with generate.py --no-render (e.g. on a CPU-only Colab runtime): ego view
  python scripts/datagen/can_nav_pnp/render_views.py .data/can_nav_500 --views ego --jobs 4
  # add the right wrist camera to episodes generated before it existed
  python scripts/datagen/can_nav_pnp/render_views.py .data/can_nav_160 --views wrist --jobs 4

Rendering only reads the recorded states (observation.qpos + object.box_pose) and the episode's scene params
(meta/episode_meta.jsonl, incl. the pick/place layout), as writer.py does at generation time; cameras change no
physics, so the result is what generate.py would have written. Videos that already exist are skipped; a killed run
never leaves a truncated video under its final name. info.json / modality.json get the rendered video keys.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from render import render_to_videos  # noqa: E402
from scene import SceneParams  # noqa: E402
from writer import CHUNK, EGO_KEY, VIDEO_FEATURE, WRIST_KEY  # noqa: E402

# view -> (video dir relative to the dataset root, modality.json name or None for the review-only debug video)
VIEWS = {"ego": (Path("videos") / CHUNK / EGO_KEY, "ego_view"),
         "wrist": (Path("videos") / CHUNK / WRIST_KEY, "right_wrist_view"),
         "debug": (Path("debug") / "third_person", None)}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def scene_params(scene: dict) -> SceneParams:
    names = {f.name for f in dataclasses.fields(SceneParams)}
    return SceneParams(**{k: tuple(v) if isinstance(v, list) else v for k, v in scene.items() if k in names})


def render_one(job):
    root, ep, scene, views = job
    paths = {v: root / VIEWS[v][0] / f"episode_{ep:06d}.mp4" for v in views}
    tmp = {v: p.with_suffix(".part.mp4") for v, p in paths.items()}
    df = pd.read_parquet(root / "data" / CHUNK / f"episode_{ep:06d}.parquet", columns=["observation.qpos", "object.box_pose"])
    render_to_videos(scene_params(scene), np.stack(df["observation.qpos"].values), np.stack(df["object.box_pose"].values),
                     tmp.get("ego"), tmp.get("debug"), tmp.get("wrist"))
    for v, p in paths.items():
        tmp[v].rename(p)
    return ep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset written by generate.py / merge_shards.py")
    ap.add_argument("--views", default="ego", help="comma-separated subset of ego,wrist,debug")
    ap.add_argument("--jobs", type=int, default=4, help="parallel render processes (~1 GB RAM each)")
    a = ap.parse_args()

    root, views = a.root, [v.strip() for v in a.views.split(",") if v.strip()]
    if unknown := set(views) - set(VIEWS):
        raise SystemExit(f"unknown views {sorted(unknown)} (choose from {sorted(VIEWS)})")
    metas = {m["episode_index"]: m for m in read_jsonl(root / "meta" / "episode_meta.jsonl")}
    eps = [e["episode_index"] for e in read_jsonl(root / "meta" / "episodes.jsonl")]
    for v in views:
        (root / VIEWS[v][0]).mkdir(parents=True, exist_ok=True)
    todo = []
    for ep in eps:
        missing = tuple(v for v in views if not (root / VIEWS[v][0] / f"episode_{ep:06d}.mp4").exists())
        if missing:
            todo.append((root, ep, metas[ep]["scene"], missing))
    print(f"{root}: {len(eps)} episodes, {len(todo)} with missing {'/'.join(views)} videos")
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, _ in enumerate(ex.map(render_one, todo), 1):
            print(f"\r  rendered {i}/{len(todo)}", end="", flush=True)
    print()

    info = json.loads((root / "meta" / "info.json").read_text())
    modality = json.loads((root / "meta" / "modality.json").read_text())
    for v in views:
        rel, name = VIEWS[v]
        if name is not None:
            info["features"][str(rel.name)] = VIDEO_FEATURE
            modality.setdefault("video", {})[name] = {"original_key": rel.name}
    info["total_videos"] = len(eps) * sum(1 for f in info["features"].values() if f.get("dtype") == "video")
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (root / "meta" / "modality.json").write_text(json.dumps(modality, indent=2))
    print(f"-> {root}")


if __name__ == "__main__":
    main()
