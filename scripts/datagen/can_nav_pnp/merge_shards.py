"""Merge generation shards (each a raw can_pnp dataset written by generate.py) into one dataset.

  python scripts/datagen/can_pnp/merge_shards.py --out .data/can_pnp_200 .data/gen/normal_s0 .data/gen/regrasp_s0 ...

Episodes are renumbered in shard order; parquet episode_index/index, the ego + third-person videos and
meta/{episodes,episodes_stats,episode_meta}.jsonl are rewritten accordingly (episode_meta keeps the
source shard + episode), generation_log.jsonl is concatenated (with the shard name) and info.json gets
the new totals. The shards themselves are not modified. Shards generated before the wrist camera existed have
no wrist video: run render_views.py --views wrist on the merged dataset afterwards (episodes simulated with
--no-render: render_views.py --views ego).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from writer import CHUNK, EGO_KEY, WRIST_KEY


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_jsonl(p: Path, rows: list[dict]):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "data" / CHUNK).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    episodes, stats, metas, gen_log = [], [], [], []
    total = 0
    for shard in map(Path, a.shards):
        by_ep = {m["episode_index"]: m for m in read_jsonl(shard / "meta" / "episode_meta.jsonl")}
        st_by_ep = {s["episode_index"]: s for s in read_jsonl(shard / "meta" / "episodes_stats.jsonl")}
        gen_log += [{**r, "shard": shard.name} for r in read_jsonl(shard / "meta" / "generation_log.jsonl")]
        for e in read_jsonl(shard / "meta" / "episodes.jsonl"):
            old, new = e["episode_index"], len(episodes)
            df = pd.read_parquet(shard / "data" / CHUNK / f"episode_{old:06d}.parquet")
            df["episode_index"] = new
            df["index"] = range(total, total + len(df))
            df.to_parquet(out / "data" / CHUNK / f"episode_{new:06d}.parquet", index=False)
            for sub in (Path("videos") / CHUNK / EGO_KEY, Path("videos") / CHUNK / WRIST_KEY, Path("debug") / "third_person"):
                src = shard / sub / f"episode_{old:06d}.mp4"
                if src.exists():
                    (out / sub).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, out / sub / f"episode_{new:06d}.mp4")
            episodes.append({**e, "episode_index": new})
            stats.append({**st_by_ep[old], "episode_index": new})
            metas.append({**by_ep[old], "episode_index": new, "source": {"shard": str(shard), "episode_index": old}})
            total += len(df)
        print(f"{shard}: {len(by_ep)} episodes")

    first = Path(a.shards[0]) / "meta"
    write_jsonl(out / "meta" / "episodes.jsonl", episodes)
    write_jsonl(out / "meta" / "episodes_stats.jsonl", stats)
    write_jsonl(out / "meta" / "episode_meta.jsonl", metas)
    write_jsonl(out / "meta" / "generation_log.jsonl", gen_log)
    for name in ("tasks.jsonl", "modality.json"):
        shutil.copyfile(first / name, out / "meta" / name)
    info = json.loads((first / "info.json").read_text())
    n = len(episodes)
    info.update(total_episodes=n, total_frames=total, total_videos=n, splits={"train": f"0:{n}"})
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    n_re = sum(1 for m in metas if m.get("info", {}).get("regrasp"))
    print(f"-> {out}: {n} episodes ({n_re} regrasp), {total} frames")


if __name__ == "__main__":
    main()
