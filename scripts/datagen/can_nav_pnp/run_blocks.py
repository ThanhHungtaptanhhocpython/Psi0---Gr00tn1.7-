"""Generate episodes as a queue of small seed blocks run in parallel, each block copied to a persistent directory
(Google Drive on Colab) as soon as it finishes. A rerun (e.g. after the Colab runtime was recycled) skips the blocks
already there, so at most the blocks in flight are lost.

  python scripts/datagen/can_nav_pnp/run_blocks.py --out-dir /content/drive/MyDrive/can_nav/blocks \\
      --work-dir /content/work --seed-base 100000 --target 170 --workers 7 --level 1

Block k simulates seeds seed_base + k*1000 ... (up to --block-size successes in --block-size*5 attempts) with
generate.py into <work-dir>/blk_<seed>, then moves it to <out-dir>/blk_<seed> (copied under a .part name, renamed
when complete). New blocks are started until the finished + running blocks cover --target episodes. Give every
machine / Colab session its own --seed-base (keep all seeds below 1 000 000, the eval range starts there); the
blocks are later merged with merge_shards.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED_STRIDE = 1000           # seeds per block slot (a block uses at most block_size * 5 of them)


def episodes_in(block: Path) -> int:
    p = block / "meta" / "episodes.jsonl"
    return sum(1 for l in p.read_text().splitlines() if l.strip()) if p.exists() else 0


def is_done(block: Path) -> bool:
    return (block / "DONE").exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="persistent dir the finished blocks go to")
    ap.add_argument("--work-dir", type=Path, required=True, help="fast local scratch dir")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--target", type=int, required=True, help="episodes to have in --out-dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--block-size", type=int, default=8, help="successful episodes per block")
    ap.add_argument("--level", type=int, default=0, help="randomization level (generate.py --level)")
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime threads per worker")
    ap.add_argument("--no-render", action="store_true", help="generate.py --no-render (render later: render_views.py)")
    ap.add_argument("--debug-video", action="store_true", help="also render the third-person review video")
    a = ap.parse_args()
    # blocks needed for the target, with room for blocks that come out short or fail
    last_seed = a.seed_base + SEED_STRIDE * (2 * -(-a.target // a.block_size) + 10)
    if last_seed > 1_000_000:
        raise SystemExit(f"blocks may reach seed {last_seed}: past 1 000 000 they overlap the eval seeds, "
                         f"lower --seed-base")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.work_dir / "logs").mkdir(parents=True, exist_ok=True)

    def block_dir(k: int) -> tuple[int, str]:
        seed = a.seed_base + k * SEED_STRIDE
        return seed, f"blk_{seed:06d}"

    def done_episodes() -> int:
        return sum(episodes_in(b) for b in a.out_dir.glob("blk_*") if is_done(b) and not b.name.endswith(".part"))

    running: dict[str, tuple[subprocess.Popen, float]] = {}
    k, t0, failed = 0, time.time(), 0
    have0 = done_episodes()
    print(f"{a.out_dir}: {have0} episodes already there; target {a.target}, {a.workers} workers", flush=True)
    last_report = 0.0
    while True:
        # finished blocks -> out_dir
        for name, (proc, t_start) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            work = a.work_dir / name
            if proc.returncode != 0:
                failed += 1
                tail = (a.work_dir / "logs" / f"{name}.log").read_text(errors="replace").splitlines()[-15:]
                print(f"[{name}] generate.py exited {proc.returncode}, see {a.work_dir}/logs/{name}.log:\n  "
                      + "\n  ".join(tail), flush=True)
                if failed >= 2 * a.workers and done_episodes() == have0:
                    for p, _ in running.values():
                        p.terminate()
                    raise SystemExit(f"{failed} blocks crashed and none finished: stopping (see the log above)")
                continue
            dst, part = a.out_dir / name, a.out_dir / f"{name}.part"
            shutil.rmtree(part, ignore_errors=True)
            shutil.copytree(work, part)
            shutil.copyfile(a.work_dir / "logs" / f"{name}.log", part / "generate.log")
            (part / "DONE").write_text(json.dumps({"episodes": episodes_in(work), "seconds": round(time.time() - t_start)}))
            shutil.rmtree(dst, ignore_errors=True)
            part.rename(dst)
            shutil.rmtree(work, ignore_errors=True)
            print(f"[{name}] {episodes_in(dst)} episodes in {(time.time() - t_start) / 60:.1f} min -> {dst}", flush=True)
        have = done_episodes()
        # start new blocks while finished + in-flight blocks do not cover the target
        while len(running) < a.workers and have + a.block_size * len(running) < a.target:
            seed, name = block_dir(k)
            k += 1
            if is_done(a.out_dir / name):
                continue
            work = a.work_dir / name
            shutil.rmtree(work, ignore_errors=True)
            cmd = [sys.executable, str(HERE / "generate.py"), "--out", str(work), "--num-episodes", str(a.block_size),
                   "--max-attempts", str(a.block_size * 5), "--seed-start", str(seed), "--level", str(a.level),
                   "--threads", str(a.threads)]
            if a.no_render:
                cmd.append("--no-render")
            if not a.debug_video:
                cmd.append("--no-debug-video")
            log = open(a.work_dir / "logs" / f"{name}.log", "w")
            running[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), time.time())
        if not running:
            break
        if time.time() - last_report > 300:
            last_report = time.time()
            rate = (have - have0) / max((time.time() - t0) / 3600, 1e-6)
            print(f"{time.strftime('%H:%M')} {have}/{a.target} episodes, {len(running)} blocks running, "
                  f"{rate:.0f} episodes/h", flush=True)
        time.sleep(5)
    print(f"done: {done_episodes()} episodes in {a.out_dir} ({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
