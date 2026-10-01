"""Generate successful episodes of the box pick -> walk -> place -> home task.

Examples (from the repo root, with the datagen venv):
  # preview: stop after the first successful episode, write the review video/plots
  .venv-datagen/Scripts/python scripts/datagen/box_pnp/generate.py --out .data/box_pnp_preview --num-episodes 1 --review

  # full run (shard across processes with disjoint seed ranges, then merge)
  python scripts/datagen/box_pnp/generate.py --out .data/box_pnp_s0 --num-episodes 125 --seed-start 0
  python scripts/datagen/box_pnp/generate.py --out .data/box_pnp_s1 --num-episodes 125 --seed-start 100000

  # pilot / success-rate check: exactly 12 seeds, physics only (no rendering, nothing written but
  # the attempt log), then summarize the per-phase diagnostics
  python scripts/datagen/box_pnp/generate.py --out .data/box_pnp_pilot12 --dry-run --num-episodes 12 --max-attempts 12
  python scripts/datagen/box_pnp/summarize_log.py .data/box_pnp_pilot12

  # regrasp episodes (scripted slipping first grasp, then a successful one)
  python scripts/datagen/box_pnp/generate.py --out .data/gen/regrasp_s0 --num-episodes 20 --seed-start 900000 --set regrasp=True

Only clean episodes are written; every attempt (and its rejection reason + per-phase
diagnostics) goes to meta/generation_log.jsonl. The physics always runs headless; a successful
episode is then rendered from its recorded states straight into the mp4s (render.py), so rejected
attempts cost no rendering and no frames are held in memory.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from episode import Config, Episode  # noqa: E402
from planner import KinematicPlanner  # noqa: E402
from scene import REPO_ROOT, sample_params  # noqa: E402
from sonic_policy import SonicPolicy  # noqa: E402
from writer import DatasetWriter  # noqa: E402

SONIC_DIR = REPO_ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
PLANNER = REPO_ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-episodes", type=int, default=1, help="successful episodes to write")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = 20x num-episodes")
    ap.add_argument("--level", type=int, default=0, help="randomization level (0 = baseline)")
    ap.add_argument("--no-debug-video", action="store_true", help="skip the third-person review video")
    ap.add_argument("--review", action="store_true", help="also write review video + plots per episode")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate + log attempts only (no rendering, no episodes written)")
    ap.add_argument("--save-failed", action="store_true",
                    help="write review videos + a contact sheet of rejected attempts to <out>/failed/")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an episode.Config field (python literal), e.g. --set feet_slip_tol=0.05")
    args = ap.parse_args()

    policy = SonicPolicy(SONIC_DIR, threads=args.threads)
    planner = KinematicPlanner(PLANNER, threads=args.threads)
    writer = DatasetWriter(args.out, SONIC_DIR, PLANNER)
    overrides = {k: ast.literal_eval(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = Config(debug_render=not args.no_debug_video, render=False, **overrides)
    max_attempts = args.max_attempts or 20 * args.num_episodes

    ok = tried = 0
    t0 = time.time()
    seed = args.seed_start
    while ok < args.num_episodes and tried < max_attempts:
        params = sample_params(seed, level=args.level)
        policy.reset()
        result = Episode(params, policy, planner, cfg).run()
        writer.log_attempt(seed, result, params)
        tried += 1
        if result.success and args.dry_run:
            ok += 1
            print(f"[seed {seed}] SUCCESS (dry run, {result.info['n_frames']} frames, "
                  f"sim {result.info['sim_time']:.1f}s, wall {result.info['wall_time']:.1f}s)", flush=True)
        elif result.success:
            ep = writer.add_episode(result, params, cfg)
            ok += 1
            print(f"[seed {seed}] SUCCESS -> episode {ep:06d}  ({result.info['n_frames']} frames, "
                  f"sim {result.info['sim_time']:.1f}s, wall {result.info['wall_time']:.1f}s)", flush=True)
            if args.review:
                from review_episode import review
                review(args.out, ep)
        else:
            print(f"[seed {seed}] rejected in {result.info.get('fail_phase')}: {result.fail_reason}", flush=True)
            if args.save_failed and not args.dry_run and result.frames.get("frame_index"):
                writer.save_failed(result, seed, params)
        seed += 1
    dt = time.time() - t0
    print(f"\n{ok}/{tried} attempts succeeded ({100 * ok / max(tried, 1):.0f}%), {dt / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
