"""Generate successful episodes of the can_nav task: walk around the obstacle -> squat -> pick the can off the
pedestal -> stand -> walk -> place in the tray -> home. SONIC 1.1 by default (SONIC_VARIANT=release for 1.0, which
cannot reach the can).

Examples (Ubuntu terminal, from the psi0 repo root):
  # preview: stop after the first successful episode, write the review video/plots
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py --out .data/can_nav_preview --num-episodes 1 --review

  # full run (shard across processes with disjoint seed ranges, then merge); or run_blocks.py (local / Colab)
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py --out .data/can_nav_s0 --num-episodes 32 --seed-start 0
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py --out .data/can_nav_s1 --num-episodes 32 --seed-start 10000
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/merge_shards.py --out .data/can_nav_64 .data/can_nav_s0 .data/can_nav_s1

  # pilot / success-rate check: exactly 12 seeds, physics only (no rendering, nothing written but
  # the attempt log), then summarize the per-phase diagnostics
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py --out .data/can_nav_pilot12 --dry-run \
      --num-episodes 12 --max-attempts 12 --seed-start 900000 --level 1
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/summarize_log.py .data/can_nav_pilot12

Only clean episodes are written; every attempt (and its rejection reason + per-phase
diagnostics) goes to meta/generation_log.jsonl. The physics always runs headless; a successful
episode is then rendered from its recorded states straight into the mp4s (render.py), so rejected
attempts cost no rendering and no frames are held in memory (--no-render: later, with render_views.py).
"""

from __future__ import annotations

import argparse
import ast
import os
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

POLICY_ROOT = REPO_ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy"
# SONIC controller: "release" (1.0) or "sonic_v1_1" (download: python download_from_hf.py --sonic-v1-1 in
# third_party/GR00T-WholeBodyControl). The env var lets closed_loop_eval.py and the probes pick it too.
SONIC_VARIANT = os.environ.get("SONIC_VARIANT", "sonic_v1_1")
SONIC_DIR = POLICY_ROOT / SONIC_VARIANT
PLANNER = REPO_ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx"


def check_sonic_dir():
    """Fail early with a useful message (sonic_v1_1 is often a symlink to a checkpoint dir elsewhere)."""
    missing = [f for f in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml")
               if not (SONIC_DIR / f).is_file()]
    if missing:
        raise SystemExit(f"SONIC policy '{SONIC_VARIANT}' not usable at {SONIC_DIR} (missing {missing}). "
                         f"Download it: cd third_party/GR00T-WholeBodyControl && "
                         f"python download_from_hf.py --sonic-v1-1, or fix the symlink.")


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
    ap.add_argument("--wrist-video", action="store_true", help="also render the right wrist camera video")
    ap.add_argument("--no-render", action="store_true",
                    help="write parquet + meta only; render the videos later with render_views.py (e.g. a CPU-only "
                         "Colab runtime simulates, a GPU machine renders)")
    ap.add_argument("--save-failed", action="store_true",
                    help="write review videos + a contact sheet of rejected attempts to <out>/failed/")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an episode.Config field (python literal), e.g. --set feet_slip_tol=0.05")
    args = ap.parse_args()

    check_sonic_dir()
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
        ep_sim = Episode(params, policy, planner, cfg)
        result = ep_sim.run()
        params = ep_sim.p          # the pick/place layout follows the READY stance (Episode._relayout)
        writer.log_attempt(seed, result, params)
        tried += 1
        if result.success and args.dry_run:
            ok += 1
            print(f"[seed {seed}] SUCCESS (dry run, {result.info['n_frames']} frames, "
                  f"sim {result.info['sim_time']:.1f}s, wall {result.info['wall_time']:.1f}s)", flush=True)
        elif result.success:
            ep = writer.add_episode(result, params, cfg, render=not args.no_render, wrist=args.wrist_video)
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
