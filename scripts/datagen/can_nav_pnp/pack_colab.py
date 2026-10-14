"""Pack what can_nav_pnp data generation needs into one zip for Colab (colab_datagen.ipynb):

  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/pack_colab.py            # data generation (colab_datagen.ipynb)
  .venv-datagen/bin/python scripts/datagen/can_nav_pnp/pack_colab.py --eval     # closed-loop eval only (server notebooks)

--eval leaves out the SONIC planner ONNX (~740 MB, only the expert's walk / squat references use it; in closed-loop
eval the actions come from the policy) and writes .data/colab/can_nav_eval.zip instead of can_nav_datagen.zip.

The zip keeps the repo layout under psi0/ (scene.REPO_ROOT resolves the same way on Colab):
  psi0/scripts/datagen/can_nav_pnp/*.py
  psi0/third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/   G1 + Dex3 MJCF and meshes
  psi0/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/<SONIC_VARIANT>/     encoder, decoder, obs config
  psi0/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/planner/.../planner_sonic.onnx
  psi0/.data/<name>/meta/episode_meta.jsonl   the pick/place layouts of --layouts datasets (closed_loop_eval.py
                                    --layout-from on Colab), only episode_index / seed / layout_xy / layout_yaw
  psi0/requirements_datagen.txt     every package of the running interpreter, pinned (run it with .venv-datagen)
  psi0/PACK_INFO.json               python version, git commit, sha256 of the ONNX files
Symlinks (the SONIC checkpoints point into another checkout) are stored as the files they point to.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate import PLANNER, SONIC_DIR  # noqa: E402
from scene import G1_SCENE, REPO_ROOT  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="closed-loop eval pack: no planner (default --out can_nav_eval.zip)")
    ap.add_argument("--out", type=Path, default=None, help="default .data/colab/can_nav_datagen.zip (--eval: can_nav_eval.zip)")
    ap.add_argument("--layouts", type=Path, nargs="*", default=[REPO_ROOT / ".data/can_nav_160"],
                    help="generated datasets whose layouts go in (for closed_loop_eval.py --layout-from)")
    a = ap.parse_args()
    a.out = a.out or REPO_ROOT / ".data/colab" / ("can_nav_eval.zip" if a.eval else "can_nav_datagen.zip")

    files = sorted(p for p in HERE.glob("*.py"))
    files += sorted(p for p in G1_SCENE.parent.rglob("*") if p.is_file())
    files += [SONIC_DIR / n for n in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml")]
    if not a.eval:
        files.append(PLANNER)
    missing = [str(p) for p in files if not p.resolve().is_file()]
    if missing:
        raise SystemExit(f"missing: {missing}")

    reqs = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()
                  if d.metadata["Name"] and d.metadata["Name"].lower() not in ("pip", "setuptools", "wheel"))
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", str(HERE)],
                                    capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    info = {"pack": "eval" if a.eval else "datagen", "python": platform.python_version(), "git_commit": commit, "can_nav_pnp_uncommitted_changes": dirty,
            "sonic_dir": str(SONIC_DIR.relative_to(REPO_ROOT)),
            "sha256": {str(p.relative_to(REPO_ROOT)): sha256(p.resolve()) for p in files if p.suffix == ".onnx"}}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p.resolve(), (Path("psi0") / p.relative_to(REPO_ROOT)).as_posix())
        z.writestr("psi0/requirements_datagen.txt", "\n".join(reqs) + "\n")
        for ds in a.layouts:
            src = ds / "meta" / "episode_meta.jsonl"
            if not src.is_file():
                print(f"skip layouts of {ds}: no {src}")
                continue
            rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
            slim = [{"episode_index": r["episode_index"], "seed": r["seed"],
                     "scene": {k: r["scene"][k] for k in ("layout_xy", "layout_yaw")}} for r in rows]
            z.writestr(f"psi0/.data/{ds.resolve().name}/meta/episode_meta.jsonl",
                       "".join(json.dumps(r) + "\n" for r in slim))
            print(f"layouts of {ds.resolve().name}: {len(slim)} episodes")
        z.writestr("psi0/PACK_INFO.json", json.dumps(info, indent=2))
    print(f"{len(files)} files, {len(reqs)} pinned packages, python {info['python']}, commit {commit}"
          f"{' (+ uncommitted can_nav_pnp changes)' if dirty else ''}")
    print(f"-> {a.out} ({a.out.stat().st_size / 2**20:.0f} MiB)")


if __name__ == "__main__":
    main()
