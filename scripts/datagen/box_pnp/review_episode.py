"""Review one generated episode: annotated video, signal timeline, per-phase contact sheet, QA checks.

  python scripts/datagen/box_pnp/review_episode.py --root .data/box_pnp_preview --episode 0

Writes <root>/review/episode_XXXXXX_{review.mp4,timeline.png,phases.png,qa.json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

PHASE_COLORS = plt.get_cmap("tab20").colors


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def qa_checks(df: pd.DataFrame, n_video: int, meta: dict) -> dict:
    tok = np.stack(df["action.sonic_token"])
    act = np.stack(df["action"])
    st = np.stack(df["observation.state"])
    on_grid = np.abs(np.round(tok / 0.0625) * 0.0625 - tok).max()
    checks = {
        "frames_parquet": len(df), "frames_video": n_video,
        "video_matches_parquet": n_video == len(df),
        "fps_30_timestamps": bool(np.allclose(np.diff(df["timestamp"]), 1 / 30, atol=1e-4)),
        "action_dim_78": act.shape[1] == 78, "state_dim_43": st.shape[1] == 43,
        "no_nan": bool(np.isfinite(act).all() and np.isfinite(st).all()),
        "token_on_fsq_grid(max_dev)": float(on_grid),
        "token_range": [float(tok.min()), float(tok.max())],
        "token_unique_values": int(len(np.unique(np.round(tok, 5)))),
        "all_events_present": sorted(meta["events"]) == sorted(
            ["grasp_success", "object_lifted", "upright_reached", "walk_started", "table_reached",
             "object_supported", "release_success", "home_reached"]),
        "events_in_order": list(dict(sorted(meta["events"].items(), key=lambda kv: kv[1]))) == [
            "grasp_success", "object_lifted", "upright_reached", "walk_started", "table_reached",
            "object_supported", "release_success", "home_reached"],
        "feet_travel_before_walk_m": meta["info"]["max_foot_travel_prewalk"],
        "max_grasp_slip_m": meta["info"]["max_slip"],
        "max_robot_table_force_N": meta["info"]["max_table_force"],
        "walk_liftoffs": meta["info"]["walk_liftoffs"],
    }
    if "error.box_tilt_deg" in df:
        tilt = df["error.box_tilt_deg"].to_numpy()
        checks["max_box_tilt_while_grasped_deg"] = float(tilt[df["grasp.assist_active"].to_numpy()].max())
        checks["final_box_tilt_deg"] = float(tilt[-1])
        checks["final_place_xy_err_m"] = float(df["error.place_xy"].iloc[-1])
        checks["table_support_at_end"] = bool(df["state.table_support"].iloc[-1])
    if "state.box_in_tray" in df:
        checks["box_in_tray_at_end"] = bool(df["state.box_in_tray"].iloc[-1])
    return checks


def review(root: str | Path, episode: int):
    root = Path(root)
    df = pd.read_parquet(root / "data/chunk-000" / f"episode_{episode:06d}.parquet")
    meta = [json.loads(l) for l in (root / "meta/episode_meta.jsonl").read_text().splitlines() if l.strip()]
    meta = next(m for m in meta if m["episode_index"] == episode)
    phases = meta["phase_names"]
    ego = iio.mimread(root / "videos/chunk-000/observation.images.ego_view" / f"episode_{episode:06d}.mp4", memtest=False)
    dbg_p = root / "debug/third_person" / f"episode_{episode:06d}.mp4"
    dbg = iio.mimread(dbg_p, memtest=False) if dbg_p.exists() else None
    out = root / "review"; out.mkdir(exist_ok=True)
    stem = f"episode_{episode:06d}"
    events_at = {v: k for k, v in meta["events"].items()}

    # ---------- QA
    qa = qa_checks(df, len(ego), meta)
    (out / f"{stem}_qa.json").write_text(json.dumps(qa, indent=2))

    # ---------- annotated side-by-side video
    f_big, f_small = _font(22), _font(16)
    w = iio.get_writer(str(out / f"{stem}_review.mp4"), fps=30, codec="libx264", quality=7,
                       macro_block_size=8, ffmpeg_params=["-pix_fmt", "yuv420p"])
    ev_shown = []
    for i in range(len(df)):
        panels = [ego[i]] + ([dbg[i]] if dbg is not None else [])
        img = Image.fromarray(np.concatenate(panels, axis=1))
        dr = ImageDraw.Draw(img)
        ph = phases[int(df["phase"].iloc[i])]
        dr.rectangle([0, 0, img.width, 34], fill=(0, 0, 0))
        dr.text((10, 6), f"t={df['timestamp'].iloc[i]:5.2f}s  frame {i:4d}  phase: {ph}", fill=(255, 255, 255), font=f_big)
        grip = "GRASP ASSIST ON" if df["grasp.assist_active"].iloc[i] else "hand free"
        dr.text((img.width - 230, 8), grip, fill=(255, 200, 0) if df["grasp.assist_active"].iloc[i] else (180, 180, 180), font=f_small)
        if i in events_at:
            ev_shown.append((events_at[i], i))
        for j, (name, fi) in enumerate(ev_shown[-4:]):
            age = i - fi
            col = (80, 255, 80) if age < 45 else (160, 160, 160)
            dr.text((10, 44 + 22 * j), f"event: {name} @ {fi / 30:.2f}s", fill=col, font=f_small)
        dr.text((10, img.height - 26), "left: ego camera (policy input)" + ("   |   right: third-person debug view" if dbg is not None else ""),
                fill=(230, 230, 230), font=f_small)
        w.append_data(np.asarray(img))
    w.close()

    # ---------- timeline plot
    t = df["timestamp"].to_numpy()
    ph = df["phase"].to_numpy()
    root_pos = np.stack(df["observation.root_pos"])
    box = np.stack(df["object.box_pose"])
    tok = np.stack(df["action.sonic_token"])
    rh = np.stack(df["action.right_hand"])
    fig, axs = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    for ax in axs:
        start = 0
        for k in range(1, len(ph) + 1):
            if k == len(ph) or ph[k] != ph[start]:
                ax.axvspan(t[start], t[k - 1] + 1 / 30, color=PHASE_COLORS[ph[start] % 20], alpha=0.25, lw=0)
                if ax is axs[0]:
                    ax.text((t[start] + t[k - 1]) / 2, 1.02, phases[ph[start]], rotation=60, fontsize=7,
                            ha="left", va="bottom", transform=ax.get_xaxis_transform())
                start = k
        for name, fi in meta["events"].items():
            ax.axvline(t[fi], color="k", lw=0.8, ls="--")
    axs[0].plot(t, root_pos[:, 2], label="pelvis z"); axs[0].plot(t, box[:, 2], label="box z")
    axs[0].set_ylabel("height [m]"); axs[0].legend(loc="upper right")
    axs[1].plot(t, root_pos[:, 0], label="pelvis x"); axs[1].plot(t, box[:, 0], label="box x")
    axs[1].set_ylabel("forward [m]"); axs[1].legend(loc="upper right")
    axs[2].plot(t, np.linalg.norm(tok, axis=1), label="|token|"); axs[2].plot(t, tok[:, :4], lw=0.6)
    axs[2].set_ylabel("SONIC token"); axs[2].legend(loc="upper right")
    axs[3].plot(t, rh); axs[3].set_ylabel("right-hand cmd [rad]")
    axs[4].plot(t, df["contact.left_foot"].astype(int) + 2.2, label="L foot")
    axs[4].plot(t, df["contact.right_foot"].astype(int) + 1.1, label="R foot")
    axs[4].plot(t, df["grasp.assist_active"].astype(int), label="grasp assist")
    axs[4].plot(t, df["contact.box_table"].astype(int) * 0.9 - 1.1, label="box-table")
    if "contact.box_tray" in df:
        axs[4].plot(t, df["contact.box_tray"].astype(int) * 0.9 - 1.1, label="box-tray")
    axs[4].set_yticks([]); axs[4].legend(loc="upper right", ncol=4); axs[4].set_xlabel("time [s]")
    for name, fi in meta["events"].items():
        axs[4].text(t[fi], -1.3, name, rotation=90, fontsize=7, va="top")
    fig.suptitle(f"episode {episode} (seed {meta['seed']})")
    fig.tight_layout(); fig.savefig(out / f"{stem}_timeline.png", dpi=90); plt.close(fig)

    contact_sheet(ego, dbg, ph, t, phases, out / f"{stem}_phases.png", f"episode {episode} (seed {meta['seed']})")
    print(json.dumps(qa, indent=2))
    return out


def contact_sheet(ego, dbg, ph, t, phases, path, title=""):
    """Middle frame of each phase (ego on top of third-person), one tile per phase."""
    ph = np.asarray(ph)
    shown = [p for p in range(len(phases)) if (ph == p).any()]
    cols = 4
    rows = max(1, int(np.ceil(len(shown) / cols)))
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.0 * rows * (2 if dbg is not None else 1)))
    for a in np.ravel(axs):
        a.axis("off")
    for k, p in enumerate(shown):
        idx = np.where(ph == p)[0]
        i = idx[len(idx) // 2]
        img = ego[i] if dbg is None else np.concatenate([ego[i], dbg[i]], axis=0)
        a = np.ravel(axs)[k]
        a.imshow(img); a.set_title(f"{phases[p]}  (t={t[i]:.1f}s)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episode", type=int, default=0)
    a = ap.parse_args()
    review(a.root, a.episode)
