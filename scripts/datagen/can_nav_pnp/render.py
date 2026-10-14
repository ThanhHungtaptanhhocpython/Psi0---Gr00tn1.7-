"""Render ego / debug frames from recorded states (qpos + box pose), no physics.

Simulation runs headless; only successful episodes are rendered, from the states saved in
their parquet (generate.py). Also lets a finished dataset be re-rendered with a different camera.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np

from scene import IMG_H, IMG_W, RECORD_FPS, SceneParams, build_model, Idx


def open_video(path: Path, fps: int = RECORD_FPS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return iio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=8,
                          ffmpeg_params=["-pix_fmt", "yuv420p"])


def render_to_videos(params: SceneParams, qpos50: np.ndarray, box_pose: np.ndarray, ego_path: Path | None,
                     debug_path: Path | None = None, wrist_path: Path | None = None):
    """Like render_states, but each frame goes straight into the mp4 (no frame lists: a ~1100-frame
    episode is ~2.5 GB of raw ego + debug frames). Any path left None is not rendered."""
    m = build_model(params)
    d = mujoco.MjData(m)
    ix = Idx(m)
    r = mujoco.Renderer(m, IMG_H, IMG_W)
    writers = [(cam, open_video(path)) for cam, path in (("head_camera", ego_path), ("debug_cam", debug_path),
                                                         ("right_wrist_cam", wrist_path)) if path is not None]
    try:
        for q, b in zip(qpos50, box_pose):
            d.qpos[:50] = q
            d.qpos[ix.box_qadr:ix.box_qadr + 7] = b
            mujoco.mj_forward(m, d)
            for cam, w in writers:
                r.update_scene(d, camera=cam); w.append_data(r.render())
    finally:
        for _, w in writers:
            w.close()
        r.close()


def render_states(params: SceneParams, qpos50: np.ndarray, box_pose: np.ndarray, debug: bool = True):
    m = build_model(params)
    d = mujoco.MjData(m)
    ix = Idx(m)
    r = mujoco.Renderer(m, IMG_H, IMG_W)
    ego, dbg = [], []
    for q, b in zip(qpos50, box_pose):
        d.qpos[:50] = q
        d.qpos[ix.box_qadr:ix.box_qadr + 7] = b
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera="head_camera"); ego.append(r.render().copy())
        if debug:
            r.update_scene(d, camera="debug_cam"); dbg.append(r.render().copy())
    r.close()
    return ego, dbg
