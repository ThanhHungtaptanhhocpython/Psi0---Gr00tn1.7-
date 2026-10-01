"""GEAR-SONIC kinematic planner (planner_sonic.onnx) wrapper used for the WALK phases.

The planner turns (context qpos, mode, target position/heading) into a 30 Hz whole-body
motion (MuJoCo qpos, 36-D). We resample it to the 50 Hz reference stream and override
the upper body with the box-carrying pose, so SONIC walks with the arms held still.
Spec: third_party/GR00T-WholeBodyControl/docs/source/references/planner_onnx.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from reference import REF_DT, slerp

PLANNER_FPS = 30.0
MODE_IDLE, MODE_SLOW_WALK, MODE_WALK, MODE_SQUAT = 0, 1, 2, 4


class KinematicPlanner:
    def __init__(self, onnx_path: str | Path, threads: int = 8):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        k = self.inputs["allowed_pred_num_tokens"].shape[1]
        self.k = k if isinstance(k, int) else 11

    def plan(self, context_qpos36: np.ndarray, mode: int, target_xy=None, target_heading: float = 0.0,
             direction=(1.0, 0.0, 0.0), target_vel: float = -1.0, seed: int = 1234,
             height: float = -1.0) -> np.ndarray:
        """context_qpos36: (4, 36) at 30 Hz. Returns (N, 36) at 30 Hz, world frame."""
        allowed = np.zeros((1, self.k), dtype=np.int64)
        allowed[0, :6] = 1  # same as the C++ stack (24..44 frames)
        feed = {
            "context_mujoco_qpos": context_qpos36[None].astype(np.float32),
            "target_vel": np.array([target_vel], dtype=np.float32),
            "mode": np.array([mode], dtype=np.int64),
            "movement_direction": np.array([direction], dtype=np.float32),
            "facing_direction": np.array([[np.cos(target_heading), np.sin(target_heading), 0.0]], dtype=np.float32),
            "height": np.array([height], dtype=np.float32),
            "random_seed": np.array([seed], dtype=np.int64),
            "has_specific_target": np.array([[0 if target_xy is None else 1]], dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.full((1, 4), target_heading, dtype=np.float32),
            "allowed_pred_num_tokens": allowed,
        }
        if target_xy is not None:
            feed["specific_target_positions"][0, :, 0] = target_xy[0]
            feed["specific_target_positions"][0, :, 1] = target_xy[1]
            feed["specific_target_positions"][0, :, 2] = context_qpos36[-1, 2]
        feed = {k: v for k, v in feed.items() if k in self.inputs}
        out = self.sess.run(None, feed)
        qpos, n = out[0][0], int(np.asarray(out[1]).reshape(-1)[0])
        return qpos[:n].astype(np.float64)


def resample(qpos30: np.ndarray, t0: float = 0.0, fps_in: float = PLANNER_FPS, dt_out: float = REF_DT):
    """30 Hz (N,36) -> 50 Hz lists of (joints29, root_pos, root_quat); lerp joints/pos, slerp quat.

    Time 0 is qpos30[0]. Pass the planner output from its last context frame (out[3]) with
    t0=dt_out so the appended frames continue the reference at the right time; resampling
    out[4:] from t=0 instead compresses every replan by 13 ms and the walk runs away.
    """
    n = len(qpos30)
    T = (n - 1) / fps_in
    ts = np.arange(t0, T + 1e-9, dt_out)
    J, P, Q = [], [], []
    for t in ts:
        f = t * fps_in
        i = min(int(np.floor(f)), n - 2)
        s = f - i
        a, b = qpos30[i], qpos30[i + 1]
        P.append((1 - s) * a[:3] + s * b[:3])
        Q.append(slerp(a[3:7], b[3:7], s))
        J.append((1 - s) * a[7:] + s * b[7:])
    return J, P, Q


def splice(out: np.ndarray, ctx: np.ndarray, blend_frames: int = 8):
    """Planner output -> 50 Hz frames continuing the reference seamlessly.

    The planner blends its first 4 frames with the context, so out[3] is NOT exactly the
    context end: it runs a centimetre or two ahead. Appending from it as-is makes every replan
    jump forward, the next context reads that jump as extra speed, and the walk runs away.
    Remove the root xy offset and cross-fade the joints over the first frames (as the C++ does).
    """
    J, P, Q = resample(out[3:], t0=REF_DT)
    dp = out[3, :2] - ctx[3, :2]
    dj = ctx[3, 7:] - out[3, 7:]
    for k in range(len(J)):
        P[k] = P[k].copy(); P[k][:2] -= dp
        w = max(0.0, 1.0 - (k + 1) / blend_frames)
        J[k] = J[k] + w * dj
    return J, P, Q


def context_from_reference(q_list, pos_list, quat_list, dt_in: float = REF_DT) -> np.ndarray:
    """Last 4 frames of the 50 Hz reference, resampled to 30 Hz -> (4, 36)."""
    n = len(q_list)
    out = []
    for k in range(4):
        t = (n - 1) * dt_in - (3 - k) / PLANNER_FPS
        f = max(t / dt_in, 0.0)
        i = min(int(np.floor(f)), n - 1)
        j = min(i + 1, n - 1)
        s = f - i
        out.append(np.r_[(1 - s) * pos_list[i] + s * pos_list[j],
                         slerp(quat_list[i], quat_list[j], s),
                         (1 - s) * q_list[i] + s * q_list[j]])
    return np.array(out)
