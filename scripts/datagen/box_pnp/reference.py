"""50 Hz whole-body reference stream fed to the SONIC encoder.

Keyframes are full MuJoCo qpos vectors (from WholeBodyIK or the walking planner).
Segments are min-jerk interpolated in joint space; the pelvis orientation is slerped.
The buffer grows phase by phase as the state machine advances; the encoder clamps its
10-frame lookahead to the last appended frame, exactly like the C++ deploy.
"""

from __future__ import annotations

import numpy as np

from sonic_policy import MUJOCO_TO_ISAACLAB, ReferenceFrame

REF_DT = 0.02  # 50 Hz


def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(dot)
    return (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)


class ReferenceBuffer:
    def __init__(self, body_qadr: np.ndarray):
        self.body_qadr = body_qadr
        self.q: list[np.ndarray] = []        # 29 MuJoCo-order joints per frame
        self.root_quat: list[np.ndarray] = []
        self.root_pos: list[np.ndarray] = []
        self.frames: list[ReferenceFrame] = []

    def __len__(self):
        return len(self.frames)

    @property
    def last_qpos_parts(self):
        return self.q[-1].copy(), self.root_pos[-1].copy(), self.root_quat[-1].copy()

    def _push(self, qj, rpos, rquat):
        self.q.append(np.asarray(qj, float).copy())
        self.root_pos.append(np.asarray(rpos, float).copy())
        self.root_quat.append(np.asarray(rquat, float).copy())
        prev = self.q[-2] if len(self.q) > 1 else self.q[-1]
        vel = (self.q[-1] - prev) / REF_DT
        self.frames.append(ReferenceFrame(self.q[-1][MUJOCO_TO_ISAACLAB].copy(),
                                          vel[MUJOCO_TO_ISAACLAB].copy(), self.root_quat[-1].copy()))
        # make the previous frame's velocity forward-consistent too
        if len(self.frames) > 1:
            self.frames[-2].joint_vel = vel[MUJOCO_TO_ISAACLAB].copy()

    def start(self, qpos: np.ndarray):
        self._push(qpos[self.body_qadr], qpos[:3], qpos[3:7])

    def add_segment(self, qpos_to: np.ndarray, duration: float):
        """Min-jerk move from the last frame to `qpos_to` (full MuJoCo qpos)."""
        q0, p0, r0 = self.last_qpos_parts
        q1, p1, r1 = qpos_to[self.body_qadr], qpos_to[:3], qpos_to[3:7]
        n = max(1, int(round(duration / REF_DT)))
        for k in range(1, n + 1):
            s = float(min_jerk(np.array(k / n)))
            self._push(q0 + s * (q1 - q0), p0 + s * (p1 - p0), slerp(r0, r1, s))

    def add_hold(self, duration: float):
        q, p, r = self.last_qpos_parts
        for _ in range(max(1, int(round(duration / REF_DT)))):
            self._push(q, p, r)

    def truncate(self, n: int):
        """Drop frames >= n (used when replanning: the future of the old plan is replaced)."""
        n = max(1, n)
        del self.q[n:], self.root_pos[n:], self.root_quat[n:], self.frames[n:]

    def add_raw(self, qj_list, rpos_list, rquat_list):
        """Append externally generated 50 Hz frames (e.g. the resampled walking planner output)."""
        for qj, rp, rq in zip(qj_list, rpos_list, rquat_list):
            self._push(qj, rp, rq)
