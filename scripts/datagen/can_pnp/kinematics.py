"""Whole-body kinematic IK for building SONIC reference keyframes.

Damped least-squares over the floating base (6) + 29 body joints, with weighted tasks:
feet pose locked, CoM xy inside the support polygon, right-hand grasp frame, pelvis height,
torso uprightness, and a posture prior. The result is a *kinematic* reference; SONIC
tracks it in physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from scene import Idx

# Grasp pocket of the right Dex3 hand, in the right_wrist_yaw_link frame (x along the
# fingers, +y towards the thumb). Found empirically: the palm occupies x<0.106, the
# thumb/fingers pinch region is around here.
GRASP_OFFSET = np.array([0.135, 0.05, 0.0])
# Power (whole-hand) grasp: where the axis of a 5.3 cm can sits when the hand wraps it, the can axis along the
# wrist z (the fingers curl about z). Found by sweeping the can through the hand model: the palm touches it,
# index/middle reach it at ~25-43 % closure, the thumb at ~70-80 %. x 0.125 rather than 0.115: 1 cm more room
# for the (opened, see Config.power_preshape) thumb while the can slides in.
POWER_OFFSET = np.array([0.125, 0.045, 0.0])


def rot_err(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """World-frame rotation error (small-angle), R_cur -> R_tgt."""
    a, b = R_cur, R_tgt   # sum_i cross(a[:, i], b[:, i]), written out row-wise
    return 0.5 * np.array([(a[1] * b[2] - a[2] * b[1]).sum(),
                           (a[2] * b[0] - a[0] * b[2]).sum(),
                           (a[0] * b[1] - a[1] * b[0]).sum()])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class IKTargets:
    feet: dict[int, tuple[np.ndarray, np.ndarray]]          # body id -> (pos, R)
    com_xy: np.ndarray | None = None
    hand_pos: np.ndarray | None = None                      # grasp point target (world)
    hand_R: np.ndarray | None = None                        # wrist frame target (world)
    pelvis_z: float | None = None
    pelvis_yaw: float | None = None                         # world yaw of the pelvis (floating base)
    torso_upright: float = 0.0                              # weight on torso z-axis == torso_axis
    torso_axis: np.ndarray | None = None                    # desired torso z-axis (default world z)
    posture: np.ndarray | None = None                       # 29 MuJoCo-order joint prior
    posture_w: np.ndarray | None = None                     # per-joint weights
    w_feet: float = 30.0
    w_com: float = 10.0
    w_hand_pos: float = 10.0
    w_hand_rot: float = 3.0
    w_pelvis: float = 3.0
    w_pelvis_yaw: float = 10.0


class WholeBodyIK:
    def __init__(self, m: mujoco.MjModel, ix: Idx):
        self.m, self.ix = m, ix
        self.d = mujoco.MjData(m)
        self.offset = GRASP_OFFSET.copy()          # grasp point in the right_wrist_yaw_link frame
        self.dofs = np.r_[np.arange(6), ix.body_vadr]           # optimized dofs
        self.n = len(self.dofs)
        lo = m.jnt_range[[m.dof_jntid[v] for v in ix.body_vadr], 0]
        hi = m.jnt_range[[m.dof_jntid[v] for v in ix.body_vadr], 1]
        self.lo, self.hi = lo + 0.02, hi - 0.02
        # Keep IK inside the range SONIC actually tracks (it straightens the knees and
        # stoops when the reference hip pitch goes past ~-2.0 rad).
        for j in (0, 6):          # hip pitch
            self.lo[j] = max(self.lo[j], -1.9)
        for j in (3, 9):          # knee
            self.hi[j] = min(self.hi[j], 2.2)
        # Wrist pitch/yaw run on the small 4010 motors (action scale 0.075): SONIC reaches
        # ~+-0.8 rad at most, so hand orientation has to come from shoulder/elbow/wrist roll.
        for j in (20, 21, 27, 28):
            self.lo[j], self.hi[j] = max(self.lo[j], -0.8), min(self.hi[j], 0.8)

    # -- helpers
    def fk(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d

    def grasp_point(self, d=None):
        d = d or self.d
        w = d.body(self.ix.rpalm)
        return w.xpos + w.xmat.reshape(3, 3) @ self.offset

    def com(self, d=None):
        d = d or self.d
        return d.subtree_com[self.ix.pelvis].copy()

    def _jac_point(self, body, point):
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.d, jp, jr, point, body)
        return jp[:, self.dofs], jr[:, self.dofs]

    # -- solve
    def solve(self, qpos0: np.ndarray, t: IKTargets, iters: int = 300, damping: float = 1e-3,
              step: float = 0.7) -> tuple[np.ndarray, dict]:
        m, ix = self.m, self.ix
        q = qpos0.copy()
        for _ in range(iters):
            d = self.fk(q)
            rows, errs, ws = [], [], []

            def add(J, e, w):
                rows.append(J); errs.append(e); ws.append(np.full(len(e), w))

            for b, (p, R) in t.feet.items():
                jp, jr = self._jac_point(b, d.xpos[b])
                add(jp, p - d.xpos[b], t.w_feet)
                add(jr, rot_err(d.xmat[b].reshape(3, 3), R), t.w_feet)
            if t.com_xy is not None:
                jc = np.zeros((3, m.nv))
                mujoco.mj_jacSubtreeCom(m, d, jc, ix.pelvis)
                add(jc[:2, self.dofs], t.com_xy - self.com(d)[:2], t.w_com)
            if t.hand_pos is not None:
                gp = self.grasp_point(d)
                jp, _ = self._jac_point(ix.rpalm, gp)
                add(jp, t.hand_pos - gp, t.w_hand_pos)
            if t.hand_R is not None:
                _, jr = self._jac_point(ix.rpalm, d.xpos[ix.rpalm])
                add(jr, rot_err(d.xmat[ix.rpalm].reshape(3, 3), t.hand_R), t.w_hand_rot)
            if t.pelvis_z is not None:
                jp, _ = self._jac_point(ix.pelvis, d.xpos[ix.pelvis])
                add(jp[2:3], np.array([t.pelvis_z - d.xpos[ix.pelvis][2]]), t.w_pelvis)
            if t.pelvis_yaw is not None:
                Rp = d.xmat[ix.pelvis].reshape(3, 3)
                e = (t.pelvis_yaw - np.arctan2(Rp[1, 0], Rp[0, 0]) + np.pi) % (2 * np.pi) - np.pi
                _, jr = self._jac_point(ix.pelvis, d.xpos[ix.pelvis])
                add(jr[2:3], np.array([e]), t.w_pelvis_yaw)
            if t.torso_upright > 0:
                Rt = d.xmat[ix.torso].reshape(3, 3)
                _, jr = self._jac_point(ix.torso, d.xpos[ix.torso])
                axis = t.torso_axis if t.torso_axis is not None else np.array([0, 0, 1.0])
                e = np.cross(Rt[:, 2], axis / np.linalg.norm(axis))
                add(jr, e, t.torso_upright)
            if t.posture is not None:
                J = np.zeros((29, self.n)); J[:, 6:] = np.eye(29)
                pw = t.posture_w if t.posture_w is not None else np.full(29, 0.05)
                e = t.posture - q[ix.body_qadr]
                rows.append(J); errs.append(e); ws.append(pw)

            J = np.vstack(rows); e = np.concatenate(errs); w = np.concatenate(ws)
            JW = J * w[:, None]
            H = JW.T @ JW + damping * np.eye(self.n)
            dq = np.linalg.solve(H, JW.T @ (e * w))
            full = np.zeros(m.nv); full[self.dofs] = step * dq
            mujoco.mj_integratePos(m, q, full, 1.0)
            q[ix.body_qadr] = np.clip(q[ix.body_qadr], self.lo, self.hi)
            q[3:7] /= np.linalg.norm(q[3:7])

        d = self.fk(q)
        info = {
            "feet_err": max(np.linalg.norm(p - d.xpos[b]) for b, (p, _) in t.feet.items()),
            "com_err": float(np.linalg.norm(t.com_xy - self.com(d)[:2])) if t.com_xy is not None else 0.0,
            "hand_err": float(np.linalg.norm(t.hand_pos - self.grasp_point(d))) if t.hand_pos is not None else 0.0,
            "hand_rot_err": float(np.linalg.norm(rot_err(d.xmat[ix.rpalm].reshape(3, 3), t.hand_R)))
            if t.hand_R is not None else 0.0,
            "pelvis_z": float(d.xpos[ix.pelvis][2]),
        }
        return q, info
