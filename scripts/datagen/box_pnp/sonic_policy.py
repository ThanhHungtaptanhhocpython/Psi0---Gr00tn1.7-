"""Pure-Python SONIC (GEAR-SONIC release) inference for G1 in MuJoCo.

Re-implements the observation/action path of the C++ deploy program
(third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref)
so the data generator can run offline, on CPU, without the TensorRT build:

  reference motion (50 Hz, IsaacLab joint order)
      -> encoder (g1 mode, 1762-D obs)          -> 64-D token
  token + 10-frame proprio history (994-D obs)  -> decoder -> 29 joint actions
      -> q_des = default + action * action_scale (MuJoCo order), PD at sim rate

Source references (GR00T-WholeBodyControl):
  policy_parameters.hpp                     gains, scales, default pose, joint maps
  g1_deploy_onnx_ref.cpp:753-920            motion_joint_{positions,velocities}_10frame_step5
  g1_deploy_onnx_ref.cpp:555-708            motion_anchor_orientation (heading-aligned, 6D)
  g1_deploy_onnx_ref.cpp:1467-1647          his_* decoder history (oldest -> newest)
  g1_deploy_onnx_ref.cpp:3152-3159          action -> joint target
  policy/release/observation_config.yaml    encoder / decoder obs layout
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

# --------------------------------------------------------------------------- constants
# Joint maps, named exactly as in policy_parameters.hpp:
#   isaac_vec = mj_vec[MUJOCO_TO_ISAACLAB] ;  mj_vec = isaac_vec[ISAACLAB_TO_MUJOCO]
ISAACLAB_TO_MUJOCO = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                               11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                               16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])

# Standing pose, MuJoCo order (policy_parameters.hpp default_angles)
DEFAULT_ANGLES_MJ = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])

_W = 10 * 2.0 * 3.1415926535
_ARM = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFF = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_KP = {k: a * _W * _W for k, a in _ARM.items()}
_KD = {k: 2.0 * 2.0 * a * _W for k, a in _ARM.items()}
_LEG = ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"]
_ARM_J = ["5020", "5020", "5020", "5020", "5020", "4010", "4010"]
MOTOR_TYPE_MJ = _LEG + _LEG + ["7520_14", "5020", "5020"] + _ARM_J + _ARM_J
# ankles and waist roll/pitch run at 2x the 5020 gains (but keep the 5020 action scale)
_DOUBLE = {4, 5, 10, 11, 13, 14}
KP_MJ = np.array([_KP[t] * (2.0 if i in _DOUBLE else 1.0) for i, t in enumerate(MOTOR_TYPE_MJ)])
KD_MJ = np.array([_KD[t] * (2.0 if i in _DOUBLE else 1.0) for i, t in enumerate(MOTOR_TYPE_MJ)])
ACTION_SCALE_MJ = np.array([0.25 * _EFF[t] / _KP[t] for t in MOTOR_TYPE_MJ])

# Dex3 (per side: thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1)
HAND_KP, HAND_KD = 1.5, 0.1
LEFT_HAND_CLOSED = np.array([0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75])
RIGHT_HAND_CLOSED = np.array([0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75])
HAND_OPEN = np.zeros(7)

# Encoder obs layout (policy/release/observation_config.yaml, total 1762)
ENC_DIM, ENC_MODE, ENC_JPOS, ENC_JVEL, ENC_ANCHOR = 1762, 0, 4, 294, 601
DEC_DIM, HIST = 994, 10
N_FUTURE, FUTURE_STEP = 10, 5  # frames t, t+5, ..., t+45 of the 50 Hz reference

# Token grid the Psi0 client snaps to before publishing (psi_rtc_sonic_client.py)
FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 1.0 / 16.0


def fsq_snap(token: np.ndarray) -> np.ndarray:
    return np.round(np.clip(token, FSQ_MIN, FSQ_MAX) / FSQ_STEP) * FSQ_STEP


# --------------------------------------------------------------------------- quaternions (wxyz)
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    return quat_mul(quat_mul(q, np.array([0.0, *v])), quat_conj(q))[1:]


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_of(q) -> float:
    fx = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(fx[1], fx[0]))


def quat_from_yaw(yaw: float):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


# --------------------------------------------------------------------------- reference
@dataclass
class ReferenceFrame:
    """One 50 Hz reference frame: IsaacLab-order joints (absolute rad) + pelvis quat (wxyz, world)."""
    joint_pos: np.ndarray  # (29,)
    joint_vel: np.ndarray  # (29,)
    root_quat: np.ndarray  # (4,)


# --------------------------------------------------------------------------- policy
class SonicPolicy:
    def __init__(self, policy_dir: str | Path, threads: int = 4):
        policy_dir = Path(policy_dir)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(policy_dir / "model_encoder.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(str(policy_dir / "model_decoder.onnx"), so, providers=prov)
        self.reset()

    def reset(self):
        self.last_action = np.zeros(29, dtype=np.float32)
        # history entries: (ang_vel(3), body_q_rel(29), body_dq(29), last_action(29), gravity(3))
        pad = (np.zeros(3), np.zeros(29), np.zeros(29), np.zeros(29), np.array([0.0, 0.0, 1.0]))
        self.hist = deque([pad] * HIST, maxlen=HIST)  # padded frames reproduce the C++ quirk
        self.heading_delta = None  # quat applied to reference roots (set on first encode)

    # ---- encoder -----------------------------------------------------------------
    def align_heading(self, robot_quat, ref_root_quat_at_start):
        """apply_delta_heading = heading(robot) * heading(ref)^-1, captured once at start."""
        self.heading_delta = quat_from_yaw(yaw_of(robot_quat) - yaw_of(ref_root_quat_at_start))

    def encode(self, ref: list[ReferenceFrame], t: int, robot_quat: np.ndarray) -> np.ndarray:
        """Token for reference frame index t (future frames clamped to the last one)."""
        if self.heading_delta is None:
            self.align_heading(robot_quat, ref[0].root_quat)
        obs = np.zeros(ENC_DIM, dtype=np.float32)  # g1 mode: encoder_mode_4 = [0,0,0,0]
        last = len(ref) - 1
        conj_base = quat_conj(robot_quat)
        for k in range(N_FUTURE):
            f = ref[min(t + k * FUTURE_STEP, last)]
            obs[ENC_JPOS + 29 * k: ENC_JPOS + 29 * (k + 1)] = f.joint_pos
            obs[ENC_JVEL + 29 * k: ENC_JVEL + 29 * (k + 1)] = f.joint_vel
            rel = quat_mul(conj_base, quat_mul(self.heading_delta, f.root_quat))
            R = quat_to_mat(rel)
            obs[ENC_ANCHOR + 6 * k: ENC_ANCHOR + 6 * (k + 1)] = R[:, :2].reshape(-1)
        return self.enc.run(None, {"obs_dict": obs[None]})[0][0].astype(np.float32)

    # ---- decoder -----------------------------------------------------------------
    def log_state(self, q_mj: np.ndarray, dq_mj: np.ndarray, base_quat: np.ndarray, gyro: np.ndarray):
        """Push the current tick's proprio state (call BEFORE decode, like the C++ loop)."""
        body_q = q_mj[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_MJ[MUJOCO_TO_ISAACLAB]
        body_dq = dq_mj[MUJOCO_TO_ISAACLAB]
        grav = quat_rotate(quat_conj(base_quat), np.array([0.0, 0.0, -1.0]))
        self.hist.append((gyro.copy(), body_q, body_dq, self.last_action.copy(), grav))

    def decode(self, token: np.ndarray) -> np.ndarray:
        """Returns the MuJoCo-order joint position target (29,)."""
        h = list(self.hist)  # oldest -> newest
        obs = np.concatenate([
            token.astype(np.float32),
            np.concatenate([e[0] for e in h]),
            np.concatenate([e[1] for e in h]),
            np.concatenate([e[2] for e in h]),
            np.concatenate([e[3] for e in h]),
            np.concatenate([e[4] for e in h]),
        ]).astype(np.float32)
        assert obs.shape[0] == DEC_DIM, obs.shape
        action = self.dec.run(None, {"obs_dict": obs[None]})[0][0].astype(np.float32)
        self.last_action = action
        return DEFAULT_ANGLES_MJ + action[ISAACLAB_TO_MUJOCO] * ACTION_SCALE_MJ


def pd_torque(q_des, q, dq, kp=KP_MJ, kd=KD_MJ):
    return kp * (q_des - q) - kd * dq
