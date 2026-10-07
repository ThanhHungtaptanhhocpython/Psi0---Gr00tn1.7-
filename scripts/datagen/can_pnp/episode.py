"""One episode of the can task: squat -> pick the can off the floor -> stand -> walk -> place in the tray -> home,
SONIC in the loop. Derived from box_pnp/episode.py; the grasped object (a slim can standing on the floor beside
the right foot) keeps the "box" names of that code (box_pose, box_geom, ...).

Phase machine (the spec's labels, stored per frame for debugging only). One thing at a time, each
step ends balanced (settle) before the next one starts:
  B1 HOME -> CROUCH                                  squat down in two stages (half, then full depth),
                                                     balance after each
  B2 -> TURN_TORSO -> REACH -> GRASP -> LIFT        turn the torso right (waist yaw), reach down to the can on
                                                     the floor, grasp it from the top, lift it clear
        [regrasp episodes: REACH (off target) -> GRASP (no assist) -> SLIP -> REGRASP_REACH -> GRASP]
  B3 -> RAISE_TORSO -> TURN_BACK                     raise the upper body + box, turn the torso back
  B4 -> STAND_UP -> STABILIZE_UPRIGHT                stand up, balance
  B5 -> WALK_STEP_1 -> WALK_STEP_2 -> STABILIZE_AT_TABLE -> ADJUST_ARM
                                                     walk, stop, balance, re-set the carry arm
  B6 -> APPROACH_PLACE -> LOWER_BOX -> RELEASE -> RETRACT -> RETURN_HOME -> SUCCESS
                                                     move the box over the tray, put it in, go home

Baseline design (simple, consistent expert data rather than pretty motion):
  * the wrist orientation reached at GRASP is held through LIFT/STAND_UP/WALK/PLACE; the box is
    moved by shoulder/elbow/torso translation only (no "upright box" re-orientation, no wrist servo
    while walking, at most one small (<10 deg) tilt correction before lowering);
  * the table is below the carry height, so PLACE is a straight lowering move into the tray in the
    table's near-right corner; success = the box ends up inside the tray (it may tip over in it:
    an upright, neatly placed box is not required yet, see Config.require_upright);
  * every phase logs its own diagnostics (EpisodeResult.info["diag"]).

Control stack per 50 Hz tick:
  expert appends kinematic reference frames (IK keyframes / planner walk) -> SONIC encoder
  -> 64-D token (latched at the 30 Hz record clock, as the Psi0 client does at deploy time)
  -> SONIC decoder -> PD at 200 Hz. The right hand is driven directly by 7 joint targets.

What gets recorded at 30 Hz is exactly what was sent: the latched token and the hand targets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from kinematics import GRASP_OFFSET, POWER_OFFSET, IKTargets, WholeBodyIK, rot_z
from planner import MODE_IDLE, MODE_SLOW_WALK, MODE_SQUAT, KinematicPlanner, context_from_reference, splice
from reference import REF_DT, ReferenceBuffer
from scene import CTRL_DT, IMG_H, IMG_W, RECORD_FPS, SIM_DT, Idx, SceneParams, build_model
from sonic_policy import (DEFAULT_ANGLES_MJ, HAND_KD, HAND_KP, HAND_OPEN, RIGHT_HAND_CLOSED,
                          SonicPolicy, pd_torque, quat_rotate, yaw_of)

PHASES = ["HOME", "CROUCH", "TURN_TORSO", "REACH", "GRASP", "SLIP", "REGRASP_REACH", "LIFT", "RAISE_TORSO",
          "TURN_BACK", "STAND_UP",
          "STABILIZE_UPRIGHT", "WALK_STEP_1", "WALK_STEP_2", "STABILIZE_AT_TABLE", "ADJUST_ARM",
          "APPROACH_PLACE", "LOWER_BOX", "RELEASE", "RETRACT", "RETURN_HOME", "SUCCESS"]
PH = {p: i for i, p in enumerate(PHASES)}
EVENTS = ["grasp_success", "object_lifted", "upright_reached", "walk_started", "table_reached",
          "object_supported", "release_success", "home_reached"]
INSTRUCTION = ("Squat down, pick up the can from the low stand on your right with your right hand, stand up, walk to "
               "the table, put the can in the tray on the table, then return to the home pose.")

# fingers down, thumb towards the robot's left (+y); rotated by the box yaw at runtime
R_GRASP = np.column_stack([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]])
SUBSTEPS = int(round(CTRL_DT / SIM_DT))  # 4
GRASP_BELOW_RIM = 0.010                  # m, pinch pocket centre below the can rim. The floor-level can is beyond
                                         # SONIC 1.1's reach: aiming lower made the reach servo push the hand down
                                         # until the squat lurched and the feet slid. 0.025 -> 0/48 fresh seeds,
                                         # 0.015 -> 5/48, 0.010 -> 8/48


class EpisodeFailed(Exception):
    pass


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 5)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


@dataclass
class Config:
    token_rate_hz: float = 30.0          # decoder sees a token latched at this rate (deploy parity)
    grasp_style: str = "power"           # "power": the whole hand wraps the can body from the side (can on the
                                         # pedestal); "pinch": thumb + fingertips on the rim from above (floor can)
    power_below_rim: float = 0.05        # power: hand (pocket) centre this far below the can rim
    power_heading_deg: float = -60.0     # power: finger direction, relative to the stance heading: fingers forward-
                                         # right, the palm facing forward-left onto the can (6 seeds: -60 -> 3 whole
                                         # episodes; -75 put the pre-grasp spot by the right knee, the IK could not
                                         # follow; -15/-45 reached out to the right and the feet slid)
    power_tilt_deg: float = 25.0         # power: fingers pitched down in the palm plane (the arm reaches down;
                                         # SONIC tracks wrist pitch only to ~+-0.8 rad)
    pregrasp_side: float = 0.08          # power: pre-grasp, the hand this far out from the can on the palm side ...
    pregrasp_up: float = 0.03            # ... and this much higher, then it closes in sideways
    pregrasp_high: float = 0.0           # power: the arm first swings this much above the pre-grasp spot (0.10:
                                         # meant to keep the swing off the can, but the feet slid 6-10 cm)
    power_preshape: tuple = (0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)   # power: hand pose while reaching, thumb_1
                                         # swung back out of the can's way: 21/50 vs 18/50 whole episodes (opened
                                         # straight, the thumb stopped the can 4-7 cm short of the palm)
    pregrasp_tol: float = 0.03           # power: grasp point error at the pre-grasp spot before sliding in ...
    pregrasp_gain: float = 0.6           # ... servo gain there (0.7 overshot back and forth by 3-5 cm) ...
    pregrasp_iters: int = 12             # ... at most this many corrections (8 stopped 6/12 seeds at 2.7-4 cm)
    pregrasp_hold: float = 0.4           # ... and the wait after each correction (SONIC settles slowly in the squat)
    slide_mode: str = "open"             # power slide-in: "open" (4 fixed steps along the palm normal, then <= 3
                                         # corrections, the last ones may push in 2 cm while the hand is off the
                                         # can): 8/30 whole episodes; "pocket" (steered by the can's position in
                                         # the hand): ~6/30
    power_offset_x: float = 0.115        # power: pocket position along the fingers (wrist frame x)
    slide_iters: int = 10                # power: slide-in steps at most ...
    slide_step: float = 0.025            # ... advancing towards the can by at most this per step ...
    slide_depth_tol: float = 0.01        # ... until the can is within this of the pocket depth
    lock_arms_crouch: bool = True        # CROUCH: both arms stay in the HOME pose until the squat has settled
    reach_comp: tuple = (0.0, 0.0, -0.02)  # initial aim offset (SONIC tracks the arm slightly short)
    squat_half_height: float = 0.60      # CROUCH stage 1: planner squat target pelvis height (m) ...
    squat_height: float = 0.42           # ... stage 2, the full squat (SONIC settles ~8-10 cm higher)
    turn_yaw: float = 0.0                # rad, TURN_TORSO: waist yaw (absolute) to the right, towards the box
                                         # can_pnp: 0 = only re-centre the waist (the planner squat yaws it ~+0.13);
                                         # with the reach IK's waist yaw free, a 0.7 turn was undone by the reach
                                         # anyway and halved the yield (1/6 vs 2/6 with SONIC 1.1)
                                         # (torso ~-30 deg; the box is ~-43 deg from the pelvis) ...
    turn_time: float = 1.2               # ... in this long; TURN_BACK undoes it (box held) ...
    turn_back_time: float = 1.5
    adjust_time: float = 1.0             # ADJUST_ARM: carry pose re-solved from the stance after the walk
    reach_tol: float = 0.04              # m, grasp point error to accept REACH (pocket is ~4-5 cm; GRASP
                                         # still requires finger contact with the box inside the pocket)
    pregrasp_time: float = 2.0           # slow pre-grasp swing down to the floor (a fast one rocks the squat)
    pregrasp_height: float = 0.0         # m above the grasp point (0: one slow move straight to it; stopping
                                         # above first and then servoing down rocked the squat more)
    reach_aim_clip: float = 0.15         # m, max aim offset per axis from the nominal target (SONIC tracks
                                         # the crouched arm ~8 cm high, the servo must be able to cover it)
    reach_aim_below: float = 0.10        # m, max aim offset below the nominal grasp height (pushing the aim
                                         # further down only pitched the squatting torso into the table)
    reach_com: bool = True               # REACH IK keeps the CoM over the feet (the arm reaches far down)
    reach_pelvis_w: float = 10.0         # REACH IK holds the pelvis at the squat height (0 = free)
    reach_waist_rp_w: float = 20.0       # REACH IK posture weight on waist roll/pitch (kept at the squat values:
                                         # 1.0 let them swing and the stance slid 6-9 cm)
    reach_hand_rot_w: float = 0.3        # REACH IK weight on the grasp wrist orientation (soft: the can is round)
    reach_com_w: float = 20.0            # REACH IK weight on the CoM (reach_com)
    reach_com_back: float = 0.03         # m, REACH CoM target this far behind the support centre
    reach_floor_force: float = 5.0       # N, fingertips pressing on the floor -> stop pushing down ...
    reach_backoff: float = 0.01          # ... and raise the aim by this much
    reach_iters: int = 8
    crouch_pelvis_z: float = 0.42        # IK pelvis target; SONIC settles ~8-10 cm higher
    reach_gain: float = 0.7
    reach_yaw_w: tuple = (10.0, 0.05, 0.5)  # REACH posture weights: hip yaw, waist yaw (at turn_yaw), right wrist yaw
                                            # (waist yaw free: holding the turn while reaching to the floor slid the feet)
    grasp_close_time: float = 0.6
    grasp_pocket_tol: float = 0.045      # m, weld engages with the box this close to the pinch pocket: the
                                         # closing index/middle fingers shove the box ~2 cm towards the thumb
                                         # (and turn it ~10 deg) before the thumb holds it
    grasp_squeeze: float = 0.15          # rad, finger command past the measured angle once the box is held
                                         # (0 = keep closing to RIGHT_HAND_CLOSED)
    lift_height: float = 0.09            # LIFT: raise the grasp point 8-10 cm ...
    lift_pullback: float = 0.0           # ... and pull it towards the robot (floor can: 0.07 dragged it into the
                                         # right knee in 11/11 episodes; pedestal can: -0.04, forward, slid the feet) ...
    lift_out: float = 0.08               # ... and this far out to the right, away from the right knee (floor can:
                                         # 0.05, 0.08 cleared the knee but the robot fell in 4/11; pedestal can:
                                         # 0.05 left 0.5-3 cm to the knee)
    lift_from_reference: bool = True    # offset applied to the reference hand (keeps the tracking compensation)
    lift_hold_pelvis: bool = True        # LIFT IK keeps the pelvis height and the CoM over the feet (arm-only lift)
    raise_torso: float = 0.10            # RAISE_TORSO (after LIFT): pelvis this much higher ...
    raise_hand: float = 0.14             # ... and the held box this much higher, so TURN_BACK swings the box
                                         # over the knees (still crouched it hit the legs in 8/8 episodes)
    raise_out: float = 0.03              # ... and this much further out to the right (clear of the knee)
    raise_time: float = 1.0
    regrasp: bool = False                # scripted failed first grasp: reach off target, close WITHOUT the grasp
                                         # assist, raise the hand -> the box slips out of the fingers; open, back
                                         # above the box, reach + grasp again (then the rest of the task)
    slip_aim_offset: tuple = (0.0, 0.0, 0.0)    # first attempt: grasp point this far off (world)
    slip_close_frac: float = 0.35        # first attempt closes only this far (a loose, light grip): closing fully
                                         # without the assist shoved the box 3-6 cm / tipped it up to 50 deg
    slip_lift: float = 0.05              # SLIP: hand raised this much with the fingers closed
    slip_max_tilt_deg: float = 15.0      # the can must still stand on the floor after the slip ...
    slip_max_shift: float = 0.03         # ... within this distance of where it stood
    carry_box_pos: tuple = (0.18, -0.27, 0.67)   # box center: feet frame (x fwd, y left), world z;
                                                 # 7-12 cm above the 0.50-0.55 m tabletop; y -0.22 left the
                                                 # can 1-2 cm off the right thigh and it rubbed it in STAND_UP
    w_hold_rot: float = 3.0              # IK weight holding the grasp wrist orientation (moderate)
    hold_actual_wrist: bool = True       # from STAND_UP on, hold the MEASURED wrist orientation at GRASP
                                         # instead of the reference one, which carries SONIC's crouch tracking
                                         # error (waist pitch untracked, wrist pitch ~0 for a 0.45 rad command):
                                         # holding the reference drove wrist pitch to its limit once upright
                                         # and tilted the box ~20-30 deg. With the box level at GRASP
                                         # (grasp_squeeze) the measured orientation = a level box.
    wrist_lock_w: float = 0.0            # carry/place IK: posture weight keeping right wrist pitch/yaw at their
                                         # measured GRASP values (0 = off; 8 made placing fail, 3/30)
    upright_pelvis_z: float = 0.72
    upright_tilt_deg: float = 10.0       # STABILIZE_UPRIGHT: pelvis tilt gate
    stable_lin_vel: float = 0.06
    stable_ang_vel: float = 0.35
    stable_time: float = 0.5
    stabilize_timeout: float = 4.0
    settle_hand_vel: float = 0.03        # m/s (filtered): "still" gate between sub-actions (hand, held box) ...
    settle_tau: float = 0.2              # s, low-pass time constant of the settle velocities ...
    settle_time: float = 0.4             # ... held this long ...
    settle_timeout: float = 3.0          # ... within this time, else the episode is rejected
    walk_max_time: float = 9.0
    carry_upright_box: bool = False      # False: hold the wrist orientation reached at GRASP (baseline)
    walk_mode: int = 1                   # planner mode (1 slowWalk, 2 walk); slowWalk keeps the 2 steps
                                         # short (~0.14 m) so the feet stop before the pedestal
    place_standoff: float = 0.35         # WALK goal: pelvis this far (x, heading frame) short of the tray center
                                         # (counting steps instead stopped the walk anywhere in 0.06-0.21 m:
                                         # far trays were over-reached, the box hit the tray, one robot fell)
    walk_stop_lead: float = 0.03         # switch the planner to idle this much before the goal (closing step)
    walk_min_travel: float = 0.08
    walk_max_travel: float = 0.45        # hard cap (the table stands ~0.6 m away: the squatting robot leans far)
    max_walk_liftoffs: int = 10          # a few slowWalk steps + a closing half-step
    walk_vel: float = -1.0               # planner target velocity (m/s), -1 = mode default
    place_above: float = 0.05
    approach_time: float = 1.4           # APPROACH_PLACE: one move, carry pose -> above the placement point
    place_max_corrections: int = 2       # small position (+ optional tilt) corrections after the approach
    place_corr_xy_tol: float = 0.025     # m, correct only above this xy error (the tray is forgiving)
    place_corr_step: float = 0.03        # m, max per correction
    tilt_fix_min_deg: float = 3.0        # tilt correction only when 3 deg < tilt < 10 deg ...
    tilt_fix_max_deg: float = 10.0       # ... larger tilts are placed as they are (no wrist servo)
    tilt_fix_max_count: int = 0          # at most this many tilt corrections (0: off, the box may tip in the tray)
    lower_step: float = 0.008            # LOWER_BOX: m per 0.25 s
    lower_xy_gain: float = 0.5           # xy re-centering while lowering
    lower_max_steps: int = 25
    release_height: float = 0.015        # LOWER_BOX: stop at tray contact or once the box is this close to
                                         # standing on the tray floor (it is then dropped the rest of the way)
    tray_release_margin: float = 0.03    # m, box center this far inside the tray walls to release (0.075 m
                                         # inner half -> within 4.5 cm of the tray center per axis)
    place_tilt_tol_deg: float = 60.0     # box tilt when it reaches the tray (60: effectively off)
    seat_depth: float = 0.008            # LOWER_BOX: extra press once the tray supports the box
    release_vz_tol: float = 0.02         # RELEASE gate: |box vz| (m/s) over the tray, held for ...
    release_gate_time: float = 0.2       # ... this long; at the timeout the box is released anyway
    release_gate_timeout: float = 1.5    # as long as it is over the tray
    reject_hand_box_after_retract: bool = False  # fingers still touching the box after RETRACT (logged only)
    require_upright: bool = False        # final check: box standing upright (< final_upright_deg) in the
    final_upright_deg: float = 10.0      # tray; off for now: only "the box is in the tray" is required
    release_open_frac: float = 0.0       # RELEASE: fingers open to this fraction of "closed" (0.5 still
                                         # pinches the 5 cm box by friction and the retract lifts it off) ...
    retract_dist: float = 0.13           # RETRACT: out along the finger axis (the opened, straight fingers
                                         # reach ~5 cm further than the curled ones) ...
    retract_back: float = 0.15           # ... then this far back towards the body ...
    retract_up: float = 0.08             # ... and this much higher (the swing to the neutral arm passes over
                                         # the box instead of sweeping through it)
    feet_slip_tol: float = 0.04          # m, foot travel before the walk that is always accepted
    feet_slip_tol_contact: float = 0.06  # m, accepted only while both feet stay in floor contact
                                         # (no lift-off) and the feet midpoint does not translate
    body_shift_tol: float = 0.04         # m, feet-midpoint translation before the walk
    liftoff_min_ticks: int = 5           # a foot off the floor for >= 0.1 s counts as a lift-off
    reject_prewalk_liftoff: bool = True  # any foot lift-off before the walk rejects the episode
    reject_leg_can_contact: bool = True  # a knee/shin touching the can before the grasp (it tips it over)
    reject_held_can_body_contact: bool = True  # the held can touching the robot anywhere but the hand (the
                                               # policy would learn to rub it along the legs)
    slip_tol: float = 0.025              # m, box drift relative to the wrist while grasped
    table_force_tol: float = 120.0       # N, robot<->table contact force
    home_joint_tol: float = 0.12         # rad, mean |q - default|
    render: bool = True
    debug_render: bool = True
    max_episode_time: float = 60.0         # regrasp episodes run ~8 s longer


@dataclass
class EpisodeResult:
    success: bool
    fail_reason: str | None
    frames: dict                          # column -> list (30 Hz)
    events: dict                          # event -> frame index
    ego_frames: list
    debug_frames: list
    info: dict = field(default_factory=dict)


class Episode:
    def __init__(self, params: SceneParams, policy: SonicPolicy, planner: KinematicPlanner,
                 cfg: Config | None = None, renderer_cache: dict | None = None):
        self.p, self.pol, self.pl = params, policy, planner
        self.cfg = cfg or Config()
        self.m = build_model(params)
        self.d = mujoco.MjData(self.m)
        self.ix = Idx(self.m)
        self.ik = WholeBodyIK(self.m, self.ix)
        self.power = self.cfg.grasp_style == "power"
        if self.power:
            self.ik.offset = POWER_OFFSET.copy()
            self.ik.offset[0] = self.cfg.power_offset_x
        self.grasp_below_rim = self.cfg.power_below_rim if self.power else GRASP_BELOW_RIM
        self.rng = np.random.default_rng(params.seed + 7)
        self.ref = ReferenceBuffer(self.ix.body_qadr)
        self.tick = 0
        self.phase = "HOME"
        self.hand_target = HAND_OPEN.copy()
        self.token = np.zeros(64, dtype=np.float32)
        self.next_token_t = 0.0
        self.next_rec_t = 0.0
        self.recording = False
        self.rec_t0 = 0.0
        self.cols: dict[str, list] = {}
        self.events: dict[str, int] = {}
        self.ego_frames, self.debug_frames = [], []
        self.grasped = False
        self.box_in_wrist = None      # box pose in wrist frame at weld time
        self.max_table_force = 0.0
        self.box_floor_touch_while_grasped = False
        self.max_slip = 0.0
        self.foot_ref = None
        self.max_foot_travel_prewalk = 0.0
        self.contacts_hist = []
        self.walk_liftoffs = []
        self.reach_log = []
        self.place_tilt_log = []
        self.prewalk_liftoffs = 0
        self.offfloor_ticks = [0, 0]
        self.R_grasp_world = None
        self.diag = {"prewalk": {}, "leg_can_clear": {}, "slip_max": {}, "robot_table_ticks": {},
                     "checkpoints": {}, "place": {}, "walk": {}, "hand_box_after_release": {},
                     "hand_table_clear": {}, "box_leg_clear": {}, "box_body_ticks": {},
                     "hand_floor_ticks": {}}
        self.renderer = None
        if self.cfg.render:
            self.renderer = mujoco.Renderer(self.m, IMG_H, IMG_W)

    # ------------------------------------------------------------------ sim core
    @property
    def t(self):
        return self.d.time

    def _step_tick(self):
        """One 50 Hz SONIC tick + 4 physics substeps (+ 30 Hz token latch / recording)."""
        d, ix = self.d, self.ix
        q = d.qpos[ix.body_qadr].copy()
        self.pol.log_state(q, d.qvel[ix.body_vadr].copy(), d.qpos[3:7].copy(), d.qvel[3:6].copy())
        if self.t + 1e-9 >= self.next_token_t:
            fr = min(self.tick, len(self.ref) - 1)
            self.token = self.pol.encode(self.ref.frames, fr, d.qpos[3:7].copy())
            self.next_token_t += 1.0 / self.cfg.token_rate_hz
        q_des = self.pol.decode(self.token)
        self.q_des = q_des
        for _ in range(SUBSTEPS):
            tau = pd_torque(q_des, d.qpos[ix.body_qadr], d.qvel[ix.body_vadr], self.pol.kp, self.pol.kd)
            d.ctrl[ix.body_act] = np.clip(tau, ix.ctrl_range[ix.body_act, 0], ix.ctrl_range[ix.body_act, 1])
            for act, qa, va, tgt in ((ix.rhand_act, ix.rhand_qadr, ix.rhand_vadr, self.hand_target),
                                     (ix.lhand_act, ix.lhand_qadr, ix.lhand_vadr, HAND_OPEN)):
                tq = HAND_KP * (tgt - d.qpos[qa]) - HAND_KD * d.qvel[va]
                d.ctrl[act] = np.clip(tq, ix.ctrl_range[act, 0], ix.ctrl_range[act, 1])
            mujoco.mj_step(self.m, d)
            if self.recording and self.t + 1e-9 >= self.next_rec_t:
                self._record()
                self.next_rec_t += 1.0 / RECORD_FPS
        self.tick += 1
        self._monitor()

    def run_until_ref_end(self, extra: int = 0):
        while self.tick < len(self.ref) + extra:
            self._step_tick()

    def run_ticks(self, n: int):
        for _ in range(n):
            self._step_tick()

    # ------------------------------------------------------------------ sensing
    def _contacts(self):
        d, ix = self.d, self.ix
        out = {"thumb_box": False, "finger_box": False, "hand_box": False, "box_floor": False,
               "box_table": False, "box_tray": False, "feet": [False, False],
               "robot_table_force": 0.0, "hand_floor_force": 0.0, "leg_box": False, "box_body": False,
               "robot_pedestal": False}
        furniture = ix.table_geoms | ix.tray_geoms
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g = (c.geom1, c.geom2)
            if ix.box_geom in g:
                o = g[1] if g[0] == ix.box_geom else g[0]
                if o in ix.rthumb_geoms: out["thumb_box"] = True
                if o in ix.rfinger_geoms: out["finger_box"] = True
                if o in ix.rhand_geoms: out["hand_box"] = True
                elif o in ix.robot_geoms:
                    out["box_body"] = True   # can on the robot anywhere but the hand
                    if o in ix.leg_geoms: out["leg_box"] = True
                if o == ix.floor_geom or o == ix.pedestal_geom: out["box_floor"] = True   # its start support
                if o in ix.table_geoms: out["box_table"] = True
                if o in ix.tray_geoms: out["box_tray"] = True
            if ix.pedestal_geom in g:
                o = g[1] if g[0] == ix.pedestal_geom else g[0]
                if o in ix.robot_geoms and o not in ix.rhand_geoms:
                    out["robot_pedestal"] = True
            if ix.floor_geom in g:
                o = g[1] if g[0] == ix.floor_geom else g[0]
                b = self.m.geom_bodyid[o]
                if b == ix.foot_bodies[0]: out["feet"][0] = True
                if b == ix.foot_bodies[1]: out["feet"][1] = True
                if o in ix.rhand_geoms:
                    mujoco.mj_contactForce(self.m, d, i, f6)
                    out["hand_floor_force"] += abs(f6[0])
            if (g[0] in furniture and g[1] in ix.robot_geoms) or (g[1] in furniture and g[0] in ix.robot_geoms):
                mujoco.mj_contactForce(self.m, d, i, f6)
                out["robot_table_force"] += abs(f6[0])
        out["box_support"] = out["box_table"] or out["box_tray"]
        return out

    def box_pose(self):
        a = self.ix.box_qadr
        return self.d.qpos[a:a + 3].copy(), self.d.qpos[a + 3:a + 7].copy()

    def wrist_pose(self):
        b = self.d.body(self.ix.rpalm)
        return b.xpos.copy(), b.xmat.reshape(3, 3).copy()

    def grasp_point(self):
        p, R = self.wrist_pose()
        return p + R @ self.ik.offset

    def _grasp_slip(self) -> float:
        """Box drift (m) relative to the wrist since the grasp assist engaged."""
        wp, wR = self.wrist_pose()
        return float(np.linalg.norm(wR.T @ (self.box_pose()[0] - wp) - self.box_in_wrist))

    def feet_now(self):
        return {b: (self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()) for b in self.ix.foot_bodies}

    def is_stable(self):
        d = self.d
        return (np.linalg.norm(d.qvel[:3]) < self.cfg.stable_lin_vel
                and np.linalg.norm(d.qvel[3:6]) < self.cfg.stable_ang_vel)

    def torso_tilt(self):
        z = self.d.xmat[self.ix.torso].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))

    def pelvis_tilt(self):
        z = self.d.xmat[self.ix.pelvis].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))

    def box_tilt(self):
        up = quat_rotate(self.box_pose()[1], np.array([0, 0, 1.0]))
        return float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))

    def box_place_xy_err(self):
        return float(np.linalg.norm(self.box_pose()[0][:2] - np.asarray(self.p.place_xy)))

    def box_in_tray(self, margin: float = 0.0, resting: bool = False) -> bool:
        """Box center inside the tray walls (by `margin`); `resting`: also low, i.e. standing or lying on
        the tray floor (not propped up on a wall)."""
        p, tb = self.p, self.d.body(self.ix.tray_body)
        local = tb.xmat.reshape(3, 3).T @ (self.box_pose()[0] - tb.xpos)
        inside = max(abs(local[0]), abs(local[1])) < p.tray_inner_half - margin
        return bool(inside and (not resting or local[2] < p.tray_floor_t + p.box_half[2] + 0.01))

    def _geoms_dist(self, geoms, other, distmax: float = 0.25):
        ft = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.m, self.d, g, other, distmax, ft) for g in geoms)

    def _leg_can_dist(self, geoms):
        return self._geoms_dist(geoms, self.ix.box_geom)

    # ------------------------------------------------------------------ diagnostics
    def _pstat(self, group, key, v, how):
        """Per-phase statistic: diag[group][phase][key] aggregated with max / min / sum."""
        slot = self.diag[group].setdefault(self.phase, {})
        if key not in slot:
            slot[key] = v
        elif how == "max":
            slot[key] = max(slot[key], v)
        elif how == "min":
            slot[key] = min(slot[key], v)
        else:
            slot[key] += v

    def _checkpoint(self, name):
        """Box tilt / wrist orientation / grasp slip / placement error at a named moment."""
        cp = {"t": self.t, "phase": self.phase, "box_tilt_deg": self.box_tilt(),
              "box_place_xy_err": self.box_place_xy_err(), "box_support": self.last_contacts["box_support"],
              "box_in_tray": self.box_in_tray()}
        wp, wR = self.wrist_pose()
        if self.R_grasp_world is not None:
            # actual wrist orientation vs the one at GRASP, with the stance yaw change removed
            dyaw = self._feet_frame(self.feet_now())[1] - self.grasp_feet_yaw
            cp["wrist_drift_deg"] = _angle_deg(rot_z(dyaw) @ self.R_grasp_world, wR)
            # actual wrist orientation vs the commanded (reference) one
            R_cmd_ref = self.ik.fk(self._seed_q()).xmat[self.ix.rpalm].reshape(3, 3).copy()
            cp["wrist_track_deg"] = _angle_deg(rot_z(-self.yaw_to_ref()) @ R_cmd_ref, wR)
        if self.grasped:
            bp, _ = self.box_pose()
            cp["grasp_slip"] = float(np.linalg.norm(wR.T @ (bp - wp) - self.box_in_wrist))
        self.diag["checkpoints"][name] = cp

    # ------------------------------------------------------------------ monitoring / events
    def _event(self, name):
        if name not in self.events:
            self.events[name] = max(0, len(self.cols.get("frame_index", [])) - 1)

    def _monitor(self):
        c = self._contacts()
        self.last_contacts = c
        d = self.d
        # fall
        if d.qpos[2] < 0.28 or (self.phase not in ("CROUCH", "TURN_TORSO", "REACH", "GRASP", "SLIP", "REGRASP_REACH", "LIFT",
                                                   "RAISE_TORSO", "TURN_BACK")
                                and self.torso_tilt() > 45):
            raise EpisodeFailed("fall_or_balance_loss")
        # feet must stay put until the walk. Small sliding/pivoting while SONIC balances is fine as
        # long as both feet keep floor contact and the stance does not translate; a lift-off (a
        # step) or a real translation rejects.
        if self.foot_ref is not None and PH[self.phase] <= PH["STABILIZE_UPRIGHT"]:
            cfg, fb = self.cfg, self.ix.foot_bodies
            travel = max(np.linalg.norm(d.xpos[b][:2] - self.foot_ref[b][:2]) for b in fb)
            mid = 0.5 * (d.xpos[fb[0]][:2] + d.xpos[fb[1]][:2])
            mid_ref = 0.5 * (self.foot_ref[fb[0]][:2] + self.foot_ref[fb[1]][:2])
            shift = float(np.linalg.norm(mid - mid_ref))
            for s in (0, 1):
                self.offfloor_ticks[s] = 0 if c["feet"][s] else self.offfloor_ticks[s] + 1
                if self.offfloor_ticks[s] == cfg.liftoff_min_ticks:
                    self.prewalk_liftoffs += 1
                    self._pstat("prewalk", "liftoffs", 1, "sum")
            self.max_foot_travel_prewalk = max(self.max_foot_travel_prewalk, travel)
            self._pstat("prewalk", "max_foot_travel", travel, "max")
            self._pstat("prewalk", "max_mid_shift", shift, "max")
            self._pstat("prewalk", "ticks", 1, "sum")
            self._pstat("prewalk", "both_contact_ticks", int(all(c["feet"])), "sum")
            if self.prewalk_liftoffs and cfg.reject_prewalk_liftoff:
                raise EpisodeFailed("foot_liftoff_before_walk")
            if travel > cfg.feet_slip_tol_contact:
                raise EpisodeFailed(f"feet_moved_before_walk(travel={travel:.3f})")
            if travel > cfg.feet_slip_tol and shift > cfg.body_shift_tol:
                raise EpisodeFailed(f"stance_translated_before_walk(travel={travel:.3f},shift={shift:.3f})")
        # legs vs the can on the floor: clearance log + contact rejection until it is grasped
        if not self.grasped and PH[self.phase] <= PH["GRASP"]:
            self._pstat("leg_can_clear", "right", self._leg_can_dist(self.ix.rleg_coll_geoms), "min")
            self._pstat("leg_can_clear", "left", self._leg_can_dist(self.ix.lleg_coll_geoms), "min")
            if c["leg_box"] and self.cfg.reject_leg_can_contact:
                raise EpisodeFailed("leg_can_collision")
        if PH[self.phase] <= PH["STABILIZE_UPRIGHT"]:   # right hand/forearm vs the tabletop while picking
            self._pstat("hand_table_clear", "min", self._geoms_dist(self.ix.rhand_coll_geoms, self.ix.table_top_geom), "min")
        if c["robot_pedestal"]:   # a knee / shin / forearm on the pedestal (the hand may brush it)
            raise EpisodeFailed("robot_pedestal_collision")
        # table collision
        self.max_table_force = max(self.max_table_force, c["robot_table_force"])
        if c["robot_table_force"] > 0:
            self._pstat("robot_table_ticks", "n", 1, "sum")
        if c["hand_box"] and PH[self.phase] >= PH["RETRACT"]:
            self._pstat("hand_box_after_release", "ticks", 1, "sum")
        if c["box_support"] and "box_first_table_phase" not in self.diag["place"]:
            self.diag["place"]["box_first_table_phase"] = self.phase
        if c["robot_table_force"] > self.cfg.table_force_tol:
            raise EpisodeFailed("strong_collision_with_table")
        if c["hand_floor_force"] > 0:
            self._pstat("hand_floor_ticks", "n", 1, "sum")
        if c["hand_floor_force"] > self.cfg.table_force_tol:
            raise EpisodeFailed("hand_pushed_into_floor")
        # held box vs the legs (knees/shins) and any other body part
        if self.grasped:
            self._pstat("box_leg_clear", "min", self._geoms_dist(self.ix.rleg_coll_geoms + self.ix.lleg_coll_geoms,
                                                                 self.ix.box_geom), "min")
        if c["box_body"]:
            self._pstat("box_body_ticks", "n", 1, "sum")
            if self.grasped and self.cfg.reject_held_can_body_contact:
                raise EpisodeFailed("held_can_touched_body")
        # grasp integrity
        if self.grasped:
            drift = self._grasp_slip()
            self.max_slip = max(self.max_slip, drift)
            self._pstat("slip_max", "m", drift, "max")
            # while the tray/table already supports the box, pressing on the soft weld is not a slip
            if drift > self.cfg.slip_tol and not c["box_support"]:
                raise EpisodeFailed("grasp_slip")
            if c["box_floor"] and PH[self.phase] >= PH["RAISE_TORSO"]:
                raise EpisodeFailed("can_touched_floor_after_lift")
        elif PH[self.phase] > PH["LIFT"] and PH[self.phase] < PH["RELEASE"]:
            raise EpisodeFailed("box_dropped")
        # walk step bookkeeping
        # (a lift-off = the foot has stayed off the floor for liftoff_min_ticks; shorter contact flickers
        # while SONIC shifts its weight are not steps)
        self.contacts_hist.append(tuple(c["feet"]))
        n = self.cfg.liftoff_min_ticks
        if self.phase.startswith("WALK") and len(self.contacts_hist) > n:
            for s in (0, 1):
                if self.contacts_hist[-n - 1][s] and not any(h[s] for h in self.contacts_hist[-n:]):
                    self.walk_liftoffs.append((self.tick, s))
        if self.t > self.cfg.max_episode_time:
            raise EpisodeFailed("timeout")

    # ------------------------------------------------------------------ recording
    def _record(self):
        d, ix = self.d, self.ix
        k = len(self.cols.get("frame_index", []))
        bp, bq = self.box_pose()
        c = getattr(self, "last_contacts", None) or self._contacts()
        tp = self.m.body(ix.table_body)
        tr = self.m.body(ix.tray_body)
        row = {
            "timestamp": np.float32(k / RECORD_FPS),
            "frame_index": k,
            "sim_time": np.float32(self.t),
            "phase": PH[self.phase],
            # Psi0 targets: [token 64 | left hand 7 | right hand 7]  (SIMPLE source layout)
            "action": np.r_[self.token, HAND_OPEN, self.hand_target].astype(np.float32),
            "action.sonic_token": self.token.astype(np.float32),
            "action.left_hand": HAND_OPEN.astype(np.float32),
            "action.right_hand": self.hand_target.astype(np.float32),
            # proprio (SIMPLE 43-D layout = MJCF qpos[7:50])
            "observation.state": d.qpos[7:50].astype(np.float32),
            "observation.qpos": d.qpos[:50].astype(np.float32),
            "observation.qvel": d.qvel[:49].astype(np.float32),
            "observation.root_pos": d.qpos[:3].astype(np.float32),
            "observation.root_quat": d.qpos[3:7].astype(np.float32),
            "observation.root_ang_vel": d.qvel[3:6].astype(np.float32),
            "observation.root_lin_vel": d.qvel[:3].astype(np.float32),
            "observation.projected_gravity": quat_rotate(np.array([d.qpos[3], -d.qpos[4], -d.qpos[5], -d.qpos[6]]),
                                                         np.array([0, 0, -1.0])).astype(np.float32),
            "observation.left_hand_state": d.qpos[ix.lhand_qadr].astype(np.float32),
            "observation.right_hand_state": d.qpos[ix.rhand_qadr].astype(np.float32),
            # raw whole-body targets (debug / latent regeneration / SONIC finetune)
            "wbc.joint_target_mj": self.q_des.astype(np.float32),
            "wbc.ref_joint_pos_isaac": self.ref.frames[min(self.tick, len(self.ref) - 1)].joint_pos.astype(np.float32),
            "wbc.ref_root_quat": self.ref.root_quat[min(self.tick, len(self.ref) - 1)].astype(np.float32),
            # objects
            "object.box_pose": np.r_[bp, bq].astype(np.float32),
            "object.table_pose": np.r_[tp.pos, tp.quat].astype(np.float32),
            "object.tray_pose": np.r_[tr.pos, tr.quat].astype(np.float32),
            "object.place_target": np.r_[self.p.place_xy, self.p.support_height].astype(np.float32),
            # contacts / grasp
            "contact.thumb_box": c["thumb_box"], "contact.finger_box": c["finger_box"],
            "contact.hand_box": c["hand_box"], "contact.box_floor": c["box_floor"],
            "contact.box_table": c["box_table"], "contact.box_tray": c["box_tray"],
            "contact.hand_floor": bool(c["hand_floor_force"] > 0),
            "contact.left_foot": c["feet"][0],
            "contact.right_foot": c["feet"][1],
            "grasp.assist_active": bool(self.grasped),
            # support: box touching the tray (or table) and resting on it (standing, or lower: tipped over)
            "state.table_support": bool(c["box_support"] and bp[2] - self.p.support_height - self.p.box_half[2] < 0.01),
            "state.box_in_tray": self.box_in_tray(resting=True),
            # errors (debug / filtering)
            "error.box_tilt_deg": np.float32(self.box_tilt()),
            "error.place_xy": np.float32(self.box_place_xy_err()),
            "error.grasp_slip": np.float32(self._grasp_slip() if self.grasped else 0.0),
        }
        for kk, v in row.items():
            self.cols.setdefault(kk, []).append(v)
        if self.renderer is not None:
            self.renderer.update_scene(d, camera="head_camera")
            self.ego_frames.append(self.renderer.render().copy())
            if self.cfg.debug_render:
                self.renderer.update_scene(d, camera="debug_cam")
                self.debug_frames.append(self.renderer.render().copy())

    # ------------------------------------------------------------------ expert helpers
    # The reference is a kinematic plan; the robot realizes it with some offset. All IK runs in
    # the reference's own frame: feet/CoM targets come from the reference FK, and world targets
    # (box, table) are mapped from the ACTUAL feet frame into the REFERENCE feet frame.
    def ref_feet(self, q=None):
        d = self.ik.fk(self._seed_q() if q is None else q)
        return {b: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for b in self.ix.foot_bodies}

    @staticmethod
    def _feet_frame(feet):
        (p0, R0), (p1, R1) = feet.values()
        c = 0.5 * (p0 + p1)
        fwd = R0[:, 0] + R1[:, 0]
        return c, float(np.arctan2(fwd[1], fwd[0]))

    def to_ref(self, p_world):
        ca, ya = self._feet_frame(self.feet_now())
        cr, yr = self._feet_frame(self.ref_feet())
        local = rot_z(-ya) @ (np.asarray(p_world) - ca)
        return cr + rot_z(yr) @ local

    def Rref(self, R_world):
        return rot_z(self.yaw_to_ref()) @ R_world

    def yaw_to_ref(self):
        return self._feet_frame(self.ref_feet())[1] - self._feet_frame(self.feet_now())[1]

    def _support_center(self, feet=None):
        f = feet or self.ref_feet()
        c, yaw = self._feet_frame(f)
        return c[:2] + np.array([np.cos(yaw), np.sin(yaw)]) * 0.03

    def _posture(self, legs=None, arm_w=0.3):
        nom = DEFAULT_ANGLES_MJ.copy()
        if legs is not None:
            h, k, a = legs
            nom[[0, 6]], nom[[3, 9]], nom[[4, 10]] = h, k, a
        w = np.full(29, arm_w)
        w[15:22] = 2.0            # left arm stays neutral
        w[[13, 14]] = 20.0        # SONIC does not track waist roll/pitch: keep them at 0
        w[[2, 8]] = 10.0          # hip yaw ~0: a yawing pelvis makes SONIC pivot the feet on the floor
        w[[1, 7, 5, 11]] = 3.0    # hip / ankle roll: symmetric stance
        if getattr(self, "q_wrist_grasp", None) is not None and self.cfg.wrist_lock_w > 0:
            nom[27:29] = self.q_wrist_grasp   # right wrist pitch/yaw: SONIC barely tracks them (4010)
            w[27:29] = self.cfg.wrist_lock_w
        return nom, w

    def _seed_q(self):
        """IK seed = current reference end (keeps the reference continuous)."""
        q = self.d.qpos.copy()
        qj, rp, rq = self.ref.last_qpos_parts
        q[self.ix.body_qadr], q[:3], q[3:7] = qj, rp, rq
        return q

    def _settle(self, name: str, timeout: float | None = None):
        """One thing at a time: hold the reference until the body is still (base linear / angular
        velocity), both feet are on the floor, the right hand is still and, while grasped, the box too;
        for settle_time. Velocities are low-pass filtered (tau settle_tau): SONIC holds a squat with a
        steady ~0.7 s bob of the pelvis (+-6 mm) that averages out, while a real drift does not.
        Logged in diag["settle"][name] (seconds waited)."""
        cfg, d, t0 = self.cfg, self.d, self.t
        timeout = cfg.settle_timeout if timeout is None else timeout
        a = np.exp(-CTRL_DT / cfg.settle_tau)
        vf = np.zeros(12)                 # filtered [base lin, base ang, hand lin, box lin]
        t_ok = None
        while True:
            self.ref.add_hold(CTRL_DT)
            self._step_tick()
            hv = np.zeros(6)
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.ix.rpalm, hv, 0)
            b = self.ix.box_vadr
            vf = a * vf + (1 - a) * np.r_[d.qvel[:3], d.qvel[3:6], hv[3:], d.qvel[b:b + 3]]
            still = (np.linalg.norm(vf[:3]) < cfg.stable_lin_vel and np.linalg.norm(vf[3:6]) < cfg.stable_ang_vel
                     and all(self.last_contacts["feet"]) and np.linalg.norm(vf[6:9]) < cfg.settle_hand_vel
                     and (not self.grasped or np.linalg.norm(vf[9:]) < cfg.settle_hand_vel))
            if still:
                t_ok = t_ok if t_ok is not None else self.t
                if self.t - t_ok >= cfg.settle_time:
                    break
            else:
                t_ok = None
            if self.t - t0 > timeout:
                raise EpisodeFailed(f"not_stable_in_{self.phase}({name})")
        self.diag.setdefault("settle", {})[name] = round(self.t - t0, 3)

    def _hold_until_stable(self, cond, timeout, hold_time=None, name=None):
        hold_time = self.cfg.stable_time if hold_time is None else hold_time
        t_ok, t0 = None, self.t
        while True:
            self.ref.add_hold(CTRL_DT)
            self._step_tick()
            if cond():
                t_ok = t_ok if t_ok is not None else self.t
                if self.t - t_ok >= hold_time:
                    return
            else:
                t_ok = None
            if self.t - t0 > timeout:
                raise EpisodeFailed(f"not_stable_in_{self.phase}" + (f"({name})" if name else ""))

    # ------------------------------------------------------------------ the episode
    def run(self) -> EpisodeResult:
        t_wall = time.time()
        try:
            self._run()
            ok, reason = True, None
        except EpisodeFailed as e:
            ok, reason = False, str(e)
        info = dict(wall_time=time.time() - t_wall, sim_time=self.t,
                    max_foot_travel_prewalk=self.max_foot_travel_prewalk, max_slip=self.max_slip,
                    max_table_force=self.max_table_force, walk_liftoffs=len(self.walk_liftoffs),
                    n_frames=len(self.cols.get("frame_index", [])),
                    fail_phase=None if ok else self.phase, regrasp=bool(self.cfg.regrasp),
                    diag=_jsonable(self.diag))
        return EpisodeResult(ok, reason, self.cols, self.events, self.ego_frames, self.debug_frames, info)

    def _run(self):
        m, d, ix, p, cfg = self.m, self.d, self.ix, self.p, self.cfg
        # ---- init: HOME pose with small joint noise, settle (not recorded)
        d.qpos[2] = 0.793
        d.qpos[3:7] = [np.cos(p.init_yaw / 2), 0, 0, np.sin(p.init_yaw / 2)]
        d.qpos[ix.body_qadr] = DEFAULT_ANGLES_MJ + self.rng.normal(0, p.init_joint_noise, 29)
        mujoco.mj_forward(m, d)
        q_home = d.qpos.copy(); q_home[ix.body_qadr] = DEFAULT_ANGLES_MJ; q_home[2] = 0.762
        self.ref.start(q_home)
        self.ref.add_hold(1.5)
        self.run_until_ref_end()
        if d.qpos[2] < 0.72:
            raise EpisodeFailed("home_settle_failed")
        self.foot_ref = {b: v[0] for b, v in self.feet_now().items()}
        self.home_feet_yaw = yaw_of(d.qpos[3:7])

        # ---- HOME: start recording once the default pose is fully settled
        self.recording, self.next_rec_t = True, self.t
        self.phase = "HOME"
        self._hold_until_stable(lambda: self.is_stable() and all(self.last_contacts["feet"]), cfg.stabilize_timeout)
        self.ref.add_hold(0.3); self.run_until_ref_end()

        box_yaw = yaw_of(self.box_pose()[1])

        # ---- CROUCH: SONIC-native squat from the kinematic planner (mode 4, target pelvis height),
        # heading locked to the HOME heading (no yaw)
        self.phase = "CROUCH"
        self._squat(cfg.squat_half_height)     # gradually: half way down, balance ...
        self._settle("crouch_half")
        q_squat = self._squat(cfg.squat_height + self.rng.uniform(-0.02, 0.02))   # ... then the full squat
        self._settle("crouch")                 # balanced in the squat before the arm moves

        # ---- TURN_TORSO: waist yaw to the right towards the box (legs/pelvis stay as squatted), balance
        self.phase = "TURN_TORSO"
        q_turn = q_squat.copy()
        q_turn[ix.body_qadr[12]] = -cfg.turn_yaw     # absolute: the planner's squat already yaws the waist ~+0.13
        self.ref.add_segment(q_turn, cfg.turn_time); self.run_until_ref_end()
        self._settle("turn")
        self._checkpoint("TURN_TORSO")

        # ---- REACH: right arm onto the box by shoulder/elbow + torso pitch; hip yaw, waist yaw and
        # wrist yaw stay near the turned squat values (the turn did the twisting, the arm translates)
        self.phase = "REACH"
        nom = q_turn[ix.body_qadr].copy()
        pw = np.full(29, 0.05)    # soft prior towards the planner's squat (hard weights stall the DLS)
        w_hip_yaw, w_waist_yaw, w_wrist_yaw = cfg.reach_yaw_w
        pw[[2, 8]] = w_hip_yaw    # hip yaw ~ squat value (no pelvis pivot)
        pw[[13, 14]] = cfg.reach_waist_rp_w   # waist roll/pitch near the squat values
        pw[12] = w_waist_yaw      # waist yaw ~ squat value: the box is in front, no twisting
        pw[15:22] = 2.0           # left arm keeps its squat pose
        pw[28] = w_wrist_yaw      # right wrist yaw
        com_xy = None
        if cfg.reach_com:   # support centre, shifted back by reach_com_back along the heading (the arm reaches forward)
            yaw_f = self._feet_frame(self.ref_feet())[1]
            com_xy = self._support_center() - cfg.reach_com_back * np.array([np.cos(yaw_f), np.sin(yaw_f)])
        # the pelvis keeps its squat heading: without this the reach IK yawed it ~15-20 deg back to the left
        # (the arm reaching right), undoing the torso turn
        pelvis_yaw = yaw_of(q_turn[3:7])
        # the pelvis stays at the squat height: reaching to the floor, an unconstrained IK dropped it to ~0.25 m,
        # far below what SONIC squats to (~0.47 m), and the robot stepped / lost balance
        pelvis_z = float(q_turn[2]) if cfg.reach_pelvis_w > 0 else None
        self._reach_ctx = dict(nom=nom, pw=pw, com_xy=com_xy, pelvis_yaw=pelvis_yaw, pelvis_z=pelvis_z)

        grasp_c, R_hand = self._grasp_target()
        q_from = q_turn
        if cfg.regrasp:
            # first attempt off target, closes without the grasp assist and slips; then reach again
            self._reach(grasp_c + np.asarray(cfg.slip_aim_offset), R_hand, q_from, diag_key="reach_slip")
            self._settle("reach_slip")
            self._slip_grasp()
            # arm back to the turned squat, balance, then reach again from scratch (re-reaching straight from
            # the extended arm made the stance creep 4-5 cm)
            self.phase = "REGRASP_REACH"
            self.ref.add_segment(q_turn, 1.2); self.run_until_ref_end()
            self._settle("slip_reset")
            grasp_c, R_hand = self._grasp_target()
        self.q_reach = self._reach(grasp_c, R_hand, q_from)
        self._settle("reach")                  # hand and body still before the fingers close

        # ---- GRASP: close the fingers; the weld engages at the first finger/thumb contact once the
        # hand is >= 30 % closed and the can sits in the pinch pocket (before the fingers can shove
        # it over: the Dex3 model alone cannot hold it).
        self.phase = "GRASP"
        self._close_grasp(assist=True)
        if not self.grasped:
            c = self.last_contacts
            raise EpisodeFailed(f"grasp_failed(thumb={c['thumb_box']},finger={c['finger_box']})")
        # no settle here: while the welded can still stands on the floor the hand is anchored, and SONIC
        # pulling it towards the (servo-compensated, lower) reference makes the squat creep forward
        # until a foot steps. Lift first, settle once the box is free (after LIFT).
        self.ref.add_hold(0.3); self.run_until_ref_end()
        # the wrist orientation to hold from here on: the reference orientation the grasp was made
        # with (stored relative to the reference stance yaw, so it follows the walk heading)
        R_grasp_ref = self.ik.fk(self._seed_q()).xmat[ix.rpalm].reshape(3, 3).copy()
        self.R_hold_local = rot_z(-self._feet_frame(self.ref_feet())[1]) @ R_grasp_ref
        # measured wrist at GRASP (stance yaw removed) + its pitch/yaw joints, held from STAND_UP on
        self.R_hold_actual_local = rot_z(-self._feet_frame(self.feet_now())[1]) @ self.wrist_pose()[1]
        self.q_wrist_grasp = d.qpos[ix.body_qadr[27:29]].copy()
        self._checkpoint("GRASP")

        # ---- LIFT: grasp point 8-10 cm up and slightly back, still crouched, same wrist orientation
        # The offset is applied to the REFERENCE hand: the reach servo left a ~8 cm tracking
        # compensation in it, and a target built from the actual hand drops that compensation, so the
        # hand shot ~10 cm forward/up towards the table and the robot leaned onto its toes.
        self.phase = "LIFT"
        yaw_ref = self._feet_frame(self.ref_feet())[1] + self._seed_q()[ix.body_qadr[12]]   # the turned torso heading
        if cfg.lift_from_reference:
            gp_ref = self.ik.grasp_point(self.ik.fk(self._seed_q())).copy()
            lift_ref = gp_ref + rot_z(yaw_ref) @ np.array([-cfg.lift_pullback, -cfg.lift_out, cfg.lift_height])
        else:   # legacy: offset from the actual hand
            yaw = yaw_of(d.qpos[3:7])
            lift_ref = self.to_ref(self.grasp_point() + rot_z(yaw) @ np.array([-cfg.lift_pullback, -cfg.lift_out,
                                                                               cfg.lift_height]))
        q0 = self._seed_q()
        # lift_hold_pelvis: the pelvis stays (height + CoM over the feet) and only the arm lifts. Free, the IK moved the
        # reference pelvis 5-12 cm forward and 4-8 cm down; SONIC followed and the held can swung back and in onto the
        # right knee / thigh (pedestal can: 8/50 episodes)
        hold = cfg.lift_hold_pelvis
        q_l, _ = self.ik.solve(q0, IKTargets(feet=self.ref_feet(), com_xy=self._support_center() if hold else None,
                                             hand_pos=lift_ref, hand_R=self._hold_R(), posture=nom, posture_w=pw,
                                             w_feet=100, w_com=10, w_hand_rot=cfg.w_hold_rot, pelvis_yaw=pelvis_yaw,
                                             pelvis_z=float(q0[2]) if hold else None, w_pelvis=20))
        self.ref.add_segment(q_l, 0.9)
        while self.tick < len(self.ref):
            self._step_tick()
            if self.box_pose()[0][2] - p.rest_z > 0.03:
                self._event("object_lifted")
        for _ in range(int(1.0 / CTRL_DT)):   # an arm-only lift (pelvis held) can take a little longer
            if "object_lifted" in self.events:
                break
            self.ref.add_hold(CTRL_DT); self._step_tick()
            if self.box_pose()[0][2] - p.rest_z > 0.03:
                self._event("object_lifted")
        if "object_lifted" not in self.events:
            raise EpisodeFailed("lift_failed")
        self._settle("lift")                   # box lifted and still before raising the torso
        self._checkpoint("LIFT")

        # ---- RAISE_TORSO: still turned, the pelvis rises raise_torso and the box raise_hand, balance; the
        # box then clears the knees when the torso turns back
        self.phase = "RAISE_TORSO"
        q0 = self._seed_q()
        dref = self.ik.fk(q0)
        pz, gp_ref = float(dref.xpos[ix.pelvis][2]), self.ik.grasp_point(dref).copy()
        q_r, _ = self.ik.solve(q0, IKTargets(feet=self.ref_feet(), com_xy=self._support_center(),
                                            hand_pos=gp_ref + rot_z(yaw_ref) @ np.array([0, -cfg.raise_out, cfg.raise_hand]),
                                            hand_R=self._hold_R(),
                                            pelvis_z=pz + cfg.raise_torso, pelvis_yaw=pelvis_yaw,
                                            posture=q0[ix.body_qadr], posture_w=pw, w_feet=100, w_com=10,
                                            w_pelvis=10, w_hand_rot=cfg.w_hold_rot), iters=400)
        self.ref.add_segment(q_r, cfg.raise_time); self.run_until_ref_end()
        self._settle("raise_torso")
        self._checkpoint("RAISE_TORSO")

        # ---- TURN_BACK: waist yaw back to the squat value, arm joints unchanged (the held box swings back
        # in front with the torso), balance. The wrist orientation held from here on is measured after it.
        self.phase = "TURN_BACK"
        q_back = self._seed_q()
        q_back[ix.body_qadr[12]] = q_squat[ix.body_qadr[12]]
        self.ref.add_segment(q_back, cfg.turn_back_time); self.run_until_ref_end()
        self._settle("turn_back")
        self.R_hold_actual_local = rot_z(-self._feet_frame(self.feet_now())[1]) @ self.wrist_pose()[1]
        R_back_ref = self.ik.fk(self._seed_q()).xmat[ix.rpalm].reshape(3, 3).copy()
        self.R_hold_local = rot_z(-self._feet_frame(self.ref_feet())[1]) @ R_back_ref
        self.q_wrist_grasp = d.qpos[ix.body_qadr[27:29]].copy()
        self._checkpoint("TURN_BACK")

        # ---- STAND_UP: legs/torso extend to standing, the arm brings the box to the carry point by
        # shoulder/elbow only; the wrist keeps the grasp orientation (no upright-box rotation)
        self.phase = "STAND_UP"
        if cfg.carry_upright_box:   # legacy option, off for the baseline
            R_up = self.Rref(self.hand_R_for_upright_box(box_yaw))
            self.R_hold_local = rot_z(-self._feet_frame(self.ref_feet())[1]) @ R_up
        elif cfg.hold_actual_wrist:
            self.R_hold_local = self.R_hold_actual_local
        q_carry = self._carry_pose()
        self.ref.add_segment(q_carry, 2.2 + self.rng.uniform(-0.2, 0.2))
        self.run_until_ref_end()
        self._checkpoint("STAND_UP")

        # ---- STABILIZE_UPRIGHT: gate before walking
        self.phase = "STABILIZE_UPRIGHT"
        self._hold_until_stable(lambda: d.qpos[2] > cfg.upright_pelvis_z and self.pelvis_tilt() < cfg.upright_tilt_deg
                                and self.is_stable() and all(self.last_contacts["feet"]) and self.grasped
                                and not self.last_contacts["box_floor"],
                                cfg.stabilize_timeout)
        self._settle("upright")                # hand and box still too, before the first step
        self._event("upright_reached")
        self._checkpoint("STABILIZE_UPRIGHT")

        # ---- WALK: two slowWalk steps with the SONIC planner, upper body frozen in the carry pose
        self._walk(q_carry)

        # ---- STABILIZE_AT_TABLE: planner idle, wait until fully settled
        self.phase = "STABILIZE_AT_TABLE"
        self._settle("at_table", cfg.stabilize_timeout)   # walk fully stopped before the arm moves
        self._event("table_reached")
        self._checkpoint("AFTER_WALK")

        # ---- ADJUST_ARM: carry pose re-solved from where the feet actually are now (the walk shifts the
        # stance and SONIC lets the frozen arm sag), balance; the box is then carried over the tray from here
        self.phase = "ADJUST_ARM"
        self.ref.add_segment(self._carry_pose(), cfg.adjust_time); self.run_until_ref_end()
        self._settle("adjust_arm")
        self._checkpoint("ADJUST_ARM")

        # ---- APPROACH_PLACE / LOWER_BOX
        self.phase = "APPROACH_PLACE"
        self._approach_place()
        self._settle("approach")               # box still above the placement point before lowering
        self.phase = "LOWER_BOX"
        self._lower_box()

        # ---- RELEASE: open only once the table carries the box (contact, no vertical motion, COM
        # projection well inside the tabletop)
        self.phase = "RELEASE"
        self._release_gate()
        self._checkpoint("PRE_RELEASE")
        # (a) relax the squeeze: the fingers are commanded "fully closed" but blocked by the box, so
        #     they press into it; dropping the weld with that stored force flicks the box over (or off
        #     the edge). Commanding the measured finger angles removes it.
        # (b) drop the weld; the box settles on the table (flat) with the hand around it.
        # (c) open only part way: fully straightened, the middle finger swings onto the box's top edge
        #     and any later hand motion drags the box.
        self.hand_target = self.d.qpos[ix.rhand_qadr].copy()
        self.ref.add_hold(0.2); self.run_until_ref_end()
        self._deactivate_weld()
        self.ref.add_hold(0.3); self.run_until_ref_end()
        self._open_hand(cfg.release_open_frac, 0.6)
        self.ref.add_hold(0.3); self.run_until_ref_end()

        # ---- RETRACT, straight-line moves with the open hand, wrist orientation kept:
        # (1) out along the finger axis (the way the hand went in; with the wrist ~20 deg off vertical
        #     a world-up move drags the fingers across the box);
        # (2) back towards the body and up (retract_up), so that
        # (3) the swing to the neutral arm joints passes over the box instead of through it.
        self.phase = "RETRACT"
        # translate only: target = the reference wrist orientation/joints as they are now (re-solving for
        # R_place_ref re-balances the arm on every move and turns the open hand into the released box)
        q_now = self._seed_q()
        self.R_place_ref = self.ik.fk(q_now).xmat[ix.rpalm].reshape(3, 3).copy()
        self.q_wrist_grasp = q_now[ix.body_qadr[27:29]].copy()
        wR = self.wrist_pose()[1]
        out_dir = -wR[:, 1] if self.power else -wR[:, 0]   # power: away from the palm side; pinch: up the fingers
        if self.power:
            out_dir = out_dir + np.array([0, 0, 0.3 - out_dir[2]])
        elif out_dir[2] < 0.7:
            out_dir = out_dir + np.array([0, 0, 0.7 - out_dir[2]])
        out_dir /= np.linalg.norm(out_dir)
        q_r = self._place_ik(self.grasp_point() + cfg.retract_dist * out_dir, self.R_place_ref, self._seed_q(), iters=200)
        self.ref.add_segment(q_r, 0.8); self.ref.add_hold(0.15); self.run_until_ref_end()
        self._open_hand(0.0, 0.4)
        self._hand_box_check("retract_out")
        yaw = self._feet_frame(self.feet_now())[1]
        back = self.grasp_point() + rot_z(yaw) @ np.array([-cfg.retract_back, 0.0, cfg.retract_up])
        q_b = self._place_ik(back, self.R_place_ref, q_r, iters=200)
        self.ref.add_segment(q_b, 0.8); self.run_until_ref_end()
        q_n = self._seed_q()
        q_n[ix.body_qadr[22:29]] = DEFAULT_ANGLES_MJ[22:29]
        self.ref.add_segment(q_n, 1.0); self.ref.add_hold(0.2); self.run_until_ref_end()
        self._hand_box_check("retract_neutral")
        if not self.box_in_tray(resting=True):
            raise EpisodeFailed("box_not_in_tray_after_retract")
        self._event("release_success")

        # ---- RETURN_HOME: default whole-body pose where the robot stands now (not the start xy)
        self.phase = "RETURN_HOME"
        q_h = self._seed_q(); q_h[ix.body_qadr] = DEFAULT_ANGLES_MJ
        q_h[2] = 0.762
        self.ref.add_segment(q_h, 1.8)
        self.run_until_ref_end()
        self._hold_until_stable(lambda: self.is_stable() and np.mean(np.abs(d.qpos[ix.body_qadr] - DEFAULT_ANGLES_MJ)) < cfg.home_joint_tol,
                                cfg.stabilize_timeout)
        self._event("home_reached")

        # ---- final checks: the box stands or lies in the tray, not moving; upright only if required
        bp, bq = self.box_pose()
        bv = np.linalg.norm(d.qvel[ix.box_vadr:ix.box_vadr + 3])
        up = quat_rotate(bq, np.array([0, 0, 1.0]))[2]
        in_tray = self.box_in_tray(resting=True)
        upright = up > np.cos(np.radians(cfg.final_upright_deg))
        self._checkpoint("FINAL")
        self.diag["place"]["final_upright"] = bool(upright)
        if not (in_tray and bv < 0.03 and (upright or not cfg.require_upright)):
            raise EpisodeFailed(f"box_not_in_tray(in_tray={in_tray},z={bp[2] - p.support_height:+.3f},"
                                f"v={bv:.3f},up={up:.3f})")
        self.phase = "SUCCESS"
        self.ref.add_hold(0.2); self.run_until_ref_end()

    # ------------------------------------------------------------------ sub-routines
    def _grasp_target(self):
        """Grasp point (pocket centered 3 cm below the can rim) and wrist orientation (the can's body yaw: the
        can is round, its yaw only selects the grasp heading)."""
        bp, bq = self.box_pose()
        point = bp + np.array([0, 0, self.p.box_half[2] - self.grasp_below_rim])
        if not self.power:
            return point, rot_z(yaw_of(bq)) @ R_GRASP
        # power: wrist z (the finger curl axis) along the can, fingers at power_heading from the stance heading,
        # pitched down by power_tilt within the palm plane; the palm side (+y) faces the can
        yaw = self._feet_frame(self.feet_now())[1] + np.radians(self.cfg.power_heading_deg)
        x = np.array([np.cos(yaw), np.sin(yaw), 0.0]); z = np.array([0, 0, 1.0])
        R = np.column_stack([x, np.cross(z, x), z])
        t = np.radians(self.cfg.power_tilt_deg)
        return point, R @ np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])

    def _reach_ik(self, q0, hand_ref, R_ref):
        """REACH IK (reference-frame targets), turned-squat posture prior, pelvis heading held."""
        c = self._reach_ctx
        return self.ik.solve(q0, IKTargets(feet=self.ref_feet(), com_xy=c["com_xy"], hand_pos=hand_ref, hand_R=R_ref,
                                           posture=c["nom"], posture_w=c["pw"], w_feet=100, w_com=self.cfg.reach_com_w,
                                           w_hand_rot=self.cfg.reach_hand_rot_w, pelvis_yaw=c["pelvis_yaw"], pelvis_z=c["pelvis_z"],
                                           w_pelvis=self.cfg.reach_pelvis_w))[0]

    def _reach_power(self, target, R_hand, q_from, diag_key="reach"):
        """Side approach for the whole-hand grasp. Aiming straight at the can, the position servo overshot by
        5-9 cm sideways and swept it off the pedestal, so: (1) servo the grasp point to a pre-grasp spot
        pregrasp_side behind the can (palm side) and pregrasp_up higher, clear of it; (2) slide in along the palm
        normal in a few steps, keeping the tracking compensation found in (1); (3) at most two small corrections
        across the palm normal (height / along the fingers), never pushing further into the can."""
        cfg = self.cfg
        reach_ik = lambda q0, tgt: self._reach_ik(q0, self.to_ref(tgt), self.Rref(R_hand))
        n = R_hand[:, 1] / np.linalg.norm(R_hand[:, 1])          # palm normal, towards the can
        pre = target - cfg.pregrasp_side * n + np.array([0, 0, cfg.pregrasp_up])
        log = self._reach_dbg = []
        self.hand_target = np.asarray(cfg.power_preshape, dtype=float)
        q_k = reach_ik(q_from, pre + np.array([0, 0, cfg.pregrasp_high]))
        self.ref.add_segment(q_k, cfg.pregrasp_time); self.run_until_ref_end()
        q_k = reach_ik(q_k, pre)
        self.ref.add_segment(q_k, 0.8); self.run_until_ref_end()
        aim = pre + np.array(cfg.reach_comp)
        for it in range(cfg.pregrasp_iters):
            q_k = reach_ik(q_k, aim)
            self.ref.add_segment(q_k, 0.8 if it == 0 else 0.5); self.ref.add_hold(cfg.pregrasp_hold)
            self.run_until_ref_end()
            err = pre - self.grasp_point()
            ik_res = float(np.linalg.norm(self.ik.grasp_point(self.ik.fk(q_k)) - self.to_ref(aim)))
            log.append((it, aim.round(3).tolist(), err.round(3).tolist(), float(self.d.qpos[2]), round(ik_res, 3)))
            if np.linalg.norm(err) < cfg.pregrasp_tol:
                break
            aim = pre + np.clip(aim + cfg.pregrasp_gain * err - pre, -cfg.reach_aim_clip, cfg.reach_aim_clip)
        else:
            self.diag[diag_key] = {"errs": [np.round(e[2], 3).tolist() for e in log], "ik_res": [e[4] for e in log],
                                   "aims": [e[1] for e in log]}
            raise EpisodeFailed(f"pregrasp_failed(err={np.linalg.norm(err):.3f})")
        # slide in, steered by where the can actually is in the hand (pocket error, wrist frame: x along the fingers,
        # y the palm normal towards the can, z the curl axis), so a nudged can is followed, not aimed past.
        # Across the palm the error is corrected with the servo gain; towards the can the hand only advances, by
        # at most slide_step, and not at all once the can is at the pocket depth or touches the palm. Aiming at a
        # fixed target instead stopped 4-7 cm short (open loop) or shoved the can off the pedestal (closed loop).
        if cfg.slide_mode == "open":
            comp = aim - pre
            for k in range(1, 5):
                q_k = reach_ik(q_k, pre + (target - pre) * k / 4 + comp)
                self.ref.add_segment(q_k, 0.4); self.ref.add_hold(0.15); self.run_until_ref_end()
            self.ref.add_hold(0.2); self.run_until_ref_end()
            for it in range(3):
                e = target - self.grasp_point()
                e_n = float(e @ n)
                log.append(("side", comp.round(3).tolist(), e.round(3).tolist(), float(self.d.qpos[2]), 0.0))
                push = 0.0 if self.last_contacts["hand_box"] else float(np.clip(cfg.reach_gain * e_n, 0.0, 0.02))
                if np.linalg.norm(e - e_n * n) < 0.015 and push < 0.005:
                    break
                comp = comp + cfg.reach_gain * (e - e_n * n) + push * n
                q_k = reach_ik(q_k, target + comp)
                self.ref.add_segment(q_k, 0.4); self.ref.add_hold(0.2); self.run_until_ref_end()
        for it in range(cfg.slide_iters if cfg.slide_mode == "pocket" else 0):
            P = self._pocket_err()
            log.append(("slide", aim.round(3).tolist(), P.round(3).tolist(), float(self.d.qpos[2]), 0.0))
            touching = bool(self.last_contacts["hand_box"])
            adv = 0.0 if (touching or P[1] < cfg.slide_depth_tol) else float(np.clip(P[1], 0.0, cfg.slide_step))
            if adv == 0.0 and np.hypot(P[0], P[2]) < 0.012:
                break
            wR = self.wrist_pose()[1]
            aim = aim + wR @ np.array([cfg.reach_gain * P[0], adv, cfg.reach_gain * P[2]])
            q_k = reach_ik(q_k, aim)
            self.ref.add_segment(q_k, 0.4); self.ref.add_hold(0.2); self.run_until_ref_end()
        err = target - self.grasp_point()
        self.reach_log += log
        self.diag[diag_key] = {"iters": len(log), "err": float(np.linalg.norm(err)),
                               "errs": [np.round(e[2], 3).tolist() for e in log], "ik_res": [e[4] for e in log],
                               "can_shift": float(np.linalg.norm(self.box_pose()[0][:2] - np.asarray(self.p.box_xy))),
                               "contacts": {k: bool(self.last_contacts[k]) for k in ("thumb_box", "finger_box", "hand_box")},
                               "pocket_err_wrist": self._pocket_err().round(3).tolist()}
        if not self._box_in_pocket(cfg.reach_tol):
            raise EpisodeFailed(f"{diag_key}_failed(err={np.linalg.norm(err):.3f})")
        return q_k

    def _reach(self, target, R_hand, q_from, diag_key="reach"):
        if self.power:
            return self._reach_power(target, R_hand, q_from, diag_key)
        """Pre-grasp above the box (slow swing), then down onto `target` (grasp point, world) with a bounded
        position servo. The aim may not be integrated down into the box: with a fingertip on the box top the
        unbounded servo drove the reference hand several cm into box + pedestal, so the feet slid and the
        wrist sprang back (box tilt) at LIFT. Returns the last IK keyframe."""
        cfg, d, ix = self.cfg, self.d, self.ix
        reach_ik = lambda q0, tgt: self._reach_ik(q0, self.to_ref(tgt), self.Rref(R_hand))
        log = self._reach_dbg = []   # kept on the episode: readable after a failed reach too
        if self.power:   # beside the can on the palm side, a little higher; then in sideways
            pre = target - cfg.pregrasp_side * R_hand[:, 1] + np.array([0, 0, cfg.pregrasp_up])
        else:
            pre = target + np.array([0, 0, cfg.pregrasp_height])
        q_k = reach_ik(q_from, pre)
        self.ref.add_segment(q_k, cfg.pregrasp_time); self.run_until_ref_end()
        base = target + np.array(cfg.reach_comp)
        aim = base.copy()
        for it in range(cfg.reach_iters):
            q_k = reach_ik(q_k, aim)
            self.ref.add_segment(q_k, 0.8 if it == 0 else 0.5)
            self.ref.add_hold(0.25)
            self.run_until_ref_end()
            err = target - self.grasp_point()
            floor_f = self._mean_floor_force(0.2)
            log.append((it, aim.round(3).tolist(), err.round(3).tolist(), float(self.d.qpos[2]), floor_f))
            if np.linalg.norm(err) < cfg.reach_tol:
                break
            step = cfg.reach_gain * err
            if floor_f > cfg.reach_floor_force:
                # the fingertips already rest on the floor next to the can: pushing further down loads the
                # arm with body weight and the robot lurches forward (feet slide). Back off instead; the
                # grasp is judged by the pocket.
                if np.linalg.norm(err[:2]) < cfg.reach_tol and self._box_in_pocket():
                    aim[2] += cfg.reach_backoff
                    q_k = reach_ik(q_k, aim)
                    self.ref.add_segment(q_k, 0.4); self.ref.add_hold(0.2); self.run_until_ref_end()
                    break
                step[2] = cfg.reach_backoff
            aim = aim + step
            d_aim = np.clip(aim - base, -cfg.reach_aim_clip, cfg.reach_aim_clip)
            d_aim[2] = np.clip(d_aim[2], -cfg.reach_aim_below, cfg.reach_aim_clip)
            aim = base + d_aim
        else:
            self.diag[diag_key] = {"errs": [np.round(e[2], 3).tolist() for e in log]}
            raise EpisodeFailed(f"{diag_key}_failed(err={np.linalg.norm(err):.3f})")
        self.reach_log += log
        self.diag[diag_key] = {"iters": len(log), "err": float(np.linalg.norm(err)),
                               "errs": [np.round(e[2], 3).tolist() for e in log],
                               "floor_force": [round(e[4], 1) for e in log],
                               "waist_yaw": float(d.qpos[ix.body_qadr[12]]),
                               "hip_yaw_r": float(d.qpos[ix.body_qadr[8]]),
                               "wrist_yaw_r": float(d.qpos[ix.body_qadr[28]]),
                               "pelvis_yaw_deg": float(np.degrees(yaw_of(d.qpos[3:7]) - self.home_feet_yaw))}
        return q_k

    def _close_grasp(self, assist: bool, frac: float = 1.0) -> bool:
        """Ramp the right hand closed. With `assist` the weld engages at the first finger/thumb contact once
        the hand is >= 30 % closed and the box sits in the pinch pocket. Returns whether the hand touched
        the box."""
        cfg, ix = self.cfg, self.ix
        n = int(cfg.grasp_close_time / CTRL_DT)
        touched = False
        start = np.asarray(cfg.power_preshape, dtype=float) if self.power else HAND_OPEN
        for k in range(n + int(0.3 / CTRL_DT)):
            s = min(1.0, (k + 1) / n)
            self.hand_target = start + s * (frac * RIGHT_HAND_CLOSED - start)
            if self.grasped and cfg.grasp_squeeze > 0:
                # once the grasp assist holds the box, each finger closes only up to grasp_squeeze past its
                # measured angle: squeezing on to "fully closed" (1.4 N.m per joint on a 0.15 kg box)
                # twisted the box in the soft weld by ~8 deg before it even left the pedestal
                q = self.d.qpos[ix.rhand_qadr]
                lim = np.abs(q) + cfg.grasp_squeeze
                self.hand_target = np.sign(self.hand_target) * np.minimum(np.abs(self.hand_target), lim)
            self.ref.add_hold(CTRL_DT); self._step_tick()
            c = self.last_contacts
            touched |= bool(c["hand_box"])
            if (assist and not self.grasped and s >= 0.3 and (c["thumb_box"] or c["finger_box"])
                    and self._box_in_pocket(cfg.grasp_pocket_tol)):
                self._activate_weld()
                self._event("grasp_success")
        return touched

    def _slip_grasp(self):
        """Regrasp episodes, failed first attempt: the fingers close on the box's top edge without the grasp
        assist (the Dex3 model alone cannot hold the box), the hand rises slip_lift and the box slips out;
        the hand opens and goes back above the can. The can must still stand on the floor."""
        cfg, p = self.cfg, self.p
        b0 = self.box_pose()[0]
        self.phase = "GRASP"
        touched = self._close_grasp(assist=False, frac=cfg.slip_close_frac)
        self.phase = "SLIP"
        q0 = self._seed_q()
        dref = self.ik.fk(q0)
        gp_ref, R_ref = self.ik.grasp_point(dref).copy(), dref.xmat[self.ix.rpalm].reshape(3, 3).copy()
        q_s = self._reach_ik(q0, gp_ref + np.array([0, 0, cfg.slip_lift]), R_ref)
        self.ref.add_segment(q_s, 0.6)
        rise = 0.0
        while self.tick < len(self.ref):
            self._step_tick()
            rise = max(rise, self.box_pose()[0][2] - b0[2])
        self.ref.add_hold(0.2); self.run_until_ref_end()
        self._open_hand(0.0, 0.4)
        self.ref.add_hold(0.3); self.run_until_ref_end()
        bp = self.box_pose()[0]
        c = self.last_contacts
        rest = abs(bp[2] - p.rest_z) < 0.01
        shift, tilt = float(np.linalg.norm(bp[:2] - b0[:2])), self.box_tilt()
        self.diag["regrasp"] = {"touched": touched, "box_max_rise": float(rise), "box_shift": shift,
                                "box_tilt_deg": tilt, "on_floor": bool(c["box_floor"] and rest)}
        if not touched:
            raise EpisodeFailed("slip_attempt_missed_box")
        if not (c["box_floor"] and rest and tilt < cfg.slip_max_tilt_deg and shift < cfg.slip_max_shift):
            raise EpisodeFailed(f"box_disturbed_by_slip(shift={shift:.3f},tilt={tilt:.1f},rest={rest})")

    def _activate_weld(self):
        m, d, ix = self.m, self.d, self.ix
        wp, wR = self.wrist_pose()
        bp, bq = self.box_pose()
        rel_p = wR.T @ (bp - wp)
        wq = np.zeros(4); mujoco.mju_mat2Quat(wq, wR.flatten())
        rel_q = np.zeros(4); mujoco.mju_mulQuat(rel_q, np.array([wq[0], -wq[1], -wq[2], -wq[3]]), bq)
        m.eq_data[ix.weld, 0:3] = 0.0            # anchor
        m.eq_data[ix.weld, 3:6] = rel_p
        m.eq_data[ix.weld, 6:10] = rel_q
        m.eq_data[ix.weld, 10] = 1.0             # torquescale
        d.eq_active[ix.weld] = 1
        self.grasped, self.box_in_wrist = True, rel_p
        R_rel = np.zeros(9); mujoco.mju_quat2Mat(R_rel, rel_q)
        self.R_box_in_wrist = R_rel.reshape(3, 3)
        self.R_grasp_world = wR.copy()
        self.grasp_feet_yaw = self._feet_frame(self.feet_now())[1]

    def _hand_box_check(self, name: str):
        """Fingers still on the released box: logged (diag hand_box_after_release), fatal only if configured."""
        if self.last_contacts["hand_box"]:
            self.diag["hand_box_after_release"][name] = True
            if self.cfg.reject_hand_box_after_retract:
                raise EpisodeFailed("hand_still_touching_box_after_retract")

    def _mean_floor_force(self, duration: float) -> float:
        """Right hand<->floor normal force, averaged over a short hold (the reference end is held)."""
        f = []
        for _ in range(max(1, int(duration / CTRL_DT))):
            self.ref.add_hold(CTRL_DT); self._step_tick()
            f.append(self.last_contacts["hand_floor_force"])
        return float(np.mean(f))

    def _pocket_err(self) -> np.ndarray:
        """Can grasp-height axis point minus the hand pocket, in the wrist frame."""
        wp, wR = self.wrist_pose()
        bp, bq = self.box_pose()
        top_pt = bp + quat_rotate(bq, np.array([0, 0, self.p.box_half[2] - self.grasp_below_rim]))
        return wR.T @ (top_pt - wp) - self.ik.offset

    def _box_in_pocket(self, tol: float = 0.03) -> bool:
        """Box grasp region (3 cm below its top) within `tol` of the hand's pinch pocket."""
        wp, wR = self.wrist_pose()
        bp, bq = self.box_pose()
        top_pt = bp + quat_rotate(bq, np.array([0, 0, self.p.box_half[2] - self.grasp_below_rim]))
        return bool(np.linalg.norm(wR.T @ (top_pt - wp) - self.ik.offset) < tol)

    def hand_R_for_upright_box(self, yaw: float) -> np.ndarray:
        """Wrist orientation that holds the (welded) box upright with the given yaw (legacy option)."""
        return rot_z(yaw) @ self.R_box_in_wrist.T

    def _hold_R(self) -> np.ndarray:
        """Grasp wrist orientation in the reference frame, following the reference stance yaw."""
        return rot_z(self._feet_frame(self.ref_feet())[1]) @ self.R_hold_local

    def _open_hand(self, frac: float, duration: float):
        """Ramp the right-hand command from its current value towards frac * closed (body holds still).
        Never closes a joint further than its current command (a finger blocked straight by the box
        must not be driven into it)."""
        start = self.hand_target.copy()
        goal = frac * RIGHT_HAND_CLOSED
        goal = np.where(np.abs(start) < np.abs(goal), start, goal)
        n = max(1, int(duration / CTRL_DT))
        for k in range(n):
            self.hand_target = start + (k + 1) / n * (goal - start)
            self.ref.add_hold(CTRL_DT); self._step_tick()

    def _deactivate_weld(self):
        self.d.eq_active[self.ix.weld] = 0
        self.grasped = False

    def _carry_pose(self):
        """Standing whole-body pose with the box center at carry_box_pos (feet frame), grasp wrist
        orientation held."""
        cx, cy, cz = self.cfg.carry_box_pos
        c, yaw = self._feet_frame(self.feet_now())
        box_t = np.r_[c[:2] + rot_z(yaw)[:2, :2] @ np.array([cx, cy]), cz]
        hand_t = box_t - (self.box_pose()[0] - self.grasp_point())   # the offset is fixed by the held orientation
        nom, pw = self._posture()
        pw[:12] = 2.0                                                  # standing legs
        q, info = self.ik.solve(self._seed_q(), IKTargets(feet=self.ref_feet(), com_xy=self._support_center(),
                                                         hand_pos=self.to_ref(hand_t), hand_R=self._hold_R(),
                                                         posture=nom, posture_w=pw, w_feet=100, w_com=20,
                                                         w_hand_rot=self.cfg.w_hold_rot, torso_upright=2.0), iters=400)
        self.diag["carry_ik"] = {k: float(v) for k, v in info.items()}
        return q

    def _squat(self, height):
        """Replan the planner's squat mode until the reference settles at `height`; returns that qpos."""
        yaw = self.home_feet_yaw
        for k in range(12):
            self.ref.truncate(self.tick + 3)
            ctx = context_from_reference(self.ref.q, self.ref.root_pos, self.ref.root_quat)
            out = self.pl.plan(ctx, MODE_SQUAT, direction=(0, 0, 0), target_heading=yaw, height=height,
                               seed=int(self.p.seed))
            J, P, Q = splice(out, ctx)
            if self.cfg.lock_arms_crouch:   # the planner swings the arms forward to balance: keep them at HOME
                J = [np.r_[j[:15], DEFAULT_ANGLES_MJ[15:29]] for j in J]
            self.ref.add_raw(J, P, Q)
            self.run_ticks(15)
            zs = [pp[2] for pp in self.ref.root_pos[-10:]]
            if abs(zs[-1] - height) < 0.015 and max(zs) - min(zs) < 0.005:
                break
        self.ref.truncate(len(self.ref))   # keep the settled tail as lookahead
        self.run_until_ref_end()
        q = self._seed_q()
        return q

    def _walk(self, q_carry):
        ix, d = self.ix, self.d
        upper = q_carry[ix.body_qadr][12:].copy()     # carry pose frozen: no wrist/arm servo while walking
        yaw = self.home_feet_yaw
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        mode = self.cfg.walk_mode
        self.phase = "WALK_STEP_1"
        self._event("walk_started")
        t0, touchdowns = self.t, 0
        x_start = self.d.qpos[:2].copy()
        tray_ahead = float(np.dot(np.asarray(self.p.tray_xy) - x_start, heading[:2]))
        goal = float(np.clip(tray_ahead - self.cfg.place_standoff, self.cfg.walk_min_travel, self.cfg.walk_max_travel))
        self.walk_liftoffs = []
        prev = tuple(self.last_contacts["feet"])
        off = [0, 0]              # ticks each foot has been off the floor
        idle_since = None
        while True:
            self.ref.truncate(self.tick + 3)
            ctx = context_from_reference(self.ref.q, self.ref.root_pos, self.ref.root_quat)
            out = self.pl.plan(ctx, mode, direction=heading if mode != MODE_IDLE else (0, 0, 0),
                               target_heading=yaw, seed=int(self.p.seed),
                               target_vel=self.cfg.walk_vel if mode != MODE_IDLE else -1.0)
            J, P, Q = splice(out, ctx)
            self.ref.add_raw([np.r_[j[:12], upper] for j in J], P, Q)
            for _ in range(10):          # replan every 0.2 s
                self._step_tick()
                cur = tuple(self.last_contacts["feet"])
                for s in (0, 1):
                    # a touchdown only ends a real swing: 20-40 ms contact flickers while SONIC shifts its
                    # weight were counted as steps and stopped the walk with ~0 travel
                    if not prev[s] and cur[s] and off[s] >= self.cfg.liftoff_min_ticks:
                        touchdowns += 1
                        if touchdowns == 1:
                            self.phase = "WALK_STEP_2"
                    off[s] = 0 if cur[s] else off[s] + 1
                prev = cur
            travel = float(np.dot(self.d.qpos[:2] - x_start, heading[:2]))
            if mode != MODE_IDLE and travel > goal - self.cfg.walk_stop_lead:
                mode, idle_since = MODE_IDLE, self.t      # near the goal -> ask the planner to stop
                travel_at_idle = travel
            if mode == MODE_IDLE and self.t - idle_since > 1.2 and all(prev) and self.is_stable():
                break
            if self.t - t0 > self.cfg.walk_max_time:
                raise EpisodeFailed("walk_timeout")
        self.walk_travelled = float(np.dot(self.d.qpos[:2] - x_start, heading[:2]))
        self.diag["walk"] = {"touchdowns": touchdowns, "liftoffs": len(self.walk_liftoffs),
                             "travel": self.walk_travelled, "time": self.t - t0, "goal": goal,
                             "travel_at_idle": travel_at_idle}
        if len(self.walk_liftoffs) > self.cfg.max_walk_liftoffs:
            raise EpisodeFailed(f"too_many_steps({len(self.walk_liftoffs)})")
        # freeze the reference at the current pose (drop the planner's remaining future)
        self.ref.truncate(self.tick + 1)

    # ---- place
    def _place_ik(self, aim_world, R_ref, q0, iters=300):
        """Standing IK for the grasp point: shoulder/elbow + a little torso lean, legs near default."""
        nom, pw = self._posture()
        pw[:12] = 2.0             # legs stay (bent knees would hit the table)
        pw[[0, 6]] = 0.5          # hip pitch: the torso may lean a little towards the table
        return self.ik.solve(q0, IKTargets(feet=self.ref_feet(), com_xy=self._support_center(),
                                           hand_pos=self.to_ref(aim_world), hand_R=R_ref, posture=nom, posture_w=pw,
                                           w_feet=100, w_com=20, w_hand_rot=self.cfg.w_hold_rot,
                                           torso_upright=0.5), iters=iters)[0]

    def _place_top(self):
        p = self.p
        return np.r_[p.place_xy, p.support_height + p.box_half[2]]   # box center standing on the tray floor

    def _approach_place(self):
        """One move from the carry pose to `place_above` over the placement point (the box bottom
        stays above the tabletop the whole way), then at most `place_max_corrections` small
        corrections. The only orientation change allowed is one tilt fix below tilt_fix_max_deg."""
        cfg = self.cfg
        target = self._place_top() + np.array([0, 0, cfg.place_above])
        R_ref = self._hold_R()
        aim0 = aim = target - (self.box_pose()[0] - self.grasp_point())
        q_k = self._place_ik(aim, R_ref, self._seed_q())
        self.ref.add_segment(q_k, cfg.approach_time); self.ref.add_hold(0.3); self.run_until_ref_end()
        pl = self.diag["place"]
        pl["approach_xy_err"] = float(np.linalg.norm((target - self.box_pose()[0])[:2]))
        pl["approach_z_err"] = float((target - self.box_pose()[0])[2])
        pl["tilt_after_approach_deg"] = self.box_tilt()
        pl["corrections"], pl["tilt_fix_deg"] = 0, []
        for _ in range(cfg.place_max_corrections):
            bp, bq = self.box_pose()
            err = target - bp
            tilt = self.box_tilt()
            need_pos = np.linalg.norm(err[:2]) > cfg.place_corr_xy_tol or abs(err[2]) > 0.02
            need_tilt = (len(pl["tilt_fix_deg"]) < cfg.tilt_fix_max_count
                         and cfg.tilt_fix_min_deg < tilt < cfg.tilt_fix_max_deg)
            if not (need_pos or need_tilt):
                break
            if need_tilt:
                # a small rotation about the horizontal axis that levels the box; the grasp point
                # target is recomputed so the box center stays over the target
                up = quat_rotate(bq, np.array([0, 0, 1.0]))
                axis = np.cross(up, [0, 0, 1.0]); axis /= np.linalg.norm(axis)
                qc = np.zeros(4); mujoco.mju_axisAngle2Quat(qc, axis, np.radians(tilt))
                Rc = np.zeros(9); mujoco.mju_quat2Mat(Rc, qc); Rc = Rc.reshape(3, 3)
                Ry = rot_z(self.yaw_to_ref())
                R_ref = Ry @ Rc @ Ry.T @ R_ref
                aim = aim0 = target - Rc @ (bp - self.grasp_point())
                pl["tilt_fix_deg"].append(tilt)
            if need_pos:
                step = 0.9 * err
                step *= min(1.0, cfg.place_corr_step / (np.linalg.norm(step) + 1e-9))
                aim = aim + step
                total = aim - aim0
                aim = aim0 + total * min(1.0, 0.08 / (np.linalg.norm(total) + 1e-9))   # <= 8 cm overall
            q_k = self._place_ik(aim, R_ref, q_k)
            self.ref.add_segment(q_k, 0.5); self.ref.add_hold(0.2); self.run_until_ref_end()
            pl["corrections"] += 1
        self.R_place_ref, self.place_aim = R_ref, aim
        self._checkpoint("APPROACH_END")

    def _lower_box(self):
        """Straight down in small steps with a gentle xy re-centering until the tray carries the box or the
        box is within release_height of the tray floor. The box only has to be well inside the tray
        (tray_release_margin); its tilt is not checked (place_tilt_tol_deg), it may tip over when released."""
        cfg = self.cfg
        top = self._place_top()
        aim, R_ref, q_k = self.place_aim.copy(), self.R_place_ref, self._seed_q()
        pl = self.diag["place"]
        for k in range(cfg.lower_max_steps):
            bp = self.box_pose()[0]
            e_xy = np.r_[top[:2] - bp[:2], 0.0]
            aim = aim + np.array([0, 0, -cfg.lower_step]) + cfg.lower_xy_gain * e_xy
            q_k = self._place_ik(aim, R_ref, q_k, iters=150)
            self.ref.add_segment(q_k, 0.25); self.run_until_ref_end()
            bp = self.box_pose()[0]
            supported = self.last_contacts["box_support"]
            if supported and "FIRST_TABLE_CONTACT" not in self.diag["checkpoints"]:
                self._checkpoint("FIRST_TABLE_CONTACT")
            if supported or bp[2] - top[2] < cfg.release_height:
                xy_err = float(np.linalg.norm(bp[:2] - top[:2]))
                tilt = self.box_tilt()
                pl.update(lower_steps=k + 1, supported_xy_err=xy_err, supported_tilt_deg=tilt,
                          supported_contact=bool(supported), release_z=float(bp[2] - top[2]))
                if not self.box_in_tray(cfg.tray_release_margin) or tilt > cfg.place_tilt_tol_deg:
                    raise EpisodeFailed(f"place_misaligned(xy_err={xy_err:.3f},tilt={tilt:.1f})")
                self._event("object_supported")
                if supported:
                    # seat the box: SONIC tracks the hand ~1.5 cm off the reference and the loop stops at the
                    # first corner contact, so the box otherwise hovers 1-4 mm up with the contact flickering
                    aim = aim + np.array([0, 0, -cfg.seat_depth])
                    q_k = self._place_ik(aim, R_ref, q_k, iters=150)
                    self.ref.add_segment(q_k, 0.3)
                self.ref.add_hold(0.3); self.run_until_ref_end()
                self.place_aim, self.q_place = aim, q_k
                return
        raise EpisodeFailed("place_not_supported")

    def _release_gate(self):
        """Open once the box is still (|vz|) over the inside of the tray; tray contact is not required. At the
        timeout the box is released anyway as long as it is over the tray."""
        cfg, d, ix = self.cfg, self.d, self.ix
        t0, t_ok = self.t, None
        while True:
            self.ref.add_hold(CTRL_DT); self._step_tick()
            vz = float(d.qvel[ix.box_vadr + 2])
            inside = self.box_in_tray(cfg.tray_release_margin)
            if inside and abs(vz) < cfg.release_vz_tol:
                t_ok = t_ok if t_ok is not None else self.t
                if self.t - t_ok >= cfg.release_gate_time:
                    break
            else:
                t_ok = None
            if self.t - t0 > cfg.release_gate_timeout:
                if not inside:
                    raise EpisodeFailed(f"release_gate_not_met(inside_tray={inside},vz={vz:+.3f})")
                self.diag["place"]["release_gate_timeout"] = True
                break
        self.diag["place"]["release_wait"] = self.t - t0
        self.diag["place"]["release_contact"] = bool(self.last_contacts["box_support"])
