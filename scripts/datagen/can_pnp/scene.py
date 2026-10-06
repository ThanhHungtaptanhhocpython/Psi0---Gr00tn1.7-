"""Scene construction + per-episode randomization for the can pick (from the floor) -> walk -> place task.

The grasped object is a slim can standing on the floor beside the robot's right foot. Internally it
keeps the "box" names of box_pnp (body "box", Idx.box_*, SceneParams.box_half/box_xy/box_yaw), so the
episode / writer / render code is shared: box_half = (radius, radius, half height).

The robot model is the one SONIC's own MuJoCo sim loop uses
(gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml, 29 body DoF + 2x7 Dex3),
so qpos[7:50] is exactly the 43-D SIMPLE state layout
[lleg6 rleg6 waist3 larm7 lhand7 rarm7 rhand7].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
G1_SCENE = REPO_ROOT / "third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

SIM_DT = 0.005          # SONIC sim loop SIMULATE_DT (200 Hz)
CTRL_DT = 0.02          # SONIC policy rate (50 Hz)
RECORD_FPS = 30         # Psi0 dataset rate
IMG_H, IMG_W = 480, 640


@dataclass
class SceneParams:
    """Everything randomized for one episode. Saved verbatim in the episode meta."""
    seed: int
    # can (cylinder) standing on the FLOOR, world frame, robot starts at origin facing +x. Slim can (5.3 cm
    # across, like the 5 cm box of box_pnp it fits the Dex3 thumb/finger pocket), grasped from the top, the
    # pocket 3 cm below the rim.
    can_radius: float = 0.0265
    can_half_height: float = 0.067
    box_mass: float = 0.15
    box_xy: tuple[float, float] = (0.20, -0.25)   # forward-right of the right foot (y ~ -0.13): beside the right
                                                  # knee in the squat, outside the right foot's walking lane
    box_yaw: float = -1.2                # grasp heading (the can is round: this only sets the wrist yaw it is
                                         # grasped with, see Episode._grasp_target)
    pedestal_height: float = 0.0         # 0: the can stands on the floor; > 0: on a static pedestal this high
    pedestal_half: tuple[float, float] = (0.06, 0.06)   # 12 x 12 cm top
    pedestal_offset: tuple[float, float] = (0.02, -0.02)  # pedestal center - can center: away from the legs
    # table: ~2 natural SONIC steps (~0.23 m of pelvis travel) + a placement standoff
    table_edge_x: float = 0.38          # near edge of the tabletop
    table_y: float = 0.09               # shifted left so its near-RIGHT corner (the tray) is in reach of
                                        # the right hand after the 2 steps
    table_height: float = 0.52           # top surface z (low: place with shoulder/elbow, not the wrist)
    table_half: tuple[float, float] = (0.25, 0.35)  # tabletop half depth (x), half width (y)
    # tray: static, open-top, sitting in the table's near-right corner; the task is "box into the tray"
    tray_inset: tuple[float, float] = (0.09, 0.09)  # tray center - near-right table corner (x into the
                                                    # table, y towards +y)
    tray_yaw: float = 0.0
    tray_inner_half: float = 0.075       # 15 x 15 cm inside (box 5 x 5 cm + a few cm of placement error)
    tray_wall_h: float = 0.025           # wall top above the tabletop (low: the fingers clear it)
    tray_wall_t: float = 0.008
    tray_floor_t: float = 0.005
    tray_rgb: tuple[float, float, float] = (0.20, 0.55, 0.35)
    # robot / rendering
    init_joint_noise: float = 0.0
    init_yaw: float = 0.0
    walk_distance: float = 0.15          # measured pelvis travel of 2 SONIC 'walk' steps (not a command)
    cam_fovy: float = 80.0               # deg; the MJCF default (45) misses the box at HOME
    cam_pitch_jitter: float = 0.0        # rad
    cam_yaw_jitter: float = 0.0
    light_dir: tuple[float, float, float] = (0.0, 0.0, -1.0)
    light_diffuse: float = 0.6
    floor_rgb: tuple[float, float, float] = (0.2, 0.3, 0.4)
    table_rgb: tuple[float, float, float] = (0.55, 0.42, 0.30)
    box_rgb: tuple[float, float, float] = (0.80, 0.12, 0.10)

    @property
    def box_half(self):
        """(radius, radius, half height): the box_pnp code reads the can's half extents from here."""
        return (self.can_radius, self.can_radius, self.can_half_height)

    @property
    def rest_z(self):
        """Can center z when standing on its start support (floor or pedestal)."""
        return self.pedestal_height + self.can_half_height

    @property
    def tray_xy(self):
        return (self.table_edge_x + self.tray_inset[0], self.table_y - self.table_half[1] + self.tray_inset[1])

    @property
    def place_xy(self):
        """Placement target = tray center."""
        return self.tray_xy

    @property
    def support_height(self):
        """z of the surface the box is put down on (the tray floor)."""
        return self.table_height + self.tray_floor_t

    def to_dict(self):
        d = asdict(self)
        d["place_xy"] = list(self.place_xy)
        d["box_half"] = list(self.box_half)
        d["tray_xy"] = list(self.tray_xy)
        return d


PEDESTAL_HEIGHT = 0.20   # the can stands on a pedestal this high (0: on the floor, the original can_pnp layout)
PEDESTAL_CAN_XY = (0.27, -0.30)   # nominal can position on the pedestal (+-1.5 cm)


def sample_params(seed: int, level: int = 0) -> SceneParams:
    """Small, controlled randomization. level 0 = baseline ranges from the task spec."""
    r = np.random.default_rng(seed)
    u = lambda lo, hi: float(r.uniform(lo, hi))
    s = 1.0 + (0.5 * level)  # widen ranges for higher levels
    return SceneParams(
        seed=seed,
        can_radius=0.0265,
        can_half_height=u(0.064, 0.070),  # 12.8-14 cm tall
        box_mass=u(0.13, 0.17),
        # small jitter beside the right foot; on the pedestal further out to the right: the squatting legs pass
        # it with >= 4.5 cm to spare and the right hand still wraps the can from behind (palm forward)
        box_xy=((PEDESTAL_CAN_XY[0] + u(-0.015, 0.015) * s, PEDESTAL_CAN_XY[1] + u(-0.015, 0.015) * s) if PEDESTAL_HEIGHT > 0 else
                (0.20 + u(-0.02, 0.02) * s, -0.25 + u(-0.02, 0.02) * s)),
        box_yaw=-1.2 + u(-0.1, 0.1) * s,
        # 2 SONIC steps move the pelvis ~0.15 m; the table edge then sits ~0.28 m ahead of it
        table_edge_x=u(0.62, 0.66),      # can_pnp: further than box_pnp (0.43-0.46): reaching down to the floor
                                         # the squatting torso pitches forward and hit the near table edge      # 2 slowWalk steps only move the pelvis ~0.1 m; with the low
                                         # (0.50-0.55 m) top, a closer edge is hit by the right hand while
                                         # it reaches/lifts the box (underside at 0.46-0.51 m)
        table_y=u(0.075, 0.105),         # tray center y = table_y - 0.35 + inset_y ~ -0.18 .. -0.14
        table_height=u(0.50, 0.55),      # below the carry height: the box is lowered by the arm, wrist fixed
        tray_inset=(u(0.093, 0.103), u(0.093, 0.105)),  # tray (outer half 8.3 cm, +-8 deg yaw) ~1-2 cm in
                                                        # from the near/right edges: as close to the robot as
                                                        # it fits (center 9.3-10.3 cm from the near edge)
        init_joint_noise=0.02 * s,
        init_yaw=u(-0.03, 0.03) * s,
        cam_pitch_jitter=u(-0.02, 0.02) * s,
        cam_yaw_jitter=u(-0.02, 0.02) * s,
        light_dir=tuple((np.array([u(-0.3, 0.3), u(-0.3, 0.3), -1.0])).tolist()),
        light_diffuse=u(0.45, 0.8),
        floor_rgb=tuple(np.clip(np.array([0.2, 0.3, 0.4]) + r.uniform(-0.05, 0.05, 3), 0, 1).tolist()),
        table_rgb=tuple(np.clip(np.array([0.55, 0.42, 0.30]) + r.uniform(-0.05, 0.05, 3), 0, 1).tolist()),
        box_rgb=tuple(np.clip(np.array([0.80, 0.12, 0.10]) + r.uniform(-0.05, 0.05, 3), 0, 1).tolist()),
        # drawn last so the draws above stay the same per seed
        tray_yaw=u(-0.14, 0.14),         # +-8 deg
        tray_rgb=tuple(np.clip(np.array([0.20, 0.55, 0.35]) + r.uniform(-0.05, 0.05, 3), 0, 1).tolist()),
        pedestal_height=(PEDESTAL_HEIGHT + u(-0.01, 0.01) * s) if PEDESTAL_HEIGHT > 0 else 0.0,
    )


def build_model(p: SceneParams) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(G1_SCENE))
    spec.option.timestep = SIM_DT
    spec.visual.global_.offwidth = max(IMG_W, 1280)
    spec.visual.global_.offheight = max(IMG_H, 720)
    wb = spec.worldbody

    # floor color / light
    for m in spec.materials:
        if m.name == "groundplane":
            m.rgba = [*p.floor_rgb, 1.0]
    for lt in wb.lights:
        lt.dir = list(p.light_dir)
        lt.diffuse = [p.light_diffuse] * 3

    # hide the scene's CoM debug marker (it would show up in the ego images)
    for site in wb.sites:
        if site.name == "com_marker":
            site.rgba = [0, 0, 0, 0]

    # optional low pedestal (static) the can stands on
    if p.pedestal_height > 0:
        ped = wb.add_body(name="pedestal", pos=[p.box_xy[0] + p.pedestal_offset[0], p.box_xy[1] + p.pedestal_offset[1], 0.0])
        ped.add_geom(name="pedestal_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[p.pedestal_half[0], p.pedestal_half[1], p.pedestal_height / 2],
                     pos=[0, 0, p.pedestal_height / 2], rgba=[0.35, 0.36, 0.40, 1.0], friction=[1.0, 0.01, 0.001])

    # can standing on the floor / pedestal (free body)
    box = wb.add_body(name="box", pos=[p.box_xy[0], p.box_xy[1], p.rest_z + 1e-3],
                      quat=[np.cos(p.box_yaw / 2), 0, 0, np.sin(p.box_yaw / 2)])
    box.add_freejoint(name="box_free")
    box.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[p.can_radius, p.can_half_height, 0],
                 mass=p.box_mass, rgba=[*p.box_rgb, 1.0], friction=[1.5, 0.01, 0.001],
                 condim=4, solref=[0.01, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    # silver lid: makes the can read as a can in the ego view
    box.add_geom(name="box_lid", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[p.can_radius * 0.92, 0.002, 0],
                 pos=[0, 0, p.can_half_height], rgba=[0.78, 0.78, 0.80, 1.0], contype=0, conaffinity=0, mass=0)

    # table: static
    hx, hy = p.table_half
    tz = p.table_height
    table = wb.add_body(name="table", pos=[p.table_edge_x + hx, p.table_y, 0.0])
    # thin (2 cm) top: with the low table the underside is what the right hand passes under while picking
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.01],
                   pos=[0, 0, tz - 0.01], rgba=[*p.table_rgb, 1.0], friction=[1.0, 0.01, 0.001])
    # near legs set 15 cm back from the near edge (kept from box_pnp)
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        table.add_geom(name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.02, 0.02, (tz - 0.02) / 2],
                       pos=[sx * (hx - (0.03 if sx > 0 else 0.15)), sy * (hy - 0.03), (tz - 0.02) / 2],
                       rgba=[*p.table_rgb, 1.0])

    # tray: static open box on the tabletop (floor + 4 low walls), rotated by tray_yaw about its center
    ih, wt, wh, ft = p.tray_inner_half, p.tray_wall_t, p.tray_wall_h, p.tray_floor_t
    oh = ih + wt
    tray = wb.add_body(name="tray", pos=[p.tray_xy[0], p.tray_xy[1], tz],
                       quat=[np.cos(p.tray_yaw / 2), 0, 0, np.sin(p.tray_yaw / 2)])
    tray_geom = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[*p.tray_rgb, 1.0], friction=[1.0, 0.01, 0.001])
    tray.add_geom(name="tray_floor", size=[oh, oh, ft / 2], pos=[0, 0, ft / 2], **tray_geom)
    for i, (sx, sy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        size = [wt / 2, oh, wh / 2] if sx else [oh, wt / 2, wh / 2]
        tray.add_geom(name=f"tray_wall{i}", size=size,
                      pos=[sx * (ih + wt / 2), sy * (ih + wt / 2), wh / 2], **tray_geom)

    # grasp assist: soft weld wrist<->box, inactive until a valid finger closure is detected
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name = "grasp_assist"
    eq.name1 = "right_wrist_yaw_link"
    eq.name2 = "box"
    eq.active = False
    eq.solref = [0.01, 1.0]

    # placement marker (visual only, no collision)
    wb.add_site(name="place_target", pos=[p.place_xy[0], p.place_xy[1], p.support_height + 0.001],
                size=[0.01, 0.01, 0.001], type=mujoco.mjtGeom.mjGEOM_CYLINDER, rgba=[0, 1, 0, 0.0])

    # third-person debug camera (NOT used as policy input)
    wb.add_camera(name="debug_cam", pos=[0.45, -1.9, 1.25], xyaxes=[0.97, 0.24, 0, -0.10, 0.40, 0.91])

    # ego camera jitter (head_camera lives in torso_link)
    for body in spec.bodies:
        for cam in body.cameras:
            if cam.name == "head_camera":
                e = np.array([0.0, -0.8 + p.cam_pitch_jitter, -1.57 + p.cam_yaw_jitter])
                cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
                cam.alt.euler = e.tolist()
                cam.fovy = p.cam_fovy
    return spec.compile()


# ---------------------------------------------------------------------------- indices
class Idx:
    """Name-based indices (the Dex3 joints sit between the arms in this MJCF)."""

    def __init__(self, m: mujoco.MjModel):
        jn = lambda n: m.joint(n)
        self.body_joint_names = [
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "waist_yaw", "waist_roll", "waist_pitch",
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
            "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
            "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
        ]
        hand = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
        self.body_qadr = np.array([jn(n + "_joint").qposadr[0] for n in self.body_joint_names])
        self.body_vadr = np.array([jn(n + "_joint").dofadr[0] for n in self.body_joint_names])
        self.lhand_qadr = np.array([jn(f"left_hand_{h}_joint").qposadr[0] for h in hand])
        self.rhand_qadr = np.array([jn(f"right_hand_{h}_joint").qposadr[0] for h in hand])
        self.lhand_vadr = np.array([jn(f"left_hand_{h}_joint").dofadr[0] for h in hand])
        self.rhand_vadr = np.array([jn(f"right_hand_{h}_joint").dofadr[0] for h in hand])
        act = lambda n: m.actuator(n).id
        self.body_act = np.array([act(n) for n in self.body_joint_names])
        self.lhand_act = np.array([act(f"left_hand_{h}") for h in hand])
        self.rhand_act = np.array([act(f"right_hand_{h}") for h in hand])
        self.box_qadr = m.joint("box_free").qposadr[0]
        self.box_vadr = m.joint("box_free").dofadr[0]
        self.box_geom = m.geom("box_geom").id
        self.floor_geom = m.geom("floor").id
        self.pedestal_geom = m.geom("pedestal_geom").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pedestal_geom") >= 0 else -1
        self.table_geoms = {m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("table")}
        self.tray_geoms = {m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("tray")}
        self.tray_body = m.body("tray").id
        rh_bodies = {m.body(n).id for n in ["right_wrist_yaw_link"] + [f"right_hand_{h}_link" for h in hand]}
        lh_bodies = {m.body(n).id for n in ["left_wrist_yaw_link"] + [f"left_hand_{h}_link" for h in hand]}
        self.rhand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in rh_bodies}
        self.lhand_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in lh_bodies}
        robot_root = m.body("pelvis").id
        self.robot_geoms = {g for g in range(m.ngeom)
                            if m.body_rootid[m.geom_bodyid[g]] == robot_root}
        self.foot_bodies = (m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id)
        leg_links = ("hip_yaw_link", "knee_link", "ankle_pitch_link", "ankle_roll_link")
        leg = lambda side: {m.body(f"{side}_{n}").id for n in leg_links}
        # collision geoms only (the visual meshes have contype 0)
        rarm = {m.body(n).id for n in ("right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link")}
        self.rhand_coll_geoms = [g for g in range(m.ngeom) if (g in self.rhand_geoms or m.geom_bodyid[g] in rarm)
                                 and m.geom_contype[g]]
        self.table_top_geom = m.geom("table_top").id
        self.rleg_coll_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in leg("right") and m.geom_contype[g]]
        self.lleg_coll_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in leg("left") and m.geom_contype[g]]
        self.leg_geoms = set(self.rleg_coll_geoms) | set(self.lleg_coll_geoms) | {
            g for g in range(m.ngeom) if m.geom_bodyid[g] in leg("right") | leg("left")}
        self.pelvis = m.body("pelvis").id
        self.torso = m.body("torso_link").id
        self.rpalm = m.body("right_wrist_yaw_link").id
        self.ctrl_range = m.actuator_ctrlrange.copy()
        self.box_body = m.body("box").id
        self.table_body = m.body("table").id
        self.weld = m.equality("grasp_assist").id
        thumb = {m.body(f"right_hand_thumb_{i}_link").id for i in range(3)}
        fingers = {m.body(f"right_hand_{f}_{i}_link").id for f in ("index", "middle") for i in range(2)}
        self.rthumb_geoms = {g for g in self.rhand_geoms if m.geom_bodyid[g] in thumb}
        self.rfinger_geoms = {g for g in self.rhand_geoms if m.geom_bodyid[g] in fingers}
