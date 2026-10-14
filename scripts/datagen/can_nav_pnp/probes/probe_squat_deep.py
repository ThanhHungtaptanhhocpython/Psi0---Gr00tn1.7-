"""How low does SONIC 1.1 actually squat? For each target pelvis height: (a) the SONIC planner's squat mode
(what generate.py uses), (b) a whole-body IK squat reference (feet fixed, CoM over the feet). Robot at READY,
no obstacle/pedestal in the way (pedestal moved away). Prints actual pelvis z / knee / hip pitch after settling."""
import os, sys, json, dataclasses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, mujoco
from PIL import Image, ImageDraw
from episode import Config, Episode, EpisodeFailed
from kinematics import IKTargets
from planner import KinematicPlanner
from scene import sample_params
from sonic_policy import SonicPolicy, DEFAULT_ANGLES_MJ
from generate import SONIC_DIR, PLANNER

OUT = sys.argv[1]
pol = SonicPolicy(SONIC_DIR, threads=4); pl = KinematicPlanner(PLANNER, threads=4)
# READY start, pick/place layout and obstacle moved far away so nothing touches the legs
P = dataclasses.replace(sample_params(1), start_xy=(0.0, 0.0), start_yaw=0.0, obstacle_xy=(-5.0, 0.0),
                        layout_xy=(3.0, 0.0), init_joint_noise=0.0)

def start(ep):
    m, d, ix = ep.m, ep.d, ep.ix
    d.qpos[:2] = 0; d.qpos[2] = 0.793; d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[ix.body_qadr] = DEFAULT_ANGLES_MJ; mujoco.mj_forward(m, d)
    q = d.qpos.copy(); q[2] = 0.762
    ep.ref.start(q); ep.ref.add_hold(1.5); ep.run_until_ref_end()
    ep.home_feet_yaw = 0.0; ep.phase = "CROUCH"

def measure(ep, t_hold=2.0):
    zs = []
    for _ in range(int(t_hold / 0.02)):
        ep.ref.add_hold(0.02); ep._step_tick(); zs.append(ep.d.qpos[2])
    q = ep.d.qpos[ep.ix.body_qadr]
    return {"pelvis_z": round(float(np.mean(zs[-25:])), 3), "pelvis_z_min": round(float(min(zs)), 3),
            "ref_pelvis_z": round(float(ep.ref.root_pos[-1][2]), 3),
            "knee_deg": round(float(np.degrees((q[3] + q[9]) / 2)), 1),
            "hip_pitch_deg": round(float(np.degrees((q[0] + q[6]) / 2)), 1),
            "ankle_pitch_deg": round(float(np.degrees((q[4] + q[10]) / 2)), 1),
            "torso_tilt_deg": round(ep.torso_tilt(), 1),
            "feet_on_floor": bool(all(ep.last_contacts["feet"]))}

def snapshot(ep, label):
    r = mujoco.Renderer(ep.m, 360, 360); cam = mujoco.MjvCamera()
    cam.lookat[:] = [ep.d.qpos[0], ep.d.qpos[1], 0.45]; cam.distance, cam.azimuth, cam.elevation = 2.0, 90.0, -5.0
    r.update_scene(ep.d, camera=cam); im = Image.fromarray(r.render()); r.close()
    ImageDraw.Draw(im).text((6, 6), label, fill=(255, 255, 0)); return im

def ik_squat(ep, h, unclamped=False):
    if unclamped:   # the MJCF joint ranges only (kinematics.py caps knee <= 2.2, hip pitch >= -1.9 rad)
        m, ix = ep.m, ep.ix
        ep.ik.lo = m.jnt_range[[m.dof_jntid[v] for v in ix.body_vadr], 0] + 0.02
        ep.ik.hi = m.jnt_range[[m.dof_jntid[v] for v in ix.body_vadr], 1] - 0.02
    nom, pw = ep._posture(); pw[:12] = 0.05
    feet = ep.ref_feet()
    q, info = ep.ik.solve(ep._seed_q(), IKTargets(feet=feet, com_xy=ep._support_center(), pelvis_z=h, posture=nom,
                                                  posture_w=pw, w_feet=100, w_com=30, w_pelvis=10, torso_upright=1.0),
                          iters=400)
    q[ep.ix.body_qadr[15:29]] = DEFAULT_ANGLES_MJ[15:29]       # arms at HOME
    ep.ref.add_segment(q, 3.0); ep.run_until_ref_end()
    qb = q[ep.ix.body_qadr]
    return round(float(info["pelvis_z"]), 3), round(float(np.degrees((qb[3] + qb[9]) / 2)), 1), round(float(np.degrees((qb[0] + qb[6]) / 2)), 1)

rows, shots = [], []
for h in (0.40, 0.35, 0.30, 0.27, 0.25, 0.22):
    pol.reset()
    ep = Episode(P, pol, pl, Config(render=False))
    start(ep)
    row = {"method": "ik_unclamped", "target": h}
    try:
        row["ik_pelvis_z"], row["cmd_knee_deg"], row["cmd_hip_pitch_deg"] = ik_squat(ep, h, unclamped=True)
        row.update(measure(ep, t_hold=3.0)); row["fell"] = False
    except EpisodeFailed as e:
        row.update(fell=True, reason=str(e), pelvis_z=round(float(ep.d.qpos[2]), 3))
    rows.append(row); print(json.dumps(row), flush=True)
    shots.append(snapshot(ep, f"IK no clamp {h:.2f}: z={row.get('pelvis_z')}" + (" FELL" if row["fell"] else "")))
W = 360; sheet = Image.new("RGB", (W * 6, W))
for k, im in enumerate(shots):
    sheet.paste(im, (k * W, 0))
sheet.save(f"{OUT}/squat_unclamped.png"); json.dump(rows, open(f"{OUT}/squat_unclamped.json", "w"), indent=1)
