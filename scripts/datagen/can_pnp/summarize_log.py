"""Summarize meta/generation_log.jsonl of one or more generation runs (pilot QA).

  python scripts/datagen/can_pnp/summarize_log.py .data/can_pnp_pilot12 [.data/other_shard ...] [--csv out.csv]

Prints the success rate, failure reasons (grouped, with the phase they happened in) and, per
phase / checkpoint, the distribution of the logged diagnostics:
  * foot contact + foot displacement before the walk (max travel, stance shift, lift-offs,
    fraction of ticks with both feet on the floor), leg <-> pedestal clearance;
  * box tilt and wrist orientation (drift vs. the grasp orientation, tracking vs. the command)
    at GRASP, LIFT, STAND_UP, AFTER_WALK, APPROACH_END, FIRST_TABLE_CONTACT, PRE_RELEASE;
  * grasp slip per phase, XY distance box -> placement target, table contacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CHECKPOINTS = ["TURN_TORSO", "GRASP", "LIFT", "RAISE_TORSO", "TURN_BACK", "STAND_UP", "STABILIZE_UPRIGHT", "AFTER_WALK",
               "ADJUST_ARM", "APPROACH_END",
               "FIRST_TABLE_CONTACT", "PRE_RELEASE", "FINAL"]
PREWALK = ["HOME", "CROUCH", "TURN_TORSO", "REACH", "GRASP", "SLIP", "REGRASP_REACH", "LIFT", "RAISE_TORSO",
           "TURN_BACK", "STAND_UP", "STABILIZE_UPRIGHT"]
PED = ["CROUCH", "TURN_TORSO", "REACH", "GRASP", "LIFT", "TURN_BACK", "STAND_UP", "WALK_STEP_1", "WALK_STEP_2",
       "STABILIZE_AT_TABLE"]


def load(dirs):
    rows = []
    for d in dirs:
        p = Path(d) / "meta" / "generation_log.jsonl"
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def q(vals, fmt="{:.3f}"):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return "-"
    return (f"n={len(v):3d}  med " + fmt.format(np.median(v)) + "  p90 " + fmt.format(np.percentile(v, 90))
            + "  max " + fmt.format(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--csv", help="also write one row per attempt")
    a = ap.parse_args()
    rows = load(a.dirs)
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    print(f"attempts {n}   SUCCESS {len(ok)}  ({100 * len(ok) / max(n, 1):.0f}%)\n")

    print("failure reasons (phase: reason)")
    short = lambda reason: re.sub(r"\(.*", "", reason)
    reasons = Counter(f"{r['info'].get('fail_phase')}: {short(r['fail_reason'])}" for r in rows if not r["success"])
    for k, v in reasons.most_common():
        print(f"  {v:3d}  {k}")
    for r in rows:
        if not r["success"]:
            print(f"      seed {r['seed']:>6}: {r['fail_reason']}")

    D = [r["info"].get("diag", {}) for r in rows]
    print("\nbefore WALK: feet (all attempts that reached the phase)")
    for ph in PREWALK:
        st = [d.get("prewalk", {}).get(ph) for d in D]
        st = [s for s in st if s]
        if not st:
            continue
        both = [s["both_contact_ticks"] / max(s["ticks"], 1) for s in st]
        lo = sum(s.get("liftoffs", 0) for s in st)
        print(f"  {ph:18s} travel {q([s['max_foot_travel'] for s in st])} | shift {q([s['max_mid_shift'] for s in st])}"
              f" | both-feet-contact min {min(both):.2f} | lift-offs {lo}")

    print("\nright leg <-> pedestal clearance [m] (min over the phase; < 0.01 is a near miss)")
    for ph in PED:
        v = [d.get("ped_clear", {}).get(ph, {}).get("right") for d in D]
        v = [x for x in v if x is not None]
        if v:
            print(f"  {ph:18s} min {min(v):.3f}  med {np.median(v):.3f}")

    print("\nright hand/forearm <-> tabletop clearance before WALK [m] (min over the phase)")
    for ph in PREWALK:
        v = [d.get("hand_table_clear", {}).get(ph, {}).get("min") for d in D]
        v = [x for x in v if x is not None]
        if v:
            print(f"  {ph:18s} min {min(v):.3f}  med {np.median(v):.3f}")

    print("\ncheckpoints (box tilt [deg] | wrist drift vs grasp [deg] | wrist tracking [deg] | box->target xy [m])")
    for cp in CHECKPOINTS:
        c = [d.get("checkpoints", {}).get(cp) for d in D]
        c = [x for x in c if x]
        if not c:
            continue
        print(f"  {cp:20s} tilt   {q([x['box_tilt_deg'] for x in c], '{:.1f}')}")
        print(f"  {'':20s} drift  {q([x.get('wrist_drift_deg') for x in c], '{:.1f}')}")
        print(f"  {'':20s} track  {q([x.get('wrist_track_deg') for x in c], '{:.1f}')}")
        print(f"  {'':20s} xy     {q([x['box_place_xy_err'] for x in c])}")

    print("\ngrasp slip [m] (max per phase)")
    phases = sorted({ph for d in D for ph in d.get("slip_max", {})},
                    key=lambda p: [*CHECKPOINTS, *PED, "APPROACH_PLACE", "LOWER_BOX", "RELEASE"].index(p)
                    if p in [*CHECKPOINTS, *PED, "APPROACH_PLACE", "LOWER_BOX", "RELEASE"] else 99)
    for ph in phases:
        print(f"  {ph:18s} {q([d['slip_max'][ph]['m'] for d in D if ph in d.get('slip_max', {})], '{:.4f}')}")

    print("\nheld box <-> legs clearance [m] (min over the phase) | episodes where the box touches the body (not the hand)")
    for ph in [*PREWALK, "WALK_STEP_1", "WALK_STEP_2", "STABILIZE_AT_TABLE", "ADJUST_ARM", "APPROACH_PLACE"]:
        v = [r["info"]["diag"].get("box_leg_clear", {}).get(ph, {}).get("min") for r in rows]
        t = [r["info"]["diag"].get("box_body_ticks", {}).get(ph, {}).get("n", 0) for r in rows]
        if any(x is not None for x in v):
            print(f"  {ph:18s} {q(v)} | body contact in {sum(1 for x in t if x)} episodes")

    print("\ntable contact")
    rt = defaultdict(int)
    for d in D:
        for ph, v in d.get("robot_table_ticks", {}).items():
            rt[ph] += 1
    print("  robot<->table touching, #episodes per phase:", dict(rt) or "none")
    print("  box first touches the table in:", dict(Counter(d.get("place", {}).get("box_first_table_phase") for d in D if d.get("place", {}).get("box_first_table_phase"))))
    pl = [d.get("place", {}) for d in D if d.get("place", {}).get("corrections") is not None]
    if pl:
        print(f"  approach xy err {q([p['approach_xy_err'] for p in pl])}")
        print(f"  corrections used: {dict(Counter(p['corrections'] for p in pl))}; tilt fixes: "
              f"{sum(bool(p.get('tilt_fix_deg')) for p in pl)}")
        sup = [p for p in pl if 'supported_xy_err' in p]
        if sup:
            print(f"  supported: xy err {q([p['supported_xy_err'] for p in sup])} | tilt {q([p['supported_tilt_deg'] for p in sup], '{:.1f}')}")
    hb = [sum(v["ticks"] for v in d.get("hand_box_after_release", {}).values()) for d in D if "RETRACT" in d.get("prewalk", {}) or d.get("hand_box_after_release")]
    if hb:
        print(f"  hand touching the box after release [ticks]: {q(hb, '{:.0f}')}")
    w = [d.get("walk") for d in D if d.get("walk")]
    if w:
        print(f"\nwalk: travel {q([x['travel'] for x in w])} | touchdowns {dict(Counter(x['touchdowns'] for x in w))}")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            hdr = ["seed", "success", "fail_phase", "fail_reason"] + [f"{cp}_{k}" for cp in CHECKPOINTS
                                                                       for k in ("tilt", "drift", "track", "xy")]
            wr.writerow(hdr)
            for r, d in zip(rows, D):
                row = [r["seed"], r["success"], r["info"].get("fail_phase"), r["fail_reason"]]
                for cp in CHECKPOINTS:
                    c = d.get("checkpoints", {}).get(cp, {})
                    row += [c.get("box_tilt_deg"), c.get("wrist_drift_deg"), c.get("wrist_track_deg"), c.get("box_place_xy_err")]
                wr.writerow(row)


if __name__ == "__main__":
    main()
