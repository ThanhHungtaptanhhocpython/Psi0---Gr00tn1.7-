# can_nav_pnp: walk around an obstacle → squat and pick a can off a low stand → stand up → walk to the table → place it in the tray

Data generation for a Unitree G1 (Dex3) driven by SONIC 1.1 in MuJoCo, shared by **Ψ₀** and **GR00T N1.7**. The code was
copied from `../can_pnp` and a walking segment was added at the start; the pick and place part is unchanged from `can_pnp`.
Design, every approach that was tried, and the numbers: [docs/sonic_pick_can/06_memo_can_nav_pnp.md](../../../docs/sonic_pick_can/06_memo_can_nav_pnp.md) (Vietnamese).

## Task

1. The robot starts at **START** (−1.8, 0) m facing +x, jittered by ±5 cm and ±5° (×1.5 with `--level 1`). In front of it
   stands an **obstacle**: a box 30–50 cm wide and 40–60 cm tall, in one of 7 colours (never red like the can or green
   like the tray). The box is at x = −0.8 ± 0.1 m with a **random lateral offset of ±(8–25) cm**.
2. The robot walks around the **wider side** of the obstacle: box offset to the right → detour on the left, and vice versa
   (`SceneParams.nav_side`). The side can be inferred from the head-camera image; the instruction never changes.
   - Waypoints (`scene.nav_waypoints`): a lane 0.5 m beside the box, from 0.2 m before its back face to 0.15 m past its
     front face, then to the goal (−0.1, 0.05).
   - The robot always **faces +x** and steps diagonally or sideways (`nav_face_max_deg = 0`). Turning the body along the
     path made the right hand brush the corner of the obstacle.
   - The first 160 episodes (`can_nav_s0`–`s4`, merged as `can_nav_160`) have the box in the centre (±3 cm,
     40 × 40 × 50 cm, yellow) and always detour on the left. They follow the same rule: a box at y ≤ 4 cm means left.
3. The robot stops. If it is more than 12 cm off the goal, it takes a few short correction steps. It then **steps each
   foot back to the nominal stance** (phase `STEP_ADJUST`): after walking, SONIC stands with the toes splayed
   (+10° / −20°), and squatting from that stance makes the feet slip.
4. **The stand, can, table and tray are placed relative to where the robot actually stopped** (`Episode._relayout`, see
   below). Then the `can_pnp` part runs unchanged: squat, wrap the whole hand around the can, stand up, step to the
   table, put the can in the tray, return to the home pose.

Instruction: *"Walk around the box in front of you to the low stand, squat down, pick up the can from the stand with your
right hand, stand up, walk to the table, put the can in the tray on the table, then return to the home pose."*

### Why the layout follows the robot

SONIC stops (and places its feet) only to within about ±5 cm of the commanded spot. The `can_pnp` grasp needs the can
within ±1.5 cm of its nominal position, and in the squat the feet are only 4.5 cm from the stand. So once the robot has
settled, the code computes the rigid transform from the nominal to the actual standing pose and moves the stand, can,
table and tray by it:

- These objects are static (the can stands still on the stand) and far from the path, so moving them now is equivalent
  to having placed them there from the start. Verified: replaying the recorded actions from START in a scene built with
  the final layout (`closed_loop_eval.py --replay`) still succeeds.
- The frames recorded before the move get their `object.*` columns rewritten for the new layout.
  `meta/episode_meta.jsonl` stores `layout_xy` and `layout_yaw`, and the videos are rendered with the final layout.
- The layout ends up (−8, +8) cm ± 6 cm from nominal on average. Closed-loop evaluation samples layouts from this range
  (`scene.sample_eval_layout`) or takes them from a dataset (`--layout-from`).

## Results so far

| Pilot | Design | Success | Main failures |
| --- | --- | --- | --- |
| 1 (15 seeds) | Goal = READY, no stance correction, no relayout | 1/15 | feet slip in the squat 6, off READY 5, hit the stand 1 |
| 2 (18 seeds) | + stance correction + relayout | 9/18 | hit the stand on approach 2, > 10 cm off the goal 2, too close to the obstacle 1 |
| 3 (18 seeds) | + goal 10 cm short of READY, lane at y = 0.70, relaxed thresholds | **14/18** | feet slip / body shift while reaching 3, missed reach 1 |
| `--level 1`, random obstacle (40 attempts) | current design | **18/40** | missed grasps 9, stance still off after correction 4, slip while reaching 3 |

Walking alone with `--level 1` reached the goal in 12/12 seeds (7 left, 5 right), with ≥ 8 cm of clearance to the
obstacle. Episodes last 67–83 s (about 2,000–2,450 frames). `--level 1` also multiplies the can-position jitter by 1.5
(±2.25 cm instead of ±1.5 cm), which explains most of the drop; leaving the can jitter unscaled would raise the yield
while keeping the walking diversity.

## Running

Each process uses about 2.2–2.8 GB of RAM. On a 31 GB machine, run at most 5–6 at once.

```bash
# pilot: success rate (physics only); --set stop_after_ready=True runs only the walking part
SONIC_VARIANT=sonic_v1_1 .venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py \
    --out .data/can_nav_pilot --dry-run --num-episodes 6 --max-attempts 6 --seed-start 900000 --threads 2

# real run: shard by seed range (below 1,000,000; seeds from 1,000,000 are reserved for evaluation), then merge
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py --out .data/can_nav_s0 --num-episodes 32 --seed-start 0 --threads 2
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py --out .data/can_nav_s1 --num-episodes 32 --seed-start 10000 --threads 2
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/merge_shards.py --out .data/can_nav_160 .data/can_nav_s0 .data/can_nav_s1 ...

# export: the same --seed gives Ψ₀ and GR00T the same validation episodes
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/export_psi0.py  .data/can_nav_160 --out .data/psi0  --zip
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/export_gr00t.py .data/can_nav_160 --out .data/gr00t --zip

# closed-loop evaluation
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py --server http://HOST:PORT --episodes 10 --out .data/eval/psi0
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py --gr00t HOST:PORT --episodes 10 --out .data/eval/gr00t
```

`--gr00t` needs the ZMQ libraries (once): `uv pip install --python .venv-datagen/bin/python pyzmq msgpack`.

`generate.py` options:
- `--level 1`: ×1.5 jitter amplitude.
- `--wrist-video`: also render the right wrist camera. Off by default: both models currently use only the head camera.
- `--no-render`: write only parquet and meta. Render the videos later with `render_views.py`, which works from the
  recorded states, so nothing is simulated again.
- `--set KEY=VALUE`: override a field of `episode.Config`, e.g. `--set feet_slip_tol=0.05`.

```bash
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/render_views.py .data/can_nav_500 --views ego --jobs 4      # missing videos
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/render_views.py .data/can_nav_500 --views wrist --jobs 4    # add the wrist camera
```

### Generating in blocks on Colab

`run_blocks.py` splits the seeds into small blocks (8 episodes each by default) and runs N processes in parallel. Each
finished block is copied straight to persistent storage (Google Drive). A rerun skips the blocks already there, so a
recycled runtime loses only the blocks in flight. If blocks keep crashing (twice the number of workers, with none
finished), it prints the crash log and stops instead of retrying forever.

1. Build the pack: `.venv-datagen/bin/python scripts/datagen/can_nav_pnp/pack_colab.py` → `.data/colab/can_nav_datagen.zip`
   (about 850 MB: code, G1 model, SONIC 1.1, planner, pinned library versions).
2. Upload it to `My Drive/can_nav/` and open [`colab_datagen.ipynb`](colab_datagen.ipynb). Give every session its own
   `SEED_BASE` (100000, 300000, 500000, …).
3. Download `My Drive/can_nav/blocks` and merge the blocks with `merge_shards.py`.

Choosing a runtime:
- **GPU (L4):** with `RENDER = "auto"`, videos are rendered with EGL on the GPU. Cell [4] prints the renderer and stops if
  EGL fell back to the CPU (Mesa llvmpipe); cell [3] writes the NVIDIA EGL vendor file that prevents this.
- **CPU only:** `RENDER = "auto"` skips rendering; render later on a GPU machine with `render_views.py --views ego`.
  OSMesa (CPU rendering) is too slow with many processes and is used only with `RENDER = "osmesa"`.
- Simulation speed depends on CPU cores and RAM (about 2.5 GB per process), not on the GPU.

Cell [5] prints a line every minute: episodes on Drive, successes inside the blocks still running, episodes per hour,
an ETA (Vietnam time), and each running block's successes/attempts.

## Data

The raw dataset is LeRobot v2.1 at 30 FPS, written by `writer.py`. Action 78 = SONIC token 64 + left hand 7 + right
hand 7. State 43 follows the MJCF qpos order. Compared with `can_pnp` there are these extra columns:

| Column | Content |
| --- | --- |
| `object.obstacle_pose` | obstacle pose (7) |
| `nav.planner_cmd` | last SONIC planner command: mode, xyz movement direction, facing direction, speed, height (7). Mode −1 means no planner segment is running. Usable if GR00T should later learn REAL_G1-style walking commands. |
| `nav.waypoint` | current target waypoint (2) |
| `contact.robot_obstacle`, `error.obstacle_clearance` | contact with and clearance to the obstacle |

New phases: `NAV_WALK`, `NAV_AROUND`, `NAV_APPROACH`, `STEP_ADJUST`, `STABILIZE_READY`. New events: `nav_started`,
`obstacle_passed`, `ready_reached`.

| Export | Format |
| --- | --- |
| `export_psi0.py` | As in `can_pnp`: state 43 in the 0810 order, `action.body_token` 64, `action` 36 (hands in `[:14]`), video `observation.images.egocentric` 320×240, `stats_psi0.json` |
| `export_gr00t.py` | Embodiment `unitree_g1_sonic`. State 46 = 0810 + projected gravity 3; action 78 = `motion_token` 64, `left_hand_joints` 7, `right_hand_joints` 7; video `observation.images.ego_view` 320×240; language `annotation.human.task_description`; `meta/stats.json` |

The right wrist camera (`right_wrist_cam`, above the wrist, looking along the fingers) exists in the code but is off by
default. On a real robot it would have to be mounted at exactly that pose.

## Rejection reasons (on top of those from `can_pnp`)

| Reason | When |
| --- | --- |
| `robot_obstacle_collision` | The robot touches the obstacle |
| `obstacle_too_close` | Clearance to the obstacle below 5 cm |
| `nav_timeout` | Not standing still at the goal after 30 s |
| `ready_pose_off` | More than 18 cm or 8° off the goal |
| `ready_stance_off` | After the correction steps, a foot is still more than 8 cm or 6° off the nominal stance |

The pre-squat checks (foot slip, body shift) are measured from the stance at READY.

## Finetuning, serving and evaluation

| | Ψ₀ | GR00T N1.7 |
| --- | --- | --- |
| Data | `.data/psi0/can_nav_160_psi0.zip` → `My Drive/psi0/` | `.data/gr00t/can_nav_160_gr00t.zip` → `My Drive/gr00t/` |
| Finetune | [`colab_finetune_psi0_sonic11.ipynb`](colab_finetune_psi0_sonic11.ipynb), `EXP = cannav160-s11` | [`colab_finetune_gr00t_sonic11.ipynb`](colab_finetune_gr00t_sonic11.ipynb), tag `UNITREE_G1_SONIC` |
| Server | [`colab_psi0_server.ipynb`](colab_psi0_server.ipynb): `serve_psi0_sonic_http`, Ψ₀ code `2830f93` | [`colab_gr00t_server.ipynb`](colab_gr00t_server.ipynb): `gr00t/eval/run_gr00t_server.py` (ZMQ) |
| Evaluation | `closed_loop_eval.py --server http://HOST:PORT` | `closed_loop_eval.py --gr00t HOST:PORT` |

- Both models use an effective batch of 128 and 8,000 steps; every other setting stays at each model's defaults.
- Use a new `EXP` name for every new dataset: the finetune notebook resumes the run named `EXP`.
- The base GR00T checkpoint has no statistics for the `UNITREE_G1_SONIC` tag, so the statistics come from our data.
- The Ψ₀ server patches `serve_psi0_simple.py` so the SONIC 1.1 checkpoint gets its CLIP text embedding (upstream passes
  `None`), and installs `fastapi` / `uvicorn`.

**Evaluating on Colab (recommended).** The server notebooks (Ψ₀ cells 6b/6c, GR00T steps 5b/5c) run the MuJoCo simulation
on the same Colab machine and call the server over `localhost`: no tunnel, about 0.6 s per request instead of 1.2 s.
They use the eval pack `can_nav_eval.zip` (`pack_colab.py --eval`, about 200 MB, no planner), render with EGL on the
GPU, show a live preview and upload each scene's evidence to Drive.

**Evaluating locally.** Each server notebook prints the command, for example:

```bash
cd ~/Code/PSI/psi0
SONIC_VARIANT=sonic_v1_1 .venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py \
    --server http://HOST:PORT --obstacle center --layout-from .data/can_nav_160 \
    --episodes 10 --viewer --video --out .data/eval/cannav160
```

- `--obstacle center` builds the centred obstacle of the first 160 episodes. Leave it out for checkpoints trained on the
  500-episode set.
- On a machine without a display (SSH), leave out `--viewer` and prefix the command with `MUJOCO_GL=egl` (GPU) or
  `MUJOCO_GL=osmesa`.
- With `--viewer`, the model's images are rendered with EGL, separately from the viewer's GLFW context: closing the viewer
  used to corrupt the offscreen renderer and feed noise to the model. A guard stops a scene with `ERROR` if the head
  image looks like noise.

**Evidence per scene.** Every scene gets a folder `seed_<seed>_<RESULT>` (`SUCCESS`, `FAIL_<reason>`, `ERROR` or
`STOPPED`) with `ego.mp4`, `third.mp4`, `overview.png` (12 moments, outside view + head camera, with a result banner),
`result.json` and `trace.npz` (time, qpos, can pose, executed actions, contacts at 30 Hz). `results.jsonl`,
`summary.md` / `summary.json` and `run_config.json` are rewritten after every scene. A can that tips over inside the
tray counts as a success, as in the data.

Rates: physics 200 Hz, SONIC 50 Hz, actions 30 Hz, the model is queried every 15 steps (2 Hz).

## SONIC 1.1 is required

As in `can_pnp`, SONIC 1.0 does not lower the body and arms far enough for this task. Download SONIC 1.1 with
`python download_from_hf.py --sonic-v1-1` in `third_party/GR00T-WholeBodyControl`. `SONIC_VARIANT` defaults to `sonic_v1_1`.

SONIC 1.1 cannot squat fully either: the lowest pelvis height is about 0.42–0.47 m (0.76 m standing), and commanding
deeper only lifts the heels (`probes/probe_squat.py`, `probes/probe_squat_deep.py`). That is why the can sits on a low
stand; picking it off the floor would need SONIC itself to be finetuned.
