# can_nav_pnp: đi vòng vật cản → ngồi xổm lấy lon trên bục → đứng dậy → đi tới bàn → đặt vào khay

Pipeline sinh dữ liệu cho G1 (Dex3) với SONIC 1.1 trong MuJoCo, dùng chung cho **Ψ₀** và **GR00T N1.7**. Được chép từ
`../can_pnp` rồi thêm đoạn đi bộ ở đầu; phần nhặt và đặt lon giữ nguyên như `can_pnp`.
Thiết kế, các phương án đã thử và số liệu: [docs/sonic_pick_can/06_memo_can_nav_pnp.md](../../../docs/sonic_pick_can/06_memo_can_nav_pnp.md).

## Task

1. Robot bắt đầu ở **START** (−1,8; 0) m, hướng +x, dao động ±5 cm, ±5° (nhân 1,5 với `--level 1`). Phía trước có một
   **vật cản**: hộp rộng 30–50 cm, cao 40–60 cm, màu chọn ngẫu nhiên trong một bảng màu. Hộp đặt ở x = −0,8 ± 0,1 m và
   **lệch ngang ngẫu nhiên ±(8–25) cm**.
2. Robot đi vòng **bên rộng hơn** của vật cản: hộp lệch phải thì đi trái, hộp lệch trái thì đi phải (`SceneParams.nav_side`).
   Hướng đi suy ra được từ ảnh camera đầu, câu lệnh không đổi.
   - Waypoint (`scene.nav_waypoints`): làn đi cách mặt bên của hộp 0,5 m, bắt đầu 0,2 m trước mặt sau của hộp và kết thúc 0,15 m
     sau mặt trước, rồi đi tới đích (−0,1; 0,05).
   - Robot luôn **nhìn về +x** và bước chéo/ngang (`nav_face_max_deg = 0`): nếu xoay người theo hướng đi, tay phải quệt vào góc
     vật cản.
   - 160 episode đầu (`can_nav_s0`–`s4`) có hộp ở giữa (±3 cm, 40 × 40 × 50 cm, màu vàng) và luôn đi trái. Chúng khớp quy tắc
     trên: hộp có y ≤ 4 cm thì đi trái.
3. Robot dừng. Nếu lệch đích hơn 12 cm, robot bước thêm vài bước ngắn để chỉnh. Sau đó robot **bước từng chân về tư thế đứng chuẩn**
   (pha `STEP_ADJUST`). Lý do: sau khi đi, SONIC đứng xoè mũi chân +10°/−20°, và ngồi xổm từ tư thế đó làm chân trượt.
4. **Bục, lon, bàn và khay được đặt theo chỗ robot thật sự dừng** (`Episode._relayout`, xem bên dưới), rồi chạy nguyên
   phần `can_pnp`: ngồi xổm, ôm lon bằng cả bàn tay, đứng dậy, bước tới bàn, đặt lon vào khay, về tư thế ban đầu.

Câu lệnh: *"Walk around the box in front of you to the low stand, squat down, pick up the can from the stand with your right
hand, stand up, walk to the table, put the can in the tray on the table, then return to the home pose."*

### Vì sao dời layout theo chỗ robot dừng

SONIC chỉ dừng (và đặt chân) chính xác khoảng ±5 cm so với điểm được lệnh. Phần nhặt lon của `can_pnp` cần lon nằm trong ±1,5 cm
so với vị trí chuẩn, và chân chỉ cách bục 4,5 cm lúc ngồi xổm. Vì thế, khi robot đã đứng yên, code tính phép biến đổi cứng từ tư thế
đứng chuẩn sang tư thế đứng thật rồi dời bục, lon, bàn và khay theo phép biến đổi đó:

- Các vật này là vật tĩnh (lon đứng yên trên bục) và nằm xa đường đi. Vì thế dời chúng lúc này tương đương với đặt sẵn chúng ở đó từ
  đầu. Đã kiểm tra: phát lại action đã ghi từ START trong cảnh đặt sẵn layout cuối (`closed_loop_eval.py --replay`) vẫn thành công.
- Các khung đã ghi trước đó được viết lại `object.*` theo layout mới. `meta/episode_meta.jsonl` lưu `layout_xy` và `layout_yaw`,
  và video được render với layout cuối.
- Layout lệch trung bình (−8; +8) cm ± 6 cm so với chuẩn. Khi đánh giá vòng kín, layout được rút ngẫu nhiên trong khoảng này
  (`scene.sample_eval_layout`), hoặc lấy từ một dataset (`--layout-from`).

## Chạy

Mỗi tiến trình tốn khoảng 2,2–2,8 GB RAM. Trên máy 31 GB, chạy tối đa 5–6 tiến trình cùng lúc.

```bash
# pilot đo tỉ lệ đạt (chỉ mô phỏng vật lý); --set stop_after_ready=True để chỉ chạy đoạn đi bộ
SONIC_VARIANT=sonic_v1_1 .venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py \
    --out .data/can_nav_pilot --dry-run --num-episodes 6 --max-attempts 6 --seed-start 900000 --threads 2

# chạy thật: shard theo dải seed (dưới 1 000 000; dải từ 1 000 000 dành cho đánh giá) rồi gộp
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py --out .data/can_nav_s0 --num-episodes 32 --seed-start 0 --threads 2
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/generate.py --out .data/can_nav_s1 --num-episodes 32 --seed-start 10000 --threads 2
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/merge_shards.py --out .data/can_nav_160 .data/can_nav_s0 .data/can_nav_s1 ...

# export: cùng --seed nên Ψ₀ và GR00T có cùng các episode val
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/export_psi0.py  .data/can_nav_160 --out .data/psi0  --zip
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/export_gr00t.py .data/can_nav_160 --out .data/gr00t --zip

# đánh giá vòng kín
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py --server http://HOST:PORT --episodes 10 --out .data/eval/psi0
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py --gr00t HOST:PORT --episodes 10 --out .data/eval/gr00t
```

`--gr00t` cần thêm thư viện ZMQ (một lần): `uv pip install --python .venv-datagen/bin/python pyzmq msgpack`.

Tuỳ chọn của `generate.py`:
- `--level 1`: biên độ dao động ×1,5.
- `--wrist-video`: render thêm camera cổ tay phải. Mặc định tắt: hiện cả hai model chỉ dùng camera đầu.
- `--no-render`: chỉ ghi parquet và meta. Video render sau bằng `render_views.py`, đọc từ trạng thái đã ghi nên không cần mô
  phỏng lại.

```bash
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/render_views.py .data/can_nav_500 --views ego --jobs 4      # video còn thiếu
.venv-datagen/bin/python scripts/datagen/can_nav_pnp/render_views.py .data/can_nav_500 --views wrist --jobs 4    # thêm camera cổ tay
```

### Sinh dữ liệu theo khối, trên Colab

`run_blocks.py` chia seed thành các khối nhỏ (mặc định 8 episode) và chạy N tiến trình song song. Khối nào xong được chép
ngay sang thư mục lưu lâu dài (Drive). Chạy lại thì bỏ qua các khối đã có. Trên Colab:

1. Tạo gói: `.venv-datagen/bin/python scripts/datagen/can_nav_pnp/pack_colab.py` → `.data/colab/can_nav_datagen.zip`
   (khoảng 850 MB: code, G1, SONIC 1.1, planner, thư viện ghim phiên bản).
2. Tải gói lên `My Drive/can_nav/`, mở [`colab_datagen.ipynb`](colab_datagen.ipynb). Mỗi phiên đặt `SEED_BASE` riêng.
3. Tải `My Drive/can_nav/blocks` về rồi gộp với `merge_shards.py`.

Tốc độ phụ thuộc CPU và RAM (mỗi tiến trình khoảng 2,5 GB), không phụ thuộc GPU. Runtime CPU High-RAM là lựa chọn kinh tế nhất.

## Dữ liệu

Bộ dữ liệu gốc dùng định dạng LeRobot v2.1, 30 FPS, ghi bởi `writer.py`. Action 78 = token SONIC 64 + tay trái 7 + tay phải 7.
State 43 theo thứ tự qpos của MJCF. So với `can_pnp`, có thêm các cột sau:

| Cột | Nội dung |
| --- | --- |
| `object.obstacle_pose` | pose vật cản (7) |
| `nav.planner_cmd` | lệnh planner SONIC cuối cùng: mode, hướng đi xyz, hướng mặt, vận tốc, độ cao (7). Mode −1 nghĩa là không có đoạn planner nào đang chạy. Dùng được nếu sau này muốn GR00T học lệnh đi kiểu REAL_G1. |
| `nav.waypoint` | waypoint đang nhắm tới (2) |
| `contact.robot_obstacle`, `error.obstacle_clearance` | va chạm và khoảng hở tới vật cản |

Pha mới: `NAV_WALK`, `NAV_AROUND`, `NAV_APPROACH`, `STEP_ADJUST`, `STABILIZE_READY`. Sự kiện mới: `nav_started`,
`obstacle_passed`, `ready_reached`.

| Export | Định dạng |
| --- | --- |
| `export_psi0.py` | Như `can_pnp`: state 43 theo thứ tự 0810, `action.body_token` 64, `action` 36 (tay ở `[:14]`), video `observation.images.egocentric` 320×240, `stats_psi0.json` |
| `export_gr00t.py` | Embodiment `unitree_g1_sonic`. State 46 = 0810 + trọng lực chiếu 3; action 78 = `motion_token` 64, `left_hand_joints` 7, `right_hand_joints` 7; video `observation.images.ego_view` 320×240; ngôn ngữ `annotation.human.task_description`; `meta/stats.json` |

## Lý do loại episode (ngoài các lý do của `can_pnp`)

| Lý do | Khi nào |
| --- | --- |
| `robot_obstacle_collision` | Robot chạm vật cản |
| `obstacle_too_close` | Khoảng hở tới vật cản dưới 5 cm |
| `nav_timeout` | Quá 30 s mà chưa đứng yên ở đích |
| `ready_pose_off` | Lệch đích hơn 18 cm hoặc 8° |
| `ready_stance_off` | Sau khi bước chỉnh, một chân vẫn lệch tư thế chuẩn hơn 8 cm hoặc 6° |

Kiểm tra trước khi ngồi xổm (trượt chân, dịch người) tính lại từ tư thế đứng ở READY.

## Finetune

| | Ψ₀ | GR00T N1.7 |
| --- | --- | --- |
| Dữ liệu | `.data/psi0/can_nav_160_psi0.zip` → `My Drive/psi0/` | `.data/gr00t/can_nav_160_gr00t.zip` → `My Drive/gr00t/` |
| Finetune | [`colab_finetune_psi0_sonic11.ipynb`](colab_finetune_psi0_sonic11.ipynb), `EXP = cannav160-s11` | [`colab_finetune_gr00t_sonic11.ipynb`](colab_finetune_gr00t_sonic11.ipynb), tag `UNITREE_G1_SONIC` |
| Server | [`colab_psi0_server.ipynb`](colab_psi0_server.ipynb): `serve_psi0_sonic_http`, code Ψ₀ `2830f93` | [`colab_gr00t_server.ipynb`](colab_gr00t_server.ipynb): `gr00t/eval/run_gr00t_server.py` (ZMQ) |
| Đánh giá (local) | `closed_loop_eval.py --server http://HOST:PORT` | `closed_loop_eval.py --gr00t HOST:PORT` |

- Checkpoint gốc GR00T chưa có thống kê cho tag `UNITREE_G1_SONIC`, nên thống kê lấy từ dữ liệu của ta.
- Mỗi notebook server in sẵn lệnh đánh giá cho máy local, dạng:

```bash
cd ~/Code/PSI/psi0
SONIC_VARIANT=sonic_v1_1 .venv-datagen/bin/python scripts/datagen/can_nav_pnp/closed_loop_eval.py \
    --server http://HOST:PORT --obstacle center --layout-from .data/can_nav_160 \
    --episodes 10 --viewer --video --out .data/eval/cannav160
```

- `--obstacle center` dựng vật cản ở giữa như 160 episode đầu. Checkpoint train trên bộ 500 episode thì bỏ cờ này.
- Máy không có màn hình (SSH): bỏ `--viewer`, thêm `MUJOCO_GL=egl` (có GPU) hoặc `MUJOCO_GL=osmesa` trước lệnh.

## Bắt buộc dùng SONIC 1.1

Giống `can_pnp`: SONIC 1.0 không hạ người và tay đủ thấp để làm task này. Tải SONIC 1.1 bằng
`python download_from_hf.py --sonic-v1-1` trong `third_party/GR00T-WholeBodyControl`. Biến `SONIC_VARIANT` mặc định là `sonic_v1_1`.
