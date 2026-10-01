# Báo cáo lần chạy gần nhất của bộ sinh dữ liệu `scripts/datagen/box_pnp`

> Mục đích: ghi lại **kịch bản, toàn bộ thông số, kết quả 12 seed** của lần chạy cuối và **chẩn đoán từng kiểu thất bại**, để quyết định cách khắc phục trước khi sinh 500 episode.
>
> Trạng thái code: [scripts/datagen/box_pnp/](../../scripts/datagen/box_pnp/) (chưa commit). Môi trường: `.venv-datagen` (Python 3.11, MuJoCo 3.14, onnxruntime CPU). SONIC ONNX tại `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/{policy/release, planner/target_vel/V2}`.
>
> Lệnh tái tạo lần chạy: `.venv-datagen/Scripts/python scripts/datagen/box_pnp/generate.py --out .data/box_pnp_test --num-episodes 1 --seed-start <seed>` (hoặc script trace trong phần 6).

---

## 1. Kịch bản đang mô phỏng

Bố cục đã chốt sau hai lần bạn góp ý: **hộp ở phía trước, lệch phải nhẹ (khoảng 35°), đặt trên một bệ thấp, ngoài làn chân phải**, để robot cầm bằng tay phải mà không phải xoay người và đường đi thẳng tới bàn không bị chắn.

```
   y (trái)
   ^
   |            +-----------------------+
   |            |        BÀN  z=0.62    |   mép gần x≈0.40, điểm đặt ≈ (0.45, -0.16)
   |            +-----------------------+
   |
   |     [G1]  ─ x (trước) ─►   ▪ bệ 16×16 cm, cao 0.20 m, tại (0.33, -0.27)
   |    chân trái y=+0.12          hộp 5×5×10 cm đứng trên bệ, cầm ở 3 cm dưới đỉnh
   |    chân phải y=-0.12          => điểm cầm z ≈ 0.27 m
```

Máy trạng thái (nhãn pha được lưu vào từng frame):

`HOME → CROUCH → REACH → GRASP → LIFT → STAND_UP → STABILIZE_UPRIGHT → WALK_STEP_1 → WALK_STEP_2 → STABILIZE_AT_TABLE → PLACE → RELEASE → RETRACT → RETURN_HOME → SUCCESS`

Cách từng pha được thực hiện (expert):

| Pha | Cách tạo tham chiếu cho SONIC | Điều kiện chuyển pha |
|---|---|---|
| HOME | giữ pose mặc định 1,5 s (không ghi) rồi 0,6 s (ghi) | pelvis > 0,72 m |
| CROUCH | **planner SONIC mode 4 (squat)**, `height=0.42` | tham chiếu ổn định |
| REACH | IK toàn thân, chân giữ tư thế squat, **servo vị trí** tối đa 8 lần, gain 0,7 | sai số điểm cầm < 2 cm |
| GRASP | đóng 7 khớp Dex3 trong 0,6 s; **weld mềm** bật khi ≥ 30 % đóng, có tiếp xúc ngón/ngón cái và hộp nằm trong "túi kẹp" (< 3 cm) | weld đã bật |
| LIFT | IK: nâng điểm cầm 10 cm và lùi 6 cm | hộp cao hơn bệ > 3 cm |
| STAND_UP | IK về chân đứng, hộp ở `carry_box_pos` | — |
| STABILIZE_UPRIGHT | giữ | pelvis > 0,72 m, vận tốc < 0,06 m/s, 0,35 rad/s trong 0,5 s |
| WALK | **planner mode 1 (slowWalk)** hướng thẳng, replan mỗi 0,2 s, chuyển idle sau 2 lần đặt chân | idle > 1,2 s, hai chân chạm, ổn định |
| PLACE | IK 2 giai đoạn (ngang → hạ xuống 5 cm trên mặt bàn), rồi **servo hướng** để dựng hộp thẳng, rồi hạ 8 mm/bước tới khi bàn đỡ | hộp chạm bàn, lệch xy < 4 cm, nghiêng < 20° |
| RELEASE | mở tay 0,8 s, tắt weld | — |
| RETRACT | gập khuỷu tay phải +0,45 rad, vai +0,1 rad | tay không chạm hộp, hộp nằm trên bàn |
| RETURN_HOME | IK về pose mặc định tại vị trí mới | ổn định, sai khớp trung bình < 0,12 rad |
| SUCCESS | hộp nằm đúng độ cao bàn (± 1 cm), v < 2 cm/s, nghiêng < 10° | |

Ghi dữ liệu 30 fps: ảnh ego 640×480 (fovy 80°), state 43-D, action 78-D = `[token SONIC 64 | tay trái 7 | tay phải 7]`, cùng qpos/qvel/pose vật thể/tiếp xúc/pha/sự kiện. Token được **latch ở 30 Hz** trước khi đưa vào decoder 50 Hz, giống client Ψ₀ khi deploy.

---

## 2. Thông số của lần chạy gần nhất

### 2.1 Scene (`scene.py::sample_params`, level 0)

| Tham số | Giá trị / khoảng random | Ghi chú |
|---|---|---|
| Hộp `box_half` | (0.025, 0.025, **0.05**) → 5×5×10 cm | lần trước đó là 8 cm (thất bại toàn bộ ở REACH) và 12 cm |
| `box_mass` | U(0.13, 0.17) kg | |
| `box_xy` | x ∈ U(0.31, 0.35), y ∈ U(−0.29, −0.26) | góc lệch ≈ 38–43° so với trục trước |
| `box_yaw` | U(−0.15, 0.15) rad | |
| Bệ `pedestal_height` | U(0.19, 0.21) m, mặt 16×16 cm | điểm cầm z ≈ 0.26–0.28 |
| Bàn `table_edge_x` | U(0.39, 0.42) | |
| `table_height` | **U(0.60, 0.64)** | lần trước 0.53–0.57 |
| `table_y` | U(−0.03, 0.03) | |
| Điểm đặt `place_xy_offset` | dx ∈ U(0.045, 0.06), dy ∈ U(−0.18, −0.14) | |
| `init_joint_noise` | 0.02 rad | |
| `init_yaw` | U(−0.03, 0.03) | |
| Camera | fovy 80°, jitter pitch/yaw ± 0.02 rad | |
| Ánh sáng, màu sàn/bàn | random nhẹ | |

### 2.2 Điều khiển (`episode.py::Config`)

| Tham số | Giá trị |
|---|---|
| `token_rate_hz` | 30 |
| `squat_height` | 0.42 (pelvis thực tế đạt ≈ 0.50–0.52) |
| `reach_tol` / `reach_iters` / `reach_gain` | 0.02 m / 8 / 0.7 |
| `grasp_close_time` | 0.6 s |
| `lift_height` | 0.10 m (+ lùi 0.06 m) |
| `carry_box_pos` | (0.20, −0.22, **0.76**) trong hệ pelvis-yaw |
| `carry_upright_box` | **True** trong lần chạy này (cờ `False` vừa thêm, chưa chạy) |
| `walk_mode` / `walk_steps` / `max_walk_liftoffs` | 1 (slowWalk) / 2 / 4 |
| `place_above` / `place_xy_tol` / `place_tilt_tol_deg` | 0.05 m / 0.04 m / 20° |
| `feet_slip_tol` | **0.04 m** |
| `slip_tol` | 0.025 m (bỏ qua khi bàn đã đỡ hộp) |
| `table_force_tol` | 120 N |
| Weld cầm | `solref=[0.01, 1.0]` |

### 2.3 Hằng số vật lý và SONIC

- Sim dt 0.005 s (200 Hz), policy 50 Hz, ghi 30 Hz.
- PD thân: kp/kd/action-scale theo `policy_parameters.hpp` (7520_22, 7520_14, 5020, 4010; cổ chân và waist roll/pitch ×2).
- Tay Dex3: kp 1.5, kd 0.1, giới hạn torque 1.4 N·m; pose đóng `RIGHT_HAND_CLOSED = [0, 0, −1.75, 1.57, 1.75, 1.57, 1.75]`.
- Túi kẹp `GRASP_OFFSET = (0.135, 0.05, 0)` trong frame `right_wrist_yaw_link` (tìm bằng thực nghiệm).
- Encoder g1-mode: 1762-D, 10 frame tương lai cách 5 tick; decoder 994-D với lịch sử 10 tick.
- Giới hạn IK bổ sung: hip pitch ≥ −1.9, gối ≤ 2.2, cổ tay pitch/yaw ±0.8 rad.

---

## 3. Kết quả 12 seed (0–11)

| Seed | Kết quả | Pha dừng | t (s) | Chân trượt (m) | Nghiêng hộp qua các vòng servo (°) |
|---|---|---|---|---|---|
| 0 | `feet_moved_before_grasp_or_walk` | GRASP | 10.6 | **0.043** | — |
| 1 | `box_not_on_table` | RETRACT | 29.8 | 0.021 | 38.5 → 22.7 → 22.5 → 21.4 |
| 2 | `place_misaligned` (xy 4.2 cm, nghiêng 8°) | PLACE | 25.0 | 0.017 | 47.6 → 29.7 → 10.7 → 13.7 |
| 3 | `grasp_slip` | PLACE | 26.9 | 0.018 | 35.7 → 21.2 → 8.1 → 11.2 |
| 4 | `place_misaligned` (xy 5.1 cm, nghiêng 32°) | PLACE | 26.1 | 0.017 | 41.6 → 53.6 → 20.5 → 30.5 |
| 5 | `feet_moved_before_grasp_or_walk` | LIFT | 11.4 | **0.052** | — |
| 6 | `box_not_on_table` | RETRACT | 29.4 | 0.019 | 39.8 → 17.7 → 10.3 → 9.8 |
| 7 | `feet_moved_before_grasp_or_walk` | LIFT | 10.7 | **0.041** | — |
| 8 | `feet_moved_before_grasp_or_walk` | REACH | 8.1 | **0.041** | — |
| 9 | `place_misaligned` (xy 5.1 cm, nghiêng 12°) | PLACE | 27.9 | 0.015 | 37.1 → 20.3 → 20.8 → 19.6 |
| 10 | `box_not_on_table` | RETRACT | 29.2 | 0.017 | 41.7 → 25.9 → 29.9 → 36.5 |
| 11 | `grasp_slip` | PLACE | 29.5 | 0.016 | 41.0 → 23.1 → 14.5 → 10.0 |

Tóm tắt: **0/12 thành công**. 8/12 đi được tới PLACE (cầm, nhấc, đứng dậy, đi 2 bước đều ổn), 4/12 rớt sớm vì chân trượt trong lúc với/nhấc.

Để so sánh, cấu hình **trước đó** (hộp 5×5×12 cm trên sàn tại (0.16, −0.34), bàn thấp 0.40 m, không có bệ) đã cho **1 episode SUCCESS** (seed 3, 737 frame, review tại `.data/box_pnp_preview/review/`), nhưng bạn đã yêu cầu đổi sang bố cục hộp trước mặt vì robot phải xoay người quá nhiều.

---

## 4. Chẩn đoán từng kiểu thất bại

### 4.1 `feet_moved_before_grasp_or_walk` (4/12) — chân trượt 4,1–5,2 cm

- Ngưỡng 4 cm; các ca thất bại chỉ vượt 0,1–1,2 cm. Các ca qua được cũng ở mức 1,5–2,1 cm, tức là **luôn có trượt** khi SONIC bám tư thế squat + với tay.
- Nguyên nhân: SONIC là policy RL bám chuyển động, không giữ chân cố định tuyệt đối. Khi tham chiếu xoay hông/eo (để với sang phải) hoặc khi trọng tâm dịch, policy tự "xoay bàn chân" trên sàn. Đã giảm bằng cách khóa hip yaw, waist yaw trong IK (trọng số 5 và 3), nhưng vẫn còn.
- Với bố cục hiện tại, hộp lệch phải 38–43°, nhiều hơn mức "trước mặt" mong muốn.

### 4.2 `place_misaligned` / `box_not_on_table` / `grasp_slip` ở PLACE (8/12) — hộp nghiêng 35–48° khi tới bàn

- Ở STAND_UP, code xoay lại bàn tay để hộp thẳng đứng (`carry_upright_box=True`). Tư thế tay mới đòi cổ tay pitch/yaw lớn, nhưng SONIC **bám cổ tay rất kém** (motor 4010, action scale 0,075): sai số hướng cổ tay đo được 17–35°.
- Kết quả: hộp treo nghiêng ~40° suốt lúc đi. Servo hướng ở PLACE kéo xuống được 10–20° sau 4 vòng nhưng không đủ; mỗi vòng thêm 0,8 s và làm episode dài 26–30 s.
- Khi hạ với hộp nghiêng: cạnh hộp chạm bàn trước → trượt trong weld (`grasp_slip`), hoặc đặt lệch, hoặc đổ sau khi thả (`box_not_on_table`).
- Bằng chứng: trong cấu hình cũ (bàn 0,40 m, giữ nguyên hướng tay lúc cầm, không xoay lại) hộp đặt được thẳng và có SUCCESS.

### 4.3 Những gì đã thử cho pha PLACE trong phiên này

| Thử | Kết quả |
|---|---|
| Bàn 0,70 → 0,60 → 0,40 → 0,55 → 0,62 m | 0,40 m đặt được nhưng gối va bàn khi đi; 0,55–0,62 nghiêng hộp |
| Hạ 12 mm → 8 mm/bước, thêm căn xy mỗi bước | ít va, vẫn nghiêng |
| Giới hạn cổ tay ±0,8 rad trong IK, `w_hand_rot` 1 → 4 | giảm nhẹ |
| Giữ khuỷu/cổ tay của tư thế mang, chỉ dùng vai/thân | tệ hơn (va bệ) |
| Servo hướng hộp (tối đa 4 vòng, 0,8·góc) | giảm 40° → 10–20°, chưa đủ |
| Weld `solref` 0,02 → 0,01 | không đổi |
| Hộp 12 → 8 → 10 cm | 8 cm: ngón tay chạm bệ, REACH thất bại toàn bộ; 10 cm OK |

---

## 5. Hướng khắc phục đề xuất (xếp theo mức ưu tiên)

1. **Không xoay lại tay sau khi cầm** (`carry_upright_box=False`, cờ đã có sẵn, chưa chạy). Giữ đúng hướng "ngón hướng xuống" lúc cầm cho suốt mang và đặt. Hộp sẽ thẳng vì lúc cầm nó thẳng. Cần bàn thấp hơn tầm tay khi đứng (thử 0,50–0,55 m) và `carry_box_pos` z ≈ 0,65–0,70. Đây là điểm khác biệt duy nhất so với lần có SUCCESS.
2. **Đưa hộp về gần trục trước hơn** (bạn yêu cầu): y ≈ −0,18 … −0,22 thay vì −0,26 … −0,29, x ≈ 0,33–0,36, bệ thu nhỏ 12×12 cm để không lấn làn chân phải (chân phải ở y = −0,12, rộng ≈ 0,1 m → làn tới y ≈ −0,17). Cần kiểm tra lại planner đi 2 bước không vướng bệ.
3. **Chân trượt**: tăng `feet_slip_tol` lên 0,06 m **chỉ nếu** video cho thấy đó là xoay tại chỗ, không phải bước; đồng thời giảm góc xoay bằng cách (2). Có thể thêm tiêu chí "không có lift-off" thay vì khoảng cách.
4. **Servo hướng**: nếu vẫn cần, chạy servo hướng **ngay sau STAND_UP** (trước khi đi), không phải lúc tới bàn, để pha PLACE ngắn và ổn định.
5. Sau khi có ≥ 5/12 SUCCESS: chạy 50 seed pilot, xem review video của 5 thành công + 5 thất bại, rồi mới sinh 500.

---

## 6. Script trace hữu ích (trong scratchpad phiên này)

- `ep_phasefail.py <seeds…>`: chạy nhiều seed, in lý do dừng, pha, chân trượt, log nghiêng hộp.
- `ep_dbg.py <seed>`: in dấu mốc mỗi lần đổi pha.
- `ep_slip.py`, `ep_coll.py`, `ep_tilt.py`, `ep_grasptrace2.py`: trace trượt kẹp, va chạm bàn/bệ, hướng cổ tay, hình học lúc cầm.
- `ep_snap2.py <seed> <t1> <t2>…`: chụp 4 góc nhìn tại các thời điểm.
- Sau khi có SUCCESS: `review_episode.py --root <out> --episode N` tạo video chú thích, timeline, contact sheet, `qa.json`.
