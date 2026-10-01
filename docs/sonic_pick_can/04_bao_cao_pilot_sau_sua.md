# Báo cáo pilot sau khi sửa bộ sinh dữ liệu `scripts/datagen/box_pnp`

> Tiếp theo [03_bao_cao_lan_chay_gan_nhat.md](03_bao_cao_lan_chay_gan_nhat.md) (0/12 SUCCESS).
> Mục tiêu phiên này: sửa hai lỗi chính (chân dịch trước grasp/walk, hộp nghiêng khi carry/place), chạy lại pilot 12 seed, và nếu đạt ≥ 5/12 thì chạy pilot 50 seed. **Chưa sinh 500 episode.**
>
> Kết quả: **pilot 12 seed (0–11): 9/12 SUCCESS**, **pilot 50 seed (0–49): 26/50 SUCCESS (52 %)**.
>
> Tái tạo (dry-run, chỉ vật lý, không render):
> `.venv-datagen/Scripts/python scripts/datagen/box_pnp/generate.py --out <dir> --dry-run --num-episodes 1 --max-attempts 1 --seed-start <seed>`
> rồi `summarize_log.py <dir>...`. Số liệu của báo cáo này nằm ở `.data/final_p12/s<seed>/meta/generation_log.jsonl` (+ `per_seed.csv`).

---

## 1. Tóm tắt

| | Lần chạy trước (03) | Phiên trước chưa báo cáo | **Hiện tại** |
|---|---|---|---|
| Pilot 12 seed (0–11) | 0/12 | 6/12 | **9/12** |
| Pilot 50 seed | – | 21/50 (seed 100–149) | **26/50 (seed 0–49)** |
| Chân dịch trước WALK (max, các ca thành công) | 1,5–2,1 cm (4 ca bị loại ở 4,1–5,2 cm) | – | median 1,7 cm, max 2,9 cm, **0 lift-off** |
| Va chạm chân–bệ khi đi | không đo (robot gần như không đi) | – | **0/50** (khe hở min 5,4 cm) |
| Sai số XY khi bàn đỡ hộp | 4–5 cm | – | median 1,1 cm, max 2,9 cm |

Hai lỗi của báo cáo 03 đã được xử lý như sau:

1. **Chân dịch trước grasp/walk**: kiểm tra mới dựa trên contact và lift-off (mục 3.4). Không còn ca nào bị loại chỉ vì vượt 4 cm một chút. 7/50 ca còn bị loại vì **cả hai chân cùng trượt** (tâm hai bàn chân dịch 4,1–4,8 cm) trong lúc đóng tay, xem mục 5.2.
2. **Hộp nghiêng khi carry/place**: `carry_upright_box=False` đã bỏ được lệnh xoay cổ tay lớn. Hộp **vẫn nghiêng khoảng 30° khi mang**, do SONIC bám hướng cổ tay kém chứ không phải do lệnh (mục 4). Đây là nguyên nhân gốc của **14/24** ca thất bại trong pilot 50 seed.

---

## 2. Cấu hình cuối

### 2.1 Scene (`scene.py::sample_params`, level 0)

| Tham số | Giá trị / phân phối |
|---|---|
| Hộp | 5 × 5 × 10 cm, `box_mass` U(0,13; 0,17) kg, `box_yaw` U(−0,15; 0,15) rad |
| `box_xy` | x U(0,33; 0,36), y U(−0,22; −0,18) (lệch phải ~28–33°, trước là 38–43°) |
| Bệ | 12 × 12 cm, cao U(0,19; 0,21) m, tâm bệ = tâm hộp + (0,025; −0,025) |
| Bàn | mép gần `table_edge_x` U(0,43; 0,46), cao `table_height` **U(0,50; 0,55)**, `table_y` U(−0,03; 0,03) |
| Điểm đặt | (mép + U(0,045; 0,06), `table_y` + U(−0,18; −0,14)) |
| Nhiễu khởi tạo | khớp 0,02 rad, yaw U(−0,03; 0,03), camera pitch/yaw ± 0,02 rad, ánh sáng / màu nhẹ |

### 2.2 Controller (`episode.py::Config`, các giá trị mặc định mới)

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Carry | `carry_upright_box` | **False** (giữ hướng cổ tay reference lúc GRASP qua LIFT → STAND_UP → WALK → PLACE) |
| | `carry_box_pos` | (0,18; −0,22; **0,67**) (x, y theo khung chân; z world) |
| | `hold_actual_wrist` / `wrist_lock_w` | False / 0 (thử nghiệm, tắt, xem mục 4.2) |
| Walk | `walk_mode` | 1 (slowWalk) |
| | `walk_steps` | **1** bước thật, sau đó planner idle và tự thêm bước khép chân, tổng 2 bước |
| | đếm bước | chỉ tính khi chân rời sàn ≥ `liftoff_min_ticks`=5 (0,1 s) |
| Chân trước WALK | `feet_slip_tol` / `feet_slip_tol_contact` / `body_shift_tol` | 0,04 / **0,06** / 0,04 m |
| | `reject_prewalk_liftoff` | True (chân rời sàn ≥ 0,1 s trước WALK → loại) |
| PLACE | `approach_time`, `place_above` | 1,4 s, 5 cm trên điểm đặt |
| | `place_max_corrections` | 2 (vị trí; sửa nghiêng tối đa 1 lần, chỉ khi 3° < tilt < 10°) |
| | `lower_step`, `lower_xy_gain` | 8 mm / 0,25 s, 0,5 |
| | `seat_depth` | **8 mm** (ấn thêm khi bàn đã đỡ hộp) |
| | `release_press_step` × `release_max_presses` | **4 mm × 3** (mất tiếp xúc trong cổng RELEASE → ấn thêm) |
| | `place_xy_tol`, `place_tilt_tol_deg` | 4 cm, 20° |
| RELEASE | cổng | hộp chạm bàn, \|vz\| < 2 cm/s, COM trong mặt bàn ≥ 2 cm, giữ 0,2 s |
| | `release_open_frac` | **0,0** (mở hẳn; 0,5 vẫn kẹp hộp 5 cm bằng ma sát) |
| RETRACT | `retract_dist` / `retract_back` / `retract_up` | 8 cm theo trục ngón / 15 cm về phía thân / **+8 cm** lên cao |
| Kiểm tra cuối | hộp nằm trên bàn ± 1 cm, v < 2 cm/s, nghiêng < 10°; robot đứng ổn định, sai khớp trung bình so với HOME < 0,12 rad |

SONIC: sim 200 Hz, policy 50 Hz, token latch **30 Hz** (giống client Ψ₀ khi deploy), ghi 30 FPS.

---

## 3. Những thay đổi so với lần trước

### 3.1 Đã có trong code khi bắt đầu phiên này (phiên trước, chưa được báo cáo)

1. `carry_upright_box=False`: không xoay lại cổ tay sau khi cầm.
2. Hộp gần trục trước hơn (y U(−0,22; −0,18), x U(0,33; 0,36)), bệ 12 × 12 cm.
3. Bàn thấp 0,50–0,55 m, carry z 0,67 m.
4. Kiểm tra chân theo contact/lift-off (`feet_slip_tol_contact`=0,06).
5. PLACE đơn giản hóa: bỏ servo hướng 4 vòng, thay bằng APPROACH_PLACE (1 lần di chuyển + ≤ 2 lần sửa nhỏ), sau đó LOWER_BOX.

Với code này (trước khi sửa trong phiên), 12 seed đạt 7/12, và cả 5 ca thất bại đều nằm ở LOWER_BOX/RELEASE.

### 3.2 Sửa trong phiên này (mỗi thay đổi được đo trên 12 seed)

| # | Thay đổi | Lỗi được sửa (bằng chứng từ trace) |
|---|---|---|
| 1 | **Đếm bước chân thật**: một lần chạm đất chỉ được tính sau khi chân rời sàn ≥ 0,1 s. Walk lift-off cũng dùng cùng ngưỡng. | Trước đây các lần contact nhấp nháy 20–40 ms khi SONIC dồn trọng tâm bị đếm là bước, nên planner idle sau 0,7 s trong khi robot đi được **−1 cm** (seed 1, 9). Khi đó tay phải với thêm ~30 cm lúc đặt. |
| 2 | **`walk_steps=1`**: idle sau bước thật đầu tiên, bước khép chân của planner là bước thứ 2. | slowWalk bước ~0,24 m. Dừng sau 2 bước thật thì bước khép chân phải đá vào mép bệ (x ≈ 0,30): **6/12 `leg_pedestal_collision`**. Hiện tại: 0/50, travel median 13,8 cm. |
| 3 | **Seat press** 8 mm khi bàn đã đỡ hộp, và ấn thêm 4 mm (tối đa 3 lần) trong cổng RELEASE nếu mất tiếp xúc. | SONIC bám tay thấp hơn reference ~1,5 cm, và vòng hạ dừng ngay khi một góc hộp chạm bàn, nên hộp **lơ lửng 1–4 mm**, contact nhấp nháy, cổng RELEASE không bao giờ thỏa (seed 7, 11). |
| 4 | **Mở hẳn tay khi RELEASE** (`release_open_frac` 0,5 → 0). | Ở mức mở 0,5, ngón vẫn kẹp hộp 5 cm bằng ma sát: hộp đi theo tay khi rút (vị trí hộp trong khung cổ tay không đổi) rồi rơi. |
| 5 | **RETRACT**: rút theo trục ngón (không theo z world), giữ nguyên hướng/khớp cổ tay reference hiện tại (chỉ tịnh tiến), lùi 15 cm và **nâng 8 cm** trước khi đưa tay về neutral. | Cổ tay lệch ~20° khỏi phương thẳng đứng nên rút theo z world kéo ngón qua hộp. Bước "về neutral" nội suy khớp làm tay quét xuống xuyên qua hộp (seed 7). |
| 6 | **Ghi dữ liệu**: thêm cột theo frame `error.box_tilt_deg`, `error.place_xy`, `error.grasp_slip`, `state.table_support`. `episode_meta` có thêm `success`/`fail_reason`. QA review có thêm tilt/XY/support. | Mục 8 của yêu cầu. |
| 7 | **Công cụ**: `generate.py --set KEY=VALUE` (override Config để A/B), `--save-failed` (video ego + third-person + contact sheet cho episode thất bại, lưu vào `<out>/failed/`). | Review failure và thử nghiệm không cần sửa code. |

Tác động đo trên seed 0–11 (các lần chạy nối tiếp): 7/12 → (1) lộ va chạm bệ → (1 + 2) 4/12, các lỗi dồn sang sau RELEASE → (+ 4) 8/12 → (+ 5, nâng tay) **9/12**.

### 3.3 Đã thử nhưng không giữ (để tránh lặp lại)

| Thử | Kết quả | Lý do bỏ |
|---|---|---|
| Giữ hướng cổ tay **thực tế** lúc GRASP + khóa wrist pitch/yaw trong IK (`hold_actual_wrist`, `wrist_lock_w=8`) | Nghiêng sau WALK giảm từ 20–40° xuống 4–23° | Chuỗi đặt/thả/rút kém ổn định: 1–4/12, sau khi sửa RETRACT vẫn chỉ 2/12. Giữ code dưới dạng tùy chọn, mặc định tắt. |
| Dịch bệ ra ngoài (mép trong = tâm hộp + 2 cm), y hộp ≤ −0,195 | Vẫn va bệ | Nguyên nhân là bước dài 0,24 m và thân lắc 6,5 cm, không phải vị trí bệ. Pha cầm còn hỏng thêm (seed 3, 4, 6, 8). Đã hoàn tác. |
| Planner `target_vel` 0,15 / 0,25 m/s; đích cụ thể 0,12 / 0,15 m | 0,15 m/s và đích cụ thể: SONIC **không nhấc chân** (walk_timeout 12/12); 0,25 m/s: vẫn va bệ | Đã bỏ. |
| Dựng hộp bằng mặt bàn (weld mềm xoay sau khi chạm + dịch tay ngang) | Không đổi / tệ hơn | Sau khi hộp tì xuống bàn, tay thực tế **không di chuyển nữa** (reference hạ 8 cm, tay chỉ hạ 0,7 cm). Góc tì nằm gần như ngay dưới điểm kẹp nên không có mô-men dựng hộp. Đã gỡ. |
| Sửa nghiêng bằng vai/khuỷu (khóa wrist, tối đa 2 lần, tới 35°) | 6/18 (không tốt hơn) | Nghiêng thường còn **tăng** sau lần sửa đầu (12→17°, 16→25°), không hội tụ. Đã gỡ. |
| Side step tay trước khi rút | Không cải thiện | Đã gỡ, vì mở hẳn tay đã đủ. |

---

## 4. Vấn đề còn lại chính: hộp nghiêng khi mang (SONIC bám cổ tay kém)

### 4.1 Nguồn gốc (đã đo)

Phân rã sai số hướng cổ tay (reference so với thực tế), seed 0 và 3:

| Checkpoint | Sai số cổ tay | Sai số torso | Wrist pitch: ref → sai số |
|---|---|---|---|
| GRASP | 26–29° | **16–18°** (waist pitch không được bám, −0,3 rad) | 0,45 → −0,5 rad |
| STAND_UP | 20–22° | 2–3° | **0,80 (giới hạn IK)** → −0,52 rad |

Hướng cần giữ được lấy từ reference lúc GRASP, mà reference đó đã chứa sẵn sai số torso ~17°. Khi đứng thẳng, sai số torso biến mất, và để giữ đúng hướng reference, IK đẩy wrist pitch lên giới hạn 0,8 rad mà SONIC không bám được. Hệ quả là hộp nghiêng median **32° (p90 45°)** sau STAND_UP và sau WALK. Lúc APPROACH, tư thế tay thay đổi và hộp giảm còn median 16°. Khi bàn đỡ, nghiêng median 12,8° (p90 20°).

### 4.2 Ảnh hưởng tới kết quả (pilot 50 seed)

- 5 ca `place_misaligned`: nghiêng 20–27° khi bàn đỡ (XY đều tốt, ≤ 2 cm).
- 9 ca hộp đổ sau khi thả (RETRACT/RETURN_HOME). Trace seed 22: hộp được thả khi còn nghiêng ~11°, một cạnh chạm bàn; khi ngón trỏ và ngón giữa duỗi ra, chúng quét vào hộp và kéo đổ (11° → 90°).
- Nghiêng lúc thả < 10°: 12 thành công / 2 đổ. Nghiêng ≥ 10°: 14 thành công / 7 đổ.

Tổng cộng **14/24** ca thất bại có gốc từ độ nghiêng này.

---

## 5. Kết quả

### 5.1 Pilot 12 seed (0–11), cấu hình cuối

| Seed | Kết quả | Pha dừng | t (s) | Chân dịch max trước WALK (cm) | Lift-off trước WALK | Nghiêng hộp GRASP / LIFT / STAND_UP / sau WALK / trước RELEASE (°) | Wrist drift sau WALK (°) | Grasp slip max (mm) | Sai số XY khi bàn đỡ (cm) | Bàn đỡ hộp |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `place_misaligned` | LOWER_BOX | 23.8 | 1.8 | 0 | 6.1 / 11.6 / 30.2 / 35.2 / – | 25.5 | 3.0 | 1.5 | có (nghiêng 22,6°) |
| 1 | SUCCESS | – | 29.6 | 2.3 | 0 | 2.4 / 6.3 / 21.9 / 28.5 / 17.9 | 29.3 | 2.7 | 1.1 | có |
| 2 | SUCCESS | – | 30.4 | 2.3 | 0 | 7.9 / 19.0 / 26.5 / 38.5 / 13.3 | 19.7 | 3.4 | 0.5 | có |
| 3 | SUCCESS | – | 30.8 | 1.8 | 0 | 12.3 / 16.6 / 35.3 / 40.6 / 6.8 | 35.4 | 2.5 | 1.7 | có |
| 4 | SUCCESS | – | 30.7 | 1.5 | 0 | 8.8 / 18.7 / 24.0 / 18.7 / 4.9 | 19.5 | 3.6 | 0.7 | có |
| 5 | SUCCESS | – | 31.9 | 1.5 | 0 | 7.9 / 16.1 / 32.3 / 27.7 / 4.5 | 16.4 | 3.6 | 1.2 | có |
| 6 | `box_not_on_table` | RETRACT | 29.2 | 3.3 | 0 | 7.5 / 14.7 / 16.8 / 19.8 / 4.7 | 16.1 | 3.4 | 0.4 | có |
| 7 | SUCCESS | – | 31.8 | 2.5 | 0 | 8.9 / 17.1 / 30.5 / 30.0 / 14.7 | 21.8 | 3.4 | 1.3 | có |
| 8 | SUCCESS | – | 32.4 | 1.7 | 0 | 13.0 / 29.1 / 27.6 / 28.6 / 19.1 | 19.9 | 3.3 | 0.6 | có |
| 9 | `box_not_stable_on_table` | RETURN_HOME | 30.4 | 1.5 | 0 | 13.4 / 16.6 / 31.9 / 41.9 / 12.2 | 33.0 | 3.7 | 0.5 | có |
| 10 | SUCCESS | – | 33.6 | 1.5 | 0 | 8.7 / 18.1 / 31.0 / 27.8 / 14.6 | 19.1 | 3.0 | 0.3 | có |
| 11 | SUCCESS | – | 30.9 | 1.6 | 0 | 8.3 / 16.6 / 37.5 / 38.5 / 1.5 | 29.7 | 3.2 | 0.2 | có |

**9/12 SUCCESS.** Grasp slip (drift của hộp trong weld) ≤ 3,7 mm ở mọi pha, so với ngưỡng 25 mm.

### 5.2 Pilot 50 seed (0–49)

**Success rate: 26/50 = 52 %.**

Phân loại thất bại theo pha:

| Pha | Lý do | Số ca | Seed |
|---|---|---|---|
| GRASP | `stance_translated_before_walk` (chân dịch 4,6–5,9 cm **và** tâm hai chân dịch 4,1–4,2 cm) | 5 | 18, 34, 38, 41, 47 |
| GRASP | `grasp_failed` (không có tiếp xúc ngón/ngón cái) | 3 | 15, 33, 37 |
| GRASP | `feet_moved_before_walk` (6,1 cm) | 1 | 48 |
| LIFT | `stance_translated_before_walk` (tâm dịch 4,8 cm) | 1 | 21 |
| LOWER_BOX | `place_misaligned` (nghiêng 20–27°, XY ≤ 2 cm) | 5 | 0, 17, 20, 25, 26 |
| RETRACT | `box_not_on_table` (hộp bị kéo khỏi bàn khi rút/mở tay) | 3 | 6, 16, 43 |
| RETURN_HOME | `box_not_stable_on_table` (hộp đổ nằm ngang, đổ từ lúc RELEASE/RETRACT) | 6 | 9, 22, 27, 29, 42, 44 |

Không có ca nào thất bại ở STAND_UP, STABILIZE_UPRIGHT, WALK hay STABILIZE_AT_TABLE: 40/40 episode đứng dậy được đều đi tới bàn.

Chỉ số chính:

| Chỉ số | Giá trị |
|---|---|
| Thời lượng (ca thành công) | 29,6–33,6 s, median 31,4 s (~896 frame @ 30 FPS) |
| Chân dịch max trước WALK (ca thành công) | median 1,7 cm, p90 2,3 cm, max 2,9 cm; **0 lift-off** ở mọi ca |
| Tỉ lệ thời gian cả hai chân chạm sàn trước WALK | 100 % trừ GRASP/LIFT (min 88–91 %, ở các ca bị loại) |
| Nghiêng hộp (median / p90) | GRASP 10 / 15° · LIFT 18 / 26° · STAND_UP 32 / 45° · sau WALK 32 / 45° · cuối APPROACH 16 / 28° · trước RELEASE 12 / 16° |
| Wrist drift so với lúc GRASP | sau WALK median 24°, p90 33° |
| Sai số XY | khi bàn đỡ: median 1,1 cm, p90 1,7 cm; cuối episode (ca thành công): median 1,7 cm, p90 2,8 cm, max 5,1 cm |
| Walk | travel median 13,8 cm, p90 21 cm; khe hở chân–bệ min 5,4 cm |
| Va chạm robot–bàn | chạm nhẹ ở REACH (7 ep), RELEASE (12), RETRACT (11), đều dưới ngưỡng 120 N |
| Thời gian tính toán | ~20 s wall / attempt (dry-run, 6 tiến trình song song trên 16 core) |

---

## 6. Review video / contact sheet

Tạo bằng `generate.py --review` (ca thành công) và `--save-failed` (ca thất bại), lưu trong `.data/final_review/`:

| Loại | Seed | File |
|---|---|---|
| Thành công | 1, 12, 30 | `review/episode_00000{0,1,2}_{review.mp4, timeline.png, phases.png, qa.json}` |
| Thất bại: hộp đổ khi mở tay | 22 | `failed/seed_000022_{ego,third}.mp4`, `_phases.png` |
| Thất bại: đặt nghiêng > 20° | 20 | `failed/seed_000020_*` |
| Thất bại: stance dịch lúc đóng tay | 18 | `failed/seed_000018_*` |

Lưu ý: render hai camera 640 × 480 giữ toàn bộ frame trong RAM (~1,7 GB mỗi episode ~900 frame). Không nên render nhiều tiến trình song song trên máy 16 GB, vì lần trước đã bị OOM.

---

## 7. Khuyến nghị

**Pilot 12 seed đạt tiêu chí** (9/12 ≥ 5/12), nên đã chạy pilot 50 seed. Với 50 seed, success rate 52 % và các lỗi **không dồn vào một chỗ duy nhất**. Tuy vậy, vẫn còn **một nguyên nhân hệ thống**: độ nghiêng hộp do SONIC bám cổ tay kém, chiếm 14/24 ca thất bại. Ngay cả các ca thành công cũng **mang hộp nghiêng ~30°** khi đi.

**Chưa nên sinh 500 episode ngay.** Đề xuất theo thứ tự:

1. **Quyết định ngưỡng "stance translate"** (7/50 ca bị loại). Tâm hai chân dịch 4,1–4,8 cm, chỉ nhỉnh hơn ngưỡng `body_shift_tol`=0,04 m, không có lift-off, và cả hai chân vẫn chạm sàn. Nới lên 0,05 m có thể cứu ~6 ca. Nhưng đây là robot trượt nguyên cả người chứ không chỉ xoay chân, nên cần bạn quyết định có chấp nhận cho dữ liệu hay không.
2. **Giảm nghiêng hộp ở gốc**: dựng reference lúc cầm (REACH/GRASP) với waist pitch ≈ 0 và wrist pitch nhỏ, để reference khớp với những gì SONIC thực hiện được. Khi đó "giữ hướng lúc GRASP" sẽ giữ được hướng thực tế. Thay đổi này đụng tới REACH/GRASP (đang ổn), nên cần pilot riêng.
3. Nếu chấp nhận hiện trạng, có thể sinh dữ liệu với hiệu suất ~50 %: 500 episode cần ~1000 attempt, tức ~1 giờ vật lý với 6 tiến trình, cộng thời gian render tuần tự. Mọi episode được ghi đều đã qua kiểm tra nghiêm ngặt (hộp nằm phẳng trên bàn, lệch XY < 4 cm, robot về HOME ổn định). Nhưng hộp nghiêng khi mang là một đặc trưng mà policy Ψ₀ sẽ học theo.
