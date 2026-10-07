# Memo: sinh dữ liệu `can_pnp` (SONIC 1.1)

Cập nhật 2026-10-07

## Tóm tắt

Nhiệm vụ đã đổi: **lon đứng trên bục cao 20 cm**, robot **ngồi xổm** và **ôm lon bằng cả bàn tay** (trước đây lon
đặt dưới sàn và tay kẹp miệng lon bằng 2 ngón). Pilot mới nhất trên 50 seed mới đạt **21/50 (42%)**.

- So với thiết kế cũ (lon dưới sàn: 4/50), tỉ lệ đạt cao gấp khoảng 5 lần.
- **Lỗi lon chạm gối/đùi lúc nhấc đã hết** (0/50, trước đó 8/50). Nguyên nhân là bước nhấc dịch cả hông; nay chỉ tay cử động.
- **Không còn ca nào bị loại vì chân trượt khi với tay** (thiết kế cũ: 30/50 ca).
- Lỗi lớn nhất còn lại: gắp hụt (17/50 tay dừng thiếu, 8/50 khép tay không bắt được lon).
- **Đã sinh xong bộ dữ liệu 144 episode**: `.data/can_ped_144` (xem mục "Bộ dữ liệu"), từ commit `463d74e` trên nhánh
  `can-pnp-datagen-sonic11`.

## Các quyết định

| Ngày | Quyết định |
| --- | --- |
| 2026-10-05 | Lon đổ nằm trong khay vẫn tính là đạt. Chỉ cần lon nằm trong khay, không rơi ra ngoài. |
| 2026-10-05 | Khi ngồi xuống, hai tay giữ nguyên tư thế ban đầu. Tay chỉ cử động sau khi thân đã hạ xong. |
| 2026-10-05 | Gắp bằng cả bàn tay (3 ngón ôm thân lon), không kẹp bằng đầu ngón. |
| 2026-10-05 | Lon đặt trên bục cao 20 cm, robot ngồi xổm. Không dùng tư thế quỳ một gối. |
| 2026-10-05 | Dừng đợt sinh dữ liệu theo thiết kế cũ. 32 episode đã sinh (`.data/can_gen_s*`) không dùng nữa (đã xoá 2026-10-06, cùng các dữ liệu lon dưới sàn khác). |

## Vì sao phải đặt lon lên bục

Đã thử cho robot **quỳ một gối** (planner SONIC mode 6) để hạ thấp người:

- Quỳ được và đứng dậy lại được, gối phải chạm sàn. Hông chỉ thấp hơn ngồi xổm 4–5 cm (0,48 m so với 0,52 m).
- Độ cao mục tiêu 0,30–0,40 m cho cùng một tư thế; 0,45 m thì robot đi lung tung khoảng 50 cm; không đặt độ cao thì
  robot ngã. Quỳ hai gối (mode 5) cũng ngã.

Dù quỳ hay ngồi xổm, **tay không xuống đủ thấp để ôm thân lon dưới sàn**. Điểm thấp nhất tay đạt được tại chỗ lon
là khoảng 13 cm; ôm thân lon cần 7–10 cm. Nguyên nhân: SONIC chỉ hạ hông tới 0,48–0,52 m, thân trên cúi có giới hạn,
cổ tay chỉ bẻ được ±46° (bộ giải IK giới hạn ở ±0,8 rad vì SONIC không bám được góc lớn hơn).

Với lon trên bục, độ cao điểm gắp mà tay ôm được (tính hình học, chưa chạy vật lý):

| Tư thế | Hông cao | Tay ôm lon được khi điểm gắp cao từ |
| --- | --- | --- |
| Đứng thẳng | 0,77 m | trên 45 cm |
| Khuỵu nửa chừng | 0,65 m | 30 cm |
| Ngồi xổm | 0,53 m | 20–25 cm |

## Thiết kế hiện tại

- **Bục:** 12 × 12 cm, cao 0,19–0,21 m. Lon đặt ở (0,27; −0,30) m, dao động ±1,5 cm.
  - Vị trí này để chân cách bục và lon ít nhất 4,5 cm suốt lúc ngồi xổm.
  - Ở vị trí thử đầu tiên, (0,25; −0,21) m, gối và ống chân phải va vào bục ở 5/6 seed.
- **Ngồi xổm:** giữ như cũ. Hai tay khoá ở tư thế ban đầu cho tới khi ngồi xong.
- **Hướng bàn tay khi gắp:** lòng bàn tay quay về phía trước bên trái, các ngón chĩa sang phải 60° và nghiêng xuống 25°.
  Lon nằm trong lòng tay, cách cổ tay 11,5 cm theo hướng ngón và 4,5 cm theo hướng lòng bàn tay.
- **Cách tay tiếp cận lon:**
  1. Đưa tay tới điểm chờ, cách lon 8 cm về phía lòng bàn tay và cao hơn 3 cm. Chỉnh tối đa 12 lần, tới khi lệch dưới 3 cm.
  2. Đẩy tay vào lon qua 4 nấc.
  3. Chỉnh thêm tối đa 3 lần. Mỗi lần được đẩy vào thêm tối đa 2 cm, nhưng chỉ khi tay chưa chạm lon.
- **Ngón cái khi tiếp cận:** bẻ ra sau (khớp `thumb_1` = 0,5 rad) để không chặn lon; khi khép tay mới quay về ôm lon.
- **Nhấc lon:** nhấc lên 9 cm và đưa ra xa gối 8 cm. **Hông giữ nguyên** (độ cao và trọng tâm), chỉ cánh tay cử động.
- **Thả lon:** mở tay rồi rút tay ra theo hướng ngược với lòng bàn tay, hơi chếch lên.

## Pilot mới nhất: 21/50

Seed 900000–900049 (chưa dùng ở lần thử nào trước đó), chỉ mô phỏng vật lý.

| Lý do bị loại | Pha | Số ca | Ghi chú |
| --- | --- | --- | --- |
| Gắp hụt sau khi đẩy tay vào | REACH | 17 | Tay dừng thiếu 3–7 cm, hoặc lon bị xô lệch 8–17 cm |
| Khép tay nhưng ngón không bắt được lon | GRASP | 8 | Lon đã nằm đúng lòng bàn tay (lệch < 2 cm) |
| Không tới được điểm chờ | REACH | 2 | |
| Cả người dịch quá 4 cm | REACH | 1 | |
| Va mạnh vào bàn | WALK | 1 | |

Hai lỗi đã sửa trong lần này:

| Lỗi | Nguyên nhân | Cách sửa | Kết quả |
| --- | --- | --- | --- |
| Lon chạm gối/đùi lúc nhấc | Bộ giải IK của bước nhấc dời hông ra trước 5–12 cm, hạ xuống 4–8 cm. SONIC bám theo nên tay kéo lon lùi về phía người, đè lên gối | Khoá độ cao hông và giữ trọng tâm trên hai bàn chân khi nhấc (`lift_hold_pelvis`) | 8/50 → 0/50 |
| Ngón cái chặn lon | Tay mở thẳng thì ngón cái thò ra phía lòng bàn tay, lon chạm ngón cái khi còn cách lòng bàn tay 4,5–7 cm (11/17 ca gắp hụt) | Bẻ ngón cái ra sau khi tiếp cận (`power_preshape`) | 18/50 → 21/50 |

So sánh trên cùng 50 seed: chỉ sửa bước nhấc 18/50; sửa cả hai 21/50.

Số liệu chính:

- Episode thành công dài khoảng 1450–1590 khung hình (khoảng 50 s).
- Ước tính: 140 episode cần khoảng 330 seed, tức khoảng 35–40 phút với 5 tiến trình, chưa tính render.
- Trong pilot trước (20%), 8/10 episode đạt có lon đứng thẳng trong khay; pilot mới chưa đo lại chỉ số này.

## Bộ dữ liệu `.data/can_ped_144` (2026-10-06)

| Mục | Giá trị |
| --- | --- |
| Số episode | 144 (6 tiến trình × 24), mỗi episode một seed khác nhau |
| Số lần thử | 394, tức tỉ lệ đạt 37% |
| Thời gian sinh | khoảng 85 phút (20:25 → 21:55), không lần nào hết RAM |
| Tổng khung hình | 214.297 khung hình @ 30 FPS (khoảng 119 phút) |
| Độ dài episode | 45–56 s, trung vị 49,5 s |
| Lon cuối episode | 131 đứng thẳng (< 15°), 13 đổ nằm trong khay |
| Phiên bản SONIC | 1.1 |
| Dung lượng | 3,5 GB |
| Code | commit `463d74e`, nhánh `can-pnp-datagen-sonic11` |

Kiểm tra: action 78 chiều, state 43 chiều, không có NaN; số khung hình video bằng số dòng parquet ở cả 144 episode;
chỉ số episode liên tục 0–143.

Lý do loại trong 250 lần thử hỏng: gắp hụt 147, khép tay không bắt được lon 30, nhấc hụt 18, không tới được điểm chờ 16,
chân hoặc người dịch quá ngưỡng 19, lon chạm người lúc nhấc 7, lon rơi khỏi khay khi rút tay 7, các lỗi khác 6.

Các thư mục shard `.data/can_ped_s0`–`s5` đã xoá sau khi gộp và kiểm tra (2026-10-06). Log sinh dữ liệu còn trong
`.data/can_ped_logs/` và `meta/generation_log.jsonl` của bộ gộp.

## Chuẩn bị finetune Ψ₀ với SONIC 1.1 (2026-10-07)

Quyết định: train tiếp từ checkpoint Ψ₀ cho SONIC 1.1 (`postpre.sonic1.1.unifolm.2609181726.40k`), không dùng `sonic-wbcbox`,
vì `sonic-wbcbox` học token SONIC 1.0 còn dữ liệu sinh bằng SONIC 1.1.

**Không cần thêm 2 chiều cổ khi export.** Đọc code Ψ₀ bản phát hành checkpoint này (commit `2830f93`) thì thấy:

- Checkpoint được post-train trên dữ liệu UniFoLM G1, cũng **không có cổ**: action = token 64 + tay 14, state 43 theo
  thứ tự khớp 0810, ảnh 320×240. Đây đúng là định dạng `export_psi0.py` đang xuất.
- Trình nạp dữ liệu tự đệm 0 lên action 80 / state 45 (`pad-action-dim=80`, `pad-state-dim=45`). Hai chiều action đệm
  không tính vào loss.
- Đã đọc phần đầu file `action_header.safetensors` trên HF: action 80, state 45, 12 khối, có token null cho state, mã
  hoá câu lệnh CLIP 768 chiều, `qk-norm`, chuỗi action dài 30. Notebook kiểm tra lại các con số này trước khi train.

Những gì đã làm:

| Việc | Chi tiết |
| --- | --- |
| Notebook mới | `scripts/datagen/can_pnp/colab_finetune_psi0_sonic11.ipynb`: code Ψ₀ `2830f93`, warm-start SONIC 1.1, tham số model theo công thức 1.1 của Ψ₀ (12 khối, `combined-temb` + CLIP, `state-as-action-token`, `state-null-token`, `qk-norm`). VLM đóng băng, chỉ train action header. |
| Câu lệnh của task | Sửa từ "pick up the can next to your right foot" thành "pick up the can from the low stand on your right" trong `episode.py` và trong meta của `can_ped_144`. Đánh giá dùng chung câu lệnh này. |
| Client đánh giá | `closed_loop_eval.py` nhận action ≥ 78 chiều và lấy 78 chiều đầu (model mới xuất 80). |
| Export | `.data/psi0/can_ped_144_train` (130 episode), `can_ped_144_val` (14 episode), `can_ped_144_psi0.zip`. |

Lưu ý khi train trên Colab:

- Dùng GPU G4, H100 hoặc A100. Header 12 khối làm bộ nhớ và checkpoint lớn hơn bản `box_pnp` (ước tính 15–17 GB mỗi
  checkpoint, cần Drive trống ≥ 40 GB). Smoke test (cell 8) đo con số thật.
- `uv.lock` ở commit `2830f93` vẫn lỗi (khoá trùng). Notebook dùng lại file lock của commit `47993ee` như bản cũ; bản mới chỉ
  thêm `decord2` và `wandb[media]`, cả hai không dùng khi train.
- Server đánh giá cũng phải chạy code `2830f93`. Notebook server của `box_pnp` vẫn dùng `4f3720d`, chưa cập nhật.
- Token SONIC 1.1 trong dữ liệu nằm trong [−0,75; 0,6875] (lưới 1/16); 0,7% giá trị vượt ±0,625. `closed_loop_eval.py` chỉ
  kẹp về ±0,625 khi bật `--clip-token` (mặc định tắt). Client trên robot thật của Ψ₀ có kẹp: cần kiểm tra lại khoảng kẹp
  cho SONIC 1.1 trước khi deploy.

## Các phương án gắp đã thử

Phần lớn phương án thử trên 6–18 seed. Kết quả phụ thuộc nhiều vào seed: cùng một phương án có thể đạt 6/18 trên nhóm
seed này nhưng 0/12 trên nhóm khác. Bảng dưới chỉ để biết hướng nào không nên thử lại.

| Thay đổi | Kết quả | Kết luận |
| --- | --- | --- |
| Hướng tay −75° (tay ở sau lon) | Điểm chờ nằm cạnh gối phải, tay lệch tới 7 cm | Bỏ |
| Hướng tay −15° / −45° (tay ở bên phải lon) | Vươn tay xa, chân trượt 6–8 cm | Bỏ |
| **Hướng tay −60°** | 3/6 seed chạy trọn vẹn ở lần thử đầu | **Giữ** |
| Bẻ ngón cái ra sau khi tiếp cận (lúc bước nhấc còn lỗi) | Không tốt hơn trên 18 seed | Thử lại sau, xem dòng cuối bảng |
| Đưa tay lên cao 10 cm trước khi tới điểm chờ | Chân trượt 6–10 cm | Bỏ |
| Đẩy tay vào có đo và chỉnh liên tục | Xô lon khỏi bục (6/18) | Bỏ |
| Đẩy tay vào theo vị trí thật của lon trong tay | Khoảng 6/30 | Giữ làm tuỳ chọn `slide_mode="pocket"` |
| Nhấc lon chếch về trước 4 cm | Chân trượt (6/18) | Bỏ |
| Ngồi xổm nông hơn (hông 0,58 m) | Chân trượt nhiều hơn (6/12) | Bỏ |
| Dời bục ra trước 3 cm | Lon chạm người nhiều hơn (5/12) | Bỏ |
| Dời điểm giữ lon ra phía đầu ngón 1,5 cm | Lon bị hất xa hơn (16–66 cm) | Bỏ |
| **Khoá hông khi nhấc lon** | 7/8 seed từng bị lon chạm gối chạy trọn vẹn | **Giữ** |
| **Bẻ ngón cái ra sau khi tiếp cận** (cùng với khoá hông) | 21/50 so với 18/50 | **Giữ** |

## Việc tiếp theo

- [ ] Xem lại video thiết kế mới, xác nhận chuyển động ổn (`.data/xem_video/moi_om_lon_tren_buc_*.mp4`).
- [x] Sửa lỗi lon chạm đùi/gối lúc nhấc (8/50 → 0/50).
- [x] Sửa lỗi ngón cái chặn lon.
- [ ] Giảm lỗi gắp hụt còn lại (17/50) và lỗi khép tay không bắt được lon (8/50).
- [x] Commit code thiết kế mới (`463d74e`).
- [x] Sinh 144 episode: `.data/can_ped_144`.
- [x] Export sang định dạng Ψ₀: `.data/psi0/can_ped_144_psi0.zip`.
- [x] Notebook finetune SONIC 1.1: `scripts/datagen/can_pnp/colab_finetune_psi0_sonic11.ipynb`.
- [ ] Tải zip lên `My Drive/psi0/`, chạy smoke test rồi train trên Colab.
- [ ] Cập nhật notebook server đánh giá sang code Ψ₀ `2830f93`.

## Ghi chú kỹ thuật

**Cảnh** (`scripts/datagen/can_pnp/scene.py`):

- `PEDESTAL_HEIGHT = 0.20`: đặt bằng 0 để quay về lon dưới sàn.
- `PEDESTAL_CAN_XY = (0.27, -0.30)`: vị trí danh định của lon trên bục.

**Tham số trong `Config`** (`scripts/datagen/can_pnp/episode.py`):

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `grasp_style` | `"power"` | `"power"`: ôm cả bàn tay; `"pinch"`: kẹp miệng lon (thiết kế cũ) |
| `lock_arms_crouch` | True | Khoá hai tay ở tư thế ban đầu khi ngồi xổm |
| `power_heading_deg` / `power_tilt_deg` | −60 / 25 | Hướng các ngón và độ nghiêng bàn tay khi gắp |
| `power_below_rim` | 0,05 m | Tâm lòng bàn tay nằm dưới miệng lon bao nhiêu |
| `power_offset_x` | 0,115 m | Vị trí lon trong lòng tay theo hướng ngón |
| `pregrasp_side` / `pregrasp_up` | 0,08 / 0,03 m | Điểm chờ: cách lon về phía lòng bàn tay / cao hơn |
| `slide_mode` | `"open"` | Cách đẩy tay vào lon |
| `power_preshape` | `thumb_1` = 0,5 | Dạng tay khi tiếp cận: ngón cái bẻ ra sau |
| `lift_out` | 0,08 m | Khi nhấc, đưa lon ra xa gối |
| `lift_hold_pelvis` | True | Khi nhấc, giữ độ cao hông và trọng tâm, chỉ tay cử động |
| `reject_held_can_body_contact` | True | Loại episode khi lon đang cầm chạm thân robot |

Ngoài ra, robot chạm vào bục (trừ bàn tay) cũng bị loại episode (`robot_pedestal_collision`).

**Lệnh chạy pilot** (dry-run, 10 seed mỗi tiến trình):

```bash
SONIC_VARIANT=sonic_v1_1 .venv-datagen/bin/python scripts/datagen/can_pnp/generate.py \
    --out .data/can_ped_pilot50/p0 --dry-run --num-episodes 10 --max-attempts 10 --seed-start 800000 --threads 3
```

**Giới hạn RAM:** mỗi tiến trình tốn khoảng 2,2–2,8 GB. Trên máy 31 GB, **chạy tối đa 6 tiến trình cùng lúc**.
Chạy 12 tiến trình đã làm máy hết RAM và hệ thống phải tắt các tab Chrome.

**Dữ liệu:**

- Pilot thiết kế mới (trước khi sửa 2 lỗi): `.data/can_ped_pilot50/`. Pilot mới nhất chạy bằng script thử nghiệm,
  không ghi vào `.data/`.
- Video xem nhanh: `.data/xem_video/`.
- Các script thử nghiệm (đo tầm với, khoảng cách chân–bục, vị trí lon trong tay) nằm trong thư mục tạm của phiên làm
  việc, chưa đưa vào repo.

## Lịch sử: lon đặt dưới sàn (thiết kế cũ, 2026-10-04 → 2026-10-05)

- Pilot đầu tiên: 5/50. Phần lớn hỏng vì chân trượt lúc với tay xuống sàn (30/45 ca hỏng).
- Gần như mọi episode có lon cọ vào gối hoặc đùi phải: lúc ngồi xổm, lon nằm ngay trước gối phải, và code cũ nhấc
  lon thẳng lên rồi kéo lùi về phía người.
- Đã sửa bằng cách đưa lon ra phía ngoài khi nhấc và giữ lon xa đùi hơn khi đứng dậy. Thời gian lon chạm chân giảm
  khoảng 88%. Thêm luật loại episode khi lon chạm thân. Pilot sau đó: 4/50, không còn lon chạm thân.
- Lon nghiêng khoảng 25° khi đặt và 13/14 lần đổ nằm trong khay. Đã thử đặt lon tựa vào thành khay nhưng lon lật qua
  thành và rơi khỏi bàn, nên đã bỏ cách này.
- Đã commit trên nhánh `can-pnp-datagen-sonic11` (`3cde088`, `72c4f4b`).
