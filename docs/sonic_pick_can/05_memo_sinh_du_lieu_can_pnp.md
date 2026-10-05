# Memo: sinh dữ liệu `can_pnp` (SONIC 1.1)

Ngày 2026-10-04

## Tóm tắt

Code đã chạy được với SONIC 1.1, nhưng **chưa nên sinh dữ liệu hàng loạt**. Pilot 50 seed chỉ giữ được 5 episode,
và trong các lần thử, gần như episode nào cũng có lon cọ vào gối hoặc đùi robot.

- Tỉ lệ thấp không phải vấn đề lớn: sinh 140 episode mất khoảng 1–1,5 giờ máy.
- Vấn đề chính là chất lượng: dạy AI bằng dữ liệu này thì AI sẽ học cả kiểu cọ lon vào chân.
- Đã thử 7 phương án. Phương án **D** giảm thời gian lon chạm chân khoảng 88% và lon không còn chạm khi đứng dậy.
  Lon vẫn còn chạm gối lúc robot vừa nâng người lên.
- Giá trị mặc định trong code chưa đổi. Các phương án chỉ được thử qua tham số.

## Pilot 50 seed: 5 thành công

Pilot chạy 50 seed mới (500000–500049), chỉ mô phỏng vật lý, 5 tiến trình, mất 3,6 phút.
36/45 ca hỏng xảy ra lúc robot ngồi xổm với tay xuống sàn.

| Lý do bị loại | Số ca | Ghi chú |
| --- | --- | --- |
| Chân trượt quá 6 cm | 23 | Phần lớn chỉ vượt nhẹ, khoảng 6,7 cm |
| Cả người dịch quá 4 cm | 7 | Dịch 4,2–4,7 cm |
| Chân đá vào lon trước khi gắp | 5 | |
| Hỏng sau khi đứng dậy | 7 | Lon rơi khỏi khay (3), mất thăng bằng (2), va bàn (2) |
| Gắp trượt | 2 | |
| Nhấc chân trước khi đi | 1 | |

Ngoài chuyện chạm chân, khi bàn đỡ lon thì lon còn nghiêng khoảng 22° (median, đo trên các ca đi được tới bước đặt).
Ngưỡng chân trượt được giữ nguyên để dữ liệu sạch.

## Vì sao lon chạm chân

- Khi robot ngồi xổm, lon nằm ngay trước **đầu gối phải**: lệch sang phải khoảng 21 cm, đúng vị trí đầu gối.
- Code cũ nhấc lon thẳng lên và kéo lùi 7 cm về phía người, nên lon va vào gối.
- Khi đứng dậy, lon được giữ ở vị trí lệch phải 22 cm, chỉ cách đùi 1–2 cm. Tay robot hơi xệ vào trong nên lon cọ đùi.
- Trong 11 seed qua được bước gắp, cả 11 seed đều có lon chạm chân. Tổng thời gian chạm khoảng 22 s
  (1087 nhịp × 20 ms).

## Các phương án đã thử

Thử trên cùng 11 seed (những seed qua được bước gắp trong pilot). Mẫu nhỏ nên chênh 1 ca thành công chưa có ý nghĩa.

| Phương án | Thay đổi | Thành công | Số ca chạm chân | Thời gian chạm | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| Gốc | Nhấc lên, kéo lùi 7 cm | 5/11 | 11/11 | 21,7 s | |
| A | Đưa lon ra ngoài 5 cm, không kéo lùi | 5/11 | 7/11 | 3,1 s | Còn chạm khi đứng dậy |
| B | Đưa ra ngoài 8 cm | 1/11 | 1/11 | 0,1 s | Robot ngã 4 lần |
| C | Ra ngoài 5 cm, vẫn kéo lùi 3 cm | 5/11 | 10/11 | 6,9 s | Kéo lùi là sai hướng |
| **D** | **A + giữ lon xa đùi thêm 5 cm khi đứng** | **4/11** | **5/11** | **2,6 s** | **Hết chạm khi đứng dậy** |
| E | D, nhưng đưa ra xa hơn (6 cm, thêm 5 cm khi nâng người) | 2/11 | 2/11 | 0,2 s | Robot ngã 3 lần |
| F | D + nâng tay cao hơn khi nâng người | 4/11 | 6/11 | 4,7 s | Chạm gối nhiều hơn |
| G | F + nhấc lon cao hơn ngay từ đầu | 4/11 | 5/11 | 3,7 s | Không tốt hơn D |

Kết luận:

- Đưa lon ra ngoài là đúng hướng. Đưa xa quá (B, E) thì robot mất thăng bằng.
- D tốt nhất: chỗ chạm còn lại chỉ ở pha nâng người khỏi tư thế ngồi xổm, chủ yếu ở 2 seed.
- Giữ lon xa đùi (D, F) có thể gây thêm ca va bàn. Gặp 1–2 ca, cần theo dõi.

## Cập nhật 2026-10-05: D làm mặc định + luật loại khi lon chạm thân

Từ 2026-10-05, D là cấu hình mặc định. Thêm luật mới: lon đang cầm mà chạm bất kỳ chỗ nào trên người robot
(trừ bàn tay) thì loại episode (`held_can_touched_body`).
Pilot 50 seed mới (600000–600049): **4/50 thành công, cả 4 không có lần nào lon chạm thân**.

- 12 seed qua được bước gắp: 4 thành công, 4 bị loại vì lon chạm gối (3 ca lúc nâng người, 1 ca lúc nhấc lon).
- 30 ca vẫn hỏng vì chân trượt lúc với tay xuống sàn (giống trước).
- Lon vẫn nghiêng khi đặt: median 26°.
- Ước tính: 140 episode cần khoảng 1750 seed, tức khoảng 1,5 giờ với 6 tiến trình, chưa tính render.

## Việc tiếp theo

- [x] Chốt D làm cấu hình mặc định trong `episode.py`.
- [x] Thêm luật loại episode khi lon chạm thân robot.
- [ ] Xử lý chỗ lon còn chạm gối lúc nâng người. Hướng thử: đưa lon ra ngoài trước rồi mới nâng người.
- [x] Lon nghiêng khi đặt: **chấp nhận** (quyết định 2026-10-05). Chỉ yêu cầu lon nằm trong khay, không rơi
      ra ngoài; lon đổ nằm trong khay vẫn tính là đạt. Đã thử đặt lon tựa vào thành khay để lon không đổ, nhưng
      lon nghiêng về phía robot nên lật qua thành và rơi khỏi bàn. Đã bỏ cách này.
- [ ] Commit code trước khi sinh dữ liệu thật.
- [ ] Trước khi finetune: `export_psi0.py` cần thêm 2 chiều cổ để dùng được checkpoint Ψ₀ cho SONIC 1.1.

## Ghi chú kỹ thuật

**Tham số mới trong `scripts/datagen/can_pnp/episode.py`** (mặc định theo cấu hình D):

- `lift_out` = 0,05: khi nhấc lon, đưa lon ra phía ngoài bên phải (m).
- `raise_out` = 0,03: khi nâng người, đưa lon ra ngoài thêm (m).
- `reject_held_can_body_contact` = True: loại episode khi lon đang cầm chạm thân robot.

**Quay lại cấu hình cũ** (trước D):

```bash
--set lift_out=0.0 --set lift_pullback=0.07 --set raise_out=0.0 --set "carry_box_pos=(0.18,-0.22,0.67)" \
--set reject_held_can_body_contact=False
```

**Lệnh chạy pilot** (dry-run, 10 seed mỗi tiến trình):

```bash
SONIC_VARIANT=sonic_v1_1 .venv-datagen/bin/python scripts/datagen/can_pnp/generate.py \
    --out .data/can_pilot50/p0 --dry-run --num-episodes 10 --max-attempts 10 --seed-start 500000 --threads 3
```

**Giới hạn RAM:** mỗi tiến trình mô phỏng tốn khoảng 2,2 GB. Trên máy 31 GB, **chạy tối đa 6 tiến trình cùng lúc**.
Chạy 12 tiến trình đã làm máy hết RAM và hệ thống phải tắt các tab Chrome.

**Dữ liệu:**

- Kết quả pilot: `.data/can_pilot50/` (`p0`–`p4`, log ở `logs/`).
- Script đo lon chạm chân và kết quả các phương án nằm trong thư mục tạm của phiên làm việc, chưa đưa vào repo.
