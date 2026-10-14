# Memo: task `can_nav_pnp`: đi vòng vật cản rồi nhặt lon (SONIC 1.1, Ψ₀ + GR00T N1.7)

Cập nhật 2026-10-13

## Trạng thái hiện tại (2026-10-13)

| Việc | Trạng thái |
| --- | --- |
| Dữ liệu `can_nav_160` (160 episode, vật cản ở giữa, đi trái) | ✅ xong, đã export cho Ψ₀ và GR00T |
| **Train Ψ₀** trên `can_nav_160` (run `cannav160-s11`) | ✅ xong 8 000 bước trên G4, 1,04 s/bước (khoảng 2 giờ 20 phút); loss 1,71 ở bước 1 696 |
| Train GR00T N1.7 trên `can_nav_160` | chưa ghi nhận kết quả (notebook sẵn sàng) |
| **Đánh giá vòng kín Ψ₀ ckpt 8000** | 🔄 **đang chạy trên Colab** (cell 6c, `cannav160-s11_ckpt8000_center_colab`, 10 cảnh từ seed 1 000 000, `--obstacle center`), sau khi sửa các lỗi ở mục "Server Ψ₀ và đánh giá trên Colab" |
| Kết quả đánh giá hợp lệ tới giờ (chạy local) | seed 1000000 ✅ 77,7 s; 1000001 ❌ hết giờ (không nhặt được lon); 1000002 ✅ 61,3 s. Thêm một lần seed 1000000 đặt được lon vào khay nhưng bị chấm sai (lỗi đã sửa) |
| Sinh thêm 340 episode (vật cản ngẫu nhiên, `--level 1`) trên Colab | chưa chạy (gói `can_nav_datagen.zip` + `colab_datagen.ipynb` sẵn sàng) |

Đang làm: chờ cell 6c chạy xong 10 cảnh. Trước đó đã dừng lần chạy chậm, xoá cell "sửa nhanh" cũ (có `pkill` server) và bật EGL
trên GPU NVIDIA (xem mục dưới).

## Tóm tắt

- Task mới thêm một đoạn đi bộ vòng qua vật cản trước task `can_pnp`. Phần nhặt lon trên bục và đặt vào khay giữ nguyên.
  Code nằm trong `scripts/datagen/can_nav_pnp/`, chép từ `can_pnp`. `can_pnp` không đổi, nên vẫn sinh lại được `can_ped_144`.
- **Pilot cuối: 14/18 (78%)** episode chạy trọn vẹn, so với 37% của `can_pnp`. Không ca nào hỏng ở đoạn đi bộ. Episode dài
  67–83 s (khoảng 2 000–2 450 khung).
- Một bộ dữ liệu dùng cho cả hai model:
  - Ψ₀: `export_psi0.py`, định dạng như cũ.
  - GR00T N1.7: `export_gr00t.py`, embodiment `unitree_g1_sonic`, action là token SONIC + tay (cách 1 trong memo 05).
  - Mỗi model có notebook Colab riêng. `closed_loop_eval.py` có thêm client GR00T (ZMQ).
- **Đã thêm camera cổ tay phải** (`right_wrist_view`). Episode đã sinh được render bù bằng `render_views.py --views wrist`, không cần mô phỏng lại.
- **Đã kiểm tra code Ψ₀ (`2830f93`): Ψ₀ nhận được nhiều camera** và checkpoint warm-start vẫn nạp được.
- **Đã sinh xong `.data/can_nav_160`**: 160 episode, 344 090 khung (3,2 giờ), 5,2 GB, đã kiểm tra, không lỗi.
- **Mục tiêu nâng lên 500 episode.** 340 episode còn lại dùng thiết kế mới:
  - Vật cản lệch ngẫu nhiên, đổi kích thước và màu. Robot đi vòng **bên rộng hơn**, câu lệnh không đổi.
  - `--level 1`.
  - Sinh trên Colab bằng `run_blocks.py` + `colab_datagen.ipynb`. Pilot: 18/40 (45%).
- **Mỗi model chỉ dùng 1 camera (camera đầu).** Camera cổ tay vẫn có trong code nhưng mặc định tắt.
- **SONIC 1.1 không ngồi xổm hẳn được:** hông thấp nhất 0,42–0,47 m; ra lệnh sâu hơn thì gót nhấc khỏi sàn (mục "SONIC 1.1 ngồi xổm sâu được đến đâu").

## Các quyết định

| Ngày | Quyết định |
| --- | --- |
| 2026-10-09 | Lon vẫn đặt trên bục thấp. Vật cản cố định, chỉ dao động nhỏ. GR00T học token SONIC + tay, giống Ψ₀. |
| 2026-10-09 | Robot đi vòng bên trái vật cản và luôn nhìn về +x, bước chéo hoặc ngang. |
| 2026-10-09 | Bục, lon, bàn và khay được dời theo chỗ robot thật sự dừng (`_relayout`), không ép robot dừng chính xác. |
| 2026-10-09 | Sau khi đi, robot bước từng chân về tư thế đứng chuẩn rồi mới ngồi xổm (`STEP_ADJUST`). |
| 2026-10-09 | Thêm camera cổ tay phải cho cả hai model. |
| 2026-10-09 | Không lật video để làm giàu dữ liệu (lý do ở mục "Làm giàu dữ liệu"). |
| 2026-10-09 | Cả Ψ₀ và GR00T chỉ dùng 1 camera (đầu). Video cổ tay mặc định tắt (`generate.py --wrist-video` để bật). |
| 2026-10-09 | Tổng 500 episode; 340 episode mới dùng `--level 1`. |
| 2026-10-10 | Train Ψ₀ và GR00T: chỉ đặt giống nhau số bước (8 000) và batch hiệu dụng (128); mọi thiết lập khác giữ mặc định từng model. |
| 2026-10-13 | Đánh giá vòng kín chạy ngay trên Colab (mô phỏng gọi server qua `localhost`), không qua tunnel. |
| 2026-10-13 | Khi đánh giá, lon đổ nằm trong khay vẫn tính là đạt (giống quy ước dữ liệu); chỉ tính "lon đổ" là thất bại khi lon ngoài khay. |
| 2026-10-09 | Vật cản không còn cố định: lệch ngang ngẫu nhiên, robot đi vòng bên rộng hơn. Hướng đi suy ra từ ảnh, câu lệnh không đổi. Không dùng câu lệnh "trái/phải" và không chọn hướng ngẫu nhiên khi không có gợi ý (model sẽ lưỡng lự, có thể lấy trung bình hai hướng và đi thẳng vào vật cản). |

## Cảnh

- **Vật cản:** hộp 40 × 40 × 50 cm ở (−0,8; 0) m, dao động ±3 cm, màu vàng (dao động nhẹ).
- **START:** (−1,8; 0) m, dao động ±5 cm và ±5°.
- **Waypoint:** (−1,2; 0,7) → (−0,45; 0,7) → (−0,1; 0,05). Đích cuối nằm sau READY 10 cm và lệch trái 5 cm. Robot đi quá đích
  5–10 cm, nên nếu đặt đích đúng READY thì robot va vào bục (2/18 seed).
- **Layout bục/bàn/khay** giống hệt `can_pnp`, tính tương đối theo tư thế đứng tại READY.
- Camera bên ngoài (`debug_cam`) được kéo xa ra để thấy cả đoạn từ START tới bàn.

## Những gì đã thử ở đoạn đi bộ

| Thử | Kết quả | Kết luận |
| --- | --- | --- |
| Quay người theo hướng đi | Tay phải quệt góc sau của vật cản, khoảng hở 4 cm sau 1,4 s | Bỏ. Luôn nhìn +x |
| Nhìn +x, bước chéo/ngang | Khoảng hở 7–15 cm, đi hết 9–12 s | **Giữ** |
| Cho phép quay tối đa 30° | Robot quay quá đà (34°), không tốt hơn | Bỏ |
| Dùng `specific_target` của planner cho đoạn cuối | Robot đi lùi gần vật cản (hở 2–4 cm) | Bỏ |
| Giảm vận tốc đoạn cuối (0,12 m/s) | Planner không đi, hết giờ | Bỏ |
| Chuyển sang idle sớm hơn (12 cm) | Vẫn đi quá 5–9 cm | Không giúp |
| Bước ngắn để chỉnh vị trí sau khi dừng | Mỗi bước dịch 5–10 cm theo hướng ngẫu nhiên, sai số vẫn khoảng ±5 cm | Chỉ dùng khi lệch > 12 cm |

### Tư thế chân sau khi đi

Sau khi đi, SONIC luôn đứng xoè mũi chân, chân trái +10°, chân phải −20°, hai chân lệch trước sau tới 10 cm (tư thế chuẩn là hai
chân song song, cách tâm ±11,9 cm). Ngồi xổm từ tư thế này làm chân trượt (`feet_moved_before_walk`, 6/15 ca trong pilot đầu).

Cách sửa là `_square_feet` / `_step_foot`. Với mỗi chân, theo IK keyframe:

1. Dồn trọng tâm sang chân trụ (0,8 s).
2. Nhấc chân 4 cm (0,35 s).
3. Đặt chân xuống đúng góc chuẩn (0,35 s).
4. Dồn trọng tâm về giữa (0,8 s).

Tối đa 2 lượt cho cả hai chân. Kết quả:

- **Góc chân sửa tốt: còn < 2–5°.**
- **Vị trí chân không sửa chính xác được.** SONIC đặt chân lệch khoảng ±5 cm so với tham chiếu, kể cả khi đã neo mục tiêu theo chân
  trụ (cách tính đầu tiên theo tâm hai chân chỉ ra lệnh dịch 1,9 cm cho sai số 5,2 cm).
- Nhấc cao hơn (6–8 cm) và chậm hơn làm chân nhấc thật, nhưng vị trí đặt vẫn lệch ~5 cm.

Vì vị trí không ép chính xác được, layout được dời theo robot (`_relayout`). Robot chỉ cần đứng đúng **hình dạng** tư thế: mỗi chân
lệch ≤ 8 cm so với vị trí đối xứng quanh tâm hai chân, góc ≤ 6°. Trong pilot, các ca chạy trọn vẹn lệch tới 7,4 cm; ca lệch 9,5 cm
bị trượt chân khi ngồi.

## Pilot

| Lần | Thiết kế | Đạt | Lỗi chính |
| --- | --- | --- | --- |
| 1 (15 seed) | Đích = READY, chưa bước chỉnh chân, chưa dời layout | 1/15 | trượt chân khi ngồi 6, lệch READY 5, va bục 1 |
| 2 (18 seed) | + bước chỉnh chân + dời layout | 9/18 | va bục khi tới gần 2, lệch đích > 10 cm 2, vật cản sát 1 |
| 3 (18 seed, 900018–900035) | + đích lùi 10 cm, làn y = 0,70, nới ngưỡng | **14/18** | trượt chân/dịch người khi với tay 3, với hụt 1 |

Layout lệch so với chuẩn (14 ca ở pilot 3): x −0,08 ± 0,06 m, y +0,08 ± 0,06 m, góc −0,1 ± 1,0°.

## Dữ liệu cho hai model

| | Ψ₀ (`export_psi0.py`) | GR00T N1.7 (`export_gr00t.py`) |
| --- | --- | --- |
| State | 43, thứ tự 0810 | 46 = 0810 + trọng lực chiếu 3 |
| Action | `action.body_token` 64 + `action[:14]` tay | `action` 78: `motion_token` 64, `left_hand_joints` 7, `right_hand_joints` 7 |
| Video | `observation.images.egocentric` 320×240 | `observation.images.ego_view` 320×240 |
| Ngôn ngữ | `task_description` (cột chuỗi) | `annotation.human.task_description` → `task_index` |
| Thống kê | `stats_psi0.json` (min/max/mean/std/q01/q99) | `stats.json` (mean/std/min/max/q01/q99) |
| Chia train/val | `--seed 0`, 10% | cùng seed, nên cùng các episode val |
| Notebook | `colab_finetune_psi0_sonic11.ipynb` | `colab_finetune_gr00t_sonic11.ipynb` (tag `UNITREE_G1_SONIC`) |
| Đánh giá vòng kín | `--server http://HOST:PORT` | `--gr00t HOST:PORT` (cần `pyzmq msgpack`) |

- Checkpoint gốc `nvidia/GR00T-N1.7-3B` không có thống kê cho `unitree_g1_sonic` (đã đọc `statistics.json`). Vì vậy thống kê từ dữ
  liệu của ta được dùng, không bị bỏ qua như khi train dưới một tag đã có sẵn thống kê.
- Data loader GR00T **chưa chạy thử được ở máy local** (môi trường sibling thiếu pandas). Cell 6 của notebook GR00T nạp thử dataset
  trước khi train.
- Client GR00T trong `closed_loop_eval.py` viết theo định dạng của `gr00t/policy/server_client.py` (gửi ndarray dạng `.npy`, nhận dạng
  msgpack_numpy). **Chưa thử với server thật.**

## Camera cổ tay phải

- Camera `right_wrist_cam` gắn vào `right_wrist_yaw_link`, cấu hình bằng `scene.WRIST_CAM_*`.
  - Vị trí (−0,03; 0; 0,10) m, tức phía trên cổ tay theo trục gập ngón (gần thẳng đứng khi ôm lon), lùi 3 cm.
  - Hướng nhìn: dọc theo ngón tay, chúc xuống 40°, lệch 20° về phía lòng bàn tay. FOV 90°. Ảnh lấy trục gập ngón làm chiều "lên",
    nên lon hiện ra thẳng đứng.
- Các vị trí đã thử:

  | Vị trí | Kết quả |
  | --- | --- |
  | Mu bàn tay, sát cổ tay | Mesh bàn tay che gần hết khung hình |
  | Mu bàn tay, xa hơn 8 cm | Chỉ thấy mu bàn tay; lon bị các ngón che |
  | **Phía trên cổ tay** | Lúc tiếp cận thấy lon trên bục, lúc ôm thấy ngón quanh lon, lúc đặt thấy khay. **Chọn** |

- Camera chỉ là hình ảnh, không ảnh hưởng vật lý.
  - Episode mới: `writer.py` render luôn `observation.images.right_wrist_view` (640×480).
  - Episode cũ: `render_views.py <dataset> --views wrist --jobs N` render từ trạng thái đã ghi rồi cập nhật `info.json` và `modality.json`.
    Thử trên 3 episode: khoảng 95 s/episode khi máy đang bận sinh dữ liệu.
  - `merge_shards.py` chép cả video cổ tay.
- Trên robot thật phải gắn camera ở đúng vị trí và hướng này. Dữ liệu sim chỉ khớp với một cách lắp.

## Ψ₀ với nhiều camera (đọc code Ψ₀ `2830f93`)

- `--data.transform.repack.image-keys` nhận danh sách key. `SonicRepackTransform` trả về `observations` là danh sách ảnh, mỗi key
  một ảnh.
- Mọi ảnh được đưa vào cùng một prompt của Qwen3-VL, mỗi ảnh là một mục `{"type": "image"}` đặt trước câu lệnh (`psi0.py`, khoảng
  dòng 1926: `List of PIL Image as [view1, view2]`).
- `views_proj` là `Linear(2048 → d)` áp lên từng token VLM, không phụ thuộc số camera. Thêm camera chỉ làm dãy token dài hơn, nên
  **checkpoint warm-start vẫn nạp được**. Giới hạn prompt là 4096 token; mỗi ảnh 320×240 chỉ chiếm vài trăm token.
- `serve_psi0_sonic.py` lặp qua mọi `image_keys`, nên server cũng hỗ trợ nhiều camera.
- Lưu ý:
  - Mọi script train chính thức của Ψ₀ chỉ dùng 1 camera, nên phần action phải học cách dùng ảnh cổ tay khi finetune.
  - Train chậm hơn vì mỗi mẫu có gấp đôi ảnh đi qua VLM.
  - Thứ tự camera (đầu trước, cổ tay sau) phải giống nhau ở export, train, server và client.
- Việc cần làm:
  - `export_psi0.py`: xuất thêm `observation.images.right_wrist`.
  - Notebook: `--data.transform.repack.image-keys observation.images.egocentric observation.images.right_wrist`.
  - `closed_loop_eval.py`: gửi cả hai ảnh.

GR00T: cấu hình `unitree_g1_sonic` chỉ có `ego_view`. Muốn dùng 2 camera thì đăng ký cấu hình riêng dưới tag `NEW_EMBODIMENT`
(video `ego_view` + `right_wrist_view`, state/action giữ nguyên). Checkpoint gốc cũng không có thống kê cho `new_embodiment`.

## Làm giàu dữ liệu

- **Không lật video.**
  - Action là token SONIC, không có phép "lật gương" cho token. Ảnh lật cho thấy lon bên trái, nhưng action vẫn là tay phải gắp
    bên phải, nên dữ liệu tự mâu thuẫn.
  - Câu lệnh ("on your right", "right hand"), state trái/phải và hướng vòng qua vật cản cũng phải đảo theo.
  - Muốn có dữ liệu "gương" thì phải mô phỏng một task tay trái.
- Nên dùng thay thế:
  - Augment ảnh lúc train: màu, độ sáng, crop hoặc dịch nhẹ.
  - Tăng độ đa dạng lúc sinh: `--level 1` (biên độ dao động ×1,5), nhiều màu và ánh sáng hơn.
- Độ cao bàn: rút trong 0,50–0,55 m. Trên 100 episode đầu: nhỏ nhất 0,50, trung bình 0,525, lớn nhất 0,55 m.

## Vật cản ngẫu nhiên (từ episode 161)

- Vật cản: x = −0,8 ± 0,1 m, **y = ±(0,08–0,25) m** (dấu ngẫu nhiên). Rộng 30–50 cm, cao 40–60 cm. Màu chọn trong 7 màu, không dùng đỏ
  (trùng màu lon) và xanh lá (trùng màu khay).
- `SceneParams.nav_side`: đi trái nếu y ≤ 4 cm, ngược lại đi phải. 160 episode đầu (hộp ở giữa ±3 cm, đi trái) vẫn khớp quy tắc này.
- `scene.nav_waypoints(p)`: làn đi cách mặt bên của hộp 0,5 m, từ 0,2 m trước mặt sau tới 0,15 m sau mặt trước của hộp, rồi tới đích
  (−0,1; 0,05). Từ làn phải, robot đi quá đà theo hướng chéo lên trái, tức ra xa bục, nên vẫn an toàn.
- Pilot chỉ đoạn đi bộ (level 1, 12 seed): **12/12** tới đích (7 trái, 5 phải), khoảng hở tới vật cản ≥ 8 cm, đi hết 9–13 s.
- **Pilot đầy đủ (level 1, 40 lần thử, chạy bằng `run_blocks.py`): 18/40 (45%)**, 12 phải / 6 trái. Thấp hơn pilot level 0 (78%)
  vì `--level 1` cũng nhân 1,5 biên độ dao động vị trí lon (±2,25 cm thay vì ±1,5 cm):

  | Lý do bị loại | Số ca |
  | --- | --- |
  | Gắp hụt (`reach_failed`, `pregrasp_failed`, `grasp_failed`, `lift_failed`) | 9 |
  | Tư thế chân sau khi bước chỉnh vẫn lệch (`ready_stance_off`) | 4 |
  | Dịch người hoặc trượt chân khi với tay | 3 |
  | Vật cản sát (3,5 cm), lệch đích 24 cm, va bục | 3 |
  | Khác (va bàn, ngã khi đứng dậy, lon rơi khỏi khay) | 3 |

  Nếu muốn tỉ lệ đạt cao hơn mà vẫn đa dạng đoạn đi bộ, có thể bỏ việc nhân biên độ cho vị trí lon.
- Video xem nhanh: `.data/xem_video/can_nav_level1_left_oy-0.2_ngoai_va_dau.mp4`, `can_nav_level1_right_oy0.23_ngoai_va_dau.mp4`.

## Sinh dữ liệu trên Colab (đã làm)

- Thời gian sinh dữ liệu chủ yếu nằm ở **CPU**: vật lý MuJoCo và SONIC/planner (onnxruntime, batch 1). GPU chỉ giúp phần render
  (`MUJOCO_GL=egl`), khoảng 1/3 thời gian mỗi episode.
- Tốc độ tăng theo số nhân CPU và RAM (mỗi tiến trình khoảng 2,5 GB).
  - Máy local có 20 luồng nhưng chỉ chạy được 5 tiến trình vì RAM.
  - Colab High-RAM chạy được khoảng 10–12 tiến trình. Chạy nhiều phiên Colab song song, mỗi phiên một dải seed.
- Đã làm:
  - `pack_colab.py` → `.data/colab/can_nav_datagen.zip` (851 MB). Đã thử: giải nén vào thư mục trống rồi chạy `generate.py` từ đó
    thành công.
  - `run_blocks.py`: chia seed thành các khối, khối nào xong thì chép sang Drive và đánh dấu `DONE`, chạy lại thì bỏ qua khối đã
    có. Đã chạy thử cho pilot đầy đủ ở trên.
  - `colab_datagen.ipynb`: tự chọn số tiến trình theo CPU và RAM, render bằng EGL nếu có GPU, không thì OSMesa, hoặc
    `RENDER = "none"` rồi render sau ở local bằng `render_views.py`.
- Cách đóng gói:
  - `pack_colab.py`: zip khoảng 1 GB gồm code, model G1, SONIC 1.1, planner (chép file thật thay vì symlink) và danh sách thư viện
    ghim phiên bản (mujoco 3.14.0, onnxruntime 1.30.0, numpy 2.4.6, pandas 3.0.6, pyarrow 25.0.1).
  - Notebook chạy hàng đợi các khối seed nhỏ, khối nào xong chép ngay về Drive, chạy lại thì bỏ qua khối đã có.
- Colab dùng Python 3.12, khác máy local (3.11). Cùng một seed có thể ra episode hơi khác, nhưng dữ liệu vẫn hợp lệ. Chỉ cần các máy
  dùng dải seed khác nhau.

## Finetune trước trên 160 episode, server và đánh giá

- Đã export `can_nav_160`: `.data/psi0/can_nav_160_psi0.zip` (733 MB) và `.data/gr00t/can_nav_160_gr00t.zip` (742 MB).
  144 episode train / 16 val, hai bản dùng cùng các episode val.
- Notebook finetune Ψ₀: đổi `EXP` thành `cannav160-s11`. Tên cũ `canped-s11` trùng với run của `can_ped_144`, mà cell 2 tự resume
  theo `EXP`, nên có thể đã train tiếp checkpoint cũ.
- **Server Ψ₀** `colab_psi0_server.ipynb`:
  - Dựng từ notebook server của `box_pnp`, đổi sang code Ψ₀ `2830f93`.
  - Dùng cùng cell cài môi trường với notebook finetune, nên môi trường server giống hệt lúc train.
  - `EXP` mặc định là `cannav160-s11`. Request gọi thử dùng `dataset_name = can_ped`, như lúc train.
- **Server GR00T** `colab_gr00t_server.ipynb`:
  - Dùng `gr00t/eval/run_gr00t_server.py` chuẩn, tag `UNITREE_G1_SONIC`.
  - Checkpoint lấy từ `My Drive/gr00t/checkpoints/can_nav_160_gr00t/checkpoint-N` và được chép về ổ local của Colab trước khi nạp.
  - Sau khi khởi động, notebook gọi thử bằng `PolicyClient`, với đúng cấu trúc quan sát mà `closed_loop_eval.py --gr00t` gửi.
- **Client GR00T đã kiểm tra giao thức:**
  - Dựng server ZMQ giả, dùng đúng class `MsgSerializer` của Isaac-GR00T.
  - Server giả nhận đúng ảnh `(1,1,240,320,3)` uint8, 8 nhóm state float32 và câu lệnh.
  - Client giải mã được action `(40, 78)`.
- `closed_loop_eval.py --obstacle center`: vật cản ở giữa (±3 cm, 40 × 40 × 50 cm, màu vàng), dùng để đánh giá checkpoint 160
  episode. Thêm `--layout-from .data/can_nav_160` để lấy layout bục/bàn từ các episode đã sinh.
- Lệnh trong README, notebook và docstring đã chuyển sang terminal Ubuntu:
  - Chạy bằng `.venv-datagen/bin/python`.
  - Cài thư viện bằng `uv pip install --python .venv-datagen/bin/python ...`, vì venv này không có pip.

## Thiết lập train của Ψ₀ và GR00T (2026-10-10)

Quyết định: **chỉ đặt số bước và batch giống nhau**, mọi thiết lập khác giữ mặc định của từng model.

| Thiết lập | Ψ₀ | GR00T N1.7 |
| --- | --- | --- |
| **Batch hiệu dụng** (giống nhau) | 128 (`GLOBAL_BATCH`, tự chia BS × ACC theo VRAM) | 128 = `GLOBAL_BATCH_SIZE` 64 × `GRAD_ACC` 2 (GPU 40 GB: 32 × 4) |
| **Số bước / số mẫu** (giống nhau) | 8 000 / 1,02 triệu (~3,3 lần qua 310 000 khung) | như Ψ₀ |
| Dữ liệu / chia val | `can_nav_160`, 144 train / 16 val | như Ψ₀ (cùng episode val) |
| LR, warmup | 1e-4 cosine, warmup 500 bước (mặc định notebook) | 1e-4 cosine, warmup 5% (mặc định `finetune.sh`) |
| Đổi màu ảnh | bật (`img-aug`, mặc định công thức Ψ₀) | tắt (mặc định `finetune.sh`) |
| Bỏ state ngẫu nhiên | 10% (mặc định công thức Ψ₀ 1.1) | 20% (mặc định `finetune.sh`) |
| Weight decay | 1e-6 | 1e-5 |
| VLM | đóng băng | đóng băng |
| Checkpoint | mỗi 1 000 bước | mỗi 1 000 bước, chép về Drive từ bước 2 000 |

Khác biệt do kiến trúc: chunk 30 so với 40 bước, cách cắt ảnh trước VLM, phần được train 0,67 tỷ so với 1,62 tỷ tham số,
pre-train (Ψ₀ đã post-train với token SONIC 1.1; projector `unitree_g1_sonic` của GR00T học từ đầu).

Đánh giá: cùng seed (từ 1 000 000), cùng cảnh (`--obstacle center --layout-from .data/can_nav_160`), cùng `--exec-steps 15`
và `--max-seconds`, so ở cùng các mốc checkpoint (ví dụ 4 000 / 6 000 / 8 000).

## Đánh giá vòng kín: lỗi ảnh nhiễu khi dùng `--viewer`, và lưu bằng chứng (2026-10-13)

**Lần đánh giá đầu tiên** (Ψ₀ `cannav160-s11`, ckpt 8000, G4, `--obstacle center`):
- Seed 1000000: **thành công** sau 66 s (đi vòng vật cản, nhặt lon, đặt vào khay).
- Seed 1000001: hết giờ, robot đứng gần điểm xuất phát.
- Seed 1000002: robot cũng đứng lưỡng lự.

**Nguyên nhân (lỗi của `closed_loop_eval.py`, không phải của model):**
- Cửa sổ viewer MuJoCo (GLFW) khi đóng lúc hết cảnh đã huỷ luôn ngữ cảnh OpenGL của renderer offscreen (cũng chạy trên GLFW).
- Từ cảnh thứ hai trở đi, **mọi ảnh render đều là nhiễu**: cả ảnh camera đầu gửi cho model lẫn video. Mức nhiễu (trung bình
  |chênh lệch pixel ngang|) là 38–47, so với 1–4 ở ảnh thường. Model "nhìn" nhiễu nên đứng yên.
- Vì vậy kết quả seed 1000001, 1000002 **không có giá trị**.

**Đã sửa:**
- Khi có `--viewer`, ảnh cho model và video render bằng **EGL** (Mesa EGL chạy được trên GPU Intel của máy local, khoảng
  8 ms/khung), độc lập với GLFW của viewer. Renderer được đóng sau mỗi cảnh.
- Thêm chốt chặn: nếu ảnh camera đầu có mức nhiễu > 20 thì dừng ngay và ghi `ERROR`, không đưa nhiễu vào model.
- Đã thử 3 cảnh liên tiếp có viewer: mức nhiễu của ảnh ở cả 3 cảnh là 1,2–2,9.

**Lưu bằng chứng (mặc định, mọi kết quả):** mỗi cảnh có thư mục `seed_<seed>_<KẾT QUẢ>`, với KẾT QUẢ là `SUCCESS`,
`FAIL_<lý do>`, `ERROR` hoặc `STOPPED`. Thư mục gồm:
- `ego.mp4`, `third.mp4`.
- `overview.png`: 12 thời điểm (góc ngoài + camera đầu) và banner kết quả.
- `result.json`.
- `trace.npz`: thời gian, qpos, pose lon, action đã thực thi, tiếp xúc, ở 30 Hz.

Ngoài ra:
- `results.jsonl`, `summary.md` / `summary.json` được ghi lại sau mỗi cảnh, và có `run_config.json`.
- Mặc định `--out` là `.data/eval/<policy>_<ngày>_<giờ>`. `--no-video` để bỏ mp4.
- Lỗi server/mạng (ví dụ tunnel hết hạn) được ghi là `ERROR`, rồi chương trình in lệnh chạy tiếp bằng `--seed-start`.
- `--replay` nhận danh sách episode, ví dụ `--replay-episode 5,7,9`.

**Đánh giá ngay trên Colab** (thêm 2026-10-13):
- Notebook server Ψ₀ có thêm cell 6b / 6c; notebook GR00T có thêm Bước 5b / 5c.
- Mô phỏng chạy trên máy Colab và gọi server qua `localhost`, nên bỏ được tunnel. Mỗi request qua tunnel trung bình mất 1,2 s
  (tối đa 6 s), làm viewer local trông giật, và tunnel còn hết hạn sau 60 phút.
- Môi trường mô phỏng: `uv venv` Python 3.11 cài từ `requirements_datagen.txt` của gói đánh giá **`can_nav_eval.zip`**
  (`pack_colab.py --eval`, 198 MB: không có planner 740 MB, chỉ dùng khi sinh dữ liệu), render bằng EGL trên GPU NVIDIA.
- Gói zip có thêm layout của `can_nav_160` (`--layout-from`).
- Bằng chứng từng cảnh được tải lên Drive (`My Drive/psi0/runs/eval/<run>/`, `My Drive/gr00t/eval/<run>/`). Ảnh `overview.png`
  hiện ngay trong notebook.
- Đã thử phần cài đặt trong venv mới ở máy local: phát lại 1 episode, kết quả `SUCCESS`.

## Server Ψ₀ và đánh giá trên Colab: các lỗi đã gặp và cách sửa (2026-10-12 → 13)

| Lỗi | Nguyên nhân | Đã sửa |
| --- | --- | --- |
| Cell 6: `No module named 'uvicorn'` | Notebook server dùng lại cell cài môi trường của notebook finetune, cell đó không cài nhóm `serve` (fastapi, uvicorn) | Cell 3 cài `fastapi==0.119.1 uvicorn==0.38.0` (đúng phiên bản trong lock), và kiểm tra import ngay ở cell 3 |
| `/act`: `linear(): argument 'input' must be Tensor, not NoneType` | Checkpoint SONIC 1.1 dùng `combined-temb` (vector CLIP của câu lệnh); ở Ψ₀ `2830f93`, `serve_psi0_simple` (đằng sau `serve_psi0_sonic_http`) không tính vector này mà truyền `None`. Chỉ `serve_psi0_sonic.py` (RTC) có xử lý | Cell 3 vá `serve_psi0_simple.py`: lấy vector từ cache lúc train (`clip_pooled_cache.pt`), câu lệnh mới thì mã hoá bằng CLIP đóng băng, giống `serve_psi0_sonic.py`. Bản vá chạy lại nhiều lần không sao |
| Viewer local: từ cảnh thứ 2, ảnh toàn nhiễu, viewer hiện khung dây vàng | Đóng viewer (GLFW) làm hỏng renderer offscreen dùng chung GLFW | Có `--viewer` thì render bằng EGL; chốt chặn ảnh nhiễu (mục trên) |
| Seed đặt được lon vào khay bị chấm `can_tipped_over` | Điều kiện "lon đổ > 45°" áp dụng cả khi lon đã nằm trong khay | Chỉ tính khi lon ngoài khay |
| Đánh giá báo `ERROR ... Connection refused` giữa chừng (3 lần) | Server bị tắt và bật lại trong lúc đang đánh giá. Khả năng cao là cell "sửa nhanh" (có `pkill -f serve_psi0_sonic_http`) vẫn nằm dưới cell 6 và được chạy lại mỗi lần chạy các cell theo thứ tự. Server mới mất khoảng 2,5 phút nạp model nên request bị từ chối | Xoá cell sửa nhanh (bản vá đã nằm trong file của server) |
| Mô phỏng trên Colab cực chậm: mỗi request cách nhau khoảng 21 s, CPU 297% | EGL rơi về Mesa llvmpipe (render bằng CPU) vì thiếu file vendor EGL của NVIDIA | Cell 6b (tiếp) / Bước 5b ghi `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` (cách của hướng dẫn dm_control), in tên GPU đang render và ms/khung, báo lỗi nếu vẫn là `llvmpipe` |

Thêm khi đánh giá trên Colab:
- **Xem trực tiếp:** mỗi 0,5 s mô phỏng ghi `live.jpg` (góc ngoài | camera đầu) + `live.json`. Cell 6c / 5c cập nhật ảnh tại
  chỗ, kèm tiến độ (cảnh thứ mấy, giây thứ mấy, đã gắp chưa). Viewer MuJoCo không mở được trên Colab vì máy không có màn hình.
- Cell 6c / 5c in **link thẳng tới thư mục Drive** chứa video và bằng chứng.
- **Nơi lưu kết quả:** Ψ₀ ở `My Drive/psi0/runs/eval/<EXP>_ckpt<bước>_<OBSTACLE>_colab/`, GR00T ở
  `My Drive/gr00t/eval/<...>/`, chạy local ở `.data/eval/<--out>/`. Chạy lại cùng tên thì ghi tiếp; `summary.md` lấy kết quả
  mới nhất của mỗi seed.
- **Tốc độ:** vật lý 200 Hz, SONIC 50 Hz, action 30 Hz, hỏi model mỗi 15 bước (2 Hz). Qua tunnel mỗi request mất khoảng 1,2 s,
  nên viewer local đứng chờ và trông giật; trên Colab qua `localhost` mất khoảng 0,6 s.
- **Thứ tự chạy notebook server Ψ₀:** 1 → 6 → 6b → 6b (tiếp) → 6c. Cell 7 (tunnel) chỉ dùng khi xem bằng viewer ở máy local;
  cell 8 xem log server; cell 9 dừng server (khi đổi checkpoint).

## SONIC 1.1 ngồi xổm sâu được đến đâu (thử 2026-10-10)

**Kết luận: SONIC 1.1 không ngồi xổm hẳn xuống được.** Hông thấp nhất khoảng **0,42–0,47 m** (lúc đứng 0,76 m). Ra lệnh sâu hơn
thì hông không hạ thêm, góc gối có tăng nhưng gót chân bị nhấc khỏi sàn. Vì vậy lon vẫn phải đặt trên bục, không đặt dưới sàn
được, nếu không finetune lại SONIC.

**Cách thử** (`scripts/datagen/can_nav_pnp/probes/probe_squat.py`, `probe_squat_deep.py`):
- Robot đứng ở READY, xung quanh không có vật cản.
- Ra lệnh ngồi xổm tới độ cao hông mục tiêu, giữ 2–3 s, rồi đo hông thật, góc gối, góc hông, cổ chân và chân có chạm sàn không.
- Ba cách ra lệnh:
  1. Planner của SONIC (mode SQUAT), đúng cách `generate.py` đang dùng.
  2. Tham chiếu IK toàn thân (chân cố định, trọng tâm trên bàn chân), có giới hạn khớp của `kinematics.py` (gối ≤ 2,2 rad = 126°,
     hông ≥ −1,9 rad).
  3. Tham chiếu IK như trên nhưng bỏ hai giới hạn đó, chỉ giữ giới hạn khớp trong MJCF (gối tới khoảng 165°).

| Hông mục tiêu | Planner: hông thật / gối | IK có giới hạn: hông thật / gối | IK bỏ giới hạn: gối lệnh → hông thật / gối thật |
| --- | --- | --- | --- |
| 0,60 m | 0,64 m / 77° | 0,62 m / 87° | |
| 0,50 m | 0,57 m / 95° | 0,59 m / 93° | |
| 0,42 m | **0,53 m / 105°** (đang dùng khi sinh dữ liệu) | 0,56 m / 99° | |
| 0,40 m | | | 136° → 0,55 m / 101° |
| 0,35 m | 0,50 m / 110° | 0,51 m / 109° | 145° → 0,54 m / 109° |
| 0,30 m | 0,48 m / 114° | 0,45 m / 120° | 154° → 0,46 m / 126°, **gót nhấc khỏi sàn** |
| 0,27 m | | | 160° → 0,45 m / 132° |
| 0,25 m | **0,47 m / 115°** | **0,42 m / 125°** (chạm giới hạn gối) | 163° → 0,43 m / 142°, gót nhấc |
| 0,22 m | | | 164° → 0,44 m / 141°, gót nhấc |

Không lần nào robot ngã.

- Planner tự chặn tham chiếu ở khoảng 0,30 m (lệnh 0,25 m thì tham chiếu chỉ xuống 0,296 m). Ở mọi độ sâu, robot thật cao hơn tham
  chiếu 4–17 cm, và thân cúi về trước 24–33°.
- Tham chiếu IK giữ thân gần thẳng (< 11°) và xuống sâu hơn planner khoảng 5 cm.
- Bỏ giới hạn khớp thì SONIC gập gối tới 142° (lệnh 163°), nhưng hông vẫn ở 0,43–0,46 m. Bàn chân xoay lên thành nhón gót, mất
  tiếp xúc sàn. Không có cách ra lệnh nào đưa hông xuống dưới khoảng 0,42 m mà vẫn đứng vững.
- Ảnh: `.data/xem_video/sonic11_squat_test.png` (planner và IK có giới hạn) và `sonic11_squat_test_unclamped.png` (IK bỏ giới hạn).

Hệ quả: muốn nhặt vật thấp hơn (dưới sàn) thì phải **finetune SONIC** (RL trong Isaac Lab với chuyển động ngồi xổm sâu / quỳ),
hoặc giữ cách hiện tại là đặt vật trên bục cao từ khoảng 20 cm.

## Việc tiếp theo

- [x] Sinh 160 episode → `.data/can_nav_160`. Các shard `.data/can_nav_s0`–`s4` vẫn còn (5,2 GB trùng lặp), có thể xoá.
- [x] Vật cản ngẫu nhiên, đi vòng bên rộng hơn; pilot 18/40.
- [x] Gói Colab + `run_blocks.py` + `colab_datagen.ipynb`; gói đánh giá `can_nav_eval.zip`.
- [x] Export `can_nav_160` cho Ψ₀ và GR00T; notebook server cho cả hai, có đánh giá ngay trên Colab.
- [x] Train Ψ₀ `cannav160-s11` (8 000 bước).
- [ ] 🔄 Đánh giá vòng kín Ψ₀ ckpt 8000: 10 cảnh từ seed 1 000 000, `OBSTACLE = "center"` (đang chạy trên Colab).
      Sau đó xoá các thư mục `ERROR` do server bị tắt giữa chừng.
- [ ] Train GR00T trên `can_nav_160` (cùng 8 000 bước × batch 128), rồi đánh giá trên cùng 10 seed để so sánh.
- [ ] Chạy `colab_datagen.ipynb` (runtime CPU High-RAM, mỗi phiên một `SEED_BASE`: 100000, 300000, 500000…) tới khi đủ
      340 episode.
- [ ] Tải các khối về rồi gộp: `merge_shards.py --out .data/can_nav_500 .data/can_nav_s{0..4} <blocks>/blk_*`
      (nếu `RENDER = "none"` thì chạy thêm `render_views.py --views ego`).
- [ ] Export bộ 500 episode, train lại cả hai model từ checkpoint gốc, đánh giá với `OBSTACLE = "random"` và `--level 1`;
      tách tỉ lệ thành công theo `nav_side` (trái / phải).
