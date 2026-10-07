# can_pnp: ngồi xổm lấy lon trên bục → đứng dậy → đi tới bàn → đặt vào khay

Pipeline sinh dữ liệu cho G1 (Dex3) với SONIC 1.1 trong MuJoCo. Được chép từ `../box_pnp` rồi sửa lại.
Quá trình thiết kế, các phương án đã thử và số liệu nằm trong
[docs/sonic_pick_can/05_memo_sinh_du_lieu_can_pnp.md](../../../docs/sonic_pick_can/05_memo_sinh_du_lieu_can_pnp.md).

- Vật cần lấy là **lon slim cao 12,8–14 cm** (đường kính 5,3 cm), đứng trên **bục 12 × 12 cm cao 0,19–0,21 m**, phía trước
  bên phải robot, ở (0,27; −0,30) m ± 1,5 cm. Ở vị trí này chân cách bục và lon ít nhất 4,5 cm lúc ngồi xổm.
- Robot ngồi xổm theo từng nấc. **Hai tay giữ nguyên tư thế ban đầu cho tới khi ngồi xong**, rồi tay phải mới đưa ra.
- **Gắp bằng cả bàn tay**: lòng bàn tay áp vào thân lon từ phía sau bên phải, 3 ngón ôm quanh lon (có hỗ trợ gắp).
  Tay tới một điểm chờ cạnh lon rồi đẩy vào; ngón cái bẻ ra sau trong lúc tiếp cận để không chặn lon.
- Nhấc lon (hông giữ nguyên, chỉ tay cử động), đứng dậy, cân bằng, đi tới bàn, đặt lon vào khay, về tư thế ban đầu.
- Episode bị loại nếu lon đang cầm chạm thân robot, robot chạm bục (trừ bàn tay), hoặc chân trượt quá ngưỡng.
- Đặt `scene.PEDESTAL_HEIGHT = 0` và `--set "grasp_style='pinch'"` để quay về thiết kế cũ (lon dưới sàn, kẹp miệng lon).
- Code bên trong vẫn giữ tên "box" của `box_pnp` (`box_pose`, body `box`, `SceneParams.box_half`, ...), nên
  `episode`/`writer`/`render` dùng chung được. Với lon, `box_half` = (bán kính, bán kính, nửa chiều cao).

## Bắt buộc dùng SONIC 1.1

SONIC 1.0 (`policy/release`) không gập eo theo lệnh (lệnh 0,5 rad, thực tế chỉ 0,02–0,05) và đặt tay cao hơn lệnh
5–13 cm, nên không hạ người và tay đủ thấp cho task này.

Tải SONIC 1.1 (repo public `nvidia/GEAR-SONIC`):

```bash
cd third_party/GR00T-WholeBodyControl
python download_from_hf.py --sonic-v1-1      # -> gear_sonic_deploy/policy/sonic_v1_1/
```

Biến môi trường `SONIC_VARIANT` dùng để chọn controller, mặc định là `sonic_v1_1`; đặt `release` để chạy 1.0. Code tự nhận phiên bản
từ `observation_config.yaml`. So với 1.0, bản 1.1 khác ở ba điểm:

- encoder 1751-D, không có root z;
- hướng neo tính tương đối theo hướng thân robot (chỉ yaw);
- gain PD cổ chân pitch nhân 1,5, theo cấu hình deploy đã được NVIDIA kiểm thử.

Token của 1.1 vẫn nằm trên lưới 1/16. `meta/info.json` và `meta/episode_meta.jsonl` ghi lại `sonic_version`.

## Chạy

Mỗi tiến trình tốn khoảng 2,2–2,8 GB RAM. Trên máy 31 GB, chạy tối đa 6 tiến trình cùng lúc.

```bash
# 1 episode mẫu + video review
SONIC_VARIANT=sonic_v1_1 .venv-datagen/bin/python scripts/datagen/can_pnp/generate.py \
    --out .data/can_ped_sample --num-episodes 1 --seed-start 1 --max-attempts 8 --review

# pilot đo tỉ lệ đạt (chỉ mô phỏng vật lý, không ghi dữ liệu)
python scripts/datagen/can_pnp/generate.py --out .data/can_ped_pilot --dry-run --num-episodes 10 --max-attempts 10 --seed-start 800000

# chạy thật: chia shard theo dải seed (dưới 1 000 000, dải trên dành cho đánh giá) rồi gộp lại
python scripts/datagen/can_pnp/generate.py --out .data/can_ped_s0 --num-episodes 24 --seed-start 0
python scripts/datagen/can_pnp/generate.py --out .data/can_ped_s1 --num-episodes 24 --seed-start 100000
python scripts/datagen/can_pnp/merge_shards.py --out .data/can_ped_48 .data/can_ped_s0 .data/can_ped_s1

# export cho Psi0 (train/val + zip để tải lên Drive)
python scripts/datagen/can_pnp/export_psi0.py .data/can_ped_144 --out .data/psi0 --zip

# đánh giá closed-loop với server Psi0
python scripts/datagen/can_pnp/closed_loop_eval.py --server http://HOST:PORT --episodes 10 --out .data/eval/can
```

`--max-attempts` mặc định bằng 20 lần `--num-episodes`.

## Finetune Ψ₀ (SONIC 1.1)

Dùng [`colab_finetune_psi0_sonic11.ipynb`](colab_finetune_psi0_sonic11.ipynb), **không** dùng notebook của `box_pnp`
(notebook đó warm-start từ `sonic-wbcbox`, checkpoint học token SONIC 1.0).

- Warm-start: `psi0/postpre.sonic1.1.unifolm.2609181726.40k` (HF `USC-PSI-Lab/psi-model`), code Ψ₀ `2830f93`.
- Không cần sửa export: dữ liệu vẫn là action 78 / state 43; trình nạp dữ liệu tự đệm 0 lên 80 / 45, đúng như lúc checkpoint
  được post-train trên dữ liệu UniFoLM G1 (cũng không có cổ). Hai chiều đệm của action không tính vào loss.
- Server đánh giá phải chạy code Ψ₀ `2830f93`. Model xuất action 80 chiều; `closed_loop_eval.py` lấy 78 chiều đầu.

## Tình trạng

- Bộ dữ liệu `.data/can_ped_144`: 144 episode, tỉ lệ đạt khi sinh 37%, 131/144 lon đứng thẳng trong khay ở cuối episode.
- Lỗi chính còn lại khi sinh dữ liệu là gắp hụt (tay dừng thiếu vài cm hoặc khép tay không bắt được lon).
- Camera đầu nhìn thấy lon và bục ngay từ tư thế ban đầu.
