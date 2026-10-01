# can_pnp: ngồi xổm nhặt lon trên sàn → đứng dậy → đi tới bàn → đặt vào khay

Pipeline sinh dữ liệu cho G1 (Dex3) với SONIC trong MuJoCo. Được chép từ `../box_pnp` rồi sửa lại.

- Bỏ bệ đỡ. Vật cần nhặt là **lon slim cao 12,8–14 cm** (đường kính 5,3 cm), đứng trên sàn, phía trước bên phải
  bàn chân phải, ở (0,20; −0,25) m. Vị trí được random ±2 cm.
- Ngồi xuống theo từng nấc: ngồi nửa chừng rồi cân bằng, ngồi hẳn rồi cân bằng. Sau đó với tay phải xuống, gắp lon từ trên
  xuống (có hỗ trợ gắp), nhấc lên, đứng dậy, cân bằng, đi tới bàn, đặt lon vào khay, rồi về tư thế ban đầu.
- Bàn đặt xa hơn so với `box_pnp`: mép bàn ở 0,62–0,66 m. Lý do: khi với xuống sàn, thân robot chúi về trước và đập
  vào mép bàn nếu bàn ở gần.
- Code bên trong vẫn giữ tên "box" của `box_pnp` (`box_pose`, body `box`, `SceneParams.box_half`, ...), nên
  `episode`/`writer`/`render` dùng chung được. Với lon, `box_half` = (bán kính, bán kính, nửa chiều cao).

## Bắt buộc dùng SONIC 1.1

SONIC 1.0 (`policy/release`) **không với tới** lon đặt trên sàn. Trong sim đã đo được:

| | SONIC 1.0 | SONIC 1.1 |
|---|---|---|
| Ngồi xổm, điểm gắp thấp nhất tay phải xuống được | 0,235 m | **0,145 m** (với vòng servo: ~0,13 m) |
| Quỳ hai gối (planner mode 5) | 0,235 m | robot lùi ~1 m, không dùng được |
| Điểm gắp cần đạt (2,5 cm dưới miệng lon) | 0,10–0,115 m | |

SONIC 1.0 không gập eo theo lệnh (lệnh 0,5 rad, thực tế chỉ 0,02–0,05) và đặt tay cao hơn lệnh 5–13 cm.

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

Token của 1.1 vẫn nằm trên lưới 1/16, trong khoảng ±0,5. `meta/info.json` và `meta/episode_meta.jsonl` ghi lại `sonic_version`.

## Chạy

```bash
# 1 episode mẫu + video review
SONIC_VARIANT=sonic_v1_1 .venv-datagen/Scripts/python scripts/datagen/can_pnp/generate.py \
    --out .data/can_pnp_sample --num-episodes 1 --seed-start 1 --max-attempts 8 --review

# chạy thật: chia shard theo dải seed rồi gộp lại
python scripts/datagen/can_pnp/generate.py --out .data/can_pnp_s0 --num-episodes 70 --seed-start 0
python scripts/datagen/can_pnp/generate.py --out .data/can_pnp_s1 --num-episodes 70 --seed-start 100000
python scripts/datagen/can_pnp/merge_shards.py --out .data/can_pnp_140 .data/can_pnp_s0 .data/can_pnp_s1

# export cho Psi0, đánh giá closed-loop (giống box_pnp)
python scripts/datagen/can_pnp/export_psi0.py .data/can_pnp_140 --out .data/psi0 --zip
python scripts/datagen/can_pnp/closed_loop_eval.py --server http://HOST:PORT --episodes 10 --out .data/eval/can
```

`--max-attempts` mặc định bằng 20 lần `--num-episodes`.

## Tình trạng và việc còn lại

- **Tỉ lệ episode đạt khoảng 1/3** (2/6 seed khi thử). Gần như mọi lần bị loại đều do **bàn chân trượt hơn 6 cm** lúc với tay
  hoặc lúc nhấc lon. Ngưỡng này được giữ chặt như `box_pnp` để dữ liệu sạch, nên chỉ cần chạy nhiều seed hơn.
  Nếu muốn yield cao hơn, có thể nới bằng `--set feet_slip_tol_contact=0.09 --set body_shift_tol=0.06`, đổi lại dữ liệu
  sẽ có chân trượt nhiều hơn.
- Khi mang lon tới khay, lon nghiêng 15–40° vì giữ nguyên hướng cổ tay lúc gắp. Lon vẫn vào khay nhưng thường bị đổ
  trong khay (`require_upright=False`).
- **Finetune Ψ₀:** checkpoint khởi động hiện dùng trong `box_pnp/colab_finetune_psi0.ipynb` (`sonic-wbcbox`) học
  token của **SONIC 1.0**. Với dữ liệu này cần checkpoint SONIC 1.1. HF `USC-PSI-Lab/psi-model` có
  `psi0/postpre.sonic1.1.unifolm.2609181726.40k`, nhưng checkpoint này dùng layout có cổ (action 80 / state 45), nên phải pad
  thêm 2 chiều cổ khi export. Khi deploy hoặc đánh giá cũng phải dùng decoder 1.1.
- Camera đầu nhìn thấy lon ở góc dưới bên phải ngay từ tư thế ban đầu, và thấy rõ trong lúc với tay.
