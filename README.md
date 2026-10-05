# Ψ₀ + SONIC: G1 gắp vật đặt vào khay

Bộ công cụ để huấn luyện [Ψ₀ (Psi0)](https://github.com/physical-superintelligence-lab/Psi0) cho task
**box pick-and-place** trên robot Unitree G1 (Dex3). Robot gắp hộp trên bệ, đứng dậy, đi tới bàn, đặt hộp vào khay rồi
về tư thế ban đầu. Toàn thân được điều khiển qua SONIC 1.1 (`gear_sonic_deploy/policy/sonic_v1_1`, chọn bằng biến `SONIC_VARIANT`; đặt `SONIC_VARIANT=release` để chạy 1.0).

Repo gồm:  

1. **Sinh dữ liệu** bằng expert có kịch bản trong MuJoCo + SONIC. Chỉ giữ episode sạch.
2. **Export sang LeRobot** đúng layout Ψ₀ neckless: action 78 = token SONIC 64 + tay 14, state 43.
3. **Finetune Ψ₀ trên Colab** (L4 hoặc G4), checkpoint lưu lên Google Drive.
4. **Server policy trên Colab + mô phỏng MuJoCo trên máy local** để đánh giá closed-loop.

Có hai task:

- **`box_pnp`**: gắp hộp trên bệ. Code mặc định dùng SONIC 1.1; dataset 140 episode và model đã finetune hiện có được sinh bằng SONIC 1.0 (cần sinh lại).
- **`can_pnp`** (mới): ngồi xổm nhặt lon trên sàn cạnh chân phải, đứng dậy, đi tới bàn, đặt lon vào khay. Task này **bắt buộc
  dùng SONIC 1.1**, vì SONIC 1.0 không với tới được vật nằm ở độ cao sàn. Xem
  [scripts/datagen/can_pnp/README.md](scripts/datagen/can_pnp/README.md).

Repo chỉ chứa phần thêm vào. Nó chạy bên trong một bản checkout của Psi0 gốc, đặt đúng các thư mục sau:

```
scripts/datagen/box_pnp/    task hộp: code sinh dữ liệu, export, đánh giá, notebook Colab
scripts/datagen/can_pnp/    task lon trên sàn (SONIC 1.1): sinh dữ liệu, export, đánh giá
docs/sonic_pick_can/        ghi chú: pipeline, kế hoạch finetune/kiểm thử, báo cáo các lần chạy
```

## Cài đặt (máy local, Windows hoặc Linux)

```bash
git clone https://github.com/physical-superintelligence-lab/Psi0.git
cd Psi0
git checkout 4f3720d45e102b36d7c3e9465ab8062274170518
git submodule update --init third_party/GR00T-WholeBodyControl
git -C third_party/GR00T-WholeBodyControl lfs pull     # model G1 (MJCF), ONNX của SONIC
# chỉ cần cho can_pnp: SONIC 1.1 -> gear_sonic_deploy/policy/sonic_v1_1/
(cd third_party/GR00T-WholeBodyControl && python download_from_hf.py --sonic-v1-1)

# chép các thư mục của repo này vào cùng vị trí
git clone https://github.com/ndhoang2k5/psi0.git /tmp/psi0-boxpnp
cp -r /tmp/psi0-boxpnp/scripts/datagen scripts/
mkdir -p docs && cp -r /tmp/psi0-boxpnp/docs/sonic_pick_can docs/

# môi trường cho sinh dữ liệu / đánh giá (Python 3.11)
python -m venv .venv-datagen
.venv-datagen/Scripts/python -m pip install mujoco==3.14.0 numpy onnxruntime imageio imageio-ffmpeg pandas pyarrow pillow scipy matplotlib
```

Trên Linux thay `.venv-datagen/Scripts/python` bằng `.venv-datagen/bin/python`.

## Quy trình

**1. Sinh dữ liệu**

```bash
# xem trước 1 episode (video + biểu đồ để kiểm tra)
.venv-datagen/Scripts/python scripts/datagen/box_pnp/generate.py --out .data/box_pnp_preview --num-episodes 1 --review

# chạy thật: chia shard theo dải seed, rồi gộp
.venv-datagen/Scripts/python scripts/datagen/box_pnp/generate.py --out .data/box_pnp_s0 --num-episodes 70 --seed-start 0
.venv-datagen/Scripts/python scripts/datagen/box_pnp/generate.py --out .data/box_pnp_s1 --num-episodes 70 --seed-start 100000
.venv-datagen/Scripts/python scripts/datagen/box_pnp/merge_shards.py --out .data/box_pnp_tray_140 .data/box_pnp_s0 .data/box_pnp_s1

# tỉ lệ thành công / lý do loại theo từng pha
.venv-datagen/Scripts/python scripts/datagen/box_pnp/summarize_log.py .data/box_pnp_s0
```

**2. Export cho Ψ₀**

```bash
.venv-datagen/Scripts/python scripts/datagen/box_pnp/export_psi0.py .data/box_pnp_tray_140 --out .data/psi0 --zip
```

Lệnh tạo các pack `<name>_train` / `<name>_val`, video 320×240, cùng file `.data/psi0/box_pnp_tray_140_psi0.zip`.
Tải file zip lên `My Drive/psi0/` của tài khoản sẽ chạy Colab.

**3. Finetune trên Colab**: [`colab_finetune_psi0.ipynb`](scripts/datagen/box_pnp/colab_finetune_psi0.ipynb)

- Chạy lần lượt các cell 1 → 9. Cell 9 phải được giữ chạy, vì nó vừa train vừa tải checkpoint lên Drive.
- Warm start từ `sonic-wbcbox` ckpt 40000. Global batch 128. Mặc định `MAX_STEPS` = 15000; 8000 là đủ để thử.
- GPU L4 dùng torch cu126 + flash-attn. GPU G4 (RTX PRO 6000 Blackwell) dùng torch cu128 + SDPA. Cell 3 tự chọn.
- Checkpoint nằm ở `My Drive/psi0/runs/finetune/<run>/checkpoints/ckpt_<N>/`. Nếu Colab ngắt, chạy lại notebook là tự resume.
- Khóa W&B (nếu dùng) đặt trong **Colab Secrets**, không ghi vào notebook.

**4. Đánh giá closed-loop**

- Trên Colab, chạy [`colab_psi0_server.ipynb`](scripts/datagen/box_pnp/colab_psi0_server.ipynb), cell 1 → 7. Notebook tải checkpoint từ Drive,
  chạy `serve_psi0_sonic_http`, mở tunnel Pinggy rồi in ra lệnh để chạy ở máy local.
- Trên máy local:

```bash
.venv-datagen/Scripts/python scripts/datagen/box_pnp/closed_loop_eval.py --server http://HOST:PORT --episodes 10 --out .data/eval/run1
```

  - `--viewer` mở cửa sổ MuJoCo để xem trực tiếp.
  - `--video` ghi mp4 cho từng episode. Tùy chọn này chậm vì phải render mỗi khung hình, nên chỉ dùng cho vài episode.
  - Seed đánh giá bắt đầu từ 1 000 000, không trùng với dữ liệu train.
  - Kiểm tra pipeline mà không cần model: `--replay .data/box_pnp_tray_140 --replay-episode 0` phát lại action đã ghi.
  - `results.jsonl` ghi kết quả từng episode, kèm chẩn đoán: gắp lúc nào, thùng được nâng bao cao, cuối episode thùng nằm ở đâu.

## Kết quả hiện tại

Checkpoint finetune khoảng 8000 bước trên 126 episode train: **1/10** cảnh chưa thấy thành công. Phần lớn lỗi nằm ở
**bước gắp**: trượt thùng rồi loay hoay đến hết 60 s, hoặc làm rơi thùng. Khi đã gắp được thì robot thường mang đi và
đặt vào khay thành công. Hướng cải thiện tiếp:

- Chạy với `--exec-steps 8` để model lập lại kế hoạch thường hơn.
- Thêm dữ liệu cho bước gắp.
- So sánh nhiều checkpoint.

## Ghi chú

- **Thứ tự state (43):** chân 12, eo 3, tay trái 7, tay phải 7, bàn tay trái 7, bàn tay phải 7. `export_psi0.py`
  hoán vị `qpos[7:50]` của MJCF theo tên khớp. Phần mô tả thứ tự state trong `docs/sonic_pick_can/01` và `02` được viết
  trước khi kiểm tra lại, nên khi hai nơi khác nhau thì tin theo code.
- **Grasp assist:** khi tay phải khép ≥ 30%, có tiếp xúc và hộp nằm trong lòng bàn tay, một weld mềm giữ hộp. Mô hình
  Dex3 tự nó không giữ nổi hộp. Cơ chế này được dùng cả khi sinh dữ liệu lẫn khi đánh giá.
- **Phiên bản SONIC:** mặc định SONIC 1.1 (`sonic_v1_1`), khớp với các script trong `real/SONIC` của Psi0. `meta/info.json`
  ghi `sonic_version`; dữ liệu, checkpoint finetune và decoder khi deploy/đánh giá phải cùng một phiên bản.
- `colab_gr00t_*.ipynb` là các notebook cũ dùng cho GR00T, giữ lại để tham khảo.

Code của Psi0 gốc theo license của repo đó.
