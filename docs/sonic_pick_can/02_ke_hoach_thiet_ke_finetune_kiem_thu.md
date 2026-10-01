# Kế hoạch: thiết kế, finetune (Colab) và kiểm thử Ψ₀ + SONIC cho task "cúi nhặt lon"

> Tài liệu 2/2. Cách hệ thống vận hành (kiến trúc, định dạng dữ liệu, runtime) nằm ở [01_luong_pipeline_psi0_sonic.md](01_luong_pipeline_psi0_sonic.md). Tài liệu này trả lời câu hỏi **làm gì, theo thứ tự nào, chạy lệnh gì, và thế nào là đạt**.
>
> Ký hiệu: ✅ là tiêu chí nghiệm thu, ⚠️ là giả định cần xác minh, 🔧 là việc phải sửa hoặc viết code.

---

## 0. Mục tiêu và phạm vi

| | |
|---|---|
| **Task** | G1 (29 DoF + 2 tay Dex3) đứng trước một lon nằm/đứng trên sàn, cúi xuống, cầm lon bằng tay phải, đứng thẳng dậy và giữ lon |
| **Câu lệnh** | "bend down and pick up the can" (kèm 4–6 cách diễn đạt khác) |
| **Policy** | Ψ₀ (Qwen3-VL-2B + action expert), xuất chunk 30×78 = token SONIC 64 + tay 14 |
| **Controller** | SONIC giải mã token thành chuyển động toàn thân |
| **Dữ liệu** | 100% tự sinh trong MuJoCo, có domain randomization và lọc episode thành công |
| **Train** | Google Colab, 1 GPU (A100 80GB là ưu tiên, A100 40GB hoặc L4 cũng chạy được) |
| **Ngoài phạm vi (giai đoạn này)** | Sim-to-real trên robot thật, tune VLM, đa task |

**Định nghĩa "xong" của toàn dự án:**

- ✅ Closed-loop trong MuJoCo đạt **≥ 70% thành công ở DR level 0** và **≥ 50% ở level 1**, tính trên 50 episode mỗi mức với seed chưa từng thấy.
- ✅ Tỉ lệ ngã ≤ 10%.

---

## 1. Các quyết định thiết kế

| # | Quyết định | Lý do | Phương án thay thế |
|---|---|---|---|
| D1 | Dùng layout **neckless 78/43** (token 64 + tay 14; state 43 theo layout SIMPLE) | Sim không có cổ. Trùng khớp với checkpoint `sonic-wbcbox`, nên nạp được **toàn bộ** action header (xem 01 §1.3) | Layout 80/45 (robot thật): phải pad, và header chỉ nạp được một phần |
| D2 | Warm-start từ **`sonic-wbcbox.neckle...2608260223/ckpt_40000`** | Đã học không gian token SONIC trên chính task cúi và bê trong sim, cùng camera và cùng layout. Gần task nhặt lon nhất | `postpre.sonic1.0.unifolm...40k` (80/45, 12 block): tốt hơn cho sim-to-real sau này |
| D3 | **Đóng băng VLM** (`--model.no-tune-vlm`), chỉ train action header | Vừa bộ nhớ Colab (VLM bf16 khoảng 4 GB, header khoảng 0.5B tham số × AdamW fp32 khoảng 8 GB) | `--model.tune-vlm` + gradient checkpointing, chỉ với A100 80GB, ở Phase 5 |
| D4 | Train **không RTC** (`--model.no-rtc`), dùng **test-time RTC** khi chạy thử | Theo release note Ψ₀-SONIC, cách này hồi phục lỗi tốt hơn | Train-time RTC (`--model.rtc`) |
| D5 | Dữ liệu 30 fps, chunk 30 (1 s), ảnh 240×320 | Trùng recipe `finetune-sonic-simple-psi0.sh` | — |
| D6 | **Action ghi lại là lệnh thực sự gửi cho SONIC** (token FSQ + mục tiêu tay), state là qpos **đo được** | Ψ₀ học đúng tín hiệu mà nó sẽ phải phát ra lúc deploy | Ghi quỹ đạo tham chiếu: sai lệch phân phối |
| D7 | Global batch **128** | README yêu cầu giữ global batch 128 cho mọi thí nghiệm finetune | — |
| D8 | Chọn đường sinh dữ liệu theo phần cứng (xem §3.1) | Isaac Sim (cần cho SIMPLE) không chạy được trên Colab | — |

---

## 2. Phase 0: Spike xác minh (2–3 ngày)

Mục đích: xoá hết các ⚠️ trong tài liệu 01 trước khi viết code.

```bash
git submodule update --init --recursive third_party/SIMPLE third_party/GR00T-WholeBodyControl
```

| ID | Câu hỏi | Tìm ở đâu | Ảnh hưởng nếu sai |
|---|---|---|---|
| Q1 | SIMPLE sinh dữ liệu `...Sonic-v0` bằng cách nào (motion planning, teleop, hay SONIC trong vòng lặp)? Token được tính ở đâu? | `third_party/SIMPLE/src/simple/tasks/**`, `simple/cli/datagen.py`, grep `sonic`, `token` | Quyết định chọn đường A hay B |
| Q2 | Input spec của **SONIC encoder**: chuyển động robot (qpos tương lai) hay SMPL? Cửa sổ bao nhiêu frame, bao nhiêu Hz? Input và output của decoder? | `GR00T-WholeBodyControl/gear_sonic/**` (encoder/decoder, `run_sim_loop.py`), file ONNX | Đường B không chạy được nếu sai |
| Q3 | **Phiên bản SONIC** mà dữ liệu `wbcbox` dùng (`sonic_v1_1` hay `release`)? | SIMPLE config, `real/SONIC/scripts/sonic_conda_env.sh` | Token sai phiên bản thì decoder hiểu sai hoàn toàn |
| Q4 | Thứ tự 7 khớp tay Dex3 và thứ tự 43 state trong SIMPLE so với MJCF | SIMPLE robot config, `simple_to_sonic_lerobot.py:16-18` | Warm-start mất tác dụng, policy lệch |
| Q5 | Vị trí và intrinsics của camera `ego_view` trong SIMPLE | SIMPLE MJCF / camera config | Ảnh khác phân phối lúc warm-start |
| Q6 | File zip `G1WholebodyXMoveBendCarryBoxSonic-v0` trên HF đã ở schema loader (có cột `action.body_token`) hay còn schema gốc? | Giải nén và đọc 1 parquet (cell C4) | Có cần chạy converter hay không |
| Q7 | `run_config.json` của `wbcbox` có khoá `model.final_layer_norm` không? Giá trị `qk_norm`, `num_blocks`, `dropout` là gì? | Tải về và đọc (cell C3) | **Nếu thiếu `final_layer_norm` thì checkpoint là legacy, bắt buộc thêm `--model.no-final-layer-norm`.** Nếu không, output head bị bỏ qua âm thầm (nạp với `strict=False`) và bị khởi tạo lại |
| Q8 | SIMPLE có chế độ render chỉ bằng MuJoCo (không Isaac) không? | `--sim-mode` choices | Có thể chạy SIMPLE trên Colab |

✅ **Nghiệm thu Phase 0**: có bảng trả lời Q1–Q8 và một quyết định đường A/B/C đã được chốt.

---

## 3. Phase 1: Thiết kế task và bộ sinh dữ liệu (1–2 tuần)

### 3.1 Chọn đường sinh dữ liệu

```
Có máy Ubuntu 22.04 + GPU RTX (≥ 3090) và Q1 cho thấy SIMPLE tự sinh được dữ liệu Sonic?
 ├─ Có  → ĐƯỜNG A: task SIMPLE mới "G1WholebodyBendPickCanSonic-v0"
 └─ Không → ĐƯỜNG B: generator MuJoCo thuần (chạy được trên Colab, MUJOCO_GL=egl)
Sau khi A hoặc B chạy ổn, nếu chuyển động kém tự nhiên thì bổ sung ĐƯỜNG C (MimicGen từ demo teleop mồi).
```

### 3.2 Đặc tả scene

| Thành phần | Giá trị mặc định | Randomization (level 0 / 1 / 2) |
|---|---|---|
| Robot | G1 29 DoF + Dex3, đứng thẳng, qpos mặc định của SONIC | yaw ban đầu: 0 / ±10° / ±20° |
| Lon | trụ r = 3.3 cm, h = 12.2 cm, m = 0.35 kg (lon soda 330 ml) | vị trí x ∈ [0.40, 0.45] / [0.35, 0.50] / [0.30, 0.55] m, y ∈ ±0.05 / ±0.12 / ±0.18 m; đứng hoặc nằm (chỉ level 2); 3 / 8 / 15 texture và màu |
| Vật lý | ma sát lon–tay 1.0, `condim=4` | ± 0 / 15 / 30% ma sát và khối lượng |
| Sàn và ánh sáng | sàn xám, 1 đèn | 1 / 5 / 20 texture sàn; cường độ và hướng đèn |
| Camera ego | theo Q5 | jitter 0 / ±1° / ±2°, FOV ±0 / 2 / 4° |
| Vật gây nhiễu | không | 0 / 0–1 / 0–3 vật khác trên sàn |

### 3.3 Đặc tả expert (đường B)

Expert tạo **chuyển động tham chiếu kinematic** cho G1. Chuyển động này được đưa qua SONIC encoder để ra token, rồi SONIC decoder bám theo trong vật lý. Tay được điều khiển trực tiếp bằng mục tiêu khớp.

| Pha | Thời lượng | Tham chiếu |
|---|---|---|
| 1. settle | 0.5 s | đứng yên |
| 2. bend | 1.5–2.0 s | pelvis z 0.75 → 0.42–0.50 m (tuỳ vị trí lon), gập hông và gối; giữ CoM chiếu trong đa giác chân |
| 3. reach | 1.0 s | IK (mink) cho cổ tay phải tới điểm pre-grasp cách lon 5 cm, lòng bàn tay hướng vào lon |
| 4. approach + grasp | 0.8 s | tiến 5 cm rồi đóng Dex3 theo profile, giới hạn lực (PD) |
| 5. lift | 0.7 s | nâng cổ tay 15 cm |
| 6. stand | 1.5–2.0 s | trở về đứng thẳng, giữ lon trước ngực |
| 7. hold | 1.0 s | đứng yên (dùng để kiểm tra thành công) |

- Nội suy giữa các keyframe bằng min-jerk. Thêm nhiễu thời lượng ±15% và nhiễu keyframe nhỏ để đa dạng hoá.
- **Tiêu chí thành công**: lon cao hơn ban đầu > 0.10 m, có tiếp xúc tay–lon trong ≥ 1 s cuối, pelvis z > 0.5 m, |tilt torso| < 30°, không có va chạm đầu gối/tay với sàn quá ngưỡng.
- **Không dùng "magic grasp"** (weld/attach): dữ liệu phải phản ánh vật lý thật của việc cầm.

🔧 Bố cục code đề xuất (đường B), đặt trong `scripts/datagen/pick_can/`:

| File | Trách nhiệm |
|---|---|
| `scene.py` | Ghép MJCF: G1 (lấy từ GR00T-WBC/SIMPLE), lon, sàn; áp randomization theo level |
| `sonic_runner.py` | Chạy encoder/decoder ONNX bằng onnxruntime-gpu; FSQ quantize; vòng 50 Hz |
| `expert.py` | Keyframe → IK (mink) → quỹ đạo tham chiếu; profile tay |
| `recorder.py` | Ghi LeRobot v2.1 **đúng schema gốc của SIMPLE** (`observation.images.ego_view`, `action` 78, `observation.state` 43, `task_index`) ở 30 fps, để tái dùng `simple_to_sonic_lerobot.py` mà không phải sửa |
| `generate.py` | Vòng seed, lọc thành công, chạy song song bằng multiprocessing, log tỉ lệ thành công của expert |
| `qa_check.py` | Kiểm tra chất lượng dữ liệu (§3.5) |
| `closed_loop_eval.py` | Dùng lại `scene.py` + `sonic_runner.py`, thay expert bằng client gọi Ψ₀ `/act` (Phase 4, mức L4) |

Với đường A thì không cần các file trên (trừ `qa_check.py`): chỉ cần thêm task vào SIMPLE bằng cách clone cấu hình `XMoveBendCarryBoxSonic`, thay hộp bằng lon và thay điều kiện thành công.

### 3.4 Kích thước dữ liệu

| Đợt | Số episode thành công | Mục đích |
|---|---|---|
| Pilot | 50 (level 0) | Kiểm tra pipeline end-to-end, overfit test |
| Main v1 | 400 (200 L0 + 150 L1 + 50 L2) | Finetune chính. Ψ₀ học được task mới từ khoảng 80 quỹ đạo (README), để dư cho randomization |
| Main v2 | 800+ | Chỉ khi closed-loop L1 < 50% |

Chia **val = 10% số episode, chia theo episode** (không chia theo frame) bằng `scripts/data/split_lerobot_train_val.py --val-episodes ...`. ⚠️ Script ghi là dành cho LeRobot v3, nhưng đọc path template từ `info.json`. Phải chạy thử trên pilot trước; nếu lỗi thì chia bằng cách sinh hai thư mục riêng với seed khác nhau.

### 3.5 Kiểm tra chất lượng dữ liệu (`qa_check.py`)

✅ Mỗi dataset phải qua đủ các kiểm tra sau:

- [ ] `info.json` có `fps == 30`; số frame video của mỗi episode bằng số dòng parquet.
- [ ] `action` có shape (78,), `observation.state` có shape (43,); không có NaN/Inf.
- [ ] `action[:, :64]` nằm trên lưới FSQ: `abs(round(x/0.0625)*0.0625 - x) < 1e-5` và `|x| ≤ 0.625`.
- [ ] Không có chiều action hay state nào có `max − min < 1e-4`, trừ những chiều hằng đã biết (ghi rõ lý do). Chiều hằng sẽ được bỏ qua khi chuẩn hoá.
- [ ] Phân phối chiều dài episode: không có episode < 3 s hoặc > 15 s.
- [ ] Xem 10 video ngẫu nhiên: lon nằm trong khung hình ego ở frame đầu, tay nằm trong khung lúc cầm.
- [ ] Replay 5 episode (gửi lại token và tay đã ghi, open-loop, trong cùng seed): lon vẫn được nhấc lên. Kiểm tra này chứng minh action ghi lại đủ để tái tạo hành vi.

---

## 4. Phase 2: Dựng Colab và chạy khói trên dữ liệu có sẵn (1–2 ngày)

Mục đích: chứng minh môi trường Colab train và serve được **trước khi** có dữ liệu mới. Dùng dataset `G1WholebodyXMoveBendCarryBoxSonic-v0` trên HF.

### 4.1 Yêu cầu runtime

| | Khuyến nghị | Tối thiểu |
|---|---|---|
| GPU | A100 80GB | A100 40GB hoặc L4 24GB. **Không dùng T4**: flash-attn 2 bị hard-code khi train và cần GPU sm80+ |
| RAM hệ thống | High-RAM | Dataloader mặc định dùng **12 worker** (`finetune.py:571-590`), RAM thường dễ bị OOM |
| Đĩa | khoảng 60 GB trống trong `/content` | checkpoint warm-start 6.25 GB + bản export khoảng 6 GB + dữ liệu + mỗi checkpoint lúc train khoảng 10 GB |
| Lưu trữ lâu dài | Google Drive (dataset đã xử lý, thư mục `.runs`) | — |

### 4.2 Các cell Colab

> Mỗi cell `%%bash` là một shell mới, nên cell nào cũng phải `source` lại venv và `.env`. Biến `REPO_URL` trỏ tới repo (hoặc fork) có thư mục `docs/sonic_pick_can`.

**C0: Kiểm tra GPU và mount Drive**

```python
!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")            # Colab Secrets
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")  # hoặc os.environ["WANDB_MODE"] = "offline"
os.environ["REPO_URL"] = "https://github.com/physical-superintelligence-lab/Psi0.git"
```

**C1: Cài môi trường (khoảng 10–15 phút)**

```bash
%%bash
set -e
cd /content
[ -d Psi0 ] || git clone "$REPO_URL" Psi0
cd Psi0
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv python install 3.11
uv venv .venv-psi --python 3.11
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group psi --index-strategy unsafe-best-match --active
# Wheel flash-attn dựng sẵn, cùng bản mà scripts/train/Dockerfile dùng (cu12 / torch 2.7 / cp311)
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
python -c "import psi, torch, flash_attn; print(psi.__version__, torch.__version__, torch.cuda.get_device_name(0), flash_attn.__version__)"
python -c "from psi.data.lerobot.compat import LEROBOT_LAYOUT; print(LEROBOT_LAYOUT)"
```

**C2: File `.env`** (`scripts/train.py` và `calc_modality_stats.py` đều assert là file này tồn tại)

```bash
%%bash
cat > /content/Psi0/.env <<'EOF'
PSI_HOME=/content/psi_home
DATA_HOME=/content/psi_home/data
HF_HOME=/content/psi_home/cache/hf
TORCH_HOME=/content/psi_home/cache/hf
HF_LEROBOT_HOME=/content/psi_home/data/lerobot
OMP_NUM_THREADS=8
TOKENIZERS_PARALLELISM=false
CUDA_LAUNCH_BLOCKING=0
TF_CPP_MIN_LOG_LEVEL=3
AV_LOG_LEVEL=quiet
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
NO_ALBUMENTATIONS_UPDATE=1
EOF
mkdir -p /content/psi_home/{data,cache/checkpoints}
```

**C3: Tải checkpoint warm-start và export (trả lời Q7)**

```bash
%%bash
set -e
cd /content/Psi0 && source .venv-psi/bin/activate && set -a && source .env && set +a
CK=$PSI_HOME/cache/checkpoints
WB_NAME=sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223
PRE=psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k
hf download USC-PSI-Lab/psi-model --include "psi0/simple-checkpoints/$WB_NAME/*" --local-dir $CK
# export chỉ cần các file config/tokenizer của VLM gốc, không cần trọng số
hf download USC-PSI-Lab/psi-model --include "$PRE/*.json" "$PRE/*.txt" "$PRE/*.jinja" --local-dir $CK
WB=$CK/psi0/simple-checkpoints/$WB_NAME
ls -la $WB/checkpoints/ckpt_40000
python scripts/export_psi0_ckpt.py $WB/checkpoints/ckpt_40000 $CK/psi0/init-wbcbox-40k \
    --hf-from $CK/$PRE --dtype bf16
# Q7: xác định cờ kiến trúc cần khớp
python - <<PY
import json; m = json.load(open("$WB/run_config.json"))["model"]
for k in ["final_layer_norm","qk_norm","num_blocks","vlm_layer_indices","dropout",
          "state_feature_dropout","state_as_action_token","combined_temb","action_dim","odim"]:
    print(f"{k:24s}", m.get(k, "<THIẾU -> legacy>"))
PY
```

> ✅ Nếu `final_layer_norm` hiện `<THIẾU -> legacy>` hoặc `False`, **mọi lệnh train phía sau phải thêm `--model.no-final-layer-norm`**. Nếu `dropout` hoặc `state_feature_dropout` thiếu thì giá trị legacy là 0.1 / 0.2, trùng với mặc định hiện tại.

**C4: Tải dataset có sẵn, kiểm tra schema (Q6) và chuyển đổi nếu cần**

```bash
%%bash
set -e
cd /content/Psi0 && source .venv-psi/bin/activate && set -a && source .env && set +a
T=G1WholebodyXMoveBendCarryBoxSonic-v0
hf download USC-PSI-Lab/psi-data simple/$T.zip --repo-type dataset --local-dir $PSI_HOME/data
unzip -q -o $PSI_HOME/data/simple/$T.zip -d $PSI_HOME/data/simple
find $PSI_HOME/data/simple/$T -maxdepth 3 -name info.json
python - <<PY
import glob, pandas as pd
p = sorted(glob.glob("$PSI_HOME/data/simple/$T/**/episode_*.parquet", recursive=True))[0]
df = pd.read_parquet(p); print(p); print(df.columns.tolist())
print({c: getattr(df[c].iloc[0], "shape", None) for c in df.columns})
PY
# Nếu KHÔNG có cột action.body_token (schema gốc SIMPLE) thì chuyển đổi.
# --copy-videos: Drive không hỗ trợ symlink tốt.
# python scripts/data/simple_to_sonic_lerobot.py --src <thư mục level chứa data/ meta/ videos/> \
#     --out $PSI_HOME/data/sonic_sim/$T --copy-videos
```

**C5: Chạy thử cấu hình, chưa train**

```bash
%%bash
cd /content/Psi0 && source .venv-psi/bin/activate && set -a && source .env && set +a
python - <<'PY'
import os
from psi.config.train.finetune_sonic_psi0_config import DynamicDataConfig  # import được là đủ
print("config import OK")
PY
```

Sau đó chạy **C6 với `MAX_STEPS=200`, `CKPT_EVERY=100`, `VAL_EVERY=100`**. Đây là bài khói (smoke test).

**C6: Lệnh train (dùng chung cho Phase 2 và Phase 3)**

```bash
%%bash
set -e
cd /content/Psi0 && source .venv-psi/bin/activate && set -a && source .env && set +a
# ---- tham số (sửa ở đây) ----
EXP=${EXP:-pickcan}
DATA_ROOT=${DATA_ROOT:-$PSI_HOME/data/simple}
TRAIN_ID=${TRAIN_ID:-G1WholebodyXMoveBendCarryBoxSonic-v0}
VAL_ID=${VAL_ID:-$TRAIN_ID}
INIT=$PSI_HOME/cache/checkpoints/psi0/init-wbcbox-40k
BS=${BS:-16}; ACC=${ACC:-8}             # A100-80: 32/4 · A100-40: 16/8 · L4: 8/16  → global 128
MAX_STEPS=${MAX_STEPS:-15000}; CKPT_EVERY=${CKPT_EVERY:-2500}; VAL_EVERY=${VAL_EVERY:-2500}
LEGACY_HEAD=${LEGACY_HEAD:---model.no-final-layer-norm}   # đặt rỗng nếu Q7 cho final_layer_norm=True
TS=${TS:-}                              # đặt TS=<yymmddHHMM> để resume đúng run
OUT=/content/drive/MyDrive/psi0/.runs   # lưu trên Drive để sống sót qua disconnect
# -----------------------------
python scripts/train.py finetune_sonic_psi0_config \
  --seed=292285 --exp=$EXP \
  ${TS:+--timestamp=$TS --train.resume_from_checkpoint=latest} \
  --train.name=finetune --train.data_parallel=ddp --train.mixed_precision=bf16 \
  --train.output_dir=$OUT \
  --train.train_batch_size=$BS --train.gradient_accumulation_steps=$ACC \
  --train.learning_rate=1e-4 --train.max_training_steps=$MAX_STEPS \
  --train.warmup_ratio=None --train.warmup_steps=500 \
  --train.checkpointing_steps=$CKPT_EVERY --train.max_checkpoints_to_keep=2 \
  --train.validation_steps=$VAL_EVERY --train.val_num_batches=5 \
  --train.max_grad_norm=1.0 --train.optimizer_foreach=False \
  --train.lr_scheduler_type=cosine \
  --train.lr_scheduler_kwargs.weight_decay=1e-6 \
  --train.lr_scheduler_kwargs.betas 0.95 0.999 \
  --log.report_to=wandb \
  --data.root_dir=$DATA_ROOT --data.train_repo_ids=$TRAIN_ID --data.val_repo_ids=$VAL_ID \
  --data.transform.repack.action-keys action.body_token 'action[:14]' \
  --data.transform.repack.dataset-name=neckless \
  --data.transform.repack.pad-action-dim=78 --data.transform.repack.pad-state-dim=43 \
  --data.transform.repack.instruction-key=task_description \
  --data.transform.field.stat-path=$DATA_ROOT/$TRAIN_ID/meta/stats_psi0.json \
  --data.transform.field.stat-action-keys action.body_token 'action[:14]' \
  --data.transform.field.action_norm_type=bounds --data.transform.field.normalize-state \
  --data.transform.field.pad-action-dim=78 --data.transform.field.pad-state-dim=43 \
  --data.transform.model.img-aug \
  --data.transform.model.resize.size 240 320 --data.transform.model.center_crop.size 240 320 \
  --model.model_name_or_path=$INIT --model.pretrained-action-header-path=$INIT \
  --model.noise-scheduler=flow --model.train-diffusion-steps=1000 --model.n_conditions=0 \
  --model.action-chunk-size=30 --model.action-dim=78 --model.action-exec-horizon=30 \
  --model.observation-horizon=1 --model.odim=43 --model.view_feature_dim=2048 \
  --model.no-tune-vlm --model.no-use_film --model.no-combined_temb \
  --model.no-rtc --model.max-delay=8 $LEGACY_HEAD 2>&1 | tee /content/train_$EXP.log
```

- Chạy một tiến trình bằng `python` là được; **không cần `torchrun`** khi dùng 1 GPU (overwatch tự nhận world_size = 1).
- Ngay sau khi model được nạp, kiểm tra log: ✅ **không có key `action_proj_out.*` nào trong danh sách missing/unexpected** của action header. Nếu có, nghĩa là cờ `final_layer_norm` sai.
- Resume sau khi Colab ngắt kết nối: lấy `<ts>` ở cuối tên thư mục run trong `$OUT/finetune/`, rồi chạy lại C6 với `TS=<ts>`.

**C7: Đánh giá open-loop (L2)**

[examples/psi0/psi0_inference_simple_sonic.py](../../examples/psi0/psi0_inference_simple_sonic.py) đang hard-code `run_dir`, `ckpt_step`, `eps_idx`. 🔧 Cần sửa thành tham số dòng lệnh, hoặc chép sang notebook rồi sửa 3 biến này. Kết quả in ra `denormed_err_l1_action.body_token` và `denormed_err_l1_action[:14]`.

**C8: Serve và client mock (L3)**

```bash
%%bash
cd /content/Psi0 && source .venv-psi/bin/activate && set -a && source .env && set +a
RUN=<đường dẫn run>; STEP=<bước>
nohup serve_psi0_sonic_http --policy psi0 --port 8014 --run-dir $RUN --ckpt-step $STEP \
      --rtc --action-exec-horizon 15 > /content/serve.log 2>&1 &
sleep 90; curl -s localhost:8014/health; echo
python src/psi/deploy/mock_psi0_client_sonic_http.py --run-dir $RUN --host localhost --port 8014 \
      --eps-idx 0 --target-hz 30 --output-dir /content/mock_out
```

### 4.3 Nghiệm thu Phase 2

- ✅ C1–C6 chạy hết mà không lỗi; loss giảm trong 200 bước khói.
- ✅ Đo được **throughput** (giây/bước) và **VRAM đỉnh**. Hai số này dùng để hiệu chỉnh `BS/ACC/MAX_STEPS` ở Phase 3 và ghi vào bảng §5.2.
- ✅ Open-loop L1 của **checkpoint `wbcbox` gốc** trên dataset BendCarryBox đã được ghi lại. Đây là **baseline tham chiếu** cho mức sai số "tốt".
- ✅ Client mock chạy được, độ trễ `/act` trung bình < 300 ms trên A100 (8 bước khử nhiễu).

🔧 Hai bản vá nhỏ nên làm trong Phase 2 (mỗi bản vài dòng):

1. Thêm `num_workers: int | None = None` và `prefetch_factor: int | None = None` vào `LerobotDataConfig` ([src/psi/config/data_lerobot.py](../../src/psi/config/data_lerobot.py)). Trainer đã đọc hai trường này qua `getattr` ([finetune.py:571-578](../../src/psi/trainers/finetune.py#L571-L578)) nhưng chưa có cách truyền vào. Nhờ vậy chỉnh được `--data.num-workers=6` cho Colab có ít CPU.
2. Cho `examples/psi0/psi0_inference_simple_sonic.py` nhận `--run-dir --ckpt-step --eps-idx` (C7).

---

## 5. Phase 3: Finetune trên dữ liệu nhặt lon (2–4 ngày)

### 5.1 Chuẩn bị dữ liệu

```bash
# 1) Output của generator (schema gốc SIMPLE) → schema loader
python scripts/data/simple_to_sonic_lerobot.py --src <raw_pickcan> \
    --out $PSI_HOME/data/sonic_sim/G1BendPickCanSonic-v0 --copy-videos
# 2) Chia train/val theo episode
python scripts/data/split_lerobot_train_val.py \
    --src   $PSI_HOME/data/sonic_sim/G1BendPickCanSonic-v0 \
    --train $PSI_HOME/data/sonic_sim/G1BendPickCanSonic-v0_train \
    --val   $PSI_HOME/data/sonic_sim/G1BendPickCanSonic-v0_val \
    --val-episodes <danh sách 10% episode> --mode copy
# 3) Kiểm tra chất lượng (§3.5)
python scripts/datagen/pick_can/qa_check.py $PSI_HOME/data/sonic_sim/G1BendPickCanSonic-v0_train
```

⚠️ **Thống kê chuẩn hoá**: nên giữ `stats_psi0.json` của **dữ liệu mới**. Tuy nhiên, nếu dải của token hoặc tay khác xa dataset `wbcbox`, header warm-start sẽ gặp phân phối action đã chuẩn hoá khác trước. Cần so sánh min/max của hai file stats. Nếu 64 chiều token đều có dải xấp xỉ [−0.625, 0.625] thì không có vấn đề.

### 5.2 Kế hoạch chạy

| Run | Dữ liệu | Bước | Mục đích | Cấu hình khác biệt |
|---|---|---|---|---|
| R0 overfit | 2 episode (train = val) | 500–1000 | **L1**: loss → ~0, L1 open-loop gần 0 | `BS=8 ACC=1`, tắt `img-aug` |
| R1 pilot | 50 episode | 5 000 | Kiểm tra closed-loop sớm | mặc định C6 |
| R2 main | 400 episode | 15 000 (đánh giá tại 5k/10k/15k) | Run chính | mặc định C6 |
| R3 robust | 400 episode | 15 000 | Chống lệch state và camera | thêm `--data.transform.repack.state-temporal-jitter=10 --data.transform.repack.state-temporal-jitter-prob=0.5 --data.transform.field.state-noise-std=0.05 --data.transform.model.view-aug --data.transform.model.view-aug-min-scale=0.85` |
| R4 (tuỳ chọn) | 400 episode | 15 000 | Header mới 12 block | chỉ khi R2/R3 < mục tiêu; cần A100 80GB |

Ước tính thời gian: thay bằng số đo ở Phase 2. Colab thường ngắt sau vài giờ đến 24 giờ, nên với R2/R3 phải có kế hoạch resume (C6 với `TS=`). Checkpoint lưu thẳng lên Drive, giữ 2 bản gần nhất; các bản ở bước chia hết cho 10k được giữ nguyên theo `preserve_checkpoint_every`.

### 5.3 Theo dõi trong lúc train (W&B)

- `train/loss` giảm đều. Nếu gradient không hữu hạn, trainer bỏ qua bước đó và log lại. ✅ Số bước bị bỏ qua < 0.1%.
- `val/...denormed_err_l1` tách theo `latent_sonic` và `hand_joints`. Chọn checkpoint theo **closed-loop**, không chỉ theo val L1.
- `lr` theo từng nhóm: chỉ có nhóm `action_header` (VLM đóng băng).

---

## 6. Phase 4: Kiểm thử (3–5 ngày, chạy xen kẽ với Phase 3)

| Mức | Khi nào | Cách làm | ✅ Tiêu chí |
|---|---|---|---|
| **L0 tĩnh** | mỗi dataset, mỗi lần đổi cấu hình | `qa_check.py`; C5; `DRY_RUN` | 100% qua checklist §3.5; validator config không lỗi |
| **L1 overfit** | R0 | C6 trên 2 episode | loss giảm > 95%; L1 token đã denormalize trên chính episode đó < 0.01 |
| **L2 open-loop** | mỗi checkpoint 5k | C7 trên val | L1 token ≤ 1.2 × baseline `wbcbox` (Phase 2); L1 tay < 0.05 rad |
| **L3 server** | checkpoint ứng viên | C8 | L1 qua server ≈ L2 (±5%); độ trễ trung bình < 300 ms, p95 < 500 ms |
| **L4 closed-loop** | checkpoint ứng viên | `closed_loop_eval.py` (đường B) hoặc `eval-sonic-wbc` của SIMPLE (đường A); 50 seed chưa thấy mỗi DR level | Pilot: ≥ 40% ở L0. Main: **≥ 70% ở L0, ≥ 50% ở L1**, ngã ≤ 10% |
| **L5 robustness** | checkpoint cuối | L4 có nhiễu: trễ state 3–5 frame, camera lệch 2°, lon màu mới, có vật gây nhiễu | Tỉ lệ thành công giảm ≤ 15 điểm % so với L4 |

Chỉ số cần ghi cho mỗi lần chạy L4: tỉ lệ thành công, tỉ lệ ngã, thời gian đến lúc thành công, số lần cầm trượt, phân loại lỗi (cúi không đủ sâu / với trượt / cầm trượt / mất thăng bằng / đứng dậy làm rơi), video 5 thành công và 5 thất bại.

Closed-loop với SONIC trong MuJoCo (đường B), mỗi 1/30 s:

```
obs = render(ego) ; s = qpos43 (đo được)
nếu hết action trong hàng đợi (hoặc đã dùng 15/30 bước): chunk = POST /act {image, state:s, instruction}
a = chunk.pop(0)
token = fsq_quantize(a[:64]) ; tay = a[64:78]
SONIC decoder (50 Hz, nội suy/giữ token giữa các tick 30 Hz) → mục tiêu khớp thân ; tay → PD Dex3
kết thúc khi thành công, khi ngã, hoặc sau 20 s
```

⚠️ Cách SONIC nhận token ở 30 Hz trong khi chạy 50 Hz (giữ token hay nội suy) phải lấy **đúng như trong sim loop của GR00T-WBC** (Q2), để hành vi lúc sinh dữ liệu và lúc đánh giá giống hệt nhau.

---

## 7. Phase 5: Cải tiến (tuỳ kết quả)

Thứ tự thử, mỗi bước chỉ thay đổi một yếu tố:

1. Thêm dữ liệu (lên 800 episode), tăng tỉ lệ DR level 1/2.
2. Bật robustness knobs (R3), nếu chưa bật.
3. DAgger-lite: lấy các trạng thái mà policy thất bại trong L4, cho expert chạy tiếp từ trạng thái đó, rồi thêm các episode thành công vào dữ liệu.
4. `--model.tune-vlm` với các LR của release note (`lang 1e-6`, `vision 1e-5`, `projector 1e-4`) + `--model.gradient-checkpointing` (chỉ A100 80GB, batch nhỏ, tăng ACC).
5. Chuyển warm-start sang `postpre.sonic1.0.unifolm` (80/45, 12 block, `state-as-action-token`, `combined-temb`). Chỉ làm khi chuẩn bị sim-to-real.

---

## 8. Rủi ro và cách giảm thiểu

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
|---|---|---|---|
| Sai input spec của SONIC encoder (đường B) | Trung bình | Cao | Q2 ở Phase 0; kiểm tra bằng cách replay một chuyển động mẫu của GR00T-WBC và so token với bản của họ |
| Token lệch phiên bản SONIC giữa lúc sinh dữ liệu và lúc đánh giá | Trung bình | Cao | Q3; ghim một bộ ONNX duy nhất, ghi hash vào `meta/info.json` |
| Thứ tự state hoặc tay khác layout SIMPLE | Trung bình | Cao | Q4; so với vài frame dataset `wbcbox` ở cùng tư thế đứng |
| Output head bị bỏ qua âm thầm do cờ `final_layer_norm` | **Cao** nếu quên | Cao | Q7 + kiểm tra log key missing (C6) |
| Expert SONIC không giữ được thăng bằng khi cúi sâu | Trung bình | Trung bình | Hạ độ sâu cúi, kéo dài pha bend, giữ CoM trong đa giác chân; lọc bằng tiêu chí ngã |
| Cầm lon trượt trong MuJoCo | Trung bình | Trung bình | `condim=4/6`, tăng ma sát, profile đóng tay chậm; kiểm tra replay (§3.5) |
| Colab ngắt kết nối hoặc hết compute unit | Cao | Trung bình | Lưu `.runs` lên Drive, resume bằng `TS`, giữ tối đa 2 checkpoint |
| Hết RAM do 12 dataloader worker | Trung bình | Trung bình | High-RAM; bản vá `num_workers` (Phase 2) |
| flash-attn không cài được | Thấp | Cao | Dùng đúng wheel cp311/torch 2.7/cu12 (C1); không dùng T4 |
| Val L1 tốt nhưng closed-loop kém (compounding error) | Cao | Cao | Chọn checkpoint theo L4; bật R3; DAgger-lite |

---

## 9. Lộ trình và checklist

| Tuần | Việc | Đầu ra |
|---|---|---|
| 1 | Phase 0 + Phase 2 (song song) | Bảng Q1–Q8, Colab chạy được, số baseline |
| 2 | Phase 1: generator + QA, pilot 50 episode | Dataset pilot qua QA |
| 3 | R0, R1, L1–L4 pilot; sinh bộ main 400 episode | Báo cáo pilot |
| 4 | R2/R3, L2–L5 | Checkpoint ứng viên + báo cáo đánh giá |
| 5 | Phase 5 nếu chưa đạt mục tiêu | — |

- [ ] Phase 0: đã trả lời Q1–Q8 và chốt đường A/B/C
- [ ] Phase 2: C0–C8 chạy được; đã ghi throughput, VRAM, baseline L1; đã vá `num_workers`
- [ ] Phase 1: `scene.py`, `sonic_runner.py`, `expert.py`, `recorder.py`, `generate.py`, `qa_check.py`
- [ ] Pilot 50 episode qua QA; R0 overfit đạt
- [ ] Main 400 episode qua QA; chia train/val theo episode
- [ ] R2 (và R3) train xong; đã chọn checkpoint theo L4
- [ ] L4 ≥ 70% (DR0), ≥ 50% (DR1), ngã ≤ 10%
- [ ] Đã lưu báo cáo, video, và cấu hình (`argv.txt`, `run_config.json`, hash ONNX của SONIC)
