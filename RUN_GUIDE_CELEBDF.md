# Hướng dẫn chạy trên máy GPU: Celeb-DF + LLM (P0 → P2)

Tài liệu này đi kèm [`ROADMAP_LLM_CELEBDF.md`](ROADMAP_LLM_CELEBDF.md). Mỗi bước ghi rõ: **script**, **đầu vào → đầu ra**, **tài nguyên/thời gian ước lượng**, **cần kiểm tra gì trước khi sang bước sau**.

Toàn bộ pipeline đã được chạy thử end-to-end trên một bộ Celeb-DF giả lập (CPU). **Chưa chạy trên Celeb-DF thật và chưa chạy với InsightFace/Qwen thật**, nên lần đầu nên chạy thử với `--limit-per-folder` / `--limit-videos` / `--n-fake` nhỏ.

Số thời gian bên dưới là ước lượng thô cho 1 GPU 24 GB, không phải số đo.

---

## 0. Cài đặt

```bash
pip install -r requirements.txt
```

```bash
pip install -r requirements-llm.txt
```

Cấu trúc thư mục khuyến nghị (đều đã nằm trong `.gitignore`):

```text
data/celebdf/
├── splits.json          # P0.1
├── frames/              # P0.2  crop 224×224 + _shards/
├── meta.csv             # P0.3  manifest: 1 dòng / frame
├── masks/  masks.csv    # P1.1
├── annotations/         # P1.2–1.3
└── synth/               # P2   D1/, D2/, D3/, probe_val/
results/  runs/  plans/  checkpoints/
```

Tất cả đường dẫn trong `meta.csv` / `synth_meta.csv` là **đường dẫn tương đối kiểu POSIX**, nên chuyển dữ liệu giữa Linux và Windows không phải sửa CSV.

---

## P0 — Dữ liệu và baseline

### P0.1 Chia split — `scripts/prepare_celebdf.py split`

```bash
python scripts/prepare_celebdf.py split --celebdf-root /path/to/Celeb-DF-v2 --out data/celebdf/splits.json --sid-heldout 12 --sid-mode strict
```

- **Đầu vào:** thư mục Celeb-DF v2 gốc (`Celeb-real/`, `Celeb-synthesis/`, `YouTube-real/`, `List_of_testing_videos.txt`).
- **Đầu ra:** `splits.json`, trong đó mỗi video có 2 cột split:
  - `official`: test = 518 video chính thức; phần còn lại chia train/val 90/10 theo video.
  - `sid` (tách identity): giữ lại 12 identity cho test. Fake có cả 2 identity thuộc nhóm giữ lại → test; cả 2 thuộc nhóm còn lại → train; trộn lẫn → `drop` (chế độ `strict`).
- **Thời gian:** vài giây, chỉ CPU.
- **Kiểm tra:** script in số video theo (split, real/fake).
  - Test `official` phải có đúng 518 video (178 real + 340 fake). Nếu có cảnh báo "test videos not found" thì kiểm tra lại thư mục gốc.
  - Nếu `sid/test` có quá ít fake (< 150) thì tăng `--sid-heldout` (ví dụ 16) hoặc dùng `--sid-mode loose`.

### P0.2 Trích frame + crop mặt — `scripts/prepare_celebdf.py extract`

Chạy thử trước với ít video:

```bash
python scripts/prepare_celebdf.py extract --celebdf-root /path/to/Celeb-DF-v2 --splits data/celebdf/splits.json --frames-root data/celebdf/frames_debug --limit-per-folder 5
```

Chạy thật. Có thể chạy song song N tiến trình (mỗi tiến trình một `--shard`, cùng hoặc khác GPU qua `--ctx-id`):

```bash
python scripts/prepare_celebdf.py extract --celebdf-root /path/to/Celeb-DF-v2 --splits data/celebdf/splits.json --frames-root data/celebdf/frames --frames-real 64 --frames-fake 32 --shard 0 --num-shards 4 --ctx-id 0
```

- **Cách làm:**
  - Giải mã tuần tự để lấy đúng chỉ số frame (không dùng seek, vì seek của OpenCV hay lệch).
  - SCRFD (`buffalo_l` của InsightFace) chọn mặt lớn nhất, crop vuông margin 1.3, resize về 224.
  - Lưu kèm **toạ độ crop (x1, y1, side)** và **5 keypoint** trong toạ độ crop. P1 cần toạ độ crop để cắt video real cùng một khung; P1 và P2 cần keypoint để dựng 5 vùng mặt.
- **Lấy mẫu:** 64 frame/video real, 32 frame/video fake. Celeb-DF có tỉ lệ khoảng 1 real : 6.3 fake theo số video, nên sau khi lấy mẫu còn khoảng 1 : 3 theo số frame.
- **Đầu ra:** `frames/<thư mục>/<video>/<frame>.png` và `frames/_shards/meta_shardXXX.jsonl`.
- **Resume:** chạy lại cùng lệnh sẽ bỏ qua các video đã xong.
- **Tài nguyên:** khoảng 57k frame real + 180k frame fake ≈ 237k PNG, tốn **khoảng 20–25 GB** đĩa. Nếu thiếu đĩa, giảm `--frames-fake 16`. Thời gian chủ yếu nằm ở bước giải mã video; ước lượng vài giờ với 4 shard.
- **Kiểm tra:** mở ngẫu nhiên khoảng 20 crop trong `frames_debug` xem mặt có nằm giữa không. Nếu crop bị lệch thì chỉnh `--margin` hoặc `--min-score`.

> Bạn đã có sẵn crop SCRFD (`celebdf_faces_scrfd`), nhưng **vẫn phải trích lại**, vì bản cũ không lưu toạ độ crop và keypoint. P1 cần các thông tin này.

### P0.3 Gộp manifest — `scripts/prepare_celebdf.py merge`

```bash
python scripts/prepare_celebdf.py merge --splits data/celebdf/splits.json --frames-root data/celebdf/frames --out data/celebdf/meta.csv
```

- **Đầu ra:** `meta.csv` gồm các cột `rel_path, video_id, label, frame_idx, x1, y1, side, kps, target_id, source_id, official, sid, …`.
- **Kiểm tra:** script in số frame/video theo split và liệt kê các video không detect được mặt. Nếu tỉ lệ video không detect được mặt vượt khoảng 1% thì xem lại bước P0.2.

### P0.4 Train baseline — `train.py` + `configs/celebdf.yaml`

`train.py` giờ đọc trực tiếp manifest khi config có khoá `manifest`. Chế độ thư mục cũ vẫn giữ nguyên, và `dataset_type` cũng đã đọc từ config. Tạo một config cho mỗi lần chạy, chỉ khác `seed`, `save_dir`, `protocol`, `synth`:

| Run | Thay đổi trong config |
|---|---|
| B0 | giữ nguyên `configs/celebdf.yaml` |
| B0-sid | `protocol: "sid"` |
| B0-low10 | `train_real_percent: 10`, `train_fake_percent: 10` |
| B1 (GAN cũ) | dùng `train_with_gan.py` với `dataset_type: celebdf` và layout thư mục cũ |
| D1/D2/D3 | thêm `synth:` (xem P2) |

```bash
python train.py --config configs/celebdf.yaml
```

Đánh giá trên test, có thể thêm bản làm nhiễu cho P-robust:

```bash
python scripts/predict_csv.py --checkpoint checkpoints/celebdf_B0_seed0/best.pth --manifest data/celebdf/meta.csv --frames-root data/celebdf/frames --protocol official --split test --out results/B0_seed0/test_official.csv
```

```bash
python scripts/predict_csv.py --checkpoint checkpoints/celebdf_B0_seed0/best.pth --manifest data/celebdf/meta.csv --frames-root data/celebdf/frames --protocol official --split test --perturb jpeg:30 --out results/B0_seed0/test_jpeg30.csv
```

```bash
python scripts/eval_predictions.py results/B0_seed0/test_official.csv results/B0_seed1/test_official.csv results/B0_seed2/test_official.csv --out results/B0_official.csv
```

- `--perturb` hỗ trợ `jpeg:Q`, `blur:K`, `resize:F`.
- `eval_predictions.py` báo frame-AUC, video-AUC (trung bình xác suất các frame), ACC, AP, EER, kèm **CI 95% bootstrap theo video**. Khi đưa vào nhiều file, script in thêm mean ± std giữa các seed.
- **Tài nguyên:** FA-ViT với ViT đóng băng, batch 32 trên GPU 24 GB, khoảng 240k frame/epoch, ước lượng 30–60 phút/epoch.
- **Go/no-go:** nếu B0 trên `official` đã đạt video-AUC ≥ 0.995 thì lấy `sid`, `low` và `robust` làm protocol chính (xem roadmap).

---

## P1 — Pseudo-mask và annotation MLLM

### P1.1 Pseudo-mask — `scripts/make_pseudo_masks.py`

Chạy thử trước:

```bash
python scripts/make_pseudo_masks.py --celebdf-root /path/to/Celeb-DF-v2 --meta data/celebdf/meta.csv --masks-root data/celebdf/masks --out-csv data/celebdf/masks.csv --limit-videos 20
```

- **Cách làm:**
  - Với mỗi frame fake, giải mã video real nguồn ở **cùng chỉ số frame** và crop bằng **khung crop của fake**.
  - Mask = |fake − real| (làm mờ nhẹ, ngưỡng Otsu, giới hạn trong vùng mặt), sau đó tính tỉ lệ phủ của mask trên 5 vùng.
  - Độ khớp của cặp được kiểm tra bằng `border_mad`: chênh lệch ở phần nền ngoài mặt phải gần 0.
- **Rất quan trọng:** quy ước `id{A}_id{B}_{k} → Celeb-real/id{A}_{k}` là **giả định chưa kiểm chứng**.
  - Nếu script cảnh báo dưới 50% cặp khớp, chạy lại với `--src-naming source` (tức `id{B}_{k}`).
  - Nếu cả hai cách đều thấp, có thể video fake và real lệch frame. Khi đó bỏ P1.1 và chỉ dùng nhãn tổng hợp từ P2.
- **Đầu ra:** `masks/<rel_path>.png` và `masks.csv` với các cột `aligned, border_mad, mask_area, ratio_eyes … ratio_boundary, dominant_region`.
- **Tài nguyên:** chỉ CPU, phải giải mã 2 video cho mỗi video fake. Có thể chia `--shard/--num-shards`; mỗi shard ghi ra một file `masks_shardXXX.csv` riêng, sau đó gộp bằng pandas/`cat`, nhớ bỏ dòng header lặp lại.
- **Kiểm tra:**
  - Tỉ lệ `aligned` nên trên 80%.
  - Xem khoảng 20 mask: chúng phải nằm trên vùng mặt, không phủ cả ảnh.
  - Chỉnh `--min-thr` nếu mask quá loang (tăng lên) hoặc quá rỗng (giảm xuống).

### P1.2 Annotation MLLM — `scripts/annotate_mllm.py`

1. **Xem trước prompt và ảnh, chưa cần model:**

   ```bash
   python scripts/annotate_mllm.py --meta data/celebdf/meta.csv --frames-root data/celebdf/frames --masks-csv data/celebdf/masks.csv --masks-root data/celebdf/masks --out data/celebdf/annotations/train_mllm.jsonl --n-fake 20 --n-real 5 --dry-run
   ```

2. **Chạy thật với backend transformers** (Qwen2.5-VL-7B: bf16 cần khoảng 17 GB VRAM, `--load-4bit` khoảng 7 GB):

   ```bash
   python scripts/annotate_mllm.py --meta data/celebdf/meta.csv --frames-root data/celebdf/frames --masks-csv data/celebdf/masks.csv --masks-root data/celebdf/masks --out data/celebdf/annotations/train_mllm.jsonl --n-fake 8000 --n-real 2000 --batch-size 8
   ```

3. **Hoặc dùng server vLLM (chỉ chạy được trên Linux, nhanh hơn).** Mở server ở một terminal:

   ```bash
   vllm serve Qwen/Qwen2.5-VL-7B-Instruct --max-model-len 8192
   ```

   rồi chạy annotate với backend openai:

   ```bash
   python scripts/annotate_mllm.py --meta data/celebdf/meta.csv --frames-root data/celebdf/frames --masks-csv data/celebdf/masks.csv --masks-root data/celebdf/masks --out data/celebdf/annotations/train_mllm.jsonl --backend openai --api-base http://localhost:8000/v1
   ```

- **Chỉ annotate frame thuộc split train** của protocol chọn (`--protocol`). Frame được trải đều theo video.
  - Frame fake được gửi kèm ảnh overlay mask làm gợi ý; frame real gửi ảnh đơn.
  - MLLM phải trả về JSON đúng schema (vùng, loại artifact, severity, confidence). Câu trả lời sai schema bị loại, nhưng câu trả lời gốc vẫn được lưu vào cột `raw`.
- **Resume:** chạy lại cùng lệnh sẽ bỏ qua các frame đã có trong file `.jsonl`.
- **Thời gian:** ước lượng 1–3 giờ cho 10k frame bằng transformers, nhanh hơn nhiều với vLLM.
- **Lưu ý thiên lệch:** gợi ý mask có thể làm MLLM chỉ "lặp lại" mask. Vì vậy cần bước QC 1.3, và cần tỉ lệ false positive trên frame real (không có gợi ý) để biết MLLM có bịa artifact hay không.

### P1.3 QC và độ tin cậy q — `scripts/qc_annotations.py`

1. **Xuất 200 frame cho 2 người chấm độc lập:**

   ```bash
   python scripts/qc_annotations.py sample-review --ann data/celebdf/annotations/train_mllm.jsonl --frames-root data/celebdf/frames --masks-root data/celebdf/masks --n 200 --out-dir data/celebdf/review
   ```

2. **Mỗi người điền `region_ok` và `artifact_ok` (1/0)** cho từng finding trong file của mình: `reviewer_A.csv` hoặc `reviewer_B.csv`. Ảnh đã được phóng to 2× trong `review/images/`.

3. **Tính calib (độ chính xác theo từng loại artifact) và kappa:**

   ```bash
   python scripts/qc_annotations.py calibrate --reviews data/celebdf/review/reviewer_A.csv data/celebdf/review/reviewer_B.csv --out data/celebdf/annotations/calib.json
   ```

4. **Tính q cho toàn bộ annotation:**

   ```bash
   python scripts/qc_annotations.py score --ann data/celebdf/annotations/train_mllm.jsonl --masks-csv data/celebdf/masks.csv --calib data/celebdf/annotations/calib.json --out data/celebdf/annotations/train_qc.jsonl
   ```

- **Công thức:** `q = confidence × calib[artifact] × (0.2 nếu vùng MLLM nêu không trùng mask hoặc nếu là frame real)`.
- **Đầu ra:** mỗi frame có `region_vec[5]` và `artifact_vec[7]`. P3/P4 dùng hai vector này.
- **Go/no-go:** `region_precision` trong `calib.json` phải ≥ 0.6 (script tự in cảnh báo nếu thấp hơn), và `kappa` nên ≥ 0.4. Nếu không đạt, P3 dùng vùng từ pseudo-mask và nhãn artifact từ P2.

---

## P2 — Làm dày dữ liệu có cấu trúc + LLM planner

### P2.1 Bộ sinh — `python -m synth.generate`

- **6 artifact giả**, áp trong 1 trong 5 vùng với độ mạnh (strength) từ 0 đến 1: `blend_boundary` (kiểu SBI), `color_mismatch`, `blur_resample`, `over_smooth`, `warp_geometry`, `texture_inconsistency` (dán vùng từ mặt khác, đã căn theo keypoint).
- **2 phép cho lớp real (nhãn 0):**
  - `authentic_degrade`: nén JPEG, làm mờ, thêm nhiễu trên toàn ảnh.
  - `authentic_seam`: dán vùng từ một frame khác **của cùng video**, để model không học lối tắt "cứ có đường dán là giả".
- **Mask và nhãn đều chính xác**, vì mỗi ảnh được sinh theo công thức biết trước.
- **QC:** ảnh mà artifact không nhìn thấy được (chênh lệch trung bình trong vùng < `--min-visible`) sẽ được tăng strength rồi thử lại, tối đa 3 lần, rồi bị loại. Script cảnh báo các ô bị loại trên 30%.
- **Chống leakage:** ảnh nguồn chỉ lấy từ split `train` (dữ liệu train) hoặc `val` (bộ probe). Script không cho chọn `test`.
- **Tài nguyên:** chỉ CPU, đa tiến trình (`--workers`). 30k ảnh mất cỡ vài phút đến vài chục phút.

### P2.2 Bộ probe chẩn đoán (sinh 1 lần, từ frame real của val)

```bash
python -m synth.generate --meta data/celebdf/meta.csv --frames-root data/celebdf/frames --protocol official --split val --grid --per-cell 40 --strengths 0.3 0.6 0.9 --out-dir data/celebdf/synth/probe_val
```

Bộ probe có 32 ô × 3 mức strength × 40 ảnh ≈ 3.8k ảnh. Mỗi vòng dùng bộ này để đo xem detector yếu ở ô (vùng × artifact) nào.

### P2.3 Vòng lặp planner (3–5 vòng)

Mỗi vòng `r` (bắt đầu `r = 1`; vòng 0 là D1 ngẫu nhiên với ngân sách nhỏ, dùng làm điểm khởi đầu):

1. **Train ngắn** (ví dụ `epochs: 5`) trên Celeb-DF cộng các vòng synth ≤ r−1. Ví dụ khai báo trong config:

   ```yaml
   synth:
     - manifest: data/celebdf/synth/D2/synth_meta.csv
       rounds: [0, 1]
   ```

2. **Dự đoán trên val và trên probe:**

   ```bash
   python scripts/predict_csv.py --checkpoint checkpoints/D2_r1/best.pth --manifest data/celebdf/meta.csv --frames-root data/celebdf/frames --split val --out results/D2_r1/val.csv
   ```

   ```bash
   python scripts/predict_csv.py --checkpoint checkpoints/D2_r1/best.pth --synth data/celebdf/synth/probe_val/synth_meta.csv --out results/D2_r1/probe.csv
   ```

3. **Báo cáo lỗi theo ô:**

   ```bash
   python -m synth.error_report --probe-preds results/D2_r1/probe.csv --val-preds results/D2_r1/val.csv --masks-csv data/celebdf/masks.csv --out results/D2_r1/report
   ```

   Báo cáo gồm AUC của từng ô so với real val, TPR theo strength, FPR trên các probe real, và FNR trên fake val theo vùng bị sửa nhiều nhất. Kết quả ghi ra `report.json` và `report.md`; file `.md` là phần được đưa vào prompt cho LLM.

4. **Lập kế hoạch.** Ba chế độ dùng cùng ngân sách để so sánh công bằng:
   - D2, LLM (cần server `vllm serve Qwen/Qwen2.5-7B-Instruct`, hoặc API khác qua `--api-base` + biến môi trường `OPENAI_API_KEY`):

     ```bash
     python -m synth.llm_planner --report results/D2_r1/report.json --mode llm --budget 10000 --round 2 --out plans/D2_r2.json --log runs/planner_log.jsonl
     ```

   - D3, luật cứng:

     ```bash
     python -m synth.llm_planner --report results/D3_r1/report.json --mode rule --budget 10000 --round 2 --out plans/D3_r2.json --log runs/planner_log.jsonl
     ```

   - D1, ngẫu nhiên đều:

     ```bash
     python -m synth.llm_planner --mode random --budget 10000 --round 2 --out plans/D1_r2.json --log runs/planner_log.jsonl
     ```

   Kế hoạch luôn được **validate** trước khi dùng: bỏ ô không hợp lệ, kẹp strength vào khoảng cho phép, giới hạn số ảnh mỗi ô, thu nhỏ cho vừa ngân sách. Mọi prompt và câu trả lời đều được ghi vào `runs/planner_log.jsonl` để đưa vào báo cáo.

5. **Sinh ảnh theo kế hoạch:**

   ```bash
   python -m synth.generate --meta data/celebdf/meta.csv --frames-root data/celebdf/frames --protocol official --split train --plan plans/D2_r2.json --round 2 --out-dir data/celebdf/synth/D2
   ```

   Mỗi vòng phải có `--round` mới; script từ chối ghi đè vòng đã có.

**Cuối cùng:** train đầy đủ (30 epoch, 3 seed) cho D1, D2, D3 với **cùng tổng số ảnh synth**, rồi đánh giá theo P0.4 trên `official`, `sid`, `low` và `robust`.

**Go/no-go P2:**
- D2 phải hơn D1 với CI không chồng nhau, trên ít nhất một protocol trong `sid`, `low`, `robust`.
- Nếu D2 ≈ D3 thì báo cáo trung thực rằng phần tăng đến từ dữ liệu có cấu trúc, không phải từ LLM.

---

## Danh sách script

| Bước | Script | GPU? | Đầu ra chính |
|---|---|---|---|
| P0.1 | `scripts/prepare_celebdf.py split` | không | `splits.json` |
| P0.2 | `scripts/prepare_celebdf.py extract` | có (SCRFD) | `frames/`, shard jsonl |
| P0.3 | `scripts/prepare_celebdf.py merge` | không | `meta.csv` |
| P0.4 | `train.py --config configs/celebdf.yaml` | có | checkpoint |
| P0.4 | `scripts/predict_csv.py` | có | CSV dự đoán theo frame |
| P0.4 | `scripts/eval_predictions.py` | không | bảng chỉ số + CI |
| P1.1 | `scripts/make_pseudo_masks.py` | không | `masks/`, `masks.csv` |
| P1.2 | `scripts/annotate_mllm.py` | có (MLLM) | `train_mllm.jsonl` |
| P1.3 | `scripts/qc_annotations.py` | không | `calib.json`, `train_qc.jsonl` |
| P2.1–2 | `synth/generate.py` | không | `synth/*/synth_meta.csv` |
| P2.3 | `synth/error_report.py` | không | `report.json/.md` |
| P2.3 | `synth/llm_planner.py` | có nếu tự chạy LLM | `plans/*.json`, log |

Chưa có trong đợt này (làm ở vòng sau): `models/favit_align.py` và `train_celebdf_llm.py` cho P3 (alignment patch × artifact), cùng phần distillation P4.
