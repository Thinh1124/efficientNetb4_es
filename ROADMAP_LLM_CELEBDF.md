# Lộ trình: Nâng cấp FA-ViT + GAN bằng LLM trên Celeb-DF

> Phạm vi: **train, làm dày dữ liệu và test đều trên Celeb-DF (v2)**.
> Hệ hiện tại: `RedesignedFAViT` (ViT-B/16 đóng băng + GAM/LAM adapter + CNN local, tùy chọn nhánh FFT, FALoss) + dữ liệu GAN bổ sung (`train_with_gan.py`).
> Nguồn ý tưởng: sheet *Deepfake Detection using LLM* — tab "Hướng nghiên cứu" (hướng 2, 3, 4, 9) và tab "Đề xuất" (đề xuất 1, 2, 3/CompFFD).

---

## 0. Nguyên tắc chung (đọc trước khi làm)

1. **LLM không có mặt lúc suy luận chính.** Detector cuối cùng vẫn là FA-ViT (ảnh → logit). LLM/MLLM chỉ đóng vai *planner* (lên công thức sinh dữ liệu), *annotator* (gán nhãn vùng/artifact) và *teacher* (distill). Nhánh giải thích là tùy chọn.
2. **Chống leakage tuyệt đối.** Mọi dữ liệu sinh thêm, pseudo-mask, annotation chỉ lấy từ **video thuộc tập train**. Không bao giờ sinh/annotate từ video test.
3. **Celeb-DF in-domain đã gần bão hòa** (nhiều detector đạt AUC ≥ 0.99). Vì vậy phần đóng góp của LLM phải được chứng minh ở các protocol khó hơn (mục 7): split tách identity, ít dữ liệu, nén/nhiễu, và cross-dataset. Nếu chỉ báo AUC in-domain thì khó thấy khác biệt.
4. **Mỗi thành phần LLM phải có ablation riêng**, cùng split, cùng seed, cùng số epoch, cùng lượng ảnh train.

---

## 1. Tổng quan các giai đoạn

| Giai đoạn | Nội dung | Thời lượng gợi ý | Đầu ra |
|---|---|---|---|
| P0 | Chuẩn bị Celeb-DF, split, baseline | Tuần 1–2 | Split cố định, bảng baseline B0–B2 |
| P1 | Pseudo-mask + annotation MLLM | Tuần 2–3 | `masks/`, `annotations/*.jsonl` đã QC |
| P2 | Làm dày dữ liệu có cấu trúc do LLM lập kế hoạch | Tuần 3–5 | `synth/` + bộ ảnh có nhãn (vùng, artifact) |
| P3 | Căn chỉnh patch × loại artifact trong ViT | Tuần 5–7 | Model `favit_align` |
| P4 | Distill MLLM teacher → FA-ViT | Tuần 7–9 | Model `favit_distill` |
| P5 | (Tùy chọn) Nhánh giải thích: ViT làm expert cho MLLM | Tuần 9–10 | Demo verdict + rationale |
| P6 | Đánh giá tổng hợp, ablation, viết báo cáo | Tuần 10–12 | Bảng kết quả + CI, báo cáo |

Thứ tự ưu tiên: **P0 → P1 → P2 → P3** là lõi (không tốn chi phí inference). P4 làm khi P3 đã có số. P5 chỉ làm nếu đề tài cần giải thích bằng ngôn ngữ.

---

## P0. Chuẩn bị dữ liệu Celeb-DF và baseline (Tuần 1–2)

### Bước 0.1 — Chia split theo video
- Celeb-DF v2: `Celeb-real/` (590), `YouTube-real/` (300), `Celeb-synthesis/` (5 639). Danh sách test chính thức: `List_of_testing_videos.txt` (518 video).
- **Test**: đúng 518 video chính thức (giữ nguyên để so với paper).
- **Train/Val**: phần còn lại, chia theo **video** (val ≈ 10%), không chia theo frame.
- **Split phụ khó hơn — `S-id` (tách identity)**: chọn khoảng 10 identity (`idXX`) chỉ xuất hiện ở test. Video fake có dạng `id{target}_id{source}_{xxxx}`: loại khỏi train mọi video có target *hoặc* source thuộc nhóm identity test. Split này dùng để chứng minh tổng quát hóa vì split chính thức có trùng identity giữa train và test.
- Lưu split ra `splits/celebdf_official.json` và `splits/celebdf_sid.json` (danh sách video cho train/val/test) để mọi thí nghiệm dùng chung.

### Bước 0.2 — Trích frame và crop mặt
- Dùng lại pipeline SCRFD đang dùng (`celebdf_faces_scrfd`). Gợi ý: 32 frame/video fake, 64–96 frame/video real để bù mất cân bằng (≈ 1 real : 6.5 fake theo số video).
- Lưu thêm `meta.csv`: `path, video_id, frame_idx, label, target_id, source_id, split`. File này cần cho pseudo-mask (P1) và đánh giá video-level.
- Bố cục thư mục theo layout `real/` + `fake/` mà repo đã hỗ trợ (`dataset_type="celebdf"`):
  ```text
  celebdf_root/
  ├── train/{real,fake}/<video_id>/*.png
  ├── val/{real,fake}/<video_id>/*.png
  └── test/{real,fake}/<video_id>/*.png
  ```

### Bước 0.3 — Code
- Thêm `configs/celebdf.yaml` (copy từ `configs/config.yaml`, đổi `data_root`, `save_dir`).
- Trong `train.py` / `train_with_gan.py`: đổi `dataset_type="ffpp"` thành giá trị đọc từ config (`config.get("dataset_type", "ffpp")`) để chạy được Celeb-DF mà không phải sửa code mỗi lần.
- Kiểm tra `eval_video_level.py` chạy được trên test Celeb-DF (đã hỗ trợ thư mục `0/1` và `real/fake`).

### Bước 0.4 — Baseline (3 seed mỗi cái)
| ID | Cấu hình |
|---|---|
| B0 | FA-ViT, chỉ Celeb-DF train, augmentation hiện tại |
| B1 | B0 + dữ liệu GAN hiện có (`train_with_gan.py`) |
| B2 | B0 + SBI/self-blending chuẩn (baseline làm dày dữ liệu không dùng LLM) |

**Go/no-go P0:** có bảng frame-AUC, video-AUC, ACC, AP, EER cho B0–B2 trên cả split chính thức và `S-id`. Nếu B0 trên split chính thức đã ≥ 0.995 video-AUC thì chuyển trọng tâm đánh giá sang `S-id`, low-data và robustness.

---

## P1. Pseudo-mask và annotation bằng MLLM (Tuần 2–3)

### Bước 1.1 — Pseudo-mask từ cặp real–fake (không cần LLM)
- Video fake `id{t}_id{s}_{k}` được tạo từ video real `id{t}_{k}` (**kiểm tra quy ước tên này trên vài video trước khi chạy hàng loạt**: so frame đầu của fake với real tương ứng). Căn frame theo `frame_idx`, crop cùng bbox, tính `|fake − real|` → làm mượt → ngưỡng → mask nhị phân 224×224.
- Script: `scripts/make_pseudo_masks.py --meta meta.csv --out masks/`. Bỏ cặp lệch (SSIM toàn ảnh thấp → frame không khớp).
- Chiếu mask lên 5 vùng bằng face parsing (BiSeNet hoặc landmarks): **mắt, mũi, miệng, da/má, biên mặt** → vector `region_ratio[5]` = tỉ lệ diện tích vùng bị sửa.

### Bước 1.2 — Annotation artifact bằng MLLM
- Model gợi ý: **Qwen2.5-VL-7B-Instruct** chạy local (vLLM/transformers, 1 GPU 24 GB với 4-bit) hoặc API nếu có ngân sách.
- Chỉ annotate **frame train fake** (lấy mẫu ~5–10k frame là đủ) và một ít frame real (để có câu trả lời "authentic").
- Đầu vào: ảnh crop + overlay mask từ 1.1 (gợi ý vùng). Yêu cầu **JSON có schema cố định**, không cho viết tự do:
  ```json
  {
    "frame": "train/fake/id0_id16_0003/0012.png",
    "findings": [
      {"region": "mouth", "artifact": "blur_resample", "severity": 2, "confidence": 0.7},
      {"region": "boundary", "artifact": "color_mismatch", "severity": 1, "confidence": 0.5}
    ]
  }
  ```
  - `region` ∈ {eyes, nose, mouth, skin, boundary}
  - `artifact` ∈ {blend_boundary, color_mismatch, blur_resample, over_smooth, warp_geometry, texture_inconsistency, authentic}
- Lưu `annotations/celebdf_train_mllm.jsonl`.

### Bước 1.3 — Kiểm tra chất lượng (QC) và tính độ tin cậy q
- **Kiểm chéo với pseudo-mask:** finding ở vùng có `region_ratio ≈ 0` → hạ confidence (hoặc bỏ). Đây là cách chống MLLM bịa artifact.
- **Kiểm tra tay** 200 frame (2 người chấm độc lập), đo độ chính xác vùng/artifact của MLLM và Cohen's κ giữa người.
- Độ tin cậy cuối cùng: `q = conf_MLLM × agree_mask × calib`. Trong đó `calib` lấy từ tập 200 frame chấm tay (tỉ lệ đúng theo từng loại artifact).

**Go/no-go P1:** độ chính xác vùng của MLLM trên tập chấm tay ≥ 60%. Nếu thấp hơn, P3/P4 chỉ dùng nhãn vùng từ pseudo-mask, nhãn artifact lấy từ P2 (dữ liệu tổng hợp có nhãn chắc chắn).

---

## P2. Làm dày dữ liệu có cấu trúc, do LLM lập kế hoạch (Tuần 3–5)

Ý tưởng: thay vì GAN sinh ảnh "giả chung chung", mỗi ảnh tổng hợp được sinh theo **công thức (vùng, artifact, cường độ)** biết trước, nên có sẵn **mask + nhãn chính xác**. LLM đóng vai *planner*: quyết định công thức nào cần sinh thêm, dựa trên lỗi của detector.

### Bước 2.1 — Bộ sinh có tham số (`synth/`)
- `synth/recipes.py`: định nghĩa `Recipe(region, artifact, strength, source_video, seed)`.
- `synth/generate.py`: từ **frame real của tập train**, áp artifact lên vùng chọn:
  | artifact | cách sinh |
  |---|---|
  | blend_boundary | self-blending kiểu SBI, mask theo vùng, đường biên Gaussian |
  | color_mismatch | lệch histogram / color transfer trong vùng |
  | blur_resample | downscale–upscale hoặc blur trong vùng |
  | over_smooth | bilateral / guided filter mạnh |
  | warp_geometry | warp affine/TPS nhẹ quanh landmark vùng |
  | texture_inconsistency | dán patch từ frame khác (cùng video hoặc identity khác) |
  | (GAN) | dùng GAN hiện có để tái tạo riêng vùng chọn, rồi blend |
- Mỗi ảnh xuất ra: `img.png`, `mask.png`, `meta.json` (recipe).
- **Làm dày cả lớp real:** augmentation an toàn (nén JPEG/H.264, đổi màu nhẹ, blur toàn ảnh) và "seam real" (dán vùng từ frame real khác cùng video, như arm `raw_aug` trong tab CompFFD). Mục đích là để model không học shortcut "có đường dán = fake".

### Bước 2.2 — QC chống shortcut
- Loại ảnh mà artifact không nhìn thấy được (khác biệt trong mask < ngưỡng).
- Cân bằng diện tích mask giữa các ô (vùng × artifact) để model không học kích thước vùng.
- Áp cùng pipeline nén cho real và synth.

### Bước 2.3 — LLM planner (vòng lặp kín)
- Model: LLM text (Qwen2.5-7B-Instruct local hoặc API), không cần thị giác.
- Mỗi vòng:
  1. Train FA-ViT (vài epoch) trên Celeb-DF train + synth hiện có.
  2. Chạy trên **val** → script `synth/error_report.py` tổng hợp lỗi theo ô (vùng × artifact) dựa vào pseudo-mask/annotation P1: AUC từng ô, false negative theo vùng, false positive trên real bị nén.
  3. Gửi báo cáo (dạng bảng số, không gửi ảnh) cho LLM, yêu cầu trả **JSON recipe**:
     ```json
     {"plan": [
       {"region": "mouth", "artifact": "blur_resample", "strength": [0.3, 0.6], "count": 3000,
        "reason": "FN cao ở miệng khi blur nhẹ"},
       {"region": "real_compressed", "artifact": "authentic", "strength": [0.5, 0.8], "count": 2000,
        "reason": "FP trên real nén mạnh"}
     ]}
     ```
  4. `synth/llm_planner.py` **validate** (schema, giới hạn tổng số ảnh mỗi vòng, giới hạn tỉ lệ synth/real) → sinh ảnh → vòng tiếp theo.
- Chạy 3–5 vòng. Log toàn bộ prompt, kế hoạch và AUC val mỗi vòng vào `runs/planner_log.jsonl`.

### Bước 2.4 — Dataset loader
- Thêm `StructuredSynthDataset` vào `datasets/deepfake_dataset.py`, trả về `(img, label, mask, region_vec[5], artifact_vec[7], q)`. Với frame Celeb-DF gốc: mask/vec lấy từ P1, `q` từ 1.3. Với synth: nhãn chắc chắn, `q = 1`.
- Trộn bằng `ConcatDataset` như `train_with_gan.py`. Tỉ lệ synth/real là tham số config (`synth_ratio`).

### Ablation bắt buộc cho P2
| ID | Cấu hình |
|---|---|
| D1 | B0 + synth **ngẫu nhiên đều** trên mọi ô (cùng tổng số ảnh với D2) |
| D2 | B0 + synth do **LLM planner** chọn |
| D3 | D2 nhưng planner là **luật cứng** (sinh thêm vào ô có AUC thấp nhất) → kiểm tra LLM có hơn heuristic đơn giản không |

**Go/no-go P2:** D2 > D1 trên `S-id` hoặc robustness (khoảng CI không chồng nhau). Nếu D2 ≈ D3 thì báo cáo trung thực rằng phần lợi đến từ *dữ liệu có cấu trúc*, không phải từ LLM. Vẫn giữ lại P2 vì nó cung cấp nhãn cho P3.

---

## P3. Căn chỉnh patch × loại artifact trong FA-ViT (Tuần 5–7)

### Bước 3.1 — Đổi backbone sang CLIP ViT-B/16
- `RedesignedFAViT` đã nhận `backbone_name` → dùng `vit_base_patch16_clip_224.openai` (timm). Patch token khi đó cùng không gian với text encoder CLIP.
- Đổi mean/std chuẩn hóa sang của CLIP trong `get_train_transform` / `get_eval_transform` (thêm tham số `norm="clip"`).
- Chạy lại B0 với backbone CLIP (gọi là B0-CLIP) để tách phần tăng do đổi backbone khỏi phần tăng do alignment.

### Bước 3.2 — Text anchor
- Với mỗi trong 7 lớp artifact, viết 3–5 cụm mô tả (có thể nhờ LLM paraphrase rồi lọc tay). Mã hóa bằng CLIP text encoder **một lần**, lưu `anchors/artifact_anchors.pt` (shape `[7, n_phrase, 512]` → lấy trung bình thành `[7, 512]`).

### Bước 3.3 — Thay đổi model (`models/favit_align.py`)
- `forward` trả thêm `patch_tokens` (196 × D) sau `self.vit.norm`.
- `ArtifactAlignHead`:
  - chiếu patch → 512 (`proj`), chuẩn hóa;
  - `S = patch @ anchors.T / τ` → `[B, 196, 7]` = heatmap theo từng loại;
  - `z_j = logsumexp`-pool theo patch → `[B, 7]`.
- Classifier mới: `MLP([CLS ; z_1..z_7]) → logit`. Classifier dùng trực tiếp bằng chứng theo loại.

### Bước 3.4 — Loss (`train_celebdf_llm.py`)
```
L = BCE(logit, y)
  + λ1 · L_type   # BCE/InfoNCE giữa z_j và artifact_vec (nhân q)
  + λ2 · L_ground # Dice giữa heatmap S_·j và mask (chỉ với ảnh có mask, nhân q)
  + λ3 · L_rank   # hard negative: cùng vùng nhưng sai loại; real có seam
  + λ4 · FALoss   # giữ từ hệ hiện tại nếu có ích
```
Gợi ý khởi đầu: λ1 = 0.5, λ2 = 0.5, λ3 = 0.1, τ = 0.07. Tune trên val.

### Ablation P3
| ID | Cấu hình |
|---|---|
| A0 | B0-CLIP + dữ liệu D2 (không alignment) |
| A1 | A0 + chỉ 2 anchor real/fake (kiểu VLAForge/KFD) |
| A2 | A0 + 7 anchor theo loại, chỉ là loss phụ (z không vào classifier) |
| A3 | A2 + z vào classifier |
| A4 | A3 + L_ground (mask) |
| Full | A4 + trọng số q + L_rank |
| Kiểm tra shortcut | xáo nhãn loại; thay anchor bằng tên vùng; zero hóa z lúc test (để biết classifier có dùng z thật không) |

**Go/no-go P3:** Full > A1 trên `S-id`, và zero hóa z làm giảm AUC rõ rệt (chứng minh z được dùng).

---

## P4. Distill MLLM teacher → FA-ViT (Tuần 7–9)

### Bước 4.1 — Chuẩn bị teacher
- Fine-tune MLLM bằng LoRA (Qwen2.5-VL-7B) trên frame train với nhãn thật/giả và JSON findings từ P1/P2. Có thể dùng luôn model Full của P3 làm teacher thứ hai (đề xuất 1 làm teacher, đề xuất 2 làm student theo ghi chú trong sheet).
- Chạy teacher **offline một lần** trên toàn bộ ảnh train, cache: `logit_t`, `heatmap_t[7,14,14]` (MLLM: lấy từ attention, hoặc dùng teacher P3), `score_t[7]`, `q`.

### Bước 4.2 — Loss distill
```
L = BCE(s, y)
  + α · T² · KL(σ(z_t/T) ‖ σ(z_s/T))
  + β · Σ_j q_j · Dice(A_s,j , A_t,j)
  + γ · Σ_j q_j · ‖s_s,j − s_t,j‖²
```
Nhãn GT luôn được giữ. Khi teacher sai so với GT thì hạ `q` (hoặc bỏ mẫu đó khỏi phần distill).

### Bước 4.3 — Stress test teacher
- Cố ý làm hỏng 10/20/40% bằng chứng của teacher (đổi loại, dời vùng) → vẽ đường AUC theo % nhiễu. Kỳ vọng là có `q` thì kết quả giảm từ từ, không có `q` thì sụp nhanh.

### Ablation P4
chỉ logit → + heatmap → + score theo loại → + q → xáo q (đối chứng).

---

## P5. (Tùy chọn) Nhánh giải thích: FA-ViT làm expert cho MLLM (Tuần 9–10)

- Pipeline: ảnh → FA-ViT (score, heatmap 7 loại, top vùng) → điền vào prompt có cấu trúc → MLLM → verdict + giải thích ngắn.
- Chỉ dùng cho demo/báo cáo. Verdict chính vẫn lấy từ FA-ViT.
- Đánh giá giải thích:
  - **grounding:** vùng nêu trong giải thích có trùng heatmap/mask không;
  - **false-yes:** hỏi phủ định về vùng không bị sửa ("miệng có bị chỉnh không?");
  - **faithfulness:** đưa evidence sai vào prompt, xem MLLM có nói theo không.

---

## P6. Đánh giá tổng hợp (Tuần 10–12)

### Protocol
| Protocol | Mô tả | Vai trò |
|---|---|---|
| P-official | Train/test theo danh sách chính thức Celeb-DF v2 | So sánh với paper |
| P-sid | Split tách identity (P0.1) | **Tổng quát hóa — protocol chính để chứng minh LLM có ích** |
| P-low | Chỉ dùng 10% / 25% video train (`train_real_percent` + lọc fake tương ứng) | Làm dày dữ liệu có ý nghĩa nhất khi ít dữ liệu |
| P-robust | Test với JPEG q=50/30, H.264 CRF 28/35, blur, resize 0.5× | Độ bền |
| P-cross (phụ) | Model train Celeb-DF → test FF++ c23, DFDC | Trả lời câu hỏi chắc chắn sẽ bị hỏi về cross-dataset |

### Chỉ số
- frame-AUC, **video-AUC** (`eval_video_level.py`), ACC@0.5, AP, EER.
- **CI 95% bootstrap theo video**, 3 seed (0, 1, 2), báo mean ± std.
- Với P3/P4: IoU/pointing-game của heatmap trên tập có pseudo-mask; độ chính xác phân loại artifact.
- Chi phí: số tham số trainable, latency, VRAM lúc inference; GPU-hours cho annotation và sinh dữ liệu.

### Bảng kết quả cuối (mẫu)
| Model | P-official vAUC | P-sid vAUC | P-low 10% | P-robust (avg) | P-cross FF++ | Latency |
|---|---|---|---|---|---|---|
| B0 FA-ViT | | | | | | |
| B1 + GAN cũ | | | | | | |
| B2 + SBI | | | | | | |
| D2 + synth LLM-planned | | | | | | |
| Full P3 | | | | | | |
| Full P3 + P4 | | | | | | |

---

## 8. Danh sách file cần thêm/sửa

| File | Việc |
|---|---|
| `scripts/prepare_celebdf.py` | Split (official + S-id), trích frame, `meta.csv` |
| `scripts/make_pseudo_masks.py` | Mask từ cặp real–fake + chiếu 5 vùng |
| `scripts/annotate_mllm.py` | Gọi MLLM, xuất JSONL theo schema |
| `scripts/qc_annotations.py` | Đối chiếu mask, tính q, xuất tập chấm tay |
| `synth/recipes.py`, `synth/generate.py` | Bộ sinh có tham số |
| `synth/error_report.py`, `synth/llm_planner.py` | Vòng lặp planner |
| `datasets/deepfake_dataset.py` | `StructuredSynthDataset`; chuẩn hóa CLIP |
| `models/favit_align.py` | FA-ViT CLIP + `ArtifactAlignHead` |
| `train_celebdf_llm.py` | Train với L_type / L_ground / L_rank / distill |
| `configs/celebdf.yaml`, `configs/celebdf_llm.yaml` | Config mới |
| `train.py`, `train_with_gan.py` | Đọc `dataset_type` từ config |

---

## 9. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Celeb-DF in-domain bão hòa | B0 ≥ 0.995 vAUC | Lấy P-sid / P-low / P-robust làm kết quả chính |
| MLLM bịa artifact | Độ chính xác vùng < 60% ở QC | Dùng pseudo-mask + nhãn synth, bỏ annotation MLLM |
| Model học shortcut đường dán | FP cao trên "seam real" | Tăng real-seam, cân bằng diện tích mask |
| Pseudo-mask lệch frame | Mask phủ cả mặt hoặc rỗng | Lọc theo SSIM, chỉ giữ cặp khớp |
| LLM planner không hơn heuristic | D2 ≈ D3 | Báo cáo trung thực; đóng góp là dữ liệu có cấu trúc |
| Novelty bị chồng (KFD, VLAForge, DF-CBM, ReAlign, AuthGuard) | — | Nhấn mạnh: q theo vùng/loại, planner vòng kín, protocol S-id; so trực tiếp nếu tái hiện được |
| Thiếu GPU cho MLLM 7B | OOM | 4-bit + LoRA; annotate tập con 5k frame; hoặc dùng teacher P3 thay MLLM ở P4 |

---

## 10. Checklist theo tuần

- [ ] **T1:** Split official + S-id, trích frame, `meta.csv`
- [ ] **T2:** Baseline B0/B1/B2 × 3 seed; pseudo-mask
- [ ] **T3:** Annotation MLLM + QC 200 frame; bộ sinh `synth/` 7 artifact
- [ ] **T4:** D1 (synth ngẫu nhiên); planner vòng 1–2
- [ ] **T5:** Planner vòng 3–5; D2, D3; quyết định go/no-go P2
- [ ] **T6:** B0-CLIP; `favit_align` A0–A2
- [ ] **T7:** A3, A4, Full, kiểm tra shortcut
- [ ] **T8:** Teacher LoRA + cache đầu ra
- [ ] **T9:** Distill + stress test teacher
- [ ] **T10:** (Tùy chọn) nhánh giải thích
- [ ] **T11:** Toàn bộ protocol P6, CI bootstrap
- [ ] **T12:** Viết báo cáo, bảng, hình heatmap
