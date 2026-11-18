# Nâng cấp hệ thống phát hiện deepfake FA-ViT + GAN bằng LLM

**Dữ liệu:** Celeb-DF v2 (train, làm dày dữ liệu và test) · **Thời lượng dự kiến:** 12 tuần · **Trạng thái:** đã có code P0–P2, đã chạy thử trên dữ liệu giả lập

---

## 1. Vấn đề và mục tiêu

- **Hệ hiện tại:** FA-ViT (ViT-B/16 đóng băng, adapter GAM/LAM kết hợp đặc trưng CNN cục bộ), có thêm dữ liệu do GAN sinh. Mô hình chỉ học nhãn nhị phân thật/giả, **không biết vùng nào bị sửa và bị sửa kiểu gì**.
- **Điểm yếu:** khả năng tổng quát hóa sang identity, mức nén và phương pháp giả mạo chưa thấy khi train. Dữ liệu GAN được sinh "chung chung", không nhắm vào điểm yếu cụ thể của mô hình.
- **Mục tiêu:** dùng LLM/MLLM để tạo **giám sát có cấu trúc (vùng × loại artifact)** và **sinh dữ liệu có chủ đích**, mà **không cần LLM lúc suy luận**.

## 2. Ý tưởng chính: LLM có 3 vai trò, cả 3 đều chạy ngoài pha suy luận

| Vai trò | LLM làm gì | Thành phần dùng kết quả |
|---|---|---|
| **Planner** | Đọc báo cáo lỗi của detector theo từng ô (vùng × artifact), quyết định sinh thêm dữ liệu gì cho vòng sau | Bộ sinh dữ liệu (thay cho GAN sinh ngẫu nhiên) |
| **Annotator** | MLLM (Qwen2.5-VL) mô tả artifact theo JSON cố định: vùng, loại, mức độ, độ tin cậy | Loss căn chỉnh vùng/artifact (P3) |
| **Teacher** | MLLM đã fine-tune sinh logit và heatmap để distill vào FA-ViT | Distillation có trọng số tin cậy (P4) |

Lúc suy luận chỉ chạy: **ảnh → crop mặt → FA-ViT → xác suất giả**. Chi phí suy luận không đổi so với hệ hiện tại.

## 3. So sánh hệ hiện tại và hệ đề xuất

| Tiêu chí | FA-ViT + GAN (hiện tại) | FA-ViT + LLM (đề xuất) |
|---|---|---|
| Nhãn huấn luyện | Chỉ thật/giả | Thật/giả + mask + (vùng, loại artifact, độ tin cậy) |
| Dữ liệu bổ sung | GAN sinh ngẫu nhiên, không có nhãn vùng | Sinh theo công thức (vùng × artifact × cường độ), mask chính xác |
| Cách chọn dữ liệu sinh thêm | Cố định từ đầu | Vòng lặp kín: lỗi của detector → LLM lập kế hoạch → sinh thêm |
| Chống học lối tắt | Không có | Thêm "real có đường dán" và "real bị nén" làm mẫu âm |
| Chi phí suy luận | 1 lần chạy ViT | Như cũ (không gọi LLM) |
| Khả năng giải thích | Không có | Heatmap theo loại artifact (P3); tùy chọn thêm lời giải thích bằng MLLM |
| Đánh giá | AUC in-domain | 5 protocol, CI bootstrap theo video, 3 seed |

## 4. Lựa chọn hướng dùng LLM (tổng hợp từ khảo sát khoảng 90 bài)

| Hướng trong khảo sát | Ưu điểm | Hạn chế với đề tài này | Chọn? |
|---|---|---|---|
| MLLM làm detector end-to-end (Veritas, FakeXplain, …) | Ra được cả quyết định lẫn lời giải thích | Mô hình 7B+ chạy lúc suy luận, tốn GPU; lời giải thích chưa chắc đúng với lý do thật | Không |
| Detector dẫn đường cho MLLM (X²-DFD, KFD) | Dùng lại được detector sẵn có | Vẫn cần MLLM lúc suy luận | Tùy chọn (chỉ cho phần demo giải thích) |
| Fusion vision–language (MLLM-Enhanced, MM-Det++) | Thông tin ngôn ngữ tác động trực tiếp vào dự đoán | Dễ học lối tắt ngữ nghĩa; chi phí cao | Một phần (text anchor ở P3) |
| **LLM làm annotator/teacher** (FFTG, ReAlign, AuthGuard) | Không cần LLM lúc suy luận | Nhãn từ LLM có thể sai (hallucination) | **Có** (P1, P4) |
| **LLM sinh/điều khiển dữ liệu** (Agent4FaceForgery) | Sinh dữ liệu có chủ đích | Phụ thuộc chất lượng bộ sinh | **Có** (P2) |
| Multi-agent / tool-use | Có kiểm tra chéo giữa nhiều agent | Độ trễ cao, tốn nhiều lượt gọi | Không |

## 5. So sánh với các công trình gần nhất (điểm mới)

| Công trình | Đã làm | Đề tài bổ sung |
|---|---|---|
| SBI, SLADD, CDFA | Tổng hợp giả mạo có tham số, nhưng phân bố cố định | **Planner vòng kín** chọn ô (vùng × artifact) theo lỗi đo được của detector |
| Agent4FaceForgery (CVPR'26) | Dùng LLM agent để sinh dữ liệu giả mạo | Kế hoạch dựa trên **báo cáo lỗi định lượng theo ô**; so sánh với luật cứng và ngẫu nhiên có cùng ngân sách (*cần đọc kỹ để xác nhận khác biệt*) |
| KFD, VLAForge | Bản đồ tương đồng patch–prompt chỉ phân biệt real/fake | Tách thành **7 loại artifact** và đưa điểm từng loại vào classifier |
| DF-CBM, AIFIND | Có concept/anchor theo loại artifact | Gán **trọng số tin cậy q theo từng vùng và từng loại** cho nhãn nhiễu từ MLLM |
| ReAlign, AuthGuard | Distill lập luận của LLM ở mức toàn ảnh | Distill **cục bộ** (heatmap + điểm theo loại) có trọng số q; stress test khi teacher sai 10–40% |

> Theo khảo sát (22/10/2026), từng thành phần riêng lẻ có mức mới **thấp đến trung bình**. Đóng góp nằm ở **tổ hợp** các thành phần, ở **trọng số tin cậy theo vùng/loại**, và ở **protocol đánh giá tách identity**. Phải chứng minh bằng ablation.

## 6. Các giai đoạn

| GĐ | Nội dung | Đầu ra | Tuần | Trạng thái |
|---|---|---|---|---|
| P0 | Chia split (official, tách identity), crop mặt, baseline | `meta.csv`, bảng B0–B2 | 1–2 | Code xong |
| P1 | Pseudo-mask từ cặp real–fake; MLLM gán nhãn; QC 200 frame với 2 người chấm | Mask, nhãn kèm độ tin cậy q | 2–3 | Code xong |
| P2 | Bộ sinh 6 artifact × 5 vùng; vòng LLM planner 3–5 vòng | Dữ liệu D1/D2/D3 | 3–5 | Code xong |
| P3 | FA-ViT + căn chỉnh patch × text anchor theo loại artifact | Model `favit_align` | 5–7 | Chưa làm |
| P4 | Distill MLLM teacher → FA-ViT | Model `favit_distill` | 7–9 | Chưa làm |
| P5 | Đánh giá tổng hợp, viết báo cáo | Bảng kết quả, báo cáo | 10–12 | — |

## 7. Thiết kế thí nghiệm

**Baseline và ablation** (mọi cấu hình dùng cùng split, cùng seed và cùng tổng số ảnh):

| Ký hiệu | Cấu hình | Câu hỏi cần trả lời |
|---|---|---|
| B0 / B1 / B2 | FA-ViT / + GAN cũ / + SBI | Mốc so sánh |
| D1 | + dữ liệu sinh **ngẫu nhiên đều** | Lợi ích của dữ liệu có cấu trúc |
| D2 | + dữ liệu sinh do **LLM planner** chọn | LLM có giúp chọn dữ liệu tốt hơn không? |
| D3 | + dữ liệu sinh theo **luật cứng** | LLM có hơn một heuristic đơn giản không? |
| A1 → Full | 2 anchor real/fake → 7 anchor theo loại → + mask → + q | Từng thành phần của P3 đóng góp bao nhiêu |

**Protocol đánh giá:**

| Protocol | Mô tả | Vai trò |
|---|---|---|
| Official | 518 video test chính thức | So sánh với các bài báo |
| **S-id** | Identity ở test không xuất hiện khi train | **Protocol chính** (in-domain Celeb-DF đã gần bão hòa) |
| Low-data | Chỉ dùng 10% / 25% số video train | Đo giá trị của việc làm dày dữ liệu |
| Robust | JPEG 30/50, blur, resize 0.5× | Độ bền |
| Cross | Train Celeb-DF → test FF++, DFDC | Tổng quát hóa sang dataset khác |

**Chỉ số:** frame-AUC, video-AUC, AP, EER, ACC; CI 95% bootstrap theo video; trung bình ± độ lệch chuẩn trên 3 seed.

## 8. Rủi ro và phương án

| Rủi ro | Phương án |
|---|---|
| Celeb-DF in-domain đã gần AUC 0.99 | Lấy S-id, low-data, robust làm kết quả chính |
| MLLM bịa artifact | Đối chiếu với pseudo-mask + chấm tay; nếu region precision < 0.6 thì dùng nhãn tổng hợp |
| LLM planner không hơn luật cứng (D2 ≈ D3) | Báo cáo trung thực; khi đó đóng góp là dữ liệu có cấu trúc |
| Thiếu GPU cho MLLM 7B | Lượng tử hóa 4-bit, chỉ gán nhãn tập con khoảng 10k frame, chạy được trên Colab T4 |

## 9. Đóng góp dự kiến

1. **Pipeline làm dày dữ liệu vòng kín do LLM điều khiển**, có đối chứng công bằng với phân bổ ngẫu nhiên và luật cứng ở cùng ngân sách.
2. **Giám sát theo vùng × loại artifact có trọng số tin cậy** cho nhãn MLLM nhiễu, đưa trực tiếp vào bộ phân loại.
3. **Protocol đánh giá tách identity trên Celeb-DF** kèm CI theo video, giúp đo tổng quát hóa thay vì chỉ AUC in-domain.
