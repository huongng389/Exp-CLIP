# Ý tưởng ban đầu

## Hướng chính: Lọc "ảo giác" (hallucination filtering) cho tri thức LLM

### Bối cảnh: Exp-CLIP dùng LLM thế nào
- Ảnh và instruction ("describe the facial actions…") được đưa vào BLIP-2 + FlanT5-XXL. Target của contrastive loss là **hidden state của T5 encoder** (mean-pool, 4096-d), không phải câu chữ.
- Câu mô tả **không hề được sinh ra**: `generate()` trong `models/BLIP2_T5.py` bị thay `forward` bằng `new_forward`, hàm này chỉ chạy T5 encoder rồi dừng, decoder không chạy (`models/Exp_CLIP.py:38` nhận `None` cho phần text).
- Hệ quả: nếu LLM "hiểu sai" khuôn mặt, embedding target bị nhiễu, và projection head học theo cái nhiễu đó. Không thể kiểm tra trực tiếp một vector 4096-d có đúng hay không.

### Ý tưởng cốt lõi
Cho decoder chạy thật (dùng `generate` gốc của LAVIS) để lấy **câu mô tả**, làm *bản đọc ra* (readout) những gì LLM "nghĩ" về ảnh, vì câu này được sinh từ chính embedding encoder đó. Kiểm tra câu này để ước lượng độ tin cậy của embedding target cho từng mẫu, rồi **gán trọng số cho mẫu trong InfoNCE** (hoặc bỏ mẫu). Câu chữ chỉ dùng để chấm điểm, không thay thế target.

### Ý tưởng 1: Kiểm chứng bằng AU detector (khuyên dùng)
- Chạy AU detector (VD OpenFace, LibreFace, ME-GraphAU) trên ảnh để lấy tập AU thực tế.
- Map câu mô tả LLM sang AU (bảng từ khóa → AU, VD "raised eyebrows" → AU1+AU2, "smiling" → AU12).
- Điểm tin cậy `w_AU` = mức khớp giữa AU trong câu và AU detector (VD F1/Jaccard). Mẫu bị LLM "bịa" cơ mặt thì có `w_AU` thấp và bị hạ trọng số trong loss.

### Ý tưởng 2: Độ đồng thuận của LLM (self-consistency)
- Với mỗi ảnh, chạy LLM K lần (đổi instruction paraphrase, hoặc augment ảnh nhẹ).
- Đo độ đồng thuận:
  - mức embedding: cosine giữa K vector encoder;
  - mức câu chữ: độ trùng tập AU/từ khóa giữa K câu.
- Đồng thuận thấp nghĩa là uncertainty cao, nên hạ trọng số hoặc loại mẫu (`w_cons`).

### Ý tưởng 3: Kết hợp
- Confidence score `w = f(w_AU, w_cons)` (VD tích hoặc trung bình có trọng số).
- Loss: `L = Σ_i w_i · ℓ_i / Σ_i w_i`, trong đó `ℓ_i` là InfoNCE đối xứng của mẫu i.

### Ghi chú triển khai
- Dữ liệu train (CAER-S) cố định, nên có thể **tính trước offline** câu mô tả, AU và `w` cho từng ảnh rồi lưu ra file. Khi train chỉ cần đọc `w`, không tốn thêm chi phí.
- Sinh câu bằng model BLIP-2 chưa bị patch (cùng instruction, cùng tiền xử lý ảnh như `train.py`); sửa `train.py` để dùng loss có trọng số.
