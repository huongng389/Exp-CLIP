# Ý tưởng ban đầu

## Hướng chính: Lọc "ảo giác" (hallucination filtering) cho tri thức LLM

### Bối cảnh: Exp-CLIP dùng LLM thế nào
- Ảnh và instruction ("describe the facial actions…") được đưa vào BLIP-2 + FlanT5-XXL. Target của contrastive loss là **hidden state của T5 encoder** (mean-pool, 4096-d), không phải câu chữ.
- Câu mô tả **không hề được sinh ra**: `generate()` trong `models/BLIP2_T5.py` bị thay `forward` bằng `new_forward`, hàm này chỉ chạy T5 encoder rồi dừng, decoder không chạy (`models/Exp_CLIP.py:38` nhận `None` cho phần text).
- Hệ quả: nếu LLM "hiểu sai" khuôn mặt, embedding target bị nhiễu, và projection head học theo cái nhiễu đó. Không thể kiểm tra trực tiếp một vector 4096-d có đúng hay không.

### Ý tưởng cốt lõi
Ước lượng độ tin cậy của **vector target** cho từng ảnh, rồi **gán trọng số cho ảnh trong InfoNCE** (hoặc loại ảnh). Kiểm tra thẳng trên vector, đúng không gian mà loss dùng, không cần cho decoder sinh câu.

### Ý tưởng 1: Kiểm chứng bằng AU detector (khuyên dùng)
- Chạy AU detector (VD py-feat, LibreFace, ME-GraphAU) trên ảnh để lấy tập AU thực tế.
- **Đọc AU thẳng từ vector target, không qua chữ:** encode các câu mẫu mô tả từng AU (VD "the eyebrows are lowered and drawn together" → AU4) bằng cùng T5 encoder. Cosine giữa target và các vector mẫu cho ra "hồ sơ AU theo LLM" của ảnh.
- Điểm tin cậy `w_AU` = mức khớp giữa hồ sơ AU theo LLM và AU detector. Ảnh nào vector target "nói" AU khác hẳn detector thì bị hạ trọng số trong loss.
- (Hướng cũ là decode câu rồi map sang AU bằng từ khóa; đã bỏ vì gián tiếp và câu quá ngắn, xem EXP-001.)

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
- Dữ liệu train (CAER-S) cố định, nên có thể **tính trước offline** vector target, AU và `w` cho từng ảnh rồi lưu ra file. Khi train chỉ cần đọc `w`, không tốn thêm chi phí.
- Sửa `train.py` để dùng loss có trọng số.
