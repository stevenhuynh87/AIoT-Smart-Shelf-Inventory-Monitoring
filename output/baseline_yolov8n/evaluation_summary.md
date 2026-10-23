# Báo Cáo Thực Nghiệm: Đánh Giá Hiệu Năng Cơ Sở YOLOv8n Pretrained Trên Kệ Hàng

- **Thời gian chạy**: 2026-10-23 19:37:27
- **Mô hình**: YOLOv8n (`yolov8n.pt`, MS-COCO 80 classes)
- **Lớp mục tiêu**: `bottle` (39) & `cup` (41)
- **Tham số cấu hình**: `conf=0.25`, `iou=0.45`, `imgsz=640`
- **Tổng số ảnh xử lý**: 32 ảnh
- **Số ảnh đã có nhãn Ground Truth**: 32 ảnh

---

## 1. Các Chỉ Số Hiệu Năng Toàn Cục (Overall Metrics)

| Chỉ số đo lường | Giá trị | Ý nghĩa kỹ thuật |
| :--- | :---: | :--- |
| **MAE (Mean Absolute Error)** | **3.00** chiếc | Sai số trung bình giữa mô hình và số lượng thực tế |
| **RMSE (Root Mean Square Error)** | **4.04** chiếc | Mức độ phân tán sai số (phạt nặng các lỗi đếm lệch lớn) |
| **Độ chính xác tuyệt đối (Accuracy)** | **21.9%** (7/32) | Tỷ lệ ảnh đếm chính xác hoàn toàn (Vision == GT) |
| **Tỷ lệ đếm thiếu (Undercount Rate)** | **71.9%** (23/32) | Do vật thể bị che khuất (occlusion) hoặc thiếu sáng |
| **Tỷ lệ đếm thừa (Overcount Rate)** | **6.2%** (2/32) | Do nhận diện nhầm chi tiết nền hoặc bóng đổ |

---

## 2. Thống Kê Theo Từng Nhóm Ảnh (Group-by-Group Breakdown)

| Nhóm Ảnh | Số lượng ảnh | MAE (chiếc) | Tỷ lệ Đếm Đúng | Tỷ lệ Đếm Thiếu | Tỷ lệ Đếm Thừa | Avg Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Nhóm 2` | 10 | 3.20 | 10.0% | 80.0% | 10.0% | 0.41 |
| `Nhóm 3` | 6 | 5.67 | 0.0% | 100.0% | 0.0% | 0.29 |
| `Nhóm 4` | 10 | 2.70 | 30.0% | 70.0% | 0.0% | 0.44 |
| `nhóm 1` | 6 | 0.50 | 50.0% | 33.3% | 16.7% | 0.51 |

---

## 3. Bảng Chi Tiết Kết Quả Từng Ảnh

| ID | Nhóm | Kịch bản thực nghiệm | GT | YOLOv8n | Sai số | Trạng thái | Chi tiết nhận diện | Độ tin cậy TB |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `shelf_01` | nhóm 1 | Baseline (1 sp - dat thua) | 1 | **1** | +0 | `EXACT` | 1 cup | 0.57 |
| `shelf_02` | nhóm 1 | Baseline (2 sp - dat thua) | 2 | **3** | +1 | `OVERCOUNT` | 2 bottle, 1 cup | 0.54 |
| `shelf_03` | nhóm 1 | Baseline (3 sp - dat thua) | 3 | **3** | +0 | `EXACT` | 3 bottle | 0.68 |
| `shelf_4` | nhóm 1 | Baseline (1 sp - dat thua) | 1 | **0** | -1 | `UNDERCOUNT` | none | 0.00 |
| `shelf_5` | nhóm 1 | Baseline (2 sp - dat thua) | 2 | **1** | -1 | `UNDERCOUNT` | 1 bottle | 0.62 |
| `shelf_6` | nhóm 1 | Baseline (3 sp - dat thua) | 3 | **3** | +0 | `EXACT` | 2 cup, 1 bottle | 0.64 |
| `shelf_7` | Nhóm 2 | Low Occlusion (4 sp - 2 hang so le) | 4 | **3** | -1 | `UNDERCOUNT` | 2 bottle, 1 cup | 0.38 |
| `shelf_8` | Nhóm 2 | Low Occlusion (5 sp - 2 hang so le) | 5 | **5** | +0 | `EXACT` | 3 vase, 1 cup, 1 bottle | 0.47 |
| `shelf_9` | Nhóm 2 | Low Occlusion (5 sp - 2 hang so le) | 5 | **3** | -2 | `UNDERCOUNT` | 2 bottle, 1 cup | 0.46 |
| `shelf_10` | Nhóm 2 | Low Occlusion (4 sp - 2 hang so le) | 4 | **3** | -1 | `UNDERCOUNT` | 1 cup, 2 bottle | 0.43 |
| `shelf_11` | Nhóm 2 | Low Occlusion (4 sp - 2 hang so le) | 4 | **5** | +1 | `OVERCOUNT` | 1 vase, 2 cup, 1 bottle, 1 book | 0.32 |
| `shelf_12` | Nhóm 2 | Low Occlusion (5 sp - 2 hang so le) | 5 | **1** | -4 | `UNDERCOUNT` | 1 cup | 0.34 |
| `shelf_13` | Nhóm 2 | Low Occlusion (5 sp - 2 hang so le) | 5 | **2** | -3 | `UNDERCOUNT` | 1 bottle, 1 cup | 0.51 |
| `shelf_14` | Nhóm 2 | Low Occlusion (11 sp - xep dong) | 11 | **3** | -8 | `UNDERCOUNT` | 2 cup, 1 bottle | 0.42 |
| `shelf_15` | Nhóm 2 | Low Occlusion (11 sp - xep dong) | 11 | **3** | -8 | `UNDERCOUNT` | 2 bottle, 1 vase | 0.35 |
| `shelf_16` | Nhóm 2 | Low Occlusion (11 sp - xep dong) | 11 | **7** | -4 | `UNDERCOUNT` | 2 vase, 3 bottle, 2 cup | 0.43 |
| `shelf_18` | Nhóm 3 | Heavy Occlusion (12 sp - che khuat >50%) | 12 | **5** | -7 | `UNDERCOUNT` | 2 vase, 2 bottle, 1 cup | 0.41 |
| `shelf_19` | Nhóm 3 | Heavy Occlusion (12 sp - che khuat >50%) | 12 | **5** | -7 | `UNDERCOUNT` | 3 bottle, 1 vase, 1 cup | 0.47 |
| `shelf_20` | Nhóm 3 | Heavy Occlusion (6 sp - cum day dac) | 6 | **0** | -6 | `UNDERCOUNT` | none | 0.00 |
| `shelf_21` | Nhóm 3 | Heavy Occlusion (6 sp - cum day dac) | 6 | **2** | -4 | `UNDERCOUNT` | 2 bottle | 0.54 |
| `shelf_22` | Nhóm 3 | Heavy Occlusion (6 sp - cum day dac) | 6 | **2** | -4 | `UNDERCOUNT` | 2 cup | 0.35 |
| `shelf_23` | Nhóm 3 | Heavy Occlusion (6 sp - cum day dac) | 6 | **0** | -6 | `UNDERCOUNT` | none | 0.00 |
| `shelf_24` | Nhóm 4 | Environment Noise (0 sp - ke trong) | 0 | **0** | +0 | `EXACT` | none | 0.00 |
| `shelf_25` | Nhóm 4 | Environment Noise (2 sp - thieu sang) | 2 | **1** | -1 | `UNDERCOUNT` | 1 cup | 0.84 |
| `shelf_26` | Nhóm 4 | Environment Noise (2 sp - thieu sang) | 2 | **2** | +0 | `EXACT` | 2 cup | 0.65 |
| `shelf_27` | Nhóm 4 | Foreign Object (3 sp + 1 di vat) | 3 | **1** | -2 | `UNDERCOUNT` | 1 cup | 0.82 |
| `shelf_28` | Nhóm 4 | Foreign Object (3 sp + 1 di vat) | 3 | **1** | -2 | `UNDERCOUNT` | 1 bottle | 0.51 |
| `shelf_29` | Nhóm 4 | Environment Noise (4 sp - choi sang) | 4 | **4** | +0 | `EXACT` | 4 bottle | 0.58 |
| `shelf_30` | Nhóm 4 | Environment Noise (4 sp - choi sang) | 4 | **0** | -4 | `UNDERCOUNT` | none | 0.00 |
| `shelf_31` | Nhóm 4 | Environment Noise (8 sp - bong do) | 8 | **4** | -4 | `UNDERCOUNT` | 3 bottle, 1 cup | 0.49 |
| `shelf_32` | Nhóm 4 | Environment Noise (8 sp - bong do) | 8 | **0** | -8 | `UNDERCOUNT` | none | 0.00 |
| `shelf_33` | Nhóm 4 | Environment Noise (8 sp - bong do) | 8 | **2** | -6 | `UNDERCOUNT` | 1 cup, 1 book | 0.48 |

---

## 4. Nhận Xét Khoa Học Phục Vụ Luận Văn
1. **Hiện tượng đếm thiếu (Under-detection)**: Mô hình pretrained YOLOv8n gặp khó khăn khi các chai/lon xếp chồng hàng sâu phía sau hoặc bị che khuất một phần (occlusion).
2. **Đóng góp của Đề tài**: Kết quả baseline này chứng minh rõ ràng việc chỉ sử dụng một mình camera thị giác là không đủ tin cậy cho giám sát tồn kho bán lẻ tự động.
3. **Tính tất yếu của Fusion**: Cảm biến trọng lượng (Load cell) hoàn toàn miễn nhiễm với che khuất thị giác, cung cấp kênh thông tin độc lập giúp cơ chế **Disagreement-Aware Fusion** phân xử và khôi phục số lượng hàng hóa chính xác.