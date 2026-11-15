# Thiết Kế Chi Tiết: Module Đánh Giá Hiệu Năng Đa Phương Thức (`metrics.py` - S02-C1)

Tài liệu này đặc tả kiến trúc, công thức toán học, giao diện lập trình (API) và chiến lược kiểm thử cho module `metrics.py` thuộc nhiệm vụ **S02-C1** trong dự án *Disagreement-Aware Vision and Weight Sensor Fusion for AIoT Smart Shelf Inventory Monitoring*.

---

## 1. Tóm Tắt Thấu Hiểu (Understanding Summary)

* **Đối tượng triển khai**: Module tính toán chỉ số đánh giá chuẩn hóa (`fusion/metrics.py` và re-export tại `vision/metrics.py`), đi kèm bộ kiểm thử tự động `tests/test_metrics.py`.
* **Mục đích nghiên cứu**: Cung cấp một nguồn chân lý duy nhất (Single Source of Truth) về công thức đo lường cho cả 3 tầng đánh giá trong khóa luận:
  1. **Tầng Nhận diện Thị giác (Object Detection)**: Đánh giá chất lượng phát hiện khung hình chữ nhật của mô hình YOLO thông qua `IoU`, `AP`, `mAP@0.5` và `mAP@0.5:0.95`.
  2. **Tầng Đếm Tồn kho (Inventory Counting)**: Định lượng sai số đếm sản phẩm của từng kênh (`Vision-Only`, `Weight-Only`, `Fixed Fusion`, `Proposed Disagreement-Aware Fusion`) qua `Counting MAE`, `RMSE`, `Exact Accuracy`, `Undercount Rate` và `Overcount Rate`.
  3. **Tầng Sự kiện & Bất đồng (Inventory & Conflict Events)**: Đánh giá năng lực phát hiện sự kiện lấy/thêm hàng (`TAKE`, `RESTOCK`, `IDLE`), cảnh báo (`Low-Stock`, `Stock-Out`) và phân loại trạng thái bất đồng (`CONCORDANT`, `OCCLUSION`, `FOREIGN_OBJECT`) qua `Precision`, `Recall`, `F1-Score` và `False Alarm Rate (FAR)`.
* **Đối tượng phục vụ**: Các kịch bản thực nghiệm đối sánh (`EXP-1` đến `EXP-6`), script đánh giá baseline (`eval_yolov8n_baseline.py`), và Notebook huấn luyện trên Google Colab (`S02-C3`).
* **Ràng buộc kỹ thuật**:
  * Chỉ sử dụng các thư viện chuẩn đã khai báo trong `requirements.txt` (`numpy`, `pandas`, `ultralytics`, `pytest`).
  * Không phụ thuộc cứng vào GPU hay phần cứng ESP32/Camera khi thực thi kiểm thử đơn vị (Unit Test).
* **Mục tiêu loại trừ (Explicit Non-Goals)**:
  * Không chứa logic điều khiển cảm biến hoặc thuật toán hợp nhất quyết định (thuộc module `fusion_engine.py` ở Sprint 03).

---

## 2. Giả Định Kỹ Thuật & Yêu Cầu Phi Chức Năng (Assumptions & NFRs)

1. **Hiệu năng (Performance)**: Các phép toán đếm và phân loại sự kiện được vector hóa bằng `NumPy`, đảm bảo thời gian xử lý `< 50 ms` cho 10.000 mẫu thực nghiệm trên CPU.
2. **Độ tin cậy (Reliability & Edge Cases)**:
   * Xử lý an toàn phép chia cho `0` (khi `TP + FP = 0` hoặc `TP + FN = 0`), trả về giá trị mặc định `zero_division=0.0` thay vì phát sinh `ZeroDivisionError` hoặc `NaN`.
   * Tự động loại bỏ các cặp giá trị thiếu (`None` hoặc `np.nan`) nếu chế độ `dropna=True` được bật, hoặc ném ngoại lệ `ValueError` tường minh khi dữ liệu đầu vào rỗng / lệch kích thước.
3. **Tính tương thích đa định dạng (Interoperability)**: Chấp nhận đầu vào linh hoạt gồm `list`, `tuple`, `numpy.ndarray`, hoặc `pandas.Series`.

---

## 3. Rủi Ro Chính & Biện Pháp Giảm Thiểu (Risks & Mitigations)

* **Rủi ro 1: Phụ thuộc nặng vào đối tượng nội bộ của `ultralytics` khi chạy Unit Test cho `mAP`**:
  * *Biện pháp*: Thiết kế `map_wrapper` hoạt động ở 2 chế độ: (1) tính toán độc lập bằng thuật toán ghép cặp IoU trên `NumPy` từ danh sách tọa độ Bounding Box và (2) bóc tách trực tiếp từ đối tượng `DetMetrics` / `results_dict` của `ultralytics`.
* **Rủi ro 2: Sai lệch quy ước đơn vị phần trăm (`0.0–1.0` vs `0.0–100.0`) giữa các bảng biểu báo cáo**:
  * *Biện pháp*: Chuẩn hóa đầu ra mặc định trong khoảng tỷ lệ `[0.0, 1.0]` chuẩn khoa học dữ liệu, đồng thời cung cấp tham số `as_percentage: bool = False` để xuất trực tiếp thang `[0.0, 100.0]` cho báo cáo Markdown.
* **Rủi ro 3: Chia cho 0 khi thực nghiệm trên kệ tĩnh không phát sinh sự kiện (Support = 0)**:
  * *Biện pháp*: Kiểm soát mẫu số ở mọi hàm `Precision`, `Recall`, `F1`, `FAR` với tham số `zero_division: float = 0.0`.

---

## 4. Nhật Ký Quyết Định Thiết Kế (Decision Log)

| Thời điểm | Quyết định | Các phương án đã cân nhắc | Lý do lựa chọn |
| :--- | :--- | :--- | :--- |
| Giai đoạn Brainstorming | **Đặt module chính tại `fusion/metrics.py` và re-export sang `vision/metrics.py`** | - PA 1: `fusion/metrics.py` + re-export sang `vision/`<br>- PA 2: Đặt ở thư mục gốc `metrics.py`<br>- PA 3: Đặt riêng trong `vision/metrics.py` | Giữ cấu trúc package chuyên nghiệp, cho phép cả pipeline thị giác (`vision`) lẫn động cơ hợp nhất (`fusion`) truy cập thuận tiện mà không gây trùng lặp mã nguồn. |
| Giai đoạn Brainstorming | **Hỗ trợ kép cho `map_wrapper` (Standalone NumPy IoU + Ultralytics Extractor)** | - Chỉ bọc kết quả `model.val()` của Ultralytics<br>- Cài thêm `pycocotools` hoặc `torchmetrics` | Cho phép kiểm thử đơn vị (Unit Test) nhanh, nhẹ, xác định 100% trên dữ liệu tọa độ box giả lập mà không cần tải trọng số YOLO hay cài thêm thư viện ngoài. |
| Giai đoạn Brainstorming | **Hỗ trợ cả Sự kiện Nhị phân (`binary`) và Đa lớp (`macro`, `micro`, `weighted`)** | - Chỉ hỗ trợ sự kiện nhị phân (`0/1`)<br>- Chỉ hỗ trợ sự kiện biến thiên số lượng (`Take/Restock`) | Hệ thống Smart Shelf vừa có cảnh báo nhị phân (`Low-Stock`, `Stock-Out`, `Disagreement`) vừa có phân loại đa trạng thái (`TAKE`, `RESTOCK`, `IDLE` hoặc các chế độ xung đột cảm biến). |

---

## 5. Thiết Kế Chi Tiết & Công Thức Toán Học

### 5.1. Nhóm Chỉ Số Đếm Tồn Kho (Inventory Counting Metrics)

Cho tập $N$ quan trắc với số lượng thực tế $y = (y_1, y_2, \dots, y_N)$ và số lượng dự đoán $\hat{y} = (\hat{y}_1, \hat{y}_2, \dots, \hat{y}_N)$:

1. **Counting MAE (`counting_mae`)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|$$
2. **Counting RMSE (`counting_rmse`)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}$$
3. **Exact Count Accuracy (`exact_accuracy`)**:
   $$\text{Exact Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)$$
4. **Undercount & Overcount Rate (`undercount_rate`, `overcount_rate`)**:
   $$\text{Undercount Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i < y_i), \quad \text{Overcount Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i > y_i)$$

### 5.2. Nhóm Chỉ Số Nhận Diện Vật Thể (`map_wrapper`)

1. **Độ chồng lấp IoU (`compute_iou`, `compute_iou_matrix`)**:
   $$\text{IoU}(B_{pred}, B_{gt}) = \frac{\text{Area}(B_{pred} \cap B_{gt})}{\text{Area}(B_{pred} \cup B_{gt})}$$
   Hỗ trợ cả định dạng `xyxy` ($[x_1, y_1, x_2, y_2]$) và `xywh` ($[x_c, y_c, w, h]$).
2. **Average Precision (`compute_ap`) & Mean Average Precision (`map_wrapper`)**:
   * Sắp xếp các dự đoán của từng lớp theo độ tự tin (`conf`) giảm dần.
   * Ghép cặp tham lam (Greedy Matching) từng dự đoán với Ground Truth cùng lớp trong cùng ảnh có $\text{IoU} \ge \tau_{\text{IoU}}$ lớn nhất chưa được ghép.
   * Nội suy đường cong Precision–Recall theo chuẩn 101 điểm của MS-COCO (`coco101`) hoặc tích phân liên tục (`continuous`).
   * Tính `mAP@0.5` ($\tau_{\text{IoU}} = 0.50$), `mAP@0.75` ($\tau_{\text{IoU}} = 0.75$) và `mAP@0.5:0.95` (trung bình trên 10 ngưỡng $\tau_{\text{IoU}} \in \{0.50, 0.55, \dots, 0.95\}$).

### 5.3. Nhóm Chỉ Số Sự Kiện & Cảnh Báo (`event_precision_recall_f1`)

Với mỗi lớp sự kiện mục tiêu, xác định `TP` (True Positive), `FP` (False Positive), `FN` (False Negative) và `TN` (True Negative):

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{False Alarm Rate (FAR)} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

Hỗ trợ hàm tiện ích `derive_inventory_events(counts_prev, counts_curr, delta_threshold=0)` để suy diễn chuỗi sự kiện `TAKE` ($\Delta N < -\delta$), `RESTOCK` ($\Delta N > +\delta$) và `IDLE` ($|\Delta N| \le \delta$) từ chuỗi số lượng tồn kho theo thời gian.

---

## 6. Chiến Lược Kiểm Thử (`tests/test_metrics.py`)

* **Unit Tests cho Counting Metrics**: Kiểm tra trường hợp đếm khớp hoàn toàn (`MAE = 0, Accuracy = 1.0`), trường hợp đếm thiếu/đếm thừa hỗn hợp, tùy chọn `as_percentage=True`, tự động bỏ qua `None`/`NaN`, và bắt lỗi mảng rỗng / lệch độ dài.
* **Unit Tests cho `map_wrapper`**: Kiểm tra tính `IoU` (`xyxy` & `xywh`), kiểm tra `mAP@0.5` và `mAP@0.5:0.95` trên tập khung hình giả lập (khớp 100%, có False Positive/False Negative, nhiều lớp SKU, không có vật thể), và kiểm tra trích xuất từ đối tượng mock `ultralytics`.
* **Unit Tests cho Event Metrics**: Kiểm tra phân loại sự kiện nhị phân (`0/1`, `bool`, nhãn chuỗi `"DISAGREEMENT"`), phân loại đa lớp (`TAKE`, `RESTOCK`, `IDLE` với `macro`, `micro`, `weighted`), xử lý an toàn khi `TP = FP = FN = 0`, và kiểm tra đồng bộ import giữa `fusion.metrics` và `vision.metrics`.

