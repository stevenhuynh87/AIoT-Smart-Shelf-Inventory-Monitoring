# Thiết Kế Chi Tiết: Đánh Giá Hiệu Năng Cơ Sở YOLOv8n Pretrained Trên 33 Ảnh Kệ Hàng Tự Chụp

Tài liệu này đặc tả toàn diện kiến trúc, quy trình thực thi và phương pháp đánh giá thực nghiệm cho bài toán kiểm thử mô hình thị giác **YOLOv8n (pretrained COCO)** trên tập dữ liệu 33 ảnh kệ hàng thực tế trong dự án *AIoT Smart Shelf Inventory Monitoring*.

---

## 1. Tóm Tắt Thấu Hiểu (Understanding Summary)

* **Đối tượng triển khai**: Pipeline thực nghiệm đánh giá độc lập mô hình học sâu `YOLOv8n` (pretrained weights `yolov8n.pt` trên tập nhãn MS-COCO) trên 33 ảnh kệ hàng tự chụp.
* **Mục đích nghiên cứu**: Thiết lập chỉ số hiệu năng cơ sở ban đầu (Vision Baseline Benchmark) cho hệ thống AIoT Smart Shelf; định lượng sai số đếm giữa mô hình thị giác tổng quát chưa fine-tune và số lượng thực tế đếm bằng mắt (Human Ground Truth); chỉ ra nguyên nhân và tỷ lệ phát sinh bất đồng (che khuất, góc tối, phản quang) để chứng minh tính tất yếu của việc dung hợp với cảm biến tải trọng (Load cell).
* **Đối tượng phục vụ**: Nghiên cứu sinh / người phát triển đề tài AIoT Smart Shelf phục vụ đo đạc thực nghiệm cho luận văn.
* **Ràng buộc kỹ thuật**:
  * Dữ liệu đầu vào: 33 tệp ảnh kệ hàng hiện có tại thư mục `Công việc/` (gồm 4 nhóm: `nhóm 1` đến `Nhóm 4`).
  * Lớp mục tiêu lọc từ 80 nhãn COCO: `bottle` (class ID 39) và `cup` (class ID 41) đại diện cho chai/lon nước giải khát.
  * Tham số suy luận tiêu chuẩn: `conf=0.25`, `iou=0.45`, `imgsz=640` trên toàn khung hình gốc.
  * Kết quả đầu ra: Báo cáo số liệu chi tiết (`evaluation_results.csv`), báo cáo tổng hợp Markdown (`evaluation_summary.md`), và lưu toàn bộ 33 ảnh đã vẽ Bounding Box kèm nhãn trực quan vào `output/baseline_yolov8n/annotated_frames/`.
* **Mục tiêu loại trừ (Explicit Non-Goals)**:
  * Chưa huấn luyện lại hay tinh chỉnh (fine-tune) trọng số YOLOv8 ở giai đoạn này.
  * Chưa tích hợp kênh cảm biến trọng lượng (Load cell) hay thuật toán phân xử bất đồng tại bước thử nghiệm độc lập này.
  * Tuyệt đối không thay đổi, chỉnh sửa hay xóa ảnh gốc trong thư mục `Công việc/`.

---

## 2. Giả Định Kỹ Thuật (Assumptions)

1. Thư viện `ultralytics` và môi trường Python sẽ tải mô hình chính hãng `yolov8n.pt` tự động trong lần chạy đầu tiên nếu chưa có trong bộ nhớ đệm (cache).
2. Tên file ảnh có đuôi mở rộng kép dạng `shelf_XX.jpg.jpg` sẽ được script tự động chuẩn hóa ID (`shelf_01` -> `shelf_33`) khi ánh xạ nhãn Ground Truth.
3. Người dùng cung cấp danh sách số lượng đếm bằng mắt (Ground Truth) tương ứng cho 33 ảnh thông qua bảng cấu hình hoặc file CSV tham chiếu.
4. Quá trình suy luận chạy cục bộ bằng CPU hoặc GPU (CUDA nếu khả dụng) thông qua thư viện `ultralytics`.

---

## 3. Rủi Ro Chính & Biện Pháp Giảm Thiểu (Risks & Mitigations)

* **Rủi ro 1: Sai lệch tên file do đuôi kép (`.jpg.jpg`)**:
  * *Biện pháp*: Dùng hàm bóc tách Regex trích xuất tiền tố `shelf_\d+` độc lập với phần mở rộng để tránh sai lệch ánh xạ.
* **Rủi ro 2: Thiếu nhãn Ground Truth cho một số ảnh**:
  * *Biện pháp*: Thiết kế script hoạt động ở chế độ chịu lỗi mềm (graceful degradation) — nếu thiếu Ground Truth cho một ảnh, script vẫn chạy suy luận, lưu ảnh kết quả và chỉ tạm loại ảnh đó khỏi phép tính MAE/RMSE.
* **Rủi ro 3: Tràn bộ nhớ khi xử lý tuần tự ảnh độ phân giải cao**:
  * *Biện pháp*: Giải phóng bộ nhớ OpenCV sau mỗi ảnh và đóng cửa sổ vẽ để giữ mức tiêu thụ RAM ổn định.

---

## 4. Nhật Ký Quyết Định (Decision Log)

| Thời điểm | Quyết định | Các phương án đã cân nhắc | Lý do lựa chọn |
| :--- | :--- | :--- | :--- |
| Giai đoạn Brainstorming | **Lựa chọn Phương án 1: Script Python Độc lập (`eval_yolov8n_baseline.py`)** | - PA 1: Script Python CLI<br>- PA 2: Jupyter Notebook<br>- PA 3: Tích hợp đầy đủ vào hệ thống | Tối ưu hóa tính đơn giản, gọn nhẹ, có thể chạy tự động bằng 1 dòng lệnh mà không làm xáo trộn cấu trúc tệp gốc của dự án; dễ dàng trích xuất báo cáo và số liệu cho nghiên cứu. |
| Giai đoạn Brainstorming | **Lọc nhãn mục tiêu COCO `bottle` (39) & `cup` (41)** | - Lọc chỉ `bottle`<br>- Lọc tất cả class không giới hạn | Lon nước và chai nước trong COCO thường được mô hình pretrained phân loại xen kẽ giữa `bottle` và `cup`. Kết hợp 2 class này giúp thu hồi tối đa các đối tượng đồ uống thực tế trên kệ. |
| Giai đoạn Brainstorming | **Tham số suy luận chuẩn (`conf=0.25`, `iou=0.45`, `imgsz=640`)** | - Ngưỡng tự tin cao (>= 0.40)<br>- Cắt vùng quan tâm (ROI) | Giữ nguyên baseline chuẩn của YOLOv8 giúp đánh giá khách quan năng lực nhận diện zero-shot trước khi quyết định can thiệp tiền xử lý ROI. |

---

## 5. Thiết Kế Chi Tiết Hệ Thống

### 5.1. Cấu Trúc Thư Mục & Phân Tách Trách Nhiệm

```text
AIoT-Smart-Shelf/
├── Công việc/                       # [INPUT GỐC - CHỈ ĐỌC]
│   ├── nhóm 1/                      # Chứa 6 ảnh (shelf_01 -> shelf_6)
│   ├── Nhóm 2/                      # Chứa 11 ảnh (shelf_7 -> shelf_16, she)
│   ├── Nhóm 3/                      # Chứa 6 ảnh (shelf_18 -> shelf_23)
│   └── Nhóm 4/                      # Chứa 10 ảnh (shelf_24 -> shelf_33)
├── vision/
│   ├── eval_yolov8n_baseline.py     # Script thực thi chính
│   └── ground_truth_yolo.csv        # Bảng ánh xạ số lượng đếm bằng mắt cho 33 ảnh
└── output/baseline_yolov8n/         # [OUTPUT ĐƯỢC TỰ ĐỘNG TẠO MỚI]
    ├── annotated_frames/            # 33 ảnh kết quả đã vẽ Bounding Box và số đếm
    ├── evaluation_results.csv       # File CSV lưu kết quả chi tiết từng ảnh
    └── evaluation_summary.md        # Báo cáo tổng kết chỉ số (MAE, RMSE, Accuracy)
```

### 5.2. Quy Trình Suy Luận (Inference Pipeline)

1. **Quét dữ liệu**: Đọc đệ quy toàn bộ các ảnh từ 4 nhóm trong `Công việc/`, sắp xếp tự nhiên theo số thứ tự `shelf_01` đến `shelf_33`.
2. **Nạp Ground Truth**: Đọc tệp ánh xạ `ground_truth_yolo.csv` (chứa `image_id,gt_count`).
3. **Thực thi YOLOv8n**:
   * Nạp mô hình `ultralytics.YOLO('yolov8n.pt')`.
   * Chạy suy luận với `conf=0.25, iou=0.45, imgsz=640`.
   * Lọc các bounding box có `cls in [39, 41]`.
4. **Trực quan hóa & Lưu ảnh**:
   * Vẽ khung chữ nhật bounding box và điểm tự tin cho từng vật phẩm.
   * Thêm banner tiêu đề trên ảnh: `[Ảnh ID] GT: X | Vision: Y | Error: Z`.
   * Lưu vào thư mục `output/baseline_yolov8n/annotated_frames/`.
5. **Tính toán Chỉ số Đánh giá (Metrics)**:
   * **MAE**: $\text{MAE} = \frac{1}{N} \sum_{i=1}^N |Count_{vision}^{(i)} - Count_{gt}^{(i)}|$
   * **RMSE**: $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (Count_{vision}^{(i)} - Count_{gt}^{(i)})^2}$
   * **Exact Match Accuracy**: Tỷ lệ ảnh có $Count_{vision} == Count_{gt}$.
   * **Tỷ lệ Đếm thiếu (Undercount Rate)**: Tỷ lệ ảnh có $Count_{vision} < Count_{gt}$.
   * **Tỷ lệ Đếm thừa (Overcount Rate)**: Tỷ lệ ảnh có $Count_{vision} > Count_{gt}$.
6. **Xuất báo cáo**: Ghi toàn bộ dữ liệu ra `evaluation_results.csv` và tạo báo cáo markdown `evaluation_summary.md`.

---

## 6. Chiến Lược Kiểm Thử (Testing Strategy)

1. **Smoke Test**: Kiểm thử chạy thử nghiệm 1 ảnh đầu tiên (`shelf_01`) để kiểm chứng: nạp mô hình thành công, trích xuất bounding box thành công, xuất ảnh annotate thành công.
2. **Full Batch Run**: Chạy toàn bộ 33 ảnh, đo thời gian suy luận trung bình (Inference FPS / Latency) và kiểm tra tính toàn vẹn của tệp CSV đầu ra.
