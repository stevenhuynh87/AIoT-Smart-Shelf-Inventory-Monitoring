# KẾ HOẠCH HỌC TẬP VÀ TRIỂN KHAI KHÓA LUẬN TỐT NGHIỆP
# ĐỀ TÀI: DISAGREEMENT-AWARE VISION AND WEIGHT SENSOR FUSION FOR AIoT SMART SHELF INVENTORY MONITORING

---

* **Loại tài liệu:** Bản kế hoạch hành động chi tiết (Action Plan & Curriculum Roadmap)
* **Đối tượng:** Sinh viên Đại học ngành Kỹ thuật Điện - Điện tử / Hệ thống Nhúng / AIoT / Computer Vision
* **Thời gian thực hiện:** 6 tháng hoàn thành khóa luận chính khóa + 2 tháng tối ưu & viết bài báo khoa học (Tổng: 8 tháng)
* **Địa điểm lưu trữ:** `c:\Users\ADMIN\Documents\Senior Project\Plan\KE_HOACH_HOC_TAP_VA_TRIEN_KHAI_AIOT_SMART_SHELF.md`

---

## MỤC LỤC TỔNG QUAN

1. [PHẦN 1: Phân Tích Hiện Trạng & Xác Định Lộ Trình](#phần-1-phân-tích-hiện-trạng--xác-định-lộ-trình)
2. [PHẦN 2: Kế Hoạch Tổng Thể 6–8 Tháng](#phần-2-kế-hoạch-tổng-thể-68-tháng)
3. [PHẦN 3: Kế Hoạch Chi Tiết Từng Tháng (Tháng 1 đến Tháng 8)](#phần-3-kế-hoạch-chi-tiết-từng-tháng)
4. [PHẦN 4: Phân Chia Tỷ Lệ Thời Gian Học Lý Thuyết & Thực Hành](#phần-4-phân-chia-tỷ-lệ-thời-gian-học-lý-thuyết--thực-hành)
5. [PHẦN 5: Lịch Học Tập & Làm Dự Án Mẫu Hằng Tuần](#phần-5-lịch-học-tập--làm-dự-án-mẫu-hằng-tuần)
6. [PHẦN 6: Lộ Trình Kiến Thức Theo Thứ Tự Phụ Thuộc (Dependency Graphs)](#phần-6-lộ-trình-kiến-thức-theo-thứ-tự-phụ-thuộc)
7. [PHẦN 7: Thiết Kế & Kế Hoạch Thực Nghiệm Chi Tiết (EXP-1 đến EXP-6)](#phần-7-thiết-kế--kế-hoạch-thực-nghiệm-chi-tiết)
8. [PHẦN 8: Ma Trận Quản Lý Rủi Ro & Kế Hoạch Dự Phòng (Risk Matrix & Fallback)](#phần-8-ma-trận-quản-lý-rủi-ro--kế-hoạch-dự-phòng)
9. [PHẦN 9: Khung Theo Dõi Tiến Độ (Progress Tracking Template)](#phần-9-khung-theo-dõi-tiến-độ)
10. [PHẦN 10: Bảng Câu Hỏi Làm Rõ & Xác Nhận Thiết Kế (Clarification & Next Steps)](#phần-10-bảng-câu-hỏi-làm-rõ--xác-nhận-thiết-kế)

---

# PHẦN 1: PHÂN TÍCH HIỆN TRẠNG & XÁC ĐỊNH LỘ TRÌNH

Dựa trên tài liệu Brainstorming (`2026_09_22_brainstorming.pdf` / `Idea_1_AIOT_Smart_Shelf_Thesis.md`), dưới đây là phân tích toàn diện hiện trạng kỹ thuật và phương pháp luận triển khai:

### 1.1 Mục Tiêu Cuối Cùng Của Dự Án
Xây dựng một hệ thống **AIoT Smart Shelf** hoàn chỉnh (end-to-end prototype) giải quyết bài toán giám sát tồn kho tự động trong bán lẻ quy mô nhỏ (minimart, pharmacy) với các đóng góp cốt lõi:
1. **Multimodal Fusion:** Kết hợp Computer Vision (camera) và trọng lượng (Load Cell + HX711) để khắc phục điểm yếu đếm thiếu khi bị che khuất (occlusion) của camera và điểm yếu không phân loại được SKU của cảm biến cân.
2. **Disagreement-Aware Verification:** Thuật toán phát hiện độ bất đồng $D_t = |\hat{N}_{v,t} - \hat{N}_{w,t}|$. Khi $D_t > \tau_D$ hoặc độ tin cậy $C_v < \tau_C$, kích hoạt cơ chế tái suy luận (verification mode) thích ứng.
3. **Event-Triggered Two-Tier Adaptive Inference:** Chỉ đánh thức camera và bộ suy luận khi phát hiện thay đổi trọng lượng ổn định trên kệ, giúp tiết kiệm từ 60–80% tài nguyên tính toán và năng lượng so với việc quét camera liên tục 30 FPS.
4. **End-to-End System:** Luồng dữ liệu hoàn chỉnh từ Sensor $\to$ ESP32 $\to$ MQTT $\to$ Edge Computer (Python AI Fusion) $\to$ Web Dashboard giám sát theo thời gian thực và cảnh báo thiếu hàng/lệch hàng.

### 1.2 Các Module Kỹ Thuật Chính
* **Module 1: Hardware & Sensor Node (Firmware):** ESP32 kết nối 2–4 Load Cell qua mô-đun ADC 24-bit HX711. Lập trình C/C++ xử lý lọc nhiễu số (Moving Average / Median Filter), phát hiện trạng thái cân ổn định (Settling State Machine), gửi gói tin JSON qua MQTT qua Wi-Fi.
* **Module 2: Edge Vision Node:** Camera USB/CSI gắn góc nhìn bao quát kệ. Mô hình Object Detection nhẹ (YOLOv8n / YOLO11n) phát hiện 5–10 loại sản phẩm (SKU), trích xuất bounding box, class, count $\hat{N}_v$ và vector confidence $C_v$.
* **Module 3: Disagreement-Aware Fusion Engine:** Trọng tâm nghiên cứu của khóa luận. Nhận $\hat{N}_v, C_v$ từ Vision và cân nặng $W_t$ từ ESP32, tính $\hat{N}_w = \text{round}(W_t / w_{item})$, tính toán $D_t$, điều phối chuyển đổi Tier 1 (Lightweight) sang Tier 2 (Verification: High-res inference, multi-frame averaging, hoặc Bayesian update).
* **Module 4: IoT Communication & Backend Broker:** Mosquitto MQTT Broker làm trục giao tiếp trung tâm; Backend Python (FastAPI/Pydantic) xử lý logic nghiệp vụ tồn kho, lưu trữ nhật ký biến động vào SQLite/PostgreSQL.
* **Module 5: Web Dashboard:** Giao diện trực quan thời gian thực (Streamlit hoặc FastAPI + HTML/JS/Tailwind/Charts) hiển thị số lượng tồn từng ô, trạng thái sensor, log bất đồng và cảnh báo (Low Stock, Stock-Out, Sensor Disagreement).

### 1.3 Phân Tách: Nội Dung Trong PDF vs. Nội Dung Đề Xuất Bổ Sung

| Hạng mục | Đã xác định trong PDF Brainstorming | Đề xuất bổ sung của AI Mentor (Tối ưu hóa học tập & kỹ thuật) |
|---|---|---|
| **Mô hình Vision** | YOLO Nano (v8 hoặc 11) | Bổ sung quy trình Pre-training trên SKU-110K $\to$ Fine-tune trên Custom Mini Dataset (5–10 SKU). Đóng gói pipeline inference với ONNX Runtime để chuẩn hóa trên cả PC và Edge board. |
| **Xử lý tín hiệu cân** | HX711, Moving average, Calibration | Bổ sung **Bộ phát hiện ổn định 2 pha (Two-phase Settling State Machine):** Chờ $\Delta W < \epsilon$ trong cửa sổ trượt $T_{settle} \approx 500\text{ms}$ để loại trừ hiện tượng rung tay khi khách hàng bốc/đặt sản phẩm trước khi trigger AI. |
| **Sensor Fusion** | Công thức $D = \|\hat{N}_v - \hat{N}_w\|$, ngưỡng $\tau_D$ | Bổ sung phương pháp **Bayesian Discrete State Update** làm extension toán học nâng cao: Mô hình hóa xác suất $P(N \mid \hat{N}_v, W_t)$ giúp luận văn có hàm lượng toán học và nghiên cứu sâu sắc hơn. |
| **Phần cứng Edge** | Đề xuất PC ban đầu, sau đó RPi5 hoặc Jetson | **Tách biệt 2 Phase rõ ràng:** 6 tháng đầu chỉ chạy trên Laptop/PC nhằm đảm bảo 100% tiến độ khóa luận; Tháng 7–8 mới cross-compile và benchmark sang Raspberry Pi 5 / Jetson Orin Nano để viết paper. |
| **Quy mô kệ** | 1 kệ, 5–10 SKU, 1–2 load cell | Quy định cấu trúc: **Kệ 1 tầng chia 2–4 ô cân vật lý (Weighing Zones)**. Mỗi ô chỉ chứa duy nhất 1 loại SKU để tránh lỗi tổ hợp tuyến tính (subset sum problem) của cảm biến cân trong phạm vi luận văn đại học. |

### 1.4 Những Kiến Thức Cần Học & Có Thể Học Song Song
* **Cần học tuần tự (Bắt buộc nền tảng trước):**
  * Python cơ bản $\to$ NumPy / OpenCV $\to$ Chuẩn bị Dataset YOLO $\to$ Huấn luyện & Đánh giá YOLO.
  * C/C++ vi điều khiển $\to$ ESP32 GPIO/I2C/SPI $\to$ HX711 Read $\to$ Digital Filtering (DSP) $\to$ State Machine.
* **Có thể học song song (Độc lập module):**
  * Module Vision (Python/YOLO) hoàn toàn có thể phát triển song song với Module Sensor (C++/ESP32/HX711).
  * Giao thức MQTT & xây dựng Web Dashboard có thể học song song khi đang đợi huấn luyện mô hình.
* **Chưa cần học ở giai đoạn đầu:**
  * TensorRT, NCNN, C++ Edge Deployment, Docker, Kubernetes, Cloud AWS/GCP (các công cụ này chỉ học từ Tháng 7 trở đi).

### 1.5 Phân Tích Sự Phụ Thuộc Kỹ Thuật (Critical Path)
* **Điểm nghẽn 1 (Dataset):** Không có dataset gán nhãn $\to$ Không fine-tune được YOLO $\to$ Không có $\hat{N}_v$ thực tế. *Giải pháp:* Tuần 1 dùng pre-trained COCO (detect chai nước/cốc) làm mockup code trước, thu thập dataset thật từ Tuần 3.
* **Điểm nghẽn 2 (Nhiễu tải trọng):** Nếu tín hiệu Load Cell chưa lọc sạch $\to$ Bị trigger ảo liên tục (false trigger) $\to$ Hệ thống Vision bị nghẽn suy luận. *Giải pháp:* Phải hoàn thiện bộ lọc FIR/Moving Median trên ESP32 trước khi tích hợp MQTT Trigger.
* **Điểm nghẽn 3 (Fusion Engine phụ thuộc cả hai):** Fusion Engine chỉ kiểm thử thực sự được khi cả luồng Vision và luồng Cân cùng đẩy dữ liệu về. *Giải pháp:* Viết bộ giả lập dữ liệu (Mock Sensor Generator) bằng script Python để test thuật toán Fusion độc lập ngay từ Tháng 2.

---

# PHẦN 2: KẾ HOẠCH TỔNG THỂ 6–8 THÁNG

```mermaid
gantt
    title LỘ TRÌNH TỔNG THỂ 8 THÁNG (6 THÁNG KHÓA LUẬN + 2 THÁNG NÂNG CAO)
    dateFormat  YYYY-MM-DD
    section Giai đoạn Khóa Luận (Tháng 1-6)
    T1: Khảo sát & Baseline Prototype      :active, t1, 2026-10-01, 30d
    T2: Computer Vision & Tín Hiệu Cân     :t2, after t1, 30d
    T3: Disagreement Fusion & Verification :t3, after t2, 30d
    T4: Backend, MQTT & Web Dashboard       :t4, after t3, 30d
    T5: Thực Nghiệm Đánh Giá EXP 1-6       :t5, after t4, 30d
    T6: Hoàn Thiện Luận Văn & Bảo Vệ       :t6, after t5, 30d
    section Giai đoạn Nâng Cao (Tháng 7-8)
    T7: Edge Optimization (RPi5/Jetson)    :t7, after t6, 30d
    T8: Viết & Nộp Bài Báo Khoa Học       :t8, after t7, 30d
```

### Bảng Tổng Quan 8 Tháng

| Tháng | Giai đoạn | Kiến thức trọng tâm | Module dự án | Deliverables (Sản phẩm đầu ra) | Milestone (Cột mốc nghiệm thu) |
|:---:|---|---|---|---|---|
| **Tháng 1** | **Khởi Động & Baseline** | Python data, C/C++ ESP32 cơ bản, Git, Tổng quan CV & Load cell | Khảo sát tài liệu, Dựng khung kệ mẫu, Đo đạc sơ bộ cảm biến & camera | - Báo cáo khảo sát tài liệu (Literature Review)<br>- Khung kệ demo 1–2 ô<br>- Script đọc webcam & script ESP32 đọc HX711 | **MS-1:** Đọc được raw weight từ ESP32 qua Serial và bắt được ảnh webcam trên PC. |
| **Tháng 2** | **Vision & Signal Processing** | PyTorch, Ultralytics YOLOv8/v11, OpenCV, Digital Filter (DSP), Cân chỉnh sensor | Module CV nhận diện sản phẩm & Module Firmware lọc ổn định trọng lượng | - Custom dataset 5–10 SKU (Roboflow/YOLO format)<br>- Checkpoint YOLOv8n fine-tuned (mAP@50 > 85%)<br>- Firmware ESP32 lọc ổn định $\Delta W < 2g$ | **MS-2:** Vision đếm đúng sản phẩm độc lập; Cân xuất đúng khối lượng ổn định khi đặt/nhấc vật. |
| **Tháng 3** | **Sensor Fusion & Disagreement** | Xác suất thống kê, Bayesian Fusion, Logic Two-tier inference, Thuật toán quyết định | Module Disagreement Analyzer & Verification Logic Engine | - Module `fusion_engine.py` hoàn chỉnh<br>- Bộ giả lập kiểm thử (Mock Test Suite) mô phỏng bất đồng<br>- Thuật toán chuyển Tier 1 $\leftrightarrow$ Tier 2 | **MS-3:** Thuật toán tự động kích hoạt Verification khi cố tình tạo che khuất (Vision sai nhưng Cân đúng). |
| **Tháng 4** | **Tích Hợp Hệ Thống & Dashboard** | Giao thức MQTT (paho-mqtt), FastAPI backend, WebSocket/Streamlit, SQLite/JSON logging | Tích hợp ESP32 $\to$ MQTT $\to$ Fusion $\to$ Web Dashboard | - MQTT Broker nội bộ hoạt động trơn tru<br>- Web Dashboard trực quan hiển thị tồn kho thời gian thực<br>- Hệ thống cảnh báo Low-stock, Stock-out, Disagreement | **MS-4:** Đặt/nhấc sản phẩm thực tế trên kệ $\to$ Dashboard cập nhật trong $< 1.5$ giây kèm trạng thái chính xác. |
| **Tháng 5** | **Thực Nghiệm Toàn Diện (EXP 1–6)** | Thiết kế thực nghiệm (DOE), Thống kê khoa học, Visualization (Matplotlib/Seaborn) | Chạy 6 kịch bản thực nghiệm trên kệ vật lý, thu thập dữ liệu & vẽ biểu đồ | - Bộ Raw Data log của 6 bài thí nghiệm (EXP-1 đến EXP-6)<br>- Toàn bộ biểu đồ phân tích (MAE, Latency, Occlusion curve)<br>- Bảng số liệu đối sánh baseline | **MS-5:** Hoàn thành đầy đủ số liệu thực nghiệm chứng minh tính ưu việt của phương pháp đề xuất. |
| **Tháng 6** | **Hoàn Thiện Luận Văn & Bảo Vệ** | Kỹ năng viết báo cáo học thuật, Trình bày slide, Kỹ thuật demo sản phẩm trực tiếp | Soạn thảo toàn văn luận văn tốt nghiệp, Chuẩn bị slide & Quay video backup | - Toàn văn Khóa luận Tốt nghiệp (Đủ 5 chương tiêu chuẩn)<br>- Slide thuyết trình bảo vệ chuyên nghiệp<br>- Kịch bản Demo thực tế 100% ổn định | **MS-6: BẢO VỆ THÀNH CÔNG KHÓA LUẬN TỐT NGHIỆP (Mục tiêu Xuất sắc/Thủ khoa).** |
| **Tháng 7** | **Edge Optimization (Tùy chọn)** | ONNX Runtime, TensorRT/NCNN, Linux nhúng (Raspberry Pi 5 / Jetson Orin Nano) | Đóng gói và tối ưu suy luận mô hình trên phần cứng nhúng biên | - Image OS và code chạy độc lập trên RPi5 / Jetson<br>- Báo cáo benchmark FPS, Công suất (Watts), RAM/CPU usage | **MS-7:** Hệ thống chạy standalone trên Edge Board không cần cắm Laptop phát triển. |
| **Tháng 8** | **Chuẩn Bị Bài Báo Khoa Học** | Cấu trúc bài báo IEEE/Springer, Academic English, LaTeX/Overleaf, Phân tích chuyên sâu | Chuyển đổi nội dung luận văn thành Manuscript hội nghị quốc tế (Scopus/IEEE) | - Bản thảo bài báo hoàn chỉnh (6–8 trang template IEEE)<br>- Minh chứng mã nguồn & dataset mở trên GitHub/Zenodo | **MS-8:** Nộp bài báo (Paper Submission) tới Hội nghị Quốc tế chuyên ngành (ATC, RIVF, ICAIT, v.v.). |

---

# PHẦN 3: KẾ HOẠCH CHI TIẾT TỪNG THÁNG

---

## THÁNG 1: KHỞI ĐỘNG, KHẢO SÁT TÀI LIỆU & XÂY DỰNG BASELINE

### A. Mục Tiêu Tháng
Xây dựng nền móng lý thuyết vững chắc về Retail Vision và Load Cell; thiết lập môi trường phát triển; đóng khung kệ gỗ/nhựa thử nghiệm 1 tầng có 2 ô cân; kết nối đọc dữ liệu thô độc lập từ camera và cảm biến lên máy tính.

### B. Kiến Thức Cần Học
1. **Bắt buộc:**
   * *Python nâng cao & Môi trường:* Quản lý `venv`, thư viện NumPy, OpenCV cơ bản (đọc camera, vẽ bounding box, lưu frame).
   * *C/C++ cho Vi điều khiển:* Cấu trúc Arduino IDE / ESP-IDF cơ bản, đọc tín hiệu từ thư viện `HX711.h`, cấu hình baudrate Serial 115200.
   * *Cơ cấu phần cứng:* Nguyên lý mạch cầu Wheatstone trong Load Cell, cơ chế khuếch đại tín hiệu vi sai của chip ADC 24-bit HX711.
   * *Phương pháp nghiên cứu:* Cách tìm kiếm, đọc và tóm tắt paper trên Google Scholar, IEEE Xplore, ScienceDirect; sử dụng Zotero để trích dẫn.
2. **Nên học thêm:**
   * Git & GitHub cơ bản để quản lý mã nguồn dự án ngay từ ngày đầu.
   * Cấu hình căn bản giao thức MQTT (lý thuyết Publisher/Subscriber).
3. **Chưa cần học:**
   * Tối ưu hóa TensorRT, viết Dockerfile, thiết kế Dashboard phức tạp.

### C. Kế Hoạch Theo Tuần (Tháng 1)

| Tuần | Nội dung học | Thực hành dự án | Giờ học | Giờ làm | Deliverables | Tiêu chí hoàn thành |
|:---:|---|---|:---:|:---:|---|---|
| **Tuần 1** | - Đọc 5 paper về Smart Shelf & Retail Object Detection.<br>- Ôn tập Python & OpenCV xử lý luồng video. | - Cài đặt môi trường Python 3.10+, PyTorch (GPU/CPU).<br>- Viết script Python mở webcam, chụp frame và ghi video. | 8h | 6h | File `review_notes.md` tổng hợp 5 paper; script `capture_cam.py`. | Mở camera mượt mà, ghi hình 30 FPS không drop frame, trích dẫn rõ 5 paper liên quan. |
| **Tuần 2** | - Cấu tạo Load Cell & nguyên lý mạch cầu Wheatstone.<br>- Lập trình C/C++ giao tiếp ESP32 với HX711 qua chân DOUT/SCK. | - Hàn chân linh kiện HX711, nối dây vào Load Cell 5kg/10kg và ESP32.<br>- Nạp code mẫu đọc giá trị ADC thô (Raw Digital Units). | 7h | 7h | Mô hình mạch thử nghiệm trên Breadboard; code `test_hx711.ino`. | Đọc được giá trị số nguyên thô thay đổi tương ứng khi tác dụng lực lên Load Cell. |
| **Tuần 3** | - Kỹ thuật hiệu chuẩn cảm biến tuyến tính: $Weight = (Raw - Offset) / Scale\_Factor$. | - Viết code hiệu chuẩn cân dùng quả cân chuẩn (hoặc chai nước 500g đã biết rõ khối lượng).<br>- Xuất giá trị khối lượng tính bằng gram qua Serial Monitor. | 6h | 8h | Script hiệu chuẩn `calibrate_loadcell.ino`; Bảng ghi nhận sai số tĩnh. | Cân vật mẫu 500g hiển thị sai số $\le \pm 2g$ trong 10 lần thử lặp lại. |
| **Tuần 4** | - Kiến trúc mô hình YOLOv8-Nano / YOLO11-Nano.<br>- Pipeline chạy inference mô hình pre-trained COCO. | - Lắp ráp khung kệ gỗ/nhựa sơ bộ (1 tầng, gắn camera góc xiên 45 độ hướng xuống).<br>- Chạy thử YOLO pre-trained trên webcam phát hiện chai/lọ có sẵn. | 7h | 8h | Khung kệ mô hình cơ học v1; script `baseline_vision.py`. | Khung kệ vững chắc; mô hình nhận diện được chai nước mẫu trên kệ với bounding box. |

### D. Milestone Cuối Tháng (MS-1)
* **Module hoàn thành:** Khung kệ thô 1 tầng; mạch đọc cảm biến Load Cell cân chính xác $\pm 2g$; script đọc hình ảnh từ webcam.
* **Báo cáo GVHD:** Bản đề cương chi tiết (Thesis Proposal), tổng quan tài liệu (Literature Review), demo đọc camera và cân mẫu tại chỗ.
* **Điều kiện chuyển tháng 2:** Phần cứng cân đo ổn định; camera thu hình không giật lag.

### E. Kiểm Tra & Đánh Giá
* [ ] Đã có bảng tóm tắt so sánh ít nhất 8 bài báo khoa học liên quan trực tiếp.
* [ ] ESP32 đọc giá trị gram ổn định, không bị hiện tượng nhảy số ngẫu nhiên quá $\pm 5g$ ở trạng thái tĩnh.
* [ ] Webcam cho hình ảnh rõ nét, bao quát toàn bộ mặt sàn ô cân.
* *Xử lý chậm tiến độ:* Nếu Load Cell nhảy số bất thường, kiểm tra ngay dây mass (GND), chống rung cơ học hoặc thay module HX711 khác (rất hay gặp IC lỗi).

---

## THÁNG 2: PHÁT TRIỂN COMPUTER VISION & XỬ LÝ TÍN HIỆU CÂN

### A. Mục Tiêu Tháng
Huấn luyện mô hình YOLOv8n/11n đạt độ chính xác cao trên bộ dữ liệu 5–10 loại sản phẩm bán lẻ; phát triển thuật toán lọc số (Digital Filtering) và State Machine trên ESP32 để xuất tín hiệu cân ổn định và tin cậy.

### B. Kiến Thức Cần Học
1. **Bắt buộc:**
   * *Computer Vision & Object Detection:* Quy trình gán nhãn (Roboflow / LabelImg), định dạng nhãn YOLO (class, x_center, y_center, width, height), kỹ thuật Data Augmentation.
   * *Huấn luyện Deep Learning:* Cấu hình Hyperparameters (Epochs, Batch size, LR, Optimizer), đánh giá độ đo Precision, Recall, mAP@50, mAP@50-95.
   * *Xử lý tín hiệu số (DSP):* Bộ lọc trung bình trượt (Moving Average Filter), bộ lọc trung vị (Median Filter) để loại bỏ nhiễu xung (Impulse Noise).
   * *Lý thuyết trạng thái (Finite State Machine - FSM):* Thiết kế máy trạng thái để phát hiện khi nào cân đang biến động và khi nào đã ổn định hoàn toàn ($T_{settle}$).
2. **Nên học thêm:**
   * Làm quen với bộ dữ liệu bán lẻ nổi tiếng SKU-110K để học cách thức các hệ thống lớn xử lý mật độ sản phẩm dày đặc.
3. **Chưa cần học:**
   * Lý thuyết Bayesian Fusion chi tiết (để sang Tháng 3).

### C. Kế Hoạch Theo Tuần (Tháng 2)

| Tuần | Nội dung học | Thực hành dự án | Giờ học | Giờ làm | Deliverables | Tiêu chí hoàn thành |
|:---:|---|---|:---:|:---:|---|---|
| **Tuần 5** | - Tiêu chuẩn tạo Custom Dataset cho Object Detection.<br>- Chọn 5–8 mặt hàng bán lẻ phổ biến (hộp sữa, lon nước ngọt, chai nước, gói bánh...). | - Chụp 800–1200 ảnh các sản phẩm trên kệ dưới nhiều góc, vị trí, ánh sáng khác nhau.<br>- Gán nhãn bounding box trên Roboflow; chia tập Train/Val/Test (70/20/10). | 6h | 10h | Bộ dataset Retail-Shelf-v1 hoàn chỉnh trên Roboflow/Google Drive. | Đầy đủ nhãn cho 5–8 lớp; hình ảnh đa dạng góc độ và mức che khuất nhẹ. |
| **Tuần 6** | - Quá trình huấn luyện Transfer Learning với YOLOv8n / YOLO11n.<br>- Ý nghĩa các hàm mất mát (Box loss, Cls loss, DFL loss). | - Huấn luyện mô hình trên Google Colab / GPU nội bộ.<br>- Phân tích biểu đồ huấn luyện và Confusion Matrix để tìm lớp hay bị nhầm lẫn. | 7h | 9h | File weights `best.pt`; Báo cáo kết quả huấn luyện (Validation Metrics). | Đạt **mAP@50 $\ge 88\%$** trên tập Validation độc lập. |
| **Tuần 7** | - Kỹ thuật đếm số lượng sản phẩm từ bounding box.<br>- Trích xuất vector độ tin cậy ($C_v$) theo từng sản phẩm.<br>- Lọc nhiễu cảm biến: Moving Average kết hợp Median Filter. | - Viết script Python `vision_counter.py` đếm số lượng từng loại sản phẩm.<br>- Viết code C++ bộ lọc trượt 10 mẫu trên ESP32 để làm mượt dữ liệu HX711. | 7h | 9h | Script đếm thị giác có confidence; Code ESP32 `filter_hx711.ino`. | Vision đếm chính xác khi không che khuất; Dữ liệu cân mượt mà, sai số tĩnh $\le \pm 1g$. |
| **Tuần 8** | - Thiết kế Finite State Machine (FSM) xác định trạng thái cân: `IDLE`, `WEIGHT_CHANGING`, `SETTLED`. | - Cài đặt FSM lên ESP32: Khi người dùng nhấc/đặt đồ, cân vào trạng thái chuyển tiếp; sau 500ms không đổi khối lượng $\to$ phát cờ `SETTLED`. | 6h | 10h | Firmware ESP32 `stable_weight_fsm.ino` hoàn chỉnh; Log Serial kiểm tra FSM. | ESP32 phát hiện chính xác thời điểm tải trọng ổn định sau mỗi thao tác lấy/đặt hàng. |

### D. Milestone Cuối Tháng (MS-2)
* **Module hoàn thành:** Module Vision nhận diện chính xác 5–8 sản phẩm đạt mAP cao; Module Cân có khả năng lọc nhiễu và phát cờ trạng thái ổn định (Settled Flag).
* **Báo cáo GVHD:** Trình chiếu demo mô hình YOLO detect sản phẩm thời gian thực; trình bày biểu đồ khử nhiễu của cảm biến trọng lượng trước và sau lọc.
* **Điều kiện chuyển tháng 3:** Sai số cân không vượt quá $\pm 1.5g$; Vision có khả năng đếm chính xác trong điều kiện bình thường.

### E. Kiểm Tra & Đánh Giá
* [ ] YOLOv8n/11n đạt mAP@50 > 85%, thời gian suy luận trên PC < 25ms/frame.
* [ ] Cảm biến cân không bao giờ bị đứng (hang) khi truyền dữ liệu liên tục 2 giờ.
* [ ] Bộ phát hiện ổn định (Settling Detector) không phát cờ `SETTLED` giả khi có rung động cơ học nhẹ ở sàn kệ.

---

## THÁNG 3: PHÁT TRIỂN THUẬT TOÁN SENSOR FUSION & PHÁT HIỆN BẤT ĐỒNG

### A. Mục Tiêu Tháng
Xây dựng trái tim nghiên cứu của khóa luận: Thuật toán **Disagreement-Aware Fusion** kết hợp giữa ước lượng thị giác ($\hat{N}_v$) và ước lượng tải trọng ($\hat{N}_w$); cài đặt cơ chế chuyển đổi hai tầng suy luận thích ứng (Two-Tier Adaptive Inference) khi phát hiện bất đồng.

### B. Kiến Thức Cần Học
1. **Bắt buộc:**
   * *Lý thuyết Sensor Fusion cơ bản:* Ưu/nhược điểm của Early Fusion, Late Fusion, Decision-level Fusion trong hệ thống đa cảm biến.
   * *Toán học ước lượng trọng lượng:* $\hat{N}_w = \text{round}(W_{total} / w_{unit})$, tính toán độ lệch chuẩn $\sigma_w$ và khoảng dung sai tải trọng cho phép.
   * *Thuật toán phát hiện bất đồng (Disagreement Detection):* Đo lường $D_t = |\hat{N}_v - \hat{N}_w|$ kết hợp ngưỡng tin cậy thị giác $\tau_C$ để phân loại trạng thái: `AGREEMENT`, `OCCLUSION_SUSPECTED`, `WEIGHT_ANOMALY`.
   * *Chiến lược Two-Tier Inference:*
     * *Tier 1 (Normal):* Ảnh độ phân giải thấp (ví dụ: $416\times416$ hoặc $640\times640$), ngưỡng confidence chuẩn (0.5).
     * *Tier 2 (Verification):* Tăng độ phân giải ảnh, hạ ngưỡng confidence để bắt vật che khuất, tích lũy trung bình qua $K$ frame liên tiếp (Temporal Verification).
2. **Nên học thêm:**
   * Cơ bản về Bayesian Inference: Cập nhật hàm mật độ xác suất tiền nghiệm $P(N)$ dựa trên hàm hợp lý từ cân và camera.
3. **Chưa cần học:**
   * Lập trình Web Frontend UI nâng cao.

### C. Kế Hoạch Theo Tuần (Tháng 3)

| Tuần | Nội dung học | Thực hành dự án | Giờ học | Giờ làm | Deliverables | Tiêu chí hoàn thành |
|:---:|---|---|:---:|:---:|---|---|
| **Tuần 9** | - Mô hình hóa khối lượng sản phẩm: Đo trọng lượng 30 mẫu của từng SKU để tìm giá trị trung bình $w_{item}$ và độ lệch $\sigma$. | - Viết module `weight_estimator.py`: Nhận khối lượng $W$, quy đổi ra số lượng $\hat{N}_w$ kèm mức độ bất định (uncertainty score). | 6h | 8h | Module `weight_estimator.py`; Bảng cơ sở dữ liệu đặc tính khối lượng SKU. | Ước lượng đúng số lượng sản phẩm dựa trên cân nặng với độ tin cậy được định lượng. |
| **Tuần 10** | - Xây dựng công thức đo lường bất đồng $D_t$.<br>- Phân tích các ma trận quyết định (Decision Logic Matrix) trong dung hòa dữ liệu. | - Viết core class `DisagreementAnalyzer`: So sánh $\hat{N}_v$ và $\hat{N}_w$.<br>- Viết script giả lập các trường hợp: (1) Trùng khớp, (2) Che khuất 1 phần, (3) Che khuất nặng. | 7h | 9h | File `disagreement_engine.py`; Bộ unit tests mô phỏng các ca bất đồng. | Thuật toán gắn nhãn chính xác trạng thái `AGREEMENT` hoặc `DISAGREEMENT` trong 100% test cases giả lập. |
| **Tuần 11** | - Cơ chế Two-Tier Adaptive Inference:<br>Thiết kế luồng chuyển đổi linh hoạt giữa chế độ giám sát nhẹ và chế độ xác minh sâu. | - Cài đặt chế độ Tier 2: Khi có bất đồng, tự động gọi hàm xử lý phân giải cao, quét lại 5 frame liên tiếp và phân tích temporal voting. | 7h | 9h | Code hoàn thiện `adaptive_inference.py`. | Khi cố ý lấy tay che 1 sản phẩm $\to$ Hệ thống phát hiện bất đồng và kích hoạt Tier 2 tìm lại được sản phẩm bị che. |
| **Tuần 12** | - Cơ chế Event-Triggered: Phối hợp kích hoạt giữa cờ `SETTLED` từ cân và module suy luận thị giác. | - Tích hợp toàn bộ logic vào pipeline Python duy nhất trên PC.<br>- Kiểm thử luồng: Đặt sản phẩm $\to$ Cân báo ổn định $\to$ Kích hoạt Vision $\to$ Tính Fusion. | 6h | 10h | File chạy tổng thể `main_fusion_pipeline.py`. | Camera không tốn năng lượng suy luận khi kệ đứng yên; chỉ chạy inference khi có biến động tải trọng. |

### D. Milestone Cuối Tháng (MS-3)
* **Module hoàn thành:** Module Fusion Engine hoàn chỉnh; cơ chế Two-tier Adaptive Verification chạy chính xác trên PC phát triển.
* **Báo cáo GVHD:** Trình bày nguyên lý toán học của thuật toán Disagreement-Aware; thực nghiệm đối chứng trực tiếp tại bàn: Trường hợp che khuất khiến Vision-only đếm sai nhưng Proposed Fusion sửa sai thành công.
* **Điều kiện chuyển tháng 4:** Thuật toán hợp nhất hoạt động ổn định trên các kịch bản thử nghiệm cục bộ; có log ghi nhận rõ ràng quá trình kích hoạt Tier 2.

### E. Kiểm Tra & Đánh Giá
* [ ] Khi che khuất 1 sản phẩm (trên tổng số 5 sản phẩm): Vision-only báo 4, Proposed Fusion nhận diện bất đồng và hiệu chỉnh về 5.
* [ ] Thời gian xử lý của Fusion Engine (không tính inference) cực nhanh: $< 5\text{ms}$.
* [ ] Không xảy ra xung đột bộ nhớ hoặc race condition khi nhận tín hiệu đồng thời.

---

## THÁNG 4: TÍCH HỢP HỆ THỐNG, MQTT & XÂY DỰNG WEB DASHBOARD

### A. Mục Tiêu Tháng
Chuyển đổi giao tiếp Serial thô sang giao thức chuẩn IoT (MQTT không dây qua Wi-Fi); xây dựng Web Dashboard trực quan giám sát tình trạng tồn kho, mức độ bất đồng và nhật ký hệ thống; hoàn thành sản phẩm MVP end-to-end.

### B. Kiến Thức Cần Học
1. **Bắt buộc:**
   * *IoT Communication:* Cài đặt và cấu hình Mosquitto MQTT Broker; thư viện `PubSubClient` trên ESP32; thư viện `paho-mqtt` trên Python.
   * *Định dạng dữ liệu IoT:* Thiết kế cấu trúc gói tin JSON tối ưu (gồm `shelf_id`, `zone_id`, `weight`, `status`, `timestamp`).
   * *Web Backend & Frontend cơ bản:* Sử dụng Streamlit (nhanh, tối ưu cho data dashboard) HOẶC FastAPI + HTML5/CSS/JavaScript với WebSocket để đẩy dữ liệu thời gian thực.
   * *Cơ sở dữ liệu lưu trữ:* SQLite lưu lịch sử biến động số lượng hàng và lịch sử bất đồng phục vụ báo cáo thống kê.
2. **Nên học thêm:**
   * Thiết kế giao diện theo nguyên lý Dashboard UI (KPI cards, bar charts trực quan, bảng thông báo cảnh báo màu đỏ/vàng/xanh).
3. **Chưa cần học:**
   * Triển khai Cloud Docker, Kubernetes hay server production thương mại.

### C. Kế Hoạch Theo Tuần (Tháng 4)

| Tuần | Nội dung học | Thực hành dự án | Giờ học | Giờ làm | Deliverables | Tiêu chí hoàn thành |
|:---:|---|---|:---:|:---:|---|---|
| **Tuần 13** | - Giao thức MQTT: Topics design, QoS 0/1, Retain messages, Keep-alive.<br>- Đóng gói & parse JSON trên ESP32 bằng thư viện `ArduinoJson`. | - Cài Mosquitto MQTT broker trên máy tính cục bộ.<br>- Viết code ESP32 kết nối Wi-Fi, publish dữ liệu trọng lượng lên topic `smartshelf/sensor/zone1`. | 6h | 9h | Broker Mosquitto running; Firmware ESP32 `mqtt_shelf_node.ino`. | ESP32 tự kết nối lại khi mất Wi-Fi; dữ liệu MQTT được publish ổn định 10Hz khi biến động. |
| **Tuần 14** | - Lập trình luồng nhận MQTT trên Python (`paho-mqtt`) và định tuyến vào Fusion Engine.<br>- Thiết kế Database Schema lưu lịch sử tồn kho (SQLite). | - Viết backend service: Lắng nghe MQTT, khi có payload `SETTLED` $\to$ kích hoạt camera $\to$ chạy Fusion $\to$ ghi kết quả vào SQLite. | 6h | 10h | Script backend `shelf_backend_service.py`; Database `inventory.db`. | Luồng tự động: Đặt sản phẩm lên kệ $\to$ ESP32 bắn MQTT $\to$ Backend tự động suy luận và ghi log database. |
| **Tuần 15** | - Công nghệ xây dựng Dashboard (Streamlit hoặc FastAPI + HTML/Tailwind/Charts).<br>- Thiết kế các component: Thẻ số lượng tồn, Biểu đồ lịch sử, Cảnh báo. | - Xây dựng giao diện Web Dashboard hiển thị số lượng tồn từng mặt hàng, trạng thái cảm biến (Online/Offline), độ bất đồng gần nhất. | 7h | 10h | Mã nguồn Web Dashboard `app.py`. | Dashboard hiển thị giao diện đẹp, trực quan, cập nhật số liệu ngay khi backend có kết quả mới. |
| **Tuần 16** | - Cơ chế cảnh báo tự động: Low Stock (Sắp hết hàng), Stock Out (Hết hàng), Sensor Disagreement (Cảnh báo gian lận hoặc lỗi cảm biến). | - Cài đặt logic cảnh báo pop-up trên Dashboard và gửi thông báo còi/đèn (hoặc Telegram bot notification demo).<br>- Tinh chỉnh độ trễ toàn trình. | 6h | 10h | Hệ thống hoàn chỉnh end-to-end v1.0; Video quay lại quy trình hoạt động. | Toàn bộ hệ thống chạy thông suốt từ kệ đến màn hình điều khiển không cần can thiệp thủ công. |

### D. Milestone Cuối Tháng (MS-4)
* **Module hoàn thành:** Toàn bộ hệ thống phần cứng, firmware, thuật toán AI và giao diện Web Dashboard được tích hợp thành công (End-to-End System Integration).
* **Báo cáo GVHD:** Trình diễn trực tiếp với GVHD: Thực hiện thao tác nhấc 2 món đồ trên kệ $\to$ Màn hình dashboard lập tức trừ tồn kho, hiển thị biểu đồ và log thời gian thực.
* **Điều kiện chuyển tháng 5:** Độ trễ từ lúc đặt vật ổn định đến khi Dashboard cập nhật $< 1.5$ giây; không có hiện tượng treo broker hoặc crash backend.

### E. Kiểm Tra & Đánh Giá
* [ ] Mất kết nối Wi-Fi đột ngột: ESP32 tự động reconnect khi có mạng lại mà không cần bấm nút reset.
* [ ] Dashboard hiển thị trực quan 3 trạng thái: Bình thường (Xanh), Có bất đồng đã xác minh (Vàng), Báo động lỗi cảm biến (Đỏ).
* [ ] Dữ liệu lưu trong SQLite đầy đủ các trường: `timestamp`, `sku`, `vision_count`, `weight_count`, `final_count`, `mode`.

---

## THÁNG 5: THỰC HIỆN CÁC THÍ NGHIỆM & ĐÁNH GIÁ KẾT QUẢ

### A. Mục Tiêu Tháng
Thu thập số liệu thực nghiệm khoa học dựa trên 6 kịch bản thực nghiệm chuẩn mực đã xác định trong PDF (EXP-1 đến EXP-6); phân tích định lượng các chỉ số MAE, Counting Accuracy, Latency, Energy/Computational Savings; vẽ toàn bộ đồ thị học thuật.

### B. Kiến Thức Cần Học
1. **Bắt buộc:**
   * *Kỹ thuật thiết kế thực nghiệm (Design of Experiments - DOE):* Đảm bảo tính lặp lại (reproducibility), phân chia các tổ hợp test có kiểm soát (control variables vs. independent variables).
   * *Chỉ số đánh giá độ chính xác kiểm kê:* Mean Absolute Error (MAE), Exact Count Accuracy (ECA), False Alarm Rate (FAR).
   * *Đo đạc hiệu năng máy tính (System Profiling):* Thư viện `psutil`, `time.perf_counter()`, công cụ đo công suất tiêu thụ (nếu có USB Power Meter) hoặc đo số frame suy luận được tiết kiệm.
   * *Biểu diễn trực quan khoa học (Data Visualization):* Thư viện `Matplotlib`, `Seaborn` để vẽ đồ thị chuẩn phong cách IEEE (bar chart, line plot, confusion matrix, ablation curve).
2. **Nên học thêm:**
   * Kiểm định thống kê (T-test, Wilcoxon signed-rank test) để chứng minh sự cải thiện của Proposed Fusion là có ý nghĩa thống kê ($p < 0.05$).
3. **Chưa cần học:**
   * Viết văn phong xuất bản quốc tế (để tháng 8).

### C. Kế Hoạch Theo Tuần (Tháng 5)

| Tuần | Nội dung học | Thực hành dự án | Giờ học | Giờ làm | Deliverables | Tiêu chí hoàn thành |
|:---:|---|---|:---:|:---:|---|---|
| **Tuần 17** | - Phương pháp luận đánh giá Baseline: Thiết lập môi trường đo đạc đồng bộ cho Vision-Only, Weight-Only, Fixed Fusion, Proposed Fusion. | - Tiến hành **EXP-1 (Baseline Comparison)**: Thực hiện 100 lượt thử nghiệm ngẫu nhiên với 5 cấu hình hàng hóa khác nhau.<br>- Thu thập log và tính toán MAE. | 5h | 12h | File dữ liệu `exp1_baseline.csv`; Bảng số liệu tổng hợp đối sánh. | Đủ 100 mẫu thử có ground truth; Proposed Fusion cho MAE thấp hơn rõ rệt so với Vision-only. |
| **Tuần 18** | - Phương pháp tạo che khuất nhân tạo và đo lường định lượng mức độ che khuất (0%, 20–40%, >50%).<br>- Các mức chiếu sáng thử nghiệm. | - Thực hiện **EXP-2 (Occlusion Robustness)** và **EXP-3 (Lighting Robustness)**.<br>- Thử nghiệm với các điều kiện: Sáng chuẩn, Ánh sáng yếu, Chiếu sáng một phía. | 5h | 12h | File `exp2_occlusion.csv`, `exp3_lighting.csv`; Đồ thị suy biến hiệu năng theo độ che khuất. | Dữ liệu chứng minh: Khi che khuất $>40\%$, Vision-only đếm sai trầm trọng trong khi Fusion vẫn duy trì độ chính xác cao. |
| **Tuần 19** | - Phương pháp làm Ablation Study chuẩn mực khoa học: Bóc tách từng thành phần để chứng minh giá trị.<br>- Khảo sát ngưỡng bất đồng $\tau_D$. | - Tiến hành **EXP-4 (Ablation Study)**: Đo lần lượt Camera only $\to$ Cam+Weight $\to$ Cam+Weight+Confidence $\to$ Proposed.<br>- Chạy thử nghiệm quét ngưỡng $\tau_D \in \{0, 1, 2, 3\}$. | 5h | 12h | Bảng Ablation Study hoàn chỉnh; Biểu đồ Trade-off giữa $\tau_D$, độ trễ và độ chính xác. | Chỉ ra được giá trị tối ưu của $\tau_D$ cân bằng giữa số lần trigger Tier 2 và độ chính xác. |
| **Tuần 20** | - Đo đạc độ trễ toàn trình từng chặng ($T_{sensor}, T_{comm}, T_{vision}, T_{fusion}, T_{dash}$) và hiệu quả năng lượng (Event-triggered). | - Thực hiện **EXP-5 (Energy & Inference Savings)** và **EXP-6 (End-to-End Latency Breakdown)**.<br>- Đo số frame/giờ giữa Continuous vs. Event-Triggered. | 5h | 12h | File số liệu `exp5_energy.csv`, `exp6_latency.csv`; Đồ thị phân rã độ trễ (Stacked Bar Chart). | Chứng minh giảm được $>70\%$ số frame cần suy luận; tổng độ trễ toàn trình đo đạc thực tế $< 1200\text{ms}$. |

### D. Milestone Cuối Tháng (MS-5)
* **Module hoàn thành:** Toàn bộ 6 bài thực nghiệm được thực thi nghiêm túc; bộ số liệu thô và đồ thị phân tích đã sẵn sàng đưa vào chương kết quả của luận văn.
* **Báo cáo GVHD:** Báo cáo chuyên đề kết quả thực nghiệm với đầy đủ biểu đồ, bảng so sánh và phân tích nguyên nhân các trường hợp sai lệch (Failure Case Analysis).
* **Điều kiện chuyển tháng 6:** Dữ liệu thực nghiệm logic, trung thực, chứng minh được tính hiệu quả của các đóng góp nghiên cứu trong luận văn.

### E. Kiểm Tra & Đánh Giá
* [ ] Đã có tối thiểu 50–100 samples cho mỗi kịch bản thực nghiệm để đảm bảo độ tin cậy thống kê.
* [ ] Các biểu đồ được xuất ra định dạng vector (PDF/EPS) hoặc PNG độ phân giải cao 300 DPI.
* [ ] Có phần phân tích rõ ràng: Trong tình huống nào thì Proposed Fusion thất bại (đây là điểm cộng rất lớn trong hội đồng bảo vệ).

---

## THÁNG 6: HOÀN THIỆN KHÓA LUẬN & CHUẨN BỊ BẢO VỆ

### A. Mục Tiêu Tháng
Soạn thảo hoàn chỉnh quyển Khóa luận Tốt nghiệp theo đúng định dạng học thuật của nhà trường; thiết kế slide thuyết trình ấn tượng; luyện tập thuyết trình và demo thực tế; bảo vệ thành công trước hội đồng chấm khóa luận.

### B. Kiến Thức Cần Học
1. **Bắt buộc:**
   * *Kỹ năng soạn thảo báo cáo kỹ thuật:* Cấu trúc chuẩn của khóa luận tốt nghiệp kỹ thuật (Chương 1: Mở đầu, Chương 2: Tổng quan & Cơ sở lý thuyết, Chương 3: Thiết kế hệ thống đề xuất, Chương 4: Triển khai & Thực nghiệm, Chương 5: Kết luận & Hướng phát triển).
   * *Kỹ năng trình bày (Presentation Skills):* Thiết kế slide thuyết trình súc tích, trực quan, tập trung vào đóng góp mới (Contributions), phương pháp giải quyết và kết quả định lượng.
   * *Kỹ năng ứng biến phản biện:* Chuẩn bị trước danh sách 20 câu hỏi tiềm năng từ các thầy cô phản biện và câu trả lời chuẩn xác.
2. **Nên học thêm:**
   * Soạn thảo trên LaTeX (sử dụng template Overleaf của trường) hoặc làm chủ Microsoft Word nâng cao (Heading, Cross-reference, List of Figures/Tables).

### C. Kế Hoạch Theo Tuần (Tháng 6)

| Tuần | Nội dung học | Thực hành dự án | Giờ học | Giờ làm | Deliverables | Tiêu chí hoàn thành |
|:---:|---|---|:---:|:---:|---|---|
| **Tuần 21** | - Tiêu chuẩn định dạng luận văn tốt nghiệp (Font chữ, căn lề, trích dẫn chuẩn IEEE). | - Viết Chương 1 (Giới thiệu, Lý do chọn đề tài, Mục tiêu, Đóng góp) và Chương 2 (Cơ sở lý thuyết & Nghiên cứu liên quan). | 4h | 14h | Bản thảo Chương 1 & Chương 2 (Draft v1). | Rõ ràng về Problem Statement, Research Questions và trích dẫn đầy đủ các tài liệu liên quan. |
| **Tuần 22** | - Kỹ thuật mô tả kiến trúc hệ thống và giải thuật bằng sơ đồ khối, lưu đồ thuật toán và công thức toán học. | - Viết Chương 3 (Thiết kế hệ thống: Khung kệ, Mạch điện, Mô hình YOLO, Thuật toán Disagreement-Aware Fusion, Dashboard). | 4h | 15h | Bản thảo Chương 3 kèm các sơ đồ kiến trúc chất lượng cao. | Mô tả chi tiết từng công thức toán, thuật toán và sơ đồ mạch phần cứng rõ ràng. |
| **Tuần 23** | - Kỹ thuật trình bày số liệu thực nghiệm, phân tích biểu đồ và thảo luận học thuật. | - Viết Chương 4 (Thực nghiệm & Thảo luận kết quả của 6 bài test) và Chương 5 (Kết luận, Hạn chế, Hướng phát triển tương lai). | 4h | 15h | Toàn văn bản thảo luận văn hoàn chỉnh (Full Draft). | Số liệu khớp 100% với log thực nghiệm; thảo luận sâu sắc, khách quan. |
| **Tuần 24** | - Kỹ năng bảo vệ luận văn: Phân bổ thời lượng (15 phút trình bày, 10 phút trả lời phản biện, 5 phút demo). | - Gửi GVHD duyệt quyển, chỉnh sửa format.<br>- Thiết kế bộ Slide bảo vệ (20–25 slides).<br>- Quay sẵn 1 video backup demo chất lượng cao phòng sự cố mạng. | 3h | 15h | - Quyển khóa luận in ấn chính thức.<br>- Bộ Slide thuyết trình.<br>- Video demo backup. | Trình bày thử trước bạn bè/GVHD lưu loát đúng trong 15 phút; trả lời trôi chảy các câu hỏi chất vấn. |

### D. Milestone Cuối Tháng (MS-6)
* **KẾT QUẢ ĐẠT ĐƯỢC:** **BẢO VỆ XUẤT SẮC KHÓA LUẬN TỐT NGHIỆP ĐẠI HỌC.**
* **Bàn giao:** Toàn văn khóa luận có chữ ký GVHD; mã nguồn được đóng gói gọn gàng trên GitHub cá nhân; video demo và slide thuyết trình.

### E. Kiểm Tra & Đánh Giá
* [ ] Kiểm tra lỗi chính tả và trùng lặp (Turnitin đạo văn $< 15\%$).
* [ ] Bộ sản phẩm demo hoạt động trơn tru 3 lần liên tiếp trước ngày bảo vệ.
* [ ] Chuẩn bị sẵn phương án mạng dự phòng (sử dụng 4G cá nhân phát Wi-Fi cho ESP32 và Laptop để không phụ thuộc vào Wi-Fi hội trường).

---

## THÁNG 7: TỐI ƯU HÓA EDGE AI & MỞ RỘNG NGHIÊN CỨU (HẬU KHÓA LUẬN)

### A. Mục Tiêu Tháng
Chuyển đổi toàn bộ pipeline từ máy tính cá nhân sang phần cứng biên độc lập (Standalone Edge AI Device: Raspberry Pi 5 hoặc Jetson Orin Nano); lượng tử hóa mô hình (Quantization FP16/INT8) và tối ưu hóa runtime suy luận.

### B. Kiến Thức Cần Học
1. **Bắt buộc:**
   * *Hệ điều hành Linux nhúng:* Linux commands, quản lý tiến trình systemd, build dependencies từ mã nguồn.
   * *Framework tối ưu suy luận Edge AI:* Xuất mô hình sang định dạng ONNX, sử dụng ONNX Runtime với Execution Providers; hoặc TensorRT trên Jetson Orin Nano / NCNN / TFLite trên Raspberry Pi 5.
   * *Đo đạc chỉ số phần cứng biên:* Giám sát nhiệt độ (thermal throttling), FPS thực tế, công suất tiêu thụ (Watts) bằng `jtop` hoặc script đo dòng.
2. **Nên học thêm:**
   * Thuật toán Bayesian State Estimation mở rộng (Continuous Bayesian Filter) để làm mượt số lượng tồn kho theo chuỗi thời gian.

### C. Kế Hoạch Theo Tuần (Tháng 7)

| Tuần | Nội dung học | Thực hành dự án | Giờ học | Giờ làm | Deliverables | Tiêu chí hoàn thành |
|:---:|---|---|:---:|:---:|---|---|
| **Tuần 25** | - Cài đặt OS Linux (Ubuntu / Raspberry Pi OS) trên Edge Board.<br>- Cấu hình Python, OpenCV, MQTT broker độc lập trên board. | - Cài đặt RPi5 hoặc Jetson Orin Nano gắn trực tiếp camera CSI/USB.<br>- Chạy thử nghiệm script chụp ảnh và đọc MQTT từ ESP32 tại chỗ. | 6h | 8h | Edge board khởi động độc lập, nhận diện đầy đủ camera và cổng mạng. | Edge board đóng vai trò MQTT broker và server xử lý trung tâm thay thế laptop. |
| **Tuần 26** | - Xuất mô hình YOLO sang ONNX và cấu hình ONNX Runtime.<br>- Kỹ thuật lượng tử hóa FP16 / INT8. | - Convert checkpoint `best.pt` $\to$ `best.onnx`.<br>- Viết code inference sử dụng ONNX Runtime trên CPU RPi5 hoặc GPU TensorRT trên Jetson. | 7h | 9h | File mô hình tối ưu `model_quantized.onnx` / `engine.trt`. | Tốc độ suy luận tăng ít nhất 30–50% so với PyTorch thô; độ chính xác suy giảm $< 1\%$. |
| **Tuần 27** | - Đo đạc và lập bảng so sánh hiệu năng giữa Laptop vs. RPi5 vs. Jetson Orin Nano. | - Đo lường các chỉ số: Inference Latency, FPS, RAM usage, CPU/GPU Load, Nhiệt độ hoạt động khi chạy liên tục 1 giờ. | 5h | 9h | Bảng số liệu Benchmark Edge Hardware toàn diện. | Dữ liệu so sánh rõ ràng giữa các nền tảng tính toán phục vụ viết bài báo khoa học. |
| **Tuần 28** | - Nghiên cứu mở rộng: Thử nghiệm thuật toán Temporal Bayesian State Update (lọc số lượng theo lịch sử giao dịch). | - Bổ sung thuật toán Bayesian filter vào module fusion trên edge board.<br>- Đánh giá giảm tỉ lệ nhảy số ảo (flickering count) khi khách hàng đưa tay vào lấy hàng. | 6h | 9h | Code cập nhật `bayesian_fusion_edge.py`. | Tỉ lệ cảnh báo giả (False Alert/Hour) giảm đáng kể trong quá trình thao tác liên tục. |

### D. Milestone Cuối Tháng (MS-7)
* **Hoàn thành:** Hệ thống AIoT Smart Shelf hoạt động hoàn toàn độc lập dạng thiết bị nhúng (Standalone Product Node) không cần máy tính phát triển; có đầy đủ dữ liệu đo lường Edge AI.

---

## THÁNG 8: CHUẨN BỊ BÀI BÁO KHOA HỌC & CÔNG BỐ (PAPER PUBLICATION)

### A. Mục Tiêu Tháng
Chuyển hóa kết quả nghiên cứu và dữ liệu thực nghiệm vượt trội của đề tài thành một bản thảo bài báo khoa học chuẩn mực (Conference Manuscript theo template IEEE/Springer 6–8 trang); hoàn thiện mã nguồn và dữ liệu công khai trên GitHub/Zenodo; tiến hành nộp bài (submission).

### B. Kiến Thức Cần Học
1. **Bắt buộc:**
   * *Academic Writing in English:* Cấu trúc chuẩn mực của một bài báo kỹ thuật máy tính (Title, Abstract, Introduction, Related Work, System Architecture & Methodology, Experimental Results & Discussion, Conclusion).
   * *Sử dụng LaTeX nâng cao:* Template `IEEEtran`, chèn hình ảnh vector, viết công thức toán học chuyên nghiệp, quản lý file trích dẫn `references.bib`.
   * *Đạo đức nghiên cứu & Open Science:* Chuẩn bị Artifact Zenodo (gán mã DOI), viết file `README.md` hướng dẫn tái tạo kết quả (Reproducibility Guide).

### C. Kế Hoạch Theo Tuần (Tháng 8)

| Tuần | Nội dung học | Thực hành dự án | Giờ học | Giờ làm | Deliverables | Tiêu chí hoàn thành |
|:---:|---|---|:---:|:---:|---|---|
| **Tuần 29** | - Cấu trúc một bài báo hội nghị Scopus/IEEE tiêu chuẩn.<br>- Cách viết Abstract ấn tượng và Introduction làm nổi bật Research Gap. | - Khởi tạo project Overleaf với template IEEE.<br>- Viết Section I (Introduction) và Section II (Related Work). | 5h | 9h | Bản thảo Section I & II bằng tiếng Anh học thuật. | Định vị rõ ràng đóng góp của bài báo: *Disagreement-Aware Multimodal Fusion*. |
| **Tuần 30** | - Cách thức trình bày giải thuật toán học và hệ thống AIoT trong bài báo. | - Viết Section III (Proposed Methodology) và Section IV (Hardware-Software Implementation).<br>- Vẽ lại các hình ảnh kiến trúc theo phong cách IEEE chuyên nghiệp. | 5h | 10h | Bản thảo Section III & IV kèm các hình minh họa vector. | Công thức toán mạch lạc, lưu đồ thuật toán Two-Tier Inference rõ ràng. |
| **Tuần 31** | - Cách viết phần Thực nghiệm và Thảo luận chuyên sâu (Quantitative & Qualitative Analysis). | - Viết Section V (Experiments & Results) tổng hợp toàn bộ 6 bài thử nghiệm từ khóa luận cộng với kết quả Edge Benchmark của Tháng 7. | 5h | 10h | Bản thảo Section V hoàn chỉnh với các bảng số liệu so sánh đẹp mắt. | Thể hiện rõ tính vượt trội về độ chính xác và tiết kiệm năng lượng so với baseline. |
| **Tuần 32** | - Kỹ thuật rà soát bản thảo (Proofreading), kiểm tra ngữ pháp (Grammarly), kiểm tra trích dẫn. | - Viết Abstract và Conclusion.<br>- Gửi bản thảo hoàn chỉnh cho GVHD đọc và cho ý kiến phản hồi.<br>- Hoàn thiện GitHub Repo và đóng gói dữ liệu. | 4h | 10h | - Bản thảo Manuscript PDF hoàn chỉnh.<br>- GitHub Repository công khai kèm DOI Zenodo. | Bài báo sẵn sàng được nộp tới hội nghị mục tiêu (ví dụ: IEEE RIVF, ATC, ICARCV, v.v.). |

### D. Milestone Cuối Tháng (MS-8)
* **KẾT QUẢ ĐẠT ĐƯỢC:** Hoàn thành Manuscript bài báo khoa học chất lượng cao sẵn sàng nộp (Ready for Submission).

---

# PHẦN 4: PHÂN CHIA TỶ LỆ THỜI GIAN HỌC LÝ THUYẾT & THỰC HÀNH

Tỷ lệ thời gian giữa **Học kiến thức nền tảng** và **Thực hành triển khai dự án** cần biến thiên linh hoạt theo từng giai đoạn phát triển nhằm tối ưu hóa đường cong học tập và tránh bẫy "sa đà học lý thuyết suông" (Tutorial Hell):

```text
TỶ LỆ PHÂN BỔ THỜI GIAN THEO GIAI ĐOẠN

Tháng 1 (Khởi động)   : [==== Học: 50% ====] [==== Làm: 50% ====]
Tháng 2 (Module rời)   : [=== Học: 40% ===]   [===== Làm: 60% =====]
Tháng 3 (Fusion Core)  : [=== Học: 40% ===]   [===== Làm: 60% =====]
Tháng 4 (Tích hợp)     : [== Học: 30% ==]     [====== Làm: 70% ======]
Tháng 5 (Thực nghiệm)  : [= Học: 20% =]       [======= Làm: 80% =======]
Tháng 6 (Viết luận văn): [= Học: 20% =]       [======= Làm: 80% =======]
Tháng 7 (Tối ưu Edge)  : [=== Học: 35% ===]   [===== Làm: 65% =====]
Tháng 8 (Viết Paper)   : [== Học: 25% ==]     [====== Làm: 75% ======]
```

### Giải Thích Cơ Sở Phân Bổ:
1. **Tháng 1 (50% Học - 50% Làm):** Giai đoạn khởi đầu cần nhiều thời gian đọc tài liệu, nắm vững nguyên lý cảm biến, giao tiếp điện tử và làm quen thư viện để tránh cắm sai mạch hoặc chọn nhầm hướng tiếp cận.
2. **Tháng 2 & 3 (40% Học - 60% Làm):** Chuyển dịch mạnh sang thực hành. Mỗi khi nạp một khái niệm lý thuyết (như Digital Filtering hay Bounding Box regression), lập tức viết code kiểm chứng và nạp thử nghiệm ngay trong ngày.
3. **Tháng 4 (30% Học - 70% Làm):** Giai đoạn tích hợp hệ thống đòi hỏi phần lớn thời gian cho việc viết mã kết nối, xử lý lỗi bất đồng bộ mạng (network asynchronous), gỡ lỗi dữ liệu bị mất gói (debugging dropped packets).
4. **Tháng 5 (20% Học - 80% Làm):** Trọng tâm chuyển sang thao tác vật lý trên kệ mẫu: Cố tình tạo các mức che khuất, thay đổi đèn chiếu sáng, lặp lại các bài test nhiều lần và lưu file log số liệu.
5. **Tháng 6 (20% Học - 80% Làm):** Thời gian chủ yếu dành cho kỹ năng viết học thuật, biên tập biểu đồ, rà soát văn bản và luyện tập diễn thuyết.
6. **Tháng 7 & 8:** Học các công cụ chuyên sâu về nhúng và phương pháp viết báo học thuật quốc tế, kết hợp viết bài và tối ưu hóa hệ thống.

---

# PHẦN 5: LỊCH HỌC TẬP & LÀM DỰ ÁN MẪU HẰNG TUẦN

Mỗi tuần được chia làm **5 loại phiên làm việc (Session Types)** có mục tiêu rõ ràng:
* **Session 1 (Lý Thuyết & Nghiên Cứu):** Đọc tài liệu, hiểu thuật toán, vẽ sơ đồ khối.
* **Session 2 (Thực Hành Coding):** Viết script mẫu độc lập, thử nghiệm hàm trên notebook/file nhỏ.
* **Session 3 (Triển Khai Module & Phần Cứng):** Tích hợp vào codebase chính hoặc lắp ráp mạch điện trên kệ.
* **Session 4 (Kiểm Thử & Gỡ Lỗi - Testing & Debugging):** Chạy thử nghiệm thực tế, bắt lỗi ngoại lệ, tối ưu hóa code.
* **Session 5 (Tổng Kết, Ghi Chép & Báo Cáo):** Cập nhật nhật ký kỹ thuật, commit GitHub, chuẩn bị nội dung báo cáo GVHD.

---

### KỊCH BẢN A: Dành cho sinh viên bận học trên lớp (10 – 15 giờ / tuần)

Phù hợp trong các tuần phải học nhiều môn chuyên ngành hoặc thi giữa kỳ:

| Ngày trong tuần | Loại phiên làm việc | Thời gian | Nội dung công việc mẫu |
|---|---|:---:|---|
| **Thứ Ba** | Session 1: Học kiến thức mới | 2.5 giờ (Tối 20:00 - 22:30) | Đọc tài liệu về một khái niệm cụ thể (ví dụ: Median Filter hoặc YOLO loss). |
| **Thứ Năm** | Session 2: Thực hành coding độc lập | 2.5 giờ (Tối 20:00 - 22:30) | Viết script code thử nghiệm thuật toán trên máy tính. |
| **Thứ Bảy** | Session 3 & 4: Triển khai & Gỡ lỗi | 5.0 giờ (Chiều 14:00 - 19:00) | Lắp ráp phần cứng kệ hoặc tích hợp module vào hệ thống, test thực tế. |
| **Chủ Nhật** | Session 5: Đánh giá & Báo cáo | 3.0 giờ (Sáng 09:00 - 12:00) | Viết tài liệu nhật ký tuần, chuẩn bị file tóm tắt tiến độ báo cáo GVHD. |
| **TỔNG CỘNG** | | **13.0 giờ** | *Đạt mức duy trì tiến độ ổn định liên tục mà không bị kiệt sức.* |

---

### KỊCH BẢN B: Dành cho giai đoạn tăng tốc / Kỳ nghỉ (20 – 25 giờ / tuần)

Phù hợp trong giai đoạn tập trung làm đề tài (Tháng 3, 4, 5):

| Ngày trong tuần | Loại phiên làm việc | Thời gian | Nội dung công việc mẫu |
|---|---|:---:|---|
| **Thứ Hai** | Session 1: Nghiên cứu chuyên sâu | 3.5 giờ (Sáng 08:30 - 12:00) | Đọc sâu thuật toán mới, thiết kế kiến trúc luồng dữ liệu tuần. |
| **Thứ Ba** | Session 2: Code thuật toán core | 4.0 giờ (Chiều 13:30 - 17:30) | Viết code lõi cho module (ví dụ: Disagreement Engine). |
| **Thứ Tư** | Nghỉ ngơi / Học môn khác | - | Tránh kiệt sức (burnout), nạp năng lượng. |
| **Thứ Năm** | Session 3: Tích hợp phần cứng & mạng | 4.5 giờ (Chiều 13:30 - 18:00) | Nạp code ESP32, test truyền nhận MQTT qua mạng nội bộ. |
| **Thứ Sáu** | Session 4: Debugging & Optimization | 4.5 giờ (Chiều 13:30 - 18:00) | Kiểm thử các trường hợp biên, đo lường độ trễ và sửa lỗi logic. |
| **Thứ Bảy** | Session 4 (tiếp): Chạy thực nghiệm | 4.5 giờ (Chiều 13:30 - 18:00) | Thu thập các mẫu dữ liệu thực nghiệm trên kệ vật lý. |
| **Chủ Nhật** | Session 5: Review & Báo cáo GVHD | 3.0 giờ (Sáng 09:00 - 12:00) | Phân tích số liệu, vẽ biểu đồ, viết tóm tắt tuần gửi GVHD. |
| **TỔNG CỘNG** | | **24.0 giờ** | *Tiến độ triển khai cực nhanh, dẫn đầu chất lượng đề tài.* |

---

# PHẦN 6: LỘ TRÌNH KIẾN THỨC THEO THỨ TỰ PHỤ THUỘC

Nhằm tránh việc học chắp vá hoặc học những thứ nâng cao khi chưa có nền tảng, toàn bộ kiến thức được tổ chức thành **5 luồng phụ thuộc có hướng (Directed Acyclic Dependency Graphs)**:

```text
LUỒNG 1: COMPUTER VISION & DEEP LEARNING (CV)
[Python Cơ Bản] ──> [NumPy & OpenCV] ──> [Dataset Retail (Roboflow)]
                                              │
                                              ▼
[Đánh Giá Model (mAP/F1)] <── [Huấn Luyện YOLOv8n/11n]

─────────────────────────────────────────────────────────────────────────────

LUỒNG 2: FIRMWARE & XỬ LÝ TÍN HIỆU NHÚNG (DSP)
[C/C++ Vi Điều Khiển] ──> [ESP32 GPIO & HX711 Read] ──> [Lọc Số (Moving Avg/Median)]
                                                               │
                                                               ▼
[Gói Tin JSON & Cờ Settled] <── [Finite State Machine (FSM Phát Hiện Ổn Định)]

─────────────────────────────────────────────────────────────────────────────

LUỒNG 3: SENSOR FUSION & DISAGREEMENT REASONING
[Xác Suất Thống Kê] ──> [Mô Hình Trọng Lượng Vật Thể] ──> [Công Thức Bất Đồng Dt]
                                                               │
                                                               ▼
[Bayesian State Update] <── [Two-Tier Adaptive Verification Strategy]

─────────────────────────────────────────────────────────────────────────────

LUỒNG 4: IOT SYSTEM, BACKEND & DASHBOARD
[Lý Thuyết MQTT Broker] ──> [PubSubClient ESP32 & Paho-MQTT Python]
                                    │
                                    ▼
[Cơ Sở Dữ Liệu SQLite] <── [FastAPI/Streamlit Web Dashboard] ──> [Hệ Thống Cảnh Báo]

─────────────────────────────────────────────────────────────────────────────

LUỒNG 5: EDGE AI DEPLOYMENT & PAPER WRITING (THÁNG 7-8)
[Linux Nhúng (RPi5/Jetson)] ──> [ONNX / TensorRT Export] ──> [Lượng Tử Hóa FP16/INT8]
                                                                     │
                                                                     ▼
[Nộp Bài Báo IEEE/Scopus] <── [LaTeX & Viết Bản Thảo Tiếng Anh] <── [System Benchmark]
```

### Giải Thích Lý Do Cần Học Theo Đúng Trật Tự:
1. **Tại sao phải học Lọc số và FSM trước khi truyền MQTT?**
   * Nếu gửi tín hiệu thô trực tiếp từ HX711 qua MQTT, nhiễu dao động sẽ liên tục kích hoạt cờ sai lệch, gây nghẽn đường truyền Wi-Fi và làm bộ suy luận Vision chạy quá tải vô ích.
2. **Tại sao phải làm chủ YOLO Baseline trước khi viết Fusion Engine?**
   * Thuật toán Fusion cần 2 thông số thực tế từ Vision là số lượng $\hat{N}_v$ và độ tin cậy $C_v$. Nếu chưa có mô hình phát hiện ổn định, thuật toán Fusion sẽ không có đầu vào thực để tinh chỉnh ngưỡng bất đồng $\tau_D$.
3. **Tại sao Web Dashboard làm ở Tháng 4 mà không làm từ Tháng 1?**
   * Dashboard chỉ là lớp hiển thị kết quả (Presentation Layer). Nếu làm Dashboard quá sớm khi các module lõi chưa hoạt động, sinh viên sẽ lãng phí nhiều tuần chỉnh sửa giao diện cho những dữ liệu chưa tồn tại.

---

# PHẦN 7: THIẾT KẾ & KẾ HOẠCH THỰC NGHIỆM CHI TIẾT

Bám sát chính xác **6 thí nghiệm cốt lõi** được thiết kế trong tài liệu Brainstorming nhằm đảm bảo tính chặt chẽ học thuật cho khóa luận:

```text
+------------------------------------------------------------------------------------------------+
|                                    MA TRẬN 6 BÀI THỰC NGHIỆM                                   |
+-------+-------------------------+-----------------------------------+--------------------------+
| Mã    | Tên Thực Nghiệm         | Mục Đích Học Thuật                | Công Cụ / Phương Pháp    |
+-------+-------------------------+-----------------------------------+--------------------------+
| EXP-1 | Baseline Comparison     | Chứng minh Fusion vượt trội       | 4 Phương pháp đối sánh   |
| EXP-2 | Occlusion Robustness    | Khảo sát khả năng kháng che khuất | 3 Mức che khuất          |
| EXP-3 | Lighting Robustness     | Khảo sát điều kiện chiếu sáng     | 4 Điều kiện ánh sáng     |
| EXP-4 | Ablation Study          | Định lượng giá trị từng module    | Bóc tách 4 thành phần    |
| EXP-5 | Energy & Inference Rate | Đo lường mức tiết kiệm tài nguyên | Continuous vs. Event-Trig|
| EXP-6 | End-to-End Latency      | Đo lường độ trễ toàn trình        | Phân tích 5 chặng trễ    |
+-------+-------------------------+-----------------------------------+--------------------------+
```

---

### EXP-1: SO SÁNH CÁC PHƯƠNG PHÁP BASELINE (BASELINE COMPARISON)
* **Mục đích:** Chứng minh phương pháp đề xuất (Proposed Disagreement-Aware Fusion) đạt độ chính xác kiểm kê vượt trội so với các phương pháp truyền thống.
* **4 Phương pháp đối sánh:**
  1. *Vision-Only:* Chỉ dùng kết quả từ YOLOv8n.
  2. *Weight-Only:* Chỉ dùng kết quả quy đổi từ Load Cell: $\hat{N}_w = \text{round}(W / w_{item})$.
  3. *Fixed Sensor Fusion:* Hợp nhất tuyến tính trọng số cố định: $\hat{N} = \alpha \hat{N}_v + (1-\alpha) \hat{N}_w$ (với $\alpha = 0.5$).
  4. *Proposed Method:* Hợp nhất thích ứng nhận biết bất đồng kết hợp Two-Tier Verification.
* **Dữ liệu thu thập:** 100 kịch bản kiểm kê ngẫu nhiên với số lượng hàng hóa từ 1 đến 10 sản phẩm trên mỗi ô cân.
* **Chỉ số đánh giá (Metrics):**
  * Counting MAE: $\text{MAE} = \frac{1}{M}\sum_{i=1}^M |N_{true}^{(i)} - \hat{N}^{(i)}|$.
  * Exact Count Accuracy (ECA): Tỷ lệ % số lần đếm chính xác hoàn toàn ($N_{true} = \hat{N}$).
  * Low-Stock Alert F1-Score.
* **Biểu đồ tạo ra:** Bar Chart so sánh MAE và ECA giữa 4 phương pháp.

---

### EXP-2: ĐÁNH GIÁ THEO MỨC ĐỘ CHE KHUẤT (OCCLUSION ROBUSTNESS)
* **Mục đích:** Khảo sát độ bền vững của hệ thống khi tầm nhìn camera bị suy giảm do sản phẩm xếp chồng chéo hoặc bị vật thể che khuất.
* **Các mức che khuất thiết lập:**
  1. *Che khuất nhẹ (0% - Không che):* Sản phẩm đứng tách rời, thấy rõ nhãn.
  2. *Che khuất trung bình (20% – 40% diện tích):* Sản phẩm bị che 1 phần bởi sản phẩm khác đứng trước.
  3. *Che khuất nặng (> 50% diện tích):* Sản phẩm nằm sâu bên trong góc kẹt hoặc bị che khuất gần như hoàn toàn.
* **Số mẫu:** 30 lượt thử cho mỗi mức che khuất (Tổng: 90 tests).
* **Chỉ số đánh giá:** Counting MAE và Missing Detection Rate (Tỉ lệ đếm sót).
* **Biểu đồ tạo ra:** Line Plot thể hiện đường cong MAE tăng dần theo độ che khuất; chứng minh Proposed Fusion giữ đường cong dẹt hơn rất nhiều so với Vision-Only.

---

### EXP-3: ĐÁNH GIÁ THEO ĐIỀU KIỆN ÁNH SÁNG (LIGHTING ROBUSTNESS)
* **Mục đích:** Đánh giá tính khả thi khi triển khai trong cửa hàng thực tế có điều kiện ánh sáng thay đổi giữa ngày, đêm hoặc mất cân bằng sáng.
* **4 Điều kiện ánh sáng thử nghiệm:**
  1. *Ánh sáng chuẩn (Normal Office Lighting: ~400–500 lux).*
  2. *Ánh sáng mạnh (Bright / Glare: ~800–1000 lux).*
  3. *Ánh sáng yếu (Dim / Night Mode: ~50–100 lux).*
  4. *Ánh sáng không đều (Uneven Shadows: Một bên sáng, một bên tối cục bộ).*
* **Chỉ số đánh giá:** Counting Accuracy (%) và Confidence Score trung bình của detector.
* **Biểu đồ tạo ra:** Grouped Bar Chart so sánh độ chính xác giữa Vision-Only và Proposed Fusion qua 4 môi trường sáng.

---

### EXP-4: ABLATION STUDY (NGHIÊN CỨU BÓC TÁCH THÀNH PHẦN)
* **Mục đích:** Phân tích định lượng xem đóng góp thực sự đến từ đâu; chứng minh từng khối chức năng thêm vào đều nâng cao hiệu quả hệ thống.
* **4 Cấu hình bóc tách:**
  * *Config A:* Camera Only (YOLOv8n).
  * *Config B:* Camera + Weight (Quy tắc kết hợp đơn giản, không xét confidence).
  * *Config C:* Camera + Weight + Confidence-Aware (Có xét ngưỡng tin cậy $C_v$).
  * *Config D (Full Proposed):* Camera + Weight + Confidence + Two-Tier Adaptive Verification.
* **Thí nghiệm phụ bổ sung:** Quét thông số ngưỡng bất đồng $\tau_D \in \{0, 1, 2, 3\}$ để vẽ biểu đồ đánh đổi giữa độ chính xác và tần suất phải kích hoạt Tier 2.
* **Biểu đồ tạo ra:** Bảng Ablation chuẩn IEEE và biểu đồ Trade-off Pareto giữa Verification Rate (%) vs. Accuracy (%).

---

### EXP-5: ĐÁNH GIÁ HIỆU QUẢ NĂNG LƯỢNG & TẦN SUẤT SUY LUẬN (ENERGY & INFERENCE EFFICIENCY)
* **Mục đích:** Chứng minh tính ưu việt của cơ chế **Event-Triggered** (chỉ suy luận khi có biến động cân) so với cơ chế **Continuous Inference** (chạy quét liên tục).
* **Kịch bản kiểm thử:** Giả lập hoạt động của một kệ hàng trong 2 giờ liên tục, trung bình có 15 thao tác nhấc/đặt sản phẩm của khách hàng mỗi giờ (Tổng: 30 sự kiện giao dịch).
* **Chỉ số đánh giá:**
  * Số frame phải xử lý qua AI mỗi giờ (Processed Frames/Hour).
  * Tỷ lệ tải CPU / GPU trung bình (% Utilization).
  * Năng lượng tiêu thụ ước tính (Joules hoặc Watt-hour).
  * Tỉ lệ bỏ sót sự kiện (Missed Event Rate - phải bằng 0%).
* **Biểu đồ tạo ra:** Area Chart biểu diễn tải CPU theo dòng thời gian giữa 2 chế độ; thể hiện mức tiết kiệm $> 75\%$ tài nguyên xử lý.

---

### EXP-6: ĐÁNH GIÁ ĐỘ TRỄ TOÀN TRÌNH (END-TO-END LATENCY PROFILING)
* **Mục đích:** Đo đạc chi tiết từng thành phần thời gian cấu thành nên độ trễ toàn trình của hệ thống, chứng minh hệ thống đáp ứng tiêu chuẩn thời gian thực (Real-time).
* **Công thức phân rã độ trễ:**
  $$T_{system} = T_{sensor} + T_{comm} + T_{vision} + T_{fusion} + T_{dashboard}$$
  * Trong đó:
    * $T_{sensor}$: Thời gian lấy mẫu và ổn định cân ($T_{settle} \approx 400 - 500\text{ms}$).
    * $T_{comm}$: Thời gian đóng gói JSON, gửi qua Wi-Fi MQTT broker ($\approx 20 - 50\text{ms}$).
    * $T_{vision}$: Thời gian suy luận YOLO trên frame ảnh ($\approx 25 - 40\text{ms}$ trên PC, $\approx 90\text{ms}$ trên RPi5).
    * $T_{fusion}$: Thời gian tính toán ma trận bất đồng và ra quyết định ($\approx 2 - 5\text{ms}$).
    * $T_{dashboard}$: Thời gian cập nhật trạng thái WebSocket lên màn hình Web ($\approx 10 - 20\text{ms}$).
* **Biểu đồ tạo ra:** Stacked Bar Chart thể hiện tỷ trọng thời gian của từng khâu trong chu trình.

---

# PHẦN 8: MA TRẬN QUẢN LÝ RỦI RO & KẾ HOẠCH DỰ PHÒNG

Dưới đây là bảng quản lý rủi ro toàn diện được thiết kế giúp sinh viên luôn có lối thoát an toàn (Safe Fallback) trong mọi tình huống bất lợi:

| TT | Rủi ro kỹ thuật tiềm ẩn | Giai đoạn | Mức độ | Phương án xử lý chủ động (Mitigation) | Phương án thay thế an toàn (Fallback Option) |
|:---:|---|:---:|:---:|---|---|
| **R-1** | **HX711 bị nhiễu cao tần, trôi dạt (drift) theo nhiệt độ.** | Tháng 1, 2 | **Cao** | - Nối đất (GND) cẩn thận, thêm tụ gốm 100nF giữa VCC và GND.<br>- Sử dụng Median Filter 5 mẫu + Moving Average 10 mẫu. | Định kỳ tự động Tare (về 0) khi kệ rỗng; nới rộng ngưỡng sai số cho phép của từng SKU từ $\pm 3g$ lên $\pm 8g$. |
| **R-2** | **Không đủ thời gian/dữ liệu huấn luyện YOLO.** | Tháng 2 | **Trung bình** | - Áp dụng mạnh kỹ thuật Transfer Learning từ model pre-trained SKU-110K.<br>- Tăng cường ảnh tự động (Augmentation: flip, crop, brightness). | Giới hạn bài toán xuống còn đúng **5 SKU** có hình dáng và bao bì tương phản rõ rệt để đạt độ chính xác tối đa. |
| **R-3** | **Mô hình Vision nhận diện kém khi sản phẩm xoay nghiêng.** | Tháng 2, 3 | **Trung bình** | - Bổ sung ảnh chụp sản phẩm từ nhiều góc nghiêng trong tập train.<br>- Gắn camera ở góc nghiêng 45° thay vì nhìn thẳng từ trên xuống. | Quy định chuẩn trưng bày kệ hàng: Mặt trước sản phẩm luôn hướng về phía camera (phù hợp với quy chuẩn bán lẻ). |
| **R-4** | **ESP32 mất kết nối Wi-Fi hoặc chập chờn MQTT.** | Tháng 4 | **Trung bình** | - Viết hàm tự động `reconnect()` trong vòng lặp loop của ESP32.<br>- Lưu đệm dữ liệu vào vòng đệm (Ring Buffer) trên RAM nếu mất mạng. | **Dự phòng an toàn:** Kết nối trực tiếp ESP32 với máy tính qua cổng USB cáp Serial (UART) để truyền dữ liệu nếu Wi-Fi trường đại học chặn cổng MQTT. |
| **R-5** | **Thuật toán Fusion không cải thiện rõ rệt so với Vision-only.** | Tháng 3, 5 | **Cao** | - Phân tích lại ngưỡng $\tau_D$ và $\tau_C$.<br>- Cố tình thiết kế bài toán với mật độ che khuất cao ($> 40\%$) để làm nổi bật vai trò cứu cánh của cảm biến trọng lượng. | Chứng minh giá trị ở khía cạnh khác: **Tiết kiệm năng lượng và giảm số frame suy luận** thông qua cơ chế Event-Triggered thay vì chỉ tập trung vào độ chính xác. |
| **R-6** | **Edge Hardware (RPi5/Jetson) gặp lỗi, cháy linh kiện, thiếu chip.** | Tháng 7 | **Thấp** | - Khóa luận chính khóa đã hoàn thành 100% trên PC/Laptop từ Tháng 6.<br>- Edge Hardware chỉ là phần mở rộng thêm điểm thưởng. | Trình diễn toàn bộ hệ thống bằng Laptop cá nhân đặt kín bên dưới gầm kệ đóng vai trò Local Edge Server. |
| **R-7** | **Thiếu thời gian thực hiện đầy đủ 6 bài thực nghiệm.** | Tháng 5 | **Cao** | - Tự động hóa quá trình ghi log thực nghiệm bằng script Python xuất thẳng ra CSV.<br>- Không nhập số liệu thủ công bằng tay. | Ưu tiên thực hiện trọn vẹn 3 bài then chốt: **EXP-1 (Baseline), EXP-2 (Occlusion), EXP-5 (Event-Triggered)**; 3 bài còn lại làm ở quy mô mẫu nhỏ hơn. |
| **R-8** | **Chậm tiến độ viết khóa luận.** | Tháng 6 | **Cao** | - Viết cuốn chiếu từng chương ngay sau khi hoàn thành tháng tương ứng (Ví dụ: Xong tháng 2 viết ngay Chương tổng quan và Vision). | Sử dụng khung dàn ý chi tiết có sẵn, chuyển các phần code dài và log số liệu chi tiết vào Phụ lục (Appendix). |

---

# PHẦN 9: KHUNG THEO DÕI TIẾN ĐỘ

Để sinh viên và Giảng viên Hướng dẫn dễ dàng kiểm soát công việc trên Excel, Notion hoặc Google Sheets, dưới đây là mẫu thiết kế bảng quản lý tác vụ:

### 9.1 Bảng Quản Lý Tác Vụ Dự Án (Task Management Template)

| Task ID | Tháng | Tuần | Tên Nhiệm Vụ | Mục Tiêu Học Tập | Module Kỹ Thuật | Ngày Bắt Đầu | Hạn Chót | Giờ Ước Tính | Giờ Thực Tế | Trạng Thái | Deliverable (Sản Phẩm) | Tiêu Chí Hoàn Thành | Ghi Chú |
|:---:|:---:|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|---|---|
| **T-01** | T1 | W1 | Đọc tài liệu Retail CV | Retail CV SOTA | Nghiên cứu | 01/10 | 07/10 | 14h | 12h | Completed | File `review_notes.md` | Tóm tắt 5 paper cốt lõi | Hiểu rõ SKU-110K |
| **T-02** | T1 | W2 | Mạch đọc ESP32-HX711 | C/C++, ADC 24bit | Firmware | 08/10 | 14/10 | 14h | 15h | Completed | Code `test_hx711.ino` | Đọc số nguyên raw ổn định | Chú ý dây nối GND |
| **T-03** | T1 | W3 | Hiệu chuẩn cảm biến cân | Calibration math | Firmware | 15/10 | 21/10 | 14h | 13h | Completed | Code `calibrate.ino` | Cân quả chuẩn 500g sai số $\le 2g$ | Dùng quả cân chuẩn |
| **T-04** | T1 | W4 | Lắp khung kệ mẫu v1 | Cơ khí khung kệ | Hardware | 22/10 | 28/10 | 15h | 16h | Completed | Khung kệ hoàn chỉnh | Camera góc nhìn bao quát ô cân | Góc xiên 45 độ |
| **T-05** | T2 | W5 | Thu thập dataset 5 SKU | Data labeling | Vision | 29/10 | 04/11 | 16h | - | In Progress| 1000 ảnh gán nhãn | Đủ ảnh che khuất và các góc | Dùng Roboflow |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

*Các trạng thái chuẩn (Status):*
* `Not Started` (Chưa bắt đầu)
* `Learning` (Đang học lý thuyết)
* `In Progress` (Đang thực hiện/Coding)
* `Testing` (Đang kiểm thử/Gỡ lỗi)
* `Completed` (Đã hoàn thành nghiệm thu)
* `Blocked` (Gặp lỗi bế tắc cần trợ giúp)

---

### 9.2 Khung Báo Cáo & Tổng Kết Cuối Tuần (Weekly Review Template)

Mỗi tối Chủ Nhật, sinh viên điền vào form này (dành 20–30 phút) để gửi email cập nhật cho Giảng viên Hướng dẫn:

```markdown
# BÁO CÁO TIẾN ĐỘ TUẦN SỐ: [Điền số tuần]
Ngày báo cáo: [Ngày/Tháng/Năm]
Sinh viên thực hiện: [Họ và tên]

---

### 1. KIẾN THỨC MỚI ĐÃ HỌC TRONG TUẦN
- [Khái niệm lý thuyết 1 đã nắm vững]
- [Kỹ năng công nghệ/thư viện mới đã sử dụng được]

### 2. CÔNG VIỆC DỰ ÁN ĐÃ HOÀN THÀNH
- [x] Đã hoàn thành Task [Mã Task]: [Mô tả ngắn gọn kết quả]
- [x] Đã viết xong script/firmware: [Tên file và link commit GitHub]
- [x] Kết quả kiểm thử: [Số liệu kiểm thử thực tế đạt được]

### 3. CÁC LỖI KỸ THUẬT GẶP PHẢI & CÁCH ĐÃ XỬ LÝ
- *Lỗi 1:* [Mô tả hiện tượng lỗi] -> *Cách sửa:* [Đã khắc phục như thế nào]
- *Lỗi 2 còn tồn tại:* [Mô tả lỗi chưa giải quyết được]

### 4. VẤN ĐỀ CẦN XIN Ý KIẾN THẦY/CÔ HƯỚNG DẪN
1. [Câu hỏi học thuật hoặc kỹ thuật 1]
2. [Xin ý kiến về lựa chọn linh kiện hoặc phương pháp]

### 5. KẾ HOẠCH CỤ THỂ CHO TUẦN TIẾP THEO
- Mục tiêu chính: [Mục tiêu cốt lõi tuần tới]
- Danh sách các Task dự kiến: [Liệt kê mã Task T-xx]
```

---

# PHẦN 10: BẢNG CÂU HỎI LÀM RÕ & XÁC NHẬN THIẾT KẾ

Để cá nhân hóa kế hoạch hành động này chính xác 100% theo điều kiện thực tế của bạn, dưới đây là danh sách các câu hỏi định hình cần bạn xác nhận:

1. **Về Nền Tảng Kỹ Thuật Hiện Tại:**
   * Trình độ hiện tại của bạn về **Python** (Cơ bản / Trung bình / Đã từng làm dự án với OpenCV/PyTorch)?
   * Trình độ lập trình vi điều khiển **C/C++ & ESP32** (Mới bắt đầu / Đã từng nạp code qua Arduino IDE / Đã làm chủ I2C/SPI/FreeRTOS)?
2. **Về Quỹ Thời Gian Hằng Tuần:**
   * Trong học kỳ này, bạn dự kiến dành được bao nhiêu giờ mỗi tuần cho đề tài này (Khoảng **10–15 giờ** hay **20–25 giờ**)?
3. **Về Mốc Thời Gian Khóa Luận:**
   * Ngày bắt đầu chính thức dự kiến là khi nào?
   * Tháng mấy sẽ diễn ra lễ bảo vệ chính thức của trường (Deadline nộp quyển)?
4. **Về Tài Nguyên Phần Cứng Đang Có Sẵn:**
   * Bạn đã có sẵn máy tính có card đồ họa rời GPU NVIDIA (ví dụ: RTX 3050/3060/4060) hay sử dụng Google Colab / GPU trường?
   * Bạn đã mua sẵn bộ linh kiện Load Cell + HX711 và ESP32 chưa, hay cần một danh mục gợi ý mua sắm tối ưu chi phí?
   * Bạn dự định dùng **Raspberry Pi 5** hay **Jetson Orin Nano** cho giai đoạn mở rộng Tháng 7–8?

---

> [!NOTE]
> *Bản kế hoạch hành động trên đã được thiết kế hoàn chỉnh và lưu trữ tại đường dẫn:*  
> `c:\Users\ADMIN\Documents\Senior Project\Plan\KE_HOACH_HOC_TAP_VA_TRIEN_KHAI_AIOT_SMART_SHELF.md`.  
> *Bạn có thể mở trực tiếp tập tin này trong môi trường soạn thảo để theo dõi và điều chỉnh bất kỳ thông số nào theo nhu cầu thực tế!*
