# Disagreement-Aware Vision and Weight Sensor Fusion for AIoT Smart Shelf Inventory Monitoring

Hệ thống giám sát tồn kho kệ hàng thông minh AIoT kết hợp thị giác máy tính (Computer Vision) và cảm biến trọng lượng (Weight Sensor/Load Cell) với cơ chế giải quyết bất đồng (Disagreement-Aware Sensor Fusion).

---

## 📁 Repository Structure

```text
├── firmware/         # Mã nguồn ESP32/Arduino, đọc cảm biến HX711/Load cell, MQTT publisher
├── vision/           # Module thị giác máy tính: Camera capture, YOLO detection, đếm vật thể
├── fusion/           # Động cơ Sensor Fusion nhận biết bất đồng (Disagreement-Aware Fusion Engine)
├── dashboard/        # Giao diện giám sát thời gian thực (Web Dashboard, MQTT client)
├── data/             # Dữ liệu thực nghiệm, log cảm biến, dataset mẫu, calibration
├── docs/             # Tài liệu kiến trúc, sơ đồ phần cứng, hướng dẫn vận hành
├── thesis/           # Báo cáo khóa luận tốt nghiệp, slide thuyết trình, biểu đồ
├── tests/            # Bộ kiểm thử tự động (pytest)
├── requirements.txt  # Danh sách thư viện Python
└── README.md         # Tài liệu tổng quan dự án
```

---

## 🚀 Quick Start (Môi trường phát triển)

### 1. Khởi tạo & Kích hoạt Virtual Environment

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt trên Windows PowerShell:
.venv\Scripts\Activate.ps1

# Kích hoạt trên Linux/macOS:
source .venv/bin/activate
```

### 2. Cài đặt các gói phụ thuộc

```bash
pip install -r requirements.txt
```

### 3. Kiểm thử môi trường

```bash
pytest
```
