# Vision Subsystem (Computer Vision & Object Detection)

Thư mục chứa mã nguồn xử lý thị giác máy tính cho kệ hàng thông minh:
- Thu thập ảnh / luồng video từ camera (Webcam / RTSP IP Camera / ESP32-CAM).
- Nhận diện sản phẩm trên kệ sử dụng mô hình học sâu (YOLOv8 / YOLOv11 / SKU-110k).
- Phân vùng kệ hàng (ROI - Region of Interest).
- Đếm số lượng sản phẩm trực quan (Vision Count) và tính toán độ tin cậy phát hiện (Vision Confidence).
- Đánh giá điều kiện môi trường quan sát (độ sáng, mức độ che khuất / occlusion).

## 📊 Công cụ Thực nghiệm Cơ sở (Baseline Evaluation)
- `eval_yolov8n_baseline.py`: Script chạy suy luận và đối soát mô hình `YOLOv8n` pretrained trên tập ảnh kệ hàng thực tế.
- `ground_truth_yolo.csv`: Bảng ánh xạ số lượng đếm thực tế (Ground Truth) theo 4 nhóm kịch bản thử nghiệm (Baseline, Low Occlusion, Heavy Occlusion, Environment Noise).
- Kết quả được xuất tự động ra thư mục `output/baseline_yolov8n/` (bao gồm tệp thống kê CSV, báo cáo Markdown và ảnh trực quan có vẽ Bounding Box).
