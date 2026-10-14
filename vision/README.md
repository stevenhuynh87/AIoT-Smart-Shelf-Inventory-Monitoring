# Vision Subsystem (Computer Vision & Object Detection)

Thư mục chứa mã nguồn xử lý thị giác máy tính cho kệ hàng thông minh:
- Thu thập ảnh / luồng video từ camera (Webcam / RTSP IP Camera / ESP32-CAM).
- Nhận diện sản phẩm trên kệ sử dụng mô hình học sâu (YOLOv8 / YOLOv11 / SKU-110k).
- Phân vùng kệ hàng (ROI - Region of Interest).
- Đếm số lượng sản phẩm trực quan (Vision Count) và tính toán độ tin cậy phát hiện (Vision Confidence).
- Đánh giá điều kiện môi trường quan sát (độ sáng, mức độ che khuất / occlusion).
