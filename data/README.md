# Data Directory

Thư mục lưu trữ dữ liệu dự án:
- `raw/`: Ảnh gốc, file ghi log cảm biến thô (CSV/JSON từ MQTT/Serial).
- `processed/`: Dữ liệu đã qua tiền xử lý, gán nhãn ground-truth, các tập train/val/test.
- `sample/`: Bộ dữ liệu mẫu kích thước nhỏ dùng cho unit test và demo nhanh.

> Lưu ý: Các file dữ liệu lớn, video ghi hình và tệp nén sẽ được `.gitignore` bỏ qua để tránh làm phình dung lượng git repo.
