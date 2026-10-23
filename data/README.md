# Data Directory

Thư mục lưu trữ dữ liệu dự án:
- `raw/`: Ảnh gốc, file ghi log cảm biến thô (CSV/JSON từ MQTT/Serial).
- `processed/`: Dữ liệu đã qua tiền xử lý, gán nhãn ground-truth, các tập train/val/test.
- `sample/`: Bộ dữ liệu mẫu kích thước nhỏ dùng cho unit test và demo nhanh.

> **Quy chuẩn dữ liệu**: Xem chi tiết định dạng và schema các cột CSV tại [docs/dataset_spec.md](file:///c:/Users/ADMIN/Documents/Disagreement-Aware%20Vision%20and%20Weight%20Sensor%20Fusion%20for%20AIoT%20Smart%20Shelf%20Inventory%20Monitoring/docs/dataset_spec.md).

> Lưu ý: Các file dữ liệu lớn, video ghi hình và tệp nén sẽ được `.gitignore` bỏ qua để tránh làm phình dung lượng git repo.
