# Fusion Subsystem (Disagreement-Aware Sensor Fusion)

Thư mục chứa động cơ tích hợp và phân xử dữ liệu giữa Thị giác (Vision) và Trọng lượng (Weight Sensor):
- Đồng bộ hóa mốc thời gian (Timestamp synchronization) giữa dữ liệu camera và cảm biến trọng lượng.
- Ước lượng số lượng hàng theo trọng lượng: `Weight Count = Delta Weight / Unit Weight`.
- Phát hiện bất đồng (Disagreement Detection) khi `|Vision Count - Weight Count| > threshold`.
- Cơ chế giải quyết bất đồng (Disagreement Resolution Mechanism):
  - Nhận diện trường hợp che khuất (Occlusion - Vision count < Weight count): ưu tiên Weight sensor.
  - Nhận diện trường hợp đặt dị vật / tác động cơ học ngoại lai: đối chiếu với độ tin cậy Vision.
  - Phân loại trạng thái hệ thống: Tin cậy (Concordant), Cảnh báo bất đồng (Discrepant), Lỗi cảm biến (Sensor Fault).
- Xuất kết quả tồn kho cuối cùng cùng chỉ số tin cậy (Fused Inventory State & Confidence Score).

## 📏 Module Đánh Giá Hiệu Năng (`fusion/metrics.py` - S02-C1)
- **Counting Metrics**: `counting_mae`, `counting_rmse`, `exact_accuracy`, `undercount_rate`, `overcount_rate`, `evaluate_counting`.
- **Object Detection Metrics (`map_wrapper`)**: `compute_iou`, `compute_iou_matrix`, `compute_ap`, `map_wrapper` (hỗ trợ cả tính toán độc lập bằng NumPy và trích xuất từ đối tượng `ultralytics`).
- **Event-level Metrics**: `event_precision_recall_f1`, `event_precision`, `event_recall`, `event_f1`, `derive_inventory_events` (hỗ trợ phân loại sự kiện nhị phân và đa lớp).
