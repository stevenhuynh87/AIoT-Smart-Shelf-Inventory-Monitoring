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
