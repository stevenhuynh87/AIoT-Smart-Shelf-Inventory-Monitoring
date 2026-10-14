# Firmware Subsystem (ESP32 / Microcontroller)

Thư mục chứa mã nguồn nhúng dành cho vi điều khiển (ESP32 / Arduino / RP2040) thực hiện:
- Đọc tín hiệu cảm biến tải trọng (Load Cell 40kg/50kg) thông qua ADC HX711.
- Xử lý lọc nhiễu trọng lượng thô (Moving Average Filter / Median Filter).
- Hiệu chuẩn cân điện tử (Calibration factor, Zero tare).
- Đóng gói dữ liệu trọng lượng và gửi lên MQTT Broker định kỳ hoặc theo sự kiện thay đổi (Event-driven / delta-weight).
