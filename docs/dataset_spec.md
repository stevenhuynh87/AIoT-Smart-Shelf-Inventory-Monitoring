# Đặc Tả Tập Dữ Liệu Đa Phương Thức (Multimodal Dataset Specification)

Tài liệu này định nghĩa cấu trúc, định dạng và quy chuẩn lưu trữ dữ liệu đa phương thức (Multimodal Data Contract) kết hợp giữa **Thị giác máy tính (Computer Vision)** và **Cảm biến trọng lượng (Load Cell / HX711)** trong hệ thống Kệ hàng thông minh AIoT.

---

## 1. Mục Đích & Phạm Vi Ứng Dụng

Dữ liệu thu thập từ kệ hàng thông minh AIoT là dữ liệu chuỗi thời gian đa phương thức (multimodal time-series). Để phục vụ cho việc huấn luyện, đánh giá mô hình nhận diện và kiểm thử **Động cơ phân xử bất đồng (Disagreement-Aware Fusion Engine)**, tập dữ liệu cần chuẩn hóa một schema thống nhất nhằm:
- **Đồng bộ hóa mốc thời gian**: Kết nối chính xác một khung hình camera tại thời điểm $t$ với giá trị cân nặng tương ứng.
- **Lưu trữ ground-truth tường minh**: Đảm bảo có nhãn chân lý phục vụ đo đạc các chỉ số hiệu năng (MAE, RMSE, Accuracy, Confusion Matrix).
- **Phân loại điều kiện môi trường nhiễu**: Ghi nhận định lượng mức độ che khuất thị giác (*occlusion*) và điều kiện ánh sáng (*lighting*) phục vụ các bài thực nghiệm đối sánh (EXP-1, EXP-2, EXP-3).

---

## 2. Cấu Trúc Lưu Trữ Thư Mục Dữ Liệu

Tập dữ liệu trong dự án được tổ chức theo cấu trúc sau:

```text
data/
├── raw/                      # Dữ liệu gốc thu thập từ phần cứng
│   ├── frames/               # Thư mục chứa các ảnh chụp từ camera (.jpg)
│   │   ├── exp1_0001.jpg
│   │   ├── exp1_0002.jpg
│   │   └── ...
│   └── exp1_baseline_raw.csv # File CSV ghi nhận theo schema chuẩn
├── processed/                # Dữ liệu đã làm sạch, căn chỉnh mốc thời gian
│   └── exp1_baseline.csv
└── sample/                   # Dữ liệu mẫu kích thước nhỏ dùng cho Unit Test
    ├── frames/
    └── sample_dataset.csv
```

---

## 3. Bảng Định Nghĩa Schema CSV Chi Tiết

Mỗi bản ghi (record / dòng) trong tệp CSV đại diện cho một trạng thái quan trắc của kệ hàng tại một mốc thời gian xác định:

| Tên Cột | Kiểu Dữ Liệu | Đơn Vị / Định Dạng | Ràng Buộc (Constraints) | Ý Nghĩa Kỹ Thuật & Vai Trò Trong Fusion |
| :--- | :---: | :---: | :---: | :--- |
| `timestamp` | `string` / `float` | ISO 8601 (`YYYY-MM-DDTHH:MM:SS.sssZ`) hoặc Epoch Unix ms | Đơn điệu tăng dần, không rỗng | Mốc thời gian đồng bộ giữa vi điều khiển (ESP32) và máy tính xử lý thị giác (Edge/Gateway). |
| `frame_path` | `string` | Đường dẫn tương đối dạng POSIX | Tệp `.jpg` / `.png` tồn tại, tỷ lệ 16:9 hoặc 4:3 | Vị trí ảnh chụp kệ hàng tại thời điểm quan trắc. Đầu vào cho mô hình YOLO để sinh ra $Count_{vision}$ và $Conf_{vision}$. |
| `raw_weight` | `float` | Gam ($g$) | Thường $\ge 0.0$ (có thể dao động nhẹ quanh $0.0$) | Giá trị trọng lượng thô trực tiếp từ ADC HX711 (chưa qua bộ lọc, phản ánh nhiễu rung cơ học / tiếp xúc). |
| `filtered_weight` | `float` | Gam ($g$) | $\ge 0.0$ | Trọng lượng sau khi qua bộ lọc số (Moving Average / Kalman Filter). Dùng để tính toán $Count_{weight} = \text{round}\left(\frac{W_{filtered}}{W_{unit}}\right)$. |
| `gt_count` | `integer` | Đơn vị chiếc (Items) | $\ge 0$ | **Ground-Truth**: Số lượng sản phẩm thực tế có trên mặt kệ tại thời điểm chụp/đo. Nhãn chuẩn để tính sai số MAE. |
| `gt_sku` | `string` | Mã định danh chuẩn hóa | Không rỗng (dùng `EMPTY` nếu kệ trống) | **Ground-Truth**: Mã sản phẩm (SKU). Dùng để tra bảng thông số hàng hóa ($W_{unit}$, dung sai trọng lượng $\sigma_w$, kích thước). |
| `occlusion_level` | `string` | Categorical Enum (`none`, `low`, `medium`, `high`) | Thuộc danh mục quy định | **Nhiễu thị giác**: Mức độ che khuất tầm nhìn camera đối với sản phẩm trên kệ (phục vụ bài thực nghiệm EXP-2). |
| `lighting_level` | `string` | Categorical Enum (`normal`, `low`, `harsh`, `uneven`) | Thuộc danh mục quy định | **Nhiễu quang học**: Điều kiện chiếu sáng môi trường tác động lên camera (phục vụ bài thực nghiệm EXP-3). |

---

## 4. Quy Chuẩn Giá Trị Danh Mục (Categorical Enumerations)

### 4.1. Mức Độ Che Khuất (`occlusion_level`)

| Giá Trị | Tỷ Lệ Che Khuất Tương Đương | Mô Tả Tình Huống Kệ Hàng Thực Tế | Tác Động Dự Kiến Đến Fusion |
| :---: | :---: | :--- | :--- |
| `none` | $0\%$ | Tất cả sản phẩm nằm rõ trong tầm nhìn của Camera, không bị che chắn. | $Count_{vision} \approx Count_{weight}$. Hai kênh đồng thuận cao. |
| `low` | $1\% - 20\%$ | Sản phẩm bị che khuất nhẹ một phần cạnh hoặc đáy (do mép kệ hoặc nhãn hàng xoay góc). | YOLO vẫn nhận diện tốt, độ tự tin $Conf_{vision}$ giảm nhẹ. |
| `medium` | $21\% - 50\%$ | Sản phẩm phía sau bị sản phẩm phía trước che khuất một nửa, hoặc khách hàng đang thò tay vào kệ. | Nguy cơ bỏ sót vật thể trong Vision ($Count_{vision} < Count_{weight}$). |
| `high` | $> 50\%$ | Sản phẩm xếp chồng lên nhau hoặc bị vật cản lớn che hoàn toàn khỏi góc máy camera. | Vision đếm thiếu nghiêm trọng. **Fusion Engine kích hoạt cơ chế ưu tiên Trọng lượng (Weight-Priority Resolution)**. |

### 4.2. Điều Kiện Ánh Sáng (`lighting_level`)

| Giá Trị | Cường Độ Sáng Tham Chiếu | Đặc Điểm Môi Trường | Tác Động Dự Kiến Đến Vision |
| :---: | :---: | :--- | :--- |
| `normal` | $300 - 500\text{ lux}$ | Ánh sáng tiêu chuẩn trong siêu thị / cửa hàng tiện lợi. | YOLO đạt độ chính xác cao nhất ($Conf_{vision} \ge 0.85$). |
| `low` | $< 100\text{ lux}$ | Ánh sáng yếu, tắt đèn nền, bóng tối bao phủ kệ. | Ảnh bị nhiễu hạt (noise), độ tự tin Vision giảm sâu, có thể sinh false negative. |
| `harsh` | $> 1000\text{ lux}$ | Đèn rọi trực tiếp gây lóa bề mặt vỏ sản phẩm (kim loại, nilon). | Cháy sáng (overexposed), mất vân viền nhận diện đặc trưng. |
| `uneven` | Không đồng đều | Nguồn sáng chỉ chiếu một bên góc kệ, tạo bóng đổ đậm (harsh shadows). | Dễ nhầm lẫn bóng đổ với vật thể hoặc che khuất cạnh nhận diện. |

---

## 5. Dữ Liệu Mẫu Đại Diện Cho Các Kịch Bản Thử Nghiệm

Dưới đây là nội dung mẫu minh họa một tệp CSV thực nghiệm chứa các kịch bản bất đồng điển hình:

```csv
timestamp,frame_path,raw_weight,filtered_weight,gt_count,gt_sku,occlusion_level,lighting_level
2026-10-21T10:00:00.000Z,data/raw/frames/exp1_0001.jpg,0.32,0.00,0,EMPTY,none,normal
2026-10-21T10:00:15.100Z,data/raw/frames/exp1_0002.jpg,356.10,355.00,1,COCA_CAN_330ML,none,normal
2026-10-21T10:00:30.450Z,data/raw/frames/exp1_0003.jpg,712.40,710.00,2,COCA_CAN_330ML,low,normal
2026-10-21T10:01:00.200Z,data/raw/frames/exp1_0004.jpg,1068.90,1065.00,3,COCA_CAN_330ML,high,normal
2026-10-21T10:01:30.800Z,data/raw/frames/exp1_0005.jpg,354.80,355.00,1,COCA_CAN_330ML,none,low
2026-10-21T10:02:00.050Z,data/raw/frames/exp1_0006.jpg,850.50,850.00,1,COCA_CAN_330ML,none,normal
```

### Phân tích các trạng thái quan trắc:
- **Dòng 1 (Kệ trống)**: Trọng lượng thô $0.32g$ do trôi điểm 0 (zero-drift); bộ lọc số đưa về $0.00g$. Nhãn `EMPTY`.
- **Dòng 2 & 3 (Đồng thuận - Concordant)**: Điều kiện ánh sáng chuẩn, không che khuất. Cả hai kênh đều tính toán ra 1 và 2 lon nước.
- **Dòng 4 (Bất đồng do che khuất - Occlusion Disagreement)**: 3 lon nước nhưng bị che khuất mức `high` (>50%). Khung hình camera có thể chỉ phát hiện 1–2 lon, nhưng kênh trọng lượng đo $1065g \approx 3 \times 355g$. Cơ chế giải quyết bất đồng ghi nhận $gt\_count = 3$.
- **Dòng 5 (Nhiễu ánh sáng yếu - Low-light)**: Điều kiện chiếu sáng $<100\text{ lux}$ để thử thách ngưỡng tự tin của mô hình nhận diện.
- **Dòng 6 (Bất đồng do vật ngoại lai / Đè tay - Foreign Object Disagreement)**: Kệ chỉ có 1 lon Coca ($355g$) nhưng trọng lượng ghi nhận $850g$ (do khách hàng tì tay hoặc đặt nhầm vật nặng). Kênh Vision đếm đúng 1 lon, kênh Weight báo sai lệch.

---

## 6. Quy Trình Thu Thập & Gán Nhãn Dữ Liệu (Data Logging Protocol)

Khi tiến hành đo đạc thực nghiệm trên hệ thống phần cứng, quy trình thực hiện cần tuân thủ các bước:
1. **Hiệu chuẩn cân (Tare/Calibration)**: Đảm bảo cân hiển thị $0.0g$ khi kệ trống trước khi bắt đầu phiên thử nghiệm.
2. **Thiết lập trạng thái kệ**: Đặt số lượng hàng hóa và tạo điều kiện che khuất / ánh sáng theo kịch bản thực nghiệm định trước.
3. **Chờ ổn định cơ học (Settling Time)**: Chờ $300 - 500\text{ ms}$ sau khi thao tác đặt/nhấc vật phẩm để sóng rung trên Load cell tắt hẳn trước khi kích hoạt chốt dữ liệu.
4. **Kích hoạt ghi nhận đồng bộ (Trigger Capture)**:
   - Phần mềm điều khiển gửi lệnh chụp camera $\rightarrow$ lưu ảnh vào `data/raw/frames/` $\rightarrow$ lấy `frame_path`.
   - Đọc mẫu cân từ ESP32 qua Serial/MQTT $\rightarrow$ ghi `raw_weight`, tính `filtered_weight`.
   - Ghi nhận `timestamp` hệ thống.
5. **Gán nhãn Ground-Truth**: Nhập trực tiếp các giá trị `gt_count`, `gt_sku`, `occlusion_level`, `lighting_level` vào dòng dữ liệu.

---

## 7. Mã Nguồn Kiểm Tra & Nạp Schema Tham Chiếu (Python)

```python
"""
Multimodal Dataset Loader and Schema Validator for AIoT Smart Shelf.
"""
from typing import Literal
from pydantic import BaseModel, Field, field_validator
import pandas as pd


class MultimodalRecord(BaseModel):
    timestamp: str
    frame_path: str
    raw_weight: float
    filtered_weight: float = Field(ge=0.0, description="Trọng lượng sau lọc không âm")
    gt_count: int = Field(ge=0, description="Số lượng mặt hàng thực tế >= 0")
    gt_sku: str
    occlusion_level: Literal["none", "low", "medium", "high"]
    lighting_level: Literal["normal", "low", "harsh", "uneven"]


SCHEMA_DTYPES = {
    "timestamp": "string",
    "frame_path": "string",
    "raw_weight": "float64",
    "filtered_weight": "float64",
    "gt_count": "int64",
    "gt_sku": "string",
    "occlusion_level": "category",
    "lighting_level": "category",
}


def load_and_validate_dataset(csv_path: str) -> pd.DataFrame:
    """Đọc tệp CSV và kiểm tra tính toàn vẹn theo Schema quy định."""
    df = pd.read_csv(csv_path, dtype=SCHEMA_DTYPES)
    
    # Kiểm tra các cột bắt buộc
    required_cols = list(SCHEMA_DTYPES.keys())
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc trong file CSV: {missing_cols}")
        
    # Kiểm tra giá trị logic
    assert (df["gt_count"] >= 0).all(), "Phát hiện giá trị gt_count âm!"
    assert (df["filtered_weight"] >= 0.0).all(), "Phát hiện giá trị filtered_weight âm!"
    
    return df
```
