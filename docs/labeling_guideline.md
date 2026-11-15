# Hướng Dẫn Gán Nhãn Với Label Studio (S02-C2)

Tài liệu này mô tả cách cài đặt Label Studio, quy tắc gán nhãn và quy trình gán thử **30 ảnh** kệ hàng, xuất nhãn sang định dạng **YOLO**.

## 1. Vì sao chọn Label Studio (không dùng CVAT)

| Tiêu chí | Label Studio | CVAT |
| :--- | :--- | :--- |
| Cài đặt trên máy này | `pip`/`uv`, chạy ngay | Cần Docker + WSL2 (máy chưa cài Docker) |
| Gán box + thuộc tính từng ảnh (occlusion/lighting) | Có (`RectangleLabels` + `Choices`) | Có |
| Xuất YOLO | Có (đã có script chuyển đổi riêng) | Có |

**Quyết định:** dùng Label Studio trong môi trường Python 3.12 riêng, đặt **ngoài repo** tại `%USERPROFILE%\.venvs\labelstudio`, để không xung đột với `.venv` chính (Python 3.14). Nếu sau này cần gán nhãn nhóm nhiều người, có thể chuyển sang CVAT.

> **Vì sao không đặt venv trong repo?** Đường dẫn repo dài (~117 ký tự) cộng với tên file migration dài của Label Studio vượt giới hạn `MAX_PATH = 260` của Windows (máy chưa bật `LongPathsEnabled`). Hậu quả: server crash khi khởi động với `ModuleNotFoundError: ... ml_model_providers.migrations.0006_...`. Đặt venv ở đường dẫn ngắn là cách khắc phục không cần đổi cấu hình hệ thống.

## 2. Cài đặt

```powershell
$venv = "$env:USERPROFILE\.venvs\labelstudio"
uv venv $venv --python 3.12
uv pip install --python "$venv\Scripts\python.exe" -r requirements-labeling.txt
```

Khởi động (dữ liệu lưu tại `data/labeling/ls_data`, đã nằm trong `.gitignore`; có thể đổi vị trí venv bằng biến môi trường `LS_VENV`):

```powershell
powershell -ExecutionPolicy Bypass -File vision\labeling\run_label_studio.ps1
```

Mở <http://localhost:8080>, lần đầu chọn **Sign up** để tạo tài khoản cục bộ (email/mật khẩu tùy ý, chỉ lưu trên máy).

## 3. Quy trình 6 bước

| Bước | Lệnh / thao tác | Kết quả |
| :---: | :--- | :--- |
| 1 | `python -m vision.labeling prepare --source "<thư mục ảnh gốc>" -n 30` | Chép 30 ảnh (chia đều theo nhóm kịch bản) vào `data/labeling/images/` + `manifest.csv` |
| 2 | Label Studio → **Create Project** → tab **Data Import** → kéo thả toàn bộ ảnh trong `data/labeling/images/` | 30 task |
| 3 | **Labeling Setup → Custom template → Code**: dán nội dung [label_config.xml](../vision/labeling/label_config.xml) | Giao diện vẽ box + chọn occlusion/lighting |
| 4 | Gán nhãn từng ảnh (xem quy tắc ở mục 4) rồi **Submit** | 30 annotation |
| 5 | **Export → JSON** (không chọn "JSON-MIN"), lưu file | `export.json` |
| 6 | `python -m vision.labeling convert --export export.json`<br>`python -m vision.labeling validate` | `data/labeling/labels/*.txt`, `image_attributes.csv`, báo cáo kiểm tra |

> Lớp (class) được định nghĩa duy nhất tại [classes.txt](../vision/labeling/classes.txt) (thứ tự dòng = class id của YOLO). Nếu sửa file này, chạy lại `python -m vision.labeling config` rồi dán lại cấu hình mới trước khi gán nhãn.

## 4. Quy tắc gán nhãn (để các lần gán nhất quán)

1. **Một box cho mỗi sản phẩm** (`product`), ôm sát phần nhìn thấy của sản phẩm, gồm cả nắp lon/chai.
2. **Sản phẩm bị che một phần:** vẽ box quanh **phần nhìn thấy được**; không đoán phần bị khuất.
3. **Sản phẩm bị che hoàn toàn / không nhìn thấy:** không vẽ box (số này chính là phần Vision đếm thiếu mà cảm biến cân sẽ bù).
4. **Vật lạ không phải hàng của kệ** (kịch bản Foreign Object): gán `foreign_object`. Lớp này **không** tính vào số lượng tồn kho khi đối chiếu `gt_count`.
5. **Kệ trống:** không vẽ box nào, vẫn chọn occlusion/lighting rồi **Submit** (nhãn rỗng hợp lệ — ảnh âm tính).
6. **Mỗi ảnh chọn thêm:** `occlusion_level` (`none/low/medium/high`) và `lighting_level` (`normal/low/harsh/uneven`) theo [dataset_spec.md](dataset_spec.md) mục 4.

## 5. Kiểm tra chất lượng (`validate`)

Lệnh `python -m vision.labeling validate` kiểm tra:

- Mỗi ảnh có đủ file nhãn; mỗi dòng có đúng 5 trường; class id hợp lệ; tọa độ chuẩn hóa nằm trong ảnh.
- **Đối chiếu số box `product` với `gt_count`** trong `vision/ground_truth_yolo.csv`. Lệch `[CHECK]` không hẳn là lỗi: ở ảnh bị che nặng, số box vẽ được có thể nhỏ hơn `gt_count`. Cần xem lại từng ảnh lệch và ghi chú lý do.

Tiêu chí hoàn thành S02-C2: 30 ảnh có nhãn, `validate` không báo `[ERROR]`, mọi ảnh lệch `gt_count` đều có giải thích.

## 6. Cấu trúc thư mục

```text
data/labeling/
├── images/                 # 30 ảnh đã chọn (không đưa lên git)
├── manifest.csv            # image_id, file_name, source_group, gt_count
├── labels/                 # YOLO txt: <class_id> <x_center> <y_center> <w> <h>
├── image_attributes.csv    # occlusion_level, lighting_level, num_boxes theo ảnh
└── ls_data/                # CSDL Label Studio (không đưa lên git)
```

## 7. Ghi chú kỹ thuật

- Script chuyển đổi tự bỏ tiền tố mã băm 8 ký tự mà Label Studio thêm vào tên ảnh tải lên (`ab12cd34-shelf_01.jpg`).
- Box xoay (rotation ≠ 0) bị bỏ qua vì YOLO chỉ nhận box thẳng; công cụ vẽ box đã tắt nút xoay ảnh.
- Ảnh trong `output/baseline_yolov8n/annotated_frames/` đã bị vẽ box và banner chồng lên, **không dùng được để gán nhãn**; phải dùng ảnh gốc.

