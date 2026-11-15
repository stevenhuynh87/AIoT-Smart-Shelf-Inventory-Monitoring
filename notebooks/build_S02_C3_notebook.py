"""
Generate ``S02_C3_yolov8n_sku110k_colab.ipynb`` (Task S02-C3).

The notebook is generated from this script so that (a) the cell sources are plain, reviewable
Python instead of escaped JSON, and (b) the embedded fallback copies of ``fusion/metrics.py`` and
``vision/sku110k_subset.py`` can always be refreshed from the real source files.

Usage (from the repository root)::

    python notebooks/build_S02_C3_notebook.py

``tests/test_sku110k_notebook.py`` fails if the committed notebook falls out of sync with the
sources, so re-run this script whenever ``fusion/metrics.py`` or ``vision/sku110k_subset.py`` change.
"""

from __future__ import annotations

import base64
import json
import textwrap
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "S02_C3_yolov8n_sku110k_colab.ipynb"

#: ``{key used inside the notebook: source file to embed}``
EMBEDDED_SOURCES = {
    "fusion_metrics_embedded": REPO_ROOT / "fusion" / "metrics.py",
    "sku110k_subset_embedded": REPO_ROOT / "vision" / "sku110k_subset.py",
}

REPO_URL = "https://github.com/stevenhuynh87/AIoT-Smart-Shelf-Inventory-Monitoring.git"
COLAB_URL = (
    "https://colab.research.google.com/github/stevenhuynh87/AIoT-Smart-Shelf-Inventory-Monitoring/"
    "blob/main/notebooks/S02_C3_yolov8n_sku110k_colab.ipynb"
)

COLAB_ONLY_TAG = "colab-only"


def normalise_source_bytes(data: bytes) -> bytes:
    """Normalise line endings so the embedded blob is identical on Windows and Linux checkouts."""
    return data.replace(b"\r\n", b"\n")


def encode_blob(path: Path) -> str:
    """zlib + base64 encode a source file (LF line endings)."""
    raw = normalise_source_bytes(path.read_bytes())
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def _lines(text: str) -> List[str]:
    text = textwrap.dedent(text).strip("\n") + "\n"
    parts = text.splitlines(keepends=True)
    parts[-1] = parts[-1].rstrip("\n")  # Jupyter convention: no trailing newline on the last line
    return parts


def md(text: str) -> Dict[str, Any]:
    """Markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str, tags: Optional[List[str]] = None, form: bool = False) -> Dict[str, Any]:
    """Code cell. ``tags`` are stored in cell metadata; ``form=True`` collapses the code in Colab."""
    metadata: Dict[str, Any] = {}
    if tags:
        metadata["tags"] = list(tags)
    if form:
        metadata["cellView"] = "form"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _lines(text),
    }


def _embedded_cell() -> Dict[str, Any]:
    chunks = []
    for key, path in EMBEDDED_SOURCES.items():
        blob = encode_blob(path)
        wrapped = "\n".join(f'        "{blob[i:i + 96]}"' for i in range(0, len(blob), 96))
        chunks.append(f'    "{key}": (\n{wrapped}\n    ),')
    body = "\n".join(chunks)
    text = (
        "# @title Bản sao nhúng dự phòng (không cần đọc)\n"
        "# Dùng khi `git clone` thất bại hoặc repo chưa có file tương ứng: nén zlib + base64 của\n"
        "# `fusion/metrics.py` (S02-C1) và `vision/sku110k_subset.py` (hàm cắt tập con S02-C3).\n"
        "# Được sinh tự động bởi notebooks/build_S02_C3_notebook.py - đừng sửa tay.\n"
        "EMBEDDED_SOURCES = {\n"
        f"{body}\n"
        "}\n"
        'print("Đã nạp bản sao nhúng:", ", ".join(EMBEDDED_SOURCES))\n'
    )
    return code(text, form=True)


def build_cells() -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []

    cells.append(
        md(
            f"""
            # S02-C3 · Huấn luyện thử YOLOv8n trên một phần nhỏ SKU-110K

            [![Open In Colab]({'https://colab.research.google.com/assets/colab-badge.svg'})]({COLAB_URL})

            **Mục đích:** làm quen trọn vòng **train → val → dự đoán → tính chỉ số** với YOLOv8n trên Google Colab
            (GPU T4 miễn phí). Đây là bài *làm quen quy trình*, **không** nhằm đạt độ chính xác cao nên
            không đặt ngưỡng mAP nào cả.

            | Bước | Nội dung |
            |---|---|
            | 0 | Cấu hình, kiểm tra GPU, lấy mã dự án (`fusion.metrics` của S02-C1) |
            | 1 | Tải SKU-110K bằng loader chính thức `SKU-110K.yaml` của Ultralytics (~13,6 GB) |
            | 2 | Cắt tập con cố định theo `seed`: **400 ảnh train / 100 ảnh val** |
            | 3 | Huấn luyện YOLOv8n (10 epoch, `imgsz=640`, `batch=16`) |
            | 4 | Đánh giá trên 100 ảnh val: mAP (Ultralytics + `map_wrapper`) |
            | 5 | Dự đoán & **chỉ số đếm** (Counting MAE / RMSE / Exact Accuracy) bằng `fusion.metrics` |
            | 6 | Trực quan hoá, lưu & tải `best.pt` + `results.csv` về máy |

            **Cách dùng:** `Runtime → Change runtime type → T4 GPU`, rồi `Runtime → Run all`.
            Không cần Google Drive, không cần API key. Mọi kết quả nằm ở `/content` và **mất khi phiên Colab kết thúc**
            — hãy chạy cell tải về ở cuối notebook.

            **Thời gian tham khảo (ước tính, chưa đo trên Colab):** tải + giải nén + chuyển nhãn SKU-110K khoảng
            10-30 phút (phụ thuộc đường truyền S3); huấn luyện 10 epoch trên 400 ảnh chỉ vài phút trên T4.

            > ⚠️ Ảnh SKU-110K rất dày (~147 vật thể/ảnh). Mặc định `max_det=300` của YOLO sẽ cắt bớt dự đoán và
            > làm lệch số đếm, nên notebook dùng `max_det=1000`.
            """
        )
    )

    cells.append(md("## 0. Cấu hình, môi trường và mã dự án"))
    cells.append(
        code(
            f"""
            # @title Cấu hình thí nghiệm (chỉ sửa ở đây)
            from pathlib import Path

            try:
                from IPython.display import display
            except ImportError:  # chạy ngoài Jupyter
                display = print

            # --- Quy trình train/val (giá trị đã được xác nhận cho S02-C3) ---
            SEED = 0           # seed cố định: cắt tập con + huấn luyện đều tái lập được
            N_TRAIN = 400      # số ảnh train lấy từ train.txt chính thức
            N_VAL = 100        # số ảnh val lấy từ val.txt chính thức
            EPOCHS = 10
            IMGSZ = 640
            BATCH = 16
            WORKERS = 2
            BASE_WEIGHTS = "yolov8n.pt"   # trọng số COCO pretrained (tự tải về)

            # --- Dự đoán / đếm (ảnh dày: nâng max_det) ---
            MAX_DET = 1000                # SKU-110K ~147 vật thể/ảnh, mặc định 300 là thiếu
            NMS_IOU = 0.6
            COUNT_CONF = 0.25             # ngưỡng tin cậy chính để đếm
            CONF_SWEEP = (0.1, 0.25, 0.5) # bảng tham khảo: số đếm thay đổi ra sao theo ngưỡng

            # --- Kiểm tra dữ liệu gốc (SKU-110K đầy đủ: train 8.219 / val 588 / test 2.936) ---
            MIN_FULL_IMAGES = {{"train": 8000, "val": 500}}
            MIN_FREE_GB = 35              # tar.gz + bản giải nén cần nhiều dung lượng đĩa
            SKU_TAR_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"

            # --- Mã dự án (S02-C1) ---
            REPO_URL = "{REPO_URL}"
            REPO_BRANCH = "main"
            REPO_DIR = Path("/content/AIoT-Smart-Shelf-Inventory-Monitoring")

            # --- Thư mục (đĩa tạm của Colab) ---
            DATASETS_DIR = Path("/content/datasets")      # loader Ultralytics tải SKU-110K vào đây
            SUBSET_DIR = Path("/content/sku110k_subset")  # tập con 400/100 ảnh
            RUNS_DIR = Path("/content/runs")
            RUN_NAME = "sku110k_yolov8n"
            REQUIRE_GPU = True            # dừng sớm nếu chưa bật GPU T4

            print(f"Seed={{SEED}} | train={{N_TRAIN}} | val={{N_VAL}} | epochs={{EPOCHS}} | imgsz={{IMGSZ}} | batch={{BATCH}}")
            """
        )
    )
    cells.append(
        code(
            """
            # @title Kiểm tra GPU và cài Ultralytics
            import importlib.util
            import subprocess
            import sys

            if importlib.util.find_spec("ultralytics") is None:
                print("Đang cài ultralytics (khoảng 1 phút)...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

            import torch
            import ultralytics

            if torch.cuda.is_available():
                DEVICE = 0
                print("GPU:", torch.cuda.get_device_name(0))
            elif REQUIRE_GPU:
                raise RuntimeError(
                    "Chưa bật GPU. Vào Runtime -> Change runtime type -> chọn T4 GPU, rồi chạy lại từ đầu "
                    "(hoặc đặt REQUIRE_GPU = False để chạy CPU - rất chậm)."
                )
            else:
                DEVICE = "cpu"
                print("Cảnh báo: đang chạy bằng CPU.")
            print("ultralytics", ultralytics.__version__, "| torch", torch.__version__)
            """
        )
    )
    cells.append(
        md(
            """
            ### Lấy mã dự án (S02-C1)

            Notebook `git clone` repo công khai để dùng `fusion.metrics` (đã push ở commit `d27b162`) và hàm cắt tập
            con `vision.sku110k_subset`. Nếu clone thất bại **hoặc** repo chưa có file tương ứng, notebook tự động dùng
            **bản sao nhúng** ở cell ẩn bên dưới (đảm bảo notebook vẫn chạy được khi không có mạng tới GitHub).
            """
        )
    )
    cells.append(
        code(
            """
            # @title Clone repo GitHub (có dự phòng)
            import subprocess

            repo_ok = False
            repo_commit = "n/a"
            try:
                if (REPO_DIR / ".git").exists():
                    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "-q"], check=False, timeout=120)
                else:
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(REPO_DIR)],
                        check=True,
                        timeout=300,
                    )
                repo_commit = subprocess.run(
                    ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                repo_ok = True
                print(f"Đã lấy repo ở {REPO_DIR} (commit {repo_commit})")
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"Cảnh báo: không clone được repo ({exc}). Sẽ dùng bản sao nhúng.")
            """,
            tags=[COLAB_ONLY_TAG],
        )
    )
    cells.append(_embedded_cell())
    cells.append(
        code(
            """
            # @title Nạp fusion.metrics và vision.sku110k_subset
            import base64
            import importlib
            import importlib.util
            import sys
            import zlib

            FALLBACK_DIR = Path("/content/_embedded_fallback")


            def load_with_fallback(module_name, embedded_key):
                # Ưu tiên module trong repo; nếu thiếu thì nạp bản sao nhúng. Trả về (module, nguồn).
                try:
                    if not repo_ok:
                        raise ImportError("repo không có sẵn")
                    return importlib.import_module(module_name), "repo"
                except Exception as exc:  # noqa: BLE001 - mọi lỗi import đều dẫn tới dự phòng
                    print(f"Dùng bản nhúng cho {module_name} (lý do: {exc})")
                    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
                    path = FALLBACK_DIR / f"{embedded_key}.py"
                    path.write_bytes(zlib.decompress(base64.b64decode(EMBEDDED_SOURCES[embedded_key])))
                    spec = importlib.util.spec_from_file_location(embedded_key, path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[embedded_key] = module
                    spec.loader.exec_module(module)
                    return module, "embedded"


            if repo_ok and str(REPO_DIR) not in sys.path:
                sys.path.insert(0, str(REPO_DIR))

            metrics, metrics_source = load_with_fallback("fusion.metrics", "fusion_metrics_embedded")
            subset, subset_source = load_with_fallback("vision.sku110k_subset", "sku110k_subset_embedded")
            print(f"fusion.metrics: {metrics_source} | sku110k_subset: {subset_source}")
            """
        )
    )

    cells.append(
        md(
            """
            ## 1. Tải SKU-110K bằng loader chính thức của Ultralytics

            Dùng `SKU-110K.yaml` (đi kèm Ultralytics): loader tải `SKU110K_fixed.tar.gz` (~13,6 GB) từ S3 của Trax,
            giải nén và chuyển nhãn sang định dạng YOLO (1 lớp `object`; train 8.219 / val 588 / test 2.936 ảnh).

            - Link S3 đôi khi **chập chờn**. Cell dưới kiểm tra dung lượng đĩa và link trước, và nếu tải lỗi sẽ báo
              rõ nguyên nhân kèm cách thử lại (chạy lại đúng cell này; loader sẽ tải lại từ đầu).
            - Sau khi tải xong, chỉ **400 + 100 ảnh** được dùng; phần còn lại không tham gia huấn luyện.
            """
        )
    )
    cells.append(
        code(
            """
            # @title Tải SKU-110K (có kiểm tra và báo lỗi rõ)
            import shutil
            import urllib.request

            from ultralytics import settings
            from ultralytics.data.utils import check_det_dataset

            DATASETS_DIR.mkdir(parents=True, exist_ok=True)

            free_gb = shutil.disk_usage(DATASETS_DIR).free / 1e9
            print(f"Dung lượng đĩa trống: {free_gb:.1f} GB (cần >= {MIN_FREE_GB} GB)")
            if free_gb < MIN_FREE_GB:
                raise RuntimeError(
                    f"Đĩa chỉ còn {free_gb:.1f} GB. Hãy dùng Runtime -> Disconnect and delete runtime rồi chạy lại."
                )

            try:  # kiểm tra nhanh link S3 (chỉ cảnh báo, không dừng)
                request = urllib.request.Request(SKU_TAR_URL, method="HEAD")
                with urllib.request.urlopen(request, timeout=30) as response:
                    size_gb = int(response.headers.get("Content-Length", 0)) / 1e9
                    print(f"Link S3 phản hồi HTTP {response.status}, kích thước ~{size_gb:.1f} GB")
            except Exception as exc:  # noqa: BLE001
                print(f"Cảnh báo: chưa kiểm tra được link S3 ({exc}). Vẫn thử tải bằng loader Ultralytics...")

            settings.update({"datasets_dir": str(DATASETS_DIR)})
            try:
                data_info = check_det_dataset("SKU-110K.yaml")  # tải + giải nén + chuyển nhãn nếu chưa có
                SKU_ROOT = Path(data_info["path"])
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Tải SKU-110K thất bại: " + repr(exc) + "\\n"
                    "Cách xử lý:\\n"
                    "  1) Link S3 của SKU-110K đôi khi chập chờn: đợi vài phút rồi chạy lại cell này.\\n"
                    "  2) Nếu lỗi lặp lại, xoá dữ liệu dở dang rồi chạy lại: shutil.rmtree(DATASETS_DIR, ignore_errors=True).\\n"
                    "  3) Lỗi thiếu thư viện (polars/pandas): chạy lại cell này một lần nữa.\\n"
                    "  4) Hết đĩa: Runtime -> Disconnect and delete runtime."
                ) from exc
            print("SKU-110K nằm ở:", SKU_ROOT)
            """,
            tags=[COLAB_ONLY_TAG],
        )
    )

    cells.append(
        md(
            """
            ## 2. Kiểm tra dữ liệu và cắt tập con cố định

            `build_subset` lấy ngẫu nhiên **có seed** `N_TRAIN` ảnh từ `train.txt` và `N_VAL` ảnh từ `val.txt`
            (hai tập vốn tách biệt nên không bị rò rỉ), chỉ nhận ảnh có nhãn. Chạy lại với cùng `SEED` luôn cho cùng
            một tập con; danh sách tên ảnh được lưu để tái lập.
            """
        )
    )
    cells.append(
        code(
            """
            # @title Kiểm tra gói dữ liệu đầy đủ và cắt tập con
            import numpy as np

            full_counts = subset.validate_sku110k_root(SKU_ROOT, min_images=MIN_FULL_IMAGES)
            print("Số ảnh trong gói đầy đủ:", full_counts)

            chosen = subset.build_subset(SKU_ROOT, SUBSET_DIR, n_train=N_TRAIN, n_val=N_VAL, seed=SEED)
            subset.verify_subset(SUBSET_DIR, N_TRAIN, N_VAL)
            DATA_YAML = subset.write_data_yaml(SUBSET_DIR)

            for split, images in chosen.items():
                (SUBSET_DIR / f"{split}_images.txt").write_text("\\n".join(p.name for p in images) + "\\n", encoding="utf-8")
                gt_per_image = np.array([subset.count_label_boxes(subset.label_path_for_image(p)) for p in images])
                print(
                    f"{split:5s}: {len(images)} ảnh | vật thể/ảnh: trung bình {gt_per_image.mean():.1f}, "
                    f"nhỏ nhất {gt_per_image.min()}, lớn nhất {gt_per_image.max()}"
                )
            print("data.yaml:", DATA_YAML)
            print(DATA_YAML.read_text(encoding="utf-8"))
            """
        )
    )
    cells.append(
        code(
            """
            # @title Xem thử vài ảnh val kèm nhãn gốc
            import matplotlib.pyplot as plt
            from PIL import Image, ImageDraw

            val_dir = SUBSET_DIR / "images" / "val"
            val_image_paths = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in subset.IMAGE_SUFFIXES)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax, image_path in zip(axes, val_image_paths[:2]):
                image = Image.open(image_path).convert("RGB")
                gt = subset.read_yolo_labels(SUBSET_DIR / "labels" / "val" / f"{image_path.stem}.txt")
                draw = ImageDraw.Draw(image)
                for x1, y1, x2, y2 in subset.yolo_to_xyxy(gt, *image.size):
                    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                ax.imshow(image)
                ax.set_title(f"{image_path.name}: {len(gt)} vật thể (nhãn gốc)")
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ## 3. Huấn luyện YOLOv8n

            Fine-tune từ trọng số COCO `yolov8n.pt` với đúng cấu hình đã chốt (`epochs=10`, `imgsz=640`, `batch=16`,
            `seed=0`). Không tinh chỉnh siêu tham số, không xuất ONNX. Kết quả (`best.pt`, `results.csv`, biểu đồ)
            nằm trong `/content/runs/sku110k_yolov8n/`.
            """
        )
    )
    cells.append(
        code(
            """
            # @title Train
            from ultralytics import YOLO

            model = YOLO(BASE_WEIGHTS)
            train_results = model.train(
                data=str(DATA_YAML),
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH,
                seed=SEED,
                device=DEVICE,
                workers=WORKERS,
                max_det=MAX_DET,   # validation trong lúc train cũng cần max_det lớn
                cache=True,        # 500 ảnh nhỏ -> cache RAM cho nhanh
                project=str(RUNS_DIR),
                name=RUN_NAME,
                exist_ok=True,
                plots=True,
            )

            SAVE_DIR = Path(model.trainer.save_dir)
            BEST_PT = SAVE_DIR / "weights" / "best.pt"
            RESULTS_CSV = SAVE_DIR / "results.csv"
            if not BEST_PT.is_file():
                raise FileNotFoundError(f"Không thấy {BEST_PT}: huấn luyện chưa hoàn tất?")
            print("Kết quả huấn luyện ở:", SAVE_DIR)
            """
        )
    )
    cells.append(
        code(
            """
            # @title Đường cong huấn luyện (từ results.csv)
            import matplotlib.pyplot as plt
            import pandas as pd

            history = pd.read_csv(RESULTS_CSV)
            history.columns = history.columns.str.strip()
            curve_groups = [
                ("Loss (box)", ["train/box_loss", "val/box_loss"]),
                ("mAP (val)", ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]),
            ]
            fig, axes = plt.subplots(1, len(curve_groups), figsize=(12, 4))
            for ax, (title, columns) in zip(axes, curve_groups):
                for column in columns:
                    if column in history:
                        ax.plot(history["epoch"], history[column], marker="o", label=column)
                ax.set_title(title)
                ax.set_xlabel("epoch")
                ax.grid(alpha=0.3)
                ax.legend()
            plt.tight_layout()
            plt.show()
            display(history.tail(3))
            """
        )
    )

    cells.append(
        md(
            """
            ## 4. Đánh giá (val) bằng `best.pt`

            - **mAP của Ultralytics**: `model.val()` (với `conf=0.001`, `max_det=1000`) rồi đưa kết quả qua
              `fusion.metrics.map_wrapper(ultralytics_results=...)` để có định dạng chuẩn của dự án.
            - mAP thấp sau 10 epoch trên 400 ảnh là **bình thường**; mục tiêu chỉ là chạy thông quy trình.
            """
        )
    )
    cells.append(
        code(
            """
            # @title Val bằng Ultralytics + map_wrapper (S02-C1)
            best_model = YOLO(str(BEST_PT))
            val_metrics = best_model.val(
                data=str(DATA_YAML),
                split="val",
                imgsz=IMGSZ,
                batch=BATCH,
                conf=0.001,
                iou=NMS_IOU,
                max_det=MAX_DET,
                device=DEVICE,
                workers=WORKERS,
                plots=False,
                project=str(RUNS_DIR),
                name="val",
                exist_ok=True,
            )
            map_ultralytics = metrics.map_wrapper(ultralytics_results=val_metrics)
            print(
                f"[Ultralytics -> map_wrapper] mAP@0.5 = {map_ultralytics['map50']:.4f} | "
                f"mAP@0.5:0.95 = {map_ultralytics['map50_95']:.4f} | "
                f"precision = {map_ultralytics['precision']:.4f} | recall = {map_ultralytics['recall']:.4f}"
            )
            """
        )
    )

    cells.append(
        md(
            """
            ## 5. Dự đoán và chỉ số đếm (Counting)

            Với mỗi ảnh val: **số đếm thật** = số hộp trong file nhãn; **số đếm dự đoán** = số hộp có
            `conf >= COUNT_CONF`. Mô hình chỉ dự đoán **một lần** ở `conf=0.001` (giữ đủ hộp cho mAP), rồi lọc theo
            ngưỡng để đếm. Sau đó dùng `fusion.metrics`: `counting_mae`, `counting_rmse`, `exact_accuracy`,
            `evaluate_counting` và `map_wrapper` (mAP tự tính bằng NumPy, đối chiếu với số của Ultralytics).

            > Lưu ý khi đọc kết quả: ảnh có ~147 vật thể nên **Exact Accuracy** (đếm đúng tuyệt đối) gần như luôn rất
            > thấp; hãy nhìn MAE cùng với số vật thể trung bình (cột `MAE / trung bình GT`).
            """
        )
    )
    cells.append(
        code(
            """
            # @title Dự đoán trên tập val
            gt_boxes, detections, image_names = [], [], []
            for image_path in val_image_paths:
                result = best_model.predict(
                    source=str(image_path),
                    imgsz=IMGSZ,
                    conf=0.001,
                    iou=NMS_IOU,
                    max_det=MAX_DET,
                    device=DEVICE,
                    verbose=False,
                )[0]
                height, width = result.orig_shape
                detections.append(subset.detections_to_array(result.boxes.xyxy, result.boxes.conf, result.boxes.cls))
                gt = subset.read_yolo_labels(SUBSET_DIR / "labels" / "val" / f"{image_path.stem}.txt")
                gt_boxes.append(subset.yolo_to_xyxy(gt, width, height))
                image_names.append(image_path.name)

            print(f"Đã dự đoán {len(detections)} ảnh | số hộp thô (conf>=0.001) trung bình/ảnh: "
                  f"{np.mean([len(d) for d in detections]):.0f}")
            """
        )
    )
    cells.append(
        code(
            """
            # @title Chỉ số đếm bằng fusion.metrics
            import json

            gt_counts = [len(boxes) for boxes in gt_boxes]
            thresholds = sorted({float(t) for t in CONF_SWEEP} | {float(COUNT_CONF)})
            counts_by_conf = subset.counts_for_thresholds(detections, thresholds)

            rows = []
            for thr in thresholds:
                stats = metrics.evaluate_counting(gt_counts, counts_by_conf[thr])
                rows.append(
                    {
                        "conf": thr,
                        "MAE": stats["mae"],
                        "RMSE": stats["rmse"],
                        "MAE / trung bình GT": stats["mae"] / float(np.mean(gt_counts)),
                        "Exact Accuracy": stats["exact_accuracy"],
                        "undercount_rate": stats["undercount_rate"],
                        "overcount_rate": stats["overcount_rate"],
                    }
                )
            counting_table = pd.DataFrame(rows).set_index("conf")
            display(counting_table.round(4))

            pred_counts = counts_by_conf[float(COUNT_CONF)]
            counting_main = {
                "conf": float(COUNT_CONF),
                "mae": metrics.counting_mae(gt_counts, pred_counts),
                "rmse": metrics.counting_rmse(gt_counts, pred_counts),
                "exact_accuracy": metrics.exact_accuracy(gt_counts, pred_counts),
                "mean_gt_count": float(np.mean(gt_counts)),
                "mean_pred_count": float(np.mean(pred_counts)),
                "n_images": len(gt_counts),
            }
            print(
                f"[conf={COUNT_CONF}] Counting MAE = {counting_main['mae']:.2f} vật thể/ảnh | "
                f"RMSE = {counting_main['rmse']:.2f} | Exact Accuracy = {counting_main['exact_accuracy']:.2%} | "
                f"GT trung bình = {counting_main['mean_gt_count']:.1f}, dự đoán trung bình = {counting_main['mean_pred_count']:.1f}"
            )

            val_counts = pd.DataFrame({"image": image_names, "gt_count": gt_counts, "pred_count": pred_counts})
            val_counts["error"] = val_counts["pred_count"] - val_counts["gt_count"]
            val_counts_csv = SAVE_DIR / "val_counts.csv"
            val_counts.to_csv(val_counts_csv, index=False)
            display(val_counts.head())
            """
        )
    )
    cells.append(
        code(
            """
            # @title mAP tự tính bằng map_wrapper (NumPy) và đối chiếu với Ultralytics
            import time

            started = time.time()
            map_standalone = metrics.map_wrapper(predictions=detections, ground_truths=gt_boxes, box_format="xyxy")
            print(f"map_wrapper (NumPy) chạy trong {time.time() - started:.0f}s")

            comparison = pd.DataFrame(
                {
                    "mAP@0.5": [map_ultralytics["map50"], map_standalone["map50"]],
                    "mAP@0.5:0.95": [map_ultralytics["map50_95"], map_standalone["map50_95"]],
                },
                index=["Ultralytics model.val()", "map_wrapper (NumPy, conf>=0.001)"],
            )
            display(comparison.round(4))
            print("Hai cách nội suy/ghép cặp hộp khác nhau nên số liệu gần nhau chứ không trùng khớp tuyệt đối.")
            """
        )
    )
    cells.append(
        code(
            """
            # @title Trực quan hoá dự đoán (conf = COUNT_CONF) so với nhãn gốc
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, image_path, gt_count in zip(axes, val_image_paths[:3], gt_counts[:3]):
                shown = best_model.predict(
                    source=str(image_path),
                    imgsz=IMGSZ,
                    conf=COUNT_CONF,
                    iou=NMS_IOU,
                    max_det=MAX_DET,
                    device=DEVICE,
                    verbose=False,
                )[0]
                ax.imshow(shown.plot(labels=False, conf=False, line_width=1)[:, :, ::-1])
                ax.set_title(f"{image_path.name}\\nGT = {gt_count} | dự đoán = {len(shown.boxes)}")
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ## 6. Lưu tóm tắt và tải kết quả về máy

            Cell đầu ghi `s02_c3_summary.json` (cấu hình + mAP + chỉ số đếm). Cell thứ hai tải về máy:
            `best.pt`, `results.csv`, `s02_c3_summary.json`, `val_counts.csv`. Nếu trình duyệt chặn nhiều tệp tải
            cùng lúc, hãy cho phép "tải nhiều tệp" cho trang Colab rồi chạy lại cell.
            """
        )
    )
    cells.append(
        code(
            """
            # @title Ghi tóm tắt kết quả (JSON)
            summary = {
                "task": "S02-C3",
                "config": {
                    "seed": SEED, "n_train": N_TRAIN, "n_val": N_VAL, "epochs": EPOCHS, "imgsz": IMGSZ,
                    "batch": BATCH, "max_det": MAX_DET, "nms_iou": NMS_IOU, "count_conf": COUNT_CONF,
                    "base_weights": BASE_WEIGHTS,
                },
                "sources": {"fusion.metrics": metrics_source, "sku110k_subset": subset_source, "repo_commit": repo_commit},
                "map_ultralytics": {k: v for k, v in map_ultralytics.items() if not k.startswith("per_class")},
                "map_standalone": {k: v for k, v in map_standalone.items() if not k.startswith("per_class")},
                "counting": counting_main,
                "counting_by_conf": counting_table.reset_index().to_dict(orient="records"),
            }
            SUMMARY_JSON = SAVE_DIR / "s02_c3_summary.json"
            SUMMARY_JSON.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                encoding="utf-8",
            )
            print("Đã ghi:", SUMMARY_JSON)
            """
        )
    )
    cells.append(
        code(
            """
            # @title Tải best.pt + results.csv về máy
            try:
                from google.colab import files
            except ImportError:
                print("Không chạy trên Colab: lấy tệp trực tiếp ở", SAVE_DIR)
            else:
                for path in (BEST_PT, RESULTS_CSV, SUMMARY_JSON, val_counts_csv):
                    files.download(str(path))
            """,
            tags=[COLAB_ONLY_TAG],
        )
    )

    cells.append(
        md(
            """
            ## Khắc phục sự cố thường gặp

            | Hiện tượng | Cách xử lý |
            |---|---|
            | `RuntimeError: Chưa bật GPU` | `Runtime → Change runtime type → T4 GPU`, chạy lại từ đầu |
            | Tải SKU-110K lỗi / treo (link S3 chập chờn) | Đợi vài phút, chạy lại cell tải; nếu lặp lại thì xoá `DATASETS_DIR` rồi chạy lại |
            | `CUDA out of memory` | Giảm `BATCH` (8 hoặc 4) ở cell cấu hình |
            | Số đếm dự đoán thấp bất thường | Kiểm tra `MAX_DET` (phải ~1000) và bảng chỉ số theo `conf` |
            | Mất kết quả sau khi phiên kết thúc | Chạy cell tải về ngay sau khi train xong |
            | `git clone` lỗi | Bình thường: notebook tự dùng bản sao nhúng (xem dòng `fusion.metrics: embedded`) |

            ## Bước tiếp theo (ngoài phạm vi S02-C3)
            Tinh chỉnh siêu tham số, fine-tune trên dữ liệu kệ hàng tự gán nhãn (S02-C2) và xuất ONNX **không** nằm
            trong notebook này.
            """
        )
    )
    return cells


def build_notebook() -> Dict[str, Any]:
    """Assemble the full notebook dictionary (nbformat 4)."""
    return {
        "cells": build_cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "name": "S02_C3_yolov8n_sku110k_colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {NOTEBOOK_PATH} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
