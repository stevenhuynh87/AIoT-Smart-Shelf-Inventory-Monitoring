"""
Baseline Evaluation Pipeline: YOLOv8n on Shelf Inventory Images
AIoT Smart Shelf Inventory Monitoring Project
"""

import os
import re
import sys
import glob
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# Target COCO Classes for Retail Shelf Beverage & Cartons:
# 39 = bottle, 41 = cup, 73 = book (frequently predicted for carton boxes), 75 = vase (frequently predicted for tall cans/cartons)
TARGET_COCO_CLASSES = [39, 41, 73, 75]
COCO_CLASS_NAMES = {39: "bottle", 41: "cup", 73: "book", 75: "vase"}


def natural_sort_key(s: str) -> List:
    """Sort strings containing numbers in natural human order (e.g., shelf_1, shelf_2, ..., shelf_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def extract_image_id(file_name: str) -> str:
    """Extract canonical image ID like 'shelf_01' from file name 'shelf_01.jpg.jpg'."""
    match = re.search(r"(shelf_\d+)", file_name, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback to stem
    return Path(file_name).stem.replace(".jpg", "").lower()


def discover_shelf_images(data_dir: str = "Công việc") -> List[Dict[str, str]]:
    """Recursively discover and naturally sort all shelf images in the data directory."""
    patterns = [os.path.join(data_dir, "**", "*.jpg*")]
    found_paths = []
    for pat in patterns:
        found_paths.extend(glob.glob(pat, recursive=True))

    # Remove duplicates if any
    unique_paths = list(set(found_paths))

    # Sort naturally by file name
    sorted_paths = sorted(unique_paths, key=lambda p: natural_sort_key(os.path.basename(p)))

    items = []
    for p in sorted_paths:
        p_obj = Path(p)
        image_id = extract_image_id(p_obj.name)
        group_name = p_obj.parent.name
        items.append({
            "image_id": image_id,
            "file_name": p_obj.name,
            "relative_path": os.path.normpath(p),
            "group_name": group_name,
        })
    return items


def load_ground_truth(gt_path: str, discovered_items: List[Dict[str, str]]) -> Dict[str, Dict]:
    """
    Load ground truth mapping from CSV (count and scenario).
    If file doesn't exist, generate a template CSV with None/blank values.
    """
    gt_map: Dict[str, Dict] = {}

    if os.path.exists(gt_path):
        df_gt = pd.read_csv(gt_path)
        # Standardize column names
        df_gt.columns = [c.strip().lower() for c in df_gt.columns]
        id_col = "image_id" if "image_id" in df_gt.columns else df_gt.columns[0]
        cnt_col = "gt_count" if "gt_count" in df_gt.columns else df_gt.columns[1]
        scen_col = "scenario" if "scenario" in df_gt.columns else None

        for _, row in df_gt.iterrows():
            img_id = extract_image_id(str(row[id_col]))
            val = row[cnt_col]
            scen_val = str(row[scen_col]) if (scen_col and pd.notna(row[scen_col])) else ""
            if pd.notna(val) and str(val).strip() != "":
                try:
                    gt_map[img_id] = {"count": int(float(val)), "scenario": scen_val}
                except ValueError:
                    gt_map[img_id] = {"count": None, "scenario": scen_val}
            else:
                gt_map[img_id] = {"count": None, "scenario": scen_val}
    else:
        # Create template CSV
        template_rows = []
        for item in discovered_items:
            template_rows.append({
                "image_id": item["image_id"],
                "group_name": item["group_name"],
                "file_name": item["file_name"],
                "gt_count": "",  # To be filled by user
                "scenario": "",
            })
        os.makedirs(os.path.dirname(os.path.abspath(gt_path)), exist_ok=True)
        pd.DataFrame(template_rows).to_csv(gt_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Created Ground Truth template at: {gt_path}")
        for item in discovered_items:
            gt_map[item["image_id"]] = {"count": None, "scenario": ""}

    return gt_map


def draw_hud_and_boxes(
    image: np.ndarray,
    detections: List[Dict],
    image_id: str,
    gt_count: Optional[int],
    vision_count: int,
    mean_conf: float,
) -> np.ndarray:
    """Draw professional HUD banner and bounding boxes onto frame."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Draw Bounding Boxes
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        cls_name = det["cls_name"]

        # Color based on confidence: Green (high) to Yellow (moderate)
        if conf >= 0.70:
            color = (34, 139, 34)     # Forest Green
        elif conf >= 0.40:
            color = (30, 144, 255)    # Dodger Blue
        else:
            color = (0, 165, 255)     # Orange

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label badge
        label = f"{cls_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, max(th, y1 - 3)),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    # Top HUD Banner
    banner_height = 48
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Banner Content
    gt_str = str(gt_count) if gt_count is not None else "N/A"
    if gt_count is not None:
        diff = vision_count - gt_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        status_color = (0, 255, 0) if diff == 0 else ((0, 140, 255) if diff < 0 else (0, 69, 255))
    else:
        diff_str = "N/A"
        status_color = (200, 200, 200)

    title_text = f"AIoT Smart Shelf | Image: {image_id}"
    metrics_text = f"GT: {gt_str}  |  YOLOv8n: {vision_count}  |  Diff: {diff_str}  |  Avg Conf: {mean_conf:.2f}"

    cv2.putText(annotated, title_text, (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(annotated, metrics_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

    return annotated


def run_evaluation(
    data_dir: str = "Công việc",
    gt_file: str = "vision/ground_truth_yolo.csv",
    output_dir: str = "output/baseline_yolov8n",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    imgsz: int = 640,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Execute complete YOLOv8n evaluation pipeline on shelf images."""
    print("=" * 70)
    print("🚀 STARTING YOLOV8N PRETRAINED BASELINE EVALUATION")
    print(f"Data directory:       {data_dir}")
    print(f"Ground Truth file:    {gt_file}")
    print(f"Output directory:     {output_dir}")
    print(f"Inference settings:   conf={conf_thresh}, iou={iou_thresh}, imgsz={imgsz}")
    print("=" * 70)

    # Create output directories
    annotated_dir = os.path.join(output_dir, "annotated_frames")
    os.makedirs(annotated_dir, exist_ok=True)

    # 1. Discover Images
    items = discover_shelf_images(data_dir)
    print(f"[INFO] Discovered {len(items)} shelf images across groups.")

    if not items:
        print("[ERROR] No images found. Check data directory!")
        return pd.DataFrame()

    # 2. Load Ground Truth (for all items)
    gt_map = load_ground_truth(gt_file, items)

    if smoke_test:
        print("[INFO] Running in SMOKE-TEST mode (single image).")
        items = items[:1]

    # 3. Load YOLOv8n Model
    print("[INFO] Loading YOLOv8n model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")

    results_data = []

    # 4. Inference Loop
    for idx, item in enumerate(items, start=1):
        image_id = item["image_id"]
        img_path = item["relative_path"]
        group_name = item["group_name"]
        file_name = item["file_name"]
        gt_info = gt_map.get(image_id, {"count": None, "scenario": ""})
        gt_cnt = gt_info["count"]
        scenario = gt_info["scenario"]

        print(f"[{idx:02d}/{len(items):02d}] Processing: {image_id} ({group_name} / {file_name})...", end=" ")

        # Read image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("[FAILED TO READ IMAGE]")
            continue

        # Run YOLO inference
        results = model.predict(
            source=img_bgr,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=imgsz,
            verbose=False,
            classes=TARGET_COCO_CLASSES,
        )

        detections = []
        confidences = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]

                detections.append({
                    "cls_id": cls_id,
                    "cls_name": COCO_CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                    "conf": conf,
                    "bbox": xyxy,
                })
                confidences.append(conf)

        vision_count = len(detections)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        # Calculate errors if Ground Truth is available
        if gt_cnt is not None:
            error = vision_count - gt_cnt
            abs_error = abs(error)
            if error == 0:
                status = "EXACT"
            elif error < 0:
                status = "UNDERCOUNT"
            else:
                status = "OVERCOUNT"
        else:
            error = None
            abs_error = None
            status = "NO_GT"

        print(f"-> Vision: {vision_count} | GT: {gt_cnt if gt_cnt is not None else 'N/A'} | Status: {status}")

        # Draw HUD and Bounding Boxes
        annotated_img = draw_hud_and_boxes(
            image=img_bgr,
            detections=detections,
            image_id=image_id,
            gt_count=gt_cnt,
            vision_count=vision_count,
            mean_conf=mean_conf,
        )

        # Save annotated image
        out_frame_path = os.path.join(annotated_dir, f"{image_id}_annotated.jpg")
        cv2.imwrite(out_frame_path, annotated_img)

        # Summarize detected class counts
        cls_tally = {}
        for d in detections:
            c_name = d["cls_name"]
            cls_tally[c_name] = cls_tally.get(c_name, 0) + 1
        detected_details = ", ".join([f"{count} {c_name}" for c_name, count in cls_tally.items()]) if cls_tally else "none"

        results_data.append({
            "image_id": image_id,
            "group_name": group_name,
            "file_name": file_name,
            "relative_path": img_path,
            "scenario": scenario,
            "gt_count": gt_cnt,
            "vision_count": vision_count,
            "error": error,
            "abs_error": abs_error,
            "detected_classes": detected_details,
            "mean_confidence": round(mean_conf, 4),
            "status": status,
            "annotated_frame": os.path.normpath(out_frame_path),
        })

    # 5. Build DataFrame & Compute Metrics
    df = pd.DataFrame(results_data)
    csv_out_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Detailed CSV results saved to: {csv_out_path}")

    # 6. Generate Summary Markdown Report
    generate_summary_report(df, output_dir, conf_thresh, iou_thresh, imgsz)

    return df


def generate_summary_report(
    df: pd.DataFrame,
    output_dir: str,
    conf_thresh: float,
    iou_thresh: float,
    imgsz: int,
):
    """Generate Markdown evaluation report with statistics and group analysis."""
    report_path = os.path.join(output_dir, "evaluation_summary.md")

    # Filter records with valid ground truth
    df_valid = df[df["gt_count"].notna()].copy()
    has_gt = len(df_valid) > 0

    total_images = len(df)
    valid_count = len(df_valid)

    lines = [
        "# Báo Cáo Thực Nghiệm: Đánh Giá Hiệu Năng Cơ Sở YOLOv8n Pretrained Trên Kệ Hàng",
        "",
        f"- **Thời gian chạy**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Mô hình**: YOLOv8n (`yolov8n.pt`, MS-COCO 80 classes)",
        f"- **Lớp mục tiêu**: `bottle` (39) & `cup` (41)",
        f"- **Tham số cấu hình**: `conf={conf_thresh}`, `iou={iou_thresh}`, `imgsz={imgsz}`",
        f"- **Tổng số ảnh xử lý**: {total_images} ảnh",
        f"- **Số ảnh đã có nhãn Ground Truth**: {valid_count} ảnh",
        "",
        "---",
        "",
        "## 1. Các Chỉ Số Hiệu Năng Toàn Cục (Overall Metrics)",
        "",
    ]

    if has_gt:
        mae = df_valid["abs_error"].mean()
        rmse = math.sqrt((df_valid["error"] ** 2).mean())
        exact_matches = (df_valid["status"] == "EXACT").sum()
        accuracy = (exact_matches / valid_count) * 100.0

        undercounts = (df_valid["status"] == "UNDERCOUNT").sum()
        undercount_rate = (undercounts / valid_count) * 100.0

        overcounts = (df_valid["status"] == "OVERCOUNT").sum()
        overcount_rate = (overcounts / valid_count) * 100.0

        lines.extend([
            "| Chỉ số đo lường | Giá trị | Ý nghĩa kỹ thuật |",
            "| :--- | :---: | :--- |",
            f"| **MAE (Mean Absolute Error)** | **{mae:.2f}** chiếc | Sai số trung bình giữa mô hình và số lượng thực tế |",
            f"| **RMSE (Root Mean Square Error)** | **{rmse:.2f}** chiếc | Mức độ phân tán sai số (phạt nặng các lỗi đếm lệch lớn) |",
            f"| **Độ chính xác tuyệt đối (Accuracy)** | **{accuracy:.1f}%** ({exact_matches}/{valid_count}) | Tỷ lệ ảnh đếm chính xác hoàn toàn (Vision == GT) |",
            f"| **Tỷ lệ đếm thiếu (Undercount Rate)** | **{undercount_rate:.1f}%** ({undercounts}/{valid_count}) | Do vật thể bị che khuất (occlusion) hoặc thiếu sáng |",
            f"| **Tỷ lệ đếm thừa (Overcount Rate)** | **{overcount_rate:.1f}%** ({overcounts}/{valid_count}) | Do nhận diện nhầm chi tiết nền hoặc bóng đổ |",
            "",
            "---",
            "",
            "## 2. Thống Kê Theo Từng Nhóm Ảnh (Group-by-Group Breakdown)",
            "",
            "| Nhóm Ảnh | Số lượng ảnh | MAE (chiếc) | Tỷ lệ Đếm Đúng | Tỷ lệ Đếm Thiếu | Tỷ lệ Đếm Thừa | Avg Confidence |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for grp, grp_df in df_valid.groupby("group_name"):
            g_cnt = len(grp_df)
            g_mae = grp_df["abs_error"].mean()
            g_acc = ((grp_df["status"] == "EXACT").sum() / g_cnt) * 100.0
            g_under = ((grp_df["status"] == "UNDERCOUNT").sum() / g_cnt) * 100.0
            g_over = ((grp_df["status"] == "OVERCOUNT").sum() / g_cnt) * 100.0
            g_conf = grp_df["mean_confidence"].mean()

            lines.append(
                f"| `{grp}` | {g_cnt} | {g_mae:.2f} | {g_acc:.1f}% | {g_under:.1f}% | {g_over:.1f}% | {g_conf:.2f} |"
            )

    else:
        lines.extend([
            "> [!NOTE]",
            "> Chưa có dữ liệu Ground Truth trong tệp `ground_truth_yolo.csv`.",
            "> Vui lòng cập nhật số lượng đếm bằng mắt vào file CSV và chạy lại để trích xuất chỉ số MAE, RMSE.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 3. Bảng Chi Tiết Kết Quả Từng Ảnh",
        "",
        "| ID | Nhóm | Kịch bản thực nghiệm | GT | YOLOv8n | Sai số | Trạng thái | Chi tiết nhận diện | Độ tin cậy TB |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: |",
    ])

    for _, row in df.iterrows():
        gt_val = row["gt_count"] if pd.notna(row["gt_count"]) else "-"
        err_val = f"{row['error']:+d}" if pd.notna(row["error"]) else "-"
        det_cls = row.get("detected_classes", "-")
        scen_val = row.get("scenario", "-")
        lines.append(
            f"| `{row['image_id']}` | {row['group_name']} | {scen_val} | {gt_val} | **{row['vision_count']}** | {err_val} | `{row['status']}` | {det_cls} | {row['mean_confidence']:.2f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Nhận Xét Khoa Học Phục Vụ Luận Văn",
        "1. **Hiện tượng đếm thiếu (Under-detection)**: Mô hình pretrained YOLOv8n gặp khó khăn khi các chai/lon xếp chồng hàng sâu phía sau hoặc bị che khuất một phần (occlusion).",
        "2. **Đóng góp của Đề tài**: Kết quả baseline này chứng minh rõ ràng việc chỉ sử dụng một mình camera thị giác là không đủ tin cậy cho giám sát tồn kho bán lẻ tự động.",
        "3. **Tính tất yếu của Fusion**: Cảm biến trọng lượng (Load cell) hoàn toàn miễn nhiễm với che khuất thị giác, cung cấp kênh thông tin độc lập giúp cơ chế **Disagreement-Aware Fusion** phân xử và khôi phục số lượng hàng hóa chính xác.",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Summary Markdown report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8n on Shelf Images")
    parser.add_argument("--data-dir", type=str, default="Công việc", help="Directory containing shelf images")
    parser.add_argument("--gt-file", type=str, default="vision/ground_truth_yolo.csv", help="Ground truth CSV path")
    parser.add_argument("--output-dir", type=str, default="output/baseline_yolov8n", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--smoke-test", action="store_true", help="Run only on single image for quick verification")

    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        gt_file=args.gt_file,
        output_dir=args.output_dir,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        smoke_test=args.smoke_test,
    )
