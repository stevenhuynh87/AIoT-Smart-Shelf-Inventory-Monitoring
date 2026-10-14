# Disagreement-Aware Vision and Weight Sensor Fusion for AIoT Smart Shelf Inventory Monitoring

## Thesis Brief

**Research Area:** AI + IoT + Computer Vision  
**Student Level:** Undergraduate  
**Preferred Duration:** 6–9 months  
**Preferred Direction:** Product-oriented  
**Computing Resources:** To be determined  

---

## 1. Concept / Problem

Trong các cửa hàng bán lẻ, minimart, pharmacy hoặc kho hàng quy mô nhỏ, việc theo dõi tồn kho trên kệ thường cần xác định:

- sản phẩm nào sắp hết;
- kệ nào đang trống;
- số lượng sản phẩm còn lại;
- sản phẩm có được đặt sai vị trí hay không;
- thời điểm cần bổ sung hàng;
- sự thay đổi tồn kho theo thời gian.

Một hệ thống chỉ sử dụng camera và Computer Vision có thể phát hiện và đếm sản phẩm, nhưng thường gặp khó khăn trong các tình huống:

- sản phẩm bị che khuất;
- nhiều sản phẩm chồng lấn;
- mật độ sản phẩm cao;
- các sản phẩm có hình dạng hoặc bao bì tương tự nhau;
- góc nhìn camera thay đổi;
- điều kiện ánh sáng không ổn định;
- detection confidence thấp;
- detector đếm thiếu hoặc đếm dư sản phẩm.

Trong khi đó, một hệ thống chỉ sử dụng load cell có thể phát hiện sự thay đổi trọng lượng nhưng không biết chính xác sản phẩm nào đã được lấy ra hoặc bổ sung.

Vì vậy, đề tài đề xuất xây dựng một **AIoT Smart Shelf** kết hợp:

- camera;
- Computer Vision;
- load cell;
- edge computing;
- IoT communication;
- sensor fusion;
- dashboard giám sát tồn kho.

### Product Concept

```text
Smart Shelf Node
=
Camera
+ Load Cell
+ Edge AI
+ Wi-Fi / MQTT
+ Dashboard
```

Mục tiêu sản phẩm là xây dựng một hệ thống smart shelf chi phí thấp có khả năng:

1. nhận dạng hoặc phát hiện sản phẩm;
2. đếm số lượng sản phẩm trên kệ;
3. đo trọng lượng thực tế;
4. kết hợp thông tin camera và load cell;
5. phát hiện sự không nhất quán giữa hai nguồn;
6. tự động kích hoạt quá trình kiểm tra lại khi cần;
7. cập nhật trạng thái tồn kho lên dashboard;
8. cảnh báo low-stock hoặc stock-out.

---

## 2. Research Gap

### 2.1 Limitation of Vision-Only Monitoring

Một hệ thống dựa hoàn toàn vào Computer Vision có thể gặp sai số khi:

```text
Occlusion
    ↓
Missed Detection
    ↓
Incorrect Product Count
```

Ví dụ:

```text
Ground Truth = 8 products
Vision Count = 5 products
```

Nguyên nhân có thể đến từ:

- overlapping objects;
- poor lighting;
- small objects;
- partially visible products;
- detector confidence threshold;
- camera perspective.

---

### 2.2 Limitation of Weight-Only Monitoring

Load cell có thể đo tổng trọng lượng:

\[
W_t
\]

tại thời điểm \(t\).

Nếu trọng lượng trung bình của một sản phẩm là:

\[
w_p
\]

thì số lượng sản phẩm có thể được ước lượng đơn giản:

\[
\hat{N}_{weight}
=
\frac{W_t}{w_p}
\]

Tuy nhiên phương pháp này có hạn chế:

- không xác định được loại sản phẩm bằng hình ảnh;
- sai số khi trọng lượng từng sản phẩm không hoàn toàn giống nhau;
- không xử lý tốt nhiều loại sản phẩm cùng nằm trên một khu vực;
- không nhận biết được sản phẩm đặt sai vị trí.

---

### 2.3 Research Opportunity

Thay vì sử dụng độc lập:

```text
Vision
```

hoặc:

```text
Weight Sensor
```

đề tài sử dụng:

```text
Vision + Weight Sensor + Temporal Context
```

để tạo một hệ thống inventory monitoring đáng tin cậy hơn.

Điểm nghiên cứu chính không chỉ là fusion hai nguồn dữ liệu, mà là xây dựng một:

> **Disagreement-Aware Sensor Fusion Mechanism**

Hệ thống chủ động phát hiện trường hợp camera và load cell đưa ra hai kết quả không nhất quán.

Ví dụ:

```text
Vision Estimate:
5 products

Weight Estimate:
≈ 8 products

        ↓

Large Disagreement

        ↓

Trigger Verification

        ↓

High-Resolution Re-Inference
or
Temporal Re-Check
```

Thay vì luôn chạy detector ở cấu hình mạnh nhất, hệ thống chỉ tăng mức xử lý khi phát hiện uncertainty hoặc disagreement.

Điều này giúp cân bằng:

\[
\text{Accuracy}
\leftrightarrow
\text{Latency}
\leftrightarrow
\text{Energy}
\leftrightarrow
\text{Compute Cost}
\]

---

## 3. Research Questions

Đề tài có thể tập trung vào ba research questions chính.

### RQ1 — Sensor Fusion

> **Can weight information reduce inventory counting errors from a vision-only smart shelf monitoring system?**

Mục tiêu là kiểm tra liệu thông tin từ load cell có giúp giảm các lỗi đếm do:

- occlusion;
- missed detections;
- overlapping products;
- lighting variations.

---

### RQ2 — Disagreement-Aware Verification

> **Can disagreement between visual and weight-based inventory estimates be used to trigger an effective re-analysis mechanism?**

Research question này tập trung vào việc xây dựng một cơ chế phát hiện:

\[
D_t
=
\left|
\hat{N}_{vision,t}
-
\hat{N}_{weight,t}
\right|
\]

trong đó:

- \(\hat{N}_{vision,t}\): số lượng sản phẩm ước lượng bằng camera;
- \(\hat{N}_{weight,t}\): số lượng sản phẩm ước lượng từ load cell;
- \(D_t\): mức độ disagreement.

Nếu:

\[
D_t > \tau_D
\]

với \(\tau_D\) là disagreement threshold, hệ thống kích hoạt verification.

---

### RQ3 — Event-Triggered Edge Vision

> **Can event-triggered visual inference reduce computational cost while maintaining reliable inventory monitoring?**

Thay vì xử lý camera liên tục:

```text
Camera
↓
Inference
↓
Inference
↓
Inference
↓
Inference
```

hệ thống có thể chuyển sang:

```text
Weight Change / Shelf Event
          ↓
Trigger Camera
          ↓
Run Detection
          ↓
Update Inventory
```

Cách tiếp cận này có tiềm năng giảm:

- số frame cần xử lý;
- CPU/GPU utilization;
- energy consumption;
- thermal load;
- unnecessary inference.

---

## 4. Proposed Solution

Kiến trúc tổng quát:

```text
Camera
  ↓
Product Detection
  ↓
Visual Count
  │
  │
  ├─────────────────────────────┐
  │                             │
  │                       Fusion Engine
  │                             │
  │                             ↓
Load Cell                  Inventory State
  ↓                             ↓
Weight Estimation        Agreement Analysis
                                ↓
                       ┌────────┴────────┐
                       │                 │
                    Agreement        Disagreement
                       │                 │
                       ↓                 ↓
                 Accept Result      Verification
                                         ↓
                              High-Resolution Detection
                                         ↓
                                  Updated Estimate
```

---

## 5. Main Technical Contribution

Contribution trung tâm của thesis là:

# Disagreement-Aware Sensor Fusion

Thay vì chỉ thực hiện:

\[
Final =
\alpha Vision
+
\beta Weight
\]

hệ thống sử dụng mức độ confidence và disagreement để đưa ra quyết định thích nghi.

Một formulation đơn giản có thể được xây dựng như sau.

### Visual Estimate

\[
\hat{N}_v
\]

với confidence:

\[
C_v \in [0,1]
\]

### Weight Estimate

\[
\hat{N}_w
=
\operatorname{round}
\left(
\frac{W}{w_p}
\right)
\]

### Disagreement

\[
D
=
|\hat{N}_v-\hat{N}_w|
\]

### Example Decision Policy

Nếu:

\[
D = 0
\]

và:

\[
C_v > \tau_c
\]

thì:

```text
Accept Vision Result
```

Nếu:

\[
D > \tau_D
\]

hoặc:

\[
C_v < \tau_c
\]

thì:

```text
Trigger Re-Inference
```

Verification có thể bao gồm:

- sử dụng ảnh resolution cao hơn;
- giảm confidence threshold;
- sử dụng model lớn hơn;
- phân tích nhiều frame liên tiếp;
- cập nhật lại weight measurement;
- yêu cầu sensor measurement ổn định trước khi quyết định.

---

## 6. Proposed Adaptive Inference Strategy

Một cách triển khai phù hợp undergraduate thesis là sử dụng hai inference modes.

### Mode 1 — Lightweight Monitoring

```text
Low Resolution
+
Lightweight Detector
```

Ví dụ:

```text
640 × 640
+
YOLO Nano Model
```

### Mode 2 — Verification

Khi disagreement xảy ra:

```text
Higher Resolution
+
Stronger Detection Configuration
+
Temporal Verification
```

Pipeline:

```text
Normal State
    ↓
Lightweight Detection
    ↓
Compare with Weight
    ↓
Agreement?
   ↙      ↘
 Yes      No
 ↓        ↓
Accept   Verification Mode
          ↓
      Re-Inference
          ↓
      Final Result
```

Cơ chế này giúp sản phẩm có yếu tố **adaptive Edge AI**, thay vì chỉ chạy detector liên tục.

---

## 7. System / Model Pipeline

Kiến trúc đầy đủ đề xuất:

```text
                    SMART SHELF
                         │
          ┌──────────────┴───────────────┐
          │                              │
       Camera                         Load Cell
          │                              │
          ↓                              ↓
   Image Acquisition              Weight Measurement
          │                              │
          ↓                              ↓
 Lightweight Detector            Weight Filtering
          │                              │
          ↓                              ↓
  Product Detection              Weight-Based Count
          │                              │
          ↓                              │
    Visual Count                        │
          │                              │
          └──────────────┬───────────────┘
                         ↓
                 Sensor Fusion Module
                         ↓
                 Disagreement Score
                         ↓
                ┌────────┴─────────┐
                │                  │
             Agreement         Disagreement
                │                  │
                ↓                  ↓
          Accept Estimate     Verification Mode
                                   ↓
                           High-Quality Inference
                                   ↓
                              Final Count
                                   ↓
                          Inventory Manager
                                   ↓
                         MQTT / REST API
                                   ↓
                              Dashboard
                 ┌─────────────────┼─────────────────┐
                 ↓                 ↓                 ↓
             Low Stock        Stock-Out          Mismatch
              Alert             Alert              Alert
```

---

## 8. Suggested AI Models

Mục tiêu của thesis không nên là phát triển object detector mới.

Có thể sử dụng một detector phổ biến làm nền, ví dụ:

- YOLO lightweight variant;
- RT-DETR lightweight configuration;
- MobileNet-SSD;
- EfficientDet-lite.

Đối với undergraduate thesis, ưu tiên:

```text
Existing Detector
+
Sensor Fusion
+
Adaptive Decision
```

thay vì:

```text
New Detector Architecture
+
New Backbone
+
New Loss
+
New IoT System
```

để tránh scope quá lớn.

---

## 9. IoT Architecture

Một kiến trúc phần cứng khả thi:

```text
Load Cell
    ↓
HX711
    ↓
ESP32
    ↓
MQTT
    ↓
Edge Computer
```

Camera kết nối với edge device:

```text
USB Camera / CSI Camera / IP Camera
             ↓
        Edge Computer
```

Toàn bộ:

```text
                 Camera
                   ↓
             Edge Computer
                   ↑
                   │
Load Cell → HX711 → ESP32
                   │
                   ↓
                  MQTT
                   ↓
            Inventory Service
                   ↓
               Dashboard
```

---

## 10. Dataset / Data Collection

### 10.1 Public Dataset

Có thể sử dụng public retail datasets để pre-train hoặc benchmark object detection.

Một lựa chọn phù hợp là:

### SKU-110K

Dataset này được thiết kế cho bài toán object detection trong các kệ bán lẻ có mật độ sản phẩm cao.

Nó đặc biệt hữu ích cho các tình huống:

- dense shelves;
- overlapping products;
- small objects;
- crowded retail scenes.

Reference:

- SKU-110K dataset documentation:  
  https://github.com/ultralytics/ultralytics/blob/main/docs/en/datasets/detect/sku-110k.md

---

## 10.2 Custom Smart Shelf Dataset

Đối với phần sensor fusion, nên xây dựng một mini dataset riêng.

Một sample có thể bao gồm:

```text
timestamp
image
product_id
ground_truth_count
vision_count
vision_confidence
weight
weight_estimated_count
lighting_condition
shelf_state
```

Ví dụ:

| Timestamp | Product | Ground Truth | Vision Count | Weight Count | Light | Occlusion |
|---|---|---:|---:|---:|---|---|
| T1 | Product A | 10 | 10 | 10 | Normal | Low |
| T2 | Product A | 8 | 6 | 8 | Normal | High |
| T3 | Product A | 5 | 4 | 5 | Low | Medium |

Dataset không cần quá lớn để xây dựng MVP.

Đối với product-oriented undergraduate thesis, có thể bắt đầu với:

- 5–10 product types;
- nhiều mức inventory;
- nhiều lighting conditions;
- nhiều mức occlusion;
- nhiều góc camera.

---

## 11. Data Collection Scenarios

Nên thiết kế data collection có kiểm soát.

### Scenario A — Normal Shelf

- good lighting;
- products aligned;
- low occlusion.

### Scenario B — Dense Shelf

- nhiều sản phẩm;
- overlapping objects;
- partial occlusion.

### Scenario C — Low Light

- illumination giảm;
- shadows;
- camera noise.

### Scenario D — Product Removal

```text
10 products
↓
Customer removes 1 product
↓
9 products
```

Load cell và camera cùng ghi nhận sự kiện.

### Scenario E — Multiple Product Removal

```text
10
↓
7
```

### Scenario F — Sensor Disagreement

Ví dụ cố tình tạo tình huống:

```text
Vision Count = 6
Weight Count = 8
Ground Truth = 8
```

để đánh giá verification mechanism.

---

## 12. Baseline Methods

Thesis nên có ít nhất bốn baseline.

### Baseline 1 — Vision-Only

```text
Camera
↓
Detector
↓
Product Count
```

---

### Baseline 2 — Weight-Only

```text
Load Cell
↓
Weight
↓
Estimated Count
```

---

### Baseline 3 — Fixed Sensor Fusion

Ví dụ:

\[
\hat{N}
=
\alpha\hat{N}_v
+
(1-\alpha)\hat{N}_w
\]

với \(\alpha\) cố định.

---

### Baseline 4 — Proposed Disagreement-Aware Fusion

```text
Vision
+
Weight
+
Confidence
+
Disagreement Analysis
+
Adaptive Verification
```

---

## 13. Evaluation Metrics

Không nên chỉ đánh giá mAP.

Một product thesis AIoT cần đánh giá cả:

```text
AI Accuracy
+
Inventory Accuracy
+
System Performance
+
Product Reliability
```

---

### 13.1 Object Detection Metrics

- Precision;
- Recall;
- F1-score;
- mAP@0.5;
- mAP@0.5:0.95.

---

### 13.2 Inventory Counting Metrics

Có thể sử dụng Mean Absolute Error:

\[
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|
y_i-\hat{y}_i
|
\]

trong đó:

- \(y_i\): ground-truth product count;
- \(\hat{y}_i\): predicted count.

Ngoài ra có thể đánh giá:

- Count Accuracy;
- Mean Percentage Counting Error;
- Exact Count Accuracy.

---

### 13.3 Inventory Event Metrics

- stock-out detection accuracy;
- low-stock detection accuracy;
- event Precision;
- event Recall;
- false alert rate.

Một metric rất hữu ích cho product:

\[
False\ Alerts / Hour
\]

---

### 13.4 Edge AI Metrics

- inference latency;
- FPS;
- CPU utilization;
- GPU/NPU utilization;
- memory consumption;
- model size.

---

### 13.5 IoT Metrics

- MQTT latency;
- sensor-to-dashboard latency;
- packet loss nếu phù hợp;
- sensor sampling rate.

---

### 13.6 Energy / Efficiency Metrics

Nếu có phần cứng hỗ trợ:

- average power consumption;
- energy per inference;
- number of processed frames/hour;
- camera activation rate.

Có thể so sánh:

```text
Continuous Inference
vs
Event-Triggered Inference
```

---

## 14. Experimental Plan

## Experiment 1 — Baseline Comparison

So sánh:

```text
Vision-Only
vs
Weight-Only
vs
Fixed Fusion
vs
Proposed Disagreement-Aware Fusion
```

Metrics:

- counting MAE;
- exact count accuracy;
- low-stock F1;
- false alerts.

---

## Experiment 2 — Ablation Study

Đánh giá lần lượt:

```text
Camera Only
```

```text
Camera + Weight
```

```text
Camera + Weight + Confidence
```

```text
Camera + Weight + Confidence + Verification
```

Mục tiêu là xác định contribution thực sự của từng module.

---

## Experiment 3 — Occlusion Robustness

Tạo nhiều mức occlusion:

```text
0%
20%
40%
60%
```

hoặc định nghĩa:

- low;
- medium;
- high.

Đánh giá:

\[
Counting\ Error
\]

theo mức occlusion.

Kỳ vọng không nên được khẳng định trước; experiment sẽ kiểm tra liệu sensor fusion có giảm degradation so với vision-only hay không.

---

## Experiment 4 — Lighting Robustness

Các điều kiện:

- bright;
- normal;
- dim;
- uneven lighting.

So sánh vision-only với multimodal fusion.

---

## Experiment 5 — Disagreement Threshold

Thử nhiều giá trị:

\[
\tau_D
\]

Ví dụ:

\[
0,\;1,\;2,\;3
\]

và đánh giá trade-off giữa:

- verification frequency;
- counting accuracy;
- latency;
- computational cost.

---

## Experiment 6 — Event-Triggered Inference

So sánh:

### Continuous

```text
Camera
↓
Inference continuously
```

với:

### Event-Triggered

```text
Weight Change
↓
Trigger Camera
↓
Inference
```

Metrics:

- processed frames/hour;
- CPU/GPU usage;
- energy;
- missed inventory events;
- latency.

---

## Experiment 7 — Real-Time Product Test

Mô phỏng workflow thực tế:

```text
Initial Shelf
    ↓
Customer Removes Product
    ↓
Weight Changes
    ↓
Camera Triggered
    ↓
Inventory Updated
    ↓
Dashboard Updated
```

Đo tổng latency:

\[
T_{system}
=
T_{sensor}
+
T_{communication}
+
T_{vision}
+
T_{fusion}
+
T_{dashboard}
\]

---

## 15. Expected Outcomes

Không giả định proposed method chắc chắn tốt hơn baseline.

Thesis hướng tới tạo ra:

1. một hệ thống AIoT Smart Shelf hoạt động end-to-end;
2. một Computer Vision module để detect/count sản phẩm;
3. một load-cell-based inventory estimator;
4. một disagreement-aware sensor fusion algorithm;
5. một adaptive verification mechanism;
6. một event-triggered inference mechanism;
7. một IoT communication layer;
8. một web dashboard;
9. một custom sensor–vision dataset;
10. một bộ experimental results đầy đủ.

---

## 16. Expected Product Prototype

Prototype cuối cùng:

```text
                  SMART SHELF
                      │
        ┌─────────────┴──────────────┐
        │                            │
      Camera                    Load Cell
        │                            │
        ↓                            ↓
    Edge AI                      ESP32
        │                            │
        └────────────┬───────────────┘
                     ↓
              Fusion Engine
                     ↓
              Inventory State
                     ↓
              MQTT / REST API
                     ↓
               Web Dashboard
                     ↓
        ┌────────────┼──────────────┐
        ↓            ↓              ↓
    Low Stock     Stock-Out      Mismatch
      Alert         Alert          Alert
```

---

## 17. Dashboard Functions

Dashboard MVP có thể bao gồm:

### Current Inventory

```text
Product A: 12
Product B: 5
Product C: 0
```

### Shelf Status

```text
Product A → Normal
Product B → Low Stock
Product C → Stock-Out
```

### Sensor Status

- load cell online/offline;
- camera online/offline;
- ESP32 connection;
- latest measurement.

### Event History

```text
09:32 Product A: 10 → 9
09:35 Product A: 9 → 8
09:50 Product B: Low Stock
```

### AI Diagnostics

- vision count;
- weight count;
- disagreement score;
- confidence;
- verification status.

---

## 18. Expected Contributions

Thesis nên tập trung tối đa vào 3–4 contributions.

### Contribution 1 — Vision–Weight Sensor Fusion

Một framework tích hợp:

```text
Computer Vision
+
Load Cell
```

cho smart shelf inventory estimation.

---

### Contribution 2 — Disagreement-Aware Verification

Một mechanism tự động phát hiện sự không nhất quán giữa:

\[
\hat{N}_{vision}
\]

và:

\[
\hat{N}_{weight}
\]

để kích hoạt verification.

---

### Contribution 3 — Event-Triggered Edge Vision

Sử dụng sự kiện từ sensor để giảm unnecessary visual inference.

---

### Contribution 4 — Low-Cost Smart Shelf Prototype

Một hệ thống AIoT end-to-end gồm:

```text
Sensor
+
Camera
+
Edge AI
+
IoT
+
Dashboard
```

---

## 19. Feasibility

### Difficulty

**Medium**

### Recommended For

**Undergraduate Thesis**

### Estimated Duration

**6–8 months**, có thể mở rộng đến 9 tháng nếu bổ sung edge optimization hoặc field test.

---

## 20. Computational Requirement

### Phase 1 — Development

Có thể sử dụng:

- Google Colab;
- laptop;
- desktop GPU;
- Apple Silicon;
- RTX 3060/4060 nếu có.

Không bắt buộc edge hardware ngay từ đầu.

---

### Phase 2 — Prototype

Có thể lựa chọn:

#### Option A — Low Cost

```text
ESP32
+
Load Cell
+
HX711
+
USB Camera
+
Laptop
```

#### Option B — Edge Product Prototype

```text
ESP32
+
Load Cell
+
HX711
+
Camera
+
Raspberry Pi 5
```

#### Option C — Stronger Edge AI

```text
ESP32
+
Load Cell
+
Camera
+
Jetson Orin Nano
```

Việc lựa chọn phần cứng nên được thực hiện sau khi detector và sensor fusion pipeline đã ổn định.

---

## 21. Scope Control

Để thesis không bị quá lớn, nên giới hạn MVP.

### Recommended Scope

```text
1 shelf
+
5–10 product categories
+
1 camera
+
1–2 load cells
+
1 edge device
+
1 dashboard
```

Không nên ngay từ đầu thực hiện:

- toàn bộ supermarket;
- hàng trăm SKU;
- multi-camera;
- customer tracking;
- recommendation engine;
- automatic payment;
- complex cloud architecture.

Các chức năng trên có thể để lại cho future work.

---

## 22. Suggested Development Roadmap

### Month 1 — Problem Definition & Baseline

- literature review;
- define system requirements;
- select products;
- setup detector baseline;
- setup load cell.

### Month 2 — Vision Module

- collect sample data;
- train/fine-tune detector;
- implement product counting;
- benchmark vision-only system.

### Month 3 — IoT Sensor Module

- ESP32;
- HX711;
- load cell calibration;
- MQTT communication;
- synchronization.

### Month 4 — Sensor Fusion

- implement weight estimation;
- develop fixed fusion baseline;
- develop disagreement score;
- implement verification logic.

### Month 5 — Adaptive Inference

- event-triggered inference;
- low/high inference mode;
- latency optimization.

### Month 6 — Product Integration

- dashboard;
- alerts;
- inventory history;
- end-to-end system.

### Month 7 — Experiments

- baseline comparisons;
- ablation;
- occlusion test;
- lighting test;
- latency test.

### Month 8 — Thesis & Demo

- statistical analysis;
- figures;
- thesis writing;
- demo preparation.

### Optional Month 9

- deployment on Raspberry Pi / Jetson;
- additional robustness study;
- conference-paper preparation.

---

## 23. Product Potential

### Potential Users

- minimarts;
- pharmacies;
- warehouses;
- convenience stores;
- vending systems;
- small retail stores.

### Main Value Proposition

```text
Automatic Stock Monitoring
+
Reduced Manual Counting
+
Low-Stock Alert
+
Real-Time Inventory Visibility
+
Low-Cost Edge Deployment
```

### Possible Product Evolution

```text
Undergraduate Thesis
        ↓
Working Smart Shelf Prototype
        ↓
Pilot with One Store
        ↓
Multi-Shelf Deployment
        ↓
Inventory Analytics Platform
        ↓
Commercial AIoT Product
```

---

## 24. Publication Potential

**Publication Potential:** Medium–High

Hướng này có khả năng phát triển thành conference paper nếu thesis có:

- experimental protocol rõ ràng;
- custom multimodal dataset;
- baseline comparison;
- ablation study;
- robustness experiments;
- real edge deployment;
- quantitative analysis về accuracy–latency–energy trade-off.

Paper sẽ mạnh hơn nếu contribution tập trung vào:

> **Disagreement-Aware Multimodal Sensor Fusion**

thay vì chỉ mô tả việc tích hợp YOLO, ESP32 và MQTT.

---

## 25. Main Technical Risks

### Risk 1 — Load Cell Noise

Load cell có thể bị:

- vibration;
- drift;
- sensor noise.

Mitigation:

- moving average;
- median filtering;
- calibration;
- stable-window detection.

---

### Risk 2 — Mixed Products

Nếu nhiều loại sản phẩm có trọng lượng khác nhau nằm trên cùng load cell, weight-based counting trở nên khó.

Mitigation:

> MVP nên sử dụng một product type cho mỗi weighing zone.

---

### Risk 3 — Vision Accuracy

Detector có thể hoạt động kém khi occlusion cao.

Đây chính là tình huống để kiểm chứng sensor fusion.

---

### Risk 4 — Scope Explosion

Retail product recognition có thể trở thành bài toán rất lớn nếu cố hỗ trợ hàng trăm SKU.

Mitigation:

> Giới hạn 5–10 SKU trong thesis prototype.

---

## 26. Minimum Viable Thesis

Nếu cần giảm scope, thesis tối thiểu vẫn nên giữ:

```text
Camera
+
YOLO
+
Load Cell
+
Sensor Fusion
+
Disagreement Detection
+
MQTT
+
Dashboard
```

Có thể bỏ:

- advanced cloud;
- multi-shelf;
- mobile application;
- multiple cameras;
- complex deep multimodal fusion.

---

## 27. Stronger Research Extension

Nếu sinh viên hoàn thành MVP sớm, có thể mở rộng bằng:

### Extension A — Learned Fusion

Thay fixed rule bằng lightweight model:

\[
f(
N_v,
C_v,
N_w,
\Delta W,
history
)
\rightarrow
N_{final}
\]

---

### Extension B — Temporal Fusion

Sử dụng:

```text
Previous State
+
Current Vision
+
Current Weight
```

để giảm count fluctuation.

---

### Extension C — Active Verification

Hệ thống tự quyết định:

```text
Accept
or
Re-Infer
```

dựa trên expected benefit và compute cost.

---

### Extension D — Edge Optimization

Benchmark:

- laptop;
- Raspberry Pi;
- Jetson.

Đánh giá:

\[
Accuracy
\leftrightarrow
Latency
\leftrightarrow
Energy
\]

---

## 28. Possible Thesis Titles

### Recommended Title

> **Disagreement-Aware Vision and Weight Sensor Fusion for AIoT Smart Shelf Inventory Monitoring**

### Alternative Title 1

> **Sensor-Fused Smart Shelf Monitoring for Automated Retail Inventory Management**

### Alternative Title 2

> **Event-Driven Edge Vision and Weight Sensor Fusion for Low-Cost Smart Retail Inventory Monitoring**

---

# 29. Recommended Final Thesis Architecture

```text
                           ┌────────────────────────┐
                           │       SMART SHELF      │
                           └───────────┬────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
                  ↓                                         ↓
              Camera                                    Load Cell
                  │                                         │
                  ↓                                         ↓
          Lightweight YOLO                              HX711 / ESP32
                  │                                         │
                  ↓                                         ↓
        Detection + Confidence                         Weight Estimate
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       ↓
                            Disagreement Analyzer
                                       │
                              ┌────────┴────────┐
                              │                 │
                          Agreement         Disagreement
                              │                 │
                              ↓                 ↓
                        Accept Count      Verification Mode
                                                │
                                                ↓
                                    High-Resolution / Temporal
                                             Inference
                                                │
                              ┌─────────────────┘
                              ↓
                        Final Inventory
                              ↓
                       MQTT / REST API
                              ↓
                         Web Dashboard
                              ↓
              ┌───────────────┼──────────────────┐
              ↓               ↓                  ↓
          Low Stock        Stock-Out          Mismatch
            Alert            Alert              Alert
```

---

# 30. Advisor Recommendation

Đây là một đề tài rất phù hợp cho **Undergraduate Thesis theo hướng Product-oriented trong 6–9 tháng** vì kết hợp cân bằng giữa Computer Vision, IoT, Edge AI, sensor fusion và product development.

Điểm quan trọng nhất là không biến thesis thành một bài tích hợp đơn thuần kiểu:

```text
YOLO
+
ESP32
+
MQTT
```

Thay vào đó, phần nghiên cứu nên tập trung vào:

> **Disagreement-Aware Vision–Weight Fusion + Adaptive Verification**

Đây là mechanism đủ rõ để xây dựng:

- research question;
- baseline;
- ablation study;
- robustness experiments;
- quantitative evaluation.

Đồng thời sản phẩm cuối vẫn có một MVP rất trực quan và dễ trình diễn:

> **Camera nhìn sản phẩm → load cell xác nhận → AI fusion quyết định → dashboard cập nhật tồn kho và cảnh báo.**

Nếu tài nguyên tính toán chưa xác định, sinh viên nên triển khai toàn bộ AI pipeline trên PC hoặc Google Colab trước. Edge hardware chỉ nên được đưa vào sau khi vision baseline và sensor fusion đã hoạt động ổn định.

Nếu mục tiêu sau thesis là phát triển thành startup hoặc pilot với doanh nghiệp, hệ thống có thể tiếp tục mở rộng từ:

```text
1 Smart Shelf
↓
Multiple Smart Shelves
↓
Store-Level Inventory
↓
Cloud Analytics
↓
Commercial AIoT Retail Platform
```
