# Problem Statement and Research Questions

**Project Title:** Disagreement-Aware Vision and Weight Sensor Fusion for AIoT Smart Shelf Inventory Monitoring  
**Academic Degree:** Bachelor of Engineering in Electronics and Automation Engineering  
**Document Status:** Ready for Thesis Supervisor Review  
**Date:** October 2026  

---

## 1. Problem Statement

In modern retail stores and small distribution warehouses, maintaining accurate real-time inventory on display shelves is essential for preventing stockouts and ensuring timely restocking. Traditionally, inventory tracking relies on periodic manual checks by store staff, which are labor-intensive, error-prone, and incapable of capturing rapid stock fluctuations during peak shopping hours. Automating this process using Artificial Intelligence of Things (AIoT) smart shelf systems can provide continuous product tracking and timely inventory alerts, helping small-scale businesses maintain high service availability with minimal operational overhead.

Existing automated monitoring systems typically rely either on camera-based computer vision or load-cell-based weight sensing; however, each modality exhibits inherent physical limitations when operating alone. Vision-based detection models, such as lightweight YOLO detectors, frequently suffer from severe detection drops when products are arranged in depth, partially occluded by front items, or obscured by customer hand interactions (Goldman et al., 2019). Additionally, lighting variations and visual similarities among packaging cause bounding boxes to flicker across video frames, yielding unstable count estimates. Conversely, load cells equipped with analog-to-digital converters measure shelf mass reliably in static environments but fundamentally lack semantic information; they cannot distinguish between different stock-keeping units (SKUs) sharing similar unit weights (Tamer & Koklu, 2021). Furthermore, raw weight signals are vulnerable to mechanical vibrations, zero-point drift, and transient external forces such as shoppers leaning on the shelf edge.

Combining camera vision and weight sensing represents a natural multimodal solution; however, simply merging their raw outputs does not resolve underlying sensing conflicts. Existing smart shelf studies predominantly utilize static rule-based lookups or naive decision averaging, often assuming that at least one modality provides ground-truth accuracy (Falcão et al., 2020; Zhang et al., 2023). In real retail environments, these two sources frequently produce contradictory estimates. For instance, an optical camera may detect only two items due to deep back-row occlusion, while the load cell measures a total weight corresponding to four items. Conversely, an accidental customer lean creates a transient weight surge while visual tracking confirms that the shelf contents remain unchanged. Blithely aggregating these conflicting signals without evaluating modality reliability can propagate erroneous counts or trigger false inventory alerts. A practical gap therefore remains in establishing an intelligent mechanism that explicitly detects, characterizes, and resolves vision–weight disagreements.

To bridge this gap, this undergraduate thesis investigates a disagreement-aware multimodal sensor fusion framework tailored for resource-constrained AIoT smart shelves. By integrating a front-facing RGB camera running lightweight object detection with an array of strain-gauge load cells, the proposed architecture continuously compares visual counts and bounding-box confidences against filtered weight differentials. Instead of enforcing rigid consensus, the system quantifies the degree of cross-modal disagreement and dynamically assigns adaptive trust weights—favoring load-cell readings during heavy visual occlusion while prioritizing visual confirmation during mechanical contact disturbances. The performance of this disagreement-aware approach will be systematically evaluated using standard engineering metrics, including item-counting Mean Absolute Error (MAE), Root Mean Square Error (RMSE), exact-match accuracy, conflict detection rate, and false-alert frequency under controlled retail scenarios.

---

## 2. Research Questions

To guide the experimental investigation and address the identified gaps, this thesis formulates three specific research questions:

* **RQ1 (Visual Inventory Estimation):**  
  How accurately can a lightweight computer vision model (e.g., YOLOv8n) detect and count shelf products under real-world retail constraints, specifically varying levels of inter-product occlusion, deep clustering, and ambient lighting variations?

* **RQ2 (Vision–Weight Disagreement Characterization):**  
  How can quantitative disagreements between camera-based count estimates and load-cell mass differentials be reliably identified, formulated, and classified into distinct operational conflict modes (such as visual occlusion, mechanical contact spikes, or foreign object placement)?

* **RQ3 (Disagreement-Aware Sensor Fusion Evaluation):**  
  To what extent does incorporating explicit disagreement information into a multimodal fusion strategy improve inventory estimation accuracy (reducing MAE and RMSE) and suppress false restock alerts compared to a vision-only approach and conventional naive fusion baselines?

---

## 3. Scope and Alignment Check

### 3.1. Alignment with Thesis Objectives
* **RQ1** establishes the empirical baseline, delineating where optical sensing alone degrades and defining the engineering necessity for a secondary modality.
* **RQ2** delivers the core engineering mechanism, formulating the thresholding and uncertainty metrics required to diagnose why vision and weight diverge during shelf interactions.
* **RQ3** provides the final empirical validation, benchmarking the proposed adaptive fusion engine against single-modality and static-fusion baselines on physical shelf hardware.

### 3.2. Scope Feasibility for Undergraduate Thesis
* Hardware is restricted to an edge microcontroller (ESP32) for load-cell acquisition and an embedded gateway/PC running lightweight vision inference.
* Product scope focuses on standard fast-moving retail SKUs (e.g., beverage cans and boxed dairy packages).
* Experiments are conducted on a benchtop smart shelf prototype with reproducible occlusion, lighting, and load-change scenarios.

### 3.3. Literature Verification Notes
* The claim regarding single-camera occlusion bottlenecks is well-supported by retail benchmark studies (Goldman et al., 2019; Wei et al., 2020).
* The assertion that prior retail smart shelves predominantly utilize static or non-disagreement-aware fusion is supported by recent edge architectures (Falcão et al., 2020; Zhang et al., 2023).

---

## 4. References

* **Falcão, G. et al. (2020)**. *FAIM: A Smart Shelf for Autonomous Retail with Multimodal Sensing*. Frontiers in Built Environment.
* **Goldman, E. et al. (2019)**. *Precise Detection in Densely Packed Scenes*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5227–5236.
* **Tamer, E. & Koklu, M. (2021)**. *Weight-Based Smart Shelf System for Automatic Inventory Management*. International Journal of Innovative Research and Scientific Studies, 4(3), pp. 182–189.
* **Wei, X.-S. et al. (2020)**. *RPC: A Large-Scale Retail Product Checkout Dataset*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
* **Zhang, H. et al. (2023)**. *Edge-Assisted Multimodal Sensing for Smart Retail Shelves*. ACM Transactions on Sensor Networks (TOSN).
