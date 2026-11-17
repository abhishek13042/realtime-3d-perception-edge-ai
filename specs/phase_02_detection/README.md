# Phase 02 — 2D Detection Baseline

**Exit condition (docs/architecture.md §5):** detector output is deterministic; the letterbox round trip is ≤ 1 px; a baseline mAP table exists for the classes shared by COCO and SUN RGB-D.

| Spec | File | Main outputs |
|---|---|---|
| 02.1 | [spec_02_1_detector_interface.md](spec_02_1_detector_interface.md) | `perception/detection/{base,letterbox,torch_backend,classes}.py`, `configs/detector.yaml` |
| 02.2 | [spec_02_2_baseline_inference.md](spec_02_2_baseline_inference.md) | `scripts/run_detector.py`, `benchmarks/detection/*.jsonl`, latency JSON, overlay gallery |
| 02.3 | [spec_02_3_detection_evaluation.md](spec_02_3_detection_evaluation.md) | `perception/eval/detection.py`, `scripts/eval_detection.py`, `docs/detection_baseline.md` |

**Scope rules for this phase**
- Pretrained **YOLO26-n** (COCO, AGPL-3.0; decision D3). **No training or fine-tuning.** Fine-tuning, if it is ever wanted, happens on a Kaggle GPU in a separate, later spec.
- Inference runs on the local CPU (decision D4). Every latency number names the machine, the backend, the thread count, and the input size.
- Thresholds are chosen on **val** (`configs/splits/sunrgbd_val.txt`) and reported on **test**. Test is never used for a choice.
- The ONNX backend belongs to Phase 06. The interface here is built so that ONNX can reuse the same pre- and post-processing.

**Template:** Objective · Requirements · Tasks · Acceptance criteria · Tests · Dependencies.
