# Spec 02.1 — Detector Interface, Letterbox, and PyTorch Backend

| Field | Value |
|---|---|
| Phase | 02 — Detection |
| Status | ✅ Done 2026-11-10 (specs approved 2026-11-10: "execute next steps one by one") |
| Depends on | 01.2 (Frame contract) |
| Blocks | 02.2, 02.3, 04.x (tracking input), 06.x (ONNX backend reuses pre/post) |

## Objective
Wrap the pretrained YOLO26-n behind a small `Detector` interface. It takes an RGB `uint8` image and returns detections in **original-image pixels**. Pre- and post-processing are **ours**, not hidden inside `model.predict`. That way the Phase 06 ONNX backend can run the exact same code, and any accuracy difference can only come from the model runtime.

## Requirements
- **R1. Data type.** `Detection{bbox_xyxy: float32[4] (original-image px, 0-based, pixel centres at integers), score: float, class_id: int (COCO 0–79), class_name: str}`. A batch of detections is held as arrays (`boxes N×4`, `scores N`, `class_ids N`) so that later stages stay vectorized.
- **R2. Letterbox** (`perception/detection/letterbox.py`, pure functions):
  - `letterbox(img, size, pad_value=114) -> (canvas, LetterboxInfo{scale, pad_x, pad_y, src_hw})`. Scaling keeps the aspect ratio, and the image is centred.
  - `unletterbox_boxes(boxes, info)` inverts the transform and clips boxes to the source image.
  - The default input size is 640. The size is a config value, because Phase 06 sweeps it.
- **R3. Interface** (`perception/detection/base.py`):
  - `Detector.detect(rgb) -> Detections`. It also exposes per-stage timings for preprocess, inference, and postprocess.
  - Postprocess: undo the letterbox, apply the confidence threshold, cap the count at `max_det`, and optionally keep only certain classes.
- **R4. PyTorch backend** (`torch_backend.py`):
  - Loads `yolo26n.pt` through Ultralytics, then calls the network directly on our letterboxed tensor.
  - **Finding (2026-11-10, ultralytics 8.4.208).** The Detect head has two branches:
    - **one2one** (NMS-free): with `model.end2end = True`, the fused model returns (B, 300, 6) `[x1, y1, x2, y2, score, cls]` in letterbox px, picked by top-k. Ultralytics uses it for `predict(nms=False)`.
    - **one2many**: returns (B, 84, 8400) `[cx, cy, w, h, 80 scores]`, which needs NMS. This is Ultralytics' default `predict` path. We apply class-aware `torchvision.ops.batched_nms` with IoU 0.7.
  - **Decision:** `head: one2one` by default. It is YOLO26's design and needs no NMS operator, which simplifies the Phase 06 ONNX/edge export. `one2many_nms` is kept for the ablation in 02.3.
- **R5. Determinism.** Inference runs in eval mode under `torch.inference_mode()`, with a fixed `torch.set_num_threads` taken from the config. No randomness.
- **R6. Config** (`configs/detector.yaml`): weights path and **SHA-256**, input size, confidence threshold for evaluation (`0.001`, the COCO convention) and the operational threshold (filled in by spec 02.3 from val), `max_det`, and thread count.
- **R7. Provenance.** The weights come from the official Ultralytics release. Their SHA-256 is recorded on first download and checked on every load, the same way as the dataset lock file.

## Tasks
- [x] T1. Install torch (CPU), ultralytics, onnx, onnxruntime, and pycocotools in the venv. Pin the versions in `pyproject.toml`.
- [x] T2. Download `yolo26n.pt`, record its SHA-256, and inspect the raw output layout (R4). Record the finding here.
- [x] T3. `letterbox.py`, with tests.
- [x] T4. `base.py`, `classes.py` (the 80 COCO names, taken from the model), and `torch_backend.py`.
- [x] T5. Reference parity check against Ultralytics `model.predict` on 20 SUN val images.

## Acceptance criteria
- **AC1. Letterbox round trip ≤ 1 px.** Map random boxes into letterbox space and back, for several image sizes (all 4 SUN sensor sizes, 640×480 TUM, and odd sizes) and input sizes (320, 480, 640). The maximum corner error must be ≤ 1 px. In practice we expect about 1e-4 px; the 1 px bound is the phase exit criterion.
- **AC2. Deterministic.** Two runs on the same 20 images in one process, and a run in a fresh process, give identical boxes, scores, and classes. The tolerance is exact equality; if exactness fails, we document why and use ≤ 1e-5.
- **AC3. Parity with the reference.** On 20 SUN val images, at the same input size and threshold, our detections match Ultralytics `predict` as follows: the same count, matched one-to-one by IoU ≥ 0.99 with the same class, and |Δscore| ≤ 1e-3. Any mismatch is explained. For example, Ultralytics uses a "minimal-padding rect" letterbox, while we use a fixed square; if that difference matters, we compare with `rect=False`.
- **AC4.** Boxes are in original-image coordinates. A visual overlay on 5 images shows that the boxes sit on the objects.

## Tests (`tests/test_letterbox.py`, `tests/test_detector.py`)
- The letterbox round trip on random boxes is within 1 px. For exact-scale cases, such as 640×480 at size 640, it is within 1e-4.
- Letterbox geometry: a 640×480 image at size 640 gives scale 1.0, pad_y 80, and pad_x 0.
- `unletterbox_boxes` clips to the image bounds.
- An empty detection result has shape (0, 4), not an error.
- Detector smoke test, skipped if the weights are absent: the output arrays are float32 with consistent lengths, and class ids are within 0–79.
- Determinism: calling `detect` twice on the same image gives equal arrays.

## Dependencies
torch (CPU wheel), torchvision (for NMS, if needed), ultralytics 8.4.x, numpy, opencv-python.

## Results (2026-11-10)
Produced by `scripts/check_detector.py`; raw output in `benchmarks/detection/detector_check.json`. Measured on CPU (i3-1115G4), conf 0.25, 20 SUN val images (`sample_ids("sunrgbd_val", 20, seed=0)`).

| AC | Result | Measured |
|---|---|---|
| AC1 | ✅ | Round-trip error < 1e-9 px. The letterbox is byte-identical to Ultralytics `LetterBox(auto=False)` (`tests/test_letterbox.py`). |
| AC2 | ✅ | Two runs in one process are exactly equal. A fresh process gives the same SHA-256 digest of all outputs (`eda4c6f5…58d7`). |
| AC3, one2one vs `predict(nms=False)` | ✅ | 81 / 81 detections; all 20 images have the same count; all matched with the same class; min IoU 0.999996; max \|Δscore\| 1.2e-4 |
| AC3, one2many_nms vs `predict()` | ✅ | 90 / 90 detections; all 20 images have the same count; min IoU 0.999997; max \|Δscore\| 3.2e-6 |
| AC4 | ✅ | Overlays in `benchmarks/detection/check/` (5 images): the boxes sit on the chairs and beds. |

**Note:** the one2one head can emit a low-score near-duplicate (for example, `bed 0.25` inside `bed 0.83` on image 006044), because no NMS runs. 02.3 measures this effect through the operating threshold and the one2one-vs-NMS ablation.

The development timing smoke figure (median over 20 images, no warm-up protocol, **not a benchmark**) is preprocess 5.9 ms, infer 117 ms, postprocess 0.2 ms. The real latency protocol is spec 02.2.
