# Risks and Limitations

**Spec:** 00.4 · **Status:** Drafted · **Last updated:** 2026-11-07

## 1. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Owner phase |
|---|---|---|---|---|---|
| R1 | ScanNet access is denied or delayed (institutional email required) | Med–High | Low (optional) | The MVP uses only ungated data; TUM provides the stability metric | 00 / user |
| R2 | SUN RGB-D upright-frame → camera-frame conversion is wrong, so every G3 number is wrong | Medium | **High** | Project the GT 3D box corners onto the RGB image and inspect them visually; add a unit test on a hand-checked frame | 01.3 |
| R3 | Depth decoding is wrong (bit rotation, scale) | Low–Med | **High** | Check that the depth histogram is plausible (rooms are 1–6 m); overlay depth edges on RGB edges | 01.2 |
| R4 | Poor COCO ↔ dataset class overlap gives a weak or unrepresentative mAP | Medium | Medium | Evaluate only the shared classes and disclose this; fine-tune on Kaggle later only if justified | 02.3 |
| R5 | CPU too slow for "real-time" (i3, 2 cores) | **High** | Medium | Smaller input (320/416), INT8, thread tuning, optionally OpenVINO EP; report the bottleneck honestly | 05.3, 06 |
| R6 | Camera motion in TUM breaks the 2D Kalman model, causing ID switches | Medium | Medium | Measure it; if confirmed, consider 3D tracking or camera-motion compensation | 04.3 |
| R7 | Too few COCO objects in the TUM sequences | Medium | Medium | Choose sequences in 01.1 by first running the detector on a sample; fall back to ScanNet or a self-recorded sequence | 01.1 |
| R8 | Secondary packages lack Python 3.14 wheels | Medium | Low | Fall back to a Python 3.12 venv | 01 |
| R9 | RAM limit (7.7 GB) when loading large MATLAB metadata | Medium | Low | Convert the metadata once to compact JSON/NPZ, and stream frames | 01.1 |
| R10 | AGPL-3.0 obligations if Ultralytics is used | Certain if chosen | Low for a public portfolio | Release the repo under AGPL-3.0, or choose YOLOX (Apache-2.0) | D3 |
| R11 | Surface vs volume-center bias is misread as an error | Certain | Medium | Report the signed bias explicitly; explain it in docs and interviews | 05.2 |
| R12 | Benchmark noise on a laptop (thermal throttling, background apps) | High | Medium | Plugged in, fixed power plan, warm-up, 3 repeats, report the spread | 05.1 |

## 2. Known limitations (by design)

1. **Camera-frame positions only.** World-frame positions are available only where the dataset provides poses (TUM, ScanNet).
2. **Visible-surface center, not object center.** Without learned amodal estimation (F-PointNets step 4), the estimate is biased toward the camera.
3. **No 3D boxes in the MVP.** A 2D box does not define a 3D box. Oriented boxes are an extension only.
4. **Depth-sensor failure surfaces:** glass, mirrors, glossy or black objects, and very thin objects. These produce `no_depth` or `low_coverage`, by design.
5. **Range limits.** Structured-light and ToF sensor accuracy degrades with distance, roughly with Z² for Kinect-type sensors. We expect the error to grow with distance and measure it by bucket.
6. **Pretrained COCO detector.** Classes are limited to COCO, and there is a domain gap to indoor RGB-D imagery.
7. **Offline input.** There is no live camera unless the user has a sensor (decision D5).
8. **CPU-only local benchmarks.** GPU and TensorRT numbers, if any, come from different hardware and are reported separately.
9. **The label-free stability metric** assumes static objects, and mixes detector, depth, and center-estimation noise into one number.

## 3. What we will not claim
- Implementing PointNet, PointNet++, or F-PointNets.
- Metric 3D from RGB alone.
- Any FPS figure without its boundary and hardware.
- Real-time performance, unless it is measured under the protocol.
- Results on the full ScanNet benchmark.
