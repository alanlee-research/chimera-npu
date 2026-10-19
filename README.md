# Chimera: Converting NPU Headroom into Small-Object Accuracy

Code, model definitions, and evaluation scripts for the paper

> **Converting NPU Headroom into Small-Object Accuracy: Hardware-Aware Detector Design and Adaptive Slicing for Real-Time UAV Detection**
> Van-An Le. Manuscript under review.

The work treats inference speed on an embedded NPU (Rockchip RK3582) as a budget. **Chimera** is a detector without a stride-32 stage, designed from on-device profiling. It runs in 10.88 ms on one NPU core. An **AIMD scheduler** spends the remaining NPU time on image tiles and still returns a full-frame result for every frame at 30 FPS.

## Main results

All latencies were measured on an RK3582 with NPU Core 0 at 900 MHz, INT8, 832×480 input.

| Model | Params | NPU latency | VisDrone test-dev mAP50:95 (4 runs) |
|---|---|---|---|
| YOLO11n-LeakyReLU | 2.59 M | 11.63 ms | 34.49 ± 0.06 % |
| **Chimera-LReLU** | **1.12 M** | **10.88 ms** | **34.51 ± 0.08 %** |

**Slicing gain at an equal false-positive rate** (VisDrone-DET test-dev, Chimera-LReLU, full frame vs. full frame merged with four tiles):

| Source resolution | Ratio to network input | mAP50 | Recall of objects < 32 px |
|---|---|---|---|
| 1916×1078 | 2.30 | 0.321 → 0.452 | +15.6 points |
| 1400×788 | 1.68 | 0.330 → 0.394 | +8.2 points |
| 1360×765 | 1.63 | 0.540 → 0.614 | +11.5 points |
| UAVDT 1024×540 | 1.23 | — | no gain |

See the paper for the full protocol and the INT8 robustness analysis.

## Repository structure

```
chimera/            model definition
  modules.py          HFGate and MDRF (DilatedContextSlim) blocks for Ultralytics
  modules_proto.py    blocks of the earlier design (Chimera-Proto)
  configs/            chimera_lrelu.yaml (proposed), chimera_silu.yaml, chimera_proto.yaml, yolo11n_lrelu.yaml,
                      chimera_lrelu_noMDRF.yaml, chimera_lrelu_noP2.yaml (ablation, Table 5)
data/dataset_prep.py  builds the 832x480 training/evaluation set from VisDrone-DET (4 merged classes)
train/train_dual.py   trains YOLO11n-LeakyReLU and Chimera-LReLU in parallel on two GPUs, exports ONNX
deploy/convert_rknn.py  INT8 conversion with rknn-toolkit2 (per-channel, 300 calibration images)
scheduler/          reference implementation of the AIMD scheduler and the tile merge rule (+ tests)
eval/               mAP on the 800-image subset, paired bootstrap, slicing evaluation (VisDrone-DET, UAVDT)
figures/            script that draws the paper figures from the reported numbers
results/            metric files behind the tables of the paper
```

## Setup

```bash
pip install -r requirements.txt
# rknn-toolkit2 2.3.2 is only needed for INT8 conversion (see Rockchip's installation guide).
```

To build a model from a config:

```python
import sys; sys.path.insert(0, "chimera")
import modules; modules.register()          # makes HFGate / DilatedContextSlim available to Ultralytics
from ultralytics import YOLO
model = YOLO("chimera/configs/chimera_lrelu.yaml")   # 1.125 M parameters, LeakyReLU only
```

## Reproducing the pipeline

1. **Dataset.** Run `data/dataset_prep.py`. It is written for a Kaggle notebook but also runs locally with `VD_WORK=<dir>`. It needs the VisDrone2019-DET train/val/test-dev splits, maps the ten classes to Person, Two-wheeler, Car, and Heavy, and writes `vd_deploy_v3/`:
   - training samples: full frames plus one region-of-interest crop each;
   - a validation set;
   - evaluation sets with ignore regions;
   - calibration images.
2. **Training.** `train/train_dual.py` trains both models concurrently on two GPUs with the same wall-clock budget. Set `SEED` at the top of the file for repeated runs (the paper uses 42, 42, 9, and 16). The script checks that only LeakyReLU remains in the network, then exports ONNX models with separate box and class outputs per stride. It also writes a package for INT8 conversion.
3. **INT8 conversion.** Run `deploy/convert_rknn.py [rk3588]` inside an exported package. The package must contain `model_fp32.onnx` and `calib_images/`.
4. **Evaluation.**
   - `eval/eval_map.py pack | host --onnx M.onnx | score --dets D.jsonl` computes COCO-style AP on the 800-image subset. It uses the same letterbox and decoding as the device.
   - `eval/bootstrap_map.py` gives paired bootstrap intervals clustered by sequence. It reads `<model>_832x480_{fp32,int8}.dets.jsonl` from `results/int8_800/` (or `INT8_RESULTS_DIR`); the per-image detection files are available on request, the resulting `bootstrap.json` is included.
   - `eval/eval_slicing_det.py <WxH>` and `eval/eval_slicing_uavdt.py full` evaluate full frame vs. full frame plus tiles against ground truth.
   - Paths are set with the environment variables `EVAL800_DIR`, `VISDRONE_TESTDEV_LABELS`, and `UAVDT_LABELS`.
5. **Scheduler.** `scheduler/hybrid.py` implements Algorithm 1 of the paper and the merge rule. Run the checks with `cd scheduler && python -m pytest test_hybrid.py`.

**Detection file format** (used by all evaluation scripts): one JSON object per image or frame,

```json
{"f": 0, "dets": [[x, y, w, h, score, class_id], ...]}
```

Boxes are in source-frame pixels; `class_id` is 0 Person, 1 Two-wheeler, 2 Car, 3 Heavy. The slicing evaluation expects `<model>_full.jsonl` (full frame only) and `<model>_sahi.jsonl` (the four tiles of each image) in `eval/logs/`. It merges them with the merge rule of the paper. It also reads the tile grid from a line `slice grid NxM, tile WxH` in `<model>_sahi.txt`.

## What is not included

- **The on-device C++ pipeline.** This covers hardware decoding, RGA preprocessing, the NPU runtime, and the tracker. It is part of a product and is not released. The scheduling and merging logic it implements is given in `scheduler/hybrid.py`.
- **Datasets, trained weights, and the streaming video sequences.** VisDrone and UAVDT are available from their authors. Trained weights and the evaluation subsets are available from the author on reasonable request.

## Hardware and software used in the paper

| Item | Value |
|---|---|
| Board | Rockchip RK3582 (NPU Core 0, 900 MHz; DRAM 2112 MHz) |
| Toolchain | rknn-toolkit2 / RKNN runtime 2.3.2, INT8 per-channel |
| Training | Ultralytics 8.4.175, two NVIDIA T4 GPUs |

## Citation

```bibtex
@misc{le2026chimera,
  author = {Le, Van-An},
  title  = {Converting {NPU} Headroom into Small-Object Accuracy: Hardware-Aware Detector Design
            and Adaptive Slicing for Real-Time {UAV} Detection},
  note   = {Manuscript under review},
  year   = {2026},
  url    = {https://github.com/alanlee-research/chimera-npu}
}
```

## License

This code builds on [Ultralytics](https://github.com/ultralytics/ultralytics), which is released under AGPL-3.0. See `LICENSE`.

## Contact

Van-An Le: levanan02sm@gmail.com · ORCID [0009-0009-3859-9438](https://orcid.org/0009-0009-3859-9438)
