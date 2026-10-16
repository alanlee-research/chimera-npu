#!/usr/bin/env python3
"""Full frame vs full + all four slices (offline merge) on UAVDT test images (letterboxed to 832x480
by the export, labels already in the deploy classes). UAVDT labels only vehicles: car -> car,
truck / bus -> heavy. Detections of person / two-wheeler are discarded before scoring (UAVDT does not
label them), and only the car and heavy classes are evaluated. No ignore regions in this export."""
import contextlib, io, json, os, sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CLASSES = (2, 3)
import op_point as op

import re
TAG = sys.argv[1] if len(sys.argv) > 1 else "uavdt"       # "uavdt" (832x480 letterboxed export) or "full" (1024x540)
PFX = "" if TAG == "uavdt" else f"{TAG}_"
W, H = (832, 480) if TAG == "uavdt" else (1024, 540)
S_FULL = min(832 / W, 480 / H)
YMAP = {2: 2, 3: 3}
LAB = os.environ.get("UAVDT_LABELS", "uavdt_test/labels")  # YOLO-format labels, deploy classes
_g = re.search(r"slice grid (\d+)x(\d+), tile (\d+)x(\d+)", open(f"logs/{PFX}chimera_lrelu_sahi.txt").read())
NX, NY, TW, TH = (int(v) for v in _g.groups())                      # grid from the board log
TILE_X = [i * (W - TW) / (NX - 1) if NX > 1 else 0 for i in range(NX)]
TILE_Y = [j * (H - TH) / (NY - 1) if NY > 1 else 0 for j in range(NY)]

imgs = [l.strip() for l in open(f"{PFX}images.txt")]
gt, ign = {}, {}
for i, f in enumerate(imgs):
    for line in open(os.path.join(LAB, f.rsplit(".", 1)[0] + ".txt")):
        c, cx, cy, bw, bh = map(float, line.split())
        b = [(cx - bw / 2) * W, (cy - bh / 2) * H, bw * W, bh * H]
        if int(c) in YMAP: gt.setdefault(i, []).append(b + [YMAP[int(c)]])
        else: ign.setdefault(i, []).append(b)

def load(p): return {r["f"]: [d for d in r["dets"] if int(d[5]) in CLASSES] for r in map(json.loads, open(p))}

def inner_edge(d, e=2.0):
    x1, y1, x2, y2 = d[0], d[1], d[0] + d[2], d[1] + d[3]
    inner_l = [x for x in TILE_X if x > 0]; inner_r = [x + TW for x in TILE_X if x + TW < W - 1]
    inner_t = [y for y in TILE_Y if y > 0]; inner_b = [y + TH for y in TILE_Y if y + TH < H - 1]
    return (any(abs(x1 - v) <= e for v in inner_l) or any(abs(x2 - v) <= e for v in inner_r) or
            any(abs(y1 - v) <= e for v in inner_t) or any(abs(y2 - v) <= e for v in inner_b))

def merge(full, sahi, filt):
    out = {}
    for f in full:
        t = [d for d in sahi.get(f, []) if not filt or (not inner_edge(d) and (d[2] * d[3]) ** 0.5 * S_FULL < 32)]
        ds = full[f] + t
        keep = []
        for c in CLASSES:
            keep += op.nms([d for d in ds if d[5] == c], 0.7)
        out[f] = sorted(keep, key=lambda d: -d[4])[:300]
    return out

def coco_map(D):
    g = {"images": [{"id": i} for i in range(len(imgs))], "annotations": [], "categories": [{"id": c} for c in CLASSES]}
    k = 1
    for i in range(len(imgs)):
        for b in gt.get(i, []):
            g["annotations"].append({"id": k, "image_id": i, "category_id": b[4], "iscrowd": 0, "area": b[2] * b[3], "bbox": b[:4]}); k += 1
        for b in ign.get(i, []):
            for c in CLASSES:
                g["annotations"].append({"id": k, "image_id": i, "category_id": c, "iscrowd": 1, "area": b[2] * b[3], "bbox": b}); k += 1
    C = COCO()
    with contextlib.redirect_stdout(io.StringIO()):
        C.dataset = g; C.createIndex()
        R = C.loadRes([{"image_id": f, "category_id": int(d[5]), "bbox": d[:4], "score": d[4]} for f, ds in D.items() for d in ds])
    res = {}
    for lab, rng in [("all", [0, 1e10]), ("small", [0, 1024])]:
        E = COCOeval(C, R, "bbox"); E.params.maxDets = [300]; E.params.areaRng = [rng]; E.params.areaRngLbl = ["r"]
        with contextlib.redirect_stdout(io.StringIO()):
            E.evaluate(); E.accumulate()
        p = E.eval["precision"][:, :, :, 0, 0]
        res[lab] = (float(p[0][p[0] > -1].mean()), float(p[p > -1].mean()))
    return res

def op_point(D, conf):
    tp = fp = ngt = tps = ngs = 0
    for i in range(len(imgs)):
        g = gt.get(i, []); ngt += len(g); ngs += sum((b[2] * b[3]) ** 0.5 < 32 for b in g)
        used = set()
        for d in op.nms([d for d in D.get(i, []) if d[4] >= conf], 0.45):
            best, j = 0, -1
            for k, b in enumerate(g):
                if k in used or b[4] != d[5]: continue
                v = op.iou(d, b)
                if v > best: best, j = v, k
            if best >= 0.5: used.add(j); tp += 1; tps += (g[j][2] * g[j][3]) ** 0.5 < 32
            elif not any(op.ioa(d, z) >= 0.5 for z in ign.get(i, [])): fp += 1
    return tp / ngt, tps / ngs, tp / max(1, tp + fp), fp / len(imgs)

n_gt = sum(len(v) for v in gt.values()); n_s = sum((b[2] * b[3]) ** 0.5 < 32 for v in gt.values() for b in v)
print(f"{len(imgs)} images, {n_gt} GT objects ({n_s} small, {100 * n_s / n_gt:.1f}%), {n_gt / len(imgs):.1f} per image")
res = {}
for m in ["chimera_lrelu", "yolo11n_lrelu"]:
    full, sahi = load(f"logs/{PFX}{m}_full.jsonl"), load(f"logs/{PFX}{m}_sahi.jsonl")
    for name, D, confs in [("full", full, [0.30, 0.25, 0.20, 0.15, 0.10]), ("full+4 tiles, new merge", merge(full, sahi, True), [0.30, 0.40, 0.50]),
                           ("full+4 tiles, raw merge", merge(full, sahi, False), [0.30]), ("SAHI only", sahi, [0.30])]:
        mp = coco_map(D)
        for c in confs:
            r, rs, pr, fpi = op_point(D, c)
            print(f"{m:18s} {name:24s} mAP50 {mp['all'][0]:.3f} mAP50-95 {mp['all'][1]:.3f} small mAP50 {mp['small'][0]:.3f} | "
                  f"conf {c:.2f}: recall {r:.3f} small {rs:.3f} precision {pr:.3f} FP/img {fpi:.1f}")
            res[f"{m}|{name}|{c}"] = dict(mAP50=mp["all"][0], mAP50_95=mp["all"][1], small_mAP50=mp["small"][0], recall=r, recall_small=rs, precision=pr, fp_per_img=fpi)
json.dump(res, open(f"det_{TAG or 'fhd'}_results.json", "w"), indent=1)
