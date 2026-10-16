#!/usr/bin/env python3
"""Recall / precision / false positives per frame at the deployed operating point (conf >= 0.30,
class-agnostic NMS 0.45) against VisDrone-VID ground truth. Matching: same class, IoU >= 0.5,
greedy by score; detections overlapping an ignore region (IoA >= 0.5) are neither TP nor FP.
Frames without a result count their ground truth as misses."""
import json, sys
import numpy as np
MAP = {1: 0, 2: 0, 3: 1, 10: 1, 4: 2, 5: 2, 6: 3, 9: 3}

def iou(a, b):
    ix = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); iy = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / (a[2]*a[3] + b[2]*b[3] - i + 1e-9)

def ioa(a, b):
    ix = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); iy = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2]*a[3] + 1e-9)

def nms(d, thr):
    d = sorted(d, key=lambda x: -x[4]); keep = []
    for b in d:
        if all(iou(b, k) <= thr for k in keep): keep.append(b)
    return keep

def run(ann, n, path, conf=0.30):
    gt, ign = {}, {}
    for l in open(ann):
        t = [int(v) for v in l.split(",")[:8]]
        f = t[0] - 1
        if not 0 <= f < n: continue
        if t[6] == 1 and t[7] in MAP: gt.setdefault(f, []).append(t[2:6] + [MAP[t[7]]])
        else: ign.setdefault(f, []).append(t[2:6])
    D = {}
    for l in open(path):
        r = json.loads(l); D[r["f"]] = nms([d for d in r["dets"] if d[4] >= conf], 0.45)
    tp = fp = ngt = tps = ngs = 0
    for f in range(n):
        g = gt.get(f, []); ngt += len(g); ngs += sum((x[2]*x[3]) ** 0.5 < 32 for x in g)
        used = set()
        for d in D.get(f, []):
            best, j = 0, -1
            for k, x in enumerate(g):
                if k in used or x[4] != d[5]: continue
                v = iou(d, x)
                if v > best: best, j = v, k
            if best >= 0.5:
                used.add(j); tp += 1; tps += (g[j][2]*g[j][3]) ** 0.5 < 32
            elif not any(ioa(d, z) >= 0.5 for z in ign.get(f, [])):
                fp += 1
    return dict(recall=tp/ngt, recall_small=tps/max(ngs, 1), precision=tp/max(tp+fp, 1), fp_per_frame=fp/n, tp=tp)

if __name__ == "__main__":
    ann, n = sys.argv[1], int(sys.argv[2])
    for p in sys.argv[3:]:
        r = run(ann, n, p)
        print(f"{p.split('/')[-1]:38s} recall {r['recall']:.3f} small {r['recall_small']:.3f} precision {r['precision']:.3f} FP/frame {r['fp_per_frame']:.2f}")
