#!/usr/bin/env python3
"""COCO-style mAP of detectors on the labelled 800-image evaluation subset (eval_val + eval_test:
VisDrone val/test-dev full frames and zoom crops, all 832x480, with ignore regions).

  pack                   -> eval/eval800.nv12 (+ eval/images.json): the board input "video"
  host  --onnx M.onnx    -> FP32 reference detections, same NV12 source / letterbox / decode /
                            NMS as the board (isolates the INT8 quantization gap)
  score --dets D.jsonl   -> metrics json: mAP50, mAP50-95, per class, per size bucket

Protocol (identical for every model): score >= 0.001, best class per anchor, per-class NMS
IoU 0.7, max 300 boxes, IoU 0.50:0.95 101-point AP (pycocotools). Ignore regions are crowd
boxes of every class (detections inside them are neither TP nor FP). Size buckets are on
sqrt(box area) in the 832x480 image: 12-24, 24-48, 48-96, >96 px (deploy_config SIZE_BUCKETS).
Images are streamed; nothing holds the whole set in RAM.
"""
import argparse, contextlib, glob, io, json, os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.environ.get("EVAL800_DIR", os.path.join(HERE, "eval800"))  # contains eval_val/ and eval_test/ (images, labels, ignore)
EVAL = os.environ.get("EVAL_WORK", os.path.join(HERE, "eval_work"))
NAMES = ["person", "two_wheeler", "car", "heavy"]
BUCKETS = [(12, 24), (24, 48), (48, 96), (96, 1e5)]
W, H = 832, 480


def image_list():
    return sorted(glob.glob(os.path.join(PKG, "eval_val/images/*.jpg"))) + \
        sorted(glob.glob(os.path.join(PKG, "eval_test/images/*.jpg")))


def to_nv12(bgr):
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y, u, v = i420[:W * H], i420[W * H:W * H * 5 // 4], i420[W * H * 5 // 4:]
    uv = np.empty(W * H // 2, np.uint8)
    uv[0::2], uv[1::2] = u, v
    return np.concatenate([y, uv])


def even_floor(v):
    return int(np.floor(v / 2.0)) * 2


def letterbox_geom(in_w, in_h):
    """Mirror of RknnDetector::detect for a whole 832x480 frame."""
    s = min(in_w / W, in_h / H)
    dw = min(in_w, (int(W * s) + 15) // 16 * 16)
    dh = min(in_h, even_floor(H * s))
    return dw, dh, even_floor((in_w - dw) * 0.5), even_floor((in_h - dh) * 0.5)


def nms(dets, thr):
    dets = sorted(dets, key=lambda d: -d[4])
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            ix = max(0, min(d[0] + d[2], k[0] + k[2]) - max(d[0], k[0]))
            iy = max(0, min(d[1] + d[3], k[1] + k[3]) - max(d[1], k[1]))
            inter = ix * iy
            if inter / max(1e-9, d[2] * d[3] + k[2] * k[3] - inter) > thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def host_dets(onnx_path, out_path, conf=0.001):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ih, iw = sess.get_inputs()[0].shape[2:]
    dw, dh, dx, dy = letterbox_geom(iw, ih)
    kx, ky = W / dw, H / dh
    names = [o.name for o in sess.get_outputs()]
    imgs = json.load(open(os.path.join(EVAL, "images.json")))
    with open(os.path.join(EVAL, "eval800.nv12"), "rb") as f, open(out_path, "w") as fo:
        for idx in range(len(imgs)):
            nv = np.frombuffer(f.read(W * H * 3 // 2), np.uint8).reshape(H * 3 // 2, W)
            rgb = cv2.cvtColor(nv, cv2.COLOR_YUV2RGB_NV12)
            canvas = np.full((ih, iw, 3), 114, np.uint8)
            canvas[dy:dy + dh, dx:dx + dw] = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR) \
                if (dw, dh) != (W, H) else rgb
            x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            outs = dict(zip(names, sess.run(None, {"images": x})))
            dets = []
            for n in names:
                if not n.startswith("box_"):
                    continue
                s = int(n.split("_s")[1])
                box, cls = outs[n][0], outs["cls_s%d" % s][0]
                gh, gw = cls.shape[1:]
                c = cls.reshape(4, -1)
                best, bc = c.max(0), c.argmax(0)
                sel = np.nonzero(best >= conf)[0]
                if not len(sel):
                    continue
                b = box.reshape(4, 16, -1)[:, :, sel]
                e = np.exp(b - b.max(1, keepdims=True))
                dist = (e * np.arange(16)[None, :, None]).sum(1) / e.sum(1)
                ax, ay = sel % gw + 0.5, sel // gw + 0.5
                x1 = np.maximum((ax - dist[0]) * s * kx - dx * kx, 0)
                y1 = np.maximum((ay - dist[1]) * s * ky - dy * ky, 0)
                x2 = np.minimum((ax + dist[2]) * s * kx - dx * kx, W)
                y2 = np.minimum((ay + dist[3]) * s * ky - dy * ky, H)
                for j in range(len(sel)):
                    if x2[j] - x1[j] >= 3 and y2[j] - y1[j] >= 3:
                        dets.append((float(x1[j]), float(y1[j]), float(x2[j] - x1[j]), float(y2[j] - y1[j]),
                                     float(best[sel[j]]), int(bc[sel[j]])))
            kept = []
            for cc in range(4):
                kept += nms([d for d in dets if d[5] == cc], 0.7)
            kept = sorted(kept, key=lambda d: -d[4])[:300]
            fo.write(json.dumps({"f": idx, "dets": [list(d) for d in kept]}) + "\n")


def build_gt():
    imgs = json.load(open(os.path.join(EVAL, "images.json")))
    gt = {"images": [], "annotations": [], "categories": [{"id": c, "name": n} for c, n in enumerate(NAMES)]}
    aid = 1
    for i, p in enumerate(imgs):
        gt["images"].append({"id": i, "width": W, "height": H})
        lab = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        for line in open(lab) if os.path.exists(lab) else []:
            c, cx, cy, bw, bh = map(float, line.split()[:5])
            w, h = bw * W, bh * H
            gt["annotations"].append({"id": aid, "image_id": i, "category_id": int(c), "iscrowd": 0, "area": w * h,
                                      "bbox": [(cx - bw / 2) * W, (cy - bh / 2) * H, w, h]})
            aid += 1
        ign = p.replace("/images/", "/ignore/").rsplit(".", 1)[0] + ".txt"
        for line in open(ign) if os.path.exists(ign) else []:
            x1, y1, x2, y2 = map(float, line.split()[:4])
            for c in range(4):
                gt["annotations"].append({"id": aid, "image_id": i, "category_id": c, "iscrowd": 1,
                                          "area": (x2 - x1) * (y2 - y1), "bbox": [x1, y1, x2 - x1, y2 - y1]})
                aid += 1
    return gt


def score(dets_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO()
    with contextlib.redirect_stdout(io.StringIO()):
        coco.dataset = build_gt()
        coco.createIndex()
    res = []
    for line in open(dets_path):
        r = json.loads(line)
        for x, y, w, h, s, c in r["dets"]:
            res.append({"image_id": r["f"], "category_id": int(c), "bbox": [x, y, w, h], "score": s})
    dt = coco.loadRes(res) if res else COCO()
    out = {}

    def run(area_rng, cats=None):
        E = COCOeval(coco, dt, "bbox")
        E.params.maxDets = [300]
        E.params.areaRng = [area_rng]
        E.params.areaRngLbl = ["r"]
        if cats is not None:
            E.params.catIds = cats
        with contextlib.redirect_stdout(io.StringIO()):
            E.evaluate(); E.accumulate()
        p = E.eval["precision"][:, :, :, 0, 0]  # T x R x K
        def ap(t):
            q = p[t] if t is not None else p
            q = q[q > -1]
            return float(q.mean()) if q.size else float("nan")
        return ap(0), ap(None), p

    a50, a5095, p = run([0, 1e10])
    out["mAP50"], out["mAP50_95"] = a50, a5095
    out["per_class_AP50"] = {}
    for k, n in enumerate(NAMES):
        q = p[0, :, k]; q = q[q > -1]
        out["per_class_AP50"][n] = float(q.mean()) if q.size else float("nan")
    out["bucket_AP50"], out["bucket_AP50_95"] = {}, {}
    for lo, hi in BUCKETS:
        b50, b5095, _ = run([lo * lo, hi * hi])
        key = f"{lo}-{int(hi) if hi < 1e5 else 'inf'}"
        out["bucket_AP50"][key], out["bucket_AP50_95"][key] = b50, b5095
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "host", "score"])
    ap.add_argument("--onnx")
    ap.add_argument("--dets")
    ap.add_argument("--out")
    a = ap.parse_args()
    os.makedirs(EVAL, exist_ok=True)
    if a.cmd == "pack":
        imgs = image_list()
        with open(os.path.join(EVAL, "eval800.nv12"), "wb") as f:
            for p in imgs:
                im = cv2.imread(p)
                assert im.shape[:2] == (H, W), p
                f.write(to_nv12(im).tobytes())
        json.dump(imgs, open(os.path.join(EVAL, "images.json"), "w"))
        print(f"[pack] {len(imgs)} images")
    elif a.cmd == "host":
        host_dets(a.onnx, a.out)
        print(f"[host] {a.out}")
    else:
        m = score(a.dets)
        if a.out:
            json.dump(m, open(a.out, "w"), indent=1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()})
                          for k, v in m.items()}))


if __name__ == "__main__":
    main()
