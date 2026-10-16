# =====================================================================================================
# Dataset preparation: VisDrone2019-DET -> vd_deploy_v3 (832x480, four merged classes) -> ZIP
# -----------------------------------------------------------------------------------------------------
# Runs on a CPU (no GPU needed). On Kaggle, paste the file into one notebook cell; locally, set VD_WORK=<dir>.
# Outputs:
#   <WORK>/vd_deploy_v3/            dataset directory
#   <WORK>/vd_deploy_v3.zip         archive of the dataset (optional)
#   <WORK>/preview/*.jpg            visual check (green: ground truth, red: ignore regions)
#
# On Kaggle, the output can be attached to the training notebook directly (Add Input -> Notebook Output
# Files); train/train_dual.py finds the dataset either as a directory or as the extracted ZIP.
#
# Dataset layout:
#   train/     full frames + one ROI crop per image; ignore regions optionally blurred; motion blur / JPEG degradation
#   val_sel/   full frames only, ignore regions blurred; used by Ultralytics to select best.pt (no ROI crops, no leakage)
#   val_raw/   full frames + ROI crops, not blurred, with ignore files; used for the final evaluation
#   test_raw/  same as val_raw, built from VisDrone2019-DET-test-dev
#   calib/     INT8 calibration images (full frames + ROI crops)
#   meta.json  size thresholds, class weights, statistics and build configuration (for reproducibility)
# =====================================================================================================
import os, sys, json, math, random, shutil, time, zlib, zipfile
import multiprocessing as mp
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2
from PIL import Image

# =====================================================================================================
# CONFIG
# =====================================================================================================
DATASET_ROOT = None                      # None = search /kaggle/input
WORK         = Path(os.environ.get("VD_WORK", "/kaggle/working"))
OUT_NAME     = "vd_deploy_v3"
OUT          = WORK / OUT_NAME
VERSION      = "3.0"

SRC_SPLITS = dict(train="VisDrone2019-DET-train", val="VisDrone2019-DET-val", test="VisDrone2019-DET-test-dev")

W, H   = 832, 480                         # network input (multiple of 32)
REF_H  = 1080                             # reference camera height (1080p gimbal camera)

# Size thresholds (short side, in pixels of the network input); kept fixed across all experiments
SIZE_FLOOR_PX = 8
DEGENERATE_PX = 2                         # boxes with a short side < 2 source px are dropped
THRESH        = dict(mode="auto", alpha=0.4)
MIN_PX_ROI    = {0: 8, 1: 8, 2: 11, 3: 15}    # used when mode="manual"
MIN_PX_FULL   = {0: 8, 1: 8, 2: 8, 3: 8}
ZOOMS         = (0.6, 0.8, 1.0)

CLASS_BALANCE = dict(enabled=True, max_ratio=8.0, dampening=0.5, quality_sat=2.0, quality_floor=0.15)

ORIG_NAMES  = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
CLASS_NAMES = ["person", "two_wheeler", "car", "heavy"]
CLASS_MAP_YOLO = {0: 0, 1: 0, 2: 1, 9: 1, 3: 2, 4: 2, 5: 3, 8: 3}           # YOLO labels, 0-indexed
CLASS_MAP_ORIG = {1: 0, 2: 0, 3: 1, 10: 1, 4: 2, 5: 2, 6: 3, 9: 3}          # original annotations (0 = ignored region)

BUILD = dict(
    crops_per_image   = 1,       # ROI crops per training image
    train_jitter      = 0.35,
    train_p_rand      = 0.15,
    eval_jitter       = 0.10,    # val/test ROI roughly centered on an object (simulated tracking crop)
    blur_ignore_train = True,    # blur ignore regions in training images
    degrade_p         = 0.15,    # motion-blur probability (training only)
    jpeg_q_range      = (55, 96),# random JPEG quality (training only)
    num_workers       = 4,
    limit             = 0,       # debug: N images per split (0 = all)
    seed              = 42,
    calib_full        = 250,
    calib_crop        = 250,
    use_test_dev      = True,
)
ZIP = dict(enabled=True, delete_folder_after_zip=False)
PREVIEW_N = 4

# =====================================================================================================
# TIỆN ÍCH
# =====================================================================================================
IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_imgs(d):
    d = Path(d)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


def find_raw_root():
    if DATASET_ROOT:
        return Path(DATASET_ROOT)
    for c in ["/kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset",
              "/kaggle/input/visdrone-dataset/VisDrone_Dataset", "/kaggle/input/visdrone-dataset"]:
        if (Path(c) / SRC_SPLITS["train"]).is_dir():
            return Path(c)
    for root, dirs, _ in os.walk("/kaggle/input"):
        if SRC_SPLITS["train"] in dirs:
            return Path(root)
    raise FileNotFoundError("VisDrone2019-DET-train not found under /kaggle/input")


def read_objects(split_dir, img_path, iw, ih):
    """Returns boxes (N,4) xyxy in source pixels and mapped classes (N,), -1 = ignore region. Original annotations take precedence."""
    split_dir, img_path = Path(split_dir), Path(img_path)
    boxes, cls = [], []
    ann = split_dir / "annotations" / f"{img_path.stem}.txt"
    if ann.exists():
        for line in ann.read_text().splitlines():
            t = [v for v in line.strip().split(",") if v != ""]
            if len(t) < 6:
                continue
            x, y, w, h = (float(v) for v in t[:4])
            if min(w, h) < DEGENERATE_PX:
                continue
            boxes.append([x, y, x + w, y + h]); cls.append(CLASS_MAP_ORIG.get(int(t[5]), -1))
    else:
        lbl = split_dir / "labels" / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                t = line.split()
                if len(t) != 5:
                    continue
                c = int(float(t[0])); cx, cy, w, h = (float(v) for v in t[1:])
                if min(w * iw, h * ih) < DEGENERATE_PX:
                    continue
                boxes.append([(cx - w / 2) * iw, (cy - h / 2) * ih, (cx + w / 2) * iw, (cy + h / 2) * ih])
                cls.append(CLASS_MAP_YOLO.get(c, -1))
    return np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(cls, np.int64)


def scan_split(split_dir, limit=0):
    """Reads image sizes (PIL header only, no decoding) and labels of one split."""
    out = []
    imgs = list_imgs(Path(split_dir) / "images")
    if limit:
        imgs = imgs[:limit]
    for p in imgs:
        try:
            iw, ih = Image.open(p).size
        except Exception:
            continue
        b, c = read_objects(split_dir, p, iw, ih)
        out.append((p, iw, ih, b, c))
    return out


# =====================================================================================================
# NGƯỠNG KÍCH THƯỚC + TRỌNG SỐ LỚP
# =====================================================================================================
def quality_score(size_px, thr_px, sat=2.0):
    return np.clip((size_px / np.maximum(thr_px, 1e-6) - 1) / max(sat - 1, 1e-6), 0.0, 1.0)


def compute_thresholds(objs):
    if THRESH["mode"] != "auto":
        f = lambda m: {int(c): float(max(v, SIZE_FLOOR_PX)) for c, v in m.items()}
        return f(MIN_PX_ROI), f(MIN_PX_FULL)
    sizes = {i: [] for i in range(len(CLASS_NAMES))}
    hw, r0s, kfs = Counter(), [], []
    for _, iw, ih, b, c in objs:
        hw[(iw, ih)] += 1
        r0 = REF_H / ih; r0s.append(r0); kfs.append(min(W / iw, H / ih) / r0)
        if len(b):
            m = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) * r0
            for ci, mi in zip(c, m):
                if ci >= 0:
                    sizes[int(ci)].append(float(mi))
    k_full = float(np.median(kfs)) if kfs else H / REF_H
    r0a = np.asarray(r0s) if r0s else np.ones(1)
    print("   most common source sizes:", [(f"{a}x{b}", n) for (a, b), n in hw.most_common(5)])
    print(f"   r0=REF_H/ih p10/p50/p90 = {np.percentile(r0a, 10):.2f}/{np.percentile(r0a, 50):.2f}/"
          f"{np.percentile(r0a, 90):.2f}   k_full={k_full:.3f}")
    alpha = THRESH["alpha"]
    roi_map, full_map = {}, {}
    print(f"   {'class':12s}{'n':>8s}{'p10':>7s}{'median':>8s}{'p90':>7s}{'thr_roi':>9s}{'thr_full':>10s}"
          f"{'kept roi':>10s}{'kept full':>11s}")
    for c, name in enumerate(CLASS_NAMES):
        v = np.asarray(sizes[c])
        if len(v) == 0:
            roi_map[c] = full_map[c] = float(SIZE_FLOOR_PX); continue
        med = float(np.median(v))
        roi_map[c] = float(max(SIZE_FLOOR_PX, alpha * med))
        full_map[c] = float(max(SIZE_FLOOR_PX, alpha * med * k_full))
        print(f"   {name:12s}{len(v):>8d}{np.percentile(v, 10):>7.1f}{med:>8.1f}{np.percentile(v, 90):>7.1f}"
              f"{roi_map[c]:>9.1f}{full_map[c]:>10.1f}{(v >= roi_map[c]).mean() * 100:>9.0f}%"
              f"{(v * k_full >= full_map[c]).mean() * 100:>10.0f}%")
    return roi_map, full_map


def compute_class_weights(objs, full_map, roi_map):
    q_sat = CLASS_BALANCE["quality_sat"]
    total, surv_full, surv_roi, eff_roi = Counter(), Counter(), Counter(), Counter()
    for _, iw, ih, b, c in objs:
        if not len(b):
            continue
        r0 = REF_H / ih; f_full = min(W / iw, H / ih); f_roi = max(ZOOMS) * r0
        m = np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        for ci, mi in zip(c, m):
            if ci < 0:
                continue
            ci = int(ci); total[ci] += 1
            if mi * f_full >= full_map[ci]:
                surv_full[ci] += 1
            s_roi = mi * f_roi
            if s_roi >= roi_map[ci]:
                surv_roi[ci] += 1
                eff_roi[ci] += float(quality_score(np.array([s_roi]), np.array([roi_map[ci]]), q_sat)[0])
    print(f"   {'class':14s}{'total GT':>10s}{'kept full':>14s}{'kept roi':>12s}{'roi (q-weighted)':>18s}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"   {n:14s}{total[i]:>10d}{surv_full[i]:>14d}{surv_roi[i]:>12d}{eff_roi[i]:>18.0f}")
    cnt = np.array([max(eff_roi[i], 1e-6) for i in range(len(CLASS_NAMES))])
    w = np.clip((cnt.max() / cnt) ** CLASS_BALANCE["dampening"], 1.0 / CLASS_BALANCE["max_ratio"],
                CLASS_BALANCE["max_ratio"])
    weights = {i: float(w[i] / w.min()) for i in range(len(CLASS_NAMES))}
    print("   ROI sampling weight per class:", {CLASS_NAMES[i]: round(v, 2) for i, v in weights.items()})
    stats = dict(total=[total[i] for i in range(len(CLASS_NAMES))],
                 surv_full=[surv_full[i] for i in range(len(CLASS_NAMES))],
                 surv_roi=[surv_roi[i] for i in range(len(CLASS_NAMES))],
                 eff_roi=[round(eff_roi[i], 1) for i in range(len(CLASS_NAMES))])
    return weights, stats


# =====================================================================================================
# SINH MẪU
# =====================================================================================================
def motion_blur(img, rng):
    k = int(rng.integers(5, 15)) | 1
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1
    M = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), float(rng.uniform(0, 180)), 1)
    ker = cv2.warpAffine(ker, M, (k, k)); s = ker.sum()
    return img if s <= 0 else cv2.filter2D(img, -1, ker / s)


def blur_regions(canvas, regs):
    for x1, y1, x2, y2 in regs:
        x1, y1 = int(max(0, math.floor(x1))), int(max(0, math.floor(y1)))
        x2, y2 = int(min(W, math.ceil(x2))), int(min(H, math.ceil(y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        k = int(np.clip(min(x2 - x1, y2 - y1), 3, 51)) | 1
        X1, Y1, X2, Y2 = max(0, x1 - k), max(0, y1 - k), min(W, x2 + k), min(H, y2 + k)
        patch = cv2.GaussianBlur(canvas[Y1:Y2, X1:X2], (k, k), 0)
        canvas[y1:y2, x1:x2] = patch[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1]


def make_sample(img, boxes, mcls, src, size, dst, min_px_map):
    x0, y0, x1, y1 = src; nw, nh = size; dx, dy = dst
    interp = cv2.INTER_AREA if nw < (x1 - x0) else cv2.INTER_CUBIC
    canvas = np.full((H, W, 3), 114, np.uint8)
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img[y0:y1, x0:x1], (nw, nh), interpolation=interp)
    fx, fy = nw / (x1 - x0), nh / (y1 - y0)
    pos, ign = [], []
    for b, c in zip(boxes, mcls):
        bx1, by1 = (b[0] - x0) * fx + dx, (b[1] - y0) * fy + dy
        bx2, by2 = (b[2] - x0) * fx + dx, (b[3] - y0) * fy + dy
        area0 = max(bx2 - bx1, 1e-6) * max(by2 - by1, 1e-6)
        cx1, cy1, cx2, cy2 = max(bx1, dx), max(by1, dy), min(bx2, dx + nw), min(by2, dy + nh)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        frac = (cx2 - cx1) * (cy2 - cy1) / area0
        if frac < 0.1:
            continue
        thr = min_px_map.get(int(c), 1e9) if c >= 0 else 1e9
        if c >= 0 and frac >= 0.6 and min(cx2 - cx1, cy2 - cy1) >= thr:
            pos.append((int(c), cx1, cy1, cx2, cy2))
        else:
            ign.append((cx1, cy1, cx2, cy2))
    return canvas, pos, ign


def pick_crops(rng, boxes, mcls, iw, ih, r0, n, jitter, p_rand, roi_map, class_weights=None):
    q_sat, q_floor = CLASS_BALANCE["quality_sat"], CLASS_BALANCE["quality_floor"]
    out = []
    for k in range(n):
        s = float(rng.choice(ZOOMS))
        f = max(s * r0, W / iw, H / ih)
        ww, wh = min(int(round(W / f)), iw), min(int(round(H / f)), ih)
        if not len(boxes):
            continue
        ms = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * f
        thr = np.array([roi_map.get(int(c), 1e9) if c >= 0 else 1e9 for c in mcls])
        ok = np.where((mcls >= 0) & (ms >= thr))[0]
        if len(ok) == 0:
            continue
        if rng.random() < p_rand:
            cx, cy = rng.uniform(ww / 2, iw - ww / 2 + 1e-6), rng.uniform(wh / 2, ih - wh / 2 + 1e-6)
        else:
            pool = ok
            if class_weights:
                present = np.unique(mcls[ok])
                wts = np.array([class_weights.get(int(c), 1.0) for c in present])
                chosen = int(rng.choice(present, p=wts / wts.sum()))
                pool = ok[mcls[ok] == chosen]
            qw = quality_score(ms[pool], thr[pool], q_sat) + q_floor
            j = int(rng.choice(pool, p=qw / qw.sum()))
            cx = (boxes[j, 0] + boxes[j, 2]) / 2 + rng.uniform(-jitter, jitter) * ww
            cy = (boxes[j, 1] + boxes[j, 3]) / 2 + rng.uniform(-jitter, jitter) * wh
        x0 = int(np.clip(cx - ww / 2, 0, iw - ww)); y0 = int(np.clip(cy - wh / 2, 0, ih - wh))
        out.append((f"z{int(round(s * 100))}_c{k}", (x0, y0, x0 + ww, y0 + wh), (W, H), (0, 0)))
    return out


def write_out(sub, name, canvas, pos, ign, o, rng):
    d = OUT / sub
    im = canvas.copy()
    if o["blur"] and ign:
        blur_regions(im, ign)
    q = 95
    if o["degrade"]:
        if rng.random() < BUILD["degrade_p"]:
            im = motion_blur(im, rng)
        q = int(rng.integers(*BUILD["jpeg_q_range"]))
    cv2.imwrite(str(d / "images" / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, q])
    lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
             for c, x1, y1, x2, y2 in pos]
    (d / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    if o["write_ign"]:
        (d / "ignore" / f"{name}.txt").write_text(
            "".join(f"{a:.1f} {b:.1f} {c:.1f} {e:.1f}\n" for a, b, c, e in ign))


def process_one(job):
    img_path, split_dir, cfg = job
    cv2.setNumThreads(0)
    img_path = Path(img_path); stem = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        return {}
    ih, iw = img.shape[:2]
    boxes, mcls = read_objects(split_dir, img_path, iw, ih)
    rng = np.random.default_rng((zlib.crc32(stem.encode()) + cfg["seed"]) % (2 ** 32))
    r0 = REF_H / ih
    f = min(W / iw, H / ih)
    nw, nh = min(W, int(round(iw * f))), min(H, int(round(ih * f)))
    plans = [("full", (0, 0, iw, ih), (nw, nh), ((W - nw) // 2, (H - nh) // 2), cfg["full_map"])]
    for tag, src, size, dst in pick_crops(rng, boxes, mcls, iw, ih, r0, cfg["n_crops"], cfg["jitter"],
                                          cfg["p_rand"], cfg["roi_map"], cfg["class_weights"]):
        plans.append((tag, src, size, dst, cfg["roi_map"]))
    st = Counter()
    for tag, src, size, dst, mp_ in plans:
        canvas, pos, ign = make_sample(img, boxes, mcls, src, size, dst, mp_)
        if tag != "full" and not pos:
            continue
        kind = "full" if tag == "full" else "roi"
        for sub, o in cfg["outputs"]:
            if o["full_only"] and kind != "full":
                continue
            write_out(sub, f"{stem}__{tag}", canvas, pos, ign, o, rng)
        st[f"n_{kind}"] += 1
        for c, *_ in pos:
            st[f"{kind}_{c}"] += 1
    return dict(st)


def run_jobs(jobs, workers):
    if workers <= 1:
        return [process_one(j) for j in jobs]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(process_one, jobs, chunksize=8))


def seq_of(stem):
    return "_".join(stem.split("_")[:2])


# =====================================================================================================
# BUILD
# =====================================================================================================
def build():
    raw = find_raw_root()
    print(f"[Source] {raw}\n[Output] {OUT}")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    b = BUILD
    train_dir = raw / SRC_SPLITS["train"]
    label_src = "original annotations" if (train_dir / "annotations").is_dir() else "YOLO labels"
    print(f"[Labels] {label_src}")

    t0 = time.time()
    print("\n[1/4] Prescan of the training split: per-class size thresholds ...")
    objs = scan_split(train_dir, b["limit"])
    roi_map, full_map = compute_thresholds(objs)
    print("   ROI threshold :", {CLASS_NAMES[c]: round(v, 1) for c, v in roi_map.items()})
    print("   full threshold:", {CLASS_NAMES[c]: round(v, 1) for c, v in full_map.items()})
    print("\n[2/4] Per-class sampling weights ...")
    cw, cw_stats = compute_class_weights(objs, full_map, roi_map) if CLASS_BALANCE["enabled"] else (None, {})
    print(f"   prescan {time.time() - t0:.0f}s")

    O_TRAIN = dict(blur=b["blur_ignore_train"], write_ign=True, degrade=True, full_only=False)
    O_SEL = dict(blur=True, write_ign=False, degrade=False, full_only=True)
    O_RAW = dict(blur=False, write_ign=True, degrade=False, full_only=False)
    plans = [("train", [("train", O_TRAIN)], b["crops_per_image"], b["train_jitter"], b["train_p_rand"], cw),
             ("val", [("val_sel", O_SEL), ("val_raw", O_RAW)], 1, b["eval_jitter"], 0.0, None)]
    if b["use_test_dev"]:
        plans.append(("test", [("test_raw", O_RAW)], 1, b["eval_jitter"], 0.0, None))

    print("\n[3/4] Generating samples ...")
    split_stats, seqs = {}, {}
    for key, outputs, n_crops, jitter, p_rand, cw_ in plans:
        d = raw / SRC_SPLITS[key]
        if not (d / "images").is_dir():
            print(f"   [skip] {key}: {d}/images not found"); continue
        for sub, o in outputs:
            for x in ["images", "labels"] + (["ignore"] if o["write_ign"] else []):
                (OUT / sub / x).mkdir(parents=True, exist_ok=True)
        imgs = list_imgs(d / "images")
        if b["limit"]:
            imgs = imgs[:b["limit"]]
        seqs[key] = {seq_of(p.stem) for p in imgs}
        cfg = dict(n_crops=n_crops, jitter=jitter, p_rand=p_rand, roi_map=roi_map, full_map=full_map,
                   class_weights=cw_, outputs=outputs, seed=b["seed"])
        t1 = time.time()
        res = run_jobs([(str(p), str(d), cfg) for p in imgs], b["num_workers"])
        agg = Counter()
        for r in res:
            agg.update(r)
        split_stats[key] = dict(agg)
        print(f"   {key}: {len(imgs)} source images -> full={agg['n_full']} roi={agg['n_roi']} ({time.time() - t1:.0f}s)")

    ks = list(seqs)
    leak = {}
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            n = len(seqs[ks[i]] & seqs[ks[j]]); leak[f"{ks[i]}&{ks[j]}"] = n
            print(f"   [leak-check] {ks[i]} ∩ {ks[j]} = {n} sequences (expected 0)")

    print("\n[4/4] Calibration INT8 ...")
    tr = sorted((OUT / "train" / "images").glob("*.jpg"))
    full = [p for p in tr if "__full" in p.stem]; crop = [p for p in tr if "__full" not in p.stem]
    rng = np.random.default_rng(b["seed"])
    pick = list(rng.choice(full, min(b["calib_full"], len(full)), replace=False)) if full else []
    pick += list(rng.choice(crop, min(b["calib_crop"], len(crop)), replace=False)) if crop else []
    (OUT / "calib").mkdir(parents=True, exist_ok=True)
    for p in pick:
        shutil.copy(p, OUT / "calib" / Path(p).name)
    print(f"   {len(pick)} images")

    meta = dict(
        dataset=OUT_NAME, version=VERSION, created=time.strftime("%Y-%m-%d %H:%M:%S"),
        input_w=W, input_h=H, ref_h=REF_H, class_names=CLASS_NAMES, orig_names=ORIG_NAMES,
        class_map_yolo=CLASS_MAP_YOLO, label_source=label_src,
        thresholds=dict(mode=THRESH["mode"], alpha=THRESH["alpha"], floor=SIZE_FLOOR_PX,
                        degenerate_px=DEGENERATE_PX,
                        min_px_roi={CLASS_NAMES[c]: v for c, v in roi_map.items()},
                        min_px_full={CLASS_NAMES[c]: v for c, v in full_map.items()}),
        zooms=list(ZOOMS),
        roi_sampling_class_weights={CLASS_NAMES[c]: v for c, v in (cw or {}).items()},
        class_stats_train=cw_stats, split_stats=split_stats, leak_check=leak,
        build={k: (list(v) if isinstance(v, tuple) else v) for k, v in BUILD.items()},
        splits=dict(train="full+roi, ignore blurred" if b["blur_ignore_train"] else "full+roi",
                    val_sel="full only, ignore blurred (best.pt selection)",
                    val_raw="full + ROI crops, not blurred, with ignore regions (final evaluation)",
                    test_raw="as val_raw, from test-dev"),
    )
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    summarize()
    print(f"\n[Done] build {time.time() - t0:.0f}s")


def summarize():
    for sub in ["train", "val_sel", "val_raw", "test_raw"]:
        d = OUT / sub / "labels"
        if not d.exists():
            continue
        cf, cr, ms = Counter(), Counter(), []
        files = list(d.glob("*.txt"))
        for f in files:
            tgt = cf if "__full" in f.stem else cr
            for line in f.read_text().splitlines():
                t = line.split(); tgt[int(t[0])] += 1
                ms.append(min(float(t[3]) * W, float(t[4]) * H))
        ms = np.asarray(ms) if ms else np.zeros(1)
        nfull = sum("__full" in f.stem for f in files)
        print(f"\n[{sub}] samples={len(files)} (full={nfull}, roi={len(files) - nfull})")
        print("   full:", {CLASS_NAMES[k]: cf.get(k, 0) for k in range(len(CLASS_NAMES))})
        print("   roi :", {CLASS_NAMES[k]: cr.get(k, 0) for k in range(len(CLASS_NAMES))})
        print(f"   short side px p10/p50/p90 = {np.percentile(ms, 10):.0f}/{np.percentile(ms, 50):.0f}/{np.percentile(ms, 90):.0f}")


# =====================================================================================================
# PREVIEW + ZIP
# =====================================================================================================
def preview():
    pv = WORK / "preview"
    pv.mkdir(exist_ok=True)
    rnd = random.Random(0)
    for sub in ["train", "val_raw"]:
        files = sorted((OUT / sub / "images").glob("*.jpg"))
        for f in rnd.sample(files, min(PREVIEW_N, len(files))):
            im = cv2.imread(str(f))
            for line in (OUT / sub / "labels" / f"{f.stem}.txt").read_text().splitlines():
                c, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
                p1, p2 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
                cv2.rectangle(im, p1, p2, (0, 255, 0), 1)
            ig = OUT / sub / "ignore" / f"{f.stem}.txt"
            if ig.exists():
                for line in ig.read_text().splitlines():
                    x1, y1, x2, y2 = [int(float(v)) for v in line.split()]
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imwrite(str(pv / f"{sub}__{f.name}"), im)
    print(f"[Preview] {pv}")


def make_zip():
    zp = WORK / f"{OUT_NAME}.zip"
    if zp.exists():
        zp.unlink()
    files = [p for p in OUT.rglob("*") if p.is_file()]
    t0 = time.time()
    with zipfile.ZipFile(zp, "w", allowZip64=True) as z:
        for i, p in enumerate(files):
            ct = zipfile.ZIP_STORED if p.suffix.lower() in IMG_EXT else zipfile.ZIP_DEFLATED
            z.write(p, arcname=str(p.relative_to(WORK)), compress_type=ct)
            if (i + 1) % 5000 == 0:
                print(f"   zip {i + 1}/{len(files)}")
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
    assert bad is None, f"ZIP CRC error in {bad}"
    print(f"[ZIP] {zp}  {zp.stat().st_size / 1e9:.2f} GB, {len(files)} file, {time.time() - t0:.0f}s (CRC verified)")
    if ZIP["delete_folder_after_zip"]:
        shutil.rmtree(OUT)


if __name__ == "__main__":
    build()
    preview()
    if ZIP["enabled"]:
        make_zip()
    print("\nNext step: run train/train_dual.py with this dataset attached (on Kaggle: Add Input -> "
          "Notebook Output Files, or upload vd_deploy_v3.zip as a dataset).")