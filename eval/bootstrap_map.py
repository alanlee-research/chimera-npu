#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for mAP differences on the 800-image eval set.

Same scoring as eval_map.py: pycocotools, maxDets 300, ignore regions as crowd. COCOeval.evaluate()
runs once per model and size range. Each bootstrap replicate then re-runs only the accumulate step
on the resampled images, with repeated images counted repeatedly.

Resampling is clustered by VisDrone sequence (split + sequence id). Images from one sequence are
strongly correlated, and the full frame and ROI crop of one source image share a sequence, so
sequence clusters give a conservative interval. Image-level resampling is reported as well.

  python bootstrap_map.py [--B 1000] [--seed 0]  ->  results/int8_800/bootstrap.json + table on stdout
"""
import argparse, contextlib, io, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_map  # noqa: E402

RES = os.environ.get("INT8_RESULTS_DIR", os.path.join(HERE, "..", "results", "int8_800"))  # <model>_832x480_{fp32,int8}.{dets.jsonl,metrics.json}
REC = np.linspace(0.0, 1.0, 101)
RANGES = {"all": [0, 1e10], "12-24": [12 * 12, 24 * 24]}
MODELS = ["yolo11n_silu", "chimera_proto", "chimera_silu", "yolo11n_lrelu", "chimera_lrelu"]
# (label, model A, model B): B - A, INT8 on the board
PAIRS = [("chimera_lrelu - yolo11n_lrelu (INT8)", "yolo11n_lrelu", "chimera_lrelu"),
         ("chimera_proto - yolo11n_silu (INT8)", "yolo11n_silu", "chimera_proto"),
         ("chimera_silu - yolo11n_silu (INT8)", "yolo11n_silu", "chimera_silu")]


def per_image(coco, dets_path, area):
    """Per (class, image): detection scores, match flags per IoU threshold, ignore flags, #GT."""
    from pycocotools.cocoeval import COCOeval
    res = []
    for line in open(dets_path):
        r = json.loads(line)
        for x, y, w, h, s, c in r["dets"]:
            res.append({"image_id": r["f"], "category_id": int(c), "bbox": [x, y, w, h], "score": s})
    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(res)
        E = COCOeval(coco, dt, "bbox")
        E.params.maxDets = [300]
        E.params.areaRng = [area]
        E.params.areaRngLbl = ["r"]
        E.evaluate()
    K, I = len(E.params.catIds), len(E.params.imgIds)
    out = [[None] * I for _ in range(K)]
    for k in range(K):
        for i in range(I):
            e = E.evalImgs[k * I + i]
            if e is None:
                continue
            o = np.argsort(-np.asarray(e["dtScores"]), kind="mergesort")[:300]
            out[k][i] = (np.asarray(e["dtScores"])[o], e["dtMatches"][:, o] > 0, e["dtIgnore"][:, o].astype(bool),
                         int(np.count_nonzero(e["gtIgnore"] == 0)))
    return out


def ap_table(pi, idx):
    """precision[T, R, K] like COCOeval.accumulate, over the image multiset idx."""
    K = len(pi)
    T = 10
    prec = -np.ones((T, len(REC), K))
    for k in range(K):
        es = [pi[k][i] for i in idx if pi[k][i] is not None]
        if not es:
            continue
        npig = sum(e[3] for e in es)
        if npig == 0:
            continue
        sc = np.concatenate([e[0] for e in es])
        o = np.argsort(-sc, kind="mergesort")
        dtm = np.concatenate([e[1] for e in es], axis=1)[:, o]
        dti = np.concatenate([e[2] for e in es], axis=1)[:, o]
        tp = np.cumsum(dtm & ~dti, axis=1).astype(float)
        fp = np.cumsum(~dtm & ~dti, axis=1).astype(float)
        for t in range(T):
            nd = tp.shape[1]
            rc = tp[t] / npig
            pr = tp[t] / (fp[t] + tp[t] + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1] if nd else pr
            q = np.zeros(len(REC))
            inds = np.searchsorted(rc, REC, side="left")
            ok = inds < nd
            q[ok] = pr[inds[ok]]
            prec[t, :, k] = q
    return prec


def metrics(pi_all, pi_small, idx):
    def m(p, t=None):
        q = p[t] if t is not None else p
        q = q[q > -1]
        return float(q.mean()) if q.size else float("nan")
    pa, ps = ap_table(pi_all, idx), ap_table(pi_small, idx)
    return np.array([m(pa, 0), m(pa), m(ps, 0)])  # mAP50, mAP50-95, AP50 12-24 px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from pycocotools.coco import COCO
    coco = COCO()
    with contextlib.redirect_stdout(io.StringIO()):
        coco.dataset = eval_map.build_gt()
        coco.createIndex()
    imgs = json.load(open(os.path.join(eval_map.EVAL, "images.json")))
    n = len(imgs)
    key = [p.split("/")[-3] + "/" + os.path.basename(p).split("_")[0] for p in imgs]  # split/sequence
    clusters = {}
    for i, c in enumerate(key):
        clusters.setdefault(c, []).append(i)
    cl = list(clusters.values())
    print(f"{n} images, {len(cl)} sequence clusters", flush=True)

    pi = {}
    for m in MODELS:
        for q in ("fp32", "int8"):
            p = os.path.join(RES, f"{m}_832x480_{q}.dets.jsonl")
            pi[m, q] = (per_image(coco, p, RANGES["all"]), per_image(coco, p, RANGES["12-24"]))
            print(f"  evaluated {m} {q}", flush=True)

    full = list(range(n))
    point = {k: metrics(*v, full) for k, v in pi.items()}
    for (m, q), v in point.items():  # sanity: must equal eval_map.py score
        ref = json.load(open(os.path.join(RES, f"{m}_832x480_{q}.metrics.json")))
        assert abs(v[0] - ref["mAP50"]) < 1e-3 and abs(v[2] - ref["bucket_AP50"]["12-24"]) < 1e-3, (m, q, v, ref)

    rng = np.random.default_rng(a.seed)
    draws = {"sequence": [], "image": []}
    for _ in range(a.B):
        pick = rng.integers(0, len(cl), len(cl))
        draws["sequence"].append([i for j in pick for i in cl[j]])
        draws["image"].append(list(rng.integers(0, n, n)))
    boot = {mode: {k: np.array([metrics(*pi[k], d) for d in ds]) for k in pi} for mode, ds in draws.items()}

    names = ["mAP50", "mAP50-95", "AP50 12-24px"]
    out = {"B": a.B, "seed": a.seed, "n_images": n, "n_sequences": len(cl), "pairs": {}, "int8_drop": {}}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    print(f"\nPaired bootstrap, B={a.B}, 95% percentile CI. P(B>A) = share of replicates with a positive difference.")
    for lab, A, Bm in PAIRS:
        d0 = point[Bm, "int8"] - point[A, "int8"]
        out["pairs"][lab] = {}
        print(f"\n{lab}")
        for j, nm in enumerate(names):
            row = {"diff": float(d0[j])}
            s = f"  {nm:13s} diff {d0[j]:+.4f}"
            for mode in ("sequence", "image"):
                d = boot[mode][Bm, "int8"][:, j] - boot[mode][A, "int8"][:, j]
                row[mode] = {"ci95": ci(d), "p_pos": float((d > 0).mean())}
                s += f" | {mode:8s} CI [{ci(d)[0]:+.4f}, {ci(d)[1]:+.4f}] P>0 {(d > 0).mean():.3f}"
            out["pairs"][lab][nm] = row
            print(s)
    print("\nINT8 drop (FP32 - INT8) and its difference to yolo11n with the same activation")
    for m, base in [("yolo11n_silu", None), ("chimera_proto", "yolo11n_silu"), ("chimera_silu", "yolo11n_silu"),
                    ("yolo11n_lrelu", None), ("chimera_lrelu", "yolo11n_lrelu")]:
        dr0 = point[m, "fp32"] - point[m, "int8"]
        out["int8_drop"][m] = {}
        for j, nm in enumerate(names):
            row = {"drop": float(dr0[j])}
            s = f"  {m:18s} {nm:13s} drop {dr0[j]:.4f}"
            for mode in ("sequence",):
                dr = boot[mode][m, "fp32"][:, j] - boot[mode][m, "int8"][:, j]
                row["drop_ci95"] = ci(dr)
                s += f" CI [{ci(dr)[0]:.4f}, {ci(dr)[1]:.4f}]"
                if base:
                    db = boot[mode][base, "fp32"][:, j] - boot[mode][base, "int8"][:, j]
                    row["vs_" + base] = {"diff": float(dr0[j] - (point[base, "fp32"] - point[base, "int8"])[j]),
                                         "ci95": ci(dr - db), "p_smaller_drop": float((dr < db).mean())}
                    s += f" | vs {base}: {row['vs_' + base]['diff']:+.4f} CI [{ci(dr - db)[0]:+.4f}, {ci(dr - db)[1]:+.4f}]" \
                         f" P(smaller drop) {(dr < db).mean():.3f}"
            out["int8_drop"][m][nm] = row
            print(s)
    json.dump(out, open(os.path.join(RES, "bootstrap.json"), "w"), indent=1)
    print(f"\n-> {os.path.join(RES, 'bootstrap.json')}")


if __name__ == "__main__":
    main()
