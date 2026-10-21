# =====================================================================================================
# Component ablation of Chimera-LReLU (Table 5 of the paper): two variants trained concurrently, one per GPU
#   chimera_lrelu_noMDRF : MDRF (layer 8) replaced by an identity mapping
#   chimera_lrelu_noP2   : stride-4 path and HFGate (layers 10-12) replaced by identity mappings
# Same protocol, seed and COCO initialization as train_dual.py (the paper uses SEED = 42). Layer indices are
# unchanged, so the same COCO tensors are transferred to every variant. The rest of this header describes
# the shared protocol.
# -----------------------------------------------------------------------------------------------------
# Paired training of YOLO11n-LeakyReLU and Chimera-LReLU (two GPUs, identical protocol)
# -----------------------------------------------------------------------------------------------------
#   * Activation: SiLU is replaced by LeakyReLU(0.1) in every convolution of both models. On the RK3582 NPU
#     the compiler fuses it into the convolution (ConvLeakyRelu); SiLU requires a separate Swish evaluation.
#     The activation is set through the `activation:` key of the model yaml, the mechanism Ultralytics uses
#     to store it in the checkpoint.
#   * Architecture: Chimera has no P5 stage; MDRF adds context at P4, HFGate injects P2 detail into P3,
#     and the P3 head input is squeezed to 64 channels.
#   * Protocol: both models use the same 832x480 canvas, data, schedule, seed and wall-clock budget, and are
#     trained concurrently, one per GPU. At the end, each model is exported to ONNX, evaluated on the
#     validation and test-dev splits, and packaged for INT8 conversion.
# =====================================================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics", "onnx", "onnxsim",
                "onnxruntime", "opencv-python-headless"], check=False)

import os, re, csv, json, math, time, shutil, zipfile
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
os.environ.update(WANDB_DISABLED="true", WANDB_MODE="disabled", COMET_MODE="disabled")

# =====================================================================================================
# CONFIG
# =====================================================================================================
QUICK_TEST = False

WORK        = Path("/kaggle/working")
DATA_MARKER = "vd_deploy_v3"
W, H        = 832, 480

ACT = "nn.LeakyReLU(0.1)"   # activation of every Conv in both models (written to the yaml)
SEED = 42                   # one seed per run (paper: 42, 42, 9, 16); everything else stays fixed

EXPERIMENTS = {
    "chimera_lrelu_noMDRF": dict(yaml="chimera_lrelu_noMDRF", pretrained="yolo11n.pt", gpu=0),
    "chimera_lrelu_noP2":   dict(yaml="chimera_lrelu_noP2", pretrained="yolo11n.pt", gpu=1),
}

RECT_TRAIN = True

SPEED = dict(
    local_copy=True,
    disk_cache=True,
    job_threads=1,
    monitor_poll=60,
)

TRAIN = dict(
    imgsz=832, epochs=300, batch=16, patience=0,
    seed=SEED, deterministic=False, cos_lr=True, close_mosaic=15, amp=True, cache=False, plots=True,
    optimizer="auto", scale=0.3, translate=0.1, mosaic=1.0, mixup=0.0, copy_paste=0.0,
    fliplr=0.5, flipud=0.0, degrees=0.0, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, save_period=10,
)
SESSION = dict(total_budget_hours=11.3, post_train_reserve_hours=1.1)
RESUME  = {}
POST_ONLY_FROM = None

EXPORT = dict(opset=12, cls_sigmoid=True)
EVAL   = dict(sets=["val_raw", "test_raw"], conf_lo=0.001, nms_iou=0.6, max_det=300, max_nms=3000,
              conf_ops=[0.15, 0.25, 0.35, 0.5], drop_gt_in_ignore=False, max_images=0)
INT8   = dict(enabled=True, calib_n=200, method="MinMax", eval_sets=["val_raw", "test_raw"], modes=["FULL"])
SIZE_BUCKETS = [(8, 12), (12, 24), (24, 48), (48, 96), (96, 1e9)]
PACKAGE = dict(calib_n=300, eval_n=100)

if QUICK_TEST:
    TRAIN.update(epochs=2, fraction=0.05, plots=False, save_period=-1, close_mosaic=0)
    SESSION.update(total_budget_hours=0.6, post_train_reserve_hours=0.25)
    EVAL["max_images"] = 40
    INT8["calib_n"] = 20
    PACKAGE.update(calib_n=20, eval_n=10)

SESSION_START = time.time()

def hours_left():
    return SESSION["total_budget_hours"] - (time.time() - SESSION_START) / 3600.0

# =====================================================================================================
# 1. DATASET
# =====================================================================================================
EXTRACT_DIR = Path("/tmp/vd_data")

def locate_dataset():
    for d in (EXTRACT_DIR, WORK):
        if (d / DATA_MARKER / "meta.json").exists(): return d / DATA_MARKER
    for meta in Path("/kaggle/input").rglob("meta.json"):
        if (meta.parent.name == DATA_MARKER or "vd_deploy" in meta.parent.name) and (meta.parent / "train" / "images").is_dir():
            src = meta.parent
            size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
            EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(EXTRACT_DIR).free < size * 1.2: return src
            t0 = time.time()
            print(f"[Data] Copying to local disk /tmp...", flush=True)
            shutil.copytree(src, EXTRACT_DIR / DATA_MARKER, dirs_exist_ok=True)
            print(f"[Data] Done in {time.time() - t0:.0f} s")
            return EXTRACT_DIR / DATA_MARKER
    cands = [p for p in Path("/kaggle/input").rglob(f"*{DATA_MARKER}*") if p.is_file() and zipfile.is_zipfile(p)]
    if cands:
        zp = max(cands, key=lambda p: p.stat().st_size)
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z: z.extractall(EXTRACT_DIR)
        return EXTRACT_DIR / DATA_MARKER
    raise FileNotFoundError(f"{DATA_MARKER} not found under /kaggle/input")

DATA = locate_dataset()
META = json.loads((DATA / "meta.json").read_text())
CLASS_NAMES = META["class_names"]
NC = len(CLASS_NAMES)
print(f"[Data] Source: {DATA} | classes: {NC} {CLASS_NAMES}")

DATA_YAML = WORK / "data_v3.yaml"
DATA_YAML.write_text(
    f"path: {DATA}\ntrain: train/images\nval: val_sel/images\nnc: {NC}\n"
    f"names: {json.dumps({i: n for i, n in enumerate(CLASS_NAMES)})}\n")

# =====================================================================================================
# 2. CHIMERA MODULES (activation taken from Conv.default_act, i.e. the `activation:` key of the yaml; no SiLU)
# =====================================================================================================
MODULES_SRC = r'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

class HFGate(nn.Module):
    """HFGate: single-channel spatial gate computed from stride-4 (P2) features that modulates P3."""
    def __init__(self, c_main, c_aux):
        super().__init__()
        self.c_main, self.c_aux = c_main, c_aux
        self.gate = nn.Conv2d(c_aux, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x):
        main, aux = torch.split(x, [self.c_main, self.c_aux], dim=1)
        mask = torch.sigmoid(self.gate(aux))
        return torch.cat([main * (1.0 + mask), aux], dim=1)

class DilatedContextSlim(nn.Module):
    """MDRF: two parallel 3x3 branches (dilation 1 and 3) with a 1x1 fusion and a residual connection."""
    def __init__(self, c):
        super().__init__()
        h = max(c // 2, 16)
        self.b1 = Conv(c, h, 3, 1, p=1)
        self.b2 = Conv(c, h, 3, 1, p=3, d=3)
        self.fuse = Conv(2 * h, c, 1, 1, act=False)

    def forward(self, x):
        feat = torch.cat([self.b1(x), self.b2(x)], dim=1)
        return x + self.fuse(feat)

def register():
    import ultralytics.nn.tasks as T
    for k in ("HFGate", "DilatedContextSlim"):
        setattr(T, k, globals()[k])
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([HFGate, DilatedContextSlim])
    except Exception:
        pass

def patch_rect_training(rect_h, rect_w):
    import random
    import ultralytics.data.augment as A
    import ultralytics.data.dataset as D
    if getattr(A, "_chimera_rect", None) == (rect_h, rect_w): return
    orig = getattr(A, "_chimera_orig_v8", A.v8_transforms)
    A._chimera_orig_v8 = orig

    def _walk(obj, seen):
        if obj is None or id(obj) in seen: return
        seen.add(id(obj)); yield obj
        for t in (getattr(obj, "transforms", None) or []): yield from _walk(t, seen)
        yield from _walk(getattr(obj, "pre_transform", None), seen)

    def v8_rect(dataset, imgsz, hyp, *args, **kwargs):
        T = orig(dataset, imgsz, hyp, *args, **kwargs)
        nodes = list(_walk(T, set()))
        persp = [t for t in nodes if isinstance(t, A.RandomPerspective)]
        if not persp: raise RuntimeError("patch_rect_training: RandomPerspective not found")
        if all(not hasattr(t, "pre_transform") for t in persp):
            for t in persp: t.size = (rect_w, rect_h)
        else:
            for t in persp:
                if getattr(t, "pre_transform", None) is not None:
                    t.pre_transform = A.LetterBox(new_shape=(rect_h, rect_w))
        return T

    A.v8_transforms = v8_rect
    D.v8_transforms = v8_rect
    A._chimera_rect = (rect_h, rect_w)

def assert_activation(model, where=""):
    """Fail fast if any SiLU/ReLU remains (a yaml without `activation:` silently falls back to SiLU)."""
    acts = {}
    for m in model.modules():
        if isinstance(m, (nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.Hardswish, nn.GELU)):
            acts[type(m).__name__] = acts.get(type(m).__name__, 0) + 1
    if set(acts) != {"LeakyReLU"}:
        raise RuntimeError(f"[{where}] unexpected activations: {acts} (expected LeakyReLU only)")
    return acts
'''
(WORK / "chimera_modules.py").write_text(MODULES_SRC)
sys.path.insert(0, str(WORK))

# =====================================================================================================
# MODEL YAMLS: Chimera-LReLU and YOLO11n-LeakyReLU (activation declared in the yaml)
# =====================================================================================================
YAML_CHIMERA_LRELU = f"""nc: {NC}
activation: {ACT}
scales:
  n: [0.50, 0.25, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0  P1/2   16 ch
  - [-1, 1, Conv, [128, 3, 2]]         # 1  P2/4   32 ch
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2  P2     64 ch (120x208)
  - [-1, 1, Conv, [256, 3, 2]]         # 3  P3/8   64 ch
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4  P3     128 ch (60x104)
  - [-1, 1, Conv, [512, 3, 2]]         # 5  P4/16  128 ch
  - [-1, 2, C3k2, [512, True]]         # 6  P4     128 ch (30x52)
head:
  - [-1, 1, SPPF, [512, 5]]                      # 7  128 ch @P4
  - [-1, 1, DilatedContextSlim, [128]]           # 8  128 ch @P4 (MDRF Context)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 9  128 ch @P3
  - [2, 1, Conv, [256, 3, 2]]                    # 10 64 ch  @P3 (3x3 stride-2 conv from P2)
  - [[4, 10], 1, Concat, [1]]                    # 11 128 + 64 = 192 ch @P3
  - [-1, 1, HFGate, [128, 64]]                   # 12 192 ch
  - [[-1, 9], 1, Concat, [1]]                    # 13 192 + 128 = 320 ch @P3
  - [-1, 2, C3k2, [512, False]]                  # 14 128 ch (P3 Neck Fusion)
  - [-1, 1, Conv, [256, 1, 1]]                   # 15 64 ch  (P3 head input squeezed to 64 ch)
  - [14, 1, Conv, [512, 3, 2]]                   # 16 128 ch @P4
  - [[-1, 8], 1, Concat, [1]]                    # 17 128 + 128 = 256 ch @P4
  - [-1, 2, C3k2, [512, False]]                  # 18 128 ch (P4 Head Output)
  - [[15, 18], 1, Detect, [nc]]                  # 19 Detect(P3: 64ch, P4: 128ch)
"""
(WORK / "chimera_lrelu.yaml").write_text(YAML_CHIMERA_LRELU)

# Ablation variants: removed blocks become nn.Identity so that layer indices stay identical.
def _ablate(src, repl):
    out = src
    for a, b in repl:
        assert out.count(a) == 1, a
        out = out.replace(a, b)
    return out
YAML_NO_MDRF = _ablate(YAML_CHIMERA_LRELU, [
    ("  - [-1, 1, DilatedContextSlim, [128]]           # 8  128 ch @P4 (MDRF Context)",
     "  - [-1, 1, nn.Identity, []]                     # 8  MDRF removed (identity)")])
YAML_NO_P2 = _ablate(YAML_CHIMERA_LRELU, [
    ("  - [2, 1, Conv, [256, 3, 2]]                    # 10 64 ch  @P3 (3x3 stride-2 conv from P2)",
     "  - [2, 1, nn.Identity, []]                      # 10 stride-4 path removed (output unused)"),
    ("  - [[4, 10], 1, Concat, [1]]                    # 11 128 + 64 = 192 ch @P3",
     "  - [4, 1, nn.Identity, []]                      # 11 P3 only, 128 ch"),
    ("  - [-1, 1, HFGate, [128, 64]]                   # 12 192 ch",
     "  - [-1, 1, nn.Identity, []]                     # 12 HFGate removed")])
(WORK / "chimera_lrelu_noMDRF.yaml").write_text(YAML_NO_MDRF)
(WORK / "chimera_lrelu_noP2.yaml").write_text(YAML_NO_P2)

# YOLO11n: the original Ultralytics yolo11.yaml with only activation, nc and scale n added
import yaml as _yaml
from ultralytics.utils import ROOT as _UL_ROOT
_y11 = _yaml.safe_load((_UL_ROOT / "cfg/models/11/yolo11.yaml").read_text())
_y11.update(nc=NC, activation=ACT, scale="n")
(WORK / "yolo11n_lrelu.yaml").write_text(_yaml.safe_dump(_y11, sort_keys=False))

# =====================================================================================================
# INITIALIZATION FROM COCO YOLO11n WEIGHTS (both models)
# =====================================================================================================
def load_coco_backbone_head(yaml_path, pretrained_pt_path, output_pt_path):
    """YOLO11n-LeakyReLU has the architecture of yolo11n.pt: load every tensor whose shape matches (the 80-class outputs are skipped)."""
    from ultralytics import YOLO
    m = YOLO(yaml_path)
    src = YOLO(pretrained_pt_path).model.float().state_dict()
    dst = m.model.state_dict()
    hit = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    m.model.load_state_dict(hit, strict=False)
    chimera_modules.assert_activation(m.model, "yolo11n_lrelu build")
    torch.save({"model": m.model, "yaml": m.model.yaml, "task": "detect", "train_args": {"task": "detect"}},
               output_pt_path)
    print(f"[Init] YOLO11n-LeakyReLU: {len(hit)}/{len(dst)} tensors loaded from COCO yolo11n")
    return output_pt_path

def surgical_weight_transplant_relu(yaml_path, pretrained_pt_path, output_pt_path):
    from ultralytics import YOLO
    print("\n[Init] Transferring COCO weights to Chimera-LReLU...")
    base_yolo = YOLO(pretrained_pt_path)
    base_model = base_yolo.model.state_dict()
    
    chimera_yolo = YOLO(yaml_path)
    chimera_model = chimera_yolo.model
    chimera_dict = chimera_model.state_dict()

    transplanted_count = 0
    # 1. Backbone layers 0..6
    for k, v in base_model.items():
        if any(k.startswith(f"model.{i}.") for i in range(7)):
            if k in chimera_dict and chimera_dict[k].shape == v.shape:
                chimera_dict[k] = v
                transplanted_count += 1

    # 2. Detect head (yolo11n model.23 -> Chimera model.19); class outputs truncated from 80 to 4
    src_prefix = "model.23."
    dst_prefix = "model.19."
    for k, v in base_model.items():
        if not k.startswith(src_prefix): continue
        suffix = k[len(src_prefix):]
        parts = suffix.split(".")
        if len(parts) >= 2 and parts[1] in ("0", "1"):
            dst_k = dst_prefix + suffix
            if dst_k not in chimera_dict: continue
            dst_tensor = chimera_dict[dst_k]
            if v.shape == dst_tensor.shape:
                chimera_dict[dst_k] = v
                transplanted_count += 1
            elif v.dim() == 4 and v.shape[0] == 80 and dst_tensor.shape[0] == 4 and v.shape[1:] == dst_tensor.shape[1:]:
                chimera_dict[dst_k] = v[:4]
                transplanted_count += 1
            elif v.dim() == 1 and v.shape[0] == 80 and dst_tensor.shape[0] == 4:
                chimera_dict[dst_k] = v[:4]
                transplanted_count += 1

    chimera_model.load_state_dict(chimera_dict)
    chimera_modules.assert_activation(chimera_model, "chimera build")
    save_ckpt = {
        "model": chimera_model,
        "yaml": chimera_model.yaml,
        "task": "detect",
        "train_args": {"task": "detect"}
    }
    torch.save(save_ckpt, output_pt_path)
    print(f"[Init] Chimera-LReLU: {transplanted_count} tensors transferred from COCO yolo11n")
    return output_pt_path

# =====================================================================================================
# TRAINING JOB (one process per GPU)
# =====================================================================================================
JOB_SRC = r'''
import json, os, sys
from pathlib import Path
import torch

os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
cfg = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, cfg["work"]); os.chdir(cfg["work"])
os.environ.update(WANDB_DISABLED="true", WANDB_MODE="disabled", COMET_MODE="disabled")

import chimera_modules; chimera_modules.register()
if cfg.get("rect_train"):
    chimera_modules.patch_rect_training(cfg["rect_h"], cfg["rect_w"])
from ultralytics import YOLO

m = YOLO(cfg["model"])
# The trainer rebuilds the model from the checkpoint yaml, so the activation check runs on the model actually trained
m.add_callback("on_train_start", lambda tr: print("[Activation OK]", cfg["name"],
               chimera_modules.assert_activation(tr.model, cfg["name"]), flush=True))
if cfg.get("resume"):
    m.train(resume=True, time=cfg["time_h"])
else:
    m.train(data=cfg["data"], project=cfg["project"], name=cfg["name"], exist_ok=True,
            time=cfg["time_h"], device=0, workers=cfg["workers"], **cfg["train"])

print("[JOB DONE]", cfg["name"], flush=True)
'''
(WORK / "job_train.py").write_text(JOB_SRC)

import chimera_modules
chimera_modules.register()
from ultralytics import YOLO

N_GPU = torch.cuda.device_count()
print(f"PyTorch {torch.__version__} | GPUs available: {N_GPU}")

YOLO("yolo11n.pt")
torch.manual_seed(SEED)  # layers without COCO counterparts are initialized from SEED as well
for _n in EXPERIMENTS:
    _w = str(WORK / f"{_n}_transplanted.pt")
    surgical_weight_transplant_relu(str(WORK / f"{_n}.yaml"), "yolo11n.pt", _w)
    EXPERIMENTS[_n] = dict(model=_w, gpu=EXPERIMENTS[_n]["gpu"])

# =====================================================================================================
# 3. ARCHITECTURE CHECK AND TIME BUDGET
# =====================================================================================================
def count_gmac(det, h=H, w=W):
    tot = [0]
    def hook(m, i, o):
        tot[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    hs = [m.register_forward_hook(hook) for m in det.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad(): det(torch.zeros(1, 3, h, w))
    for x in hs: x.remove()
    return tot[0] / 1e9

ARCH = {}
print("\n" + "=" * 90 + "\nARCHITECTURE CHECK (LeakyReLU only) AND COMPUTE COST @ 832x480")
for name, exp in EXPERIMENTS.items():
    det = YOLO(exp["model"]).model.float().eval().cpu()
    chimera_modules.assert_activation(det, name)
    params = sum(p.numel() for p in det.parameters()) / 1e6
    gmac = count_gmac(det)
    strides = [int(s) for s in det.stride.tolist()]
    ARCH[name] = dict(params_M=round(params, 3), gflops=round(2 * gmac, 2), strides=strides)
    print(f"  {name:18s} | Params: {params:.2f}M | FLOPs: {2 * gmac:.2f} GFLOPs | Strides: {strides}")
print("=" * 90)

# =====================================================================================================
# 4. CONCURRENT TRAINING
# =====================================================================================================
def read_last_row(csv_path):
    try: rows = list(csv.DictReader(open(csv_path)))
    except Exception: return None
    if not rows: return None
    return {k.strip(): v.strip() for k, v in rows[-1].items() if k}

def launch(name, exp, gpu, time_h, workers, target_epochs):
    train_cfg = dict(TRAIN)
    train_cfg["epochs"] = target_epochs

    cfg = dict(work=str(WORK), data=str(DATA_YAML), project=str(WORK / "runs"), name=name, time_h=round(time_h, 3),
               workers=workers, train=train_cfg, model=exp.get("model"),
               resume=RESUME.get(name), rect_train=RECT_TRAIN, rect_h=H, rect_w=W)
    cp = WORK / f"job_{name}.json"
    cp.write_text(json.dumps(cfg))
    th = str(SPEED["job_threads"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS=th,
               MKL_NUM_THREADS=th, OPENBLAS_NUM_THREADS=th, TORCH_FORCE_WEIGHTS_ONLY_LOAD="0")
    log = open(WORK / f"log_{name}.txt", "w")
    print(f"[Launch] {name:18s} -> GPU {gpu} | planned epochs: {target_epochs} | budget: {time_h:.2f} h")
    return subprocess.Popen([sys.executable, "-u", str(WORK / "job_train.py"), str(cp)],
                            stdout=log, stderr=subprocess.STDOUT, cwd=str(WORK), env=env), log

def monitor(procs, poll=None):
    poll = poll or SPEED["monitor_poll"]
    last_epoch, last_t = {}, {}
    while any(p.poll() is None for p, _ in procs.values()):
        time.sleep(poll)
        
        for name, (p, _) in procs.items():
            if p.poll() is not None and p.poll() != 0:
                print(f"\n[Error] {name} exited with code {p.poll()}")
                log_txt = Path(WORK / f"log_{name}.txt")
                if log_txt.exists():
                    print(f"--- Last log lines of {name} ---\n{log_txt.read_text(errors='ignore')[-3000:]}")
                print("-" * 60)
                for other_n, (other_p, _) in procs.items():
                    if other_p.poll() is None:
                        other_p.terminate()
                return

        for name, (p, _) in procs.items():
            r = read_last_row(WORK / "runs" / name / "results.csv")
            if r and r.get("epoch") != last_epoch.get(name):
                t_cum = float(r["time"]) if r.get("time") not in (None, "") else time.time()
                prev_ep, prev_t = last_epoch.get(name), last_t.get(name)
                try: n_ep = int(float(r["epoch"])) - (int(float(prev_ep)) if prev_ep else 0)
                except Exception: n_ep = 1
                ep_sec = (t_cum - prev_t) / max(n_ep, 1) if prev_t is not None else float("nan")
                last_epoch[name], last_t[name] = r.get("epoch"), t_cum
                print(f"  [{name:16s}] ep {r.get('epoch'):>4} ({ep_sec:4.0f}s/ep) | "
                      f"box {float(r.get('train/box_loss', 0)):.3f} cls {float(r.get('train/cls_loss', 0)):.3f} | "
                      f"P {float(r.get('metrics/precision(B)', 0)):.3f} R {float(r.get('metrics/recall(B)', 0)):.3f} | "
                      f"mAP50 {float(r.get('metrics/mAP50(B)', 0)):.4f} mAP50-95 {float(r.get('metrics/mAP50-95(B)', 0)):.4f} | "
                      f"{hours_left():.2f} h left", flush=True)

        if hours_left() < SESSION["post_train_reserve_hours"] * 0.5:
            print("[Monitor] Session time limit reached; stopping training to export ONNX.")
            for p, _ in procs.values():
                if p.poll() is None: p.terminate()

    for name, (p, log) in procs.items():
        log.close()
        status = "done" if p.returncode == 0 else f"failed ({p.returncode})"
        print(f"[Status] {name}: {status}")

def run_training():
    names = list(EXPERIMENTS)
    budget = hours_left() - SESSION["post_train_reserve_hours"]
    assert budget > 0.05, "Not enough session time left for training"

    projected = {
        "chimera_lrelu_noMDRF": int(budget * 3600 / 160),
        "chimera_lrelu_noP2": int(budget * 3600 / 150)
    }

    if N_GPU >= len(names) and len(names) > 1:
        workers = max(2, (os.cpu_count() or 4) // len(names))
        procs = {n: launch(n, EXPERIMENTS[n], EXPERIMENTS[n].get("gpu", i), budget, workers, projected.get(n, 200))
                 for i, n in enumerate(names)}
        monitor(procs)
    else:
        per = budget / len(names)
        for n in names:
            procs = {n: launch(n, EXPERIMENTS[n], 0, min(per, hours_left() - SESSION["post_train_reserve_hours"]), os.cpu_count() or 4, 150)}
            monitor(procs)

def prepare_disk_cache():
    if not SPEED["disk_cache"]: return False
    img_dir = DATA / "train" / "images"
    if not os.access(img_dir, os.W_OK): return False
    imgs = sorted(img_dir.glob("*.jpg"))
    need = len(imgs) * W * H * 3 * 1.05
    have = sum(1 for p in imgs if p.with_suffix(".npy").exists())
    free = shutil.disk_usage(img_dir).free
    if have < len(imgs) and free < (len(imgs) - have) * W * H * 3 * 1.2: return False
    t0 = time.time()
    def save(p):
        f = p.with_suffix(".npy")
        if not f.exists():
            im = cv2.imread(str(p))
            if im is not None: np.save(f.as_posix(), im, allow_pickle=False)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex: list(ex.map(save, imgs))
    print(f"[Disk cache] Cached {len(imgs)} images as .npy ({time.time() - t0:.0f} s)")
    return True

if RECT_TRAIN:
    chimera_modules.patch_rect_training(H, W)
if prepare_disk_cache():
    TRAIN["cache"] = "disk"

run_training()

def find_best(name):
    for f in ("best.pt", "last.pt"):
        p = WORK / "runs" / name / "weights" / f
        if p.exists(): return p
    return None

# =====================================================================================================
# 5. ONNX EXPORT (RKNN-ready, raw head outputs) AND VISDRONE EVALUATION
# =====================================================================================================
class RawHeadExport(nn.Module):
    def __init__(self, det, cls_sigmoid=True):
        super().__init__()
        self.net, self.layers, self.head = det, det.model, det.model[-1]
        self.cls_sigmoid = cls_sigmoid

    def forward(self, x):
        y = []
        for m in self.layers[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.net.save else None)
        h = self.head
        feats = [x if j == -1 else y[j] for j in h.f]
        outs = []
        for i in range(h.nl):
            c = h.cv3[i](feats[i])
            outs += [h.cv2[i](feats[i]), torch.sigmoid(c) if self.cls_sigmoid else c]
        return tuple(outs)

def export_onnx(weights, onnx_path):
    import onnx, onnxruntime as ort
    from onnxsim import simplify
    det = YOLO(str(weights)).model.float().eval().cpu()
    try: det.fuse(verbose=False)
    except Exception: pass
    strides = [int(s) for s in det.stride.tolist()]
    names = []
    for s in strides: names += [f"box_s{s}", f"cls_s{s}"]
    wrap = RawHeadExport(det, EXPORT["cls_sigmoid"]).eval()
    for p in wrap.parameters(): p.requires_grad_(False)
    dummy = torch.zeros(1, 3, H, W)
    kw = dict(opset_version=EXPORT["opset"], input_names=["images"], output_names=names, do_constant_folding=True)
    try: torch.onnx.export(wrap, dummy, str(onnx_path), dynamo=False, **kw)
    except TypeError: torch.onnx.export(wrap, dummy, str(onnx_path), **kw)
    try:
        m, ok = simplify(onnx.load(str(onnx_path)))
        if ok: onnx.save(m, str(onnx_path))
    except Exception: pass
    ops = sorted({n.op_type for n in onnx.load(str(onnx_path)).graph.node})
    if "LeakyRelu" not in ops:
        raise RuntimeError(f"{onnx_path.name}: no LeakyRelu in the ONNX graph, wrong activation (ops={ops})")
    print(f"   [ONNX] {onnx_path.name} | ops: {ops}")
    return names, strides, ops

def np_decode(outs, conf, cls_sigmoid=True, max_det=300, nms_iou=0.6, max_nms=3000):
    thr = conf if cls_sigmoid else float(np.log(conf / (1.0 - conf)))
    bb, ss, cc = [], [], []
    for i in range(0, len(outs), 2):
        box = np.asarray(outs[i], np.float32)[0]; cls = np.asarray(outs[i + 1], np.float32)[0]
        nc, h, w = cls.shape; stride = W / w; reg = box.shape[0] // 4
        cls = cls.reshape(nc, -1); box = box.reshape(4, reg, -1)
        ci, ai = np.nonzero(cls > thr)
        if ai.size == 0: continue
        b = box[:, :, ai]; b = b - b.max(1, keepdims=True); e = np.exp(b)
        d = ((e / e.sum(1, keepdims=True)) * np.arange(reg, dtype=np.float32)[None, :, None]).sum(1)
        gx = (ai % w).astype(np.float32) + 0.5; gy = (ai // w).astype(np.float32) + 0.5
        bb.append(np.stack([(gx - d[0]) * stride, (gy - d[1]) * stride, (gx + d[2]) * stride, (gy + d[3]) * stride], 1))
        sc = cls[ci, ai]
        ss.append(sc if cls_sigmoid else 1.0 / (1.0 + np.exp(-sc))); cc.append(ci)
    if not bb: return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    boxes, scores, cid = np.concatenate(bb), np.concatenate(ss), np.concatenate(cc)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W); boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    if len(scores) > max_nms:
        k = np.argsort(-scores)[:max_nms]; boxes, scores, cid = boxes[k], scores[k], cid[k]
    off = boxes + cid[:, None].astype(np.float32) * 8192.0
    xywh = np.stack([off[:, 0], off[:, 1], off[:, 2] - off[:, 0], off[:, 3] - off[:, 1]], 1)
    k = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, float(nms_iou)), dtype=int).reshape(-1)
    boxes, scores, cid = boxes[k], scores[k], cid[k]
    o = np.argsort(-scores)[:max_det]
    return boxes[o], scores[o], cid[o]

def read_gt(path):
    rows = [l.split() for l in path.read_text().splitlines()] if path.exists() else []
    rows = [list(map(float, r)) for r in rows if len(r) == 5]
    if not rows: return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    a = np.asarray(rows, np.float32)
    cx, cy, bw, bh = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], 1), a[:, 0].astype(np.int64)

def read_ign(path):
    rows = [list(map(float, l.split())) for l in path.read_text().splitlines() if len(l.split()) == 4] if path.exists() else []
    return np.asarray(rows, np.float32).reshape(-1, 4)

def ioa(pb, ig):
    x1 = torch.maximum(pb[:, None, 0], ig[None, :, 0])
    y1 = torch.maximum(pb[:, None, 1], ig[None, :, 1])
    x2 = torch.minimum(pb[:, None, 2], ig[None, :, 2])
    y2 = torch.minimum(pb[:, None, 3], ig[None, :, 3])
    inter = torch.clamp(x2 - x1, min=0.0) * torch.clamp(y2 - y1, min=0.0)
    return (inter / ((pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1]) + 1e-9)[:, None]).max(dim=1).values

def apply_ignore(b, s, c, gb, gc, ig):
    if len(ig) and len(b):
        pb_t = torch.from_numpy(b); ig_t = torch.from_numpy(ig)
        k = (ioa(pb_t, ig_t).numpy() < 0.5)
        b, s, c = b[k], s[k], c[k]
    return b, s, c, gb, gc

_trapz = getattr(np, "trapezoid", None) or np.trapz
def _ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return _trapz(np.interp(x, mrec, mpre), x)

def map_from_recs(recs):
    iouv = np.linspace(0.5, 0.95, 10)
    TP, CF, PC, TC = [], [], [], []
    for pb, ps, pc, gb, gc in recs:
        correct = np.zeros((len(pb), 10), bool)
        if len(gb) and len(pb):
            x1 = np.maximum(gb[:, None, 0], pb[None, :, 0]); y1 = np.maximum(gb[:, None, 1], pb[None, :, 1])
            x2 = np.minimum(gb[:, None, 2], pb[None, :, 2]); y2 = np.minimum(gb[:, None, 3], pb[None, :, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            iou = inter / ((gb[:, 2] - gb[:, 0])[:, None] * (gb[:, 3] - gb[:, 1])[:, None] +
                           (pb[:, 2] - pb[:, 0])[None, :] * (pb[:, 3] - pb[:, 1])[None, :] - inter + 1e-9)
            for j, t in enumerate(iouv):
                gi, pi = np.nonzero((iou >= t) & (gc[:, None] == pc[None, :]))
                if gi.size:
                    m = np.stack([gi, pi, iou[gi, pi]], 1); m = m[m[:, 2].argsort()[::-1]]
                    m = m[np.unique(m[:, 1], return_index=True)[1]]
                    m = m[m[:, 2].argsort()[::-1]]
                    m = m[np.unique(m[:, 0], return_index=True)[1]]
                    correct[m[:, 1].astype(int), j] = True
        TP.append(correct); CF.append(ps); PC.append(pc); TC.append(gc)
    tp, cf, pc, tc = (np.concatenate(x) for x in (TP, CF, PC, TC))
    o = np.argsort(-cf, kind="stable"); tp, pc = tp[o], pc[o]
    aps, per = [], {}
    for c in range(NC):
        n_gt = int((tc == c).sum())
        if n_gt == 0: continue
        m = pc == c
        if m.sum() == 0: ap = np.zeros(10)
        else:
            tpc = tp[m].cumsum(0); fpc = (~tp[m]).cumsum(0)
            rec = tpc / (n_gt + 1e-16); prec = tpc / (tpc + fpc + 1e-16)
            ap = np.array([_ap(rec[:, j], prec[:, j]) for j in range(10)])
        aps.append(ap); per[CLASS_NAMES[c]] = (float(ap[0]), float(ap.mean()))
    if not aps: return 0.0, 0.0, {}
    aps = np.asarray(aps)
    return float(aps[:, 0].mean()), float(aps.mean()), per

def eval_onnx_model(onnx_path, set_name, mode="FULL"):
    import onnxruntime as ort
    root = DATA / set_name
    pred = (lambda s: "__full" in s) if mode == "FULL" else (lambda s: "__full" not in s)
    imgs = sorted(p for p in (root / "images").glob("*.jpg") if pred(p.stem))
    if not imgs: return None
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    recs = []
    for p in imgs:
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        outs = sess.run(None, {"images": rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0})
        b, s, c = np_decode(outs, EVAL["conf_lo"], EXPORT["cls_sigmoid"], EVAL["max_det"], EVAL["nms_iou"], EVAL["max_nms"])
        gb, gc = read_gt(root / "labels" / f"{p.stem}.txt")
        recs.append(apply_ignore(b, s, c, gb, gc, read_ign(root / "ignore" / f"{p.stem}.txt")))
    m50, m5095, per = map_from_recs(recs)
    return dict(mAP50=m50, mAP50_95=m5095, per_class=per, n_imgs=len(recs))

# =====================================================================================================
# 6. RKNN CONVERSION PACKAGE
# =====================================================================================================
CONVERT_RKNN_SRC = r'''import json, glob, sys
from rknn.api import RKNN
platform = sys.argv[1] if len(sys.argv) > 1 else "rk3588"
cfg = json.load(open("deploy_config.json"))
imgs = sorted(glob.glob("calib_images/*.jpg"))
open("calib.txt", "w").write("\n".join(imgs))
rk = RKNN(verbose=False)
rk.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform,
          quantized_algorithm="normal", quantized_method="channel")
assert rk.load_onnx(model="model_fp32.onnx") == 0
assert rk.build(do_quantization=True, dataset="calib.txt") == 0
assert rk.export_rknn("model_int8.rknn") == 0
print("OK -> model_int8.rknn")
rk.release()
'''

def make_package(name, best, onnx_fp32, out_names, strides, summary):
    pkg = WORK / f"rknn_pkg_{name}_seed{SEED}"
    if pkg.exists(): shutil.rmtree(pkg)
    (pkg / "calib_images").mkdir(parents=True)
    shutil.copy(onnx_fp32, pkg / "model_fp32.onnx")
    shutil.copy(best, pkg / "best.pt")
    if name.startswith("chimera"):
        shutil.copy(WORK / "chimera_modules.py", pkg)
        shutil.copy(WORK / f"{name}.yaml", pkg)
    calib = sorted((DATA / "calib").glob("*.jpg"))[:PACKAGE["calib_n"]]
    for p in calib: shutil.copy(p, pkg / "calib_images")
    (pkg / "convert_rknn.py").write_text(CONVERT_RKNN_SRC)
    (pkg / "deploy_config.json").write_text(json.dumps(dict(
        model=name, input_w=W, input_h=H, outputs=out_names, strides=strides,
        reg_max=16, cls_sigmoid=EXPORT["cls_sigmoid"], class_names=CLASS_NAMES, seed=SEED, results=summary
    ), indent=2, ensure_ascii=False))
    shutil.make_archive(str(pkg), "zip", pkg)
    print(f"   [Package] {pkg}.zip written")

# =====================================================================================================
# 7. SUMMARY
# =====================================================================================================
FINAL_RESULTS = {}
for name in EXPERIMENTS:
    best = find_best(name)
    print("\n" + "=" * 90 + f"\nEVALUATION: {name} (best checkpoint: {best})\n" + "=" * 90)
    if best is None: continue

    onnx_fp32 = WORK / f"{name}_fp32.onnx"
    out_names, strides, ops = export_onnx(best, onnx_fp32)
    ev_val = eval_onnx_model(onnx_fp32, "val_raw", mode="FULL")
    ev_test = eval_onnx_model(onnx_fp32, "test_raw", mode="FULL")

    FINAL_RESULTS[name] = dict(
        arch=ARCH.get(name, {}),
        val=ev_val,
        test=ev_test,
        weights=str(best),
        strides=strides
    )
    print(f"  [{name:18s}] VAL  FULL: mAP50 = {ev_val['mAP50']*100:.2f}% | mAP50-95 = {ev_val['mAP50_95']*100:.2f}%")
    if ev_test:
        print(f"  [{name:18s}] TEST FULL: mAP50 = {ev_test['mAP50']*100:.2f}% | mAP50-95 = {ev_test['mAP50_95']*100:.2f}%")
    make_package(name, best, onnx_fp32, out_names, strides, FINAL_RESULTS[name])

print("\n" + "=" * 95)
print(f"RESULTS, VisDrone full frame 832x480 | SEED = {SEED}")
print("=" * 95)
print(f"{'Model':18s}{'Params':>8s}{'FLOPs':>8s}{'Val mAP50':>12s}{'Val 50-95':>12s}{'Test mAP50':>12s}{'Test 50-95':>12s}")
print("-" * 82)
for n, r in FINAL_RESULTS.items():
    p = f"{r['arch'].get('params_M', 0):.2f}M"
    f = f"{r['arch'].get('gflops', 0):.2f}G"
    v50 = f"{r['val']['mAP50']*100:.2f}%" if r.get('val') else "N/A"
    v95 = f"{r['val']['mAP50_95']*100:.2f}%" if r.get('val') else "N/A"
    t50 = f"{r['test']['mAP50']*100:.2f}%" if r.get('test') else "N/A"
    t95 = f"{r['test']['mAP50_95']*100:.2f}%" if r.get('test') else "N/A"
    print(f"{n:18s}{p:>8s}{f:>8s}{v50:>12s}{v95:>12s}{t50:>12s}{t95:>12s}")
print("=" * 95)

with zipfile.ZipFile(WORK / f"runs_ablation_seed{SEED}.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in (WORK / "runs").rglob("*"):
        if p.is_file() and p.suffix in (".pt", ".csv", ".yaml", ".png", ".jpg"):
            z.write(p, arcname=str(p.relative_to(WORK)))
    for f in ["chimera_modules.py", "chimera_lrelu_noMDRF.yaml", "chimera_lrelu_noP2.yaml"]:
        if (WORK / f).exists(): z.write(WORK / f, arcname=f)

print(f"\n[Done] Archived: runs_ablation_seed{SEED}.zip")