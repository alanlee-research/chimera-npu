"""Chimera building blocks for Ultralytics (register() before building a model from yaml)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

class HFGate(nn.Module):
    """HFGate: single-channel spatial gate from stride-4 (P2) features applied to P3.

    Input: concatenation [F3 (c_main ch) | F2_down (c_aux ch)]. Output: [F3 * (1 + G) | F2_down],
    with G = sigmoid(Conv1x1(F2_down)). Zero weights and bias -1 give a uniform initial gain of 1.27.
    """
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
    """MDRF: two parallel 3x3 branches (dilation 1 and 3, c/2 channels each), 1x1 fusion, residual.
    The activation follows the `activation:` key of the model yaml."""
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
    """Fail if any activation other than LeakyReLU remains (a yaml without `activation:` falls back to SiLU)."""
    acts = {}
    for m in model.modules():
        if isinstance(m, (nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.Hardswish, nn.GELU)):
            acts[type(m).__name__] = acts.get(type(m).__name__, 0) + 1
    if set(acts) != {"LeakyReLU"}:
        raise RuntimeError(f"[{where}] unexpected activations: {acts} (expected LeakyReLU only)")
    return acts
