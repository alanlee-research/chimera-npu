"""Building blocks of Chimera-Proto, the earlier design (SPDDown, three-branch DilatedContext, HFGate with a 3x3 stage)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

class SPDDown(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        w = torch.zeros(4 * c, 1, 2, 2)
        for i in range(c):
            for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                w[4 * i + k, 0, dy, dx] = 1.0
        self.register_buffer("w", w, persistent=False)
        self.proj = Conv(4 * c, c, 1, 1)

    def forward(self, x):
        return self.proj(F.conv2d(x, self.w.to(x.dtype), stride=2, groups=self.c))

class HFGate(nn.Module):
    def __init__(self, c_main, c_aux):
        super().__init__()
        self.c_main, self.c_aux = c_main, c_aux
        mid = max(c_aux // 2, 16)
        self.g1 = Conv(c_aux, mid, 3, 1)
        self.g2 = nn.Conv2d(mid, 1, 1)
        nn.init.zeros_(self.g2.weight)
        nn.init.constant_(self.g2.bias, -1.0)

    def forward(self, x):
        main, aux = torch.split(x, [self.c_main, self.c_aux], dim=1)
        gate = torch.sigmoid(self.g2(self.g1(aux)))
        return torch.cat([main * (1.0 + gate), aux], dim=1)

class DilatedContext(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = max(c // 4, 16)
        self.b1 = Conv(c, h, 3, 1, p=1)
        self.b2 = Conv(c, h, 3, 1, p=2, d=2)
        self.b3 = Conv(c, h, 3, 1, p=4, d=4)
        self.fuse = Conv(3 * h, c, 1, 1, act=False)

    def forward(self, x):
        feat = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return x + self.fuse(feat)

def register():
    import ultralytics.nn.tasks as T
    for k in ("SPDDown", "HFGate", "DilatedContext"):
        setattr(T, k, globals()[k])
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([SPDDown, HFGate, DilatedContext])
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
        if not persp:
            raise RuntimeError("patch_rect_training: RandomPerspective not found")
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
