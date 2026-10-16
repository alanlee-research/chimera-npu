"""Reference implementation of the single-core hybrid scheduler and the tile merge rule.

This module mirrors the logic of the on-device pipeline described in the paper (Algorithm 1 and
Sect. 5) in plain Python, so that the policy can be read, tested, and simulated without the board.
It is not the deployed implementation: the deployed C++ pipeline additionally handles hardware
decoding, RGA preprocessing, the NPU runtime, and tracking.

Notation (paper): T frame period, tau calibrated job time, K tiles, a additive increase,
gamma1 / gamma2 multiplicative decreases, theta wait threshold, D maximum frames in flight.
"""
from dataclasses import dataclass, field
import math


@dataclass
class AimdParams:
    frame_period_ms: float          # T
    job_ms: float                   # tau, calibrated per-job time in the running pipeline
    num_tiles: int                  # K
    a: float = 0.01
    gamma1: float = 0.03            # decrease when the full-frame job waited longer than theta
    gamma2: float = 0.10            # decrease when a frame had to be skipped
    theta_factor: float = 0.5       # theta = theta_factor * tau
    max_in_flight: int = 3          # D
    start_factor: float = 0.85      # mu_0 = start_factor * mu_max


@dataclass
class AimdScheduler:
    p: AimdParams
    mu: float = field(init=False)
    credit: float = field(init=False, default=0.0)
    next_tile: int = field(init=False, default=0)

    def __post_init__(self):
        self.mu_max = self.p.frame_period_ms / self.p.job_ms
        self.mu = self.p.start_factor * self.mu_max

    def plan_frame(self):
        """Jobs for a newly accepted frame: the mandatory full frame, then tiles from the credit."""
        jobs = [("full", None)]
        self.credit += self.mu - 1.0
        added = 0
        while self.credit >= 1.0 and added < self.p.num_tiles:
            jobs.append(("tile", self.next_tile))
            self.next_tile = (self.next_tile + 1) % self.p.num_tiles
            self.credit -= 1.0
            added += 1
        self.credit = min(1.0, max(-1.0, self.credit))  # no bursts, no long debt
        return jobs

    def on_frame_done(self, full_job_wait_ms):
        """AIMD update when the result of a frame is collected."""
        if full_job_wait_ms > self.p.theta_factor * self.p.job_ms:
            self.mu = max(1.0, self.mu * (1.0 - self.p.gamma1))
        else:
            self.mu = min(self.mu_max, self.mu + self.p.a)

    def on_frame_skipped(self):
        """A frame arrived while D frames were still in flight (after cancelling pending tiles)."""
        self.mu = max(1.0, self.mu * (1.0 - self.p.gamma2))


# ---------------------------------------------------------------------------- merge rule
def _iou(a, b):
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def nms_class_agnostic(dets, iou_thr=0.45):
    """Greedy NMS over all classes; dets are (x, y, w, h, score, cls) in frame pixels."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(_iou(d, k) <= iou_thr for k in keep):
            keep.append(d)
    return keep


def keep_tile_detection(det, tile_roi, frame_w, frame_h, full_scale, max_px=32.0, edge_px=2.0):
    """Merge rule: keep a tile detection only if no inner tile border cuts it and the object
    is small in the full-frame network input (sqrt(w*h) * full_scale < max_px)."""
    x, y, w, h = det[:4]
    rx, ry, rw, rh = tile_roi
    cut = ((rx > 0 and x <= rx + edge_px) or (ry > 0 and y <= ry + edge_px) or
           (rx + rw < frame_w - 1 and x + w >= rx + rw - edge_px) or
           (ry + rh < frame_h - 1 and y + h >= ry + rh - edge_px))
    return (not cut) and math.sqrt(w * h) * full_scale < max_px


def merge_frame(full_dets, tile_results, frame_w, frame_h, full_scale, iou_thr=0.45):
    """full_dets: detections of the full-frame job; tile_results: list of (tile_roi, dets)."""
    kept = list(full_dets)
    for roi, dets in tile_results:
        kept += [d for d in dets if keep_tile_detection(d, roi, frame_w, frame_h, full_scale)]
    return nms_class_agnostic(kept, iou_thr)
