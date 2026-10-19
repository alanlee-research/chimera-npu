#!/usr/bin/env python3
"""Draws Fig. 2 (Chimera architecture) and Fig. 4 (qualitative comparison) of the manuscript."""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})


def arch():
    fig, ax = plt.subplots(figsize=(7.0, 2.5)); ax.set_xlim(-1, 111.5); ax.set_ylim(-1, 35); ax.axis("off")
    C = {"bb": "#e2e8f0", "ctx": "#bbf7d0", "gate": "#fde68a", "head": "#bfdbfe", "op": "#f8fafc"}
    H = 4.4

    def box(x0, x1, y, t, c, fs=5.7):
        ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, H, boxstyle="round,pad=0.2,rounding_size=0.7", fc=c, ec="black", lw=0.55))
        ax.text((x0 + x1) / 2, y + H / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.05); return (x0, x1, y)
    L = lambda b: (b[0], b[2] + H / 2); R = lambda b: (b[1], b[2] + H / 2)
    T = lambda b, dx=0: ((b[0] + b[1]) / 2 + dx, b[2] + H); B = lambda b, dx=0: ((b[0] + b[1]) / 2 + dx, b[2])

    def arr(p, q):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=0.55, color="black", shrinkA=0, shrinkB=0, mutation_scale=6))
    y1, y2, y3, y4 = 29, 19, 9, -0.5
    r = [box(0, 7, y1, "Input\n832$\\times$480", C["op"]), box(8.5, 14.5, y1, "Conv s2\n16 ch", C["bb"]), box(16, 22, y1, "Conv s2\n32 ch", C["bb"]),
         box(23.5, 35, y1, "C3k2 ($P2$)\n64 ch, 120$\\times$208", C["bb"]), box(36.5, 42.5, y1, "Conv s2\n64 ch", C["bb"]),
         box(44, 55.5, y1, "C3k2 ($P3$)\n128 ch, 60$\\times$104", C["bb"]), box(57, 63, y1, "Conv s2\n128 ch", C["bb"]),
         box(64.5, 76, y1, "C3k2 ($P4$)\n128 ch, 30$\\times$52", C["bb"]), box(77.5, 83.5, y1, "SPPF", C["bb"]), box(87, 96, y1, "MDRF\n$d$ = 1, 3", C["ctx"])]
    for a, b in zip(r[:-1], r[1:]): arr(R(a), L(b))
    P2, P3, MD = r[3], r[5], r[9]
    sd = box(23.5, 35, y2, "Conv 3$\\times$3 s2\n64 ch", C["gate"]); arr(B(P2), T(sd))
    c1 = box(44, 51, y2, "Concat\n192 ch", C["op"]); arr(R(sd), L(c1)); arr(B(P3, T(c1)[0] - (P3[0] + P3[1]) / 2), T(c1))
    hf = box(53, 64, y2, "HFGate\n$\\mathbf{F}_3\\odot(1+\\mathbf{G})$", C["gate"]); arr(R(c1), L(hf))
    c2 = box(66, 73, y2, "Concat\n320 ch", C["op"]); arr(R(hf), L(c2))
    up = box(87, 96, y2, "Upsample\n$\\times$2", C["op"]); arr(B(MD), T(up)); arr(L(up), R(c2))
    fu = box(64.5, 74.5, y3, "C3k2 fusion\n($P3$) 128 ch", C["bb"]); arr(B(c2, 0), T(fu))
    sq = box(50.5, 61.5, y3, "Conv 1$\\times$1\n128$\\to$64 ch", C["head"]); arr(L(fu), R(sq))
    h3 = box(38, 48, y3, "Head $P3$\nstride 8", C["head"]); arr(L(sq), R(h3))
    ds = box(64.5, 74.5, y4, "Conv s2\n128 ch", C["bb"]); arr(B(fu), T(ds))
    c3 = box(78, 87, y4, "Concat\n256 ch", C["op"]); arr(R(ds), L(c3))
    xm = (78 + 87) / 2
    ax.plot([96, 97.8, 97.8, xm], [y1 + H / 2, y1 + H / 2, y3 + H / 2, y3 + H / 2], color="black", lw=0.55); arr((xm, y3 + H / 2), T(c3))
    c4 = box(90, 99, y4, "C3k2 ($P4$)\n128 ch", C["bb"]); arr(R(c3), L(c4))
    h4 = box(101.5, 110.5, y4, "Head $P4$\nstride 16", C["head"]); arr(R(c4), L(h4))
    ax.text(1, 13, "$P5$ stage removed\ncontext: MDRF at stride 16\nstride-4 detail: HFGate\nheads at strides 8 and 16", fontsize=6.6, va="center")
    fig.savefig("fig_arch_jr.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig("fig_arch_jr.png", dpi=220, bbox_inches="tight")


def qual():
    a = cv2.imread("fig_qual_a.jpg")[0:300, 0:980][:, :, ::-1]; b = cv2.imread("fig_qual_b.jpg")[0:300, 0:980][:, :, ::-1]
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 2.4))
    for ax, im, t in [(axs[0], a, "(a) Full frame only"), (axs[1], b, "(b) Full frame + four tiles (merge rule)")]:
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=7.5, pad=2)
    plt.subplots_adjust(hspace=0.2, left=0.01, right=0.99, top=0.94, bottom=0.01)
    fig.savefig("fig5_qualitative_jr.pdf", dpi=300); fig.savefig("fig5_qualitative_jr.png", dpi=200)


if __name__ == "__main__":
    arch(); qual()
