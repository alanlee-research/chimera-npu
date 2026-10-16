#!/usr/bin/env python3
# =====================================================================================================
# FIGURES FOR THE CHIMERA PAPER — ALL NUMBERS TAKEN FROM THE SSOT TABLE
# Single-column figures are drawn at the true IEEE column width (3.5 in) so fonts print at real size.
# =====================================================================================================
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,          # TrueType embedding (IEEE PDF eXpress rejects Type 3)
    'ps.fonttype': 42,
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.6,
    'grid.alpha': 0.35,
    'grid.linewidth': 0.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

COL = 3.5     # IEEE single column (in)
PAGE = 7.16   # IEEE double column (in)

# ---------------------------------------------------------------- SSOT
LAT = {  # pure NPU latency, ms, RK3582 Core 0 @ 900 MHz
    "YOLO11n": 18.27,
    "Chimera-Proto": 18.64,
    "Chimera-SiLU": 16.67,
    "YOLO11n-LeakyReLU": 11.63,
    "Chimera-LReLU": 10.88,
}
MAP = {  # VisDrone test mAP50-95, %
    "YOLO11n": 35.50,
    "Chimera-Proto": 35.59,
    "Chimera-SiLU": 35.79,
    "YOLO11n-LeakyReLU": 34.49,
    "Chimera-LReLU": 34.51,
}
STAGES = ['Back-\nbone', 'P5\nstage', 'MDRF', 'Neck', 'Head\nP3', 'Heads\nP4/P5', 'Mem.\ndisp.']
YOLO_STAGE = [9.04, 3.90, 0.00, 5.24, 2.14, 1.71, 0.00]      # sum = 22.03
CHIM_STAGE = [9.23, 0.00, 1.21, 4.97, 3.44, 1.16, 0.85]      # sum = 20.86
assert abs(sum(YOLO_STAGE) - 22.03) < 1e-6 and abs(sum(CHIM_STAGE) - 20.86) < 1e-6
BATCH_D = {"YOLO11n": (18.29, 11.62), "Chimera": (16.67, 10.90)}  # SiLU, LeakyReLU, same architecture, same session
TAU = 10.88   # full-frame / tile job latency used by the scheduler
T = 1000 / 30


def save(fig, name):
    name = name + "_jr"
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Fig. 1
def fig1():
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    style = {
        "YOLO11n":               ("#ef4444", "o", (46.5, 35.02)),
        "Chimera-Proto":         ("#0ea5e9", "^", (46.5, 36.02)),
        "Chimera-SiLU":          ("#6366f1", "P", (65.0, 36.02)),
        "YOLO11n-LeakyReLU":     ("#f59e0b", "D", (63.0, 34.02)),
        "Chimera-LReLU":         ("#10b981", "*", (75.0, 35.22)),
    }
    for name, (c, m, txt) in style.items():
        fps = 1000 / LAT[name]
        ax.scatter(fps, MAP[name], s=110 if m == "*" else 36, color=c, marker=m,
                   edgecolors="black", linewidth=0.6, zorder=4)
        ax.annotate(f"{name}\n{LAT[name]:.2f} ms", xy=(fps, MAP[name]), xytext=txt, fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=c, lw=0.6),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6), zorder=5)
    f = ["Chimera-SiLU", "Chimera-LReLU"]
    ax.plot([1000 / LAT[k] for k in f], [MAP[k] for k in f], "--", color="#059669",
            lw=1.0, alpha=0.8, label="Pareto front", zorder=3)
    ax.set_xlabel("NPU-only throughput, 1000 / latency (FPS)")
    ax.set_ylabel("VisDrone test mAP$_{50:95}$ (%)")
    ax.set_xlim(45, 100)
    ax.set_ylim(33.6, 36.3)
    ax.grid(True)
    ax.legend(loc="lower left", frameon=True)
    save(fig, "fig1_pareto_frontier")


# ---------------------------------------------------------------- Fig. 2
def fig2():
    fig = plt.figure(figsize=(PAGE, 2.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[3.2, 1, 1, 1.15], wspace=0.6)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(STAGES)); w = 0.38
    ax.bar(x - w/2, YOLO_STAGE, w, color='#f87171', edgecolor='black', lw=0.5,
           label=f'YOLO11n ({sum(YOLO_STAGE):.2f} ms)')
    ax.bar(x + w/2, CHIM_STAGE, w, color='#38bdf8', edgecolor='black', lw=0.5,
           label=f'Chimera-Proto ({sum(CHIM_STAGE):.2f} ms)')
    for i, (a, b) in enumerate(zip(YOLO_STAGE, CHIM_STAGE)):
        if a == 0: ax.text(i - w/2, 0.15, "—", ha='center', fontsize=6)
        if b == 0: ax.text(i + w/2, 0.15, "—", ha='center', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(STAGES, fontsize=6.5)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("(a) Stage-wise latency (SiLU models)", fontweight="bold")
    ax.grid(axis='y'); ax.legend(fontsize=6.3)

    def small(pos, title, ylab, labels, vals, ymax, colors):
        a = fig.add_subplot(gs[pos])
        a.bar(range(len(vals)), vals, 0.6, color=colors, edgecolor='black', lw=0.5)
        for j, v in enumerate(vals):
            a.text(j, v + ymax * 0.02, f"{v:g}", ha='center', fontsize=6.3)
        a.set_xticks(range(len(vals))); a.set_xticklabels(labels, fontsize=6.3)
        a.set_ylabel(ylab, fontsize=7); a.set_ylim(0, ymax * 1.15)
        a.set_title(title, fontweight="bold", fontsize=7.5); a.grid(axis='y')
        return a

    small(1, "(b) DDR traffic", "MB / frame", ["YOLO11n", "Proto"], [33.0, 38.0], 40,
          ['#f87171', '#38bdf8'])
    small(2, "(c) Peak MAC load", "%", ["YOLO11n", "Proto"], [76.2, 84.1], 90,
          ['#f87171', '#38bdf8'])
    a = fig.add_subplot(gs[3])
    xs = np.arange(2); ww = 0.36
    silu = [BATCH_D["YOLO11n"][0], BATCH_D["Chimera"][0]]
    lre = [BATCH_D["YOLO11n"][1], BATCH_D["Chimera"][1]]
    a.bar(xs - ww/2, silu, ww, color='#cbd5e1', edgecolor='black', lw=0.5, label='SiLU')
    a.bar(xs + ww/2, lre, ww, color='#10b981', edgecolor='black', lw=0.5, label='LeakyReLU')
    for j in range(2):
        a.text(j - ww/2, silu[j] + 0.4, f"{silu[j]:.2f}", ha='center', fontsize=5.6)
        a.text(j + ww/2, lre[j] + 0.4, f"{lre[j]:.2f}", ha='center', fontsize=5.6)
    a.set_xticks(xs); a.set_xticklabels(["YOLO11n", "Chimera"], fontsize=6.3)
    a.set_ylabel("Latency (ms)", fontsize=7); a.set_ylim(0, 24)
    a.set_title("(d) Activation (batch D)", fontweight="bold", fontsize=7.5)
    a.grid(axis='y'); a.legend(fontsize=5.8, loc='upper right')
    save(fig, "fig2_layer_breakdown")


# ---------------------------------------------------------------- Fig. 3
def fig3():
    """Schedule of Algorithm 1 on Core 0, 30 FPS source, calibrated job time 11.2 ms (mu_max = 2.97,
    mu_0 = 0.85 mu_max = 2.5). Tile jobs per frame follow the credit accumulator c += mu - 1:
    1, 2, 1. In frame 2 the tiles are slowed by memory contention, the full-frame job of frame 3
    waits longer than theta = 0.5 tau for the core, and mu is decreased."""
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    tau, theta = 11.2, 5.6
    b = [0, T, 2 * T, 3 * T]
    for fb in b:
        ax.axvline(fb, color="black", ls="--", lw=0.7, alpha=0.7)
    h, rows = 0.5, [2.0, 1.2, 0.4]
    full = dict(color="#10b981", ec="black", lw=0.5)
    tile = dict(color="#38bdf8", ec="black", lw=0.5)
    # frame 1: credit 1.5 -> one tile
    ax.barh(rows[0], tau, left=b[0], height=h, label="Full-frame job (mandatory)", **full)
    ax.barh(rows[0], tau, left=b[0] + tau, height=h, label="Tile job (optional)", **tile)
    # frame 2: credit 2.0 -> two tiles, slowed by memory contention (14 ms each)
    t = b[1]
    ax.barh(rows[1], tau, left=t, height=h, **full)
    ax.barh(rows[1], 14.0, left=t + tau, height=h, **tile)
    ax.barh(rows[1], 14.0, left=t + tau + 14.0, height=h, **tile)
    busy_until = t + tau + 28.0
    # frame 3: full-frame job waits for the running tile, wait > theta -> mu decreased; one tile
    t = b[2]
    wait = busy_until - t
    ax.barh(rows[2], wait, left=t, height=h, color="#fca5a5", ec="black", lw=0.5, hatch="////",
            label="Wait for the core")
    ax.barh(rows[2], tau, left=busy_until, height=h, **full)
    ax.barh(rows[2], tau, left=busy_until + tau, height=h, **tile)
    ax.annotate(f"wait {wait:.1f} ms > $\\theta$\n$\\mu \\leftarrow \\mu(1-\\gamma_1)$",
                xy=(t + wait / 2, rows[2] + h / 2), xytext=(t + 7, 0.98), fontsize=7, color="#b91c1c",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.6))
    for k in range(3):
        ax.text(b[k] + 0.8, 2.47, f"frame {k + 1} arrives", fontsize=7, color="#475569")
    ax.set_yticks(rows); ax.set_yticklabels(["Frame 1", "Frame 2", "Frame 3"], fontsize=7.5)
    ax.set_xlabel("Time on NPU Core 0 (ms)")
    ax.set_xlim(-1, 102); ax.set_ylim(-0.05, 2.75)
    ax.grid(axis='x')
    ax.legend(loc="lower left", fontsize=6.5, ncol=1, framealpha=0.95, bbox_to_anchor=(0.0, -0.02))
    save(fig, "fig3_hybrid_timeline")


# ---------------------------------------------------------------- Fig. 4
def fig4():
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    cats = ['All objects', 'Objects < 32 px']
    full_only, hybrid, sahi = [48.5, 26.2], [88.7, 85.3], [100.0, 100.0]
    x = np.arange(2); w = 0.25
    ax.bar(x - w, full_only, w, label='Full-frame only', color='#94a3b8', ec='black', lw=0.5)
    ax.bar(x, hybrid, w, label='Hybrid AIMD (ours)', color='#10b981', ec='black', lw=0.5)
    ax.bar(x + w, sahi, w, label='Pure SAHI (reference)', color='#38bdf8', ec='black', lw=0.5, hatch='..')
    for i in range(2):
        for dx, v in ((-w, full_only[i]), (0, hybrid[i]), (w, sahi[i])):
            ax.text(i + dx, v + 1.5, f"{v:g}", ha='center', fontsize=6.3)
    ax.set_ylabel("Coverage relative to SAHI (%)")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 128); ax.set_yticks(range(0, 101, 20))
    ax.grid(axis='y')
    ax.legend(loc='upper center', ncol=3, fontsize=6, framealpha=0.95, columnspacing=0.8, handlelength=1.4)
    save(fig, "fig4_coverage_sahi")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
