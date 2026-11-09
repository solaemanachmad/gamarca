"""Figures of the GAMA-RCA paper (vector PDF, print / light mode). Data: build/per_case.csv and build/report.json, exported by
scripts/analyze_v5.py from the run zips; nothing here re-runs a model.

Palette: validated categorical slots 1-3 (blue / orange / aqua; dataviz skill reference palette, light mode): GAMA statistical core,
GAMA + LLM (Qwen2.5-7B), GAMA + LLM (gpt-oss-120b). Baselines in neutral gray. Direct labels on every series (aqua < 3:1 contrast:
relief rule)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
GRAY = "#9a9994"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6, "legend.fontsize": 6.4, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

GROUPS = {'httpError5xx': 'Application logic', 'rateLimiting': 'Application logic', 'trafficSurge': 'Application logic',
          'nullPointerException': 'Application logic', 'trafficHotspot': 'Application logic', 'loadBalancerFailure': 'Application logic',
          'codeDefect': 'Application logic', 'memoryPressure': 'JVM runtime', 'threadExhaustion': 'JVM runtime', 'fullGC': 'JVM runtime',
          'nodeCpuHigh': 'Cloud resource (node)', 'nodeDown': 'Cloud resource (node)', 'nodeMemoryOOM': 'Cloud resource (node)',
          'slowSQL': 'Middleware & DB', 'redisUnavailable': 'Middleware & DB', 'dbNetworkLatency': 'Middleware & DB',
          'messageQueueBacklog': 'Middleware & DB', 'cacheBreakdown': 'Middleware & DB', 'replicaScaleDown': 'K8s lifecycle',
          'resourceLimitMisconfig': 'K8s lifecycle', 'podCrashLoop': 'K8s lifecycle', 'podPendingUnschedulable': 'K8s lifecycle',
          'podRestartFlapping': 'K8s lifecycle', 'networkPolicyIsolation': 'K8s lifecycle', 'dnsResolutionFailure': 'K8s lifecycle',
          'cpuFullLoad': 'Resource & perf.', 'cpuDeadLoop': 'Resource & perf.', 'diskIOHigh': 'Resource & perf.'}

pc = pd.read_csv(os.path.join(HERE, "build", "per_case.csv"), index_col=0)
pc["group"] = pc["gt_fault_type"].map(lambda t: GROUPS[t.split("-", 1)[1]])
rep = json.load(open(os.path.join(HERE, "build", "report.json")))


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(m, [2.5, 97.5])


# ------------------------------------------------------------------------------------------------------------------------------
# Fig. 1 architecture
# ------------------------------------------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, title, lines, num=None, face="white", edge=INK2, dashed=False, title_color=INK, num_color=BLUE, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", lw=0.9 if not dashed else 0.9,
                                ec=edge, fc=face, ls="--" if dashed else "-"))
    tx = x + 0.012
    if num is not None:
        ax.add_patch(plt.Circle((x + 0.022, y + h - 0.035), 0.015, color=num_color, zorder=3))
        ax.text(x + 0.022, y + h - 0.035, str(num), color="white", ha="center", va="center", fontsize=7, weight="bold", zorder=4)
        tx = x + 0.045
    ax.text(tx, y + h - 0.035, title, ha="left", va="center", fontsize=7.6, weight="bold", color=title_color)
    yy = y + h - 0.08
    for ln in lines:
        bold = ln.startswith("**")
        t = ln[2:] if bold else ln
        ax.text(x + 0.012, yy, t, ha="left", va="top", fontsize=fs, color=INK if bold else INK2, weight="bold" if bold else "normal")
        yy -= 0.034 if bold else 0.031


def fig_architecture():
    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    H, Y = 0.47, 0.47
    fs = 5.9
    box(ax, 0.005, Y, 0.118, H, "Input", ["Alert window", "(start, end)", "", "Metrics", "Logs", "Traces", "K8s events",
                                           "Topology (calls,", "hosting nodes)", "", "no labels used"], face="#f3f3f1", fs=fs)
    box(ax, 0.137, Y, 0.215, H, "Calibrated evidence", [
        "**Per entity and signal (M0)", "robust shift vs the case's own", "history; peer and own-volatility",
        "re-standardisation; p-value from", "the case's pseudo-windows", "**Trace structure (M2)", "error / slow origin, relay to",
        "the unreachable callee", "**Mechanism text (M5)", "**Events, availability, onsets"], num=1, fs=fs)
    box(ax, 0.366, Y, 0.228, H, "Abductive belief", [
        r"$P(h\mid O)\propto P(h)\,\Pi_o\,P(o\mid h)$", "h = (entity, type), open anomaly, h0", "**Fingerprint roles",
        "key / support / may / contradict", "**Explain-away on the topology", "errors up to callers,", "load down to callees (M4)",
        "**Propagation strength π (EM)", "**Onset precedence", "**Cross-modal corroboration (M1)"], num=2, fs=fs)
    box(ax, 0.608, Y, 0.2, H, "Decision", [
        "**(a) root entity", r"$e^*=\mathrm{argmax}_e\;\Sigma_\tau\, P(e,\tau\mid O)$", "**(b) fault type at e*", r"$\tau^*=\mathrm{argmax}_\tau\, P(\tau\mid e^*,O)$,",
        "re-weighted by the order of", "the root's own signals (M3)", "**Evidence list", "log Bayes factors vs h0"], num=3, fs=fs)
    box(ax, 0.822, Y, 0.173, H, "Output", ["Root-cause entity", "Fault type", "Confidence", "",
                                            "**Causal chain (M6)", "cause → propagation", "→ impact; each step", "cites metric = value",
                                            "", "deterministic, CPU"], face="#f3f3f1", fs=fs)
    for x0, x1 in ((0.123, 0.137), (0.352, 0.366), (0.594, 0.608), (0.808, 0.822)):
        ax.annotate("", xy=(x1, Y + H / 2), xytext=(x0 - 0.001, Y + H / 2), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    lx, ly, lw, lh = 0.366, 0.17, 0.629, 0.2
    ax.add_patch(FancyBboxPatch((lx, ly), lw, lh, boxstyle="round,pad=0.004,rounding_size=0.012", lw=0.9, ec=ORANGE, fc="#fdf1ea", ls="--"))
    ax.add_patch(plt.Circle((lx + 0.022, ly + lh - 0.035), 0.015, color=ORANGE, zorder=3))
    ax.text(lx + 0.022, ly + lh - 0.035, "4", color="white", ha="center", va="center", fontsize=7, weight="bold", zorder=4)
    ax.text(lx + 0.045, ly + lh - 0.035, "Bounded LLM (M7, optional; local 7B or API model; <= 7 calls)", ha="left", va="center",
            fontsize=7.4, weight="bold")
    cols = [("Text sensor", ["entity chosen by expected", "information gain; verbatim", "quote; answer = likelihood"]),
            ("Shortlist typing", ["one type among the top 5", "of P(τ | e*); the quote must", "name a fingerprint signal"]),
            ("Narrator", ["writes the chain of the", "decided diagnosis; scored", "apart from the computed one"])]
    for i, (t, ls_) in enumerate(cols):
        cx = lx + 0.012 + i * 0.21
        ax.text(cx, ly + lh - 0.08, t, fontsize=6.4, weight="bold", va="top")
        for j, s_ in enumerate(ls_):
            ax.text(cx, ly + lh - 0.113 - j * 0.028, s_, fontsize=fs, color=INK2, va="top")
    for xa, lab, two in ((0.48, "likelihood", True), (0.70, "type choice", True), (0.905, "narrative", False)):
        ax.annotate("", xy=(xa, Y - 0.003), xytext=(xa, ly + lh + 0.003),
                    arrowprops=dict(arrowstyle="<|-|>" if two else "-|>", color=ORANGE, lw=0.9))
        ax.text(xa + 0.008, (Y + ly + lh) / 2, lab, fontsize=6, color="#b04a1f", va="center")
    for i, t in enumerate(["P1  Calibrate, do not threshold: every test is compared with the case's own history and peers.",
                           "P2  Causes precede their effects and propagate directionally (errors up, load down).",
                           "P3  The LLM reads text and chooses among ranked options; decision and evidence are computed."]):
        ax.text(0.005, 0.25 - i * 0.045, t, fontsize=6.0, color=INK2, va="top", wrap=False) if False else None
    ax.text(0.005, ly + lh - 0.035, "Design principles", fontsize=7.0, weight="bold", va="center")
    for i, t in enumerate(["P1  Calibrate, do not threshold:", "     compare with the case's own", "     history and peers.",
                           "P2  Causes precede effects and", "     propagate directionally.", "P3  The LLM reads text, chooses",
                           "     among ranked options."]):
        ax.text(0.005, ly + lh - 0.075 - i * 0.026, t, fontsize=5.8, color=INK2, va="top")
    fig.savefig(os.path.join(OUT, "fig1_architecture.pdf"))
    fig.savefig(os.path.join(OUT, "fig1_architecture.png"), dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------------
# Fig. 2 main results: Final and Final_step with 95% bootstrap CIs
# ------------------------------------------------------------------------------------------------------------------------------
REPKEY = {"core": "GAMA_CORE", "llm7b": "GAMA_LLM", "llmoss": "GAMA_LLM@gptoss", "belief7b": "BELIEF_LLM",
          "evid7b": "EVIDENCE_LLM_FINAL", "direct7b": "DIRECT_LLM", "notrace": "GAMA_CORE_NO_TRACE"}
ARMS = [("core", "GAMA (no LLM)", BLUE), ("llm7b", "GAMA + LLM 7B", ORANGE),
        ("llmoss", "GAMA + LLM 120B", AQUA), ("belief7b", "Belief→LLM 7B", GRAY),
        ("evid7b", "Evidence→LLM 7B", GRAY), ("direct7b", "Direct LLM 7B", GRAY)]


def fig_main():
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    ys = np.arange(len(ARMS))[::-1]
    for ax, col, title in ((axes[0], "F", "Final (headline RP)"), (axes[1], "Fs", "Final (step-anchored RP)")):
        name = "Final" if col == "F" else "Final_step"
        ref = rep["arms"]["GAMA_CORE"]["all"][name]
        for y, (k, lab, c) in zip(ys, ARMS):
            a = rep["arms"][REPKEY[k]]["all"]                 # the numbers of scripts/analyze_v5.py (bootstrap 5,000, seed 0)
            m, (lo, hi) = a[name], a[name + "_ci"]
            ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round")
            ax.plot([m], [y], "o", ms=6, color=c, mec=SURF, mew=1.2, zorder=3)
            right = not (hi + 1 < ref < hi + 9)
            ax.text(hi + 1.2 if right else lo - 1.2, y, f"{m:.1f}", va="center", ha="left" if right else "right", fontsize=7, color=INK)
        ax.axvline(ref, color=BLUE, lw=0.6, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(10, 80)
        ax.set_xticks([10, 20, 30, 40, 50, 60, 70])
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("mean and 95% bootstrap CI over the 103 cases (0-100); dashed line: GAMA", fontsize=6.8, color=INK2, y=-0.05)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([a[1] for a in ARMS])
    for t, (_, _, c) in zip(axes[0].get_yticklabels(), ARMS):
        t.set_color(INK if c != GRAY else INK2)
    fig.savefig(os.path.join(OUT, "fig2_main.pdf"))
    fig.savefig(os.path.join(OUT, "fig2_main.png"), dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------------
# Fig. 3 by fault group: EL and FI of the core and the LLM arms (heatmap, sequential blue)
# ------------------------------------------------------------------------------------------------------------------------------
def fig_groups():
    order = ["Application logic", "JVM runtime", "Resource & perf.", "Middleware & DB", "K8s lifecycle", "Cloud resource (node)"]
    cols = [("core", "GAMA"), ("llm7b", "+LLM\n7B"), ("llmoss", "+LLM\n120B"), ("evid7b", "Evidence\n→LLM 7B")]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), gridspec_kw=dict(wspace=0.05))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#bcd5f3", "#6aa2e6", "#2a78d6", "#174c8f"])
    for ax, met, title in ((axes[0], "EL", "Entity localisation (EL)"), (axes[1], "FI", "Fault identification (FI)")):
        M = np.array([[100 * pc.loc[pc.group == g, f"{k}_{met}"].mean() for k, _ in cols] for g in order])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if M[i, j] > 62 else INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c[1] for c in cols])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", color=INK, pad=24)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-.5, len(order)), minor=True)
        ax.grid(which="minor", color=SURF, lw=2)
        ax.tick_params(which="minor", length=0)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([f"{g} ({int((pc.group == g).sum())})" for g in order])
    axes[1].set_yticks([])
    fig.savefig(os.path.join(OUT, "fig3_groups.pdf"))
    fig.savefig(os.path.join(OUT, "fig3_groups.png"), dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------------------
# Fig. 4 decision vs explanation: (a) per-case Final change vs the core; (b) what LLM typing did; (c) narrator checkpoint rate
# ------------------------------------------------------------------------------------------------------------------------------
TYPING = {"llm7b": (3, 10, 12), "llmoss": (2, 7, 12)}     # type changes: to the true type / away from it / wrong -> wrong (post hoc)


def fig_llm():
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.2), gridspec_kw=dict(wspace=0.6, width_ratios=[1.1, 1.3, 0.8]))
    bk = [("llmoss", "120B", AQUA), ("llm7b", "7B", ORANGE)]
    # (a) better / worse / equal vs core
    ax = axes[0]
    for i, (k, lab, c) in enumerate(bk):
        d = pc[f"{k}_F"] - pc["core_F"]
        b, w = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        ax.barh(i, -w, color=c, height=0.5, alpha=0.55)
        ax.barh(i, b, color=c, height=0.5)
        ax.text(-w - 0.6, i, f"{w} worse", ha="right", va="center", fontsize=6.3, color=INK)
        ax.text(b + 0.6, i, f"{b} better", ha="left", va="center", fontsize=6.3, color=INK)
        ax.text(-6, i + 0.4, f"ΔFinal {100 * d.mean():+.1f}", ha="center", fontsize=6.2, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([b[1] for b in bk])
    ax.set_xlim(-34, 16)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 1.75)
    ax.set_xlabel("cases (rest equal)")
    ax.set_title("(a) Decision vs GAMA", loc="left", color=INK)
    ax.tick_params(axis="y", length=0)
    # (b) typing changes
    ax = axes[1]
    labels = ["wrong\n→ true", "true\n→ wrong", "wrong\n→ wrong"]
    x = np.arange(3)
    for k, lab, c in bk[::-1]:
        v = TYPING[k]
        i = 0 if k == "llm7b" else 1
        ax.bar(x + (i - 0.5) * 0.36, v, width=0.34, color=c, label=lab)
        for xx, vv in zip(x + (i - 0.5) * 0.36, v):
            ax.text(xx, vv + 0.3, str(vv), ha="center", fontsize=6.6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("cases")
    ax.set_ylim(0, 15)
    ax.set_title("(b) Type changes", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=6.4, handlelength=1)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    # (c) narrator step-anchored checkpoint rate
    ax = axes[2]
    vals = [pc["llm7b_cpsllm"].mean(), pc["llmoss_cpsllm"].mean()]
    ax.bar([0, 1], vals, width=0.55, color=[ORANGE, AQUA])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.8, color=INK)
    ref = pc["core_cps"].mean()
    ax.axhline(ref, color=BLUE, lw=0.8, ls=(0, (3, 2)))
    ax.text(-0.5, ref + 0.008, f"computed chain {ref:.3f}", fontsize=5.8, color=BLUE, ha="left", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["7B", "120B"])
    ax.set_ylim(0, 0.42)
    ax.set_xlim(-0.6, 1.4)
    ax.set_ylabel("step-anchored checkpoint rate")
    ax.set_title("(c) Narrator", loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, "fig4_llm.pdf"))
    fig.savefig(os.path.join(OUT, "fig4_llm.png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["arch", "main", "groups", "llm"]
    if "arch" in which:
        fig_architecture()
    if "main" in which:
        fig_main()
    if "groups" in which:
        fig_groups()
    if "llm" in which:
        fig_llm()
    # numbers used in the text (sanity check against report.json)
    for k, lab, _ in ARMS:
        print(f"{lab:36s} Final {100 * pc[k + '_F'].mean():5.1f} CI {np.round(100 * boot_ci(pc[k + '_F']), 1)}  "
              f"Final_step {100 * pc[k + '_Fs'].mean():5.1f}")
