"""Figures for the optimizer study (plot environment: matplotlib).

Every figure is rendered twice, light and dark, from the same validated palette
(dataviz reference palette; categorical pair and ordinal ramps checked with the
palette validator). Color carries ONE job per figure:
  * initialization  : blue = random / uninformed start, orange = SPIQ start
  * plan-cost rank  : one-hue ordinal blue ramp (optimal -> worse)
Optimizer identity is always written on the axis, never encoded by color.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "analysis")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", random="#2a78d6", spiq="#eb6834",
                  ordinal=["#104281", "#256abf", "#5598e7", "#86b6ef"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                  axis="#383835", random="#3987e5", spiq="#d95926",
                  ordinal=["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]),
}
INIT_LABEL = {"random": "Random init (fixed 0.5 start)", "spiq": "SPIQ init (team's method)"}
PROB_LABEL = {"P1": "P1 · 3 relations · 5 qubits", "P2": "P2 · 4 relations · 12 qubits",
              "P3": "P3 · 4 relations, 6 predicates · 20 qubits"}
NICE = {"COBYLA": "COBYLA", "AQGD": "AQGD", "SPSA": "SPSA", "NELDER_MEAD": "Nelder–Mead", "POWELL": "Powell",
        "BOBYQA": "BOBYQA", "L_BFGS_B": "L-BFGS-B", "SLSQP": "SLSQP", "ADAM": "Adam", "NFT": "NFT"}

def style(t):
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": t["ink"], "axes.titlelocation": "left", "legend.frameon": False,
        "axes.grid": False, "grid.color": t["grid"], "grid.linewidth": 0.8, "lines.solid_capstyle": "round",
    })

def save(fig, name, mode):
    fig.savefig(os.path.join(FIG, f"{name}-{mode}.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

def order(runs):
    """One fixed optimizer order for every figure: best mean gap-closed (across problems & inits) first."""
    g = runs.groupby("optimizer")["gap_closed"].mean().sort_values(ascending=False)
    return list(g.index)

# ---------------------------------------------------------------------------------------------
def fig_convergence(runs, traj, gt, P, init, mode):
    t = THEMES[mode]; style(t)
    opts = order(runs)
    d = traj[(traj.problem == P) & (traj.init == init)]
    if d.empty: return
    gs = gt[P]
    lo = gs - 0.08 * (d.E_logged.quantile(0.98) - gs)
    hi = d.E_logged.quantile(0.98)
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharey=True, sharex=True)
    c = t[init]
    for ax, O in zip(axes.flat, opts):
        dd = d[d.optimizer == O]
        ax.set_title(NICE[O], fontsize=10.5)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.axhline(gs, color=t["muted"], lw=1.2)
        if dd.empty:
            ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center", color=t["muted"]); continue
        for T, tr in dd.groupby("trial"):
            ax.plot(tr["eval"], tr.E_logged, color=c, lw=0.6, alpha=0.22)
        # hold each trial's final answer until the LAST trial of this optimizer stops
        med = dd.pivot_table(index="eval", columns="trial", values="E_exact_incumbent").ffill().median(axis=1)
        ax.plot(med.index, med.values, color=c, lw=2)
        ax.set_ylim(lo, hi)
        ax.tick_params(labelsize=8.5)
    axes.flat[0].annotate("ground state", xy=(1, gs), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                          textcoords="offset points", ha="right", va="bottom", color=t["ink2"], fontsize=8.5)
    for ax in axes[1]: ax.set_xlabel("energy evaluations")
    for ax in axes[:, 0]: ax.set_ylabel("⟨H_C⟩  (QUBO energy)")
    fig.suptitle(f"Energy convergence — {PROB_LABEL[P]} — {INIT_LABEL[init]}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.925, "Faint: every logged 10,240-shot evaluation (all trials).  Bold: exact energy of the incumbent "
             "(the point the optimizer would return if stopped here), median of trials; it ends where the optimizer stops.  "
             "Grey: ground state.", ha="left", color=t["ink2"], fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, f"convergence_{P}_{init}", mode)

# ---------------------------------------------------------------------------------------------
def dotplot(runs, starts, gt, metric, xlabel, title, subtitle, name, mode, ref="ground", xlim=None):
    t = THEMES[mode]; style(t)
    opts = order(runs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
    for ax, P in zip(axes, ["P1", "P2", "P3"]):
        ax.set_title(PROB_LABEL[P], fontsize=10.5)
        ax.xaxis.grid(True); ax.set_axisbelow(True)
        for k, I in enumerate(["random", "spiq"]):
            off = -0.17 if I == "random" else 0.17
            st = starts[(starts.problem == P) & (starts.init == I)]
            if metric in st and not st.empty:
                ax.axvline(float(st[metric].iloc[0]), color=t[I], lw=1, alpha=0.55)
            for i, O in enumerate(opts):
                v = runs[(runs.problem == P) & (runs.init == I) & (runs.optimizer == O)][metric].dropna()
                if v.empty: continue
                y = i + off
                ax.scatter(v, np.full(len(v), y), s=22, color=t[I], alpha=0.55, linewidths=0, zorder=3)
                m = v.median()
                ax.plot([m, m], [y - 0.14, y + 0.14], color=t[I], lw=2.6, zorder=4, solid_capstyle="round")
        if ref == "ground":
            ax.axvline(gt[P], color=t["ink2"], lw=1.2)
            ax.annotate("ground state", xy=(gt[P], 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                        textcoords="offset points", va="top", fontsize=8.5, color=t["ink2"])
        if xlim: ax.set_xlim(*xlim)
        if metric == "evaluations": ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(opts))); axes[0].set_yticklabels([NICE[o] for o in opts], fontsize=10)
    axes[0].invert_yaxis()
    h = [plt.Line2D([], [], marker="o", ls="", color=t[I], label=INIT_LABEL[I]) for I in ["random", "spiq"]]
    fig.legend(handles=h, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 0.975), fontsize=9.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.905, subtitle, ha="left", color=t["ink2"], fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, name, mode)

# ---------------------------------------------------------------------------------------------
def fig_cost_dist(runs, dist, starts_dist, init, mode):
    t = THEMES[mode]; style(t)
    opts = order(runs)
    labels = ["optimal plan", "2nd-cheapest", "3rd-cheapest", "more expensive"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True)
    for c, P in enumerate(["P1", "P2", "P3"]):
        for r, dec in enumerate(["raw", "fallback"]):
            ax = axes[r, c]
            ax.set_title(f"{PROB_LABEL[P]}\n{'raw decoding' if dec == 'raw' else 'with fallback (Schönberger et al.)'}", fontsize=10)
            rows = [("start (before optimizing)", starts_dist[(starts_dist.problem == P) & (starts_dist.init == init) & (starts_dist.decoder == dec)])]
            rows += [(NICE[O], dist[(dist.problem == P) & (dist.init == init) & (dist.optimizer == O) & (dist.decoder == dec)]) for O in opts]
            costs = sorted(starts_dist[starts_dist.problem == P].cost.unique())   # every plan-cost class of P
            for i, (lab, dd) in enumerate(rows):
                if dd.empty: continue
                ntr = max(dd.trial.nunique(), 1) if "trial" in dd else 1
                p = dd.groupby("cost").prob.sum() / ntr
                p = p.reindex(costs, fill_value=0)
                bins = [p.iloc[0], p.iloc[1] if len(p) > 1 else 0, p.iloc[2] if len(p) > 2 else 0, p.iloc[3:].sum()]
                left = 0
                for b, v in enumerate(bins):
                    if v <= 0: continue
                    ax.barh(i, v, left=left, height=0.62, color=t["ordinal"][b], edgecolor=t["surface"], linewidth=1.5)
                    if b == 0 and v >= 0.07:
                        ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                                color="#ffffff" if mode == "light" else "#0b0b0b")
                    left += v
            ax.set_xlim(0, 1); ax.xaxis.grid(True); ax.set_axisbelow(True)
            ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.tick_params(axis="y", length=0)
            if c == 0:
                ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    axes[0, 0].invert_yaxis()
    h = [plt.Rectangle((0, 0), 1, 1, color=t["ordinal"][b], label=l) for b, l in enumerate(labels)]
    fig.legend(handles=h, loc="upper right", ncol=4, bbox_to_anchor=(0.995, 0.975), fontsize=9.5)
    fig.suptitle(f"Which plans does the final state produce? — {INIT_LABEL[init]}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.01, 0.925, "Exact probability mass on each plan-cost rank at the returned parameters, MEAN over trials "
             "(the paper's Table 1 'cost ratio', computed without shot noise; tables report medians).", ha="left", color=t["ink2"], fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, f"plan_costs_{init}", mode)

# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    runs = pd.read_csv(os.path.join(A, "runs.csv"))
    tp = os.path.join(A, "trajectories.csv")
    traj = pd.read_csv(tp if os.path.exists(tp) else tp + ".gz")       # published copy is gzipped
    starts = pd.read_csv(os.path.join(A, "starts.csv"))
    dist = pd.read_csv(os.path.join(A, "cost_dist.csv"))
    sdist = pd.read_csv(os.path.join(A, "starts_cost_dist.csv"))
    gt = {k: v["ground_state_energy"] for k, v in json.load(open(os.path.join(HERE, "ground_truth.json"))).items()}
    for mode in ("light", "dark"):
        for P in ("P1", "P2", "P3"):
            for I in ("random", "spiq"):
                fig_convergence(runs, traj, gt, P, I, mode)
        dotplot(runs, starts, gt, "E_exact", "exact energy of the returned parameters  (lower is better)",
                "Final solution quality — exact energy of what each optimizer returns",
                "Dots: individual trials.  Bars: median.  Thin coloured lines: energy at each starting point.  "
                "Noise-free statevector evaluation of the parameters the pipeline decodes into plans.",
                "final_energy", mode)
        dotplot(runs, starts, gt, "opt_ratio_fb", "probability of sampling the optimal join order  (higher is better)",
                "Optimal-plan ratio — the paper's headline metric (fallback decoding)",
                "Fraction of the pipeline's 10,240 re-sampled shots whose decoded plan (with the Schönberger fallback) has "
                "optimal cost.  Thin coloured lines: value at the start.", "optimal_ratio_fallback", mode, ref=None, xlim=(0, None))
        dotplot(runs, starts, gt, "opt_ratio_raw", "probability of sampling the optimal join order  (higher is better)",
                "Optimal-plan ratio — raw decoding (no fallback)",
                "Same metric without the fallback repair step.  Thin coloured lines: value at the start.",
                "optimal_ratio_raw", mode, ref=None, xlim=(0, None))
        dotplot(runs, starts, gt, "evaluations", "energy evaluations used  (log scale, fewer is cheaper)",
                "Cost — how many 10,240-shot circuit evaluations each optimizer spends",
                "Budget: 1,000 (random init) · 2,500 (SPIQ, P1–P2) · 1,500 (SPIQ, P3).  Runs at the budget were cut off, not converged.",
                "evaluations", mode, ref=None)
        for I in ("random", "spiq"):
            fig_cost_dist(runs, dist, sdist, I, mode)
    print("figures:", len(os.listdir(FIG)))
