"""Markdown tables for the report (plot environment). Writes analysis/tables.md."""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "analysis")
runs = pd.read_csv(os.path.join(A, "runs.csv"))
starts = pd.read_csv(os.path.join(A, "starts.csv"))
gt = json.load(open(os.path.join(HERE, "ground_truth.json")))
NICE = {"COBYLA": "COBYLA", "AQGD": "AQGD", "SPSA": "SPSA", "NELDER_MEAD": "Nelder–Mead", "POWELL": "Powell",
        "BOBYQA": "BOBYQA", "L_BFGS_B": "L-BFGS-B", "SLSQP": "SLSQP", "ADAM": "Adam", "NFT": "NFT"}
INIT = {"random": "Random initialization (fixed 0.5 start, the team's baseline)", "spiq": "SPIQ initialization (the team's method)"}

def med(s, fmt):
    s = s.dropna()
    return "–" if s.empty else fmt.format(s.median())

out = []
for I in ("random", "spiq"):
    r = runs[runs.init == I]
    if r.empty: continue
    out.append(f"### Leaderboard — {INIT[I]}\n")
    out.append("Median over trials. **Gap closed** = fraction of the distance from the starting energy to the exact ground state "
               "that the returned parameters close (1.00 = ground state reached). **P(opt)** = the paper's optimal-plan ratio "
               "(fraction of 10,240 re-sampled shots decoding, with fallback, to an optimal-cost join order).\n")
    cols = ["Rank", "Optimizer"] + [f"Gap closed {P}" for P in ("P1", "P2", "P3")] + \
           [f"P(opt) {P}" for P in ("P1", "P2", "P3")] + ["Evals (median, all problems)", "Trials"]
    rows = []
    for O, g in r.groupby("optimizer"):
        score = np.nanmean([g[g.problem == P].gap_closed.median() for P in ("P1", "P2", "P3")])
        rows.append((score, O, g))
    rows.sort(key=lambda x: -x[0] if not np.isnan(x[0]) else 1e9)
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "---|" * len(cols))
    st = {P: starts[(starts.problem == P) & (starts.init == I)] for P in ("P1", "P2", "P3")}
    out.append("| – | *start (before optimizing)* | " + " | ".join("0.00" for P in ("P1", "P2", "P3")) + " | " +
               " | ".join(f"{float(st[P].P_opt_fb_exact.iloc[0]):.3f}" if not st[P].empty else "–" for P in ("P1", "P2", "P3")) + " | – | – |")
    for k, (score, O, g) in enumerate(rows, 1):
        gc = [med(g[g.problem == P].gap_closed, "{:.2f}") for P in ("P1", "P2", "P3")]
        po = [med(g[g.problem == P].opt_ratio_fb, "{:.3f}") for P in ("P1", "P2", "P3")]
        out.append(f"| {k} | **{NICE[O]}** | " + " | ".join(gc) + " | " + " | ".join(po) +
                   f" | {int(g.evaluations.median())} | " +
                   " · ".join(f"{P} {n}" for P, n in g.groupby('problem').trial.nunique().items()) + " |")
    out.append("")

out.append("### Detailed results per problem\n")
for P in ("P1", "P2", "P3"):
    out.append(f"#### {P} — {len(gt[P]['card'])} relations, {len(gt[P]['pred'])} predicates, {gt[P]['num_qubits']} qubits · "
               f"ground state {gt[P]['ground_state_energy']:.3f} · optimal plan cost {gt[P]['optimal_cost']:g}\n")
    out.append("| Init | Optimizer | Exact energy, median [min–max] | Gap closed | P(opt) raw | P(opt) fallback | "
               "E[plan cost] fallback | Evals | Budget hit | Wall time (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for I in ("random", "spiq"):
        s = starts[(starts.problem == P) & (starts.init == I)]
        if not s.empty:
            s = s.iloc[0]
            out.append(f"| {I} | *start* | {s.E_exact:.3f} | 0.00 | {s.P_opt_raw_exact:.3f} | {s.P_opt_fb_exact:.3f} | "
                       f"{s.mean_cost_fb_exact:.0f} | – | – | – |")
        g = runs[(runs.problem == P) & (runs.init == I)]
        for O, h in sorted(g.groupby("optimizer"), key=lambda kv: kv[1].E_exact.median()):
            out.append(f"| {I} | {NICE[O]} | {h.E_exact.median():.3f} [{h.E_exact.min():.3f}–{h.E_exact.max():.3f}] | "
                       f"{h.gap_closed.median():.2f} | {h.opt_ratio_raw.median():.3f} | {h.opt_ratio_fb.median():.3f} | "
                       f"{h.mean_cost_fb_exact.median():.0f} | {int(h.evaluations.median())} | "
                       f"{int(h.budget_hit.astype(str).eq('True').sum())}/{len(h)} | {h.wall_time_s.median():.0f} |")
    out.append("")
open(os.path.join(A, "tables.md"), "w").write("\n".join(out))
print("\n".join(out[:18]))
