"""Reproduction check: the harness's COBYLA runs vs the team's own committed COBYLA logs.

Only the team's logs that are valid for the CURRENT workloads are compared (identified by
parameter count and by their deterministic first energy; see README §3). Their P1 SPIQ logs
(Week81/85) used an older 1-predicate P1 and are excluded.
"""
import csv, gzip, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
THEIRS = {  # (problem, init): committed energy log(s)
    ("P1", "random"): ["outputs/uninitialized/energy/Week71/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_10000/reps_2/COBYLA/input0/trial1/energy_per_iteration_10000_COBYLA_2_1.csv"],
    ("P2", "random"): ["outputs/uninitialized/energy/Week74/Week84/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_10000/reps_2/COBYLA/input0/trial1/energy_per_iteration_10000_COBYLA_2_1.csv"],
    ("P2", "spiq"): ["outputs/spiq/energy/Week84/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_10000/reps_2/COBYLA/input0/trial1/energy_per_iteration_10000_COBYLA_2_1.csv",
                     "outputs/spiq/energy/Week88/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_10000/reps_2/COBYLA/input0/trial1/energy_per_iteration_10000_COBYLA_2_1.csv"],
    ("P3", "spiq"): ["outputs/spiq/energy/Week60/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_10000/reps_2/COBYLA/input0/trial1/energy_per_iteration_10000_COBYLA_2_1.csv"],
}

def summarize(files):
    firsts, bests, evals = [], [], []
    for f in files:
        e = [float(r["energy"]) for r in csv.DictReader(gzip.open(f, "rt") if f.endswith(".gz") else open(f))]
        firsts.append(e[0]); bests.append(min(e)); evals.append(len(e))
    return firsts, bests, evals

def fmt(v):
    return " / ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v)

rows = ["| Problem · init | Source | Runs | First logged energy | Best logged energy | Evaluations |", "|---|---|---|---|---|---|"]
for (P, I), fs in THEIRS.items():
    f1, b1, n1 = summarize([os.path.join(ROOT, f) for f in fs])
    mine = sorted(glob.glob(os.path.join(HERE, "results", P, I, "COBYLA", "trial*", "energy_per_iteration_*.csv*")))
    f2, b2, n2 = summarize(mine)
    rows.append(f"| {P} · {I} | team's committed log | {len(fs)} | {fmt(f1)} | {fmt(b1)} | {fmt(n1)} |")
    budget_note = " (budget 1,500)" if (P, I) == ("P3", "spiq") else ""
    if mine:
        rows.append(f"| | this study's harness | {len(mine)} | median {np.median(f2):.3f} | median {np.median(b2):.3f} "
                    f"[{min(b2):.3f}–{max(b2):.3f}] | median {int(np.median(n2))}{budget_note} |")
out = "\n".join(rows)
open(os.path.join(HERE, "analysis", "repro_table.md"), "w").write(out + "\n")
print(out)
