"""Aggregate every run into tidy tables (run in the QAOA environment).

Outputs (analysis/):
  runs.csv            one row per run: evaluations, wall time, energies, plan-quality metrics
  trajectories.csv    per run & evaluation: logged energy, best-so-far, exact energy of the iterate
                      (exact values subsampled to <=60 points per run)
  cost_dist.csv       per run: probability of each plan cost at the returned state (exact), raw/fallback
  starts.csv          per problem & initialization: exact metrics at the starting point

Metric definitions
  E_logged_best  min of the driver's logged (10240-shot) energies. Biased low by shot noise, and
                 more so the more evaluations a method makes; reported for reference only.
  E_exact        exact <H_C> (QUBO scale) of the parameters the pipeline decodes into plans:
                 random init -> the optimizer's returned point (VQE optimal_point, as pickled);
                 SPIQ        -> the best-seen parameters (driver's min_params, as re-sampled).
  gap_closed     (E_exact(start) - E_exact) / (E_exact(start) - E_ground): 1 = reached ground state.
  opt_ratio_*    fraction of the pipeline's 10240 re-sampled shots whose decoded plan has optimal
                 cost, from readout_summary.csv (the paper's Table-1 / R-notebook metric).
  P_opt_*_exact  the same probability computed exactly from the state (no sampling noise).
"""
import csv, glob, gzip, json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "base"))
sys.path[:0] = [HERE, BASE]
os.chdir(BASE)
from ground_truth import load
from decode_table import build
from fastsim import qaoa_probs, compile_circuit, circuit_probs
from spiq_artifacts import load_spiq
from qiskit.quantum_info import Statevector

RES, OUT = os.path.join(HERE, "results"), os.path.join(HERE, "analysis")
os.makedirs(OUT, exist_ok=True)
GT = json.load(open(os.path.join(HERE, "ground_truth.json")))
IDX = {"P1": 0, "P2": 1, "P3": 2}

prob_cache, spiq_cache = {}, {}
def problem(P):
    if P not in prob_cache:
        card, pred, pred_sel, qubo, _ = load(IDX[P])
        d = build(P)
        op, off = qubo.to_ising()
        prob_cache[P] = dict(d=d, diag=d["energy"] - float(off), opt=float(d["optimal_cost"]))
    return prob_cache[P]

def spiq(P):
    if P not in spiq_cache:
        pcirc, x0, meta = load_spiq(P)
        spiq_cache[P] = (compile_circuit(pcirc), x0)
    return spiq_cache[P]

def probs(P, init, theta):
    if init == "random":
        return qaoa_probs(problem(P)["diag"], np.asarray(theta, float), 2)
    comp, _ = spiq(P)                      # fast exact simulator, verified == qiskit Statevector
    return circuit_probs(comp, np.asarray(theta, float))

def exact_metrics(P, p):
    d, opt = problem(P)["d"], problem(P)["opt"]
    gs = GT[P]["ground_state_energy"]
    return dict(E_exact=float(p @ d["energy"]),
                P_opt_raw_exact=float(p @ (np.round(d["raw_cost"]) == opt)),
                P_opt_fb_exact=float(p @ (np.round(d["fb_cost"]) == opt)),
                mean_cost_raw_exact=float(p @ d["raw_cost"]), mean_cost_fb_exact=float(p @ d["fb_cost"]),
                P_ground_state=float(p[np.isclose(d["energy"], gs)].sum()))

def parse_params(s):
    return [float(v) for v in s.replace("[", " ").replace("]", " ").replace(",", " ").split()]

def readout_ratios(run_dir, opt):
    import gzip
    f = glob.glob(os.path.join(run_dir, "iterations_*", "reps_*", "*", "trial_*", "readout_summary.csv*"))[0]
    text = gzip.open(f, "rt").read() if f.endswith(".gz") else open(f).read()   # published copies are gzipped
    rows = list(csv.DictReader(text.split("# all_solutions")[1].strip().splitlines()))
    out = {}
    for flag, key in (("False", "raw"), ("True", "fb")):
        cs = [float(r["cost"]) for r in rows if r["used_fallback"] == flag and float(r["probability"]) > 0]
        out[f"opt_ratio_{key}"] = float(np.mean(np.isclose(cs, opt))) if cs else np.nan
        out[f"best_cost_{key}"] = float(min(cs)) if cs else np.nan
    return out

# ---- starting points -----------------------------------------------------------------
starts, start_dists = {}, []
for P in ("P1", "P2", "P3"):
    for I, th in (("random", [0.5] * 4), ("spiq", spiq(P)[1])):
        p0 = probs(P, I, th)
        starts[(P, I)] = exact_metrics(P, p0)
        d0 = problem(P)["d"]
        for key, vec in (("raw", d0["raw_cost"]), ("fallback", d0["fb_cost"])):
            for c in sorted(set(np.round(vec))):
                start_dists.append(dict(problem=P, init=I, trial=0, decoder=key, cost=c,
                                        prob=float(p0[np.isclose(np.round(vec), c)].sum())))
with open(os.path.join(OUT, "starts_cost_dist.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(start_dists[0])); w.writeheader(); w.writerows(start_dists)
with open(os.path.join(OUT, "starts.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["problem", "init"] + list(next(iter(starts.values()))))
    for (P, I), m in starts.items():
        w.writerow([P, I] + list(m.values()))

# ---- runs ------------------------------------------------------------------------------
runs, trajs, dists = [], [], []
for meta_path in sorted(glob.glob(os.path.join(RES, "*", "*", "*", "trial*", "run_meta.json"))):
    rd = os.path.dirname(meta_path)
    m = json.load(open(meta_path))
    P, I, O, T = m["problem"], m["init"], m["optimizer"], m["trial"]
    opt, gs = problem(P)["opt"], GT[P]["ground_state_energy"]
    import gzip
    lf = glob.glob(os.path.join(rd, "energy_per_iteration_*.csv*"))[0]      # .csv, or .csv.gz in the published copy
    log = list(csv.DictReader(gzip.open(lf, "rt") if lf.endswith(".gz") else open(lf)))
    E = np.array([float(r["energy"]) for r in log])
    if I == "random":
        pk = os.path.join(rd, "results.txt")
        resp = pickle.load(open(pk, "rb") if os.path.exists(pk) else gzip.open(pk + ".gz", "rb"))  # published: gzipped
        theta = list(resp.min_eigen_solver_result.optimal_point)
    else:
        theta = parse_params(log[int(np.argmin(E))]["paramter"])
    p = probs(P, I, theta)
    ex = exact_metrics(P, p)
    E0 = starts[(P, I)]["E_exact"]
    row = dict(problem=P, init=I, optimizer=O, trial=T, evaluations=m["evaluations"],
               budget=m["budget"], budget_hit=m["budget_hit"], wall_time_s=round(m["wall_time_s"], 2),
               E_start_exact=E0, E_logged_first=float(E[0]), E_logged_best=float(E.min()),
               eval_of_logged_best=int(np.argmin(E)) + 1, ground_state=gs, **ex,
               gap_closed=(E0 - ex["E_exact"]) / (E0 - gs), **readout_ratios(rd, opt))
    runs.append(row)
    # cost distribution at the returned state (exact)
    d = problem(P)["d"]
    for key, vec in (("raw", d["raw_cost"]), ("fallback", d["fb_cost"])):
        for c in sorted(set(np.round(vec))):
            dists.append(dict(problem=P, init=I, optimizer=O, trial=T, decoder=key, cost=c,
                              prob=float(p[np.isclose(np.round(vec), c)].sum())))
    # trajectory: logged energies, and the EXACT energy of the incumbent (the point the optimizer
    # would hand back if stopped now = best point by its own noisy estimate, which is what the
    # pipeline decodes). Exact energies are memoised per distinct incumbent.
    best = np.minimum.accumulate(E)
    inc_idx = np.array([int(np.argmin(E[:k + 1])) for k in range(len(E))]) if len(E) < 20000 else None
    memo = {}
    for k, r in enumerate(log):
        j = int(inc_idx[k])
        if j not in memo:
            memo[j] = float(probs(P, I, parse_params(log[j]["paramter"])) @ problem(P)["d"]["energy"])
        trajs.append(dict(problem=P, init=I, optimizer=O, trial=T, eval=k + 1, E_logged=float(E[k]),
                          E_logged_best_so_far=float(best[k]), E_exact_incumbent=memo[j]))
    print(f"{P} {I:6} {O:11} t{T}: evals={m['evaluations']:5} E_exact={ex['E_exact']:.3f} "
          f"gap_closed={row['gap_closed']:.3f} opt_ratio raw/fb={row['opt_ratio_raw']:.3f}/{row['opt_ratio_fb']:.3f}", flush=True)

for name, rows in (("runs", runs), ("trajectories", trajs), ("cost_dist", dists)):
    if rows:
        with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote {len(runs)} runs")
