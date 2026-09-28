"""Re-check every headline finding from the published data. Prints evidence + PASS/FAIL.

Run from optimizer_study/ with the QAOA environment:   python verify_findings.py
Sources: analysis/runs.csv, analysis/starts.csv (per-run results), analysis/spiq_gradient.json
(measure_spiq_gradient.py), the team's committed energy logs (outputs/), and - for finding 4 -
ONLY the team's own code (QUBOGenerator, qubo.objective.evaluate, Postprocessing).
"""
import contextlib, csv, io, itertools, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.join(HERE, "..", "base")
R = pd.read_csv(os.path.join(HERE, "analysis", "runs.csv"))
S = pd.read_csv(os.path.join(HERE, "analysis", "starts.csv"))
GT = json.load(open(os.path.join(HERE, "ground_truth.json")))
results = []
def check(label, ok):
    results.append(ok); print(f"   [{'PASS' if ok else 'FAIL'}] {label}")
def cell(P, I, O): return R[(R.problem == P) & (R.init == I) & (R.optimizer == O)]
def start(P, I, col): return float(S[(S.problem == P) & (S.init == I)][col].iloc[0])

print("=" * 100); print("FINDING 1 - from the random start, several optimizers beat COBYLA"); print("=" * 100)
for P, others in (("P1", ["POWELL"]), ("P2", ["ADAM", "NFT"])):
    c = cell(P, "random", "COBYLA")
    print(f" {P}: start {start(P,'random','E_exact'):.3f}, ground state {GT[P]['ground_state_energy']:.3f}")
    print(f"   COBYLA  gap closed per trial {sorted(c.gap_closed.round(3))}  median {c.gap_closed.median():.2f}")
    for O in others:
        o = cell(P, "random", O)
        print(f"   {O:7} gap closed per trial {sorted(o.gap_closed.round(3))}  median {o.gap_closed.median():.2f}")
        check(f"{P}: {O} median gap closed ({o.gap_closed.median():.2f}) > COBYLA ({c.gap_closed.median():.2f})",
              o.gap_closed.median() > c.gap_closed.median())
        if P == "P1":
            check(f"P1: EVERY Powell trial beats EVERY COBYLA trial (worst Powell {o.E_exact.max():.3f} < best COBYLA {c.E_exact.min():.3f})",
                  o.E_exact.max() < c.E_exact.min())

print(); print("=" * 100); print("FINDING 2 - from the SPIQ start, COBYLA and most others do not improve anything"); print("=" * 100)
print(" (a) What the logs show vs. what is true (COBYLA, SPIQ start, median over trials):")
for P in ("P1", "P2", "P3"):
    c = cell(P, "spiq", "COBYLA")
    logged = (c.E_logged_best - c.E_logged_first).median(); exact = (c.E_exact - c.E_start_exact).median()
    print(f"   {P}: logged energy change {logged:+.3f}   |   EXACT energy change of what COBYLA returns {exact:+.4f}")
    check(f"{P}: logs show a decrease, but the exact energy did not improve by more than 0.01", logged < 0 and exact > -0.01)
print(" (b) The team's own committed logs show the same small logged decrease:")
for P, f in (("P2", "outputs/spiq/energy/Week84/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_10000/reps_2/COBYLA/input0/trial1/energy_per_iteration_10000_COBYLA_2_1.csv"),
             ("P3", "outputs/spiq/energy/Week60/ExperimentalAnalysis/IBMQ/QPUPerformance/Results/CPU_Data/iterations_10000/reps_2/COBYLA/input0/trial1/energy_per_iteration_10000_COBYLA_2_1.csv")):
    e = [float(r["energy"]) for r in csv.DictReader(open(os.path.join(HERE, "..", f)))]
    print(f"   team {P}: first logged {e[0]:.3f} -> best logged {min(e):.3f} (change {min(e)-e[0]:+.3f}) over {len(e)} evaluations")
print(" (c) Why: exact gradient and single-angle curvature at the team's SPIQ point (measure_spiq_gradient.py):")
G = json.load(open(os.path.join(HERE, "analysis", "spiq_gradient.json")))
for P in ("P1", "P2", "P3"):
    if P not in G: print(f"   {P}: (not measured yet)"); continue
    g = G[P]
    print(f"   {P}: {g['angles']} angles | |grad| = {g['grad_norm']:.1e} (vs {g['grad_norm_nearby_random_point']:.2f} at a nearby random point) | "
          f"single-angle directions: {g['n_uphill']} uphill, {g['n_flat']} flat, {g['n_downhill']} downhill")
    check(f"{P}: gradient is zero (< 1e-10) and NO single angle goes downhill", g["grad_norm"] < 1e-10 and g["n_downhill"] == 0)
print(" (d) How many optimizers improve on the SPIQ start (gap closed, median):")
g = R[R.init == "spiq"].groupby(["optimizer", "problem"]).gap_closed.median().unstack()
print(g.round(3).to_string().replace("\n", "\n   "))
stuck = sorted(g.index[g.max(axis=1) <= 0.01])
print(f"   optimizers closing <= 1% of the gap on EVERY problem: {stuck}")
print(f"   BOBYQA's best: {g.loc['BOBYQA'].max():.4f} of the gap (P3)")
check(f"{len(stuck)} of 10 optimizers (incl. COBYLA) close <= 1% on every problem", "COBYLA" in stuck and len(stuck) == 6)

print(); print("=" * 100); print("FINDING 3 - SPSA (P1) and NFT (P3) escape the SPIQ point"); print("=" * 100)
s = cell("P1", "spiq", "SPSA")
print(f" P1 SPSA: exact energy per trial {sorted(s.E_exact.round(3))} (start {start('P1','spiq','E_exact'):.3f}, ground {GT['P1']['ground_state_energy']:.3f})")
print(f"          optimal-plan ratio (team's metric, fallback) per trial {sorted(s.opt_ratio_fb.round(3))} (start {start('P1','spiq','P_opt_fb_exact'):.3f})")
check("P1: every SPSA trial ends below the SPIQ start", (s.E_exact < start("P1", "spiq", "E_exact") - 0.1).all())
check("P1: 2 of 3 SPSA trials raise the optimal-plan ratio from 0.50 to >= 0.97", int((s.opt_ratio_fb >= 0.97).sum()) == 2)
n = cell("P3", "spiq", "NFT")
print(f" P3 NFT:  exact energy per trial {sorted(n.E_exact.round(3))} (start {start('P3','spiq','E_exact'):.3f}, ground {GT['P3']['ground_state_energy']:.3f})")
check("P3: every NFT trial ends >3 energy units below the SPIQ start", (n.E_exact < start("P3", "spiq", "E_exact") - 3).all())
for P in ("P2", "P3"):
    x = cell(P, "spiq", "SPSA")
    print(f" caveat - {P} SPSA: exact energy {sorted(x.E_exact.round(3))} vs start {start(P,'spiq','E_exact'):.3f} (worse)")

print(); print("=" * 100); print("FINDING 4 - on P2 the QUBO's ground state is NOT the optimal join order (team's code only)"); print("=" * 100)
sys.path.insert(0, BASE); cwd = os.getcwd(); os.chdir(BASE)
import Scripts.ProblemGenerator as PG, Scripts.QUBOGenerator as QG, Scripts.Postprocessing as PP
card, pred, pred_sel = PG.get_join_ordering_problem("ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/1_predicates")
with contextlib.redirect_stdout(io.StringIO()):
    qubo, _ = QG.generate_IBMQ_QUBO_for_left_deep_trees_v2(card, pred, pred_sel)
os.chdir(cwd)
n_var = qubo.get_num_vars()
best = min(((qubo.objective.evaluate(list(x)), x) for x in itertools.product((0, 1), repeat=n_var)), key=lambda t: t[0])
R_ = len(card); w = np.arange(1, R_ - 1)[R_ - 3::-1]
cv = np.array(np.array_split(np.array(best[1][:R_ * (R_ - 2)]), R_)).dot(w)
raw = PP.get_raw_join_order(cv); fb = PP.postprocess_join_order(raw, cv, R_, pred)
cost = lambda o: PP.get_costs_for_leftdeep_tree(list(o), card, pred, pred_sel, {})
plans = {o: cost(o) for o in itertools.permutations(range(R_))}
opt = min(plans.values())
print(f" P2: card {card}, predicates {pred}, selectivities {pred_sel}; {n_var} binary variables -> {2**n_var} assignments checked")
print(f"   QUBO ground state: energy {best[0]:.4f}, bitstring {''.join(map(str, best[1]))}")
print(f"   decoded by the team's decoder: raw order {raw} (cost {cost(raw):.0f}), with fallback {fb} (cost {cost(fb):.0f})")
print(f"   true optimal join order(s) by brute force over all {len(plans)} orders: {[list(o) for o,c in plans.items() if c==opt]} (cost {opt:.0f})")
check(f"P2 ground state decodes to cost {cost(fb):.0f} > optimal {opt:.0f}", cost(raw) > opt and cost(fb) > opt)
print(f"   SPIQ's P2 start: probability on the ground state {start('P2','spiq','P_ground_state'):.2f}, "
      f"probability of an optimal plan {start('P2','spiq','P_opt_fb_exact'):.3f}")

print(); print("=" * 100)
print(f"SUMMARY: {sum(results)} of {len(results)} checks PASS" + ("" if all(results) else "  <-- SOME CHECKS FAILED"))
