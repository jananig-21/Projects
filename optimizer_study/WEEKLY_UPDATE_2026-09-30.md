# Weekly Update on 9/30: Optimizer study for SPIQ vs Random initialization

**Scope.** I changed only the **classical optimizer**. Everything else is the team's pipeline, unmodified:
the same join-ordering QUBO, QAOA depth p=2, the relaxed SPIQ ansatz, the team's committed SPIQ initial points,
noiseless QasmSimulator with 10,240 shots, and the team's decoder with the Schönberger fallback.
I tested 10 optimizers on 3 problem instances:

| Instance | Relations / predicates | Qubits | Ground energy | Optimal plan cost |
|---|---|---|---|---|
| P1 | 3 / 2 | 5 | 1.031 | 15 |
| P2 | 4 / 2 | 12 | 4.766 | 315 |
| P3 | 4 / 6 | 20 | 2.725 | 39 |

Optimizers: COBYLA (the team's), AQGD (the original paper's), SPSA, Nelder–Mead, Powell, BOBYQA, L-BFGS-B, SLSQP, Adam, NFT.

**Seeded trials per (optimizer, instance).**
- Random start: 5 trials on P1 and P2, 1–2 trials on P3 so far (3 planned).
- SPIQ start: 3 trials on P1 and P2, 2 trials on P3.

**Budgets.** Random: 1,000 evaluations. SPIQ: 2,500 evaluations on P1/P2 and 1,500 on P3.

**Status.** 195 runs are published and all 15 automated proof checks pass. 15 more P3 runs are in progress.

> Note: this is a separate study from the 22-instance / 400-evaluation GPU benchmark. It uses the 3 instances committed in the `spiq-joo` branch.

**Proof, for every number below:**
- all tables: [`analysis/tables.md`](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/analysis/tables.md)
- every run, one row each: [`analysis/runs.csv`](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/analysis/runs.csv)
- claims with checks: [`EVIDENCE.md`](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/EVIDENCE.md)
- full report: [`README.md`](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/README.md)
- explained from scratch: [`BRIEFING.md`](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/BRIEFING.md)

---

## 1. From the Random start, the choice of optimizer matters. COBYLA is not the best.

1. **Only the optimizer differs.** Same instances, circuit, p=2, simulator, shots, decoder and seeds scheme.
2. **Metric: gap closed.** Gap closed = (E_start − E_final) / (E_start − E_ground), using the exact noise-free energy of the returned parameters. 1.00 means the ground state was reached. I use the exact energy because the "best logged energy" is biased downward by shot noise.
3. **Results (median gap closed):**

| Optimizer | P1 | P2 | P3 | Median evals |
|---|---|---|---|---|
| **Powell** | **0.71** | 0.29 | **0.50** | 334 |
| COBYLA (team) | 0.40 | 0.25 | 0.48 | 116 |
| **NFT** | 0.55 | 0.36 | 0.20 | 1000 |
| SPSA | 0.51 | 0.30 | 0.20 | 1000 |
| SLSQP | 0.56 | 0.22 | 0.20 | 76 |
| **Adam** | 0.35 | **0.40** | 0.20 | 1000 |
| BOBYQA | 0.40 | 0.31 | 0.24 | 1000 |
| AQGD | 0.39 | 0.22 | 0.20 | 999 |
| L-BFGS-B | 0.32 | 0.22 | 0.18 | 60 |
| Nelder–Mead | 0.37 | 0.11 | 0.17 | 1000 |

4. **P1: Powell beats COBYLA in every trial.** Powell closes 0.71 of the gap vs COBYLA's 0.40. Powell's worst trial (energy 2.406) beats COBYLA's best trial (2.690).
5. **P2: Adam and NFT beat COBYLA.** Adam 0.40 and NFT 0.36 vs COBYLA 0.25.
6. **P3: Powell and COBYLA lead.** Powell 0.50 and COBYLA 0.48. Every other optimizer is at about 0.20. P3 is based on 1–2 trials so far.
7. **Cost.** COBYLA stops early (~116 evaluations), so it is the cheapest per unit of progress. Powell uses ~3× more evaluations.

![Final exact energy per optimizer](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/final_energy-light.png)
*Figure 1: Final exact energy per optimizer, instance and start. [figures/final_energy-light.png](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/figures/final_energy-light.png)*

![Convergence P1 random](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/convergence_P1_random-light.png)
![Convergence P2 random](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/convergence_P2_random-light.png)
![Convergence P3 random](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/convergence_P3_random-light.png)
*Figure 2: Exact-energy convergence from the Random start (P1, P2, P3). Files: `figures/convergence_P*_random-light.png`*

**Proof:**
- [`EVIDENCE.md` Finding 1](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/EVIDENCE.md), with per-trial values
- `verify_findings.py` checks (PASS)

---

## 2. SPIQ starts much closer to the ground energy, closer than any optimizer gets from Random

| Instance | Ground energy | Random start | SPIQ start | Best result from Random (any optimizer) |
|---|---|---|---|---|
| P1 | 1.031 | 5.091 | **1.561** | 2.215 (Powell) |
| P2 | 4.766 | 24.842 | **8.338** | 16.815 (Adam) |
| P3 | 2.725 | 30.692 | **9.867** | 16.577 (Powell) |

1. **SPIQ starts far closer to the ground state.** The remaining energy gap at the start is:
   - P1: 0.53 (SPIQ) vs 4.06 (Random)
   - P2: 3.57 vs 20.08
   - P3: 7.14 vs 27.97
2. **SPIQ's untouched start beats the best of all 10 optimizers run from Random, on every instance.** The SPIQ start is the "SPIQ start" column above; the Random results are the last column, at up to 1,000 evaluations.

**Proof:** the `*start*` rows in [`analysis/tables.md`](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/analysis/tables.md). The starting energies are exact statevector values, computed by `fastsim.py` and verified against Qiskit to 1e-10.

---

## 3. After the SPIQ start, most optimizers do not move at all

1. **6 of 10 optimizers stay at the SPIQ start on every instance.** COBYLA (the team's choice), Powell, Nelder–Mead, SLSQP, L-BFGS-B and AQGD all close ≤1% of the remaining gap on P1, P2 and P3; several end slightly worse. BOBYQA moves 1% on P3.
2. **The team's logged energy decrease after SPIQ is shot noise.**
   - The logs show 8.33→8.27 on P2.
   - The exact energy change of COBYLA's returned point on P2 is **+0.0015**, i.e. no improvement.
   - I reproduced the team's log: 8.34→8.27 over 2,247 evaluations.
3. **Why the optimizers stall: the SPIQ point sits in a flat, upward-curving spot.** I measured the gradient at the SPIQ initial point ([`analysis/spiq_gradient.json`](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/analysis/spiq_gradient.json)).
   - The gradient norm is ≈ 0: 3e-15 (P1), 1.4e-14 (P2), 2e-14 (P3).
   - For comparison, nearby random points have gradient norms of 1.7–4.7.
   - Along each individual angle, the energy curves upward or is flat. No single angle goes downhill:
     - P1: 40 up / 0 flat
     - P2: 93 up / 23 flat
     - P3: 246 up / 22 flat
   - Gradient-based optimizers (L-BFGS-B, SLSQP, AQGD) and optimizers that search one angle at a time (Powell, COBYLA, Nelder–Mead) therefore see nothing to follow.
4. **Only optimizers with random or joint multi-angle moves escape.**

| Optimizer | Instance | Energy | Optimal-plan probability |
|---|---|---|---|
| **SPSA** | P1 | 1.561 → 1.08–1.20 | 0.50 → **0.98** in 2 of 3 trials |
| **Adam** | P1 | 1.561 → 1.38 | 0.50 → 0.76 |
| **NFT** | P3 | 9.867 → **6.38** (0.49 of the gap closed) | 0.75 → 0.63 |
| SPSA | P2 | +3.0 (gets worse) | |
| SPSA | P3 | +8.1 (gets worse) | |

![Convergence P1 SPIQ](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/convergence_P1_spiq-light.png)
![Convergence P2 SPIQ](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/convergence_P2_spiq-light.png)
![Convergence P3 SPIQ](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/convergence_P3_spiq-light.png)
*Figure 3: Exact-energy convergence from the SPIQ start. Most curves are flat lines at the start energy. Files: `figures/convergence_P*_spiq-light.png`*

**Proof:**
- [`EVIDENCE.md` Findings 2–3](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/EVIDENCE.md)
- `measure_spiq_gradient.py`

---

## 4. Join-order quality: lower energy does not always mean better plans

1. **Same plan rule for both starts.** Both use the team's rule: decode each of the 10,240 shots, apply the Schönberger fallback, and report the fraction that gives an optimal-cost join order.
2. **Median optimal-plan probability:**

| Instance | Random start | Best from Random | SPIQ start | Best from SPIQ |
|---|---|---|---|---|
| P1 | 0.197 | 0.495 (Powell) | 0.500 | **0.979 (SPSA)** |
| P2 | 0.060 | **0.126 (Adam)** | **0.000** | 0.090 (SPSA) |
| P3 | 0.160 | 0.159 (no optimizer improves it) | **0.750** | 0.754 (optimizers barely move it) |

3. **SPIQ wins clearly on plan quality on P1 and P3.** On P3, SPIQ's start gives 0.75 vs about 0.16 for every Random-start run.
4. **P2 is a penalty-scaling problem, not an optimizer problem.**
   - The lowest-energy QUBO assignment (bitstring `111100001100`) decodes to join order [1,0,3,2], which costs **465**. The optimal cost is 315.
   - SPIQ's P2 start puts 25% of its probability on this ground state, yet has a 0% optimal-plan probability.
   - Minimizing energy further on P2 cannot fix this. The QUBO's `penalty_scaling` needs retuning.
5. **Random starts: energy drops, plan quality does not.** On P3, optimizers lower the energy by up to 50% of the gap, but no optimizer raises plan quality above the starting 0.16.

![Optimal-plan ratio](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/optimal_ratio_fallback-light.png)
*Figure 4: Optimal-plan probability (with fallback) per optimizer. [figures/optimal_ratio_fallback-light.png](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/figures/optimal_ratio_fallback-light.png)*

![Plan cost distribution, SPIQ](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/plan_costs_spiq-light.png)
![Plan cost distribution, Random](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/plan_costs_random-light.png)
*Figure 5: Distribution of decoded plan costs (the team's Table 1 format). Files: `figures/plan_costs_*-light.png`*

**Proof:**
- [`EVIDENCE.md` Finding 4](https://github.com/jananig-21/Projects/blob/quantumdb-optimizer-study/optimizer_study/EVIDENCE.md), showing the ground-state decode (PASS: cost 465 > optimal 315) and the 25% / 0.000 check
- per-run `readout_summary.csv` files under `results/`

---

## 5. Evaluation cost

![Evaluations used](https://raw.githubusercontent.com/jananig-21/Projects/quantumdb-optimizer-study/optimizer_study/figures/evaluations-light.png)
*Figure 6: Objective evaluations used per optimizer. Each evaluation = 10,240 shots.*

- **COBYLA, SLSQP and L-BFGS-B stop by themselves** after 60–120 evaluations from the Random start.
- **SPSA, Adam, NFT and AQGD always run to the full budget.**
- **For a QPU, cost scales with evaluations × shots.** COBYLA at ~116 × 10,240 is about 1.2M shots per run; a 1,000-evaluation optimizer is about 10M.

---

## 6. Recommendations and open questions

1. **Pair SPIQ with an optimizer that can leave a flat point.** The candidates are NFT (best on P3) or SPSA/Adam (best on P1). COBYLA effectively does nothing after SPIQ, so the reported "SPIQ + COBYLA" result is really "SPIQ alone".
2. **Retune `penalty_scaling` for P2-type instances,** so that the QUBO ground state is the optimal join order.
3. **Open question: evaluation budget.** Given that COBYLA stalls at the SPIQ point, should the budget comparison use an optimizer that actually moves after SPIQ? And what budget makes sense for a QPU (evaluations × shots × circuit depth × cost)?
4. **Open question: scale-up.** Should I run the top 3 optimizers (Powell, NFT, SPSA) on the 22-instance benchmark at 400 evaluations?

---

## Where everything is
- **Branch:** [`quantumdb-optimizer-study`](https://github.com/jananig-21/Projects/tree/quantumdb-optimizer-study/optimizer_study)
- **Code:** `run_study.py` (calls the team's unmodified `IBMQExperiments.py`), `optimizers.py` (the 10 optimizer settings), `spiq_artifacts.py` (the team's SPIQ points)
- **Results:** `results/<problem>/<init>/<optimizer>/trial<k>/`, each holding `run_meta.json`, the energy log, `readout_summary.csv` and `min_state_readout.csv`
- **Reproduction check:** `repro_check.py`, comparing mine vs the team's logs:
  - P2 Random COBYLA: 24.88→19.62 (mine) vs 24.84→19.70 (team)
  - P2 SPIQ: 8.34→8.27 vs 8.33→8.27
  - P3 SPIQ: 9.85→9.78 vs 9.84→9.79
- **Automated verification:** `python verify_findings.py` gives 15/15 PASS, with the output embedded in `EVIDENCE.md`
