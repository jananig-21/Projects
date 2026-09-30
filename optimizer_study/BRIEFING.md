# Briefing: the optimizer study, explained from zero

*For presenting this work. Everything the study did, where it lives, how it was done, what every
result means, and what to report. Numbers are from the analysis snapshot on GitHub. The P3
random-start arm is still completing, and its trial counts are stated wherever they appear.*

---

## 0. What to report (the one-minute version)

> "I benchmarked **10 classical optimizers** inside our team's QAOA join-ordering pipeline,
> changing *only* the optimizer, and reproduced our team's own COBYLA numbers first.
> Across **210 runs** (3 problems × 2 starting strategies × 10 optimizers × repeated trials):
>
> 1. **From the random start, the optimizer matters.** On P1, Powell closes 71 % of the gap to the
>    ground state vs COBYLA's 40 %, and every Powell trial beats every COBYLA trial. On P2, Adam (40 %) and NFT (36 %) beat
>    COBYLA (25 %). On P3, COBYLA (48 %) and Powell (50 %) are the only two that get far.
> 2. **From our SPIQ start, COBYLA and most optimizers do not improve anything.** The small drop
>    in our logs is shot noise. The exact energy of what COBYLA returns is unchanged. The cause is
>    measured: at the SPIQ point the gradient is exactly zero and no single angle goes downhill.
> 3. **SPSA (on P1) and NFT (on P3) do escape the SPIQ point.** NFT gives the biggest real gain
>    (P3 energy 9.87 → 6.38). SPSA is unsafe on bigger problems.
> 4. **Lower energy does not guarantee better join orders with our QUBO.** On P2 the exact ground
>    state decodes to a sub-optimal plan (cost 465 vs optimal 315). On P3 from the random start, no
>    optimizer raises the optimal-plan ratio even when the energy drops a lot.
>
> **Suggested next steps:** pair SPIQ with NFT instead of COBYLA, and tune the QUBO penalty weight
> so its lowest energy is the best join order."

Proof for 1–4: [`EVIDENCE.md`](EVIDENCE.md). One script, `verify_findings.py`, re-checks them all
(15 of 15 checks pass).

---

## 1. The team's project, from zero

### 1.1 The problem: join ordering
A database answering a query often has to **join** several tables. The **order** of the joins
changes how big the intermediate results get. A good order might produce 315 intermediate rows,
a bad one 1,350. Choosing the order is a hard combinatorial problem.

### 1.2 The team's approach (paper: *Improving Join Order Optimization on Gate-Based Quantum Computers via Structured Parameter Initialization*, Shekar, Wu, Bharadwaj, Ravi, Ma, VLDB 2026 workshop QCDKM)

| Step | What happens | Plain meaning |
|---|---|---|
| **1. QUBO** | The join-ordering problem is written as a *Quadratic Unconstrained Binary Optimization*: binary variables v[t,j] ("table t is in join j") and p[k,j] ("predicate k applies at join j"), plus penalty terms for invalid orders and a cost term (squared log of intermediate sizes). | Every possible answer gets an **energy**. Low energy = good, valid join order. The lowest possible energy is the **ground state**. |
| **2. QAOA** | A quantum circuit (Quantum Approximate Optimization Algorithm), depth p = 2, whose **angles** (β, γ) control which answers it favours. | A machine with dials. The dial settings decide what it outputs. |
| **3. Classical optimizer** | A classical routine repeatedly measures the circuit's average energy (10,240 shots per measurement) and adjusts the angles to lower it. The team used **COBYLA**. | The thing that turns the dials downhill. **This is what this study varies.** |
| **4. Decode** | Each measured bitstring is turned back into a join order (a score per table, sorted), optionally repaired by the **fallback** of Schönberger et al. (prefers joining connected tables, avoiding cross products), then costed classically. | Read out the answer and check how good it is. |
| **SPIQ** (the team's contribution) | Before QAOA, a classical genetic algorithm searches *Clifford* angles (multiples of π/2, which a classical computer can simulate quickly with `stim`) for a good starting point. It uses a "relaxed" circuit with one angle per gate (40 / 116 / 268 angles for P1 / P2 / P3). | A smart place to start turning the dials. The baseline instead starts every dial at 0.5 ("random" / "uninitialized"). |

Everything runs on a **noiseless simulator** (no real quantum hardware), exactly as in the paper.

### 1.3 The three test problems ("workloads")
Stored in `base/ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/<0|1|2>_predicates/`
(`card.txt` = table sizes, `pred.txt` = which tables are linked, `pred_sel.txt` = link selectivities).

| | Tables (sizes) | Links (selectivity) | Qubits | Ground-state energy | Optimal plan cost | Worst plan cost |
|---|---|---|---|---|---|---|
| **P1** | 10, 15, 20 | (0,1) 0.1, (1,2) 0.1 | 5 | 1.031 | 15 | 200 |
| **P2** | 10, 15, 20, 30 | (0,1) 0.1, (2,3) 0.1 | 12 | 4.766 | 315 | 1350 |
| **P3** | 10, 15, 20, 30 | six links, 0.1–0.6 | 20 | 2.725 | 39 | 990 |

Qubits = (tables + links) × (tables − 2). I computed the ground states exactly (every bitstring) and
the optimal costs by brute force over every join order, using the team's own functions.

### 1.4 The team's own results (from their committed logs, `outputs/`)
| | Plain start + COBYLA | SPIQ start + COBYLA |
|---|---|---|
| P1 | 5.09 → 2.75 (Week71) | their logs are for an *older* 1-predicate P1 |
| P2 | 24.84 → 19.70 (Week74/84) | 8.33 → 8.27 (Week84, Week88) |
| P3 | – | 9.84 → 9.79 (Week60) |

Their message: SPIQ starts far lower than the plain start ever reaches.

---

## 2. What was done, in order

1. **Copied the team's repository** (`umich-db/quantumdb`, public, never modified) into your GitHub
   as 5 branches, one per original branch (§4).
2. **Read the whole project** (README, cleanup plan, driver, QUBO generator, decoder, R notebook,
   committed results) to learn the pipeline, the metrics and the results exactly as the team defines them.
3. **Built the team's exact software environment**: Python 3.8, qiskit 0.34.2 (terra 0.19.2, Aer
   0.10.3), qiskit-optimization 0.3.2, docplex. The goal was to run *their* code, not a rewrite.
4. **Computed the ground truth** for P1–P3 (§1.3) with their QUBO generator and cost function. It
   matches the "optimal" targets hard-coded in their R notebook (15 for P1, 315 for P2).
5. **Identified which of their logged runs belong to which problem**, since folders are named by
   week. The first energy of a plain-start run is deterministic per problem, and SPIQ runs are
   identified by their number of angles. This showed that old logs (Week4/23/24/51) predate an
   energy-scale fix, and that the P1 SPIQ logs are for an older P1.
6. **Wrote a harness** (`run_study.py`) that runs their unmodified driver `base/IBMQExperiments.py`
   and swaps only the optimizer (§5).
7. **Reproduced their numbers** with COBYLA before trusting anything (§6.1).
8. **Ran 10 optimizers × 3 problems × 2 starts × several trials = 210 runs.**
9. **Built an analysis** that scores each run *exactly* (noise-free), plus figures, tables and a report.
10. **Found and fixed bugs in my own tooling** along the way, and re-ran every affected run (§8.2).
11. **Measured *why* optimizers stall at the SPIQ point** (exact gradients, §7.3).
12. **Wrote the evidence page and a verification script** (15/15 checks pass).
13. **Made the long runs survive machine restarts** (checkpointing, tested to give byte-identical results).

---

## 3. What was NOT changed
- **No file of the team's code was modified** in any branch. The study lives in one new folder,
  `optimizer_study/`, on one new branch.
- Nothing was pushed to the team's repository (`umich-db/quantumdb`), and it was never written to.
- Nothing was pushed to your `main` branch. No pull requests were opened.

---

## 4. Everything in your GitHub (github.com/jananig-21/Projects)

| Branch | What it is | Created by |
|---|---|---|
| `main` | Your original one-line README. Untouched. | you |
| `claude/github-repo-6-projects-2lvc4r` | Your 6 ML projects from Google Drive, with READMEs, license and CI | earlier session |
| `quantumdb` | Exact copy of the team repo's `main`: the original SIGMOD'23 paper package (23,559 files) | copy |
| `quantumdb-Ruokun`, `quantumdb-divya`, `quantumdb-spiq` | Exact copies of the team's working branches (history of their experiments) | copy |
| `quantumdb-spiq-joo` | Exact copy of the team's **latest, cleaned-up** branch. This is the project | copy |
| **`quantumdb-optimizer-study`** | = `quantumdb-spiq-joo` **+ the new folder `optimizer_study/`**. **This study.** | this work |

### 4.1 Every file in `optimizer_study/` and what it means

**Read these**

| File | What it is |
|---|---|
| `README.md` | The full technical report: method, reproduction check, results, discussion, notes on the team's code, limitations, how to reproduce |
| `EVIDENCE.md` | Proof of the headline findings: exact claims, per-trial numbers, figures, raw-data links, verification output |
| `BRIEFING.md` | This file |
| `figures/*.png` | All charts, each in a `-light` and `-dark` version (§7.5) |
| `analysis/tables.md` | Leaderboards and detailed per-problem tables (the same tables as README §6) |

**Data**

| File | What it contains |
|---|---|
| `analysis/runs.csv` | **One row per run**: problem, start, optimizer, trial, evaluations, wall time, energies, gap closed, optimal-plan ratios, plan costs. The master results table |
| `analysis/starts.csv` | The same metrics at each starting point, before any optimization |
| `analysis/cost_dist.csv`, `starts_cost_dist.csv` | Probability of each plan cost at each run's final state / at the start |
| `analysis/trajectories.csv.gz` | Every energy evaluation of every run, plus the exact energy of the incumbent (the convergence curves) |
| `analysis/repro_table.md` | The reproduction check against the team's logs |
| `analysis/spiq_gradient.json` | Exact gradient and curvature at the team's SPIQ points (finding 2) |
| `analysis/verify_findings_output.txt` | The output of the verification script |
| `ground_truth.json` | Ground states, optimal plans, all plan costs for P1–P3 |
| `results/<problem>/<random or spiq>/<optimizer>/trial<k>/` | **Raw output of each run, in the team's own file formats** (below) |

Inside each `results/.../trial<k>/` folder:

| File | Meaning |
|---|---|
| `energy_per_iteration_*.csv.gz` | The team's energy log: one row per energy evaluation (iteration, energy, angles, std) |
| `min_state_readout.csv` | The re-sampled best state: bitstring, energy, probability |
| `results.txt.gz` | The team's pickled result object |
| `iterations_10000/reps_2/<opt>/trial_<k>/readout_summary.csv.gz` | The team's decoded readout: every shot as a join order with its cost, raw and with fallback. The team's R notebook reads this format |
| `run_meta.json` | Evaluations used, wall time, budget hit or not, seed, optimizer settings |
| `checkpoint.jsonl` | (newer runs) saved optimizer steps, used to resume after a restart |

**Code**

| File | What it does |
|---|---|
| `run_study.py` | Runs **one** experiment through the team's unmodified driver, swapping the optimizer |
| `optimizers.py` | The 10 optimizers and their settings, the evaluation-budget guard, the AQGD batch adapter, checkpointing |
| `spiq_artifacts.py` | Which of the team's saved SPIQ starting points each problem uses |
| `ground_truth.py` | Computes §1.3 |
| `decode_table.py` | For every bitstring: which plan the team's decoder produces and its cost (for exact metrics) |
| `fastsim.py` | Exact simulators for the QAOA and SPIQ circuits (verified against qiskit to 10⁻¹⁰) |
| `measure_spiq_gradient.py` | Exact gradient/curvature at the SPIQ point (finding 2) |
| `analyze.py` | Turns all raw results into `analysis/*.csv` |
| `plots.py`, `summarize.py`, `update_readme.py` | Figures, tables, README data sections |
| `repro_check.py` | The reproduction table |
| `verify_findings.py` | Re-checks every headline finding, prints PASS/FAIL |
| `run_grid_safe.sh`, `jobs_*.txt` | Runs the whole experiment grid; cannot run a cell twice |
| `refresh_and_publish.sh`, `sync_publish.sh` | Rebuild everything and publish to GitHub |
| `requirements-study.txt` | Exact software versions |

---

## 5. How the changes were made (the method, precisely)

**Only the optimizer changes.** The team's `IBMQExperiments.py` already calls three replaceable
functions. The harness replaces exactly those, and runs the team's code for everything else:

| Hook in the team's driver | Team's version | Study's version |
|---|---|---|
| `get_optimizer()` | COBYLA | the optimizer under test, behind a budget guard |
| `get_local_QASM_backend()` | simulator, 10,240 shots, unseeded | same simulator and shots; each execution gets a fresh seed from a per-trial random stream (fresh noise every call, as in the original, but reproducible) |
| `current_optim` | "COBYLA" | the optimizer's name (used in file names) |

Kept identical: problems, QUBO, circuit, depth 2, the plain start [0.5, 0.5, 0.5, 0.5], the team's
own saved SPIQ starting points (checked by their built-in sanity test), shot count, decoding, file formats.

**The 10 optimizers**

| Optimizer | Type | Why included | Settings |
|---|---|---|---|
| COBYLA | derivative-free, trust region | the team's baseline | team's exact settings |
| AQGD | gradient, quantum-native (parameter-shift) | the original 2023 paper's choice | team's exact settings |
| SPSA | stochastic gradient | built for noisy objectives | team's exact settings |
| Nelder–Mead | derivative-free, simplex | classic | defaults |
| Powell | derivative-free, line searches | classic | defaults |
| BOBYQA | derivative-free, quadratic trust region | COBYLA's successor | defaults (bounds x₀ ± π, required) |
| L-BFGS-B | gradient, quasi-Newton | standard | finite-difference step 0.1 |
| SLSQP | gradient, SQP | qiskit's default | finite-difference step 0.1 |
| Adam | gradient, adaptive | deep-learning default | lr 0.05, step 0.1 |
| NFT | quantum-native, one angle at a time | exact for SPIQ's circuit | defaults |

Why the finite-difference step changed: the driver gives no analytic gradient, so these methods
estimate one numerically. Their default step (10⁻⁸) is a million times smaller than the measurement
noise (≈0.03–0.12), which would make the estimated gradient pure noise.

**Budget:** each run may use at most 1,000 energy evaluations (random start) or 2,500 / 1,500 (SPIQ,
P1–P2 / P3). An *evaluation* = one 10,240-shot run of the circuit. Optimizers that converge stop earlier.

**Trials:** random start: 5 per optimizer on P1 and P2, 3 on P3. SPIQ start: 3 on P1 and P2, 2 on P3.
Trials differ in measurement noise, and in internal randomness for stochastic optimizers.

---

## 6. Metrics: how to read every number

| Metric | Definition | How to read it |
|---|---|---|
| **Energy** ⟨H_C⟩ | Average QUBO energy of the circuit's output. | Lower is better. The ground state is the minimum possible. |
| **Exact energy** | The energy of the parameters the optimizer returned, computed *without* measurement noise. | The true quality of the answer. The study's main score. |
| **Gap closed** | (start energy − returned energy) / (start energy − ground state) | 1.00 = reached the ground state; 0 = no progress; negative = worse than the start. Comparable across problems. |
| **Optimal-plan ratio** (P(opt)) | Fraction of the 10,240 re-sampled shots that decode to a join order of optimal cost. **The team's headline metric.** Reported raw and with fallback. | Higher is better. 0.98 = the machine outputs the best order 98 % of the time. |
| **Plan-cost distribution** | Probability of the optimal / 2nd / 3rd-cheapest / more expensive plans. The team's Table 1 "cost ratio". | More mass on the optimal plan is better. |
| **Evaluations, wall time** | The cost of optimizing. | Fewer is cheaper. "Budget hit" = cut off, not converged. |

**Why not "best energy seen"?** Each evaluation is noisy, and the minimum of 1,000 noisy readings is
lower than the minimum of 100 just by chance. That rewards optimizers that simply evaluate more.
The study therefore scores the **exact energy of what is returned**, computed with simulators
verified against qiskit.

### 6.1 Reproduction check (the setup gives the team's numbers)
| | Team's log | This study |
|---|---|---|
| P2, plain start, COBYLA | 24.84 → 19.70 in 119 evals | 24.88 → 19.62 (median of 5), 115 evals |
| P2, SPIQ, COBYLA | 8.33 → 8.27 in 2,331 evals | 8.34 → 8.27 (median of 3), 2,247 evals |
| P3, SPIQ, COBYLA | 9.84 → 9.79 | 9.85 → 9.78 |
| P1, plain start, COBYLA | 2.75 (one run) | 2.64–3.55 across 5 runs (brackets it) |

---

## 7. Results in detail

### 7.1 Random (plain) start: the leaderboard
*Gap closed (median over trials). Higher is better. P3 random-start trial counts in brackets.*

| Optimizer | P1 | P2 | P3 | Evaluations used |
|---|---|---|---|---|
| Powell | **0.71** | 0.29 | **0.50** [2] | ~300 (stops itself) |
| COBYLA (team) | 0.40 | 0.25 | 0.48 [2] | ~120 (stops itself) |
| NFT | 0.55 | 0.36 | 0.20 [1] | 1,000 (full budget) |
| SPSA | 0.51 | 0.30 | 0.20 [1] | 1,000 |
| SLSQP | 0.56 | 0.22 | 0.20 [2] | ~70 |
| Adam | 0.35 | **0.40** | 0.20 [1] | 1,000 |
| BOBYQA | 0.40 | 0.31 | 0.24 [1] | 1,000 |
| AQGD | 0.39 | 0.22 | 0.20 [2] | 1,000 |
| L-BFGS-B | 0.32 | 0.22 | 0.18 [2] | ~60 |
| Nelder–Mead | 0.37 | 0.11 | 0.17 [1] | 1,000 |

**How to read it:** on P1, Powell closes 71 % of the distance to the ground state, and all 5 Powell
trials beat all 5 COBYLA trials. On P2, Adam and NFT lead. On P3, Powell and COBYLA reach ~0.5 while
every other method stalls near 0.2 (energy ≈ 25 vs a start of 30.7). **Nothing reaches the ground state**:
with depth 2, the plain start gets at most ~70 % of the way.

**Optimal-plan ratio (with fallback), random start:** start P1 0.20 / P2 0.06 / P3 0.16. The best
optimizers raise P1 to ~0.50 and P2 to ~0.13, **but on P3 no optimizer raises it** (0.13–0.16).
Powell and COBYLA, which lower P3's energy the most, end at 0.13, slightly *below* the start.

### 7.2 SPIQ start: the leaderboard
| Optimizer | P1 | P2 | P3 |
|---|---|---|---|
| NFT | 0.00 | 0.00 | **0.49** (9.87 → 6.38) |
| Adam | **0.33** | 0.00 | 0.00 |
| BOBYQA | 0.00 | 0.00 | 0.01 |
| AQGD, Powell, Nelder–Mead, SLSQP, L-BFGS-B, **COBYLA** | ≈0.00 | ≈0.00 | ≈0.00 |
| SPSA | **0.75** | −0.84 (worse) | −1.13 (worse) |

**How to read it:** SPIQ starts far lower than any plain-start run ends (P1 1.56 vs the best plain
result 2.22; P2 8.34 vs 16.8; P3 9.87 vs 16.6). But from there, **6 of 10 optimizers, COBYLA included,
change nothing (≤ 1 % of the gap on every problem)**. The team's logged drop (8.33 → 8.27 on P2)
is noise: the exact energy of what COBYLA returns is 8.340 vs 8.338 at the start.

**Optimal-plan ratio, SPIQ start:** starts at P1 0.50 / P2 **0.00** / P3 0.75.
- SPSA on P1 lifts it to 0.98–0.99 in 2 of 3 trials; Adam to 0.76.
- NFT on P3 *lowers* it to 0.63 even though the energy drops a lot.
- On P2 it stays 0 for everyone except SPSA (0.09).

### 7.3 Why optimizers stall at the SPIQ point (measured)
Because each SPIQ angle drives exactly one gate, the energy along each angle is an exact sine wave,
so these numbers are exact (`analysis/spiq_gradient.json`):

| | Angles | Gradient at the SPIQ point | Gradient nearby | Single-angle directions (uphill / flat / downhill) |
|---|---|---|---|---|
| P1 | 40 | 3 × 10⁻¹⁵ (zero) | 1.72 | 40 / 0 / **0** |
| P2 | 116 | 1 × 10⁻¹⁴ (zero) | 4.68 | 93 / 23 / **0** |
| P3 | 268 | 2 × 10⁻¹⁴ (zero) | 3.19 | 246 / 22 / **0** |

So no single angle, moved on its own, goes downhill, which is why step-by-step optimizers stop. SPSA
moves all angles at once and finds a downhill combination on P1. NFT refits each angle's full sine
wave and travels along P3's flat directions until descent opens up.

### 7.4 Why lower energy is not always a better plan
- P2: the exact ground state (energy 4.7664, bitstring `111100001100`) decodes, with the team's own
  decoder, to order [1, 0, 3, 2], cost **465**. The optimal orders [0, 1, 2, 3] and [1, 0, 2, 3] cost **315**.
- SPIQ's P2 start puts 25 % probability on that ground state, and 0 % on optimal plans.
- P2, SPSA from SPIQ: energy goes *up* (8.34 → 11.34) but the optimal-plan ratio goes *up* too (0 → 0.09).
- P3 from the random start: the energy drops from 30.7 to ~16.6, but the optimal-plan ratio does not rise.

Cause: the QUBO approximates plan cost by the squared log of intermediate sizes, with validity
penalties weighted by the number of joins. That weighting lets some cheap-but-wrong assignments win.

### 7.5 The figures (`figures/`, each in light and dark)
| Figure | What it shows | Takeaway |
|---|---|---|
| `final_energy` | Dot per trial: exact energy returned, per optimizer and problem; blue = random start, orange = SPIQ; lines mark the starts and the ground state | Blue spreads widely (the optimizer matters); orange sits on its start line (nothing moves), except SPSA/Adam on P1 and NFT on P3 |
| `optimal_ratio_fallback`, `optimal_ratio_raw` | The team's optimal-plan ratio per run | Little improvement on P2/P3; big jump for SPSA/SPIQ on P1 |
| `plan_costs_random`, `plan_costs_spiq` | Stacked bars: share of optimal / 2nd / 3rd / worse plans at the start and after each optimizer (the team's Table 1) | Where probability goes, not just energy |
| `convergence_<P>_<start>` | One panel per optimizer: faint = every logged energy, bold = exact energy of the incumbent (what it would return if stopped now), grey line = ground state (the team's Figs. 3–5) | How fast and how far each optimizer gets; flat bold lines = stuck |
| `evaluations` | Evaluations used per optimizer (log scale) | Cost: COBYLA/SLSQP/L-BFGS-B stop early; SPSA/Adam/NFT/AQGD use the whole budget |

---

## 8. Things found along the way

### 8.1 In the team's code (worth telling the team)
1. **Their SPIQ driver cannot run AQGD** (their own optimizer option 0). AQGD evaluates in batches,
   and the SPIQ cost function takes one point, so it crashes, or can silently return a wrong
   energy. The study adds an adapter in its own code.
2. **P2's QUBO ground state is not the optimal plan** (§7.4).
3. **Their committed P1 SPIQ logs are for an older version of P1.** Commit `ce7efac6` changed P1
   from 1 to 2 links. Older plain-start logs (Week4/23/24/51) predate an energy-scale fix.
4. **Their plain-start path is ~50× slower than the SPIQ path on 20 qubits** (≈35 s vs 0.67 s per
   evaluation), due to qiskit's expectation code (measured with a profiler).
5. **Their "best energy seen" logging is biased by noise**, as described in §6.

### 8.2 In this study's own tooling (found, fixed, re-run)
1. AQGD's batched evaluations were counted per execution instead of per circuit: fixed, and all AQGD runs redone.
2. AQGD's returned point was taken from a whole batch: fixed, and all AQGD runs redone. All other
   optimizers were verified unaffected.
3. The team's `.gitignore` (`spiq/`) silently kept the SPIQ-start raw files off GitHub: fixed; all 80 are now there.
4. Two claims were stated too strongly and are now corrected: "6 of 10" (not 7) optimizers do not
   move from SPIQ, and SPSA's 98 % on P1 happens in 2 of 3 trials.

---

## 9. Limitations (say these if asked)
- Noiseless simulation and depth p = 2, like the paper. Real hardware noise could change the ranking.
- 2–5 trials per cell: medians and ranges are reported; small differences should not be over-read.
- No per-optimizer tuning (library defaults, or the team's settings). Tuned SPSA/Adam might do better.
- Equal evaluation budgets; runs cut off at the budget are marked.
- The plain start is fixed at 0.5 (as in the team's code), so trials differ only in noise.
- P3's plain-start arm has fewer trials (2–3) because each run takes 10–13 hours.

---

## 10. Likely questions from your professor

**"How do we know the setup is faithful?"** It runs the team's unmodified driver and reproduces their
logged numbers (§6.1). Only three documented hooks change.

**"Isn't COBYLA's drop in our logs real progress?"** No. The exact energy of what it returns is
unchanged (P2: 8.340 vs 8.338). The logged drop comes from keeping the lowest of ~2,000 noisy readings.

**"Why do optimizers get stuck at SPIQ?"** Measured: zero gradient, and no single angle goes downhill (§7.3).

**"So is SPIQ useless?"** No. SPIQ's *start* is far better than anything the plain start reaches. The
finding is that COBYLA adds nothing on top of it, and that NFT/SPSA can.

**"What should we change?"** Use NFT after SPIQ, and tune the QUBO penalty (`penalty_scaling`) so the
ground state is the optimal plan. Check it with the brute-force script before running QAOA.

**"Can I re-check the numbers?"** `python verify_findings.py` re-derives the headline claims from the data.
