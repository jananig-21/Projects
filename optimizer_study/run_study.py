"""Run one (problem, initialization, optimizer, trial) cell of the optimizer study.

The project's own QAOA driver, base/IBMQExperiments.py, runs UNMODIFIED. We only
replace three module-level hooks it exposes:

  get_optimizer(iterations)  -> the optimizer under test, behind an evaluation-budget guard
  get_local_QASM_backend()   -> same QasmSimulator + 10240 shots, but each execution draws a
                                fresh simulator seed from a per-trial RNG (fresh shot noise on
                                every call, as in the original runs, yet reproducible per trial)
  current_optim              -> optimizer name, used by the driver in its output filenames

Everything else - QUBO construction (Scripts/QUBOGenerator.py), the QAOA ansatz, reps=2,
the fixed [0.5]*4 "random-initialization" start, SPIQ's relaxed per-gate ansatz and its
built-in sanity check, energy logging on the QUBO scale, re-sampling the best parameters
seen, pickling the response and decoding it with Scripts/Postprocessing.readout (raw and
fallback join orders) - is the project's code path.

Usage (any cwd):
    python run_study.py --problem P2 --init random --optimizer COBYLA --trial 1 --budget 1000
"""
import argparse, contextlib, io, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", "base"))

ap = argparse.ArgumentParser()
ap.add_argument("--problem", choices=["P1", "P2", "P3"], required=True)
ap.add_argument("--init", choices=["random", "spiq"], required=True)
ap.add_argument("--optimizer", required=True)
ap.add_argument("--trial", type=int, default=1)
ap.add_argument("--budget", type=int, default=1000, help="hard cap on energy evaluations during optimization")
ap.add_argument("--out", default=os.path.join(HERE, "results"))
a = ap.parse_args()

INPUT_IDX = {"P1": 0, "P2": 1, "P3": 2}[a.problem]
REPS, ITERATIONS, SHOTS = 2, 10000, 10240          # the paper's settings (README §4, IBMQExperiments.py)
SEED = 1000 * INPUT_IDX + 100 * (a.init == "spiq") + a.trial

# Their driver parses argv at import time; hand it the paper's flags.
sys.argv = [sys.argv[0], "--trial", str(a.trial), "--reps", str(REPS), "--optimizer", "1",
            "--iterations", str(ITERATIONS), "--input_idx", str(INPUT_IDX)]
os.chdir(BASE)                     # their relative problem paths assume cwd = base/
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)

import numpy as np
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.providers.aer import QasmSimulator
import IBMQExperiments as X                       # the project's driver, unmodified
import Scripts.Postprocessing as Postprocessing
import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator as QUBOGenerator
from optimizers import make_optimizer, BudgetGuard

algorithm_globals.random_seed = SEED               # seeds SPSA / NFT / etc. internal randomness

# ---- hook 2: seeded-per-execution simulator ----------------------------------------------
_rng = np.random.default_rng(SEED)
class _FreshSeedInstance(QuantumInstance):
    """QuantumInstance that re-seeds the simulator on every execute(): fresh shot noise per
    call (like an unseeded run) but the whole sequence is reproducible from the trial seed."""
    def execute(self, circuits, had_transpiled=False):
        self._run_config.seed_simulator = int(_rng.integers(0, 2**31 - 1))    # same order as before
        BudgetGuard.tick(len(circuits) if isinstance(circuits, (list, tuple)) else 1)
        BudgetGuard.exec_count += 1
        return super().execute(circuits, had_transpiled=had_transpiled)
X.get_local_QASM_backend = lambda: _FreshSeedInstance(backend=QasmSimulator(), shots=SHOTS)

# ---- hook 1 + 3: optimizer under test -----------------------------------------------------
X.current_optim = a.optimizer
X.get_optimizer = lambda iterations: BudgetGuard(make_optimizer(a.optimizer, iterations, a.budget), a.budget,
                                                 scalar_objective=(a.init == "spiq"))

# ---- problem, exactly as conduct_IBMQ_QPU_experiments builds it ------------------------------
card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(
    f"ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON/{INPUT_IDX}_predicates")
with contextlib.redirect_stdout(io.StringIO()):
    qubo, penalty_weight = QUBOGenerator.generate_IBMQ_QUBO_for_left_deep_trees_v2(card, pred, pred_sel)

run_dir = os.path.join(a.out, a.problem, a.init, a.optimizer, f"trial{a.trial}")
os.makedirs(run_dir, exist_ok=True)
# resumability: completed optimizer steps are checkpointed and replayed after a kill (optimizers.py)
BudgetGuard.checkpoint_path = os.path.join(run_dir, "checkpoint.jsonl")
BudgetGuard.advance_rng = lambda k: [_rng.integers(0, 2**31 - 1) for _ in range(k)]

t0 = time.time()
if a.init == "random":
    response, final_point, used_eval, min_state = X.solve_with_QAOA(
        qubo, ITERATIONS, run_dir, reps=REPS, use_local_simulator=True)
else:
    from spiq_artifacts import load_spiq
    pcirc, x0, meta = load_spiq(a.problem)
    response, final_point, used_eval, min_state = X.solve_with_QAOA_spiq(
        qubo, ITERATIONS, pcirc, x0, run_dir, use_local_simulator=True,
        expected_energy=meta["energy_best"])       # their built-in SPIQ reproduction check
wall = time.time() - t0

# ---- outputs in the project's own formats ---------------------------------------------------
if min_state is not None:        # min_state_readout.csv, as conduct_IBMQ_QPU_experiments writes it
    import csv
    with open(os.path.join(run_dir, "min_state_readout.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bitstring", "energy", "prob"])
        for s in min_state["samples"]:
            w.writerow(["".join(map(str, s["x"])), s["fval"], s["probability"]])
X.pickle_results(run_dir, response)                # results.txt (pickled response)

# readout_summary.csv via the project's decoder (raw + fallback join orders, costed)
with contextlib.redirect_stdout(io.StringIO()):
    Postprocessing.postprocess_qiskit_with_readout(
        response, card, pred, pred_sel, trial_id=a.trial, tag=REPS, current_optim=a.optimizer,
        iterations=ITERATIONS, base_dir=run_dir, shots=SHOTS)

json.dump(dict(problem=a.problem, init=a.init, optimizer=a.optimizer, trial=a.trial, seed=SEED,
               budget=a.budget, reps=REPS, shots=SHOTS, wall_time_s=wall,
               evaluations=BudgetGuard.total_during_opt, budget_hit=BudgetGuard.exhausted,
               driver_eval_count=used_eval, replayed_steps=BudgetGuard.replayed_steps,
               min_energy=None if min_state is None else min_state["min_energy"],
               min_eval_idx=None if min_state is None else min_state["min_eval_idx"],
               optimizer_settings=BudgetGuard.settings),
          open(os.path.join(run_dir, "run_meta.json"), "w"), indent=2, default=str)
print(f"[done] {a.problem} {a.init} {a.optimizer} trial{a.trial}: evals={BudgetGuard.total_during_opt} "
      f"budget_hit={BudgetGuard.exhausted} best_E={None if min_state is None else round(min_state['min_energy'], 4)} "
      f"wall={wall:.0f}s")
