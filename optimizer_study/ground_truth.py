"""Ground truth for P1-P3, computed with the project's own QUBO generator and cost model.

For each problem: qubit count, exact ground-state energy of the QUBO (the dashed
baseline in the paper's Figs. 3-5), the ground-state bitstring(s) and what they decode
to, and the cost of every possible left-deep join order (so "optimal" is
defined exactly as Scripts/Postprocessing costs a plan).
"""
import itertools, json, os, sys, contextlib, io
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base")
sys.path.insert(0, BASE)
import Scripts.ProblemGenerator as ProblemGenerator
import Scripts.QUBOGenerator as QUBOGenerator
import Scripts.Postprocessing as Postprocessing

PROBLEMS = {"P1": 0, "P2": 1, "P3": 2}

def load(idx):
    path = os.path.join(BASE, "ExperimentalAnalysis/IBMQ/QPUPerformance/Problems/JSON", f"{idx}_predicates")
    card, pred, pred_sel = ProblemGenerator.get_join_ordering_problem(path)
    with contextlib.redirect_stdout(io.StringIO()):
        qubo, pw = QUBOGenerator.generate_IBMQ_QUBO_for_left_deep_trees_v2(card, pred, pred_sel)
    return card, pred, pred_sel, qubo, pw

def all_energies(qubo):
    """Exact QUBO energy of every bitstring, vectorised (bit k of the index = variable k)."""
    n = qubo.get_num_vars()
    lin = qubo.objective.linear.to_array()
    quad = qubo.objective.quadratic.to_array(symmetric=False)
    const = qubo.objective.constant
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.float64)
    return const + X @ lin + np.einsum("bi,ij,bj->b", X, quad, X), X

def plan_costs(card, pred, pred_sel):
    d = {}
    return {tuple(p): Postprocessing.get_costs_for_leftdeep_tree(list(p), card, pred, pred_sel, d)
            for p in itertools.permutations(range(len(card)))}

def decode(x, card, pred, pred_sel):
    """Decode a bitstring exactly as Postprocessing.readout does (raw and fallback)."""
    R = len(card)
    w = np.arange(1, R - 1)[R - 3::-1]
    cv = np.array(np.array_split(np.array(x[:R * (R - 2)]), R)).dot(w)
    raw = Postprocessing.get_raw_join_order(cv)
    fb = Postprocessing.postprocess_join_order(raw, cv, R, pred)
    d = {}
    return (raw, Postprocessing.get_costs_for_leftdeep_tree(raw, card, pred, pred_sel, d),
            fb, Postprocessing.get_costs_for_leftdeep_tree(fb, card, pred, pred_sel, d))

if __name__ == "__main__":
    out = {}
    for name, idx in PROBLEMS.items():
        card, pred, pred_sel, qubo, pw = load(idx)
        E, X = all_energies(qubo)
        # cross-check the vectorised energy against qiskit's own evaluator on random strings
        rng = np.random.default_rng(0)
        for k in rng.integers(0, len(E), 50):
            assert abs(E[k] - qubo.objective.evaluate(X[k].astype(int).tolist())) < 1e-6
        gs = float(E.min())
        gs_idx = np.flatnonzero(np.isclose(E, gs))
        costs = plan_costs(card, pred, pred_sel)
        opt_cost = min(costs.values())
        gs_decoded = [decode(X[i].astype(int).tolist(), card, pred, pred_sel) for i in gs_idx]
        out[name] = dict(
            input_idx=idx, card=card, pred=[list(p) for p in pred], pred_sel=pred_sel,
            num_qubits=qubo.get_num_vars(), penalty_weight=pw,
            ground_state_energy=gs, num_ground_states=len(gs_idx),
            mean_energy_uniform=float(E.mean()), max_energy=float(E.max()),
            optimal_cost=opt_cost, worst_cost=max(costs.values()),
            distinct_plan_costs=sorted(set(round(c) for c in costs.values())),
            optimal_orders=[list(p) for p, c in costs.items() if c == opt_cost],
            ground_state_decodes_to=[dict(raw=r, raw_cost=rc, fallback=f, fallback_cost=fc) for r, rc, f, fc in gs_decoded],
        )
        o = out[name]
        print(f"{name}: {o['num_qubits']} qubits | ground state E={gs:.4f} ({len(gs_idx)} state(s)) | "
              f"uniform-mean E={o['mean_energy_uniform']:.2f} | optimal cost={opt_cost:g} worst={o['worst_cost']:g} "
              f"| plan costs={o['distinct_plan_costs']}")
        print(f"     ground state decodes to: {o['ground_state_decodes_to'][:2]}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "ground_truth.json"), "w"), indent=2, default=float)
