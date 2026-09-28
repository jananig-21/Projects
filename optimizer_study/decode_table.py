"""Per-bitstring decode table: for every bitstring of a problem, the plan the project's decoder
(Scripts/Postprocessing: raw score-vector order, and the Schoenberger fallback) produces and its
cost, plus the QUBO energy. Lets us compute EXACT (shot-noise-free) plan-quality metrics of any
QAOA state: P(optimal plan) = probabilities @ (cost == optimal_cost).

Index convention: bit k of the integer index = QUBO variable k (= qubit k), verified against
qubo.objective.evaluate and against qiskit Statevector.probabilities() ordering.
"""
import os, numpy as np
from ground_truth import load, all_energies
import Scripts.Postprocessing as Postprocessing

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

def build(problem):
    idx = {"P1": 0, "P2": 1, "P3": 2}[problem]
    path = os.path.join(CACHE, f"decode_{problem}.npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    card, pred, pred_sel, qubo, _ = load(idx)
    E, X = all_energies(qubo)
    R = len(card)
    w = np.arange(1, R - 1)[R - 3::-1]
    rel = X[:, :R * (R - 2)].reshape(len(X), R, R - 2) @ w          # score vector per bitstring
    memo, pcost = {}, {}
    raw_cost = np.empty(len(X)); fb_cost = np.empty(len(X))
    for i in range(len(X)):
        cv = rel[i]
        key = tuple(cv)
        if key not in memo:
            raw = Postprocessing.get_raw_join_order(np.array(cv))
            fb = Postprocessing.postprocess_join_order(raw, np.array(cv), R, pred)
            for o in (raw, fb):
                if tuple(o) not in pcost:
                    pcost[tuple(o)] = Postprocessing.get_costs_for_leftdeep_tree(list(o), card, pred, pred_sel, {})
            memo[key] = (pcost[tuple(raw)], pcost[tuple(fb)])
        raw_cost[i], fb_cost[i] = memo[key]
    opt = min(pcost.values()) if False else None
    import itertools
    opt = min(Postprocessing.get_costs_for_leftdeep_tree(list(p), card, pred, pred_sel, {})
              for p in itertools.permutations(range(R)))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, energy=E, raw_cost=raw_cost, fb_cost=fb_cost, optimal_cost=opt)
    return dict(np.load(path, allow_pickle=True))
