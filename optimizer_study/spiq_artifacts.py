"""SPIQ initializations used in the study: the team's own committed artifacts.

    P1 (3 relations, 2 predicates)  spiq_init_outputs/spiq_initial_point_input0_reps2_3rel_2pred.json  n_gens=4000
    P2 (4 relations, 2 predicates)  new_spiq_outputs/spiq_initial_point_input0_reps2_4rel_2pred.json   n_gens=2000
    P3 (4 relations, 6 predicates)  spiq_init_outputs/spiq_initial_point_input0_reps2_4rel2.json       n_gens=4000

P3's file carries no problem metadata; it is identified by (a) 268 relaxed angles = the P3
ansatz size and (b) its energy, 9.867 on the current P3 QUBO, matching the start of the team's
logged P3 SPIQ run (Week60, 9.84). The alternative 4rel_6pred artifact (n_gens=1000) starts at
14.75. Every artifact is re-validated at run time by the driver's own SPIQ sanity check.
"""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT = {
    "P1": "spiq_init_outputs/spiq_initial_point_input0_reps2_3rel_2pred.json",
    "P2": "new_spiq_outputs/spiq_initial_point_input0_reps2_4rel_2pred.json",
    "P3": "spiq_init_outputs/spiq_initial_point_input0_reps2_4rel2.json",
}

def load_spiq(problem):
    try:
        from qiskit import qpy
    except ImportError:
        from qiskit.circuit import qpy_serialization as qpy
    jpath = os.path.join(ROOT, ARTIFACT[problem])
    meta = json.load(open(jpath))
    raw = meta["pcirc_qpy"]
    for cand in (os.path.join(ROOT, raw), os.path.join(os.path.dirname(jpath), os.path.basename(raw))):
        if os.path.exists(cand):
            pcirc = qpy.load(open(cand, "rb"))[0]
            break
    else:
        raise FileNotFoundError(raw)
    by_name = dict(zip(meta["relaxed_param_names"], meta["relaxed_initial_point"]))
    x0 = [by_name[p.name] for p in pcirc.parameters]          # align by NAME, not position
    return pcirc, x0, meta
