"""Finding 2 evidence: the exact gradient and single-angle curvature at the team's SPIQ points.

In SPIQ's relaxed ansatz every angle drives one Rz gate, so E(θ) is an exact sinusoid in each
angle and the parameter-shift rule is exact:
    dE/dθ_i  = [E(θ + π/2 e_i) - E(θ - π/2 e_i)] / 2
    curv_i   = [E(θ + π/2 e_i) + E(θ - π/2 e_i) - 2E(θ)] / 2   (>0: uphill both ways, 0: flat, <0: downhill)
Energies are exact (noise-free statevector; fastsim.circuit_probs is verified against qiskit).
Writes analysis/spiq_gradient.json.   Usage:  python measure_spiq_gradient.py [P1 P2 P3]
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "base")]
os.chdir(os.path.join(HERE, "..", "base"))
from spiq_artifacts import load_spiq
from fastsim import compile_circuit, circuit_probs
from decode_table import build

out_path = os.path.join(HERE, "analysis", "spiq_gradient.json")
out = json.load(open(out_path)) if os.path.exists(out_path) else {}
rng = np.random.default_rng(0)
for P in (sys.argv[1:] or ["P1", "P2", "P3"]):
    pc, x0, _ = load_spiq(P); comp = compile_circuit(pc); E = build(P)["energy"]
    f = lambda th: float(circuit_probs(comp, th) @ E)
    x0 = np.array(x0); n = len(x0); S = np.eye(n) * np.pi / 2; f0 = f(x0)
    up = np.array([f(x0 + S[i]) for i in range(n)]); dn = np.array([f(x0 - S[i]) for i in range(n)])
    grad, curv = (up - dn) / 2, (up + dn - 2 * f0) / 2
    xr = x0 + rng.normal(0, 0.3, n)
    grad_r = np.array([(f(xr + S[i]) - f(xr - S[i])) / 2 for i in range(n)])
    out[P] = dict(angles=n, energy_at_spiq=f0, grad_norm=float(np.linalg.norm(grad)), grad_max_abs=float(np.abs(grad).max()),
                  curvature_min=float(curv.min()), n_uphill=int((curv > 1e-9).sum()), n_flat=int((np.abs(curv) <= 1e-9).sum()),
                  n_downhill=int((curv < -1e-9).sum()), grad_norm_nearby_random_point=float(np.linalg.norm(grad_r)))
    print(P, out[P], flush=True)
    json.dump(out, open(out_path, "w"), indent=2)
