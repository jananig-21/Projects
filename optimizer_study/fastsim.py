"""Exact (noise-free) QAOA statevector for a diagonal cost Hamiltonian, in numpy.

Reproduces qiskit-terra 0.19's QAOAAnsatz(op, reps) bit-for-bit: |+>^n, then per layer
exp(-i*gamma_l*H_C) (a phase per bitstring) followed by exp(-i*beta_l*sum X) = RX(2*beta_l)
on every qubit. Parameter order matches QAOAAnsatz.parameters: [beta_0..beta_{p-1},
gamma_0..gamma_{p-1}]. Verified against qiskit Statevector in verify_fastsim().
"""
import numpy as np

def qaoa_probs(ising_diag, params, reps):
    n = int(np.log2(len(ising_diag)))
    betas, gammas = params[:reps], params[reps:]
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for l in range(reps):
        psi = psi * np.exp(-1j * gammas[l] * ising_diag)
        c, s = np.cos(betas[l]), -1j * np.sin(betas[l])
        for q in range(n):
            v = psi.reshape(-1, 2, 2 ** q)          # axis 1 = bit q
            a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :], v[:, 1, :] = c * a + s * b, s * a + c * b
            psi = v.reshape(-1)
    return np.abs(psi) ** 2

def verify_fastsim():
    import sys
    from ground_truth import load, all_energies
    from qiskit.circuit.library.n_local.qaoa_ansatz import QAOAAnsatz
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(7)
    for name, idx in (("P1", 0), ("P2", 1), ("P3", 2)):
        card, pred, pred_sel, qubo, _ = load(idx)
        E, _ = all_energies(qubo)
        op, off = qubo.to_ising()
        diag = E - off
        for _ in range(2 if name == "P3" else 5):
            th = rng.uniform(-np.pi, np.pi, 4)
            p_ref = Statevector.from_instruction(QAOAAnsatz(op, reps=2).assign_parameters(th)).probabilities()
            err = np.max(np.abs(qaoa_probs(diag, th, 2) - p_ref))
            assert err < 1e-10, (name, err)
        print(f"{name}: fastsim == qiskit QAOAAnsatz statevector (max |dp| < 1e-10)")

if __name__ == "__main__":
    verify_fastsim()


# ---- relaxed SPIQ circuits (gate set cx / rz / s / sx, plus a few common extras) -----------
_1Q = {
    "s":  lambda th: np.array([[1, 0], [0, 1j]]),
    "sdg": lambda th: np.array([[1, 0], [0, -1j]]),
    "sx": lambda th: 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
    "h":  lambda th: np.array([[1, 1], [1, -1]]) / np.sqrt(2),
    "x":  lambda th: np.array([[0, 1], [1, 0]]),
    "rz": lambda th: np.array([[np.exp(-0.5j * th), 0], [0, np.exp(0.5j * th)]]),
    "rx": lambda th: np.array([[np.cos(th / 2), -1j * np.sin(th / 2)], [-1j * np.sin(th / 2), np.cos(th / 2)]]),
}

def compile_circuit(qc):
    """Turn a parameterised QuantumCircuit into a gate list for circuit_probs (done once per circuit)."""
    params = list(qc.parameters)
    pos = {p: i for i, p in enumerate(params)}
    ops = []
    for inst, qargs, _ in qc.data:
        qs = [qc.find_bit(q).index for q in qargs]
        if inst.name == "cx":
            ops.append(("cx", qs, None)); continue
        if inst.name in ("barrier", "measure"):
            continue
        if inst.name not in _1Q:
            raise ValueError(f"unsupported gate {inst.name}")
        arg = None
        if inst.params:
            p = inst.params[0]
            if hasattr(p, "parameters") and p.parameters:
                (sym,) = p.parameters
                coeff = float(p.bind({sym: 1.0})) if str(p) != sym.name else 1.0
                arg = ("param", pos[sym], coeff)
            else:
                arg = ("const", float(p))
        ops.append((inst.name, qs, arg))
    return qc.num_qubits, ops

def circuit_probs(compiled, theta):
    n, ops = compiled
    psi = np.zeros(2 ** n, complex); psi[0] = 1
    for name, qs, arg in ops:
        if name == "cx":
            c, t = qs
            v = psi.reshape([2] * n)                      # axis n-1-k = qubit k
            idx = [slice(None)] * n; idx[n - 1 - c] = 1
            sub = v[tuple(idx)]
            ax_t = (n - 1 - t) - (1 if (n - 1 - t) > (n - 1 - c) else 0)
            v[tuple(idx)] = np.flip(sub, axis=ax_t)
            psi = v.reshape(-1); continue
        th = None if arg is None else (arg[1] if arg[0] == "const" else arg[2] * theta[arg[1]])
        U = _1Q[name](th)
        q = qs[0]
        v = psi.reshape(-1, 2, 2 ** q)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :], v[:, 1, :] = U[0, 0] * a + U[0, 1] * b, U[1, 0] * a + U[1, 1] * b
        psi = v.reshape(-1)
    return np.abs(psi) ** 2

def verify_circuit_sim():
    import os, sys
    from spiq_artifacts import load_spiq
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(3)
    for P in ("P1", "P2", "P3"):
        pcirc, x0, _ = load_spiq(P)
        comp = compile_circuit(pcirc)
        for th in ([np.array(x0)] + [rng.uniform(-np.pi, np.pi, len(x0))]):
            ref = Statevector.from_instruction(pcirc.assign_parameters(dict(zip(pcirc.parameters, th)))).probabilities()
            err = np.max(np.abs(circuit_probs(comp, th) - ref))
            assert err < 1e-10, (P, err)
        print(f"{P}: circuit_probs == qiskit Statevector on the SPIQ relaxed circuit ({len(x0)} params, max |dp| < 1e-10)")
