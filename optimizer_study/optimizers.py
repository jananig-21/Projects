"""The optimizer roster, plus an evaluation-budget guard.

COBYLA, AQGD and SPSA are constructed EXACTLY as in the project's
IBMQExperiments.get_optimizer (options 1, 0, 2). All other optimizers use the
library (qiskit-terra 0.19.2) defaults, with two documented exceptions:

* FD_EPS - gradient-based methods (L-BFGS-B, SLSQP, ADAM) are handed no analytic
  gradient by the driver, so they difference the objective numerically. Their
  default step (1e-8 to 1e-10) is ~10^6x smaller than the shot noise of a 10240-shot
  energy estimate, which would make every gradient pure noise. We use a
  step of 0.1 rad, a standard choice for sampled VQA objectives.
* BOBYQA requires finite bounds; when the ansatz supplies none we use x0 +/- pi
  (a full period around the starting point, since all angles are periodic).
"""
import numpy as np
from qiskit.algorithms.optimizers import (ADAM, AQGD, BOBYQA, COBYLA, L_BFGS_B, NELDER_MEAD, NFT,
                                          POWELL, SLSQP, SPSA, OptimizerResult)

FD_EPS = 0.1

ROSTER = {
    # name: (family, one-line description used in the report)
    "COBYLA":      ("derivative-free, trust region", "Linear-approximation trust region. The team's paper baseline."),
    "AQGD":        ("gradient, quantum-native", "Analytic Quantum Gradient Descent (parameter-shift + momentum). The original SIGMOD'23 choice."),
    "SPSA":        ("stochastic gradient", "Simultaneous-perturbation stochastic approximation. 2 evals/step regardless of dimension; built for noisy objectives."),
    "NELDER_MEAD": ("derivative-free, simplex", "Downhill simplex."),
    "POWELL":      ("derivative-free, line search", "Conjugate-direction line searches."),
    "BOBYQA":      ("derivative-free, trust region", "Quadratic-model trust region (Powell). The quadratic successor to COBYLA's linear model."),
    "L_BFGS_B":    ("gradient, quasi-Newton", "Limited-memory BFGS with finite-difference gradients."),
    "SLSQP":       ("gradient, SQP", "Sequential least-squares quadratic programming with finite-difference gradients."),
    "ADAM":        ("gradient, adaptive", "Adam, the default optimizer of deep learning, with finite-difference gradients."),
    "NFT":         ("quantum-native, sequential", "Nakanishi-Fujii-Todo sequential minimal optimization: exact sinusoidal fits, one parameter at a time."),
}

def make_optimizer(name, iterations, budget):
    if name == "COBYLA":      return COBYLA(maxiter=iterations, rhobeg=2.0, tol=1e-12)          # = their option 1
    if name == "AQGD":        return AQGD(maxiter=iterations, eta=0.01)                         # = their option 0
    if name == "SPSA":        return SPSA(maxiter=iterations)                                   # = their option 2
    if name == "NELDER_MEAD": return NELDER_MEAD(maxfev=budget)
    if name == "POWELL":      return POWELL(maxfev=budget)
    if name == "BOBYQA":      return BOBYQA(maxiter=budget)
    if name == "L_BFGS_B":    return L_BFGS_B(maxfun=budget, maxiter=iterations, eps=FD_EPS)
    if name == "SLSQP":       return SLSQP(maxiter=iterations, eps=FD_EPS)
    if name == "ADAM":        return ADAM(maxiter=iterations, lr=0.05, eps=FD_EPS)
    if name == "NFT":         return NFT(maxiter=iterations, maxfev=budget)
    raise ValueError(f"unknown optimizer {name}; choose from {list(ROSTER)}")


class _BudgetExhausted(Exception):
    pass


class BudgetGuard:
    """Wraps an optimizer so optimization stops after `budget` energy evaluations.

    Every circuit execution during minimize() counts as one evaluation (the harness's
    QuantumInstance calls BudgetGuard.tick()); this covers function values AND the
    evaluations an optimizer spends on its own numerical gradients. When the budget
    runs out we return the best point evaluated so far, so the driver continues
    normally (re-samples the best state, logs, pickles) exactly as after a natural stop.
    """
    active = False
    total_during_opt = 0
    exhausted = False
    budget = None
    settings = {}

    def __init__(self, inner, budget, scalar_objective=False):
        self._inner = inner
        # The SPIQ driver's cost_fn evaluates ONE parameter vector. AQGD evaluates its
        # parameter-shift gradient as a batch (concatenated vectors); VQE supports that,
        # the SPIQ cost_fn does not (it would crash, or zip() would silently keep only the
        # first point). For scalar objectives we split AQGD's batch into single calls.
        self._scalar_objective = scalar_objective
        BudgetGuard.budget = budget
        try:
            BudgetGuard.settings = dict(type=type(inner).__name__, **inner.settings, fd_eps_note=FD_EPS)
        except Exception:
            BudgetGuard.settings = dict(type=type(inner).__name__)

    def __getattr__(self, item):            # delegate everything else (set_max_evals_grouped, ...)
        return getattr(self._inner, item)

    @classmethod
    def tick(cls, n_circuits=1):
        """Count n_circuits energy evaluations (an optimizer may batch several, e.g. AQGD's
        parameter-shift gradient runs 2*dim+1 circuits in one execution)."""
        if cls.active:
            if cls.total_during_opt + n_circuits > cls.budget:
                cls.exhausted = True
                raise _BudgetExhausted()
            cls.total_during_opt += n_circuits

    def minimize(self, fun, x0, jac=None, bounds=None):
        n = len(np.asarray(x0))
        best = {"x": np.asarray(x0, float), "f": np.inf}
        def tracked(x):
            f = fun(x)
            xs = np.asarray(x, float).reshape(-1, n)        # a batched call carries several points
            fs = np.atleast_1d(np.asarray(f, float))
            k = int(np.argmin(fs))
            if fs[k] < best["f"]:
                best["x"], best["f"] = xs[k].copy(), float(fs[k])
            return f
        if self._scalar_objective and isinstance(self._inner, AQGD):
            single = tracked
            def tracked(x):
                x = np.asarray(x, float)
                if x.size == n:
                    return single(x)
                return np.array([single(c) for c in x.reshape(-1, n)])
        if isinstance(self._inner, BOBYQA) and (bounds is None or any(b is None for pair in bounds for b in pair)):
            x0a = np.asarray(x0, float)
            bounds = [(x - np.pi, x + np.pi) for x in x0a]
        BudgetGuard.active = True
        try:
            res = self._inner.minimize(fun=tracked, x0=x0, jac=jac, bounds=bounds)
        except _BudgetExhausted:
            res = OptimizerResult()
            res.x, res.fun, res.nfev = best["x"], best["f"], BudgetGuard.total_during_opt
        finally:
            BudgetGuard.active = False
        return res
