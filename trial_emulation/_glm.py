"""Exact Newton/IRLS fitting for the binomial-logit and Gaussian-identity GLMs.

The weight models (switching and censoring) and the MSM are maximum-likelihood
GLMs with canonical links, for which IRLS and Newton's method coincide. This
module fits them with the same algorithm statsmodels uses (same starting
values, same iteration, same deviance convergence rule) but solves each step
from the p x p weighted Gram matrix instead of a least-squares solve on the
n x p whitened design, and skips statsmodels' final pseudo-inverse of that
design. The result is returned as an ordinary statsmodels ``GLMResults``
object built from the same formula model, so ``params``, ``bse``, ``summary()``,
``predict()``, ``score_obs`` and pickling behave as before.

``solver="statsmodels"`` keeps the reference path (``GLM.fit``). The numpy
solver also hands over to that path whenever its preconditions fail (non-binary
binomial response, offsets or variance weights, a singular or ill-conditioned
step, or no convergence), so hard problems are fitted exactly as before.
``solver="torch"`` runs the same iteration in float64 with PyTorch, on CUDA
when available; it is used only when explicitly requested.
"""

from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import scipy.linalg
import statsmodels.api as sm
from statsmodels.formula.api import glm as _formula_glm
from statsmodels.genmod.generalized_linear_model import GLMResults, GLMResultsWrapper

SOLVERS = ("numpy", "statsmodels", "torch")

# Fit options the Newton solver honours; any other GLM.fit option uses statsmodels.
_NEWTON_FIT_KWARGS = {"maxiter", "tol", "disp"}
# Rows per block when forming X'WX, bounding temporary memory to ~block x p.
_BLOCK_ROWS = 1 << 20
# Largest condition number of the Jacobi-scaled step matrix the solver accepts.
_MAX_CONDITION = 1e12


class _FallBack(Exception):
    """The Newton solver declines a problem; statsmodels fits it instead."""


def check_solver(solver: str) -> str:
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got '{solver}'")
    return solver


def family_for(name: str) -> Any:
    if name == "binomial":
        return sm.families.Binomial(link=sm.families.links.Logit())
    if name == "gaussian":
        return sm.families.Gaussian(link=sm.families.links.Identity())
    raise ValueError(f"family must be 'binomial' or 'gaussian', got '{name}'")


def fit_formula_glm(
    formula: str,
    data: Any,
    family: str,
    *,
    freq_weights: Optional[np.ndarray] = None,
    missing: str = "drop",
    solver: str = "numpy",
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Fit ``formula`` on ``data`` and return statsmodels GLM results.

    The model object is always the statsmodels formula GLM; only the
    optimisation differs between solvers. ``results.mle_settings['solver']``
    records which solver produced the estimates.
    """
    check_solver(solver)
    fit_kwargs = dict(fit_kwargs or {})
    model_kwargs: Dict[str, Any] = {"family": family_for(family), "missing": missing}
    if freq_weights is not None:
        model_kwargs["freq_weights"] = freq_weights
    model = _formula_glm(formula, data=data, **model_kwargs)
    return fit_glm_model(model, solver=solver, fit_kwargs=fit_kwargs)


def fit_glm_model(model: Any, *, solver: str = "numpy", fit_kwargs: Optional[Dict] = None) -> Any:
    """Fit a statsmodels ``GLM`` model object with the chosen solver."""
    check_solver(solver)
    fit_kwargs = dict(fit_kwargs or {})
    reason = None
    if solver != "statsmodels":
        unsupported = set(fit_kwargs) - _NEWTON_FIT_KWARGS
        if unsupported:
            reason = "fit options " + ", ".join(sorted(unsupported))
        else:
            backend = _TorchBackend() if solver == "torch" else _NumpyBackend()
            try:
                return _newton_fit(
                    model,
                    backend,
                    maxiter=int(fit_kwargs.get("maxiter", 100)),
                    tol=float(fit_kwargs.get("tol", 1e-8)),
                    solver=solver,
                )
            except _FallBack as exc:
                reason = str(exc)
    results = model.fit(**fit_kwargs)
    settings = getattr(results, "mle_settings", None)
    if isinstance(settings, dict):
        settings["solver"] = "statsmodels"
        if reason is not None:
            settings["solver_fallback_reason"] = reason
    return results


# ---------------------------------------------------------------------------
# Backends: where the O(n p) work runs. The p x p algebra always runs in numpy.


class _NumpyBackend:
    name = "numpy"

    def asarray(self, values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(values, dtype=float)

    def to_numpy(self, values: Any) -> np.ndarray:
        return np.asarray(values)

    def gram(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        n, p = x.shape
        out = np.zeros((p, p), dtype=float)
        for start in range(0, n, _BLOCK_ROWS):
            block = x[start : start + _BLOCK_ROWS]
            out += block.T @ (block * w[start : start + _BLOCK_ROWS, None])
        return (out + out.T) / 2.0

    def xtv(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return x.T @ v

    def matvec(self, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return x @ beta

    def vec(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    exp = staticmethod(np.exp)
    log = staticmethod(np.log)

    def clip(self, values, low, high):
        return np.clip(values, low, high)

    def total(self, values) -> float:
        return float(np.sum(values))


class _TorchBackend:
    name = "torch"

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "solver='torch' requires PyTorch; install torch or use solver='numpy'"
            ) from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def asarray(self, values: np.ndarray):
        return self.torch.as_tensor(
            np.ascontiguousarray(values, dtype=float), dtype=self.torch.float64, device=self.device
        )

    def to_numpy(self, values: Any) -> np.ndarray:
        return values.detach().cpu().numpy()

    def gram(self, x, w) -> np.ndarray:
        n, p = x.shape
        out = self.torch.zeros((p, p), dtype=self.torch.float64, device=self.device)
        for start in range(0, n, _BLOCK_ROWS):
            block = x[start : start + _BLOCK_ROWS]
            out += block.T @ (block * w[start : start + _BLOCK_ROWS, None])
        out = self.to_numpy(out)
        return (out + out.T) / 2.0

    def xtv(self, x, v) -> np.ndarray:
        return self.to_numpy(x.T @ v)

    def matvec(self, x, beta: np.ndarray):
        return x @ self.vec(beta)

    def vec(self, values: np.ndarray):
        return self.torch.as_tensor(values, dtype=self.torch.float64, device=self.device)

    def exp(self, values):
        return self.torch.exp(values)

    def log(self, values):
        return self.torch.log(values)

    def clip(self, values, low, high):
        return self.torch.clamp(values, low, high)

    def total(self, values) -> float:
        return float(values.sum().item())


# ---------------------------------------------------------------------------
# Newton/IRLS iteration.

_EPS = np.finfo(float).eps


def _family_kind(model: Any) -> str:
    family = model.family
    if isinstance(family, sm.families.Binomial) and isinstance(
        family.link, sm.families.links.Logit
    ):
        return "logit"
    if isinstance(family, sm.families.Gaussian) and isinstance(
        family.link, sm.families.links.Identity
    ):
        return "identity"
    raise _FallBack(f"family {type(family).__name__} with link {type(family.link).__name__}")


def _solve_step(h: np.ndarray, rhs: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, Any]]:
    """Solve ``h @ step = rhs`` by Cholesky on the Jacobi-scaled matrix."""
    diagonal = np.diag(h)
    if not np.all(np.isfinite(h)) or np.any(diagonal <= 0):
        raise _FallBack("singular step matrix")
    scale = np.sqrt(diagonal)
    scaled = h / np.outer(scale, scale)
    if np.linalg.cond(scaled) > _MAX_CONDITION:
        raise _FallBack("ill-conditioned step matrix")
    try:
        factor = scipy.linalg.cho_factor(scaled, lower=False, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise _FallBack("step matrix is not positive definite") from exc
    step = scipy.linalg.cho_solve(factor, rhs / scale, check_finite=False) / scale
    return step, (scale, factor)


def _newton_fit(model: Any, backend: Any, *, maxiter: int, tol: float, solver: str) -> Any:
    kind = _family_kind(model)
    endog = np.asarray(model.endog, dtype=float)
    if endog.ndim != 1:
        raise _FallBack("multi-column response")
    if kind == "logit" and not np.isin(endog, (0.0, 1.0)).all():
        raise _FallBack("binomial response is not 0/1")
    if np.any(np.asarray(model._offset_exposure) != 0):
        raise _FallBack("offset or exposure")
    if np.any(np.asarray(model.var_weights) != 1):
        raise _FallBack("var_weights")
    if maxiter < 1:
        raise _FallBack("maxiter < 1")
    freq = np.asarray(model.freq_weights, dtype=float)

    x = backend.asarray(model.exog)
    y = backend.vec(endog)
    wf = backend.vec(freq)

    def mean_and_weight(eta):
        if kind == "logit":
            mu = 1.0 / (1.0 + backend.exp(-eta))
            clipped = backend.clip(mu, _EPS, 1.0 - _EPS)
            return mu, clipped * (1.0 - clipped)
        return eta, None

    def deviance(mu) -> float:
        if kind == "logit":
            # y is 0/1: the likelihood term is log(mu) or log(1 - mu).
            clipped = backend.clip(mu, _EPS, 1.0 - _EPS)
            fitted_prob = y * clipped + (1.0 - y) * (1.0 - clipped)
            return -2.0 * backend.total(wf * backend.log(fitted_prob))
        resid = y - mu
        return backend.total(wf * resid * resid)

    # Statsmodels' starting values, and its first IRLS step from them.
    mu0 = np.asarray(model.family.starting_mu(endog), dtype=float)
    eta0 = np.asarray(model.family.predict(mu0), dtype=float)
    mu = backend.vec(mu0)
    eta = backend.vec(eta0)
    weight = mu * (1.0 - mu) if kind == "logit" else None
    dev_previous = deviance(mu)
    if not np.isfinite(dev_previous):
        raise _FallBack("non-finite starting deviance")

    history: Dict[str, list] = {"params": [np.inf, np.zeros(x.shape[1])], "deviance": [np.inf]}
    history["deviance"].append(dev_previous)
    beta = None
    converged = False
    iteration = 0
    for iteration in range(1, maxiter + 1):
        irls_weight = wf if weight is None else wf * weight
        h = backend.gram(x, irls_weight)
        if beta is None:
            # IRLS form: working response z = eta + (y - mu) / v.
            rhs = backend.xtv(x, irls_weight * eta + wf * (y - mu))
            beta, factor = _solve_step(h, rhs)
        else:
            # Newton form of the same step (canonical link): accurate as it shrinks.
            step, factor = _solve_step(h, backend.xtv(x, wf * (y - mu)))
            beta = beta + step
        if not np.all(np.isfinite(beta)):
            raise _FallBack("non-finite parameters")
        eta = backend.matvec(x, beta)
        mu, weight = mean_and_weight(eta)
        dev = deviance(mu)
        if not np.isfinite(dev):
            raise _FallBack("non-finite deviance")
        history["params"].append(beta.copy())
        history["deviance"].append(dev)
        # Statsmodels' rule (absolute deviance change <= tol), with a floor at
        # the rounding level of the deviance sum for very large problems.
        if abs(dev - dev_previous) <= max(tol, 1e-13 * abs(dev)):
            converged = True
            break
        dev_previous = dev
    if not converged:
        raise _FallBack(f"no convergence in {maxiter} iterations")

    # Covariance from the step matrix of the final iteration, as statsmodels
    # takes it from the weights of its final weighted least-squares fit.
    scale, (chol, lower) = factor
    inverse_scaled = scipy.linalg.cho_solve((chol, lower), np.eye(len(scale)), check_finite=False)
    normalized_cov = inverse_scaled / np.outer(scale, scale)
    normalized_cov = (normalized_cov + normalized_cov.T) / 2.0

    mu_np = backend.to_numpy(mu)
    model.mu = mu_np
    model.weights = backend.to_numpy(irls_weight)
    model.scale = model.estimate_scale(mu_np)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        results = GLMResults(model, beta, normalized_cov, model.scale, cov_type="nonrobust")
    results.method = "IRLS"
    results.mle_settings = {
        "wls_method": "newton-cholesky",
        "optimizer": "IRLS",
        "solver": solver,
        "device": str(getattr(backend, "device", "cpu")),
    }
    history["iteration"] = iteration
    results.fit_history = history
    results.converged = True
    return GLMResultsWrapper(results)
