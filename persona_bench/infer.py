"""Batched Laplace machinery for Bradley-Terry (logistic) feedback.

Used by the shipped no-transfer, oracle, and pooled baselines. A third-party
`Policy` does not have to use it.

Model for one student with prior N(m0, P0^{-1}) on w:
    P(o = 1 | x, w) = sigmoid(x . w),   x = beta * (phi_i - phi_j)   (beta absorbed into x)

`fit_laplace` runs damped Newton in batch over n independent problems (students x mixture
components) and returns the mode, the posterior precision and the Laplace evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import logsumexp, expit


def _logdet(A: np.ndarray) -> np.ndarray:
    sign, ld = np.linalg.slogdet(A)
    return ld


def fit_gauss(Xs: np.ndarray, y: np.ndarray, m0: np.ndarray, P0: np.ndarray, tau: float):
    """Batched exact Gaussian-likelihood posterior: y = x.w + N(0, tau^2). Same outputs as
    `fit_laplace` (numerical-feedback reference mode)."""
    n, b, d = Xs.shape
    if b == 0:
        return m0.copy(), P0.copy(), np.zeros(n), np.zeros(n)
    H = P0 + np.einsum("nbd,nbe->nde", Xs, Xs) / tau**2
    rhs = np.einsum("ndk,nk->nd", P0, m0) + np.einsum("nbd,nb->nd", Xs, y) / tau**2
    w = np.linalg.solve(H, rhs[..., None])[..., 0]
    res = y - np.einsum("nbd,nd->nb", Xs, w)
    dw = w - m0
    f = 0.5 * (res**2).sum(axis=1) / tau**2 + 0.5 * np.einsum("nd,ndk,nk->n", dw, P0, dw)
    log_ev = -f + 0.5 * _logdet(P0) - 0.5 * _logdet(H) - 0.5 * b * np.log(2 * np.pi * tau**2)
    return w, H, f, log_ev


def fit_laplace(Xs: np.ndarray, o: np.ndarray, m0: np.ndarray, P0: np.ndarray,
                w0: Optional[np.ndarray] = None, iters: int = 12, tol: float = 1e-5,
                lik: str = "logit", tau: float = 0.1, S_cap: Optional[float] = None):
    """Batched MAP + Laplace for logistic regression with Gaussian prior.

    Xs (n,b,d) scaled contrasts; o (n,b) in {0,1}; m0 (n,d); P0 (n,d,d) prior precision.
    Returns w (n,d), H (n,d,d) posterior precision, f (n,) negative log joint at the mode,
    log_ev (n,) Laplace log evidence (up to the common (2*pi) constants, which cancel).
    With lik="gauss" the exact Gaussian posterior is returned instead (rating feedback).
    """
    if lik == "gauss":
        return fit_gauss(Xs, o, m0, P0, tau)
    n, b, d = Xs.shape
    w = m0.copy() if w0 is None else w0.copy()
    if b == 0:
        return w, P0.copy(), np.zeros(n), np.zeros(n)
    XsT = np.ascontiguousarray(np.swapaxes(Xs, 1, 2))            # (n,d,b)

    def objective(wv):
        z = (Xs @ wv[:, :, None])[:, :, 0]
        nll = (np.logaddexp(0.0, z) - o * z).sum(axis=1)
        dw = wv - m0
        return nll + 0.5 * (dw * (P0 @ dw[:, :, None])[:, :, 0]).sum(axis=1)

    f = objective(w)
    for _ in range(iters):
        z = (Xs @ w[:, :, None])[:, :, 0]
        p = expit(z)
        g = (XsT @ (p - o)[:, :, None])[:, :, 0] + (P0 @ (w - m0)[:, :, None])[:, :, 0]
        if np.max(np.abs(g)) < tol:
            break
        Wt = p * (1.0 - p)
        H = XsT @ (Xs * Wt[:, :, None]) + P0
        step = np.linalg.solve(H, g[..., None])[..., 0]
        alpha = np.ones(n)
        ok = np.zeros(n, dtype=bool)
        w_new = w
        for _ls in range(12):
            w_try = w - alpha[:, None] * step
            f_try = objective(w_try)
            ok_now = f_try <= f + 1e-12
            take = ok_now & ~ok
            if take.any():
                w_new = np.where(take[:, None], w_try, w_new)
                f = np.where(take, f_try, f)
            ok = ok | ok_now
            if ok.all():
                break
            alpha = np.where(ok, alpha, alpha * 0.5)
        w = np.where(ok[:, None], w_new, w)
        if S_cap is not None:                                 # spec §7.1: project the mode onto the ball
            w = _project_rows(w, S_cap)
            f = objective(w)
        if not ok.any():
            break
    z = (Xs @ w[:, :, None])[:, :, 0]
    p = expit(z)
    Wt = p * (1.0 - p)
    H = XsT @ (Xs * Wt[:, :, None]) + P0
    f = objective(w)
    log_ev = -f + 0.5 * _logdet(P0) - 0.5 * _logdet(H)
    return w, H, f, log_ev


# --------------------------------------------------------------------------- mixture belief

@dataclass
class Mixture:
    """Mixture of Gaussians over w: weights pi (M,), means m (M,d), covariances C (M,d,d)."""

    pi: np.ndarray
    m: np.ndarray
    C: np.ndarray

    @property
    def M(self) -> int:
        return len(self.pi)

    @property
    def d(self) -> int:
        return self.m.shape[1]

    def precision(self) -> np.ndarray:
        return np.linalg.inv(self.C)

    def mean(self) -> np.ndarray:
        return self.pi @ self.m

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        k = rng.choice(self.M, size=n, p=self.pi)
        L = np.linalg.cholesky(self.C + 1e-10 * np.eye(self.d))
        eps = rng.normal(size=(n, self.d))
        return self.m[k] + np.einsum("nij,nj->ni", L[k], eps)

    @staticmethod
    def isotropic(d: int, var: float) -> "Mixture":
        return Mixture(np.ones(1), np.zeros((1, d)), var * np.eye(d)[None])

    @staticmethod
    def gaussian(mu: np.ndarray, Sigma: np.ndarray) -> "Mixture":
        return Mixture(np.ones(1), mu[None].copy(), Sigma[None].copy())

    @staticmethod
    def atoms(W: np.ndarray, weights: Optional[np.ndarray] = None, var: float = 1e-4) -> "Mixture":
        M, d = W.shape
        pi = np.full(M, 1.0 / M) if weights is None else weights
        return Mixture(pi.copy(), W.copy(), np.repeat(var * np.eye(d)[None], M, axis=0))


def condition(prior: Mixture, Xs: np.ndarray, o: np.ndarray, w_init: Optional[np.ndarray] = None,
              lik: str = "logit", tau: float = 0.1, S_cap: Optional[float] = None):
    """Posterior mixture for one student given scaled contrasts Xs (b,d) and labels o (b,).
    Each component is updated by Laplace; component weights by Laplace evidence."""
    M, d = prior.M, prior.d
    b = Xs.shape[0]
    P0 = prior.precision()
    XsM = np.broadcast_to(Xs[None], (M, b, d))
    oM = np.broadcast_to(o[None], (M, b))
    w, H, f, ev = fit_laplace(XsM, oM, prior.m, P0, w0=w_init, lik=lik, tau=tau, S_cap=S_cap)
    logw = np.log(np.maximum(prior.pi, 1e-300)) + ev
    logw -= logsumexp(logw)
    post = Mixture(np.exp(logw), w, np.linalg.inv(H))
    return post, ev




# --------------------------------------------------------------------------- Gaussian utilities

def expected_max_pair(mu_i, mu_j, var_diff):
    """E[max(X, Y)] for jointly Gaussian X, Y with means mu_i, mu_j and Var(X - Y) = var_diff
    (Clark 1961)."""
    from scipy.stats import norm
    s = np.sqrt(np.maximum(var_diff, 1e-14))
    a = (mu_i - mu_j) / s
    return mu_i * norm.cdf(a) + mu_j * norm.cdf(-a) + s * norm.pdf(a)


def _project_rows(W: np.ndarray, S: float) -> np.ndarray:
    n = np.linalg.norm(W, axis=1, keepdims=True)
    return np.where(n > S, S * W / np.maximum(n, 1e-12), W)

