"""Batched Laplace inference for pairwise-choice (Bradley-Terry) feedback with an optional
lapse-robust likelihood, and mixture-of-Gaussians belief utilities.

Model for one student with prior N(m0, P0^{-1}) on w:
    P(o = 1 | x, w) = (1 - eps) * sigmoid(x . w) + eps / 2,   x = beta * (phi_i - phi_j)
eps = 0 is the plain logistic model. Everything is batched over n independent problems.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp


def _logdet(A):
    return np.linalg.slogdet(A)[1]


def laplace_fit(X, o, m0, P0, w0=None, eps=0.0, iters=15, tol=1e-5):
    """MAP + Laplace (Fisher-scoring Hessian) for batched logistic/lapse regression.

    X (n,b,d), o (n,b), m0 (n,d), P0 (n,d,d). Returns w (n,d), H (n,d,d), f (n,), log_ev (n,).
    """
    n, b, d = X.shape
    w = m0.copy() if w0 is None else w0.copy()
    if b == 0:
        return w, P0.copy(), np.zeros(n), np.zeros(n)
    XT = np.ascontiguousarray(np.swapaxes(X, 1, 2))

    def probs(wv):
        z = np.einsum("nbd,nd->nb", X, wv)
        s = expit(z)
        p = (1.0 - eps) * s + 0.5 * eps
        return s, p

    def objective(wv):
        s, p = probs(wv)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        nll = -(o * np.log(p) + (1 - o) * np.log(1 - p)).sum(axis=1)
        dw = wv - m0
        return nll + 0.5 * np.einsum("nd,nde,ne->n", dw, P0, dw)

    f = objective(w)
    for _ in range(iters):
        s, p = probs(w)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        dp = (1.0 - eps) * s * (1 - s)                      # dp/dz
        g_z = -(o / p - (1 - o) / (1 - p)) * dp               # d nll / dz
        g = np.einsum("ndb,nb->nd", XT, g_z) + np.einsum("nde,ne->nd", P0, w - m0)
        if np.max(np.abs(g)) < tol:
            break
        fisher = dp * dp / (p * (1 - p))
        H = np.einsum("ndb,nb,nbe->nde", XT, fisher, X) + P0
        step = np.linalg.solve(H, g[..., None])[..., 0]
        alpha = np.ones(n)
        done = np.zeros(n, dtype=bool)
        w_new = w.copy()
        for _ls in range(12):
            w_try = w - alpha[:, None] * step
            f_try = objective(w_try)
            ok = (f_try <= f + 1e-12) & ~done
            if ok.any():
                w_new[ok] = w_try[ok]
                f = np.where(ok, f_try, f)
                done |= ok
            if done.all():
                break
            alpha = np.where(done, alpha, alpha * 0.5)
        if not done.any():
            break
        w = w_new
    s, p = probs(w)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    dp = (1.0 - eps) * s * (1 - s)
    fisher = dp * dp / (p * (1 - p))
    H = np.einsum("ndb,nb,nbe->nde", XT, fisher, X) + P0
    f = objective(w)
    log_ev = -f + 0.5 * _logdet(P0) - 0.5 * _logdet(H)
    return w, H, f, log_ev


@dataclass
class Mix:
    """Mixture of Gaussians over w: weights pi (M,), means m (M,d), covariances C (M,d,d)."""
    pi: np.ndarray
    m: np.ndarray
    C: np.ndarray

    @property
    def M(self):
        return len(self.pi)

    @property
    def d(self):
        return self.m.shape[1]

    def precision(self):
        return np.linalg.inv(self.C)

    def mean(self):
        return self.pi @ self.m

    def sample(self, rng, n=1):
        k = rng.choice(self.M, size=n, p=self.pi)
        L = np.linalg.cholesky(self.C + 1e-10 * np.eye(self.d))
        eps = rng.normal(size=(n, self.d))
        return self.m[k] + np.einsum("nij,nj->ni", L[k], eps)

    @staticmethod
    def isotropic(d, var):
        return Mix(np.ones(1), np.zeros((1, d)), var * np.eye(d)[None])

    def copy(self):
        return Mix(self.pi.copy(), self.m.copy(), self.C.copy())


def condition(prior: Mix, X, o, w_init=None, eps=0.0):
    """Posterior mixture for one student: per-component Laplace, weights by Laplace evidence.
    Returns (posterior Mix, evidence per component)."""
    M, d = prior.M, prior.d
    b = X.shape[0]
    P0 = prior.precision()
    XM = np.broadcast_to(X[None], (M, b, d))
    oM = np.broadcast_to(o[None], (M, b))
    w, H, f, ev = laplace_fit(XM, oM, prior.m, P0, w0=w_init, eps=eps)
    logw = np.log(np.maximum(prior.pi, 1e-300)) + ev
    logw -= logsumexp(logw)
    return Mix(np.exp(logw), w, np.linalg.inv(H)), ev


def probit_click_prob(mu_diff, var_diff, beta_scale=1.0):
    """E_{w~N}[sigmoid(x.w)] approx via probit matching: sigmoid(mu / sqrt(1 + pi/8 var))."""
    return expit(mu_diff / np.sqrt(1.0 + np.pi / 8.0 * np.maximum(var_diff, 0.0)))


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def project_ball(W, S):
    n = np.linalg.norm(W, axis=-1, keepdims=True)
    return np.where(n > S, S * W / np.maximum(n, 1e-12), W)


def vote_scale(U, lo, hi, norm=1):
    """How a component's predicted utilities enter a vote. norm=1: stretched to [0, 1] on this question (the v1
    objective, where every question is worth the same). norm=0: top-anchored in predicted logit units (the v2
    objective, where a question on which the student hardly cares costs little)."""
    if norm:
        return (U - lo) / np.maximum(hi - lo, 1e-9)
    return U - hi
