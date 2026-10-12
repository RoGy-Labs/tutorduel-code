"""Kernel / particle population prior: every completed student contributes one mixture component,
its Laplace posterior N(m_i, kde_scale * C_i) computed under a base prior (the isotropic no-transfer
prior, or an empirical-Bayes Gaussian fitted to all sessions). Plus the isotropic fallback.
Within-session inference and pair rules are inherited from HBMix."""
from __future__ import annotations

import numpy as np

from .hb import HBMix, parse_opts
from .infer import laplace_fit, project_ball


class KDEPrior(HBMix):
    DEFAULTS = dict(HBMix.DEFAULTS)
    DEFAULTS.update(dict(base="iso", kde_scale=1.0, max_comp=400, eb_iters=3, n_init=1,
                         refine=0, refine_every=5, refine_min=8))

    def _eb_gaussian(self, X, O):
        """One-component EM (Gaussian population) -> (mu, Sigma)."""
        n = X.shape[0]
        mu = np.zeros(self.D); Sig = self.v0 * np.eye(self.D)
        w0 = None
        for _ in range(self.opt["eb_iters"]):
            P0 = np.tile(np.linalg.inv(Sig)[None], (n, 1, 1))
            m, H, f, ev = laplace_fit(X, O, np.tile(mu, (n, 1)), P0, w0=w0, eps=self.opt["eps"])
            w0 = m
            C = np.linalg.inv(H)
            mu = m.mean(axis=0)
            dm = m - mu
            Sig = (C + dm[:, :, None] * dm[:, None, :]).mean(axis=0)
            Sig = self._regularize_cov(Sig[None])[0]
            if self.opt["cap"]:
                mu = project_ball(mu[None], self.S)[0]
        return mu, Sig

    def _em(self):
        o = self.opt
        X, O = self._stack()
        n = X.shape[0]
        if o["base"] == "eb" and n >= 3:
            mu0, Sig0 = self._eb_gaussian(X, O)
            self.eb_mu, self.eb_Sig = mu0, Sig0
        else:
            mu0, Sig0 = np.zeros(self.D), self.v0 * np.eye(self.D)
        P0 = np.tile(np.linalg.inv(Sig0)[None], (n, 1, 1))
        m, H, f, ev = laplace_fit(X, O, np.tile(mu0, (n, 1)), P0, eps=o["eps"])
        C = np.linalg.inv(H)
        if o["refine"] > 0 and n >= o["refine_min"]:
            if (n - o["refine_min"]) % o["refine_every"] == 0 or getattr(self, "_ref_cache", None) is None:
                mr, Cr = m.copy(), C.copy()
                for _ in range(o["refine"]):
                    mr, Cr = self._refine(X, O, mr, Cr)
                self._ref_cache = (mr, Cr)
            # refined posteriors for the students seen at the last refinement, base posteriors for newer ones
            mr, Cr = self._ref_cache
            k = len(mr)
            m = np.concatenate([mr, m[k:]], axis=0); C = np.concatenate([Cr, C[k:]], axis=0)
        if o["cap"]:
            m = project_ball(m, self.S)
        if n > o["max_comp"]:
            idx = np.sort(self.rng.choice(n, size=o["max_comp"], replace=False))
            m, C = m[idx], C[idx]
        self.mu = m
        self.Sig = self._regularize_cov(C * o["kde_scale"])
        self.pi = np.full(len(m), (1.0 - o["w_iso"]) / len(m))
        self.pi0 = o["w_iso"]
        self.W = None


def _refine(self, X, O, m, C):
    """One leave-one-out NPMLE-style pass: every student's posterior is recomputed under the mixture of
    all OTHER students' current posteriors (plus the isotropic fallback) and collapsed to one Gaussian."""
    from scipy.special import logsumexp
    o = self.opt
    n, b, D = X.shape
    Cs = self._regularize_cov(C)
    means = np.concatenate([np.zeros((1, D)), m], axis=0)                      # (n+1, D) component means
    covs = np.concatenate([(self.v0 * np.eye(D))[None], Cs], axis=0)
    P0 = np.linalg.inv(covs)
    M1 = n + 1
    logpi = np.full(M1, np.log((1.0 - o["w_iso"]) / max(n - 1, 1))); logpi[0] = np.log(o["w_iso"])
    m_new = np.zeros_like(m); C_new = np.zeros_like(C)
    chunk = max(1, int(2e6 // (M1 * b * D)))                                     # students per batch
    for s0 in range(0, n, chunk):
        idx = np.arange(s0, min(n, s0 + chunk)); ns = len(idx)
        Xr = np.repeat(X[idx][:, None], M1, axis=1).reshape(ns * M1, b, D)
        Or = np.repeat(O[idx][:, None], M1, axis=1).reshape(ns * M1, b)
        w, H, f, ev = laplace_fit(Xr, Or, np.tile(means, (ns, 1)), np.tile(P0, (ns, 1, 1)), eps=o["eps"], iters=8)
        w = w.reshape(ns, M1, D); Ci = np.linalg.inv(H).reshape(ns, M1, D, D); ev = ev.reshape(ns, M1)
        lw = logpi[None] + ev
        lw[np.arange(ns), idx + 1] = -np.inf                                      # leave own component out
        lw -= logsumexp(lw, axis=1, keepdims=True)
        r = np.exp(lw)                                                            # (ns, M1)
        mu = np.einsum("sk,skd->sd", r, w)
        dm = w - mu[:, None]
        C_new[idx] = np.einsum("sk,skde->sde", r, Ci + dm[..., :, None] * dm[..., None, :])
        m_new[idx] = mu
    return m_new, C_new


KDEPrior._refine = _refine


def make_kde(spec: str):
    parts = spec.split(":")
    opts = parse_opts(parts[2:], KDEPrior.DEFAULTS)
    return KDEPrior(name=spec, **opts)
