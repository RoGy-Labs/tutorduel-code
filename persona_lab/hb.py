"""Hierarchical-Bayes mixture policy: learns the population of preference vectors from its own
completed sessions and uses it as the prior for every new student.

Population model  p(w) = pi_0 N(0, v0 I) + sum_k pi_k N(mu_k, Sigma_k)
  * component 0 is the no-transfer isotropic prior (fixed; weight floored at w_iso) so that a
    student unlike every learned component falls back to no-transfer behaviour;
  * learned components are fitted by EM on the policy's own click logs, with a Laplace E-step
    (per student x component), M-step on posterior moments, and prune / merge / birth moves.
Within a session: mixture posterior (per-component Laplace + evidence weights) after every click.
Pair rules: emax (two highest posterior-mean utilities), info (emax + info bonus), disagree.
Only Phi (6 x d) and the policy's own clicks are used.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from persona_bench.protocol import Policy

from .infer import Mix, binary_entropy, condition, laplace_fit, probit_click_prob, project_ball, vote_scale


def parse_opts(parts, defaults):
    """'k8', 'rule=info', 'aug=0.5' ... -> dict with types taken from defaults."""
    out = dict(defaults)
    for p in parts:
        if not p:
            continue
        long = {"k": "n_comp", "c": "info_c", "e": "eps", "a": "aug", "r": "rule", "f": "floor",
                "w": "w_iso", "i": "n_init"}
        if "=" in p:
            k, v = p.split("=", 1)
        else:
            k, v = p[0], p[1:]
        k = long.get(k, k)
        if k not in out:
            raise ValueError(f"unknown option {p!r}; known: {sorted(out)}")
        t = type(out[k])
        out[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return out


class HBMix(Policy):
    DEFAULTS = dict(norm=1, n_comp=8, n_init=6, em_every=1, em_late=3, em_late_after=80, em_iters=3,
                    w_iso=0.05, floor=1e-3, rule="emax", aug=0.0, eps=0.0, info_c=0.0,
                    cap=True, merge_tol=0.05, birth=False, prune=True, split=True, split_pen=0.5,
                    sig_cap=1.0, init_k=1, kg_c=1.0, kg_bank=16, adapt=True, warm=0,
                    pts=0.0, cov_c=0.0, explore=0, v0_scale=1.0, cov="full", nu=0.0, n_vs=8,
                    dis_p=0.02, dis_rounds=100000, alpha=0.0, dis_from=0, info_h=True,
                    probe=-1, probe_rounds=2, ts_min=0.0, iso2=0.0,
                    temp=0.05, a_psi=0.0, a_hot=0.0, hyp=False, hyp_every=10, hyp_min=10, hyp_pen=0.0)

    def __init__(self, name="lab:hb", **opts):
        self.name = name
        self.opt = dict(self.DEFAULTS)
        self.opt.update(opts)

    # ---- lifecycle ---------------------------------------------------------------------
    def reset(self, rng, context):
        super().reset(rng, context)
        self.S, self.beta, self.b, self.N = context["S"], context["beta"], context["b"], context["N"]
        o = self.opt
        self.D = self.d + (self.K if (o["aug"] > 0 or o["a_hot"] > 0 or o["hyp"]) else 0)
        # per-dimension feature scales (used only when hyp=True: learned block scales for [psi | onehot])
        self.scale = np.ones(self.D)
        if o["hyp"]:
            a = o["aug"] if o["aug"] > 0 else 1.0
            self.scale = np.concatenate([np.full(self.d, 1.0 / np.sqrt(1 + a * a)), np.full(self.K, a / np.sqrt(1 + a * a))])
        self.hyp_log = []
        self.v0 = o["v0_scale"] * self.S ** 2 / self.d       # (scaled) matched-isotropic variance per dim
        self.iso = Mix.isotropic(self.D, self.v0)
        fw, fm, fC = self._fallback(1.0)
        self.prior = Mix(fw, fm, fC)
        # learned components
        self.mu = np.zeros((0, self.D)); self.Sig = np.zeros((0, self.D, self.D)); self.pi = np.zeros(0)
        self.pi0 = 1.0
        # completed sessions
        self.sess_X = []; self.sess_o = []
        self.W = None                                        # (n, M, D) warm-start modes for the E-step
        self.n_students = 0
        self._last = None
        self._k_hat = -1
        self.bank = []                                       # recent feature matrices (observed data)
        self.G = (self.d / self.S ** 2) * np.eye(self.D)      # pooled Fisher information of logged contrasts
        self.Ginv = np.linalg.inv(self.G)
        self.t_global = 0

    def _aug(self, Phi):
        a = self.opt["aug"]
        if self.opt["hyp"]:                # raw blocks; self.scale is applied where features are used
            return np.concatenate([Phi, np.eye(self.K)], axis=1)
        if self.opt["a_hot"] > 0:          # explicit block scales (no normalisation)
            return np.concatenate([self.opt["a_psi"] * Phi, self.opt["a_hot"] * np.eye(self.K)], axis=1)
        if a <= 0:
            return Phi
        return np.concatenate([Phi, a * np.eye(self.K)], axis=1) / np.sqrt(1.0 + a * a)

    def start_student(self):
        self.X = []; self.o = []
        self.post = self.prior
        if self.opt["pts"] > 0 and self.n_students > 0:
            # population-level Thompson sampling: perturb every component mean by one draw from the
            # pooled-model uncertainty (directions the population log has not constrained stay uncertain)
            L = np.linalg.cholesky(self.Ginv + 1e-10 * np.eye(self.D))
            eps = self.opt["pts"] * (L @ self.rng.normal(size=self.D))
            pr = self.prior
            self.post = Mix(pr.pi.copy(), pr.m + eps[None], pr.C.copy())
            self._sess_prior = self.post
        else:
            self._sess_prior = self.prior
        self.modes = self.post.m.copy()
        self.t = 0

    def _refresh(self):
        if not self.X or not self.opt["adapt"]:
            self.post = self._sess_prior
            return
        self.post, _ = condition(self._sess_prior, np.array(self.X) * self.scale[None], np.array(self.o), w_init=self.modes, eps=self.opt["eps"])
        self.modes = self.post.m

    def update(self, outcome):
        i, j, Phi = self._last
        x = self.beta * (Phi[i] - Phi[j])
        self.X.append(x)
        self.o.append(float(outcome))
        self.t += 1
        if self.opt["pts"] > 0 or self.opt["cov_c"] > 0 or self.opt["explore"] > 0:
            self.G += 0.25 * np.outer(x, x)
            self.Ginv = np.linalg.inv(self.G)
        self.t_global += 1
        self._refresh()

    def end_student(self):
        if self.X:
            self.sess_X.append(np.array(self.X)); self.sess_o.append(np.array(self.o))
            self._k_hat = int(np.argmax(self.post.pi))
        self.n_students += 1
        n = len(self.sess_X)
        o = self.opt
        if o["hyp"] and n >= o["hyp_min"] and (n - o["hyp_min"]) % o["hyp_every"] == 0:
            self._update_scales()
            self._ref_cache = None
        if n >= o["n_init"]:
            every = o["em_every"] if n < o["em_late_after"] else o["em_late"]
            if (n - o["n_init"]) % every == 0:
                self._em()
        self._rebuild_prior()

    def diagnostics(self):
        return {"M": int(len(self.pi)), "pi0": float(self.pi0), "k_hat": self._k_hat,
                "a_psi": float(self.scale[0]), "a_hot": float(self.scale[-1])}

    # ---- population EM -----------------------------------------------------------------
    def _stack(self):
        b = max(len(x) for x in self.sess_X)
        n = len(self.sess_X)
        X = np.zeros((n, b, self.D)); O = np.zeros((n, b))
        for i, (x, oo) in enumerate(zip(self.sess_X, self.sess_o)):
            X[i, :len(x)] = x; O[i, :len(oo)] = oo          # zero rows contribute a constant log(1/2)
        return X * self.scale[None, None, :], O

    def _update_scales(self):
        """Empirical Bayes for the two feature-block scales: maximise the summed Laplace evidence of all
        logged sessions under N(0, v0 I) over a multiplicative grid around the current scales."""
        o = self.opt
        n = len(self.sess_X)
        b = max(len(x) for x in self.sess_X)
        Xr = np.zeros((n, b, self.D)); O = np.zeros((n, b))
        for i, (x, oo) in enumerate(zip(self.sess_X, self.sess_o)):
            Xr[i, :len(x)] = x; O[i, :len(oo)] = oo
        P0 = np.tile((np.eye(self.D) / self.v0)[None], (n, 1, 1)); m0 = np.zeros((n, self.D))
        a0 = 1.0 / np.sqrt(2.0)
        grid = a0 * np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
        best, best_ev = None, -np.inf
        for ap in grid:
            for ah in grid:
                sc = np.concatenate([np.full(self.d, ap), np.full(self.K, ah)])
                _, _, _, ev = laplace_fit(Xr * sc[None, None, :], O, m0, P0, eps=o["eps"], iters=8)
                tot = float(ev.sum()) - o["hyp_pen"] * (np.log(ap / a0) ** 2 + np.log(ah / a0) ** 2)
                if tot > best_ev:
                    best_ev, best = tot, (ap, ah)
        self.scale = np.concatenate([np.full(self.d, best[0]), np.full(self.K, best[1])])
        self.hyp_log.append((n, float(best[0]), float(best[1])))

    def _estep(self, X, O):
        """Laplace fits of every student under every component (iso first). Returns m (n,M+1,D),
        C (n,M+1,D,D), ev (n,M+1)."""
        n = X.shape[0]
        M = len(self.pi)
        means = np.concatenate([np.zeros((1, self.D)), self.mu], axis=0)             # (M+1, D)
        covs = np.concatenate([self.v0 * np.eye(self.D)[None], self.Sig], axis=0)     # (M+1, D, D)
        P0 = np.linalg.inv(covs)
        Xr = np.repeat(X[:, None], M + 1, axis=1).reshape(n * (M + 1), X.shape[1], self.D)
        Or = np.repeat(O[:, None], M + 1, axis=1).reshape(n * (M + 1), X.shape[1])
        m0 = np.tile(means, (n, 1)); P0r = np.tile(P0, (n, 1, 1))
        w0 = None
        if self.W is not None and self.W.shape[0] == n and self.W.shape[1] == M + 1:
            w0 = self.W.reshape(n * (M + 1), self.D)
        w, H, f, ev = laplace_fit(Xr, Or, m0, P0r, w0=w0, eps=self.opt["eps"])
        C = np.linalg.inv(H)
        self.W = w.reshape(n, M + 1, self.D)
        return self.W, C.reshape(n, M + 1, self.D, self.D), ev.reshape(n, M + 1)

    def _resp(self, ev):
        logw = np.log(np.maximum(np.concatenate([[self.pi0], self.pi]), 1e-300))[None] + ev
        logw -= logw.max(axis=1, keepdims=True)
        r = np.exp(logw); r /= r.sum(axis=1, keepdims=True)
        return r

    def _init_components(self, X, O):
        """k-means++ seeds (init_k of them) on isotropic posterior means."""
        n = X.shape[0]
        M0 = max(1, min(self.opt["init_k"], self.opt["n_comp"], n))
        P0 = np.tile((np.eye(self.D) / self.v0)[None], (n, 1, 1))
        m, H, f, ev = laplace_fit(X, O, np.zeros((n, self.D)), P0, eps=self.opt["eps"])
        centres = [m.mean(axis=0)] if M0 == 1 else [m[int(self.rng.integers(n))]]
        for _ in range(1, M0):
            d2 = np.min([np.sum((m - c) ** 2, axis=1) for c in centres], axis=0)
            if d2.sum() <= 0:
                break
            centres.append(m[int(self.rng.choice(n, p=d2 / d2.sum()))])
        self.mu = np.array(centres)
        self.Sig = np.repeat((self.v0 * np.eye(self.D))[None], len(centres), axis=0)
        self.pi = np.full(len(centres), (1.0 - self.opt["w_iso"]) / len(centres))
        self.pi0 = self.opt["w_iso"]
        self.W = None

    def _regularize_cov(self, Sig):
        """Add the floor and cap eigenvalues at sig_cap * S^2."""
        o = self.opt
        Sig = 0.5 * (Sig + np.swapaxes(Sig, -1, -2)) + o["floor"] * np.eye(self.D)
        cap = o["sig_cap"] * self.S ** 2
        ev, V = np.linalg.eigh(Sig)
        ev = np.clip(ev, o["floor"], cap)
        return np.einsum("mij,mj,mkj->mik", V, ev, V)

    def _state(self):
        return (self.mu.copy(), self.Sig.copy(), self.pi.copy(), self.pi0, None if self.W is None else self.W.copy())

    def _set_state(self, st):
        self.mu, self.Sig, self.pi, self.pi0, self.W = st[0].copy(), st[1].copy(), st[2].copy(), st[3], (None if st[4] is None else st[4].copy())

    def _em_iterations(self, X, O, iters, prune=True):
        """EM iterations on the current components; returns total log marginal likelihood."""
        o = self.opt
        n = X.shape[0]
        L = -np.inf
        for _ in range(iters):
            m, C, ev = self._estep(X, O)
            logp = np.log(np.maximum(np.concatenate([[self.pi0], self.pi]), 1e-300))
            L = float(logsumexp(logp[None] + ev, axis=1).sum())
            r = self._resp(ev)
            Nk = r.sum(axis=0)
            M = len(self.pi)
            mu = np.zeros((M, self.D)); Sig = np.zeros((M, self.D, self.D))
            for k in range(M):
                rk = r[:, k + 1]
                nk = max(Nk[k + 1], 1e-9)
                mu[k] = (rk[:, None] * m[:, k + 1]).sum(axis=0) / nk
                dm = m[:, k + 1] - mu[k]
                Sig[k] = (rk[:, None, None] * (C[:, k + 1] + dm[:, :, None] * dm[:, None, :])).sum(axis=0)
                Sig[k] = (Sig[k] + o["nu"] * self.v0 * np.eye(self.D)) / (nk + o["nu"])   # shrink toward the base prior
                if o["cov"] == "diag":
                    Sig[k] = np.diag(np.diag(Sig[k]))
            if o["cap"]:
                mu = project_ball(mu, self.S)
            self.mu, self.Sig = mu, self._regularize_cov(Sig)
            pi = Nk / n
            pi0 = max(pi[0], o["w_iso"])
            self.pi0 = pi0
            self.pi = pi[1:] * (1.0 - pi0) / max(pi[1:].sum(), 1e-12)
            if prune and o["prune"] and len(self.pi) > 1:
                keep = Nk[1:] >= 1.0
                if not keep.any():
                    keep[int(np.argmax(Nk[1:]))] = True
                if not keep.all():
                    self._keep(keep)
        return L

    def _loglik(self, X, O):
        m, C, ev = self._estep(X, O)
        logp = np.log(np.maximum(np.concatenate([[self.pi0], self.pi]), 1e-300))
        return float(logsumexp(logp[None] + ev, axis=1).sum()), m, C, ev

    def _penalty(self, n):
        D = self.D
        return self.opt["split_pen"] * (D + D * (D + 1) / 2 + 1) * np.log(max(n, 2))

    def _em(self):
        o = self.opt
        X, O = self._stack()
        n = X.shape[0]
        if len(self.pi) == 0:
            self._init_components(X, O)
        self._em_iterations(X, O, o["em_iters"])
        if o["split"] and n >= 2 * o["n_init"]:
            L0, m, C, ev = self._loglik(X, O)
            base = self._state()
            accepted = False
            # ---- split proposal: the component with the largest weighted spread
            if len(self.pi) < o["n_comp"]:
                r = self._resp(ev)
                spread = np.array([np.trace(self.Sig[k]) * max(r[:, k + 1].sum(), 1e-9) for k in range(len(self.pi))])
                k = int(np.argmax(spread))
                rk = r[:, k + 1]
                mk = m[:, k + 1]
                # 2-means on the responsible posterior means, seeded along the principal axis
                w_, V_ = np.linalg.eigh(self.Sig[k])
                axis = V_[:, -1] * np.sqrt(max(w_[-1], 1e-9))
                c1, c2 = self.mu[k] + axis, self.mu[k] - axis
                for _ in range(5):
                    d1 = np.sum((mk - c1) ** 2, axis=1); d2 = np.sum((mk - c2) ** 2, axis=1)
                    a = rk * (d1 <= d2); b_ = rk * (d1 > d2)
                    if a.sum() > 1e-6 and b_.sum() > 1e-6:
                        c1 = (a[:, None] * mk).sum(axis=0) / a.sum(); c2 = (b_[:, None] * mk).sum(axis=0) / b_.sum()
                mu = np.concatenate([self.mu, c2[None]], axis=0); mu[k] = c1
                Sig = np.concatenate([self.Sig, self.Sig[k][None]], axis=0)
                pi = np.concatenate([self.pi, [self.pi[k] / 2]]); pi[k] /= 2
                self.mu, self.Sig, self.pi = mu, Sig, pi
                self.W = None
                self._em_iterations(X, O, o["em_iters"], prune=False)
                L1, _, _, _ = self._loglik(X, O)
                if L1 - L0 > self._penalty(n):
                    accepted = True
                else:
                    self._set_state(base)
            # ---- merge proposal: the two closest components (only if no split was accepted)
            if not accepted and len(self.pi) > 1:
                M = len(self.pi)
                best, bd = None, np.inf
                for a in range(M):
                    for c in range(a + 1, M):
                        dd = np.linalg.norm(self.mu[a] - self.mu[c])
                        if dd < bd:
                            bd, best = dd, (a, c)
                a, c = best
                wa, wc = self.pi[a], self.pi[c]
                mu = self.mu.copy(); Sig = self.Sig.copy(); pi = self.pi.copy()
                mu[a] = (wa * mu[a] + wc * mu[c]) / (wa + wc)
                Sig[a] = (wa * Sig[a] + wc * Sig[c]) / (wa + wc)
                pi[a] = wa + wc
                keep = np.ones(M, dtype=bool); keep[c] = False
                self.mu, self.Sig, self.pi = mu[keep], Sig[keep], pi[keep]
                self.W = None
                self._em_iterations(X, O, o["em_iters"], prune=False)
                L1, _, _, _ = self._loglik(X, O)
                if L1 - L0 > -self._penalty(n):
                    pass  # merge accepted (simpler model explains the data nearly as well)
                else:
                    self._set_state(base)
        elif o["merge_tol"] > 0:
            self._merge()

    def _keep(self, keep):
        self.mu, self.Sig, self.pi = self.mu[keep], self.Sig[keep], self.pi[keep]
        self.pi = self.pi * (1.0 - self.pi0) / max(self.pi.sum(), 1e-12)
        if self.W is not None:
            self.W = self.W[:, np.concatenate([[True], keep])]

    def _merge(self):
        tol = self.opt["merge_tol"] * self.S
        changed = True
        while changed and len(self.pi) > 1:
            changed = False
            M = len(self.pi)
            for a in range(M):
                for c in range(a + 1, M):
                    if np.linalg.norm(self.mu[a] - self.mu[c]) < tol:
                        wa, wc = self.pi[a], self.pi[c]
                        self.mu[a] = (wa * self.mu[a] + wc * self.mu[c]) / (wa + wc)
                        self.Sig[a] = (wa * self.Sig[a] + wc * self.Sig[c]) / (wa + wc)
                        self.pi[a] = wa + wc
                        keep = np.ones(M, dtype=bool); keep[c] = False
                        self.mu, self.Sig, self.pi = self.mu[keep], self.Sig[keep], self.pi[keep]
                        if self.W is not None:
                            self.W = self.W[:, np.concatenate([[True], keep])]
                        changed = True
                        break
                if changed:
                    break

    def _fallback(self, weight):
        """Isotropic fallback component(s): optionally a two-scale mixture (heavy-tailed prior)."""
        eye = np.eye(self.D)
        if self.opt["iso2"] > 0:
            return (np.array([0.5 * weight, 0.5 * weight]), np.zeros((2, self.D)),
                    np.stack([self.v0 * eye, self.opt["iso2"] * self.v0 * eye]))
        return np.array([weight]), np.zeros((1, self.D)), (self.v0 * eye)[None]

    def _rebuild_prior(self):
        if len(self.pi) == 0 or len(self.sess_X) < self.opt["warm"]:
            w, m, C = self._fallback(1.0)
            self.prior = Mix(w, m, C)
            return
        pi0 = self.pi0
        if self.opt["alpha"] > 0:
            # Dirichlet-process-style weight of "a student unlike anyone seen so far"
            n = len(self.sess_X)
            pi0 = max(pi0, self.opt["alpha"] / (self.opt["alpha"] + n))
        fw, fm, fC = self._fallback(pi0)
        pi = np.concatenate([fw, self.pi * (1.0 - pi0) / max(self.pi.sum(), 1e-12)])
        m = np.concatenate([fm, self.mu], axis=0)
        C = np.concatenate([fC, self.Sig], axis=0)
        self.prior = Mix(pi / pi.sum(), m, C)

    # ---- pair selection --------------------------------------------------------------
    def _mean_utility(self, Phi):
        return Phi @ self.post.mean()

    def predict_best(self, Phi):
        return int(np.argmax(self._mean_utility(self._aug(Phi) * self.scale[None, :])))

    def _kg_select(self, Phi):
        """Knowledge gradient over the component weights: immediate expected utility of the pair
        plus (remaining rounds) x expected improvement of the best pair value on a bank of recent
        feature matrices after observing the click."""
        post = self.post
        pi, m, C = post.pi, post.m, post.C
        K = self.K
        iu = np.triu_indices(K, 1)
        U = m @ Phi.T                                                   # (M,K)
        ubar = pi @ U
        imm = ubar[iu[0]] + ubar[iu[1]]                                  # (P,)
        remaining = max(self.b - self.t - 1, 0)
        if len(pi) < 2 or remaining == 0 or not self.bank:
            k = int(np.argmax(imm)); return int(iu[0][k]), int(iu[1][k])
        D = Phi[iu[0]] - Phi[iu[1]]
        md = self.beta * (U[:, iu[0]] - U[:, iu[1]])
        vd = self.beta ** 2 * np.einsum("pd,mde,pe->mp", D, C, D)
        p1 = probit_click_prob(md, vd)                                   # (M,P)
        pbar = pi @ p1                                                   # (P,)
        w1 = pi[:, None] * p1 / np.maximum(pbar[None], 1e-12)            # (M,P) weights after y=1
        w0 = pi[:, None] * (1 - p1) / np.maximum(1 - pbar[None], 1e-12)
        B = np.stack(self.bank)                                          # (B,K,D)
        Ub = np.einsum("bkd,md->bmk", B, m)                              # (B,M,K)
        Sb = Ub[:, :, iu[0]] + Ub[:, :, iu[1]]                           # (B,M,P')
        V0 = np.einsum("m,bmq->bq", pi, Sb).max(axis=1).mean()
        V1 = np.einsum("mp,bmq->pbq", w1, Sb).max(axis=2).mean(axis=1)   # (P,)
        V0_ = np.einsum("mp,bmq->pbq", w0, Sb).max(axis=2).mean(axis=1)
        voi = pbar * V1 + (1 - pbar) * V0_ - V0
        score = imm + self.opt["kg_c"] * remaining * np.maximum(voi, 0.0)
        k = int(np.argmax(score))
        return int(iu[0][k]), int(iu[1][k])

    def select(self, Phi):
        Phi_raw = self._aug(Phi)
        Phi = Phi_raw * self.scale[None, :]
        K = self.K
        ubar = self._mean_utility(Phi)
        order = np.argsort(-ubar)
        rule = self.opt["rule"]
        if self.opt["explore"] > 0 and self.t_global < self.opt["explore"]:
            # population-level exploration phase: the least-covered contrast (D-optimal-style)
            iu = np.triu_indices(K, 1)
            D = self.beta * (Phi[iu[0]] - Phi[iu[1]])
            cov = np.einsum("pd,de,pe->p", D, self.Ginv, D)
            k = int(np.argmax(cov))
            i, j = int(iu[0][k]), int(iu[1][k])
        elif rule == "emax" and self.opt["cov_c"] > 0:
            iu = np.triu_indices(K, 1)
            D = self.beta * (Phi[iu[0]] - Phi[iu[1]])
            bonus = np.sqrt(np.maximum(np.einsum("pd,de,pe->p", D, self.Ginv, D), 0.0))
            score = ubar[iu[0]] + ubar[iu[1]] + self.opt["cov_c"] * self.S * bonus
            k = int(np.argmax(score))
            i, j = int(iu[0][k]), int(iu[1][k])
        elif rule == "emax":
            i, j = int(order[0]), int(order[1])
        elif rule == "kg":
            i, j = self._kg_select(Phi)
        elif rule == "vote_probe":
            # DIAGNOSTIC: leader by vote, challenger = a fixed probe persona during the first rounds
            U = self.post.m @ Phi.T
            lo = U.min(axis=1, keepdims=True); hi = U.max(axis=1, keepdims=True)
            Ut = vote_scale(U, lo, hi, self.opt["norm"])
            vbar = self.post.pi @ Ut
            vorder = np.argsort(-vbar)
            i, j = int(vorder[0]), int(vorder[1])
            q = int(self.opt["probe"])
            if self.t < self.opt["probe_rounds"] and q >= 0 and q != i:
                j = q
        elif rule == "vote_info":
            U = self.post.m @ Phi.T
            lo = U.min(axis=1, keepdims=True); hi = U.max(axis=1, keepdims=True)
            Ut = vote_scale(U, lo, hi, self.opt["norm"])
            vbar = self.post.pi @ Ut
            i = int(np.argmax(vbar))
            Dj = Phi[i][None, :] - Phi                                           # (K,D) contrasts leader - j
            md = self.beta * (U[:, [i]] - U)                                      # (M,K)
            vd = self.beta ** 2 * np.einsum("kd,mde,ke->mk", Dj, self.post.C, Dj)
            p1 = probit_click_prob(md, vd)                                        # (M,K) P(click leader | comp)
            pbar = self.post.pi @ p1
            info = binary_entropy(pbar) - self.post.pi @ binary_entropy(p1)       # (K,)
            remaining = max(self.b - self.t - 1, 0)
            c = self.opt["info_c"] * (remaining / self.b if self.opt["info_h"] else 1.0)
            score = vbar + c * info
            score[i] = -np.inf
            j = int(np.argmax(score))
        elif rule in ("fav2", "vote_fav"):
            # posterior probability that each persona is this student's favourite on this question
            U = self.post.m @ Phi.T
            lo = U.min(axis=1, keepdims=True); hi = U.max(axis=1, keepdims=True)
            Ut = vote_scale(U, lo, hi, self.opt["norm"])
            vbar = self.post.pi @ Ut
            opinion = ((hi - lo)[:, 0] > 1e-9)                                  # flat components have no favourite
            pfav = np.bincount(U.argmax(axis=1), weights=self.post.pi * opinion, minlength=K)
            if rule == "fav2":
                forder = np.argsort(-(pfav + 1e-6 * vbar))
                i, j = int(forder[0]), int(forder[1])
            else:
                i = int(np.argmax(vbar))
                pf = pfav + 1e-6 * vbar; pf[i] = -np.inf
                j = int(np.argmax(pf))
        elif rule == "vote_soft":
            # softened vote: Plackett-Luce sample of two personas with probabilities ∝ exp(vote / temp)
            U = self.post.m @ Phi.T
            lo = U.min(axis=1, keepdims=True); hi = U.max(axis=1, keepdims=True)
            Ut = vote_scale(U, lo, hi, self.opt["norm"])
            vbar = self.post.pi @ Ut
            z = (vbar - vbar.max()) / max(self.opt["temp"], 1e-6)
            pr = np.exp(z); pr /= pr.sum()
            i = int(self.rng.choice(K, p=pr))
            pr2 = pr.copy(); pr2[i] = 0.0; pr2 /= pr2.sum()
            j = int(self.rng.choice(K, p=pr2))
        elif rule == "vote_ts":
            # top-two Thompson: leader by vote; challenger = favourite of a component sampled from the
            # current posterior weights (components without an opinion or agreeing with the leader excluded)
            U = self.post.m @ Phi.T
            lo = U.min(axis=1, keepdims=True); hi = U.max(axis=1, keepdims=True)
            Ut = vote_scale(U, lo, hi, self.opt["norm"])
            vbar = self.post.pi @ Ut
            vorder = np.argsort(-vbar)
            i, j = int(vorder[0]), int(vorder[1])
            if self.opt["dis_from"] <= self.t < self.opt["dis_rounds"]:
                fav = Ut.argmax(axis=1)
                ok = ((hi - lo)[:, 0] > 1e-9) & (fav != i)
                w = self.post.pi * ok
                if w.sum() > self.opt["ts_min"]:
                    k = int(self.rng.choice(len(w), p=w / w.sum()))
                    j = int(fav[k])
        elif rule in ("vote", "vote_disagree", "evote", "evote_disagree"):
            # each component votes with its predicted utilities stretched to [0,1] on this question
            # (the benchmark's utility scale), weighted by its posterior probability; evote averages
            # the stretched utilities over posterior samples of the component
            if rule.startswith("evote"):
                Mm = self.post.M
                L = np.linalg.cholesky(self.post.C + 1e-10 * np.eye(self.D))            # (M,D,D)
                eps = self.rng.normal(size=(self.opt["n_vs"], Mm, self.D))
                Th = self.post.m[None] + np.einsum("mij,smj->smi", L, eps)              # (S,M,D)
                Us = np.einsum("smd,kd->smk", Th, Phi)                                  # (S,M,K)
                lo = Us.min(axis=2, keepdims=True); hi = Us.max(axis=2, keepdims=True)
                Ut = vote_scale(Us, lo, hi, self.opt["norm"]).mean(axis=0)              # (M,K)
            else:
                U = self.post.m @ Phi.T                                          # (M,K)
                lo = U.min(axis=1, keepdims=True); hi = U.max(axis=1, keepdims=True)
                Ut = vote_scale(U, lo, hi, self.opt["norm"])
            vbar = self.post.pi @ Ut
            vorder = np.argsort(-vbar)
            i = int(vorder[0]); j = int(vorder[1])
            if rule.endswith("_disagree") and self.opt["dis_from"] <= self.t < self.opt["dis_rounds"]:
                fav = Ut.argmax(axis=1)
                has_opinion = (Ut.max(axis=1) - Ut.min(axis=1)) > (0.5 if self.opt["norm"] else 1e-9)      # flat components have no favourite
                for k in np.argsort(-self.post.pi, kind="stable"):
                    if self.post.pi[k] <= self.opt["dis_p"]:
                        break
                    if has_opinion[k] and fav[k] != i:
                        j = int(fav[k]); break
        elif rule == "disagree":
            i = int(order[0])
            mu = np.einsum("kd,md->mk", Phi, self.post.m)
            fav = mu.argmax(axis=1)
            j = None
            for k in np.argsort(-self.post.pi):
                if fav[k] != i and self.post.pi[k] > 0.02:
                    j = int(fav[k]); break
            if j is None:
                j = int(order[1])
        elif rule == "info":
            iu = np.triu_indices(K, 1)
            mu = np.einsum("kd,md->mk", Phi, self.post.m)                        # (M,K)
            D = Phi[iu[0]] - Phi[iu[1]]                                          # (P,D)
            md = self.beta * (mu[:, iu[0]] - mu[:, iu[1]])                        # (M,P)
            vd = self.beta ** 2 * np.einsum("pd,mde,pe->mp", D, self.post.C, D)  # (M,P)
            p1 = probit_click_prob(md, vd)                                        # (M,P)
            pbar = self.post.pi @ p1
            info = binary_entropy(pbar) - self.post.pi @ binary_entropy(p1)       # (P,)
            remaining = max(self.b - self.t - 1, 0)
            c = self.opt["info_c"] * self.S * remaining / self.b
            score = ubar[iu[0]] + ubar[iu[1]] + c * info
            k = int(np.argmax(score))
            i, j = int(iu[0][k]), int(iu[1][k])
        else:
            raise ValueError(rule)
        self._last = (i, j, Phi_raw)
        if self.opt["kg_bank"] > 0:
            self.bank.append(Phi)
            if len(self.bank) > self.opt["kg_bank"]:
                self.bank.pop(0)
        return i, j


def make_hb(spec: str) -> Policy:
    parts = spec.split(":")
    assert parts[0] == "lab" and parts[1] == "hb", spec
    opts = parse_opts(parts[2:], HBMix.DEFAULTS)
    return HBMix(name=spec, **opts)
