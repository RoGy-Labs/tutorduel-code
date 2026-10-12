"""COLDB, Clustering Of Linear Dueling Bandits (Wang, Sun, Kong, Xie, Hu, Lui and Dai, ICML 2025),
ported from Algorithm 1 of the paper (no code is released) to the sequential-user protocol of this
benchmark, and a one-line variant for users who are seen for a single session.

Faithful port (`ColdbPolicy`, spec "coldb[:aX][:cY]"):
  * per-user ridge-regularised logistic MLE  theta_hat_i  on the user's own contrasts (Line 11);
  * a graph over users, complete when a user first arrives (Line 2); the cluster of the current user
    is its connected component (Line 5);
  * cluster MLE theta_t on the pooled data of the component (Line 6, eq. 6) and aggregated
    information matrix V_{t-1} = V0 + sum of contrast outer products over the component (Line 7);
  * first arm  argmax theta_t . phi(x)  (Line 8);
  * second arm argmax_{x != x1} theta_t . (phi(x) - phi(x1)) + (beta_t / kappa_mu) ||phi(x) - phi(x1)||_{V^-1}
    with beta_t = sqrt(2 log(1/delta) + d log(1 + t L^2 kappa_mu / (d lambda)))  (Line 9);
  * after the click, the served user's estimate is refit and every edge (i_t, l) with
    ||theta_hat_{i_t} - theta_hat_l|| > alpha (f(T_i) + f(T_l)) is deleted (Line 12), with
    f(T) = (sqrt(lambda/kappa_mu) + sqrt(2 log(u/delta) + d log(1 + 4 T kappa_mu / (d lambda)))) / (kappa_mu sqrt(2 lambda_x T)).
  Constants: kappa_mu = 1/4 (logistic), delta = 0.05, lambda = d/S^2 (norm-matched ridge; ':lam1' gives the paper's 1), L = 2 (||x|| <= 2 for a contrast of two
  unit vectors), lambda_x = 1/d (isotropic unit features), u = number of users of the run. `alpha`
  scales the deletion threshold (alpha = 1 is the paper's constant; the clustering-of-bandits
  literature the paper follows tunes it) and `c` scales the exploration width (c = 1 is the paper's).
  In this benchmark contrasts are scaled by beta (x = beta (phi_i - phi_j)) so that theta estimates w.

Merge variant (spec "coldbm[:aX][:cY]"): a new user starts isolated (own component), is served by the
pooled model of all users only while it has no data, and after every click edges are ADDED to every
component whose pooled estimate passes the centred test and REMOVED where it fails; a mistaken merge
is thus undone as in COLDB, and a user is never pooled with a component that its own data reject.

Likelihood-ratio variant (spec "coldbl[:tX][:cY]"): as the merge variant, but the test is a
likelihood ratio calibrated for few clicks instead of a distance with the 1/sqrt(T) radius: after every
click the user is connected to the single component C whose pooled estimate (own data excluded)
predicts the user's own clicks best, provided
   LL_i(theta_hat_i) - LL_i(theta_C) <= tol * (d/2) * (1 + log(1 + T_i)),
and disconnected from every component that fails this test; the tolerance `tol` (default 1) plays
the role of alpha. This keeps COLDB's pooled MLE, pair rule and undoable merges, and replaces only
the edge test.

Variant for one-session users (`ColdbCentredPolicy`, spec "coldbc[:aX][:cY]"): identical except that
the edge test compares the served user's estimate with the pooled estimate of the other user's
component (the served user's own data excluded), theta_hat_{C(l) without i}, and the radius of that pooled
estimate uses the component's number of rounds: an edge is deleted when
||theta_hat_i - theta_hat_{C(l) without i}|| > alpha (f(T_i) + f(T_{C(l) without i})). With one short session per user
the individual estimates of two users are both noisy, whereas a component's pooled estimate is not.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from persona_bench.infer import fit_laplace
from persona_bench.protocol import Policy


def _f_radius(T: int, d: int, u: int, kappa: float, lam: float, lam_x: float, delta: float) -> float:
    if T <= 0:
        return np.inf
    num = np.sqrt(lam / kappa) + np.sqrt(2 * np.log(u / delta) + d * np.log(1 + 4 * T * kappa / (d * lam)))
    return num / (kappa * np.sqrt(2 * lam_x * T))


class ColdbPolicy(Policy):
    name = "coldb"

    def __init__(self, alpha: float = 1.0, c: float = 1.0, name: Optional[str] = None, centred: bool = False,
                 lam: Optional[float] = None, merge: bool = False, lr: bool = False, tol: float = 1.0):
        self.alpha, self.c, self.centred, self.lam_in = alpha, c, centred or merge or lr, lam
        self.merge, self.lr, self.tol = merge or lr, lr, tol
        self.name = name or ("coldbl" if lr else "coldbm" if merge else "coldbc" if centred else "coldb")

    # ---- lifecycle -------------------------------------------------------------------
    def reset(self, rng, context):
        super().reset(rng, context)
        self.S, self.beta, self.u = context["S"], context["beta"], int(context["N"])
        self.kappa, self.delta, self.L = 0.25, 0.05, 2.0
        # regularisation input lambda of Algorithm 1: matched to the norm bound (d / S^2), as for every
        # other baseline of the benchmark; lambda = 1 inflates the per-user MLE on separable sessions
        self.lam = (self.d / self.S**2) if self.lam_in is None else self.lam_in
        self.lam_x = 1.0 / self.d
        self.X: list[list[np.ndarray]] = []       # per user: contrasts
        self.o: list[list[float]] = []
        self.theta_hat: list[np.ndarray] = []     # per-user MLE
        self.adj: np.ndarray = np.zeros((0, 0), dtype=bool)
        self.t_global = 0
        self.cur = -1
        self._last = None
        self._theta_c = np.zeros(self.d)
        self._Vinv = np.eye(self.d) * self.kappa / self.lam
        self._comp_cache = None

    def _mle(self, Xs, os, w0=None):
        if len(Xs) == 0:
            return np.zeros(self.d)
        X = np.array(Xs)[None]
        o = np.array(os)[None]
        w, _, _, _ = fit_laplace(X, o, np.zeros((1, self.d)), (self.lam * np.eye(self.d))[None],
                                 w0=None if w0 is None else w0[None])
        return w[0]

    def start_student(self):
        n = len(self.X)
        # new user joins with edges to every existing user (complete graph at initialisation)
        adj = np.ones((n + 1, n + 1), dtype=bool)
        adj[:n, :n] = self.adj
        np.fill_diagonal(adj, False)
        if self.merge:                       # merge variant: the new user starts in its own component
            adj[n, :] = False
            adj[:, n] = False
        self.adj = adj
        self.X.append([]); self.o.append([]); self.theta_hat.append(np.zeros(self.d))
        self.cur = n
        self._comp_cache = None

    def _component(self, i):
        n = len(self.X)
        if n == 1:
            return np.array([0])
        _, lab = connected_components(csr_matrix(self.adj), directed=False)
        return np.nonzero(lab == lab[i])[0]

    def _pooled(self, members, exclude=None, w0=None):
        Xs, os = [], []
        T = 0
        for m in members:
            if m == exclude:
                continue
            Xs += self.X[m]; os += self.o[m]; T += len(self.X[m])
        return self._mle(Xs, os, w0), Xs, T

    # ---- pair selection (Lines 5-9) --------------------------------------------------
    def select(self, Phi):
        i = self.cur
        members = self._component(i)
        if self.merge and len(self.X[i]) == 0:
            members = np.arange(len(self.X))     # no data yet: pooled model of everyone, as in COLDB
        theta_c, Xs, _ = self._pooled(members, w0=self._theta_c)
        self._theta_c = theta_c
        V = self.lam / self.kappa * np.eye(self.d)
        if Xs:
            Xa = np.array(Xs)
            V = V + Xa.T @ Xa
        Vinv = np.linalg.inv(V)
        self.t_global += 1
        beta_t = np.sqrt(2 * np.log(1 / self.delta) + self.d * np.log(1 + self.t_global * self.L**2 * self.kappa / (self.d * self.lam)))
        u = Phi @ theta_c
        x1 = int(np.argmax(u))
        D = self.beta * (Phi - Phi[x1])
        width = np.sqrt(np.maximum(np.einsum("kd,de,ke->k", D, Vinv, D), 0.0))
        score = D @ theta_c + self.c * (beta_t / self.kappa) * width
        score[x1] = -np.inf
        x2 = int(np.argmax(score))
        self._last = (x1, x2, Phi)
        return x1, x2

    # ---- feedback, per-user MLE and edge deletion (Lines 10-12) ------------------------
    def update(self, outcome):
        x1, x2, Phi = self._last
        i = self.cur
        x = self.beta * (Phi[x1] - Phi[x2])
        self.X[i].append(x); self.o[i].append(float(outcome))
        self.theta_hat[i] = self._mle(self.X[i], self.o[i], w0=self.theta_hat[i])
        Ti = len(self.X[i])
        fi = _f_radius(Ti, self.d, self.u, self.kappa, self.lam, self.lam_x, self.delta)
        n = len(self.X)
        if self.lr:
            # likelihood-ratio test of the user's own clicks under each component's pooled estimate
            Xi, oi = np.array(self.X[i]), np.array(self.o[i])
            def ll(th):
                z = Xi @ th
                return float(-(np.logaddexp(0.0, -z) * oi + np.logaddexp(0.0, z) * (1 - oi)).sum())
            ll_own = ll(self.theta_hat[i])
            budget = self.tol * (self.d / 2.0) * (1.0 + np.log(1.0 + Ti))
            _, lab = connected_components(csr_matrix(self.adj), directed=False)
            best, best_ll = None, -np.inf
            for c in np.unique(lab):
                members = np.nonzero(lab == c)[0]
                others = [m for m in members if m != i]
                if not others:
                    continue
                th, _, T = self._pooled(members, exclude=i)
                if T == 0:
                    continue
                l = ll(th)
                if ll_own - l <= budget and l > best_ll:
                    best, best_ll = c, l
            for c in np.unique(lab):
                others = [m for m in np.nonzero(lab == c)[0] if m != i]
                for l_ in others:
                    self.adj[i, l_] = self.adj[l_, i] = (c == best)
        elif self.merge:
            # test against every other component's pooled estimate; connect to those that pass, cut those that fail
            _, lab = connected_components(csr_matrix(self.adj), directed=False)
            for c in np.unique(lab):
                members = np.nonzero(lab == c)[0]
                others = [m for m in members if m != i]
                if not others:
                    continue
                th, _, T = self._pooled(members, exclude=i)
                if T == 0:
                    continue
                fl = _f_radius(T, self.d, self.u, self.kappa, self.lam, self.lam_x, self.delta)
                passes = np.linalg.norm(self.theta_hat[i] - th) <= self.alpha * (fi + fl)
                for l in others:
                    self.adj[i, l] = self.adj[l, i] = passes
        elif self.centred:
            # compare with the pooled estimate of each neighbour's component (own data excluded)
            _, lab = connected_components(csr_matrix(self.adj), directed=False)
            cache = {}
            for l in np.nonzero(self.adj[i])[0]:
                if lab[l] not in cache:
                    members = np.nonzero(lab == lab[l])[0]
                    th, _, T = self._pooled(members, exclude=i)
                    cache[lab[l]] = (th, T)
                th, T = cache[lab[l]]
                if T == 0:
                    continue
                fl = _f_radius(T, self.d, self.u, self.kappa, self.lam, self.lam_x, self.delta)
                if np.linalg.norm(self.theta_hat[i] - th) > self.alpha * (fi + fl):
                    self.adj[i, l] = self.adj[l, i] = False
        else:
            for l in np.nonzero(self.adj[i])[0]:
                Tl = len(self.X[l])
                if Tl == 0:
                    continue
                fl = _f_radius(Tl, self.d, self.u, self.kappa, self.lam, self.lam_x, self.delta)
                if np.linalg.norm(self.theta_hat[i] - self.theta_hat[l]) > self.alpha * (fi + fl):
                    self.adj[i, l] = self.adj[l, i] = False

    def predict_best(self, Phi):
        return int(np.argmax(Phi @ self._theta_c))

    def diagnostics(self):
        n = len(self.X)
        if n < 2:
            return {"n_clusters": n}
        ncomp, _ = connected_components(csr_matrix(self.adj), directed=False)
        return {"n_clusters": int(ncomp)}


def make_coldb(spec: str) -> Policy:
    """'coldb', 'coldb:a0.05', 'coldb:a0.05:c0.5', 'coldbc:a0.1' ..."""
    parts = spec.split(":")
    centred = parts[0] == "coldbc"
    merge = parts[0] == "coldbm"
    lr = parts[0] == "coldbl"
    alpha, c, lam, tol = 1.0, 1.0, None, 1.0
    for p in parts[1:]:
        if p.startswith("lam"):
            lam = float(p[3:])
        elif p.startswith("a"):
            alpha = float(p[1:])
        elif p.startswith("c"):
            c = float(p[1:])
        elif p.startswith("t"):
            tol = float(p[1:])
        else:
            raise ValueError(p)
    return ColdbPolicy(alpha=alpha, c=c, name=spec, centred=centred, lam=lam, merge=merge, lr=lr, tol=tol)
