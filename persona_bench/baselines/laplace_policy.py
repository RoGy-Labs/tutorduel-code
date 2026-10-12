"""Laplace baselines: no-transfer, oracle, pooled ridge, CoLSTIM."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import expit

from persona_bench.infer import Mixture, condition, expected_max_pair, fit_laplace
from persona_bench.protocol import Policy


class LaplaceDueling(Policy):
    """Shipped no-transfer and oracle baselines. Not a required template for other policies.

    Within-student Laplace update and pair rules. Population prior is fixed.

    population : "none"    matched-isotropic prior N(0, (S^2/d) I) (no transfer; the claim competitor)
                 "oracle"  true population law (reference only)
    rule       : "greedy_ts" | "tts" | "emax" | "ids" | "ids_w" | "ucb" | "greedy_disagree" | "greedy_emax"
    adapt      : update the student's posterior with their own clicks (False = frozen prior)
    future     : coefficient of a population-coverage exploration bonus (0 = off)
    """

    def __init__(self, population: str = "none", rule: str = "greedy_ts", adapt: bool = True,
                 future: float = 0.0, n_mc: int = 256, ts_tries: int = 8,
                 ucb_c: float = 2.0, name: Optional[str] = None,
                 atom_var: float = 1e-4, random_ties: bool = False):
        self.population, self.rule, self.adapt, self.future = population, rule, adapt, future
        self.n_mc, self.ts_tries, self.ucb_c = n_mc, ts_tries, ucb_c
        self.atom_var = atom_var
        self.random_ties = random_ties      # spec token "rt": exact ties in posterior-mean utility are broken at random
        self.name = name or f"{population}-{rule}" + ("" if adapt else "-noadapt") + (f"-fut{future}" if future else "")

    # ---- lifecycle ---------------------------------------------------------------------
    def reset(self, rng, context):
        super().reset(rng, context)
        self.S, self.beta, self.b = context["S"], context["beta"], context["b"]
        self.feedback = context.get("feedback", "ab")
        self.lik = "gauss" if self.feedback == "rating" else "logit"
        self.tau = context.get("rating_noise", 0.1) * (2 * self.S if self.feedback == "rating" else 1.0)
        base = Mixture.isotropic(self.d, self.S**2 / self.d)
        if self.population == "oracle":
            orc = context["oracle"]
            if orc["kind"] == "types":
                self.prior = Mixture.atoms(orc["atoms"], orc["weights"], var=self.atom_var)
            else:
                self.prior = Mixture.gaussian(orc["mu"], orc["Sigma"])
        else:
            self.prior = base
        self.base = base
        self.log_X: list[np.ndarray] = []   # per past student: (b, d) scaled contrasts
        self.log_o: list[np.ndarray] = []
        self.G = np.eye(self.d)               # coverage matrix of logged contrasts (for `future`)
        self.n_students = 0

    def start_student(self):
        self.X: list[np.ndarray] = []
        self.o: list[float] = []
        self.post = self.prior
        self.modes = self.prior.m.copy()
        self._last = None

    def _refresh_posterior(self):
        if not self.adapt or len(self.X) == 0:
            self.post = self.prior
            return
        Xs = np.array(self.X)
        o = np.array(self.o)
        self.post, _ = condition(self.prior, Xs, o, w_init=self.modes, lik=self.lik, tau=self.tau)
        self.modes = self.post.m

    def update(self, outcome):
        i, j, Phi = self._last
        x = self.beta * (Phi[i] - Phi[j])
        self.X.append(x)
        self.o.append(float(outcome))
        self._refresh_posterior()

    def end_student(self):
        if len(self.X) > 0:
            self.log_X.append(np.array(self.X))
            self.log_o.append(np.array(self.o))
            X = np.array(self.X)
            self.G += X.T @ X
        self.n_students += 1

    def diagnostics(self):
        return {"prior_M": int(self.prior.M)}

    # ---- posterior summaries ---------------------------------------------------------
    def _moments(self, Phi):
        """Per-component utility means (M,K) and difference variances (M,K,K)."""
        mu = np.einsum("kd,md->mk", Phi, self.post.m)
        D = Phi[:, None, :] - Phi[None, :, :]                       # (K,K,d)
        var = np.einsum("ijd,mde,ije->mij", D, self.post.C, D)      # (M,K,K)
        return mu, var

    def _mean_utility(self, Phi):
        return np.einsum("m,mk->k", self.post.pi, np.einsum("kd,md->mk", Phi, self.post.m))

    def predict_best(self, Phi):
        return int(np.argmax(self._mean_utility(Phi)))

    def _favourite_of_samples(self, Phi, n):
        W = self.post.sample(self.rng, n)
        return np.argmax(Phi @ W.T, axis=0)

    # ---- pair rules ------------------------------------------------------------------
    def select(self, Phi):
        rule = self.rule
        K = self.K
        ubar = self._mean_utility(Phi)
        if self.random_ties:                # only exact ties are affected (e.g. the flat prior of the first round)
            order = np.lexsort((self.rng.random(K), -ubar))
        else:
            order = np.argsort(-ubar)
        if rule in ("greedy_ts", "greedy_disagree", "ucb", "greedy_emax"):
            i = int(order[0])
            j = None
            if rule == "greedy_ts":
                favs = self._favourite_of_samples(Phi, self.ts_tries)
                diff = favs[favs != i]
                j = int(diff[0]) if len(diff) else None
            elif rule == "greedy_disagree":
                # favourite of the most probable component that disagrees with slot 1 (Type-Elim)
                mu, _ = self._moments(Phi)
                comp_fav = mu.argmax(axis=1)
                for k in np.argsort(-self.post.pi):
                    if comp_fav[k] != i:
                        j = int(comp_fav[k]); break
                if j is None:  # every component agrees: Thompson challenger
                    favs = self._favourite_of_samples(Phi, self.ts_tries)
                    diff = favs[favs != i]
                    j = int(diff[0]) if len(diff) else None
            elif rule == "ucb":
                mu, var = self._moments(Phi)
                mean_diff = np.einsum("m,mk->k", self.post.pi, mu - mu[:, i][:, None])
                tot_var = np.einsum("m,mk->k", self.post.pi, var[:, :, i] + (mu - mu[:, i][:, None] - mean_diff[None]) ** 2)
                score = mean_diff + self.ucb_c * np.sqrt(np.maximum(tot_var, 0))
                score[i] = -np.inf
                j = int(np.argmax(score))
            elif rule == "greedy_emax":
                j = int(order[1])
            if j is None or j == i:
                j = int(order[1])
        elif rule == "tts":
            favs = self._favourite_of_samples(Phi, 1 + self.ts_tries)
            i = int(favs[0])
            diff = favs[1:][favs[1:] != i]
            j = int(diff[0]) if len(diff) else int(order[0] if order[0] != i else order[1])
        elif rule == "emax":
            # Strong-regret pair rule: maximise E[U_i + U_j] = two highest posterior means.
            i, j = int(order[0]), int(order[1])
            if self.future > 0:
                iu = np.triu_indices(K, 1)
                score = ubar[iu[0]] + ubar[iu[1]] + self.future * self._coverage_bonus(Phi, iu)
                k = int(np.argmax(score))
                i, j = int(iu[0][k]), int(iu[1][k])
        elif rule in ("ids", "ids_w", "ids_lin"):
            i, j = self._ids(Phi, about_w=(rule == "ids_w"), power=(1.0 if rule == "ids_lin" else 2.0))
        elif rule == "ids_h":
            # horizon-aware: information has no value in the last round(s) of a session
            if len(self.X) >= self.b - 1:
                mu, var = self._moments(Phi)
                iu = np.triu_indices(K, 1)
                score = np.einsum("m,mp->p", self.post.pi, expected_max_pair(mu[:, iu[0]], mu[:, iu[1]], var[:, iu[0], iu[1]]))
                k = int(np.argmax(score)); i, j = int(iu[0][k]), int(iu[1][k])
            else:
                i, j = self._ids(Phi)
        else:
            raise ValueError(rule)
        if self.future > 0 and rule != "emax":
            # add coverage exploration by possibly swapping the challenger
            pass
        self._last = (i, j, Phi)
        return i, j

    def _coverage_bonus(self, Phi, iu):
        Ginv = np.linalg.inv(self.G)
        D = Phi[iu[0]] - Phi[iu[1]]
        return np.sqrt(np.einsum("pd,de,pe->p", D, Ginv, D))

    def _ids(self, Phi, about_w=False, power=2.0):
        """Information-directed sampling on pairs: minimise (expected weak regret)^2 / info,
        where info is the mutual information between the click and the identity of the best
        persona (favourite-information IDS) or between the click and w (ids_w)."""
        K = self.K
        W = self.post.sample(self.rng, self.n_mc)                   # (S,d)
        U = W @ Phi.T                                               # (S,K)
        best = U.argmax(axis=1)
        Umax = U.max(axis=1)
        iu = np.triu_indices(K, 1)
        Ui, Uj = U[:, iu[0]], U[:, iu[1]]                           # (S,P)
        regret = (Umax[:, None] - np.maximum(Ui, Uj)).mean(axis=0)  # (P,)
        p1 = expit(self.beta * (Ui - Uj))                           # P(o=1 | w_s)
        H = lambda p: -(p * np.log(np.maximum(p, 1e-12)) + (1 - p) * np.log(np.maximum(1 - p, 1e-12)))
        Ho = H(p1.mean(axis=0))
        if about_w:
            info = Ho - H(p1).mean(axis=0)
        else:
            info = Ho.copy()
            for p in np.unique(best):
                mask = best == p
                frac = mask.mean()
                info -= frac * H(p1[mask].mean(axis=0))
        info = np.maximum(info, 1e-9)
        ratio = regret**power / info
        k = int(np.argmin(ratio))
        return int(iu[0][k]), int(iu[1][k])

    # ---- rating (numerical feedback) mode -------------------------------------------
    def select_single(self, Phi):
        # Thompson sampling on utilities, with the same posterior machinery
        w = self.post.sample(self.rng, 1)[0]
        p = int(np.argmax(Phi @ w))
        self._last_single = (p, Phi)
        return p

    def update_rating(self, y):
        p, Phi = self._last_single
        # y = 0.5 + phi.w / (2S) + noise  ->  target (y - 0.5) * 2S = phi . w
        self.X.append(Phi[p].copy())
        self.o.append((y - 0.5) * 2 * self.S)
        self._refresh_posterior()



class PooledW(Policy):
    """One ridge logistic vector on all completed sessions (RLHF default).
    The current student is served from that point estimate with the two highest
    expected utilities (strong-regret pair) and no within-student update."""

    name = "pooled"

    def reset(self, rng, context):
        super().reset(rng, context)
        self.S, self.beta = context["S"], context["beta"]
        self.lam = self.d / max(self.S ** 2, 1e-9)
        self.w = np.zeros(self.d)
        self.log_X: list[np.ndarray] = []
        self.log_o: list[np.ndarray] = []
        self._sess_X: list[np.ndarray] = []
        self._sess_o: list[float] = []
        self._last = None

    def start_student(self):
        self._sess_X, self._sess_o, self._last = [], [], None

    def select(self, Phi):
        V = Phi @ self.w
        iu = np.triu_indices(self.K, 1)
        k = int(np.argmax(V[iu[0]] + V[iu[1]]))
        i, j = int(iu[0][k]), int(iu[1][k])
        self._last = (i, j, Phi)
        return i, j

    def update(self, outcome):
        i, j, Phi = self._last
        self._sess_X.append(self.beta * (Phi[i] - Phi[j]))
        self._sess_o.append(float(outcome))

    def end_student(self):
        if not self._sess_X:
            return
        self.log_X.append(np.array(self._sess_X))
        self.log_o.append(np.array(self._sess_o))
        X = np.concatenate(self.log_X)
        o = np.concatenate(self.log_o)
        m0 = np.zeros((1, self.d))
        P0 = (self.lam * np.eye(self.d))[None]
        w, _, _, _ = fit_laplace(X[None], o[None], m0, P0, w0=self.w[None])
        self.w = w[0]

    def predict_best(self, Phi):
        return int(np.argmax(Phi @ self.w))



class ColstimDueling(Policy):
    """Per-student linear contextual dueling baseline in the style of CoLSTIM (Bengs, Saha and
    Huellermeier, 2022): a ridge-regularised logistic maximum-likelihood estimate of the current
    student's vector, a leader that maximises the estimated utility, and a challenger that
    maximises an optimistic estimate of the utility difference, with confidence width
    c * ||x_j - x_i||_{V^{-1}} where V is the regularised design matrix of the scaled contrasts.
    With g > 0 the leader is chosen as in CoLSTIM's Algorithm 1 (lines 7-14): each arm's estimated utility is
    perturbed by g * eps_i * ||x_i||_{V^{-1}}, eps_i i.i.d. standard Gumbel trimmed to [-cth, cth] (CoLSTIM's
    C_thresh). CoLSTIM's coupling probability p_t = min(1, d log(dT) / sqrt(t - tau)) is 1 for every round of a
    session of 24 questions with d >= 7, so the perturbations are always drawn independently here. g = 0 is the
    unperturbed leader (not used in the paper). No information is carried across students."""
    name = "colstim"

    def __init__(self, c: float = 1.0, name: Optional[str] = None, lam_mult: float = 1.0, g: float = 0.0, cth: float = 3.0):
        self.c, self.lam_mult, self.g, self.cth = c, lam_mult, g, cth
        self.name = name or "colstim"

    def reset(self, rng, context):
        super().reset(rng, context)
        self.S, self.beta = context["S"], context["beta"]
        self.lam = self.lam_mult * self.d / self.S**2     # matched-isotropic ridge (x lam_mult), as for the no-transfer prior

    def start_student(self):
        self.X: list[np.ndarray] = []
        self.o: list[float] = []
        self.theta = np.zeros(self.d)
        self.V = self.lam * np.eye(self.d)
        self._last = None

    def _refit(self):
        Xs = np.array(self.X)[None]
        o = np.array(self.o)[None]
        m0 = np.zeros((1, self.d))
        P0 = (self.lam * np.eye(self.d))[None]
        w, _, _, _ = fit_laplace(Xs, o, m0, P0, w0=self.theta[None])
        self.theta = w[0]

    def select(self, Phi):
        u = Phi @ self.theta
        Vinv = np.linalg.inv(self.V)
        if self.g > 0:                                                    # CoLSTIM leader: trimmed Gumbel perturbation x confidence width
            X = self.beta * Phi
            w = np.sqrt(np.maximum(np.einsum("kd,de,ke->k", X, Vinv, X), 0.0))
            eps = np.clip(self.rng.gumbel(size=self.K), -self.cth, self.cth)
            u = u + self.g * eps * w
        i = int(np.argmax(u))
        Dx = self.beta * (Phi - Phi[i])                                   # scaled contrasts x_j - x_i
        width = np.sqrt(np.maximum(np.einsum("kd,de,ke->k", Dx, Vinv, Dx), 0.0))
        score = Dx @ self.theta + self.c * width
        score[i] = -np.inf
        j = int(np.argmax(score))
        self._last = (i, j, Phi)
        return i, j

    def update(self, outcome):
        i, j, Phi = self._last
        x = self.beta * (Phi[i] - Phi[j])
        self.X.append(x)
        self.o.append(float(outcome))
        self.V += np.outer(x, x)
        self._refit()

    def predict_best(self, Phi):
        return int(np.argmax(Phi @ self.theta))

