"""Population mixture with births + identification-aware pair rules (CODuel's population learner).

Population p(w) = pi_0 N(0, v0 I) + sum_k pi_k N(mu_k, Sigma_k) over w in the augmented feature space
[psi | aug * onehot(persona)] / sqrt(1 + aug^2)  (the persona index is an admissible feature).

Learning (own completed sessions only): EM with a Laplace E-step per (student, component) and a moment M-step
(inherited from persona_lab.hb.HBMix), started from one empirical-Bayes Gaussian, with
  * component covariances capped at `cap_v0` times the no-transfer prior variance per direction, so a learned
    component can never be broader than the fallback (a broad Gaussian would otherwise swallow every student), and
  * BIRTHS instead of the BIC split test: after EM, a student whose clicks are explained better by the isotropic
    fallback than by every learned component (Laplace evidence gap > `birth_margin` nats) seeds a new component
    at their own posterior. Components with responsibility mass < 1 are pruned.

Within a session: mixture posterior (per-component Laplace + evidence weights) after every click.

Pair rules (strong regret charges BOTH shown answers, so information about the type must be paid for):
  emax     two highest posterior-mean utilities (myopic)
  kg       one-step lookahead: expected regret now + (remaining rounds) x expected regret of the greedy pair after
           observing the click, averaged over a bank of recent questions (analytic, components frozen)
  rollout  Monte Carlo one-step lookahead with a greedy continuation that keeps learning: sample the truth from the
           belief, show the candidate pair, then play greedy on future questions from the bank, accumulating the
           predicted strong regret; common random numbers across candidates. Components frozen inside the rollout.
  lts      latent-bandit Thompson sampling (Maillard and Mannor 2014; Hong et al. 2020, mTS): draw one component from
           the belief weights and show its greedy pair (the two highest utilities under that component's posterior)
  lucb     latent-bandit UCB (Hong et al. 2020, mUCB): the plausible components are those whose belief weight is within
           a factor exp(-lucb_c) of the largest (a log-likelihood-ratio confidence set, lucb_c = log(1/delta)); show
           the pair with the lowest strong regret under the most optimistic plausible component
Regret inside the rules is measured in predicted-utility units (linear in w), i.e. the v2 objective up to a constant.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.special import expit

from persona_lab.hb import HBMix
from persona_lab.infer import Mix, condition, laplace_fit, probit_click_prob
from persona_lab.kde import KDEPrior


class MixV3(KDEPrior):
    """pop=em: EM mixture with births (this module); pop=kde: kernel components (one per past student, persona_lab.kde)."""
    DEFAULTS = dict(KDEPrior.DEFAULTS)
    DEFAULTS.update(dict(pop="em", base="eb", kde_scale=2.0, aug=1.0, alpha=2.0, n_comp=12, birth_margin=1.0, births=1, cap_v0=1.0, nu=1.0, em_iters=3,
                         n_init=4, em_every=1, em_late=3, em_late_after=80, split=False, merge_tol=0.0,
                         rule="rollout", n_roll=64, bank=16, top_m=32, h=1.0, lucb_c=3.0, norm=0, w_iso=0.05, mstep="moment", comp_var=0.05, pool_ridge=1.0, pool_iters=8, rec0=0.0, rec_use=False, mass_min=0.0, floor_v0=0.0))

    def reset(self, rng, context):
        super().reset(rng, context)
        o = self.opt
        # eigenvalue cap of learned covariances = cap_v0 * v0 (hb expresses the cap as sig_cap * S^2); eigenvalue floor
        # = floor_v0 * v0: with floor_v0 = 1 no component is narrower than the no-transfer prior in any direction, so an
        # unstructured population gives back the isotropic prior (shifted by a near-zero population mean)
        o["sig_cap"] = o["cap_v0"] * self.v0 / (self.S ** 2)
        if o["floor_v0"] > 0:
            o["floor"] = o["floor_v0"] * self.v0
            o["sig_cap"] = max(o["sig_cap"], o["floor_v0"] * self.v0 / (self.S ** 2))
        self.pairs = np.array(list(combinations(range(self.K), 2)))
        self.fbank: list[np.ndarray] = []          # recent scaled feature matrices (exogenous questions)
        self.n_births = 0
        self.n_em = 0
        self.rec = np.zeros(0)                      # per component: prequential log-evidence record vs the fallback
        self._sess_ev = None                        # evidence per prior component on the current session's clicks
        self.mass = 0.0                             # trusted population mass in the current prior

    def _refresh(self):
        if not self.X or not self.opt["adapt"]:
            self.post = self._sess_prior
            self._sess_ev = None
            return
        self.post, ev = condition(self._sess_prior, np.array(self.X) * self.scale[None], np.array(self.o), w_init=self.modes, eps=self.opt["eps"])
        self.modes = self.post.m
        self._sess_ev = ev

    def end_student(self):
        # prequential record: the evidence each learned component (fixed before this session) earned on this session's
        # clicks, relative to the isotropic fallback. Updated before EM changes the components.
        if self._sess_ev is not None and len(self.rec) and len(self._sess_ev) == len(self.rec) + 1:
            # responsibility-weighted: only the sessions a component claims count for or against it
            self.rec = self.rec + self.post.pi[1:] * (self._sess_ev[1:] - self._sess_ev[0])
        self._sess_ev = None
        super().end_student()

    def _keep(self, keep):
        super()._keep(keep)
        if len(self.rec) == len(keep):
            self.rec = self.rec[keep]

    def _rebuild_prior(self):
        super()._rebuild_prior()
        M = len(self.pi)
        if len(self.rec) != M:                       # components were rebuilt (kde) or initialised: fresh records
            self.rec = np.zeros(M)
        if not self.opt["rec_use"] or M == 0 or self.prior.M != M + 1:
            self.mass = float(1.0 - self.prior.pi[0]) if M else 0.0
            return
        # reliability = posterior probability that the component is real, from its record (log Bayes factor)
        rel = expit(self.rec - self.opt["rec0"])
        pi = self.prior.pi.copy()
        pk = pi[1:] * rel
        pi0 = max(pi[0], 1.0 - pk.sum())
        pk = pk * (1.0 - pi0) / max(pk.sum(), 1e-12) if pk.sum() > 0 else pk
        pi = np.concatenate([[pi0], pk]); pi /= pi.sum()
        self.prior = Mix(pi, self.prior.m, self.prior.C)
        self.mass = float(1.0 - pi0)

    # ---- population EM with births ---------------------------------------------------------
    def _init_eb(self, X, O):
        n = X.shape[0]
        P0 = np.tile((np.eye(self.D) / self.v0)[None], (n, 1, 1))
        m, H, f, ev = laplace_fit(X, O, np.zeros((n, self.D)), P0, eps=self.opt["eps"])
        C = np.linalg.inv(H)
        mu = m.mean(axis=0)
        dm = m - mu
        Sig = (C + dm[:, :, None] * dm[:, None, :]).mean(axis=0)
        self.mu = mu[None]
        self.Sig = self._regularize_cov(Sig[None])
        self.pi = np.array([1.0 - self.opt["w_iso"]]); self.pi0 = self.opt["w_iso"]
        self.W = None

    def _pooled_fit(self, Xf, Of, w, th0, P0):
        """Responsibility-weighted ridge logistic regression (damped Newton) on all logged contrasts."""
        th = th0.copy()

        def obj(t):
            z = Xf @ t
            return float((w * (np.logaddexp(0.0, z) - Of * z)).sum() + 0.5 * t @ P0 @ t)

        f = obj(th)
        H = P0
        for _ in range(int(self.opt["pool_iters"])):
            z = Xf @ th; p = expit(z)
            g = Xf.T @ (w * (p - Of)) + P0 @ th
            H = (Xf * (w * p * (1 - p))[:, None]).T @ Xf + P0
            step = np.linalg.solve(H, g)
            a = 1.0
            for _ls in range(10):
                t_try = th - a * step
                f_try = obj(t_try)
                if f_try <= f + 1e-12:
                    th, f = t_try, f_try
                    break
                a *= 0.5
            if np.max(np.abs(g)) < 1e-5:
                break
        return th, H

    def _em_iterations(self, X, O, iters, prune=True):
        """As HBMix, with two alternative M-steps.
        mstep=scatter: Sigma_k from the spread of the students' posterior means only (directions no click has probed do
                       not inherit the prior variance); means as in the moment M-step.
        mstep=pooled : latent-class Bradley-Terry: mu_k = responsibility-weighted ridge logistic fit on ALL logged
                       contrasts (the students' own clicks pooled), Sigma_k = its Laplace covariance + comp_var * v0 * I.
                       Unlike the moment M-step, the mean keeps learning when the component is tight."""
        o = self.opt
        if o["mstep"] == "moment":
            return super()._em_iterations(X, O, iters, prune)
        if o["mstep"] == "pooled":
            return self._em_pooled(X, O, iters, prune)
        n = X.shape[0]
        L = -np.inf
        for _ in range(iters):
            m, C, ev = self._estep(X, O)
            logp = np.log(np.maximum(np.concatenate([[self.pi0], self.pi]), 1e-300))
            L = float(np.logaddexp.reduce(logp[None] + ev, axis=1).sum())
            r = self._resp(ev)
            Nk = r.sum(axis=0)
            M = len(self.pi)
            mu = np.zeros((M, self.D)); Sig = np.zeros((M, self.D, self.D))
            for k in range(M):
                rk = r[:, k + 1]
                nk = max(Nk[k + 1], 1e-9)
                mu[k] = (rk[:, None] * m[:, k + 1]).sum(axis=0) / nk
                dm = m[:, k + 1] - mu[k]
                Sig[k] = (rk[:, None, None] * (dm[:, :, None] * dm[:, None, :])).sum(axis=0)
                Sig[k] = (Sig[k] + o["nu"] * o["floor"] * np.eye(self.D)) / (nk + o["nu"])
            from persona_lab.infer import project_ball
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

    def _em_pooled(self, X, O, iters, prune=True):
        o = self.opt
        n, b, D = X.shape
        Xf = X.reshape(n * b, D); Of = O.reshape(n * b)
        P0 = (o["pool_ridge"] / self.v0) * np.eye(D)
        L = -np.inf
        for _ in range(iters):
            m, C, ev = self._estep(X, O)
            logp = np.log(np.maximum(np.concatenate([[self.pi0], self.pi]), 1e-300))
            L = float(np.logaddexp.reduce(logp[None] + ev, axis=1).sum())
            r = self._resp(ev)
            Nk = r.sum(axis=0)
            M = len(self.pi)
            mu = np.zeros((M, D)); Sig = np.zeros((M, D, D))
            for k in range(M):
                w = np.repeat(r[:, k + 1], b)
                th, H = self._pooled_fit(Xf, Of, w, self.mu[k], P0)
                mu[k] = th
                Sig[k] = np.linalg.inv(H) + o["comp_var"] * self.v0 * np.eye(D)
            from persona_lab.infer import project_ball
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

    def _em(self):
        o = self.opt
        if o["pop"] == "kde":
            return KDEPrior._em(self)
        X, O = self._stack()
        n = X.shape[0]
        if len(self.pi) == 0:
            if o["init_k"] > 1:
                self._init_components(X, O)                          # k-means++ seeds on isotropic posterior means
            else:
                self._init_eb(X, O)
        self._em_iterations(X, O, o["em_iters"])
        for _ in range(int(o["births"])):
            if len(self.pi) >= o["n_comp"]:
                break
            L, m, C, ev = self._loglik(X, O)                       # ev (n, M+1): fallback first
            gap = ev[:, 0] - ev[:, 1:].max(axis=1)
            s = int(np.argmax(gap))
            if gap[s] <= o["birth_margin"]:
                break
            self.mu = np.concatenate([self.mu, m[s, 0][None]], axis=0)
            self.Sig = np.concatenate([self.Sig, self._regularize_cov(C[s, 0][None])], axis=0)
            pi_new = 1.0 / n
            self.pi = np.concatenate([self.pi * (1.0 - pi_new), [pi_new * (1.0 - self.pi0)]])
            self.rec = np.concatenate([self.rec, [0.0]]) if len(self.rec) == len(self.pi) - 1 else np.zeros(len(self.pi))
            self.W = None
            self.n_births += 1
            self._em_iterations(X, O, o["em_iters"])
        if o["merge_tol"] > 0:
            self._merge()
        self.n_em += 1

    def diagnostics(self):
        d = super().diagnostics()
        d.update({"births": self.n_births, "n_em": self.n_em, "mass": self.mass, "rec_max": float(self.rec.max()) if len(self.rec) else 0.0})
        return d

    # ---- tables for the current belief -----------------------------------------------------
    def _comp_tables(self, F, post):
        """Per component: utilities (M,K), click probabilities P(first chosen) (M,15), strong regret (M,15)."""
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        U = post.m @ F.T
        x = self.beta * (F[i] - F[j])                               # (15,D)
        md = x @ post.m.T                                           # (15,M)
        vd = np.einsum("pd,mde,pe->pm", x, post.C, x)
        p = probit_click_prob(md, vd).T                             # (M,15)
        e = self.opt["eps"]
        p = (1 - e) * p + 0.5 * e                                   # lapse-robust likelihood (fixed across worlds)
        R = 2 * U.max(axis=1, keepdims=True) - U[:, i] - U[:, j]
        return U, np.clip(p, 1e-6, 1 - 1e-6), R

    def _truth_tables(self, F, th):
        """Per sampled truth: click probabilities (S,15) and strong regret (S,15)."""
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        U = th @ F.T                                                # (S,K)
        x = self.beta * (F[i] - F[j])
        e = self.opt["eps"]
        p = (1 - e) * expit(x @ th.T).T + 0.5 * e                   # (S,15)
        R = 2 * U.max(axis=1, keepdims=True) - U[:, i] - U[:, j]
        return p, R

    def observe(self, F):
        self.fbank.append(F)
        if len(self.fbank) > self.opt["bank"]:
            self.fbank.pop(0)

    def _restricted(self):
        post = self.post
        m = self.opt["top_m"]
        if post.M <= m:
            return post
        idx = np.argsort(-post.pi)[:m]
        pi = post.pi[idx] / post.pi[idx].sum()
        return Mix(pi, post.m[idx], post.C[idx])

    # ---- pair rules --------------------------------------------------------------------------
    def pair_scores(self, Phi):
        """Expected total strong regret (lower is better) of showing each of the 15 pairs now, under the rule."""
        Phi_raw = self._aug(Phi)
        F = Phi_raw * self.scale[None, :]
        self.observe(F)
        rem = max(self.b - self.t - 1, 0)
        post = self._restricted()
        U0, p0, R0 = self._comp_tables(F, post)
        imm = post.pi @ R0                                          # (15,) expected strong regret now
        rule = getattr(self, "rule_override", None) or self.opt["rule"]
        if rule == "lts" and post.M >= 2:
            # latent-bandit Thompson sampling: the greedy pair of one component drawn from the belief (every round)
            k = int(self.rng.choice(post.M, p=post.pi / post.pi.sum()))
            sc = R0[k]
        elif rule == "lucb" and post.M >= 2:
            # latent-bandit UCB: optimistic over the plausible components (belief weight within exp(-lucb_c) of the top)
            lw = np.log(np.maximum(post.pi, 1e-300))
            sc = R0[lw >= lw.max() - self.opt["lucb_c"]].min(axis=0)
        elif rule in ("emax", "vote", "vote_disagree", "lts", "lucb") or rem == 0 or post.M < 2:
            sc = imm                                                # vote with top-anchored utilities = greedy score
        elif rule == "kg":
            sc = self._kg(post, imm, p0, rem)
        elif rule == "kg2":
            sc = self._kg2(post, imm, p0, rem)
        elif rule == "rollout":
            sc = self._rollout(post, imm, p0, F, rem)
        else:
            raise ValueError(rule)
        sc = sc + 1e-9 * self.rng.random(len(sc))                  # exact ties (flat prior) broken at random, as none:emax:rt
        self._scores, self._Phi_raw = sc, Phi_raw
        return sc

    def select(self, Phi, fixed=None):
        rule = getattr(self, "rule_override", None) or self.opt["rule"]
        if rule in ("vote", "vote_disagree") and fixed is None:
            # persona_lab's vote rule (each component's predicted utilities top-anchored in logit units, averaged with
            # the posterior weights): the population-smoothing rule that suits fuzzy populations
            self.observe(self._aug(Phi) * self.scale[None, :])
            return HBMix.select(self, Phi)
        sc = self.pair_scores(Phi)
        if fixed is not None:                                       # hedge mode: one slot is imposed
            sc = np.where((self.pairs == fixed).any(axis=1), sc, np.inf)
        k = int(np.argmin(sc))
        i, j = int(self.pairs[k, 0]), int(self.pairs[k, 1])
        if fixed is not None and j == fixed:
            i, j = j, i
        self._last = (i, j, self._Phi_raw)
        return i, j

    def _kg(self, post, imm, p0, rem):
        pi = post.pi
        pbar = pi @ p0                                              # (15,)
        w1 = pi[:, None] * p0 / pbar[None]                          # (M,15) belief after y=1
        w0 = pi[:, None] * (1 - p0) / (1 - pbar)[None]
        v1 = np.zeros(len(imm)); v0 = np.zeros(len(imm))
        for F in self.fbank:
            _, _, R = self._comp_tables(F, post)                    # (M,15)
            v1 += (w1.T @ R).min(axis=1); v0 += (w0.T @ R).min(axis=1)
        nb = len(self.fbank)
        fut = (pbar * v1 + (1 - pbar) * v0) / nb
        return imm + self.opt["h"] * rem * fut

    def _kg2(self, post, imm, p0, rem):
        """Two-step lookahead: expected regret now + expected regret of the best second pair (chosen by one-step
        lookahead on a bank question) + (remaining - 1) x expected greedy regret after both clicks."""
        if rem < 2:
            return self._kg(post, imm, p0, rem)
        pi = post.pi
        NP, M = p0.shape[1], len(pi)
        pbar = pi @ p0                                                      # (NP,)
        W1 = np.stack([pi[:, None] * p0 / pbar[None], pi[:, None] * (1 - p0) / (1 - pbar)[None]], axis=1)   # (M,2,NP)
        W1 = np.transpose(W1, (2, 1, 0))                                    # (NP,2,M) beliefs after the first click
        P1 = np.stack([pbar, 1 - pbar], axis=1)                             # (NP,2) probability of each first outcome
        fut = np.zeros(NP)
        for F in self.fbank:
            _, pq, Rq = self._comp_tables(F, post)                          # (M,NP2), (M,NP2)
            imm2 = W1 @ Rq                                                  # (NP,2,NP2) expected regret of the second pair
            pb2 = W1 @ pq                                                   # (NP,2,NP2) P(first chosen) for the second pair
            # beliefs after the second click: (NP,2,NP2,2,M)
            W2y = W1[:, :, None, :] * pq.T[None, None, :, :] / np.maximum(pb2[..., None], 1e-12)
            W2n = W1[:, :, None, :] * (1 - pq.T)[None, None, :, :] / np.maximum((1 - pb2)[..., None], 1e-12)
            v2y = (W2y @ Rq).min(axis=-1); v2n = (W2n @ Rq).min(axis=-1)    # (NP,2,NP2) greedy regret after both clicks
            cont = pb2 * v2y + (1 - pb2) * v2n
            best2 = (imm2 + (rem - 1) * cont).min(axis=-1)                  # (NP,2) value of the best second pair
            fut += (P1 * best2).sum(axis=1)
        return imm + self.opt["h"] * fut / len(self.fbank)

    def _rollout(self, post, imm, p0, F0, rem):
        o = self.opt
        S, M, NP = int(o["n_roll"]), post.M, len(self.pairs)
        rng = self.rng
        th = post.sample(rng, S)                                    # (S,D) sampled truths
        nb = len(self.fbank)
        qs = rng.integers(0, nb, size=rem)
        u = rng.random(size=(rem + 1, S))
        logpi = np.log(np.maximum(post.pi, 1e-300))
        pt0, Rt0 = self._truth_tables(F0, th)                       # (S,15)
        logw = np.broadcast_to(logpi, (NP, S, M)).copy()
        lp1, lp0 = np.log(p0), np.log(1 - p0)                       # (M,15)
        y = u[0][None, :] < pt0.T                                   # (NP,S)
        logw += np.where(y[..., None], lp1.T[:, None, :], lp0.T[:, None, :])
        cost = Rt0.T.mean(axis=1)                                   # (NP,)
        cache = {}
        ar = np.arange(S)[None, :]
        for r in range(rem):
            q = int(qs[r])
            if q not in cache:
                Fq = self.fbank[q]
                _, pq, Rq = self._comp_tables(Fq, post)
                ptq, Rtq = self._truth_tables(Fq, th)
                cache[q] = (np.log(pq), np.log(1 - pq), Rq, ptq, Rtq)
            lq1, lq0, Rq, ptq, Rtq = cache[q]
            w = np.exp(logw - logw.max(axis=-1, keepdims=True)); w /= w.sum(axis=-1, keepdims=True)
            g = np.argmin(w @ Rq, axis=-1)                          # (NP,S) greedy pair under the rolled belief
            cost += Rtq[ar, g].mean(axis=1)
            yq = u[r + 1][None, :] < ptq[ar, g]
            logw += np.where(yq[..., None], lq1[:, g].transpose(1, 2, 0), lq0[:, g].transpose(1, 2, 0))
        return cost
