"""Prior-learning Thompson samplers across sessions, adapted to pairwise (Bradley-Terry) choices: `lab:metats`, `lab:hierts`,
`lab:adats` (comparison methods of the paper).

Model (the papers' Gaussian case, Kveton et al. 2021 Sec. 3.3 / App. D; Hong et al. 2022 eq. 2; Basu et al. 2021 Sec. 3):
    mu ~ N(0, Sigma_q),   theta_s ~ N(mu, Sigma_0) per session s,   choice ~ sigmoid(x . theta_s),  x = beta (F_i - F_j)
over the features F = phi, or with aug > 0 the augmented [phi | aug * onehot(persona)] / sqrt(1 + aug^2) (the paper uses aug=0:
its phi already contains the persona indicator). Sigma_0 = s0 * v0 I and
Sigma_q = sq * v0 I are KNOWN hyper-parameters in the papers; here they are tuned on the development roster like every comparison
method (v0 = S^2 / d, the no-transfer prior variance). Only the hyper-mean mu is learned, exactly as in the papers.

Non-conjugate likelihood: each session's clicks enter through a Gaussian site exp(theta' B_s - theta' G_s theta / 2) read off a Laplace
fit (G_s = posterior precision - Sigma_0^-1 = the Fisher information of the clicks at the mode, B_s = H_s m_s - Sigma_0^-1 mu_ref, mode
under the prior N(mu_ref, Sigma_0), mu_ref = the current hyper-mean, two passes per refit). With these sites the papers' Gaussian
hyper-posterior applies unchanged (HierTS eq. 5):
    Sigma_bar^-1 = Sigma_q^-1 + sum_s Lambda_s,  mu_bar = Sigma_bar sum_s nu_s,  Lambda_s = G_s (I + Sigma_0 G_s)^-1,  nu_s = (I + G_s Sigma_0)^-1 B_s
(Lambda_s = (Sigma_0 + G_s^-1)^-1 and nu_s = Lambda_s theta_hat_s written without inverting G_s, which is singular after few clicks).

Modes (no initial delay in any, as in the papers):
  metats  Meta-TS (Kveton et al. 2021, Alg. 1): at the start of each session sample mu_s ~ N(mu_bar, Sigma_bar) from the
          completed sessions, then Thompson sampling within the session from the prior N(mu_s, Sigma_0).
  hierts  HierTS (Hong et al. 2022, Alg. 1, sequential tasks): every round sample mu_t from the hyper-posterior given the completed
          sessions AND the current session's clicks, then theta ~ P(theta | mu_t, current clicks), act on theta.
  adats   AdaTS (Basu et al. 2021, Alg. 1): mu marginalised, session prior N(mu_bar, Sigma_bar + Sigma_0), Thompson sampling within.
Dueling adaptation of "act on one posterior sample": show the two personas with the highest sampled utilities (exact ties at random;
pair=top2, default); pair=dts draws a second independent sample and shows its best persona other than the first
(double-Thompson-sampling style challenger). rule=emax replaces every sample by its mean (greedy diagnostic, not used in the paper).
"""
from __future__ import annotations

import numpy as np

from persona_bench.protocol import Policy

from .hb import parse_opts
from .infer import Mix, condition, laplace_fit


class PriorTS(Policy):
    DEFAULTS = dict(mode="hierts", s0=1.0, sq=1.0, aug=1.0, rule="ts", passes=2, pair="top2")

    def __init__(self, name="lab:hierts", **opts):
        self.name = name
        self.opt = dict(self.DEFAULTS); self.opt.update(opts)
        assert self.opt["mode"] in ("metats", "hierts", "adats"), self.opt["mode"]
        assert self.opt["rule"] in ("ts", "emax"), self.opt["rule"]
        assert self.opt["pair"] in ("top2", "dts"), self.opt["pair"]      # top2: two highest utilities of one sample; dts: challenger from a second sample

    def reset(self, rng, context):
        super().reset(rng, context)
        self.S, self.beta = context["S"], context["beta"]
        self.D = self.d + (self.K if self.opt["aug"] > 0 else 0)
        self.v0 = self.S ** 2 / self.d
        I = np.eye(self.D)
        self.Sig0 = self.opt["s0"] * self.v0 * I; self.P0 = I / (self.opt["s0"] * self.v0)
        self.Pq = I / (self.opt["sq"] * self.v0)
        self.mu_bar = np.zeros(self.D); self.Sig_bar = self.opt["sq"] * self.v0 * I; self.P_bar = self.Pq.copy()
        self.sess_X: list[np.ndarray] = []; self.sess_o: list[np.ndarray] = []
        self.W0 = None
        self._last = None

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _aug(self, Phi):
        a = self.opt["aug"]
        if a <= 0:
            return Phi
        return np.concatenate([Phi, a * np.eye(self.K)], axis=1) / np.sqrt(1.0 + a * a)

    def _gauss(self, m, C):
        if self.opt["rule"] == "emax":
            return m
        L = np.linalg.cholesky(0.5 * (C + C.T) + 1e-10 * np.eye(self.D))
        return m + L @ self.rng.normal(size=self.D)

    def _sites(self, X, O, mu_ref, w0=None):
        """Gaussian sites (G, B) of n sessions' clicks, Laplace at the prior N(mu_ref, Sigma_0). X (n,b,D), O (n,b)."""
        n = X.shape[0]
        m, H, _, _ = laplace_fit(X, O, np.tile(mu_ref, (n, 1)), np.tile(self.P0[None], (n, 1, 1)), w0=w0)
        G = H - self.P0[None]
        G = 0.5 * (G + np.swapaxes(G, 1, 2))
        B = np.einsum("nde,ne->nd", H, m) - (self.P0 @ mu_ref)[None]
        return G, B, m

    def _lam_nu(self, G, B):
        I = np.eye(self.D)
        Lam = np.linalg.solve(I[None] + G @ self.Sig0, G)            # (Sigma_0 + G^-1)^-1 = (I + G Sigma_0)^-1 G
        Lam = 0.5 * (Lam + np.swapaxes(Lam, 1, 2))
        nu = np.linalg.solve(I[None] + G @ self.Sig0, B[..., None])[..., 0]
        return Lam, nu

    # ---- session lifecycle ---------------------------------------------------------------------------------------
    def start_student(self):
        self.X: list[np.ndarray] = []; self.o: list[float] = []
        mode = self.opt["mode"]
        if mode == "metats":
            self.m0 = self._gauss(self.mu_bar, self.Sig_bar); self.C0 = self.Sig0
        elif mode == "adats":
            self.m0 = self.mu_bar.copy(); self.C0 = self.Sig_bar + self.Sig0
        if mode != "hierts":
            self.post = Mix(np.ones(1), self.m0[None].copy(), self.C0[None].copy())

    def _theta(self):
        """One draw (or the mean, rule=emax) of the current user's vector, and its mean."""
        if self.opt["mode"] != "hierts":
            return self._gauss(self.post.m[0], self.post.C[0]), self.post.m[0]
        if self.X:
            G, B, _ = self._sites(np.array(self.X)[None], np.array(self.o)[None], self.mu_bar)
            Lam, nu = self._lam_nu(G, B)
            G, B, Lam, nu = G[0], B[0], Lam[0], nu[0]
        else:
            G = Lam = np.zeros((self.D, self.D)); B = nu = np.zeros(self.D)
        Pt = self.P_bar + Lam                                         # hyper-posterior incl. the current session
        Ct = np.linalg.inv(Pt); mt = Ct @ (self.P_bar @ self.mu_bar + nu)
        mu = self._gauss(mt, Ct)
        Pth = self.P0 + G; Cth = np.linalg.inv(Pth)
        th = self._gauss(Cth @ (self.P0 @ mu + B), Cth)
        return th, Cth @ (self.P0 @ mt + B)

    def select(self, Phi):
        F = self._aug(Phi)
        th, _ = self._theta()
        u = F @ th
        order = np.lexsort((self.rng.random(self.K), -u))
        i = int(order[0])
        if self.opt["pair"] == "dts":                                 # double-TS style: the challenger maximises an independent second sample
            u2 = F @ self._theta()[0]
            u2[i] = -np.inf
            j = int(np.lexsort((self.rng.random(self.K), -u2))[0])
        else:
            j = int(order[1])
        self._last = (i, j, F)
        return i, j

    def update(self, outcome):
        i, j, F = self._last
        self.X.append(self.beta * (F[i] - F[j])); self.o.append(float(outcome))
        if self.opt["mode"] != "hierts":
            prior = Mix(np.ones(1), self.m0[None].copy(), self.C0[None].copy())
            self.post, _ = condition(prior, np.array(self.X), np.array(self.o), w_init=self.post.m)

    def end_student(self):
        if self.X:
            self.sess_X.append(np.array(self.X)); self.sess_o.append(np.array(self.o))
            self._refit()

    def _refit(self):
        n = len(self.sess_X); b = max(len(x) for x in self.sess_X)
        X = np.zeros((n, b, self.D)); O = np.zeros((n, b))
        for k, (x, oo) in enumerate(zip(self.sess_X, self.sess_o)):
            X[k, :len(x)] = x; O[k, :len(oo)] = oo                     # zero rows: no information (x = 0)
        w0 = None if self.W0 is None or len(self.W0) != n else self.W0
        mu_ref = self.mu_bar
        for _ in range(int(self.opt["passes"])):
            G, B, w0 = self._sites(X, O, mu_ref, w0=w0)
            Lam, nu = self._lam_nu(G, B)
            P = self.Pq + Lam.sum(axis=0)
            C = np.linalg.inv(P)
            mu_ref = C @ nu.sum(axis=0)
        self.mu_bar, self.Sig_bar, self.P_bar, self.W0 = mu_ref, 0.5 * (C + C.T), P, w0

    def predict_best(self, Phi):
        return int(np.argmax(self._aug(Phi) @ self._theta()[1]))

    def diagnostics(self):
        ev = np.linalg.eigvalsh(self.Sig_bar)
        return {"mu_norm": float(np.linalg.norm(self.mu_bar)), "sigbar_max": float(ev.max() / self.v0), "n_sess": len(self.sess_X)}


def make_prior_ts(spec: str):
    parts = spec.split(":")
    opts = parse_opts(parts[2:], PriorTS.DEFAULTS)
    opts["mode"] = parts[1]
    return PriorTS(name=spec, **opts)
