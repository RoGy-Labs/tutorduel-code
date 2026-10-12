"""CODuel on the tutoring benchmark: the population learner of `v3:` (persona_v3.policy.GatedV3) with the rater's score
recalibrated from completed sessions' clicks (Section "Score recalibration" and Appendix "Implementation details").

Spec: `qo:<v3 options>:xcols=1:recal=1:rfull=1:xpass=0:lama=10:aug=0:sa=<s>`, e.g. the paper's configuration on dataset v3.1L

    qo:rule=kg:mstep=pooled:comp_var=0:n_comp=16:birth_margin=0:births=3:alpha=20:peg=0.5:peg_min=8:w_iso=0:n_init=1:em_late=1
      :cap=0:cap_v0=1000:xcols=1:recal=1:rfull=1:xpass=0:lama=10:aug=0:sa=1.6996

The world's features are phi(p, q) = [e_p, g(p, q)] (K + 1 = 7 columns: the persona indicator and the rater's score). With g(q)
the K scores of question q, the learners see phi'(p, q) = [e_p, (A g(q))_p centred over personas / sd], where A = a I + A' is fitted
by logistic regression on every logged choice of the completed sessions,

    logit P(first shown chosen) = (e_i - e_k) . (b + b_n) + (A g(q))_i - (A g(q))_k,

with ridge `lama` on A', 1e-3 on a and on the shared persona offsets b, and `lamb` on the per-user offsets b_n. A = I until `w_min`
sessions are complete; A is refitted after every `refit` further sessions (warm-started), and sd is the standard deviation of the
centred recalibrated coordinate over the logged rounds. After each refit every stored feature-derived quantity of the inner
learner is rebuilt from the raw rows kept here: the mixture's stacked session contrasts, the current session's contrasts (its
posterior is re-conditioned), the lookahead's question bank, and the no-transfer learner's logs; the gate's evidence is kept.

Options (besides every v3 option): refit=10, w_min=20, lama=10 (ridge on A'), lamb=1 (ridge on the per-user offsets),
sa (required) = prior scale: the inner learner's benchmark scale is S' = sa sqrt(d' / K) with d' = K + 1 (the paper sets sa = S sqrt(K / d'),
S the median-norm scale of the comparison map [e_p, g], so that S' = S: results/features/<version>/dev/S_<map>.json).
xcols=1, recal=1, rfull=1, xpass=0 name this configuration (one rater column, the whole score recalibrated, phi' replaces it).
chk=1 recomputes, before every rebuild, the stored quantities with the old A and asserts they equal what the learner holds.

Random stream: the policy draws the stream of v3:<its v3 options> (the qo options are left out of the stream key), so the
population learner and its no-transfer replica see the same uniform draws as `v3:` with the same options. Admissibility: only phi
(observed), the context (K, d, beta, b, N) and the policy's own clicks are used.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from persona_bench.protocol import Policy
from persona_lab.hb import parse_opts

from .mix import MixV3
from .policy import GATE_DEFAULTS, GatedV3

# the keys of the recalibration (left out of the random-stream key); xcols / recal / rfull / xpass are fixed to the paper's values
QO_DEFAULTS = dict(qo=1, refit=10, w_min=20, lamb=1.0, sa=-1.0, chk=0, xcols=1, recal=1, rfull=1, xpass=0, lama=10.0)
FIXED = dict(qo=1, xcols=1, recal=1, rfull=1, xpass=0)


def fit_lin(D, sess, n_sess, F, pen, y, lamb, x0=None, maxiter=1000):
    """Logistic regression of the logged choices:

        logit P(first shown chosen) = D . (b0 + B_n) + F w

    D (R, K) persona contrast of the shown pair, sess (R,) session index in [0, n_sess), F (R, m) covariates with per-column
    ridge `pen` (m,) (loss term sum_j pen_j w_j^2). b0 ridge 1e-3, B ridge lamb. Deterministic (L-BFGS-B), warm start x0.
    Returns (b0 (K,), w (m,), B (n_sess, K), info). Parameter layout: [b0, w, B]."""
    R, K = D.shape
    m = F.shape[1]
    pen = np.asarray(pen, dtype=float)
    o = np.zeros(R)

    def unpack(x):
        return x[:K], x[K:K + m], x[K + m:].reshape(n_sess, K)

    def f(x):
        b0, w, B = unpack(x)
        s = o + D @ b0 + (D * B[sess]).sum(1) + F @ w
        loss = float((np.logaddexp(0.0, s) - y * s).sum() + (pen * w ** 2).sum() + lamb * (B ** 2).sum() + 1e-3 * (b0 ** 2).sum())
        g = expit(s) - y
        gD = g[:, None] * D
        gB = np.stack([np.bincount(sess, weights=gD[:, k], minlength=n_sess) for k in range(K)], 1)
        return loss, np.concatenate([gD.sum(0) + 2e-3 * b0, F.T @ g + 2 * pen * w, (gB + 2 * lamb * B).ravel()])

    x_init = np.zeros(K + m + K * n_sess) if x0 is None else x0
    res = minimize(f, x_init, jac=True, method="L-BFGS-B", options=dict(maxiter=int(maxiter)))
    b0, w, B = unpack(res.x)
    return b0.copy(), w.copy(), B.copy(), dict(nit=int(res.nit), fun=float(res.fun), ok=bool(res.success))


def split_phi(Phi, K):
    """Rows of phi = [e_p, g] (rows possibly relabelled) -> persona part P0 (K, K), identities jmap (K,), score column (K, 1)."""
    if Phi.shape != (K, K + 1):
        raise ValueError(f"qo: phi has shape {Phi.shape}; expected (K, K + 1) = ({K}, {K + 1}): [persona indicator, rater score]")
    P0 = np.ascontiguousarray(Phi[:, :K])
    jmap = np.argmax(np.abs(P0), axis=1)
    if not ((np.abs(P0) > 1e-12).sum(1) == 1).all() or len(set(jmap.tolist())) != K:
        raise ValueError("qo: the persona part of phi is not one nonzero entry per row with distinct identities")
    return P0, jmap, np.ascontiguousarray(Phi[:, K:])


class GatedQO(Policy):
    def __init__(self, spec, stream, s2opts, qopts):
        self.name = stream                        # the runner hashes self.name before reset(): the stream key
        self._spec = spec
        self.q = dict(qopts)
        self.inner = GatedV3(name=stream, **s2opts)

    # ---- lifecycle ------------------------------------------------------------------------------------------------
    def reset(self, rng, context):
        super().reset(rng, context)
        K = context["K"]
        self.dp = K + 1                           # phi' = [e_p, recalibrated score]
        self.S_in = float(self.q["sa"]) * float(np.sqrt(self.dp / K))
        self.inner.reset(rng, dict(context, d=self.dp, S=self.S_in))
        self.name = self._spec                    # rows are filed under the spec
        self.hist = []                            # completed sessions: raw rows (P0, jmap, scores), shown pairs, clicks
        self.recent = deque(maxlen=int(self.inner.mix.opt["bank"]))   # raw rows of the questions in the mixture's lookahead bank
        self.n_refit = 0
        self.fit_info = {}
        self.model = (None, np.eye(K), 1.0)       # (unused, A, sd) that _feat uses now: A = I until the first fit
        self.sess_m = []                          # per completed session: the model its stored features were built with
        self.warm = {}                            # warm start of the last fit
        self.traj = dict(n=[], sd=[], a=[])
        self.Wq, self.sd_h = None, 0.0

    def start_student(self):
        self.inner.start_student()
        self.c_P0, self.c_j, self.c_pair, self.c_o, self.c_x = [], [], [], [], []

    def _gid(self, X, jmap):
        """Rater score of every persona IDENTITY for the question, from the score column of its rows."""
        g = np.empty(self.K)
        g[jmap] = X[:, 0]
        return g

    def _feat(self, P0, jmap, X, model=False):
        """phi' of one question (K, K + 1); model: (unused, A, sd) to build the score column with (None: zero column);
        False = the current model."""
        if model is False:
            model = self.model
        col = np.zeros(self.K)
        if model is not None and model[2] > 0:
            _, Aq, sd_h = model
            h = (Aq @ self._gid(X, jmap))[jmap]                     # (A g(q))_{j(p)} for every row p
            col = (h - h.mean()) / sd_h
        return np.concatenate([P0, col[:, None]], axis=1)

    def select(self, Phi):
        P0, jmap, X = split_phi(Phi, self.K)
        i, j = self.inner.select(self._feat(P0, jmap, X))
        self.recent.append((P0, jmap, X))
        self.c_P0.append(P0); self.c_j.append(jmap); self.c_pair.append((i, j)); self.c_x.append(X)
        return i, j

    def update(self, outcome):
        self.inner.update(outcome)
        self.c_o.append(float(outcome))

    def end_student(self):
        if self.c_o:
            self.hist.append(dict(P0=np.array(self.c_P0), j=np.array(self.c_j), pair=np.array(self.c_pair, dtype=int),
                                  o=np.array(self.c_o), x=np.array(self.c_x)))
            self.sess_m.append(self.model)                            # its stored features were built with the current model
        n = len(self.hist)
        q = self.q
        if n >= q["w_min"] and (n - q["w_min"]) % q["refit"] == 0 and self.c_o:
            if q["chk"]:
                self._rebuild(check=True)
            self._refit()
            self._rebuild()
        self.inner.end_student()

    # ---- recalibration ----------------------------------------------------------------------------------------------
    def _rows(self):
        """Every logged round of the completed sessions as flat arrays."""
        K = self.K
        ji, jk, sess, y, G = [], [], [], [], []
        for n, s in enumerate(self.hist):
            rows = np.arange(len(s["o"]))
            ji.append(s["j"][rows, s["pair"][:, 0]]); jk.append(s["j"][rows, s["pair"][:, 1]])
            sess.append(np.full(len(rows), n)); y.append(s["o"])
            G.append(np.array([self._gid(s["x"][t], s["j"][t]) for t in rows]))
        R = dict(ji=np.concatenate(ji), jk=np.concatenate(jk), sess=np.concatenate(sess), y=np.concatenate(y))
        R["D"] = np.eye(K)[R["ji"]] - np.eye(K)[R["jk"]]
        R["G"] = np.concatenate(G)
        return R

    def _fit(self, R):
        """A = a I + A' fitted on every logged round (warm-started from the previous fit)."""
        K = self.K
        ns = len(self.hist)
        D, G, y, sess = R["D"], R["G"], R["y"], R["sess"]
        lama, lamb = float(self.q["lama"]), float(self.q["lamb"])
        cols = [(D * G).sum(1)[:, None], (D[:, :, None] * G[:, None, :]).reshape(len(y), K * K)]
        pen = [np.full(1, 1e-3), np.full((K, K), lama).ravel()]
        prev = self.warm.get("all")
        x0 = None
        if prev is not None:
            B = np.concatenate([prev["B"], np.zeros((ns - len(prev["B"]), K))], axis=0)
            x0 = np.concatenate([prev["b0"], prev["w"], B.ravel()])
        b0, w, B, info = fit_lin(D, sess, ns, np.concatenate(cols, 1), np.concatenate(pen), y, lamb, x0=x0)
        self.warm["all"] = dict(b0=b0, w=w, B=B)
        a = float(w[0])
        return a, a * np.eye(K) + w[1:1 + K * K].reshape(K, K), info

    def _refit(self):
        K = self.K
        R = self._rows()
        n_sess = len(self.hist)
        a, A, info = self._fit(R)
        H = np.zeros((len(R["y"]), K))                               # recalibrated score of every logged round (identity basis)
        H = H + R["G"] @ A.T
        Hc = H - H.mean(axis=1, keepdims=True)
        sd = float(np.sqrt((Hc ** 2).mean()))
        sd = sd if sd > 1e-12 else 0.0
        self.model = (None, A, sd) if sd > 0 else None
        self.sess_m = [self.model] * n_sess
        self.Wq, self.sd_h = A, sd
        self.n_refit += 1
        self.fit_info = dict(info, n=n_sess, rows=int(len(R["y"])), sd=sd, wnorm=float(np.linalg.norm(A)))
        t = self.traj
        t["n"].append(n_sess); t["sd"].append(sd); t["a"].append(a)

    def _rebuild(self, check=False):
        """Recompute every feature-derived quantity the inner learner stores from the raw rows (current A). check=True:
        compare with what it holds (must be bit-identical when A has not changed) instead of replacing it."""
        mix, base = self.inner.mix, self.inner.base
        a_mix, a_base = mix.beta, base.beta

        def contrasts(s, model):
            Fs = [self._feat(s["P0"][t], s["j"][t], s["x"][t], model) for t in range(len(s["o"]))]
            xm = np.array([a_mix * (mix._aug(F)[i] - mix._aug(F)[j]) for F, (i, j) in zip(Fs, s["pair"])])
            xb = np.array([a_base * (F[i] - F[j]) for F, (i, j) in zip(Fs, s["pair"])])
            return xm, xb

        past = [contrasts(s, m) for s, m in zip(self.hist[:-1], self.sess_m[:-1])]
        cur_m, cur_b = contrasts(self.hist[-1], self.sess_m[-1])
        fbank = [mix._aug(self._feat(*rw)) * mix.scale[None, :] for rw in self.recent]
        G = np.eye(base.d)
        for _, xb in past:
            G += xb.T @ xb
        if len(mix.sess_X) != len(past) or len(base.log_X) != len(past) or len(mix.fbank) != len(fbank):
            raise RuntimeError("qo: the inner learner's stored sessions / bank do not match the policy's raw log")
        if check:
            same = (all(np.array_equal(a, xm) for a, (xm, _) in zip(mix.sess_X, past))
                    and all(np.array_equal(a, xb) for a, (_, xb) in zip(base.log_X, past))
                    and np.array_equal(np.array(mix.X), cur_m) and np.array_equal(np.array(base.X), cur_b)
                    and all(np.array_equal(a, b) for a, b in zip(mix.fbank, fbank)) and np.array_equal(base.G, G))
            if not same:
                raise AssertionError("qo chk: rebuilt features differ from the inner learner's stored ones (stale or inexact bookkeeping)")
            self.n_chk = getattr(self, "n_chk", 0) + 1
            return
        mix.sess_X = [xm for xm, _ in past]
        base.log_X = [xb for _, xb in past]
        base.G = G
        mix.X = list(cur_m)
        base.X = list(cur_b)
        mix.fbank = fbank
        mix.W = None                              # E-step warm starts were computed on the old features
        mix._refresh()                            # current session's posterior on the rebuilt contrasts

    # ---- predictions / reports ------------------------------------------------------------------------------------
    def predict_best(self, Phi):
        P0, jmap, X = split_phi(Phi, self.K)
        return self.inner.predict_best(self._feat(P0, jmap, X))

    def diagnostics(self):
        d = self.inner.diagnostics()
        d.update(qo_refits=self.n_refit, qo_sd=self.sd_h, qo_S=self.S_in, qo_d=self.dp,
                 qo_wnorm=float(np.linalg.norm(self.Wq)) if self.Wq is not None else 0.0, qo_nit=self.fit_info.get("nit", 0),
                 qo_chk=getattr(self, "n_chk", 0))
        t = self.traj
        d.update(qo_traj_n=list(t["n"]), qo_traj_sd=list(t["sd"]), qo_traj_a=list(t["a"]))   # per refit: sessions, sd, a
        return d


def make_qo(spec: str) -> Policy:
    parts = [p for p in spec.split(":")[1:] if p]
    s2_defaults = dict(MixV3.DEFAULTS); s2_defaults.update(GATE_DEFAULTS)
    clash = set(QO_DEFAULTS) & set(s2_defaults)
    assert not clash, clash
    defaults = dict(s2_defaults); defaults.update(QO_DEFAULTS)
    opts = parse_opts(parts, defaults)
    qopts = {k: opts.pop(k) for k in QO_DEFAULTS}
    bad = {k: qopts[k] for k, v in FIXED.items() if qopts[k] != v}
    if bad:
        raise ValueError(f"{spec}: this implementation is the paper's configuration {FIXED}; got {bad}")
    if qopts["chk"] not in (0, 1) or qopts["refit"] < 1 or qopts["w_min"] < 1 or qopts["lama"] < 0 or qopts["lamb"] <= 0 \
            or qopts["sa"] <= 0:
        raise ValueError(f"{spec}: need chk in (0, 1), refit >= 1, w_min >= 1, lama >= 0, lamb > 0 and the prior scale sa > 0")
    if opts["mode"] != "full":
        raise ValueError(f"{spec}: mode={opts['mode']!r}; only the full gate is supported")
    stream_parts = [p for p in parts if p.split("=", 1)[0] not in QO_DEFAULTS]
    stream = "v3:" + ":".join(stream_parts)
    return GatedQO(spec, stream, opts, qopts)
