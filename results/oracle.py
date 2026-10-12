"""The per-user linear oracle of the paper (Section "Shared Protocol and Evaluation"): for every user, the parameters of a utility
linear in CODuel's features that minimise that user's excess regret over all their questions, in hindsight.

Features: phi_A(p, q) = [e_p, (A g(q))_p], where g is the rater's score (the last column of the feature map) and the recalibration
A = a I + A' is fitted ONCE on all users of the roster in hindsight, by the logistic model of CODuel's recalibration on the full
choice tables (every user, question and persona pair, the stored probability as the target; ridge 10 on A', 1e-3 on a and on the
shared persona offsets, 1 on the per-user offsets). For each user, the utility b_p + w (A g(q))_p (persona offsets b, slope w)
shows the two personas with the highest utilities on each question; the search maximises the user's summed utility of the shown
pairs, i.e. minimises the user's excess regret (no closed form): every slope of a grid of 201 values in [-10, 10] with three offset
initialisations (the per-persona mean utility, the same shifted by the slope, a random draw) and the two least-squares solutions
(of the choice logits and of the utilities), each followed by coordinate ascent on the offsets with an exact line search; the best
three are then polished by alternating exact slope and offset searches. The value found is achievable, so the oracle's true score
is at least the reported one.

    python results/oracle.py --version v3.1L --map jevgw --roster test_unseen      # prints the oracle's excess and score
    python results/oracle.py --version v3.1L --map jevgw --all --out results/oracle  # every roster -> oracle_<version>_<roster>.npy

The arrays hold, per arrival of the roster (dataset order), the oracle's excess regret over the session (U units, summed over the
24 questions); results/paper_stats.py reads them. Deterministic.
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persona_v3.recal import fit_lin  # noqa: E402

PAIRS = list(combinations(range(6), 2))
PI, PJ = np.array(PAIRS).T
K = 6
EYE = np.eye(K)
ROSTERS = ["dev", "test", "test_unseen", "test_continuum", "test_idio"]
# slope grid: w = 0 and +-logspace(-3, 1, 100) (201 values in [-10, 10]; U is in [0, 1], g has unit sd)
WGRID = np.concatenate([[0.0], np.logspace(-3, 1, 100), -np.logspace(-3, 1, 100)])
N_RAND = 1                                  # random offset initialisations per slope (besides the two deterministic ones)


class Roster:
    def __init__(self, version, roster, fmap):
        self.version, self.roster, self.fmap = version, roster, fmap
        split = "dev" if roster == "dev" else "test"
        D = ROOT / "dataset" / version / "types" / roster
        self.P, self.U, self.QIDS = np.load(D / "P.npy"), np.load(D / "U.npy"), np.load(D / "qids.npy")
        self.A_, self.J = self.QIDS.shape
        U = self.U
        self.TOP = np.sort(U, -1)[..., ::-1]
        self.FLOOR = self.TOP[..., 0] - self.TOP[..., 1]
        self.PAIR_EX = 2 * self.TOP[..., :1] - U[..., PI] - U[..., PJ] - self.FLOOR[..., None]
        self.RND = self.PAIR_EX.mean()                             # random pairs' expected excess per round
        p = np.clip(self.P[..., PI, PJ], 1e-4, 1 - 1e-4)
        self.LOGIT = np.log(p / (1 - p))
        Phi = np.load(ROOT / "results" / "features" / version / split / f"phi_{fmap}.npy").astype(np.float64)
        assert Phi.shape[1:] == (K, 7) and np.allclose(Phi[..., :K], EYE), "phi must be [e_p, g]"
        self.G = Phi[self.QIDS][..., K]                            # (A, J, 6) rater scores of every arrival's questions
        self.E = np.broadcast_to(EYE, (self.A_, self.J, K, K))

    def ex_of(self, pick, idx):
        """pick (A, n, 2) on questions idx (A, n) -> per-arrival mean excess per question (A,)."""
        tk = lambda X: np.take_along_axis(X, idx, 1)
        ua = np.take_along_axis(np.take_along_axis(self.U, idx[..., None], 1), pick, -1).sum(-1)
        return (2 * tk(self.TOP[..., 0]) - ua - tk(self.FLOOR)).mean(1)

    def lsq_theta(self, F, target, tr):
        """(b, w) of the per-user least-squares fit on questions tr, for F = [e_p, g'] (initialisations of the search)."""
        th = np.zeros((self.A_, F.shape[-1]))
        for a in range(self.A_):
            Fa = F[a, tr[a]]
            if target == "logit":
                X = (Fa[:, PI] - Fa[:, PJ]).reshape(-1, F.shape[-1]); y = self.LOGIT[a, tr[a]].reshape(-1)
            else:
                X = Fa.reshape(-1, F.shape[-1]); y = self.U[a, tr[a]].reshape(-1)
            th[a] = np.linalg.lstsq(X, y, rcond=None)[0]
        return th[:, :K], th[:, K]

    def one_dim(self, g):
        return np.concatenate([self.E, g[..., None]], -1)

    def regret(self, g):
        def fp(tr, te):
            Gt = np.take_along_axis(g, tr[..., None], 1); Ut = np.take_along_axis(self.U, tr[..., None], 1)
            inits = [self.lsq_theta(self.one_dim(g), "logit", tr), self.lsq_theta(self.one_dim(g), "U", tr)]
            seed = int(tr[0, 0] * 1000 + tr[0, 1]) + 17
            b, w, _ = regret_opt(Gt, Ut, inits, seed)
            V = b[:, None, :] + w[:, None, None] * np.take_along_axis(g, te[..., None], 1)
            return top2(V)
        return fp

    def recal_all(self, lama=10.0, lamb=1.0):
        """A = a I + A' fitted on all arrivals of the roster (full choice tables); returns (A g, A, fit info)."""
        A_, J = self.A_, self.J
        y = self.P[..., PI, PJ].reshape(-1)
        D = np.broadcast_to(EYE[PI] - EYE[PJ], (A_, J, len(PAIRS), K)).reshape(-1, K)
        G = np.broadcast_to(self.G[:, :, None, :], (A_, J, len(PAIRS), K)).reshape(-1, K)
        sess = np.repeat(np.arange(A_), J * len(PAIRS))
        cols = np.concatenate([(D * G).sum(1)[:, None], (D[:, :, None] * G[:, None, :]).reshape(len(y), K * K)], 1)
        pen = np.concatenate([np.full(1, 1e-3), np.full(K * K, lama)])
        _, w, _, info = fit_lin(D, sess, A_, cols, pen, y, lamb)
        A = w[0] * EYE + w[1:].reshape(K, K)
        return self.G @ A.T, A, info

    def oracle_excess(self):
        """Per-arrival excess regret of the oracle over the session (sum over its questions), and the fitted A."""
        gA, A, _ = self.recal_all()
        allq = np.broadcast_to(np.arange(self.J), (self.A_, self.J))
        return self.ex_of(self.regret(gA)(allq, allq), allq) * self.J, A


def top2(V):
    """stable top-2 picks of V (..., 6) (ties -> lower persona index first, as argsort(kind='stable') of -V)."""
    return np.argsort(-V, -1, kind="stable")[..., :2]


def value(b, w, G, Uu):
    """sum over questions of U of the top-2 picks under b + w G. b (B, 6), w (B,), G, Uu (B, n, 6) -> (B,)."""
    pk = top2(b[:, None, :] + w[:, None, None] * G)
    return np.take_along_axis(Uu, pk, -1).sum((-1, -2))


def cd_offsets(b, w, G, Uu, max_sweeps=30):
    """Coordinate ascent on the offsets with an EXACT line search per coordinate. With the other offsets fixed, persona p is in
    the top 2 of question j iff b_p > t_j = (second-best other value) - w g_jp, and then gains d_j = U_p - U_(second other); the
    total is a step function of b_p with n breakpoints, maximised by sorting t. b_p is set to the midpoint of the best interval
    (empty intervals from tied thresholds are excluded). Returns (b, value)."""
    B, n, _ = G.shape
    b = b.copy()
    rows = np.arange(B)[:, None]; cols = np.arange(n)[None, :]
    for _ in range(max_sweeps):
        changed = False
        for p in range(K):
            V = b[:, None, :] + w[:, None, None] * G
            Vo = V.copy(); Vo[..., p] = -np.inf
            o = top2(Vo)
            o2 = o[..., 1]
            t = V[rows, cols, o2] - w[:, None] * G[..., p]
            d = Uu[..., p] - Uu[rows, cols, o2]
            order = np.argsort(t, -1, kind="stable")
            ts = np.take_along_axis(t, order, -1); ds = np.take_along_axis(d, order, -1)
            cum = np.concatenate([np.zeros((B, 1)), np.cumsum(ds, -1)], -1)            # value of b_p in (ts[k-1], ts[k])
            ok = np.ones((B, n + 1), bool)
            ok[:, 1:n] = (ts[:, 1:] - ts[:, :-1]) > 1e-9 * (1 + np.abs(ts[:, 1:]))
            cum = np.where(ok, cum, -np.inf)
            cur = np.where(b[:, p][:, None] > t, d, 0.0).sum(-1)
            k = cum.argmax(-1)
            better = cum[np.arange(B), k] > cur + 1e-12
            if not better.any():
                continue
            span = ts[:, -1] - ts[:, 0] + 1e-3
            lo = ts[np.arange(B), np.maximum(k - 1, 0)]; hi = ts[np.arange(B), np.minimum(k, n - 1)]
            new = np.where(k == 0, ts[:, 0] - 0.5 * span, np.where(k == n, ts[:, -1] + 0.5 * span, 0.5 * (lo + hi)))
            b[better, p] = new[better]
            changed = True
        if not changed:
            break
    return b, value(b, w, G, Uu)


def w_search(b, w, G, Uu, chunk=64):
    """Exact line search on the slope with the offsets fixed: the top-2 sets change only where two personas swap on a question,
    w* = -(b_i - b_k) / (g_i - g_k); every interval between consecutive crossings (and both ends, and the current w) is tried."""
    B, n, _ = G.shape
    w_new = w.copy(); v_new = value(b, w, G, Uu)
    for s in range(0, B, chunk):
        sl = slice(s, min(B, s + chunk))
        bb, GG, UU = b[sl], G[sl], Uu[sl]
        db = bb[:, PI] - bb[:, PJ]                                  # (c, 15)
        dg = GG[..., PI] - GG[..., PJ]                              # (c, n, 15)
        with np.errstate(divide="ignore", invalid="ignore"):
            cr = np.where(np.abs(dg) > 1e-12, -db[:, None, :] / dg, np.nan).reshape(len(bb), -1)
        cr = np.sort(cr, -1)                                        # nan last
        lo = np.nanmin(np.where(np.isfinite(cr), cr, np.nan), -1); hi = np.nanmax(np.where(np.isfinite(cr), cr, np.nan), -1)
        mids = 0.5 * (cr[:, 1:] + cr[:, :-1])
        cand = np.concatenate([mids, (lo - 1 - np.abs(lo))[:, None], (hi + 1 + np.abs(hi))[:, None], w[sl][:, None]], -1)
        cand = np.where(np.isfinite(cand), cand, w[sl][:, None])
        c, C = cand.shape
        V = bb[:, None, None, :] + cand[:, :, None, None] * GG[:, None]                 # (c, C, n, 6)
        pk = top2(V)
        val = np.take_along_axis(np.broadcast_to(UU[:, None], V.shape), pk, -1).sum((-1, -2))   # (c, C)
        j = val.argmax(-1)
        vbest = val[np.arange(c), j]
        upd = vbest > v_new[sl] + 1e-12
        w_new[sl] = np.where(upd, cand[np.arange(c), j], w[sl]); v_new[sl] = np.where(upd, vbest, v_new[sl])
    return w_new, v_new


def regret_opt(G, Uu, inits, seed, top_polish=3, rounds=4):
    """Per arrival, maximise sum_j U of the top-2 under b + w g (= minimise the arrival's total excess) over (b, w).
    Stage 1: for every slope of WGRID and three offset inits (per-persona mean utility = the best fixed pair; least-squares offsets
    given w; random), coordinate ascent on the offsets; plus the two least-squares solutions (logit P and U) as extra starts.
    Stage 2: the best `top_polish` solutions per arrival alternate exact slope search and offset coordinate ascent (`rounds`).
    Returns b (A, 6), w (A,), value (A,) of the best solution found (first best in start order on ties)."""
    A_, n, _ = G.shape
    rng = np.random.default_rng(seed)
    mU = Uu.mean(1)                                                 # (A, 6)
    mG = G.mean(1)
    sU = mU.std(-1, keepdims=True) + 1e-6
    starts_b, starts_w = [], []
    for b0, w0 in inits:                                            # least-squares solutions
        starts_b.append(b0); starts_w.append(w0)
    for w in WGRID:
        starts_b.append(mU); starts_w.append(np.full(A_, w))
        starts_b.append(mU - w * mG); starts_w.append(np.full(A_, w))
        for _ in range(N_RAND):
            starts_b.append(mU + rng.normal(size=(A_, K)) * (sU + abs(w) * G.std((1, 2))[:, None]))
            starts_w.append(np.full(A_, w))
    S = len(starts_b)
    Bb = np.stack(starts_b, 1)                                      # (A, S, 6)
    Ww = np.stack(starts_w, 1)                                      # (A, S)
    vals = np.zeros((A_, S))
    step = max(1, 24000 // A_)
    for s in range(0, S, step):
        sl = slice(s, min(S, s + step)); m = Bb[:, sl].shape[1]
        GG = np.repeat(G, m, 0); UU = np.repeat(Uu, m, 0)
        b, v = cd_offsets(Bb[:, sl].reshape(-1, K), Ww[:, sl].reshape(-1), GG, UU)
        Bb[:, sl] = b.reshape(A_, m, K); vals[:, sl] = v.reshape(A_, m)
    order = np.argsort(-vals, 1, kind="stable")[:, :top_polish]    # stage 2: polish the best few (first best kept on ties)
    b = np.take_along_axis(Bb, order[..., None], 1).reshape(-1, K)
    w = np.take_along_axis(Ww, order, 1).reshape(-1)
    GG = np.repeat(G, top_polish, 0); UU = np.repeat(Uu, top_polish, 0)
    v = value(b, w, GG, UU)
    for _ in range(rounds):
        w, v1 = w_search(b, w, GG, UU)
        b, v2 = cd_offsets(b, w, GG, UU)
        if np.all(v2 <= v + 1e-12):
            v = np.maximum(v, v2); break
        v = v2
    v = v.reshape(A_, top_polish)
    v0 = np.take_along_axis(vals, order, 1)
    assert np.all(v >= v0 - 1e-9)
    k = v.argmax(1)
    bb = b.reshape(A_, top_polish, K)[np.arange(A_), k]; ww = w.reshape(A_, top_polish)[np.arange(A_), k]
    return bb, ww, v[np.arange(A_), k]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", default="v3.1L", help="dataset version (v3.1L; v3.1J for the second judge)")
    ap.add_argument("--map", default=None, help="feature map [e_p, g] (default: jevgw on v3.1L, lung on v3.1J)")
    ap.add_argument("--roster", default="test_unseen")
    ap.add_argument("--all", action="store_true", help="every roster")
    ap.add_argument("--out", default=None, help="directory for oracle_<version>_<roster>.npy (per-arrival excess over the session)")
    a = ap.parse_args()
    fmap = a.map or {"v3.1L": "jevgw", "v3.1J": "lung"}[a.version]
    for roster in (ROSTERS if a.all else [a.roster]):
        R = Roster(a.version, roster, fmap)
        ex, A = R.oracle_excess()
        rnd = R.RND * R.J                                           # random pairs' expected excess per user
        print(f"{a.version} {roster:15s} oracle excess per user {ex.mean():7.3f} (random pairs {rnd:7.3f}): score "
              f"{100 * (1 - ex.mean() / rnd):5.1f}; recalibration a = {np.trace(A) / K:.3f} (mean diagonal of A)", flush=True)
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            np.save(Path(a.out) / f"oracle_{a.version}_{roster}.npy", ex)


if __name__ == "__main__":
    main()
