"""Reference rows of the ladder. Closed-form rows are computed from the hidden tables of a replay world; `ref:typelib`
is a hidden-information reference POLICY (it is handed the per-type click and utility tables of the design roster and
must only find out which type the student is). None of this is admissible as a proposed method.

All closed-form totals are for one pass over the whole roster (N = env.N arrivals, b rounds), on the [0, 1] utility
scale, as strong regret 2U* - U_i - U_j; `floor` is the unavoidable part (best pair every round)."""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from persona_bench.protocol import Policy  # noqa: E402
from persona_bench.registry import register  # noqa: E402


def _pair_regret(U):
    K = U.shape[-1]
    best = U.max(-1)
    return np.stack([2 * best - U[..., i] - U[..., j] for i, j in combinations(range(K), 2)], -1)      # (..., 15)


def closed_form(env, b: int, folds: int = 2, design_env=None) -> dict:
    """design_env: roster on which fixed pairs are chosen for the `*_design` rows (dev when env is the test roster)."""
    rounds = np.arange(b) % env.b_priv
    R = _pair_regret(env.U[:, rounds])                                  # (N, b, 15)
    top2 = np.sort(env.U[:, rounds], -1)[..., -2:]
    floor = float((top2[..., 1] - top2[..., 0]).sum())
    types = env.type_ids()
    out = {"floor": floor, "random": float(R.mean(-1).sum()), "best_fixed_pair_hindsight": float(R.sum((0, 1)).min())}
    tt = 0.0
    for m in np.unique(types):
        tt += float(R[types == m].sum((0, 1)).min())
    out["type_told_pair_hindsight"] = tt
    # cross-fitted versions: choose on one fold of arrivals, pay on the other
    rng = np.random.default_rng(0)
    fold = np.empty(env.N, dtype=int)
    for m in np.unique(types):
        ix = rng.permutation(np.where(types == m)[0]); fold[ix] = np.arange(len(ix)) % folds
    cf = ct = 0.0
    for f in range(folds):
        tr, te = fold != f, fold == f
        cf += float(R[te][:, :, R[tr].sum((0, 1)).argmin()].sum())
        for m in np.unique(types):
            a, c = tr & (types == m), te & (types == m)
            if c.any():
                ct += float(R[c][:, :, (R[a] if a.any() else R[tr]).sum((0, 1)).argmin()].sum())
    out["best_fixed_pair_crossfit"], out["type_told_pair_crossfit"] = cf, ct
    if design_env is not None:
        Rd = _pair_regret(design_env.U[:, rounds]); td = design_env.type_ids()
        out["best_fixed_pair_design"] = float(R[:, :, Rd.sum((0, 1)).argmin()].sum())
        out["type_told_pair_design"] = float(sum(R[types == m][:, :, Rd[td == m].sum((0, 1)).argmin()].sum() for m in np.unique(types)))
    out["per_student_oracle_pair"] = float(R.sum(1).min(-1).sum())      # best single pair per arrival, hindsight
    # information measures of the world (nats)
    iu = np.triu_indices(env.K, 1)
    p = np.clip(env._Pclick[:, rounds][..., iu[0], iu[1]], 1e-12, 1 - 1e-12)
    info = np.log(2) + (p * np.log(p) + (1 - p) * np.log(1 - p))
    out["info_per_click"] = float(info.mean()); out["session_info"] = float(b * info.mean())
    M = len(np.unique(types))
    if M > 1:
        Pm = np.stack([p[types == m].mean((0, 1)) for m in np.unique(types)])                # (M, 15) type-level click tables
        kl = [(Pm[a] * np.log(Pm[a] / Pm[c]) + (1 - Pm[a]) * np.log((1 - Pm[a]) / (1 - Pm[c]))).mean() for a in range(M) for c in range(M) if a != c]
        out["I_type"] = float(np.mean(kl)); out["rho_identification"] = float(b * np.mean(kl) / np.log(M))
    out["N"], out["b"], out["M"] = int(env.N), int(b), int(M)
    return out


class TypeLibOracle(Policy):
    """Knows, for every profile type of the DESIGN roster, the mean click table and mean utilities (question-independent).
    Does not know the student's type: keeps a Bayesian posterior over types from the clicks and shows the two personas
    with the highest posterior-mean utility. Reference for 'how far does pure type identification get'."""
    population = "reference"

    def __init__(self, name="ref:typelib", told: bool = False):
        self.name, self.told = name, told

    def reset(self, rng, context):
        super().reset(rng, context)
        lib = context["typelib"]
        self.P, self.U, self.w0 = np.clip(lib["P"], 1e-6, 1 - 1e-6), lib["U"], lib["weights"]

    def start_student(self):
        self.logw = np.log(self.w0)

    def select(self, Phi):
        w = np.exp(self.logw - self.logw.max()); w /= w.sum()
        if self.name.endswith(":ts"):        # Thompson sampling over types: show the favourite pair of a sampled type
            order = np.argsort(-self.U[self.rng.choice(len(w), p=w)])
            self._last = (int(order[0]), int(order[1]))
            return self._last
        if self.name.endswith(":tts"):       # top-two over types: favourite of the most probable type and of a sampled rival type
            k1 = int(np.argmax(w)); i = int(np.argmax(self.U[k1]))
            u2 = self.U[self.rng.choice(len(w), p=w)].copy(); u2[i] = -np.inf
            self._last = (i, int(np.argmax(u2)))
            return self._last
        order = np.argsort(-(w @ self.U))
        self._last = (int(order[0]), int(order[1]))
        return self._last

    def update(self, outcome):
        i, j = self._last
        self.logw = self.logw + np.log(self.P[:, i, j] if outcome else 1 - self.P[:, i, j])

    def end_student(self):
        pass

    def diagnostics(self):
        return {"k_hat": int(np.argmax(self.logw))}


register("ref", lambda spec: TypeLibOracle(name=spec))
