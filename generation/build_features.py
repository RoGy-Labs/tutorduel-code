#!/usr/bin/env python
"""Rebuild the rater feature maps phi(p, q) = [e_p, g(p, q)] (questions x 6 personas x 7) and their scales S from the raw rater
outputs, and check them against the shipped arrays.

    python generation/build_features.py               # checks results/features/{v3.1L/*/phi_jevgw, v3.1J/*/phi_lung}.npy and dev/S_*.json
    python generation/build_features.py --out <dir>   # also writes <dir>/<version>/<split>/phi_<map>.npy and <dir>/<version>/dev/S_<map>.json

Input: dataset/rater/, one gzipped JSONL row per question of the split, keyed by qid (the index into
dataset/texts/<split>/packs.json["questions"]); lists follow the persona order P0_proof ... P5_stepbystep.
  jevgw (used with v3.1L): jev_worst_<split>.jsonl.gz, {"qid", "p_worst": P_Jev(worst style = p | message) for the six styles}
                           (generation/rate_jev.py); g = -log max(p_worst, 1e-3).
  lung (used with v3.1J):  luna_badly_<split>.jsonl.gz, {"qid", "top_logprobs": for each style the [token, logprob] pairs that
                           gpt-6-luna returned at the first token of its 0-10 'how badly' rating (generation/rate_luna.py)};
                           g = -(expected rating), the listed tokens 0..10 renormalised.
g is centred over the six personas of each question and divided by its standard deviation over all dev cells (the dev
constant is reused on test). S = median norm of the per-user least-squares fits of the logit clicks on the feature contrasts
on the dev roster of the map's dataset version (persona_bench.replay.linear_fits / calibrate_S).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persona_bench.replay import calibrate_S, linear_fits  # noqa: E402

EPS = 1e-3


def rows(name, split):
    with gzip.open(ROOT / "dataset" / "rater" / f"{name}_{split}.jsonl.gz", "rt") as f:
        return [json.loads(line) for line in f]


def table(rs, value):
    X = np.full((len(rs), 6), np.nan)
    for r in rs:
        X[r["qid"]] = value(r)
    assert not np.isnan(X).any(), "questions without a rating"
    return X


def expected(top):
    v = [(int(t.strip()), np.exp(lp)) for t, lp in top if t.strip().isdigit() and int(t.strip()) <= 10]
    w = np.array([b for _, b in v])
    return float(np.dot([a for a, _ in v], w / w.sum()))


def g_jev(split):
    return -np.log(np.clip(table(rows("jev_worst", split), lambda r: r["p_worst"]), EPS, 1))


def g_luna(split):
    return table(rows("luna_badly", split), lambda r: -np.array([expected(t) for t in r["top_logprobs"]]))


MAPS = {"jevgw": ("v3.1L", g_jev), "lung": ("v3.1J", g_luna)}           # map -> (dataset version, raw g)


def build(g):
    D, T = (G - G.mean(1, keepdims=True) for G in (g("dev"), g("test")))
    sd = float(D.std())
    phi = lambda G: np.concatenate([np.broadcast_to(np.eye(6), (len(G), 6, 6)), G[..., None]], -1)
    return dict(dev=phi(D / sd), test=phi(T / sd)), sd


def scale(version, Phi):
    d = ROOT / "dataset" / version / "types" / "dev"
    P, qids = np.load(d / "P.npy"), np.load(d / "qids.npy")
    return calibrate_S(linear_fits(P, lambda s, t: Phi[int(qids[s, t])], Phi.shape[1], Phi.shape[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for name, (version, g) in MAPS.items():
        arrays, sd = build(g)
        F = ROOT / "results" / "features" / version
        for split, Phi in arrays.items():
            f = F / split / f"phi_{name}.npy"
            err = f"{np.abs(Phi - np.load(f)).max():.1e}" if f.exists() and np.load(f).shape == Phi.shape else "not shipped / shape differs"
            print(f"{version} {split:5s} phi_{name} {Phi.shape}   max |rebuilt - shipped| {err}")
        S = scale(version, arrays["dev"])
        meta = json.load(open(F / "dev" / f"S_{name}.json"))
        print(f"{version} dev   S_{name} {S!r}   shipped {meta['S']!r}   |diff| {abs(S - meta['S']):.1e}   (dev sd of g {sd:.4f})")
        if a.out:
            for split, Phi in arrays.items():
                o = Path(a.out) / version / split; o.mkdir(parents=True, exist_ok=True)
                np.save(o / f"phi_{name}.npy", Phi)
            (Path(a.out) / version / "dev" / f"S_{name}.json").write_text(json.dumps(dict(meta, S=S), indent=1))


if __name__ == "__main__":
    main()
