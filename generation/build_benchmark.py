#!/usr/bin/env python
"""Rebuild the benchmark tables of every roster from the raw judge calls and check them against the shipped arrays.

    python generation/build_benchmark.py                          # gpt-6-luna judge: checks dataset/v3.1L/types/<roster>/{P,U,Pcbt}.npy
    python generation/build_benchmark.py --judge jev              # second judge (TypeSafe Jev): checks dataset/v3.1J/types/<roster>/{P,U}.npy
    python generation/build_benchmark.py --out <dir>              # also writes the rebuilt arrays (incl. V, U_stretch) to <dir>/<roster>/
    python generation/build_benchmark.py --check-against <dir>    # also compares with every <dir>/<roster>/{P,V,U,U_stretch,Pcbt}.npy present
    python generation/build_benchmark.py --compact <calls.jsonl> dataset/judge_calls_<judge>/<roster>/calls.jsonl.gz
                                                                  # native rows of a judge client -> the shipped call file

Input: dataset/judge_calls_<judge>/<roster>/calls.jsonl.gz, one gzipped JSONL row per prompt, i.e. per (arrival, question j of
the arrival, persona pair i < k, display order pos, label lab): pos = 1 if persona i's answer is shown first, lab = 1 if the
first-shown answer is labelled A; then
  luna (v3.1L): n_i, n = picks of persona i's answer among the n valid of 4 sampled replies (generation/judge_luna.py);
                per prompt the smoothed log-odds log((n_i + 1/2) / (n - n_i + 1/2)); prompts without a valid pick are skipped.
  jev (v3.1J):  p_i, p_k = the judge's probabilities of the labels carried by the answers of personas i and k
                (generation/judge_jev.py); per prompt log(p / (1 - p)), p = p_i / (p_i + p_k) clipped to [0.005, 0.995].
--compact keeps exactly these fields of the clients' native rows (judge_luna.py also writes id, tin, tout), one row per prompt
(the last if a prompt was called twice), in request order. The 4 prompts of a pair are averaged in logit space, which gives the
click probabilities P. Scores V are ridge Bradley-Terry fits to P, utilities U = 1 - (v_max - v) / R with R = 8 logits, and
Pcbt = sigma(v_i - v_j). V and U_stretch are not shipped (derivable; --out writes them).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persona_bench.utility import bt_scores, cbt_table, stretch, top_anchored  # noqa: E402

PAIRS = list(combinations(range(6), 2))
ROSTERS = ["dev", "test", "test_unseen", "test_continuum", "test_idio"]
JUDGES = {"luna": "v3.1L", "jev": "v3.1J"}                                     # judge -> dataset version
KEY = ("arrival", "j", "i", "k", "pos", "lab")
CLIP = 0.005                                                                   # jev: per-prompt probability clip


def table_from_rows(rows, n_units):
    """rows with unit, i, j, pos, lab, y -> P[n_units, 6, 6] (logit average of the 4 prompts of each pair)."""
    Y = np.full((n_units, len(PAIRS), 2, 2), np.nan)
    pi = {p: n for n, p in enumerate(PAIRS)}
    for u, i, j, pos, lab, y in rows:
        Y[u, pi[(i, j)], (1 - pos) // 2, (1 - lab) // 2] = y
    s = np.nanmean(Y, axis=(-1, -2))
    P = np.full((n_units, 6, 6), 0.5)
    for n, (i, j) in enumerate(PAIRS):
        P[:, i, j] = expit(s[:, n]); P[:, j, i] = 1 - P[:, i, j]
    return P, int(np.isnan(Y).sum())


def log_odds(r):
    """Per-prompt log-odds that persona i's answer is preferred; None for a luna prompt without a valid pick."""
    if "n_i" in r:                                                             # luna: n_i of n sampled picks
        return float(np.log((r["n_i"] + 0.5) / (r["n"] - r["n_i"] + 0.5))) if r["n"] > 0 else None
    p = r["p_i"] / (r["p_i"] + r["p_k"]) if r["p_i"] + r["p_k"] > 0 else 0.5    # jev: probabilities of the two labels
    p = min(max(p, CLIP), 1 - CLIP)
    return float(np.log(p / (1 - p)))


def compact(src, dst):
    """Native call rows (.jsonl or .jsonl.gz) -> the shipped call file (fields, order and dedup as in the module docstring)."""
    rows = {}
    with (gzip.open if str(src).endswith(".gz") else open)(src, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[tuple(r[k] for k in KEY)] = {k: r[k] for k in KEY + (("n_i", "n") if "n_i" in r else ("p_i", "p_k"))}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        for key in sorted(rows, key=lambda t: (t[0], t[1], t[2], t[3], -t[4], -t[5])):
            f.write((json.dumps(rows[key]) + "\n").encode())
    return len(rows)


def build(roster, judge="luna"):
    version = JUDGES[judge]
    R = json.load(open(ROOT / "dataset" / version / "MANIFEST.json"))["R_logit"]
    ix = json.load(open(ROOT / "dataset" / version / "types" / roster / "index.json")); n_arr, b = ix["n_arrivals"], ix["b"]
    rows = []
    with gzip.open(ROOT / "dataset" / f"judge_calls_{judge}" / roster / "calls.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line); y = log_odds(r)
            if y is not None and r["arrival"] < n_arr:
                rows.append((r["arrival"] * b + r["j"], r["i"], r["k"], r["pos"], r["lab"], y))
    P, missing = table_from_rows(rows, n_arr * b)
    P = P.reshape(n_arr, b, 6, 6)
    V = bt_scores(P)
    return dict(P=P, V=V, U=top_anchored(V, R), U_stretch=stretch(V), Pcbt=cbt_table(V)), missing


def max_diff(arrays, d):
    return "  ".join(f"{k} {np.abs(v - np.load(d / f'{k}.npy')).max():.1e}" for k, v in arrays.items() if (d / f"{k}.npy").exists())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="luna", choices=list(JUDGES), help="luna: dataset v3.1L; jev: the second judge, v3.1J")
    ap.add_argument("--rosters", default=",".join(ROSTERS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-against", default=None, help="folder with <roster>/<array>.npy to compare with as well")
    ap.add_argument("--compact", nargs=2, metavar=("NATIVE", "OUT"), help="convert a client's call rows to the shipped format, then exit")
    a = ap.parse_args()
    if a.compact:
        print(f"{compact(*a.compact)} prompts -> {a.compact[1]}")
        return
    version = JUDGES[a.judge]
    for roster in a.rosters.split(","):
        arrays, missing = build(roster, a.judge)
        print(f"{roster:15s} missing prompts {missing}   max |rebuilt - shipped|: " + max_diff(arrays, ROOT / "dataset" / version / "types" / roster))
        if a.check_against:
            print(f"{'':15s} max |rebuilt - {a.check_against}|: " + max_diff(arrays, Path(a.check_against) / roster))
        if a.out:
            o = Path(a.out) / roster; o.mkdir(parents=True, exist_ok=True)
            for k, v in arrays.items():
                np.save(o / f"{k}.npy", v)


if __name__ == "__main__":
    main()
