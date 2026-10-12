"""Agreement analyses of the appendix: the judges against two authors' choices, and the second judge against the dataset's judge.

    python results/agreement.py human              # tab:agreement (a)-(c) and the numbers of Appendix "Agreement with human choices"
    python results/agreement.py judges             # tab:second-judge and the numbers of Appendix "A second judge"
    python results/agreement.py all --macros       # both, then every number as name = value (the names of the paper's macros)

human: two authors chose, for sampled comparisons, which of two tutor answers the described student would rather receive. Inputs:
  dataset/human_labels/labels.jsonl       one row per comparison labeled by both authors (204 comparisons, 51 (student, question)
      cells of 4 persona pairs): id, cell, roster, arrival, j (question index within the arrival), qid, i < k (the two personas,
      indices as in P.npy), left = [persona shown on the left, as answer A, to author 1, to author 2], choice = [persona chosen by
      author 1, by author 2]. The interface showed the student profile (dataset/v3.1L/types/cartoons.json, profile of the arrival in
      index.json), the question with its cue and the two answers (dataset/texts/<dev|test>/answers.jsonl, by qid and persona name).
  dataset/human_labels/chat_judges.jsonl  id, then for gpt-5.6-sol and claude-opus-5 (chat models, reasoning off) the picks of persona i's
      answer among 4 sampled replies to each of the dataset judge's 4 prompts, in the order (display order, label) = (i first, i is A),
      (i first, k is A), (k first, k is A), (k first, i is A); every reply was a valid pick. Pair log-odds: mean of the 4 prompts'
      log((n_i + 1/2) / (4 - n_i + 1/2)), as for the dataset judge.
  dataset/v3.1L/types/<roster>/P.npy (gpt-6-luna, the dataset's judge) and dataset/v3.1J/types/<roster>/P.npy (TypeSafe Jev), read
      at [arrival, j, i, k]; a judge's favorite is persona i if its log-odds are > 0 (a tie counts as persona k).
A comparison counts once, with the two authors' choices averaged. Intervals: 95% percentile intervals of a cluster bootstrap over
the 51 cells, 4000 draws, seed 0.

judges: every comparison of the three evaluation rosters (held-out profiles, Composite, Quirky) judged by gpt-6-luna (v3.1L) and by
TypeSafe Jev (v3.1J); reads P.npy, U.npy and qids.npy of both versions. Fixed-pair scores: 100 (R_rand - R) / (R_rand - G_T) of each
student's best fixed pair under Jev's utilities, with the pair chosen from either judge's utilities.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset"
HL = DS / "human_labels"
CHAT = [("gpt-5.6-sol", "Sol"), ("claude-opus-5", "Opus")]
EVAL = [("U", "test_unseen", "held-out profiles"), ("C", "test_continuum", "Composite"), ("I", "test_idio", "Quirky")]
NB, HIGH, CAP = 4000, 0.8, 0.9                         # bootstrap draws; "confident"; the dataset judge's maximum probability
BANDS = [(0.5, 0.7), (0.7, CAP), (CAP, 1.01)]          # on the judge's probability of its favorite, rounded to 2 decimals
SAMPLE = (300, 75)                                     # comparisons and cells drawn; both authors labeled every comparison of 51 cells
PAIRS = list(itertools.combinations(range(6), 2))
IU = np.triu_indices(6, 1)
M = {}                                                 # every reported number, by the name of its macro in the paper
logit = lambda p: np.log(p / (1 - p))
pc = lambda x: f"{100 * x:.1f}"


def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((np.mean(a == b) - pe) / (1 - pe))


def table(head, body, title):
    w = [max(len(str(r[j])) for r in [head] + body) for j in range(len(head))]
    print(f"\n{title}")
    for r in [head] + body:
        print("  " + "  ".join(str(x).ljust(w[j]) if j == 0 else str(x).rjust(w[j]) for j, x in enumerate(r)))


def put(name, t, f=pc):
    M[name], M[name + "Lo"], M[name + "Hi"] = f(t[0]), f(t[1]), f(t[2])
    return f"{f(t[0])} [{f(t[1])}, {f(t[2])}]"


def rep_human():
    rows = sorted((json.loads(l) for l in open(HL / "labels.jsonl")), key=lambda r: r["id"])
    chat = {r["id"]: r for r in map(json.loads, open(HL / "chat_judges.jsonl"))}
    arr = {(v, f, s): np.load(DS / v / "types" / s / f"{f}.npy", mmap_mode="r") for v, f in (("v3.1L", "P"), ("v3.1J", "P"), ("v3.1L", "qids"))
           for s in {r["roster"] for r in rows}}
    assert all(arr["v3.1L", "qids", r["roster"]][r["arrival"], r["j"]] == r["qid"] for r in rows)
    z = {t: np.array([float(logit(arr[v, "P", r["roster"]][r["arrival"], r["j"], r["i"], r["k"]])) for r in rows]) for t, v in (("luna", "v3.1L"), ("jev", "v3.1J"))}
    for m, t in CHAT:
        n = np.array([chat[r["id"]][m] for r in rows], float)
        z[t] = np.log((n + .5) / (4 - n + .5)).mean(1)
        z[t][np.abs(z[t]) < 1e-9] = 0.0                # evidence that cancels exactly is a tie, whatever the summation order
    y = np.array([[float(c == r["i"]) for c in r["choice"]] for r in rows])     # 1 if the author chose persona i; columns: author 1, 2
    fav = {t: (zz > 0).astype(float) for t, zz in z.items()}
    hit = {t: (y == f[:, None]).mean(1) for t, f in fav.items()}               # share of the two authors' choices the judge matches
    conf = 1 / (1 + np.exp(-np.abs(z["luna"])))                                  # gpt-6-luna's probability of its favorite
    rconf = np.array([round(float(c), 2) for c in conf])                         # stored values at the maximum are 0.9 +- float error
    cells = sorted({r["cell"] for r in rows}); cell = np.array([cells.index(r["cell"]) for r in rows])
    by = [np.flatnonzero(cell == u) for u in range(len(cells))]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by[u] for u in rng.choice(len(cells), len(cells))]) for _ in range(NB)]
    allx = np.arange(len(rows))

    def ci(stat):
        bs = [stat(d) for d in draws]
        return stat(allx), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))

    def mean(x, d, keep=None):
        d = d if keep is None else d[keep[d]]
        return float(np.mean(x[d])) if len(d) else np.nan

    def auc(d, t="luna"):
        s, yy = np.repeat(z[t][d], 2), y[d].ravel()
        pos, neg = s[yy == 1], s[yy == 0]
        return float(np.mean((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])))

    n = len(rows); y1, y2 = y[:, 0], y[:, 1]
    M.update(AgrSamplePairs=str(SAMPLE[0]), AgrSampleUnits=str(SAMPLE[1]), AgrCells=str(len(cells)), AgrPairs=str(n), AgrLabels=str(2 * n))
    print(f"{SAMPLE[0]} comparisons drawn in {SAMPLE[1]} (student, question) cells; both authors labeled the {n} comparisons of {len(cells)} "
          f"cells: {2 * n} choices, {n} each")

    # (a) pairs of raters
    rater = (y1 == y2).astype(float)
    a = [("author 1 and author 2", put("AgrRaterAgreeCi", ci(lambda d: mean(rater, d))), f"{kappa(y1, y2):.2f}")]
    M["AgrRaterAgree"], M["AgrKappaRaters"] = pc(rater.mean()), a[0][2]
    for t, name, key in (("luna", "gpt-6-luna", "Judge"), ("jev", "TypeSafe Jev", "Jev")):
        for r_, yr in (("one", y1), ("two", y2)):
            M[f"Agr{key}R{r_}"], M[f"AgrKappa{key}R{r_}"] = pc(np.mean(fav[t] == yr)), f"{kappa(fav[t], yr):.2f}"
            a.append((f"{name} and author {1 if r_ == 'one' else 2}", M[f"Agr{key}R{r_}"], M[f"AgrKappa{key}R{r_}"]))
    a.append(("gpt-6-luna and TypeSafe Jev", put("AgrJudgesAgree", ci(lambda d: float(np.mean(fav["luna"][d] == fav["jev"][d])))),
              f"{kappa(fav['luna'], fav['jev']):.2f}"))
    both = y1 == y2
    M["AgrBothAgreeN"], M["AgrBothAgreeJudge"] = str(both.sum()), str(int((fav["luna"][both] == y1[both]).sum()))
    a.append((f"gpt-6-luna, where the authors agree ({M['AgrBothAgreeN']})", f"{100 * int(M['AgrBothAgreeJudge']) / both.sum():.1f}", ""))
    a.append(("AUC of the log-odds of gpt-6-luna", put("AgrAuc", ci(auc), lambda x: f"{x:.2f}"), ""))
    table(["pair", "agreement (%)", "Cohen's kappa"], a, "(a) Pairs of raters [95% cluster bootstrap]")
    put("AgrDiff", ci(lambda d: mean(hit["luna"], d) - mean(rater, d)), lambda x: f"{100 * x:+.1f}")
    luna_agree = put("AgrJudgeAgree", ci(lambda d: mean(hit["luna"], d)))
    M["AgrJevAgree"] = pc(hit["jev"].mean())
    print(f"  gpt-6-luna with the authors minus the authors with each other: {M['AgrDiff']} points [{M['AgrDiffLo']}, {M['AgrDiffHi']}]; "
          f"where the authors agree, gpt-6-luna picks their answer in {M['AgrBothAgreeJudge']} of {M['AgrBothAgreeN']}")
    print(f"  agreement with the authors' choices: TypeSafe Jev {M['AgrJevAgree']}%, gpt-6-luna {M['AgrJudgeAgree']}%")

    # (b) by the confidence of gpt-6-luna
    bands, b = [], []
    for lo, hi in BANDS:
        keep = (lo <= rconf) & (rconf < hi)
        h = ci(lambda d: mean(hit["luna"], d, keep))
        bands.append((keep.sum(), float(np.mean(conf[keep])), h, float(np.mean(rater[keep])), float(np.mean(conf[keep] ** 2 + (1 - conf[keep]) ** 2))))
        b.append(["0.9 (maximum)" if lo == CAP else f"[{lo:.1f}, {hi:.1f})", bands[-1][0], f"{bands[-1][1]:.2f}",
                  f"{pc(h[0])} [{pc(h[1])}, {pc(h[2])}]", f"{100 * bands[-1][3]:.0f}, {100 * bands[-1][4]:.0f}"])
    table(["probability of its favorite", "comparisons", "mean", "authors choose it (%)", "authors agree (%), expected"], b,
          "(b) By the confidence of gpt-6-luna [95% cluster bootstrap]")
    top = rconf >= HIGH
    M.update(AgrBinTopP=f"{HIGH:.1f}", AgrBinTopShare=f"{100 * top.sum() / n:.0f}", AgrBinTopHit=f"{100 * mean(hit['luna'], allx, top):.0f}",
             AgrBinMidHit=f"{100 * bands[1][2][0]:.0f}", AgrBinMidP=f"{bands[1][1]:.2f}", AgrEce=f"{100 * sum(k / n * abs(h[0] - p) for k, p, h, _, _ in bands):.0f}",
             AgrBinLowRater=f"{100 * bands[0][3]:.0f}", AgrBinCapRater=f"{100 * bands[-1][3]:.0f}",
             AgrBinLowExp=f"{100 * bands[0][4]:.0f}", AgrBinCapExp=f"{100 * bands[-1][4]:.0f}")
    print(f"  probability >= {M['AgrBinTopP']}: {M['AgrBinTopShare']}% of comparisons, the authors choose the favorite in {M['AgrBinTopHit']}%; "
          f"[0.7, 0.9): {M['AgrBinMidHit']}% against a mean of {M['AgrBinMidP']}; expected calibration error {M['AgrEce']} points")

    # (c) other judges
    c = [["gpt-6-luna (this dataset)", n, luna_agree, "--"]]
    conf_hi = conf >= HIGH
    for m, t in CHAT:
        c.append([f"{m} (chat)", n, put(f"AgrAgree{t}", ci(lambda d: mean(hit[t], d))), pc(np.mean(fav[t] == fav["luna"]))])
        M[f"AgrSameFav{t}"] = c[-1][3]
        put(f"AgrDiff{t}", ci(lambda d: mean(hit[t], d) - mean(hit["luna"], d)), lambda x: f"{100 * x:+.1f}")
        M[f"AgrSameFavHigh{t}"] = pc(np.mean(fav[t][conf_hi] == fav["luna"][conf_hi]))
    table(["judge", "comparisons", "agreement (%)", "same favorite (%)"], c, "(c) Other judges, 4 prompts x 4 sampled picks [95% cluster bootstrap]")
    for m, t in CHAT:
        print(f"  {m} minus gpt-6-luna: {M[f'AgrDiff{t}']} points [{M[f'AgrDiff{t}Lo']}, {M[f'AgrDiff{t}Hi']}]; same favorite where "
              f"gpt-6-luna's probability >= {HIGH}: {M[f'AgrSameFavHigh{t}']}%")


def sj_stats(L, J):
    """agreement of the second judge (J) with the dataset's judge (L) on one roster (arrays of the same arrivals and questions)."""
    assert np.array_equal(L["qids"], J["qids"]), "the two judges must see the same questions"
    pl, pj = (x["P"][..., IU[0], IU[1]].ravel() for x in (L, J))
    nt = (pl != .5) & (pj != .5)                                           # neither judge indifferent
    same = (pl > .5) == (pj > .5)
    top = np.abs(pl - .5) >= .4 - 1e-9                                      # gpt-6-luna at its maximum probability 0.9
    N = L["U"].shape[0]
    reg = {k: np.stack([(2 * x["U"].max(-1) - x["U"][..., a] - x["U"][..., b]).sum(1) for a, b in PAIRS], 1) for k, x in (("L", L), ("J", J))}
    srt = np.sort(J["U"], -1); fl = (srt[..., -1] - srt[..., -2]).sum()    # floor G_T under Jev
    rnd = reg["J"].mean(1).sum()                                            # random distinct pairs under Jev
    pick = lambda k: reg["J"][np.arange(N), reg[k].argmin(1)].sum()        # each student's best fixed pair under judge k, paid under Jev
    return dict(n=pl.size, same=same[nt].sum(), strict=nt.sum(), same_top=(same & top & nt).sum(), top=(top & nt).sum(), top_all=top.sum(),
                cross=(pl * pj + (1 - pl) * (1 - pj)).sum(), self_l=(pl ** 2 + (1 - pl) ** 2).sum(), self_j=(pj ** 2 + (1 - pj) ** 2).sum(),
                top_persona=(L["U"].sum(1).argmax(1) == J["U"].sum(1).argmax(1)).sum(), students=N,
                transfer=100 * (rnd - pick("L")) / (rnd - fl), own=100 * (rnd - pick("J")) / (rnd - fl))


def rep_judges():
    load = lambda v, s: {f: np.load(DS / v / "types" / s / f"{f}.npy") for f in ("P", "U", "qids")}
    per = {k: sj_stats(load("v3.1L", s), load("v3.1J", s)) for k, s, _ in EVAL}
    tot = {f: sum(per[k][f] for k in per) for f in ("n", "same", "strict", "same_top", "top", "top_all", "cross", "self_l", "self_j", "top_persona", "students")}
    rat = lambda a, b, d=1: f"{100 * a / b:.{d}f}"
    M.update(SJAgreeComparisons=f"{tot['n']:,}", SJStudents=f"{tot['students']:,}", SJAgree=rat(tot["same"], tot["strict"]),
             SJAgreeTop=rat(tot["same_top"], tot["top"]), SJAgreeTopShare=rat(tot["top_all"], tot["n"], 0),
             SJClickCross=f"{tot['cross'] / tot['n']:.2f}", SJClickSelf=f"{tot['self_l'] / tot['n']:.2f}", SJClickSelfJ=f"{tot['self_j'] / tot['n']:.2f}",
             SJTopPersona=rat(tot["top_persona"], tot["students"], 0), SJTopPersonaChance=f"{100 / 6:.0f}")
    for k in per:
        M.update({f"SJAgree{k}": rat(per[k]["same"], per[k]["strict"]), f"SJTopPersona{k}": rat(per[k]["top_persona"], per[k]["students"], 0),
                  f"SJTransfer{k}": f"{per[k]['transfer']:.0f}", f"SJTransferOwn{k}": f"{per[k]['own']:.0f}"})
    row = lambda lab, f, pooled="": [lab] + [f(per[k]) for k in per] + [pooled]
    body = [row("same favored answer, comparisons (%)", lambda x: rat(x["same"], x["strict"]), M["SJAgree"]),
            row("  where gpt-6-luna gives 0.9 (%)", lambda x: rat(x["same_top"], x["top"]), M["SJAgreeTop"]),
            row("  share of comparisons where gpt-6-luna gives 0.9 (%)", lambda x: rat(x["top_all"], x["n"], 0), M["SJAgreeTopShare"]),
            row("click agreement, Jev with gpt-6-luna", lambda x: f"{x['cross'] / x['n']:.2f}", M["SJClickCross"]),
            row("  two gpt-6-luna clicks", lambda x: f"{x['self_l'] / x['n']:.2f}", M["SJClickSelf"]),
            row("  two Jev clicks", lambda x: f"{x['self_j'] / x['n']:.2f}", M["SJClickSelfJ"]),
            row("same most-preferred persona, students (%)", lambda x: rat(x["top_persona"], x["students"], 0), M["SJTopPersona"]),
            row("score of each student's best pair under gpt-6-luna, paid under Jev", lambda x: f"{x['transfer']:.0f}"),
            row("  its best pair under Jev", lambda x: f"{x['own']:.0f}")]
    table(["statistic"] + [h for _, _, h in EVAL] + ["all three"], body,
          f"TypeSafe Jev (v3.1J) against gpt-6-luna (v3.1L): {M['SJAgreeComparisons']} comparisons of two personas' answers, {M['SJStudents']} students")
    print(f"  favored answer: comparisons where neither judge gives 0.5; same most-preferred persona by chance: {M['SJTopPersonaChance']}%")


REPORTS = {"human": rep_human, "judges": rep_judges}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?", default="all", choices=list(REPORTS) + ["all"])
    ap.add_argument("--macros", action="store_true", help="also print every number as name = value (the paper's macro names)")
    a = ap.parse_args()
    for name in (REPORTS if a.report == "all" else [a.report]):
        print(f"\n===== {name} =====")
        REPORTS[name]()
    if a.macros:
        print("\n===== numbers by macro name =====")
        print("".join(f"  {k} = {v}\n" for k, v in sorted(M.items())), end="")


if __name__ == "__main__":
    main()
