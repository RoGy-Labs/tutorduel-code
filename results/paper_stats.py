"""The paper's tables from the run rows: benchmark scores, reference rows, paired comparisons with seed t-intervals and the
crossed bootstrap over seeds and users, and the summaries of the appendices.

    python results/paper_stats.py main                 # Table 1 (held-out profiles, Composite, Quirky) and the paired comparisons
    python results/paper_stats.py all                  # every report below
    python results/paper_stats.py main --from-runs     # the same from rows you ran yourself (results/runs/<group>/, see README)

Reports (paper table or figure in brackets):
    main            benchmark results on the three evaluation rosters and paired comparisons      [tab:main, tab:main-gains]
    topic           held-out-topic and development rosters                                       [tab:topic, tab:topic-gains]
    references      diagnostic references                                                        [tab:references]
    profiles        held-out-profile roster by profile, and the bootstrap over profiles          [tab:profiles]
    phases          excess regret per user in the first and last quarter of the arrivals         [tab:phases]
    redrawn         rosters redrawn for every seed                                                [tab:resampled]
    ablations       one-component changes on the development roster                              [tab:ablations]
    ablations-eval  one-component changes on the evaluation rosters                              [tab:ablations-heldout]
    sensitivity     settings of CODuel changed one at a time                                     [tab:sensitivity]
    controls        negative controls, the evidence gate, pair rules with the profiles known     [tab:controls, tab:ablation-gate, tab:ablation-library]
    grid            click sharpness x session length on the development and scrambled rosters    [fig:llm-maps, fig:llm-maps-typelib]
    sim             simulator anchors, the b x beta grid, COLDB with long sessions               [tab:sim and Appendix "Simulator anchors"]
    tuning          tuning of the comparison methods                                             [tab:tuning]
    tuning-eval     settings chosen on the evaluation rosters, and the double-sample pair rule   [tab:tuning-oracle, tab:ts-pair]
    second-judge    the five rosters judged by TypeSafe Jev                                      [tab:sj-main, tab:sj-main-gains, tab:sj-topic]

Rows: results/rows/<group>.npz (shipped: one row per (world, seed, policy) with the excess regret R_T - G_T, the floor G_T, and,
where a report needs them, the per-user excess regret in arrival order with the arrival and profile ids). With --from-runs the
same groups are read from results/runs/<group>/ written by results/bench.py --tag <group>.

Score (higher is better): 100 (R_rand - R) / (R_rand - G_T), with R_rand the expected regret of uniformly random distinct pairs
and G_T the floor; both are closed-form (results/refs.py). The per-user linear oracle is read from results/oracle/
(results/oracle.py writes it). Paired rows: the gain of CODuel over a competitor (competitor excess - CODuel excess) as a share
of the competitor's excess, with the seed t-interval and the crossed bootstrap (2,000 replicates; seeds and users resampled,
users within profiles on the rosters of recurring profiles).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "results"))
os.environ.setdefault("PERSONA_ALLOW_TEST", "1")                  # the reports only read the rosters' utilities
from refs import closed_form  # noqa: E402
from stats import bh, crossed_bootstrap, t_interval  # noqa: E402
from worlds import _llm_env, build, parse_world, resample_indices  # noqa: E402

S2O = ("rule=kg:mstep=pooled:comp_var=0:n_comp=16:birth_margin=0:births=3:alpha=20:peg=0.5:peg_min=8:w_iso=0:n_init=1:em_late=1"
       ":cap=0:cap_v0=1000")
CODUEL = {V: f"qo:{S2O}:xcols=1:recal=1:rfull=1:xpass=0:lama=10:aug=0:sa={sa}" for V, sa in (("v3.1L", "1.6996"), ("v3.1J", "2.5019"))}
CODUEL_SIM = f"v3:{S2O}:aug=0"                                    # CODuel on the simulator (no rater score to recalibrate)
NORECAL = f"v3:{S2O}:aug=0"                                       # CODuel without recalibration (A = I) on the tutoring benchmark
NT = "none:emax:rt"
FMAP = {"v3.1L": "jevgw", "v3.1J": "lung"}
# comparison methods with the configurations chosen on the tuning seeds (results/tuning_grids.json, `python results/paper_stats.py tuning`)
TUNED = {"v3.1L": dict(rucb="rucb:a0.51", pop="lab:pop:aug=0", colstim="colstim:c1:l0.1:g0.3", coldb="coldb:a0.03:c0.03", coldbl="coldbl:t2:c0.3",
                       metats="lab:metats:s0=0.001:sq=0.1:aug=0", hierts="lab:hierts:s0=0.001:sq=1:aug=0", adats="lab:adats:s0=0.0003:sq=10:aug=0"),
         "v3.1J": dict(rucb="rucb:a0.51", pop="lab:pop:aug=0", colstim="colstim:c3:l0.03:g0.3", coldb="coldb:a0.03:c0.03", coldbl="coldbl:t0.5:c0.3",
                       metats="lab:metats:s0=0.0003:sq=1:aug=0", hierts="lab:hierts:s0=0.0003:sq=1:aug=0", adats="lab:adats:s0=0.003:sq=1:aug=0"),
         "sim": dict(rucb="rucb:a1", pop="lab:pop:aug=0", colstim="colstim:c0.3:l3:g0.01", coldb="coldb:a0.1:c0", coldbl="coldbl:t0.5:c0.3",
                     metats="lab:metats:s0=0.03:sq=1:aug=0", hierts="lab:hierts:s0=0.01:sq=0.1:aug=0", adats="lab:adats:s0=0.03:sq=1:aug=0")}
NAMES = dict(rucb="RUCB", pop="Pooled", colstim="CoLSTIM", coldb="COLDB", coldbl="COLDB-LR", metats="Meta-TS", hierts="HierTS", adats="AdaTS")
MAIN_CMP = ("rucb", "pop", "colstim", "coldb", "coldbl", "metats", "hierts")      # Table 1 (AdaTS: appendix tables only)
# roster key -> (dataset roster, name, seeds in the paper, users resampled within profiles)
ROSTERS = {"U": ("test_unseen", "held-out profiles", 64, True), "C": ("test_continuum", "Composite", 32, False),
           "I": ("test_idio", "Quirky", 32, False), "T": ("test", "held-out topics", 64, True), "D": ("dev", "development", 32, True)}
ROWS = ROOT / "results" / "rows"
RUNS = ROOT / "results" / "runs"
FROM_RUNS = False


def world(V, k, mid="M8_t1_b24_ug_raw_rl1", suffix=""):
    return f"pl_{V}_{ROSTERS[k][0]}_{mid}{suffix}_f{FMAP[V]}"


# ---------------------------------------------------------------------------------------------------------------- rows
_cache: dict = {}


def _group(g):
    if g not in _cache:
        if FROM_RUNS:
            _cache[g] = _read_runs(g)
        else:
            z = np.load(ROWS / f"{g}.npz")
            d = {k: z[k] for k in z.files}
            idx = {}
            for n in range(len(d["seed"])):
                idx.setdefault((str(d["worlds"][d["world"][n]]), str(d["specs"][d["spec"][n]])), {})[int(d["seed"][n])] = n
            _cache[g] = (d, idx)
    return _cache[g]


def _read_runs(g):
    """results/runs/<g>/<world>/s<seed>/<spec>.json (results/bench.py) in the layout of the shipped rows."""
    recs = []
    for p in sorted((RUNS / g).glob("*/s*/*.json")):
        r = json.loads(p.read_text())
        if r.get("schema", 1) >= 2:
            recs.append(r)
    if not recs:
        raise SystemExit(f"no rows in {RUNS / g}: run the commands of the README with --tag {g}")
    nmax = max(r["N"] for r in recs)
    d = dict(excess=np.array([r["excess01"] for r in recs]), floor=np.array([r["G_T01"] for r in recs]),
             total=np.array([r["total01"] for r in recs]), n_users=np.array([r["N"] for r in recs]),
             per_user=np.full((len(recs), nmax), np.nan), arrival_ids=np.full((len(recs), nmax), -1), type_ids=np.full((len(recs), nmax), -1),
             prefix_excess=np.array([[np.cumsum(r["excess_per_student01"])[min(k, r["N"]) - 1] for k in (25, 50, 100, 200, 300)] for r in recs]),
             n_clusters=np.array([float(r.get("diag_last", {}).get("n_clusters", np.nan)) for r in recs]))
    idx = {}
    for n, r in enumerate(recs):
        m = r["N"]; d["per_user"][n, :m] = r["excess_per_student01"]
        if "arrival_ids" in r:                                    # tutoring worlds only
            d["arrival_ids"][n, :m] = r["arrival_ids"]; d["type_ids"][n, :m] = r["type_ids"]
        idx.setdefault((r["world"], r["spec"]), {})[int(r["seed"])] = n
    return d, idx


def rows(g, w, spec):
    """{seed: row index} of (group, world, spec)."""
    return _group(g)[1].get((w, spec), {})


def col(g, name, ix):
    return _group(g)[0][name][ix]


def excess(g, w, spec):
    R = rows(g, w, spec)
    return {s: float(col(g, "excess", i)) for s, i in R.items()}


def stats_of(g, w, spec):
    e = np.array(list(excess(g, w, spec).values()))
    if not len(e):
        return None
    return float(e.mean()), (float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else float("nan")), len(e)


def per_arrival(g, w, spec, seeds):
    """(seeds, arrivals) excess regret, columns ordered by arrival id, and the profile id of each column."""
    R = rows(g, w, spec)
    ids0 = col(g, "arrival_ids", R[seeds[0]]); m = int(col(g, "n_users", R[seeds[0]]))
    ids = np.sort(ids0[:m]); pos = {int(a): n for n, a in enumerate(ids)}
    D = np.zeros((len(seeds), len(ids)))
    for k, s in enumerate(seeds):
        i = R[s]
        for a, x in zip(col(g, "arrival_ids", i)[:m], col(g, "per_user", i)[:m]):
            D[k, pos[int(a)]] = x
    types = dict(zip(ids0[:m].tolist(), col(g, "type_ids", R[seeds[0]])[:m].tolist()))
    return D, np.array([types[int(a)] for a in ids])


def paired(gA, specA, gC, specC, w, boot=False, strata=True):
    """gain of A over C (C - A on excess regret): seed t-interval, optionally the crossed bootstrap."""
    A, C = excess(gA, w, specA), excess(gC, w, specC)
    seeds = sorted(set(A) & set(C))
    if len(seeds) < 3:
        return None
    d = np.array([C[s] - A[s] for s in seeds]); ti = t_interval(d)
    cex = float(np.mean([C[s] for s in seeds]))
    out = dict(gain=ti["mean"], lower=ti["lower"], upper=ti["upper"], p=ti["p"], n=len(seeds), comp=cex, share=ti["mean"] / cex,
               lo=ti["lower"] / cex, hi=ti["upper"] / cex)
    if boot:
        Dc, types = per_arrival(gC, w, specC, seeds); Da, _ = per_arrival(gA, w, specA, seeds)
        b = crossed_bootstrap(Dc - Da, strata=types if strata else None, n_boot=2000)
        out.update(blo=b["lower"] / cex, bhi=b["upper"] / cex)
    return out


# ---------------------------------------------------------------------------------------------------------- references
_cells: dict = {}


def cell(w):
    """closed-form references of a tutoring world: random pairs, floor, fixed pairs (refs.closed_form)."""
    if w not in _cells:
        p = parse_world(w)
        env, _, _, b, _ = build(w, 0, with_oracle=False)
        design = build(w.replace(f"_{p['roster']}_", "_dev_"), 0, with_oracle=False)[0] if p["roster"].startswith("test") else None
        _cells[w] = closed_form(env, b, design_env=design)
    return _cells[w]


def ref_value(c, key):
    for k in (key, key.replace("_design", "_crossfit"), key.replace("_design", "_hindsight")):
        if k in c:
            return c[k]
    return None


def known_profile_pair(c, k):
    """the best fixed pair of the user's profile: chosen on the development roster for the development profiles (T, D),
    cross-fitted for the eight new profiles of U; not defined on Composite and Quirky."""
    if k not in ("U", "T", "D"):
        return None
    return c.get("type_told_pair_crossfit") if k == "U" else ref_value(c, "type_told_pair_design")


def oracle_excess(V, k):
    """per-user linear oracle (results/oracle.py), summed over the roster's users."""
    p = ROOT / "results" / "oracle" / f"oracle_{V}_{ROSTERS[k][0]}.npy"
    if not p.exists():
        raise SystemExit(f"missing {p}: python results/oracle.py --version {V} --all --out results/oracle")
    return float(np.load(p).sum())


def score(c, ex):
    rnd = c["random"] - c["floor"]
    return 100 * (rnd - ex) / rnd


def pct(x, sign=True):
    return f"{100 * x:+.1f}%" if sign else f"{100 * x:.1f}%"


def table(head, body, title):
    w = [max(len(str(r[j])) for r in [head] + body) for j in range(len(head))]
    print(f"\n{title}")
    print("  " + "  ".join(str(h).ljust(w[j]) if j == 0 else str(h).rjust(w[j]) for j, h in enumerate(head)))
    for r in body:
        print("  " + "  ".join(str(x).ljust(w[j]) if j == 0 else str(x).rjust(w[j]) for j, x in enumerate(r)))


# ------------------------------------------------------------------------------------------------------------- reports
def ladder(V, keys, group, title, cmp_keys=MAIN_CMP, gains=True, per_user_pair=False):
    """references and learners on the rosters `keys` (excess mean +- s.e. over seeds, score), then the paired comparisons."""
    T = TUNED[V]; M = CODUEL[V]
    ws = [world(V, k) for k in keys]; cs = [cell(w) for w in ws]
    head = ["method"] + [x for k in keys for x in (f"{ROSTERS[k][1]}: excess", "score")]
    body = []
    refs = [("random pairs", lambda c, k: c["random"] - c["floor"]),
            ("best fixed pair (chosen on the development roster)", lambda c, k: ref_value(c, "best_fixed_pair_design") - c["floor"])]
    if per_user_pair:
        refs.append(("best fixed pair per user (hindsight)", lambda c, k: c["per_student_oracle_pair"] - c["floor"]))
    refs += [("per-user linear oracle", lambda c, k: oracle_excess(V, k)), ("distinct-pair floor G_T", lambda c, k: 0.0)]
    for lab, f in refs:
        r = [lab]
        for k, c in zip(keys, cs):
            ex = f(c, k); r += [f"{ex:.1f}", f"{score(c, ex):.0f}"]
        body.append(r)
    specs = [(NAMES[m], T[m]) for m in cmp_keys] + [("CODuel", M), ("ablation: no transfer", NT)]
    for lab, sp in specs:
        r = [lab]
        for w, c in zip(ws, cs):
            s = stats_of(group, w, sp)
            r += ["--", "--"] if s is None else [f"{s[0]:.1f} +- {s[1]:.1f}", f"{score(c, s[0]):.0f}"]
        body.append(r)
    table(head, body, f"{title}: excess regret R_T - G_T (mean +- s.e. over seeds) and score (share of random pairs' excess removed)")
    print("  floor G_T: " + ", ".join(f"{c['floor']:.0f} ({ROSTERS[k][1]})" for k, c in zip(keys, cs)))
    if not gains:
        return
    head = ["CODuel compared with"] + [ROSTERS[k][1] for k in keys]
    body = []
    for lab, sp in [("no transfer (ablation)", NT)] + [(NAMES[m], T[m]) for m in dict.fromkeys(cmp_keys + ("adats",))] + [("type-library reference", "ref:typelib")]:
        r = [lab]
        for k, w in zip(keys, ws):
            x = paired(group, M, group, sp, w, boot=True, strata=ROSTERS[k][3])
            r.append("--" if x is None else f"{pct(x['share'])} [{pct(x['blo'])}, {pct(x['bhi'])}]")
        body.append(r)
    table(head, body, f"{title}, paired: gain of CODuel as a share of the competitor's excess regret [crossed bootstrap 95%]")


def rep_main(a):
    ladder("v3.1L", ["U", "C", "I"], "main", "Table 1 (dataset v3.1L)", per_user_pair=True)


def rep_topic(a):
    ladder("v3.1L", ["T", "D"], "main", "Held-out-topic and development rosters (v3.1L)", cmp_keys=MAIN_CMP + ("adats",))


def rep_references(a):
    V, keys = "v3.1L", ["U", "C", "I", "T", "D"]
    head = ["reference"] + [ROSTERS[k][1] for k in keys]
    cs = {k: cell(world(V, k)) for k in keys}
    body = []
    for lab, f in (("best fixed pair per user", lambda k, c: c["per_student_oracle_pair"] - c["floor"]),
                   ("known-profile pair", lambda k, c: None if known_profile_pair(c, k) is None else known_profile_pair(c, k) - c["floor"]),
                   ("type-library reference", lambda k, c: (stats_of("main", world(V, k), "ref:typelib") or [None])[0]),
                   ("CODuel", lambda k, c: (stats_of("main", world(V, k), CODUEL[V]) or [None])[0])):
        body.append([lab] + ["--" if f(k, cs[k]) is None else f"{score(cs[k], f(k, cs[k])):.0f}" for k in keys])
    table(head, body, "Diagnostic references: score by roster (the last row, CODuel, is for comparison; the per-user linear oracle is in Table 1)")


def rep_profiles(a):
    V, k = "v3.1L", "U"; w = world(V, k); g = "main"
    ex = {m: stats_of(g, w, TUNED[V][m])[0] for m in MAIN_CMP}
    best = min(ex, key=ex.get)
    specs = [CODUEL[V], TUNED[V][best], NT]
    seeds = sorted(set.intersection(*[set(rows(g, w, s)) for s in specs + ["random"]]))
    D = {s: per_arrival(g, w, s, seeds)[0] for s in specs}
    R0 = rows(g, w, specs[0]); m = int(col(g, "n_users", R0[seeds[0]]))
    ids = np.sort(col(g, "arrival_ids", R0[seeds[0]])[:m])
    prof = json.loads((ROOT / "dataset" / V / "types" / ROSTERS[k][0] / "index.json").read_text())["profiles"]
    names = np.array([prof[i] for i in ids])
    Ub = _llm_env(parse_world(w)).U
    srt = np.sort(Ub, -1); fl = (srt[..., -1] - srt[..., -2]).sum(1)
    reg = np.stack([(2 * srt[..., -1] - Ub[..., i] - Ub[..., j]).sum(1) for i, j in itertools.combinations(range(Ub.shape[-1]), 2)], 1)
    rnd_a = (reg.mean(1) - fl)[ids]

    def sc(s, ss, idx):
        rnd = rnd_a[idx].sum()
        return 100 * (rnd - D[s][np.ix_(ss, idx)].sum(1).mean()) / rnd
    allS = np.arange(len(seeds))
    body = []
    for p in np.unique(names):
        idx = np.where(names == p)[0]
        body.append([p, len(idx)] + [f"{sc(s, allS, idx):.0f}" for s in specs])
    table(["profile", "users", "CODuel", NAMES[best], "no transfer"], body, f"Held-out profiles, score by profile ({len(seeds)} seeds)")
    rng = np.random.default_rng(0); P = np.unique(names); out = []
    for _ in range(4000):
        pp = rng.choice(P, len(P)); ss = rng.integers(0, len(seeds), len(seeds))
        out.append(sc(specs[0], ss, np.concatenate([np.where(names == q)[0] for q in pp])))
    lo, hi = np.quantile(out, [0.025, 0.975])
    print(f"  CODuel, profiles and seeds resampled after the fact: 95% range of the score {lo:.0f} to {hi:.0f}")


def rep_phases(a, frac=0.25):
    V = "v3.1L"; keys = ["U", "C", "I"]
    head = ["method"] + [x for k in keys for x in (f"{ROSTERS[k][1]}: first 25%", "last 25%")]
    per = {}
    for k in keys:
        w = world(V, k)
        for sp in (NT, CODUEL[V], "ref:typelib"):
            R = rows("main", w, sp)
            per[(sp, k)] = np.array([col("main", "per_user", i) for i in R.values()]).mean(0)
    body = []
    for lab, sp in (("no transfer", NT), ("CODuel", CODUEL[V]), ("type-library reference", "ref:typelib")):
        r = [lab]
        for k in keys:
            p = per[(sp, k)]; n = max(1, int(round(frac * len(p)))); r += [f"{p[:n].mean():.2f}", f"{p[-n:].mean():.2f}"]
        body.append(r)
    table(head, body, "Excess regret per user in the first and last quarter of the arrivals (mean over seeds)")
    for k in keys:
        x, y = per[(CODUEL[V], k)], per[(NT, k)]; n = max(1, int(round(frac * len(x))))
        print(f"  {ROSTERS[k][1]}: CODuel below no transfer by {pct((y[:n].mean() - x[:n].mean()) / y[:n].mean())} (first quarter), "
              f"{pct((y[-n:].mean() - x[-n:].mean()) / y[-n:].mean())} (last quarter)")


def rep_redrawn(a):
    V = "v3.1L"; T = TUNED[V]
    specs = [("CODuel", CODUEL[V]), ("COLDB", T["coldb"]), ("CoLSTIM", T["colstim"]), ("Meta-TS", T["metats"]), ("HierTS", T["hierts"]), ("no transfer", NT)]
    res = {}
    for k in ("U", "C", "I"):
        mode = "rsp" if k == "U" else "rss"
        w = world(V, k, suffix="_" + mode)
        seeds = sorted(set.intersection(*[set(rows("redrawn", w, s)) for _, s in specs]))
        base = _llm_env(parse_world(world(V, k)))
        Ub = base.U; srt = np.sort(Ub, -1); fl = (srt[..., -1] - srt[..., -2]).sum(1)
        reg = np.stack([(2 * srt[..., -1] - Ub[..., i] - Ub[..., j]).sum(1) for i, j in itertools.combinations(range(Ub.shape[-1]), 2)], 1)
        rnd_a = reg.mean(1) - fl
        sc = {sp: [] for _, sp in specs}; distinct = []
        for sd in seeds:
            idx = resample_indices(base, sd, mode); rnd = rnd_a[idx].sum(); distinct.append(len(set(np.asarray(base.profiles)[idx])))
            for _, sp in specs:
                sc[sp].append(100 * (rnd - excess("redrawn", w, sp)[sd]) / rnd)
        res[k] = (sc, np.mean(distinct), len(seeds))
    body = []
    for lab, sp in specs:
        body.append([lab] + [f"{np.mean(res[k][0][sp]):.0f} [{np.quantile(res[k][0][sp], 0.025):.0f}, {np.quantile(res[k][0][sp], 0.975):.0f}]" for k in res])
    for lab, sp in specs[1:]:
        r = [f"CODuel minus {lab}"]
        for k in res:
            d = np.array(res[k][0][CODUEL[V]]) - np.array(res[k][0][sp])
            r.append(f"{d.mean():+.0f} [{np.quantile(d, 0.025):+.0f}, {np.quantile(d, 0.975):+.0f}]")
        body.append(r)
    table(["policy"] + [ROSTERS[k][1] for k in res], body, "Score over rosters redrawn for every seed (mean [2.5th, 97.5th percentile over seeds])")
    print("  seeds: " + ", ".join(f"{ROSTERS[k][1]} {res[k][2]}" for k in res) + f"; distinct profiles per redrawn held-out-profile roster {res['U'][1]:.1f}")


ABL = [("no recalibration (A = I)", None), ("greedy pair rule instead of lookahead", ":rule=emax"),
       ("latent-bandit Thompson sampling instead of lookahead", ":rule=lts"), ("latent-bandit UCB instead of lookahead", ":rule=lucb:lucb_c=0.5"),
       ("no evidence gate", ":peg=0"), ("fallback concentration alpha = 5", ":alpha=5"), ("moment M-step instead of pooled fit", ":mstep=moment"),
       ("one tight type instead of the mixture", ":births=0:n_comp=1"), ("one Gaussian with learned covariance", ":births=0:n_comp=1:mstep=moment"),
       ("no population (recalibration kept)", ":warm=1000000000:n_init=1000000000:peg=0")]


def rep_ablations(a):
    V = "v3.1L"; M = CODUEL[V]; w = world(V, "D"); g = "ablations_dev"
    m, se, n = stats_of(g, w, M)
    body = [["CODuel (excess regret)", f"{m:.1f} +- {se:.1f}", ""]]
    r = paired(g, M, "main", NT, w)
    body.append(["no transfer (base learner only)", f"{r['gain']:+.1f} [{r['lower']:+.1f}, {r['upper']:+.1f}]", pct(r["gain"] / m)])
    for lab, suf in ABL:
        sp = NORECAL if suf is None else M + suf
        r = paired(g, M, g, sp, w)
        body.append([lab, "--" if r is None else f"{r['gain']:+.1f} [{r['lower']:+.1f}, {r['upper']:+.1f}]", "" if r is None else pct(r["gain"] / m)])
    table(["variant", "increase in excess regret [95% t]", "share of CODuel's"], body, f"One-component changes, development roster ({n} seeds)")


def rep_ablations_eval(a):
    V = "v3.1L"; M = CODUEL[V]; keys = ["U", "C", "I"]
    body = []
    for lab, suf in [ABL[0], ABL[1], ABL[4], ABL[7], ABL[8], ABL[9]]:
        r_ = [lab]
        for k in keys:
            w = world(V, k)
            x = paired("main", M, "main" if suf is None else "ablations_eval", NORECAL if suf is None else M + suf, w)
            base = stats_of("main", w, M)[0]
            r_.append("--" if x is None else f"{x['gain']:+.1f} [{x['lower']:+.1f}, {x['upper']:+.1f}] ({pct(x['gain'] / base)})")
        body.append(r_)
    table(["variant"] + [ROSTERS[k][1] for k in keys], body, "One-component changes on the evaluation rosters: increase in excess regret [95% t] (share of CODuel's)")


SENS = [(":peg=0.25", "gate threshold z = 0.25 (paper 0.5)"), (":peg=1", "gate threshold z = 1"), (":peg_min=4", "gate minimum 4 (paper 8)"),
        (":peg_min=16", "gate minimum 16"), (":n_comp=8", "components 8 (paper 16)"), (":n_comp=32", "components 32"),
        (":bank=8", "question bank 8 (paper 16)"), (":bank=32", "question bank 32"), (":alpha=5", "alpha = 5 (paper 20)"), (":alpha=50", "alpha = 50"),
        (":lama=3", "recalibration ridge 3 (paper 10)"), (":lama=30", "recalibration ridge 30"),
        (":refit=5", "refit every 5 sessions (paper 10)"), (":refit=20", "refit every 20 sessions")]


def rep_sensitivity(a):
    V = "v3.1L"; M = CODUEL[V]; keys = ["U", "C", "I"]
    closest = {k: min(MAIN_CMP, key=lambda m: stats_of("main", world(V, k), TUNED[V][m])[0]) for k in keys}
    body, n_cells, n_below, n_sig, d_all = [], 0, 0, 0, []
    for suf, lab in SENS:
        r_ = [lab]
        for k in keys:
            w = world(V, k); c = cell(w)
            x = paired("main", M, "sensitivity", M + suf, w); y = paired("main", TUNED[V][closest[k]], "sensitivity", M + suf, w)
            k100 = -100 / (c["random"] - c["floor"])
            r_.append(f"{x['gain'] * k100:+.1f} [{x['upper'] * k100:+.1f}, {x['lower'] * k100:+.1f}]")
            n_cells += 1; n_below += y["gain"] < 0; n_sig += y["upper"] < 0; d_all.append(x["gain"] * k100)
        body.append(r_)
    table(["setting"] + [ROSTERS[k][1] for k in keys], body, "Sensitivity: change in score against the paper's settings [95% t]")
    print(f"  score changes from {min(d_all):+.1f} to {max(d_all):+.1f} points; CODuel keeps lower excess regret than the closest comparison "
          f"method in {n_below} of {n_cells} cells (significantly in {n_sig}); closest: " + ", ".join(f"{ROSTERS[k][1]} {NAMES[closest[k]]}" for k in keys))


def rep_controls(a):
    V = "v3.1L"; M = CODUEL[V]; wn = world(V, "D", suffix="_scrl")
    body = []
    x = paired("sim_null", CODUEL_SIM, "sim_null", NT, "ps_null_be4_b20")
    body.append(["simulator, null population", f"{x['comp']:.1f}", pct(x["share"]), f"[{pct(x['lo'])}, {pct(x['hi'])}] (seed t)"])
    x = paired("null_dev", M, "null_dev", NT, wn, boot=True, strata=False)
    body.append(["tutoring, scrambled roster", f"{x['comp']:.1f}", pct(x["share"]), f"[{pct(x['blo'])}, {pct(x['bhi'])}] (crossed bootstrap)"])
    table(["control", "excess of no transfer", "change with CODuel", "95% interval"], body, "Negative controls")
    body = []
    for lab, g, w, m in (("simulator null (beta = 4, b = 20)", "sim_ablations", "ps_null_be4_b20", CODUEL_SIM), ("scrambled roster (b = 24)", "null_dev", wn, M)):
        for plab, sp in (("CODuel", m), ("without gate", m + ":peg=0"), ("no population", m + ":warm=1000000000:n_init=1000000000:peg=0")):
            s = stats_of(g, w, sp); x = paired(g, sp, g, NT, w)
            body.append([lab, plab, f"{s[0]:.1f} +- {s[1]:.1f}", f"{pct(x['gain'] / x['comp'])} [{pct(x['lower'] / x['comp'])}, {pct(x['upper'] / x['comp'])}]"])
    table(["control", "policy", "excess regret", "change vs no transfer [95% t]"], body, "The evidence gate on the negative controls")
    w = world(V, "D")
    body = [[lab, "{:.1f} +- {:.1f}".format(*stats_of("ablations_dev", w, sp)[:2])] for lab, sp in
            (("greedy", "refx:typelib:greedy"), ("one-step lookahead", "refx:typelib:kg"), ("latent-bandit Thompson sampling", "refx:typelib:lts"),
             ("latent-bandit UCB", "refx:typelib:lucb"))]
    table(["pair rule (profiles' tables known)", "excess regret"], body, "Lookahead with a known population (development roster)")


def grid_cells(g, policy, comp, worlds):
    """paired gain at every arrival prefix (25, 50, 100, 200, 300 users) of every world, Benjamini-Hochberg over all of them at
    q = 0.05; returns the full-roster record of each world (share of the competitor's excess, resolved sign)."""
    rec = []
    for w in worlds:
        A, C = rows(g, w, policy), rows(g, w, comp)
        seeds = sorted(set(A) & set(C))
        if len(seeds) < 4:
            continue
        ea = np.array([col(g, "prefix_excess", A[s]) for s in seeds]); ec = np.array([col(g, "prefix_excess", C[s]) for s in seeds])
        N = int(col(g, "n_users", A[seeds[0]]))
        for j, n in enumerate((25, 50, 100, 200, 300)):
            if n > N and j and (25, 50, 100, 200, 300)[j - 1] >= N:
                continue                                          # prefixes beyond the roster collapse onto the full roster
            d = ec[:, j] - ea[:, j]; ti = t_interval(d)
            rec.append(dict(world=w, N=min(n, N), gain=float(d.mean()), share=float(d.mean() / ec[:, j].mean()), p=ti["p"], params=parse_world(w)))
    mask = bh([r["p"] for r in rec])
    for r, m in zip(rec, mask):
        r["sign"] = int(np.sign(r["gain"])) if m else 0
    top = {}
    for r in rec:
        if r["N"] >= top.get(r["world"], {"N": -1})["N"]:
            top[r["world"]] = r
    return list(top.values())


def _grid_print(rec, ykey, title):
    bs = sorted({r["params"]["b"] for r in rec}); ys = sorted({ykey(r["params"]) for r in rec})
    body = []
    for y in ys:
        row = [f"{y:g}"]
        for b in bs:
            r = [x for x in rec if x["params"]["b"] == b and ykey(x["params"]) == y]
            row.append("--" if not r else f"{100 * r[0]['share']:+.0f}{'' if r[0]['sign'] else 'o'}")
        body.append(row)
    table([""] + [f"b={b}" for b in bs], body, title + "  (o: not resolved at Benjamini-Hochberg level 0.05)")


def rep_grid(a):
    V = "v3.1L"; g = "grid_dev"
    worlds = [str(w) for w in _group(g)[0]["worlds"]] if not FROM_RUNS else sorted({w for w, _ in _group(g)[1]})
    rec = grid_cells(g, CODUEL[V], NT, worlds)
    dev = [r for r in rec if not r["params"].get("scramble_clicks")]; scr = [r for r in rec if r["params"].get("scramble_clicks")]
    _grid_print(dev, lambda p: p["tau"], "Development roster: gain of CODuel over no transfer, % of its excess (rows: click sharpness tau)")
    _grid_print(scr, lambda p: p["tau"], "Scrambled roster: the same")
    rest = [r for r in dev if not (r["params"]["tau"] == 0.5 and r["params"]["b"] in (3, 6))]
    print(f"  development: better in {sum(r['sign'] > 0 for r in dev)} of {len(dev)} cells, worse in {sum(r['sign'] < 0 for r in dev)}; shares "
          f"{100 * min(r['share'] for r in dev):+.0f}% to {100 * max(r['share'] for r in dev):+.0f}% (all but tau = 0.5 with b = 3, 6: "
          f"{100 * min(r['share'] for r in rest):.0f}% to {100 * max(r['share'] for r in rest):.0f}%)")
    print(f"  scrambled: worse in {sum(r['sign'] < 0 for r in scr)} of {len(scr)} cells, better in {sum(r['sign'] > 0 for r in scr)}; shares "
          f"{100 * min(r['share'] for r in scr):+.0f}% to {100 * max(r['share'] for r in scr):+.0f}%")
    tl = grid_cells(g, "ref:typelib", NT, [w for w in worlds if not parse_world(w).get("scramble_clicks")])
    _grid_print(tl, lambda p: p["tau"], "Type-library reference vs no transfer (development roster)")
    print(f"  type library: better in {sum(r['sign'] > 0 for r in tl)} of {len(tl)} cells, worse in {sum(r['sign'] < 0 for r in tl)}")


SIM_POPS = [("types_M2_th60", "2 types, 60 deg"), ("types_M2_th90", "2 types, 90 deg"), ("types_M4_th60", "4 types, 60 deg"),
            ("types_M4_th90", "4 types, 90 deg"), ("types_M8_th60", "8 types, 60 deg"), ("types_M8_th90", "8 types, 90 deg"),
            ("gauss_s0.3", "Gaussian cloud"), ("lowrank_r2_s1", "low-rank cloud"), ("null", "null")]


def rep_sim(a):
    T = TUNED["sim"]
    specs = [("no transfer", NT)] + [(NAMES[m], T[m]) for m in ("rucb", "colstim", "coldb", "coldbl", "pop", "metats", "hierts", "adats")] + \
            [("CODuel", CODUEL_SIM), ("population oracle", "oracle:emax")]
    means = {}
    for beta in (4, 12):
        body = []
        for pop, lab in SIM_POPS:
            w = f"ps_{pop}_be{beta}_b20"; r = [lab]
            for _, sp in specs:
                s = stats_of("sim", w, sp)
                if s:
                    means[(w, sp)] = s[0]
                r.append("--" if s is None else f"{s[0]:.0f} +- {s[1]:.0f}")
            body.append(r)
        table(["population"] + [s for s, _ in specs], body, f"Simulator anchors, beta = {beta}, b = 20: excess regret (mean +- s.e. over seeds)")
    cells = [f"ps_{pop}_be{beta}_b20" for beta in (4, 12) for pop, _ in SIM_POPS if pop != "null"]
    beat = {lab: sum(means[(w, sp)] < means[(w, CODUEL_SIM)] for w in cells) for lab, sp in specs if sp not in (CODUEL_SIM, "oracle:emax", NT)}
    print(f"  structured cells: {len(cells)}; cells where a comparison method has lower excess regret than CODuel: {beat}")
    for key, pops in (("two types", ["types_M2_th60", "types_M2_th90"]), ("eight types", ["types_M8_th60", "types_M8_th90"]),
                      ("Gaussian cloud", ["gauss_s0.3"])):
        rem = [1 - means[(f"ps_{p}_be{b}_b20", CODUEL_SIM)] / means[(f"ps_{p}_be{b}_b20", NT)] for p in pops for b in (4, 12)]
        print(f"  {key}: CODuel removes {100 * min(rem):.0f}% to {100 * max(rem):.0f}% of no transfer's excess regret")
    ch = [100 * (means[(f"ps_null_be{b}_b20", sp)] - means[(f"ps_null_be{b}_b20", NT)]) / means[(f"ps_null_be{b}_b20", NT)]
          for b in (4, 12) for lab, sp in specs if sp in (T["pop"], T["rucb"], T["coldbl"], T["metats"], T["hierts"], T["adats"], T["coldb"])]
    print(f"  null population: methods that share information change excess regret by {min(ch):+.0f}% to {max(ch):+.0f}% relative to no transfer")
    # the b x beta grid (16 seeds per cell; the b = 20 anchors are above)
    g = "sim_grid"
    worlds = [str(w) for w in _group(g)[0]["worlds"]] if not FROM_RUNS else sorted({w for w, _ in _group(g)[1]})
    rec = grid_cells(g, CODUEL_SIM, NT, [w for w in worlds if parse_world(w)["b"] != 20])
    struct = [r for r in rec if r["params"]["make"]["population"] != "null"]; null = [r for r in rec if r["params"]["make"]["population"] == "null"]
    print(f"  b x beta grid: better in {sum(r['sign'] > 0 for r in struct)} of {len(struct)} structured cells, worse in {sum(r['sign'] < 0 for r in struct)}; "
          f"null population: worse in {sum(r['sign'] < 0 for r in null)} of {len(null)} (mean share {100 * np.mean([r['share'] for r in null]):+.1f}%)")
    for k, f in (("beta", lambda r: r["params"]["make"]["beta"]), ("b", lambda r: r["params"]["b"])):
        vals = sorted({f(r) for r in struct})
        mv = [np.mean([r["share"] for r in struct if f(r) == v]) for v in vals]
        print(f"  mean share by {k}: " + ", ".join(f"{v:g}: {100 * m:+.0f}%" for v, m in zip(vals, mv)))
    # COLDB with long sessions (two 90-degree types, beta = 4, 16 seeds)
    g = "sim_coldb_long"
    def m_(w, sp):
        R = rows(g, w, sp)
        return np.mean([col(g, "excess", i) for i in R.values()]), np.mean([col(g, "n_clusters", i) for i in R.values()])
    pooled, _ = m_("ps_types_M2_th90_be4_b80", "pooled"); split, ncl = m_("ps_types_M2_th90_be4_b80", "coldb:a0.03:c0.03")
    _, ncl1 = m_("ps_types_M2_th90_be4_b160", "coldb:a1:c1"); _, nlr = m_("ps_types_M2_th90_be4_b40", "coldbl:t0.5:c0.3")
    print(f"  COLDB, 80 choices per user, tightened margin (coldb:a0.03:c0.03): excess {split:.0f} against {pooled:.0f} for the pooled baseline "
          f"({ncl:.1f} groups); original margin (coldb:a1:c1) after 160 choices: {ncl1:.1f} groups; COLDB-LR after 40 choices: {nlr:.1f} groups")


def _ratio(g, w_list, spec, nt_world=None, seeds=range(300, 316)):
    """tuning criterion: mean over worlds of (mean excess of the configuration) / (mean excess of no transfer), tuning seeds."""
    rs = []
    for w in w_list:
        A, N = excess(g, w, spec), excess(g, nt_world or w, NT)
        ss = [s for s in seeds if s in A and s in N]
        if len(ss) < len(seeds):
            return None
        rs.append(np.mean([A[s] for s in ss]) / np.mean([N[s] for s in ss]))
    return float(np.mean(rs))


def rep_tuning(a, envs=("v3.1L", "v3.1J", "sim")):
    G = json.loads((ROOT / "results" / "tuning_grids.json").read_text())
    for env in envs:
        E = G["environments"][env]; g = E["group"]
        body = []
        for m, grid in G["grids"][env].items():
            specs = [CODUEL["v3.1L"] + s if m == "lucb" else s for s in grid]
            r = {sp: _ratio(g, E["worlds"], sp) for sp in specs}; r = {k: v for k, v in r.items() if v is not None}
            if not r:
                continue
            best = min(r, key=r.get)
            body.append([m, len(grid), len(r), best[len(CODUEL['v3.1L']):] if m == "lucb" else best, f"{r[best]:.3f}"])
        table(["method", "grid", "with rows", "chosen", "ratio"], body, f"Tuning, {env}: {E['description']}")


def rep_tuning_eval(a):
    V = "v3.1L"; G = json.loads((ROOT / "results" / "tuning_grids.json").read_text())["grids"][V]
    fams = [("colstim", G["colstim"]), ("coldb", G["coldb"]), ("coldbl", G["coldbl"]), ("rucb", G["rucb"]), ("metats", G["metats"]),
            ("hierts", G["hierts"]), ("adats", G["adats"])]
    body = []; best_any = {}
    for m, grid in fams:
        r_ = [NAMES[m]]
        for k in ("U", "C", "I"):
            w = world(V, k); c = cell(w)
            ex = {sp: stats_of("tuning_eval", w, sp) for sp in grid}
            ex = {sp: s[0] for sp, s in ex.items() if s is not None and s[2] >= ROSTERS[k][2]}
            b = min(ex, key=ex.get)
            r_ += [f"{score(c, ex[TUNED[V][m]]):.0f}", f"{score(c, ex[b]):.0f}", b]
            best_any[k] = max(best_any.get(k, -1e9), score(c, ex[b]))
        body.append(r_)
    body.append(["CODuel"] + [x for k in ("U", "C", "I") for x in (f"{score(cell(world(V, k)), stats_of('main', world(V, k), CODUEL[V])[0]):.0f}", "", "")])
    table(["method"] + [x for k in ("U", "C", "I") for x in (f"{ROSTERS[k][1]}: dev. choice", "best on roster", "configuration")], body,
          "Settings chosen on the evaluation rosters (score)")
    print("  best comparison method on each roster: " + ", ".join(f"{ROSTERS[k][1]} {v:.0f}" for k, v in best_any.items()))
    body = []
    for m in ("metats", "hierts", "adats"):
        dts = json.loads((ROOT / "results" / "tuning_grids.json").read_text())["selected"][V][m + "_dts"]
        r_ = [NAMES[m]]
        for k in ("U", "C", "I"):
            x = paired("tuning_eval", dts, "tuning_eval", TUNED[V][m], world(V, k))      # reported minus double-sample excess
            r_.append(f"{-100 * x['gain'] / x['comp']:+.1f} [{-100 * x['upper'] / x['comp']:+.1f}, {-100 * x['lower'] / x['comp']:+.1f}]")
        body.append(r_)
    table(["method"] + [ROSTERS[k][1] for k in ("U", "C", "I")], body,
          "Double-sample pair rule against the reported rule: change in excess regret, % of the reported version's [95% t]")


def rep_second_judge(a):
    V = "v3.1J"
    for keys, title in ((["U", "C", "I"], "Second judge (v3.1J), evaluation rosters"), (["T", "D"], "Second judge (v3.1J), held-out topics and development")):
        ladder(V, keys, "sj_main", title, cmp_keys=MAIN_CMP + (("adats",) if "T" in keys else ()))


REPORTS = {"main": rep_main, "topic": rep_topic, "references": rep_references, "profiles": rep_profiles, "phases": rep_phases,
           "redrawn": rep_redrawn, "ablations": rep_ablations, "ablations-eval": rep_ablations_eval, "sensitivity": rep_sensitivity,
           "controls": rep_controls, "grid": rep_grid, "sim": rep_sim, "tuning": rep_tuning, "tuning-eval": rep_tuning_eval,
           "second-judge": rep_second_judge}


def main():
    global FROM_RUNS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", choices=list(REPORTS) + ["all"])
    ap.add_argument("--from-runs", action="store_true", help="read results/runs/<group>/ (results/bench.py --tag <group>) instead of results/rows/")
    a = ap.parse_args()
    FROM_RUNS = a.from_runs
    for name in (REPORTS if a.report == "all" else [a.report]):
        print(f"\n===== {name} =====")
        REPORTS[name](a)


if __name__ == "__main__":
    main()
