"""Every run of the paper, by experiment group: the worlds, seeds and policies behind each table (results/bench.py does the runs).

    python results/experiments.py --list                  # groups, number of runs, the tables they feed
    python results/experiments.py main --print            # the results/bench.py commands of a group
    python results/experiments.py main [--jobs 32]       # run them (rows in results/runs/main/; cached, so a rerun resumes)
    python results/paper_stats.py main --from-runs        # the tables from your rows

Each group writes to results/runs/<group>/ and corresponds to one file of the shipped rows, results/rows/<group>.npz. The tuning
groups (tuning, sj_tuning, sim_tuning) are run by results/tune.py. Policy specs are explained in the README.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "results"))

S2O = ("rule=kg:mstep=pooled:comp_var=0:n_comp=16:birth_margin=0:births=3:alpha=20:peg=0.5:peg_min=8:w_iso=0:n_init=1:em_late=1"
       ":cap=0:cap_v0=1000")
CJ = f"qo:{S2O}:xcols=1:recal=1:rfull=1:xpass=0:lama=10:aug=0:sa=1.6996"      # CODuel, dataset v3.1L
CJJ = f"qo:{S2O}:xcols=1:recal=1:rfull=1:xpass=0:lama=10:aug=0:sa=2.5019"     # CODuel, dataset v3.1J (second judge)
S2 = f"v3:{S2O}:aug=0"                       # CODuel on the simulator; on the tutoring benchmark: CODuel without recalibration
NOPOP = ":warm=1000000000:n_init=1000000000:peg=0"
L = lambda r, mid="M8_t1_b24_ug_raw_rl1", suf="": f"pl_v3.1L_{r}_{mid}{suf}_fjevgw"
J = lambda r: f"pl_v3.1J_{r}_M8_t1_b24_ug_raw_rl1_flung"
CMP_L = ["none:emax:rt", "rucb:a0.51", "lab:pop:aug=0", "colstim:c1:l0.1:g0.3", "coldb:a0.03:c0.03", "coldbl:t2:c0.3",
         "lab:metats:s0=0.001:sq=0.1:aug=0", "lab:hierts:s0=0.001:sq=1:aug=0", "lab:adats:s0=0.0003:sq=10:aug=0", "ref:typelib"]
CMP_J = ["none:emax:rt", "rucb:a0.51", "lab:pop:aug=0", "colstim:c3:l0.03:g0.3", "coldb:a0.03:c0.03", "coldbl:t0.5:c0.3",
         "lab:metats:s0=0.0003:sq=1:aug=0", "lab:hierts:s0=0.0003:sq=1:aug=0", "lab:adats:s0=0.003:sq=1:aug=0", "ref:typelib"]
CMP_SIM = ["none:emax:rt", "rucb:a1", "lab:pop:aug=0", "colstim:c0.3:l3:g0.01", "coldb:a0.1:c0", "coldbl:t0.5:c0.3",
           "lab:metats:s0=0.03:sq=1:aug=0", "lab:hierts:s0=0.01:sq=0.1:aug=0", "lab:adats:s0=0.03:sq=1:aug=0", "oracle:emax"]
ABL_DEV = [":rule=emax", ":rule=lts", ":rule=lucb:lucb_c=0.5", ":peg=0", ":alpha=5", ":mstep=moment", ":births=0:n_comp=1",
           ":births=0:n_comp=1:mstep=moment", NOPOP]
ABL_EVAL = [":rule=emax", ":peg=0", ":births=0:n_comp=1", ":births=0:n_comp=1:mstep=moment", NOPOP]
SENS = [":peg=0.25", ":peg=1", ":peg_min=4", ":peg_min=16", ":n_comp=8", ":n_comp=32", ":bank=8", ":bank=32", ":alpha=5", ":alpha=50",
        ":lama=3", ":lama=30", ":refit=5", ":refit=20"]
TAUS, BS = (0.5, 1, 2, 4), (3, 6, 12, 24, 48)
GRID = [L("dev", f"M8_t{t:g}_b{b}_ug_{'raw' if t == 1 else 'cbt'}_rl1") for t in TAUS for b in BS]
GRID_SCR = [L("dev", f"M8_t{t:g}_b{b}_ug_{'raw' if t == 1 else 'cbt'}_rl1", "_scrl") for t in TAUS for b in BS]
POPS = ["types_M2_th60", "types_M2_th90", "types_M4_th60", "types_M4_th90", "types_M8_th60", "types_M8_th90", "gauss_s0.3", "lowrank_r2_s1", "null"]
ANCHORS = [f"ps_{p}_be{be}_b20" for p in POPS for be in (4, 12)]
SIM_GRID = [f"ps_{p}_be{be}_b{b}" for p in POPS for be in (1, 2, 4, 8) for b in BS]


def _tuning_eval():
    import json
    T = json.loads((ROOT / "results" / "tuning_grids.json").read_text())
    G, sel = T["grids"]["v3.1L"], T["selected"]["v3.1L"]
    return [s for m in ("colstim", "coldb", "coldbl", "rucb", "rucb_beyond_edge", "metats", "hierts", "adats") for s in G[m]] + \
           [sel[m] for m in ("metats_dts", "hierts_dts", "adats_dts")]       # the double-sample pair rule: the selected configurations


# group -> (description, [(worlds, seeds, policies, held-out rosters)])
GROUPS = {
    "main": ("Table 1, paired comparisons, held-out topics and development roster, references, profiles, phases (v3.1L)", [
        ([L("test_unseen"), L("test")], "0-63", CMP_L + [CJ], True),
        ([L("test_unseen")], "0-63", ["random", S2], True),
        ([L("test_continuum"), L("test_idio")], "0-31", CMP_L + [CJ, S2], True),
        ([L("dev")], "100-131", CMP_L + [CJ], False)]),
    "ablations_dev": ("one-component changes on the development roster, pair rules with the profiles known", [
        ([L("dev")], "100-131", [CJ] + [CJ + s for s in ABL_DEV] + [S2] + [f"refx:typelib:{r}" for r in ("greedy", "kg", "lts", "lucb")], False)]),
    "ablations_eval": ("one-component changes on the evaluation rosters (paired with main)", [
        ([L("test_unseen")], "0-63", [CJ + s for s in ABL_EVAL], True),
        ([L("test_continuum"), L("test_idio")], "0-31", [CJ + s for s in ABL_EVAL], True)]),
    "sensitivity": ("settings of CODuel changed one at a time (paired with main)", [
        ([L("test_unseen")], "0-63", [CJ + s for s in SENS], True),
        ([L("test_continuum"), L("test_idio")], "0-31", [CJ + s for s in SENS], True)]),
    "null_dev": ("scrambled development roster and the gate on it", [
        ([L("dev", suf="_scrl")], "100-131", ["none:emax:rt", CJ, CJ + ":peg=0", CJ + NOPOP], False)]),
    "grid_dev": ("click sharpness x session length on the development and scrambled rosters", [
        (GRID + GRID_SCR, "100-131", ["none:emax:rt", CJ], False),
        (GRID, "100-131", ["ref:typelib"], False)]),
    "redrawn": ("rosters redrawn for every seed", [
        ([L("test_unseen", suf="_rsp"), L("test_continuum", suf="_rss"), L("test_idio", suf="_rss")], "0-63",
         ["none:emax:rt", CJ, "coldb:a0.03:c0.03", "colstim:c1:l0.1:g0.3", "lab:metats:s0=0.001:sq=0.1:aug=0", "lab:hierts:s0=0.001:sq=1:aug=0"], True)]),
    "tuning_eval": ("every configuration of the tuning grids on the evaluation rosters; the double-sample pair rule", [
        ([L("test_unseen")], "0-63", None, True), ([L("test_continuum"), L("test_idio")], "0-31", None, True)]),
    "sj_main": ("second judge (dataset v3.1J): every roster", [
        ([J("test_unseen"), J("test")], "0-63", CMP_J + [CJJ], True),
        ([J("test_continuum"), J("test_idio")], "0-31", CMP_J + [CJJ], True),
        ([J("dev")], "100-131", CMP_J + [CJJ], False)]),
    "sim": ("simulator anchors (b = 20)", [(ANCHORS, "100-131", CMP_SIM + [S2], False)]),
    "sim_grid": ("simulator b x beta grid", [(SIM_GRID, "100-115", ["none:emax:rt", S2], False)]),
    "sim_null": ("simulator null population (negative control)", [(["ps_null_be4_b20"], "100-131", ["none:emax:rt", S2], False)]),
    "sim_ablations": ("the gate on the simulator's null population", [
        (["ps_null_be4_b20"], "100-131", ["none:emax:rt", S2, S2 + ":peg=0", S2 + NOPOP], False)]),
    "sim_coldb_long": ("COLDB with long sessions", [
        (["ps_types_M2_th90_be4_b80"], "100-115", ["pooled", "coldb:a0.03:c0.03"], False),
        (["ps_types_M2_th90_be4_b160"], "100-115", ["coldb:a1:c1"], False),
        (["ps_types_M2_th90_be4_b40"], "100-115", ["coldbl:t0.5:c0.3"], False)]),
}


def calls(group):
    out = []
    for worlds, seeds, pols, test in GROUPS[group][1]:
        pols = pols if pols is not None else ["none:emax:rt"] + _tuning_eval()
        out.append(["--tag", group, "--worlds", ",".join(worlds), "--seeds", seeds, "--policies", ",".join(pols),
                    "--competitor", "none:emax:rt", "--quiet"] + (["--allow-test"] if test else []))
    return out


def n_runs(group):
    n = 0
    for worlds, seeds, pols, _ in GROUPS[group][1]:
        a, b = seeds.split("-")
        n += len(worlds) * (int(b) - int(a) + 1) * len(pols if pols is not None else ["none:emax:rt"] + _tuning_eval())
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("group", nargs="?", choices=list(GROUPS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--print", action="store_true", help="print the results/bench.py commands instead of running them")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.list or not a.group:
        for g, (desc, _) in GROUPS.items():
            print(f"{g:16s} {n_runs(g):7d} runs  {desc}")
        return
    if a.print:
        for c in calls(a.group):
            print("python results/bench.py " + " ".join(shlex.quote(x) for x in c))
        return
    import bench
    for c in calls(a.group):
        bench.main(c + ["--jobs", str(a.jobs)])


if __name__ == "__main__":
    main()
