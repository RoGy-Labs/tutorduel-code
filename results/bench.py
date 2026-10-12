"""Cached, parallel benchmark harness over (world, seed) tasks with common random numbers.

    python results/bench.py --worlds pl_v3.1L_dev_M8_t1_b24_ug_raw_rl1_fjevgw --seeds 100-103 --policies none:emax:rt,random --tag dev

Per-run rows are cached in results/runs/<tag>/<world>/s<seed>/<policy>.json so a stalled or
partial run resumes without recomputing finished (world, seed, policy) triples. Prints per-world
mean +- se of total01 and the paired difference competitor - policy with its 95% t interval.
"""
from __future__ import annotations

import argparse
import json
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker process
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results"))

from persona_bench import run_sequence  # noqa: E402
from persona_bench.metrics import mean_ci  # noqa: E402
from persona_bench.report import excess_curves, result_row  # noqa: E402
import persona_lab  # noqa: F401,E402
import persona_v3  # noqa: F401,E402  registers v3:* and qo:* (CODuel)
import refs  # noqa: F401,E402  registers ref:typelib
import refs_v3  # noqa: F401,E402  registers refx:typelib:<rule> (diagnostic references with hidden information)
from worlds import build, parse_world, resolve  # noqa: E402

SCHEMA = 2

RUNS = ROOT / "results" / "runs"


def sanitize(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]", "_", spec)


def parse_seeds(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        if "-" in tok:
            a, b = tok.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def run_path(tag, world, seed, spec):
    return RUNS / tag / world / f"s{seed}" / (sanitize(spec) + ".json")


def _task(args):
    tag, world, seed, specs, N, force = args
    missing = [s for s in specs if force or not run_path(tag, world, seed, s).exists()]
    if not missing:
        return world, seed, []
    need_oracle = any(s.startswith("oracle") for s in missing)
    env, rk, n, b, orc = build(world, seed, N=N, with_oracle=need_oracle)
    t0 = time.time()
    if "extra_context" not in rk or rk["extra_context"].get("typelib") is None:
        missing = [s for s in missing if not s.startswith("ref:")]          # reference needs the type library (LLM worlds only)
    res = run_sequence(env, missing, N=n, b=b, seed=seed, oracle=orc, **rk)
    done = []
    for spec in missing:
        r = res[spec] if spec in res else next(v for v in res.values() if v.policy == spec)
        row = result_row(r, extra={"world": world, "seed": seed, "b": b, "spec": spec})
        row["per_student01"] = [float(x) for x in r.per_student * r.to01]
        row["diag_last"] = r.diagnostics[-1] if r.diagnostics else {}
        row["regretful_curve"] = [float(x) for x in r.regretful.mean(axis=1)]
        row.update(excess_curves(r))
        row["schema"] = SCHEMA
        row["world_params"] = parse_world(world) or {"legacy": world}
        if hasattr(env, "arrival_ids"):                                      # replay worlds: which arrival came when, and its type
            order = np.asarray(r.types)
            row["arrival_ids"] = [int(env.arrival_ids[i]) for i in order]
            row["type_ids"] = [int(x) for x in env.type_ids()[order]]
        p = run_path(tag, world, seed, spec)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(row, default=float))
        done.append((spec, row["total01"], row["seconds"]))
    return world, seed, done


def load_rows(tag, worlds, seeds, specs):
    rows = {}
    for w in worlds:
        for s in seeds:
            for spec in specs:
                p = run_path(tag, w, s, spec)
                if p.exists():
                    rows[(w, s, spec)] = json.loads(p.read_text())
    return rows


def cell_info(tag, world):
    """Closed-form reference rows and information measures of a replay world (written once per cell)."""
    p = parse_world(world)
    path = RUNS / tag / world / "_cell.json"
    if p is None or p["kind"] != "llm" or p["scramble"] or p.get("synthetic"):
        return json.loads(path.read_text()) if path.exists() else None
    if not path.exists():
        env, rk, n, b, _ = build(world, 0, with_oracle=False)
        design = build(world.replace(f"_{p['roster']}_", "_dev_"), 0, with_oracle=False)[0] if p["roster"].startswith("test") else None
        info = refs.closed_form(env, b, design_env=design)
        info["world_params"] = p
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=1, default=float))
    return json.loads(path.read_text())


def summarize(tag, worlds, seeds, specs, competitor="none:emax", proposed=None):
    rows = load_rows(tag, worlds, seeds, specs)
    for w in worlds:                                                         # homogeneity guard: one (N, b) per cached cell
        shapes = {(r["N"], r["b"]) for (ww, _, _), r in rows.items() if ww == w}
        if len(shapes) > 1:
            raise SystemExit(f"{tag}/{w}: cached rows mix different (N, b) {sorted(shapes)}; use another --tag for another --N")
    out = {"tag": tag, "competitor": competitor, "proposed": proposed, "seeds": seeds, "worlds": {}}
    for w in worlds:
        wb = {"policies": {}, "paired": {}}
        for spec in specs:
            vals = {s: rows[(w, s, spec)]["total01"] for s in seeds if (w, s, spec) in rows}
            if not vals:
                continue
            m, se, h = mean_ci(np.array(list(vals.values())))
            secs = float(np.mean([rows[(w, s, spec)]["seconds"] for s in vals]))
            ex = [rows[(w, s, spec)]["excess01"] for s in vals if "excess01" in rows[(w, s, spec)]]
            wb["policies"][spec] = {"mean": m, "se": se, "ci95": h, "n": len(vals), "per_seed": {str(s): v for s, v in vals.items()},
                                    "excess_mean": float(np.mean(ex)) if ex else None,
                                    "seconds_mean": secs, "N": rows[(w, next(iter(vals)), spec)]["N"], "b": rows[(w, next(iter(vals)), spec)]["b"]}
        for spec in specs:
            if spec == competitor or spec not in wb["policies"] or competitor not in wb["policies"]:
                continue
            common = [s for s in seeds if (w, s, spec) in rows and (w, s, competitor) in rows]
            if len(common) < 2:
                continue
            d = np.array([rows[(w, s, competitor)]["total01"] - rows[(w, s, spec)]["total01"] for s in common])
            m, se, h = mean_ci(d)
            cex = wb["policies"][competitor].get("excess_mean")
            wb["paired"][spec] = {"mean": m, "se": se, "ci95": h, "lower": m - h, "upper": m + h, "n": len(common),
                                  "gain_share_of_excess": (m / cex if cex else None),
                                  "frac_better": float(np.mean(d > 0)),
                                  "pass": bool(m - h > 0 and wb["policies"][spec]["mean"] < wb["policies"][competitor]["mean"])}
        try:
            wb["cell"] = cell_info(tag, w)
        except Exception as e:                                               # references are a convenience, never a blocker
            wb["cell"] = {"error": str(e)}
        out["worlds"][w] = wb
    if proposed:
        flags = [out["worlds"][w]["paired"].get(proposed, {}).get("pass", False) for w in worlds]
        out["pass_per_world"] = {w: f for w, f in zip(worlds, flags)}
        out["all_pass"] = bool(all(flags)) and len(flags) == len(worlds)
    return out


def print_table(summary, specs, competitor="none:emax"):
    worlds = list(summary["worlds"])
    short = {s: (s if len(s) <= 22 else s[:20] + "..") for s in specs}
    print()
    print(f"{'world':22s} " + " ".join(f"{short[s]:>30s}" for s in specs))
    for w in worlds:
        wb = summary["worlds"][w]
        cells = []
        for s in specs:
            p = wb["policies"].get(s)
            if p is None:
                cells.append(f"{'-':>30s}"); continue
            cell = f"{p['mean']:8.1f}±{p['se']:5.1f}"
            pr = wb["paired"].get(s)
            if pr is not None:
                cell += f" d={pr['mean']:+6.1f}[{pr['lower']:+6.1f}]" + ("*" if pr["pass"] else " ")
                if pr.get("gain_share_of_excess") is not None:
                    cell += f" {100 * pr['gain_share_of_excess']:+4.0f}%"
            cells.append(f"{cell:>30s}")
        print(f"{w:22s} " + " ".join(cells))
    print("(d = competitor - policy, paired over seeds; [lower 95% bound]; * = pass; % = d as a share of the competitor's excess over the floor)")
    secs = {s: np.mean([summary["worlds"][w]["policies"][s]["seconds_mean"] for w in worlds if s in summary["worlds"][w]["policies"]]) for s in specs}
    print("mean seconds/run: " + ", ".join(f"{s}={secs[s]:.0f}" for s in specs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="pl_v3.1L_dev_M8_t1_b24_ug_raw_rl1_fjevgw")
    ap.add_argument("--seeds", default="100-103")
    ap.add_argument("--policies", default="none:emax:rt,random")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--competitor", default="none:emax:rt")
    ap.add_argument("--proposed", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-out", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--allow-test", action="store_true", help="permit worlds on the held-out test rosters")
    a = ap.parse_args(argv)
    worlds = resolve(a.worlds); seeds = parse_seeds(a.seeds); specs = [s for s in a.policies.split(",") if s]
    test_worlds = [w for w in worlds if str((parse_world(w) or {}).get("roster", "")).startswith("test")]
    if test_worlds:                                    # held-out rosters: an explicit flag guards against accidental looks during development
        if not a.allow_test:
            raise SystemExit("these worlds use a held-out test roster: " + ", ".join(test_worlds) + "\npass --allow-test to run them")
        os.environ["PERSONA_ALLOW_TEST"] = "1"
    if a.competitor not in specs:
        specs.insert(0, a.competitor)
    tasks = [(a.tag, w, s, specs, a.N, a.force) for w in worlds for s in seeds]
    t0 = time.time()
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            for w, s, done in pool.imap_unordered(_task, tasks):
                if done and not a.quiet:
                    print(f"  done {w} s{s}: " + ", ".join(f"{sp}={tot:.1f} ({sec:.0f}s)" for sp, tot, sec in done), flush=True)
    else:
        for t in tasks:
            w, s, done = _task(t)
            if done and not a.quiet:
                print(f"  done {w} s{s}: " + ", ".join(f"{sp}={tot:.1f} ({sec:.0f}s)" for sp, tot, sec in done), flush=True)
    summ = summarize(a.tag, worlds, seeds, specs, competitor=a.competitor, proposed=a.proposed)
    summ["wall_seconds"] = time.time() - t0
    out = Path(a.summary_out) if a.summary_out else RUNS / a.tag / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=1, default=float))
    print_table(summ, specs, a.competitor)
    if a.proposed:
        print(f"all_pass({a.proposed}) = {summ.get('all_pass')}")
    print(f"wall {time.time()-t0:.0f}s; summary -> {out}")
    return summ


if __name__ == "__main__":
    main()
