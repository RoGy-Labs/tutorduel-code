"""Tuning of the comparison methods (Appendix "Tuning of the comparison methods"): every configuration of
results/tuning_grids.json is run on the 16 tuning seeds (300-315), which no reported result uses, then one configuration per
method is selected.

    python results/tune.py --env v3.1L                 # development roster of v3.1L (rows in results/runs/tuning/)
    python results/tune.py --env v3.1J                 # development roster of the second judge (results/runs/sj_tuning/)
    python results/tune.py --env sim                   # the 16 structured b = 20 simulator worlds (results/runs/sim_tuning/)
    python results/tune.py --env v3.1L --select-only   # selection from rows already in results/runs/<tag>/

Criterion (tuning_grids.json "rule"): for every configuration, the mean over the environment's worlds of (mean excess regret of
the configuration over the tuning seeds) / (mean excess regret of the no-transfer learner none:emax:rt over the same seeds); the
configuration with the smallest ratio is chosen. The grids in tuning_grids.json already contain the configurations added by the
edge rule. `python results/paper_stats.py tuning` gives the same selection from the shipped rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "results"))
import bench  # noqa: E402
import paper_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", required=True, choices=["v3.1L", "v3.1J", "sim"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()
    G = json.loads((ROOT / "results" / "tuning_grids.json").read_text())
    E = G["environments"][a.env]
    if not a.select_only:
        specs = [paper_stats.CODUEL["v3.1L"] + s if m == "lucb" else s for m, grid in G["grids"][a.env].items() for s in grid]
        bench.main(["--tag", E["group"], "--worlds", ",".join(E["worlds"]), "--seeds", G["seeds"], "--policies", ",".join(specs),
                    "--competitor", "none:emax:rt", "--jobs", str(a.jobs), "--quiet"])
    paper_stats.FROM_RUNS = True
    paper_stats.rep_tuning(None, envs=(a.env,))


if __name__ == "__main__":
    main()
