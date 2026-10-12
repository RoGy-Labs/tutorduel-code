"""File-based leaderboard over comparable (world, seed set) runs."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .metrics import summarize_runs


def _world_id(row: dict) -> str:
    if "world" in row:
        return str(row["world"])
    if "config" in row:
        return str(row["config"])
    bank = row.get("bank", "sim")
    pool = row.get("pool", "")
    M = row.get("M", "")
    return f"{bank}/{pool}/M={M}"


def build_leaderboard(rows: list[dict], metric: str = "total01", competitor: str = "none:emax") -> dict:
    """Group by world; rank policies by mean `metric` (lower is better)."""
    by_world: dict[str, list] = defaultdict(list)
    for r in rows:
        by_world[_world_id(r)].append(r)
    worlds = {}
    for world, wr in sorted(by_world.items()):
        summ = summarize_runs(wr, metric)
        ranking = sorted(summ["policies"].items(), key=lambda kv: kv[1]["mean"])
        vs = {}
        for key, val in summ.get("paired", {}).items():
            if competitor in key:
                vs[key] = val
        worlds[world] = {
            "metric": metric,
            "ranking": [{"policy": p, **stats} for p, stats in ranking],
            "paired_vs_competitor": vs,
        }
    return {"competitor": competitor, "worlds": worlds}


def write_leaderboard(rows: list[dict], out: str | Path, metric: str = "total01",
                      competitor: str = "none:emax") -> dict:
    table = build_leaderboard(rows, metric=metric, competitor=competitor)
    path = Path(out)
    path.write_text(json.dumps(table, indent=2, default=str))
    md = to_markdown(table)
    path.with_suffix(".md").write_text(md)
    return table


def to_markdown(table: dict) -> str:
    lines = ["# Leaderboard", ""]
    for world, block in table["worlds"].items():
        lines += [f"## {world}", "", f"metric: `{block['metric']}` (lower better)", "",
                  "| rank | policy | mean | se | n |",
                  "| --- | --- | --- | --- | --- |"]
        for i, row in enumerate(block["ranking"], 1):
            se = row.get("se")
            se_s = "nan" if se is None or se != se else f"{se:.4f}"
            lines.append(f"| {i} | {row['policy']} | {row['mean']:.4f} | {se_s} | {row['n']} |")
        lines.append("")
    return "\n".join(lines) + "\n"
