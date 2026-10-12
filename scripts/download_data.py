"""Download the TutorDuel dataset from Hugging Face (https://huggingface.co/datasets/romainpuech/tutorduel) into the places the
code reads it from: dataset/ (rosters of both judges, texts, judge and rater outputs, human labels) and results/features/ (the
persona-question features).

    pip install huggingface_hub
    python scripts/download_data.py                  # about 170 MB
    python scripts/download_data.py --revision <commit>   # a fixed version of the dataset
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "romainpuech/tutorduel"
# dataset repository folder -> local folder
TARGETS = {"v3.1L": "dataset/v3.1L", "v3.1J": "dataset/v3.1J", "texts": "dataset/texts", "judge_calls_luna": "dataset/judge_calls_luna",
           "judge_calls_jev": "dataset/judge_calls_jev", "rater": "dataset/rater", "human_labels": "dataset/human_labels",
           "features": "results/features"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--revision", default=None, help="branch, tag or commit of the dataset repository (default: main)")
    ap.add_argument("--force", action="store_true", help="replace local folders that already exist")
    a = ap.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    src = Path(snapshot_download(REPO, repo_type="dataset", revision=a.revision, allow_patterns=[f"{k}/**" for k in TARGETS]))
    for k, v in TARGETS.items():
        dst = ROOT / v
        if dst.exists():
            if not a.force:
                print(f"{v} exists, kept (--force replaces it)")
                continue
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src / k, dst)
        print(f"{v}: {sum(1 for p in dst.rglob('*') if p.is_file())} files")


if __name__ == "__main__":
    main()
