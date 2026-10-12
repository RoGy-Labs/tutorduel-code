#!/usr/bin/env python
"""The simulated-student judge: for every arrival, question and persona pair of a roster, gpt-6-luna (reasoning off)
role-plays the arrival's student profile and picks the answer that student would rather receive. Each pair gets 4 prompts
(display order x which answer carries label A), each sampled K = 4 times in one call. One output row per prompt:
{id, arrival, j, i, k, pos, lab, n_i, n, tin, tout} with n_i of n valid picks for persona i.

    python generation/judge_luna.py --roster dev [--k 4] [--workers 32] [--out calls_k4.jsonl]    # needs OPENAI_API_KEY
    python generation/build_benchmark.py --compact calls_k4.jsonl dataset/judge_calls_luna/dev/calls.jsonl.gz

The call files shipped in dataset/judge_calls_luna/<roster>/calls.jsonl.gz were produced by this script and compacted by
build_benchmark.py --compact: the same rows without id, tin and tout (input / output tokens), one row per prompt, in request
order; generation/build_benchmark.py turns them into the benchmark tables. Resumable: prompts already in --out are skipped.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKEYS = ["P0_proof", "P1_examples", "P2_socratic", "P3_intuition", "P4_concise", "P5_stepbystep"]
PAIRS = list(combinations(range(6), 2))

JUDGE_PROMPT = (
    "You are taking part in a research simulation of students choosing between tutoring answers. Fully adopt "
    "the student described below. All candidate answers are factually acceptable; judge only which style THIS "
    "student would rather receive.\n\n"
    "Student:\n{profile}\n\n"
    "Question: {question}\n\n"
    "Answer {first}:\n{text_first}\n\n"
    "Answer {second}:\n{text_second}\n\n"
    "Which answer would this student rather receive, A or B? Reply with the single letter A or B and nothing else.")


def requests(roster, n_arrivals=None):
    """Every prompt of the roster: arrivals x questions x 15 pairs x 2 display orders x 2 label assignments."""
    L = ROOT / "dataset" / "v3.1L" / "types"
    profiles = json.load(open(L / roster / "index.json"))["profiles"]
    qids = np.load(L / roster / "qids.npy")
    profile_text = json.load(open(L / "cartoons.json"))
    text_split = "dev" if roster == "dev" else "test"          # the test_* rosters play the test question packs
    ans = {}
    for line in open(ROOT / "dataset" / "texts" / text_split / "answers.jsonl"):
        a = json.loads(line)
        ans[(int(a["qid"]), a["persona"])] = a
    out = []
    for arr in range(n_arrivals or len(profiles)):
        for j in range(qids.shape[1]):
            q = int(qids[arr, j])
            question = ans[(q, PKEYS[0])]["question"].strip()
            for i, k in PAIRS:
                ti, tk = ans[(q, PKEYS[i])]["text"].strip(), ans[(q, PKEYS[k])]["text"].strip()
                for pos in (1, -1):                              # pos = 1: persona i's answer is shown first
                    for lab in (1, -1):                          # lab = 1: the first-shown answer is labelled A
                        first, second = ("A", "B") if lab == 1 else ("B", "A")
                        tf, ts = (ti, tk) if pos == 1 else (tk, ti)
                        out.append(dict(id=f"{arr}|{j}|{i}|{k}|{pos}|{lab}", arrival=arr, j=j, i=i, k=k, pos=pos, lab=lab,
                                        label_i=first if pos == 1 else second,
                                        prompt=JUDGE_PROMPT.format(profile=profile_text[profiles[arr]].strip(), question=question,
                                                                   first=first, second=second, text_first=tf, text_second=ts)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True, choices=["dev", "test", "test_unseen", "test_continuum", "test_idio"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="gpt-6-luna", help="gpt-6-luna: the dataset judge")
    ap.add_argument("--n-arrivals", type=int, default=None, help="judge only the first n arrivals of the roster")
    a = ap.parse_args()
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    cl = OpenAI()
    path = Path(a.out or f"calls_k{a.k}_{a.roster}.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    todo = [r for r in requests(a.roster, a.n_arrivals) if r["id"] not in done]
    print(f"{len(done)} cached, {len(todo)} to call", flush=True)
    lock = threading.Lock()

    def work(req):
        cap = 16
        for attempt in range(8):
            try:
                r = cl.chat.completions.create(model=a.model, messages=[{"role": "user", "content": req["prompt"]}], n=a.k,
                                               max_completion_tokens=cap, reasoning_effort="none")
                break
            except Exception as e:
                code = getattr(e, "status_code", None)
                if code == 400 and "max_tokens" in str(e):       # one of the n replies ran long: the API refuses, not truncates
                    cap *= 4
                    continue
                if code is not None and code < 500 and code != 429:
                    raise
                time.sleep(min(60, 2 ** attempt))
        picks = [(c.message.content or "").strip().lstrip("*\"'( ").upper()[:1] for c in r.choices]
        rec = {x: req[x] for x in ("id", "arrival", "j", "i", "k", "pos", "lab")}
        rec.update(n_i=sum(p == req["label_i"] for p in picks), n=sum(p in ("A", "B") for p in picks),
                   tin=r.usage.prompt_tokens, tout=r.usage.completion_tokens)
        with lock, open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))


if __name__ == "__main__":
    main()
