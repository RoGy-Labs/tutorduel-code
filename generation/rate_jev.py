#!/usr/bin/env python
"""The rater of the feature map jevgw (used with dataset v3.1L): TypeSafe Jev (jev-1.13.0, System One API) as a user-independent
rater. One call per question: the state holds only the student's message ({"student_message": ...}; no profile, no answer), and
one six-way `choice` question asks which answer style would serve the message worst, the options being short descriptions of
the six tutor personas written from their system prompts (generation/data/generation_prompts.json).

    python generation/rate_jev.py --split dev [--workers 16] [--rate 15] [--out jev_worst_dev.jsonl]   # needs TYPESAFE_API_KEY
    gzip -n jev_worst_dev.jsonl && mv jev_worst_dev.jsonl.gz dataset/rater/

One output row per question: {qid, p_worst}: qid indexes dataset/texts/<split>/packs.json["questions"], p_worst = Jev's
probabilities that each style is the worst, in persona order P0_proof ... P5_stepbystep. generation/build_features.py turns
dataset/rater/jev_worst_{dev,test}.jsonl.gz into phi_jevgw. Resumable: questions already in --out are skipped.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
KEYS = ["proof", "examples", "socratic", "intuition", "concise", "steps"]           # persona order P0..P5
STYLE_DESC = {
    "proof": "A rigorous answer: it starts from precise definitions, states the result as a claim and proves it or sketches "
             "the proof, with standard notation and the hypotheses that are needed; no analogies.",
    "examples": "A worked-example answer: it immediately works through one or two specific numerical or concrete examples in "
                "full detail, then states the general pattern briefly at the end, with light notation.",
    "socratic": "A Socratic answer: it does not give the full answer, but guides the student with a short sequence of leading "
                "questions and small hints so they reach the answer themselves, giving the key idea only at the end.",
    "intuition": "An intuitive answer: it explains the idea in plain language with a vivid analogy or mental picture, says why "
                 "the idea matters, and keeps formulas to a minimum.",
    "concise": "A minimal answer: a short bullet-point summary with the key definition, the key formula or rule, and one line "
               "on when it applies, in under 90 words.",
    "steps": "A step-by-step answer: a clearly numbered procedure the student can follow, the purpose of each step in simple "
             "words, warnings about common mistakes, and a short encouraging remark at the end.",
}
WORST_INSTR = "Which style of answer would serve what the student asks for in `student_message` worst?"
QUESTIONS = {"worst": {"type": "choice", "instructions": WORST_INSTR, "criteria": {k: STYLE_DESC[k] for k in KEYS}}}


class Limiter:
    def __init__(self, per_s):
        self.gap, self.t, self.lock = 1.0 / per_s, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.t = max(self.t + self.gap, now)
            d = self.t - now
        if d > 0:
            time.sleep(d)


def call(cl, lim, message):
    import httpx
    err = None
    for attempt in range(10):
        lim.wait()
        try:
            r = cl.post(URL, json={"model": MODEL, "state": {"student_message": message}, "questions": QUESTIONS})
            if r.status_code in (429, 500, 502, 503, 504, 520, 529):
                err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(min(30, 2 ** attempt) * (0.5 + random.random())); continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            pr = r.json()["answers"]["worst"]["probabilities"]
            assert set(pr) == set(KEYS), pr
            return [pr[k] for k in KEYS]
        except (httpx.TransportError, httpx.TimeoutException, ValueError, KeyError, AssertionError, TypeError) as e:
            err = e
            time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
    raise err


def run(split, path, workers, per_s):
    import httpx
    key = os.environ.get("TYPESAFE_API_KEY") or sys.exit("TYPESAFE_API_KEY is not set")
    msgs = json.load(open(ROOT / "dataset" / "texts" / split / "packs.json"))["questions"]
    done = {json.loads(l)["qid"] for l in open(path) if l.strip()} if path.exists() else set()
    todo = [q for q in range(len(msgs)) if q not in done]
    print(f"[{split}] {len(done)} cached, {len(todo)} to call", flush=True)
    lim, fails = Limiter(per_s), 0
    cl = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60)
    with ThreadPoolExecutor(workers) as ex, open(path, "a") as f:
        futs = {ex.submit(call, cl, lim, msgs[q]): q for q in todo}
        for fu in as_completed(futs):
            try:
                f.write(json.dumps(dict(qid=futs[fu], p_worst=fu.result())) + "\n"); f.flush()
            except Exception as e:                                  # left for the next pass
                fails += 1
                print("FAILED", futs[fu], repr(e)[:200], flush=True)
    cl.close()
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rate", type=float, default=15.0, help="calls per second")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = Path(a.out or f"jev_worst_{a.split}.jsonl")
    for _ in range(3):
        if run(a.split, path, a.workers, a.rate) == 0:
            break


if __name__ == "__main__":
    main()
