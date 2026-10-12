#!/usr/bin/env python
"""The second judge (dataset v3.1J): TypeSafe Jev (jev-1.13.0, System One API) judges the same prompts as generation/judge_luna.py,
every persona pair of every question of every arrival under 4 prompts (display order x which answer carries label A). Jev does
not write text: the luna judge prompt without its final reply instruction is sent as the `state`, and one `choice` question asks
which answer the student would rather receive. One call per prompt; the row keeps the judge's probabilities of both labels.

    python generation/judge_jev.py --roster dev [--workers 48] [--rpm 1150] [--out calls_jev_dev.jsonl]   # needs TYPESAFE_API_KEY
    python generation/build_benchmark.py --compact calls_jev_dev.jsonl dataset/judge_calls_jev/dev/calls.jsonl.gz

One output row per prompt: {arrival, j, i, k, pos, lab, p_i, p_k}: arrival index in the roster, question j of the arrival (its
qid is qids.npy[arrival, j]), persona pair i < k, pos = 1 if persona i's answer is shown first, lab = 1 if the first-shown answer
is labelled A, and p_i, p_k = Jev's probabilities of the labels carried by the answers of personas i and k. The files shipped in
dataset/judge_calls_jev/ hold these rows (one per prompt, request order, gzipped); generation/build_benchmark.py --judge jev turns
them into the v3.1J tables. Resumable: prompts already in --out are skipped. Standard library only.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_luna import requests  # noqa: E402  (same prompts, rosters and answer texts as the luna judge)

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"                                     # pinned version
STATE_END = "\n\nWhich answer would this student rather receive, A or B?"
QUESTION = dict(type="choice", instructions="Which answer would this student rather receive, A or B?",
                criteria={"A": "The student would rather receive Answer A.", "B": "The student would rather receive Answer B."})
KEY = ("arrival", "j", "i", "k", "pos", "lab")


def state_of(prompt):
    """The luna prompt without its reply instruction (Jev returns a typed choice, not a letter)."""
    head, sep, _ = prompt.rpartition(STATE_END)
    assert sep, "judge prompt changed"
    return head


class Pace:
    """At most `rpm` request starts per minute, shared by all workers; a 429 pauses everyone."""
    def __init__(self, rpm):
        self.gap, self.next, self.lock = 60.0 / rpm, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            t = max(time.monotonic(), self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - time.monotonic()))

    def pause(self, s):
        with self.lock:
            self.next = max(self.next, time.monotonic() + s)


def call(key, state, pace):
    body = json.dumps(dict(model=MODEL, state=state, questions={"pick": QUESTION})).encode()
    for attempt in range(12):
        pace.wait()
        req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                pace.pause(float(e.headers.get("Retry-After") or min(60, 2 ** attempt)))
            elif e.code < 500:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
            else:
                time.sleep(min(60, 2 ** attempt))
        except Exception:
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError("12 failed attempts")


def run(roster, path, workers, rpm, n_arrivals):
    key = os.environ.get("TYPESAFE_API_KEY") or sys.exit("TYPESAFE_API_KEY is not set")
    done = {tuple(json.loads(l)[k] for k in KEY) for l in open(path) if l.strip()} if path.exists() else set()
    todo = [r for r in requests(roster, n_arrivals) if tuple(r[k] for k in KEY) not in done]
    print(f"[{roster}] {len(done)} cached, {len(todo)} to call", flush=True)
    lock, pace, n_bad = threading.Lock(), Pace(rpm), [0]

    def work(req):
        try:
            pr = call(key, state_of(req["prompt"]), pace)["answers"]["pick"]["probabilities"]
        except Exception as e:                           # left for the next pass
            with lock:
                n_bad[0] += 1
                if n_bad[0] <= 20:
                    print(f"[{roster}] {req['id']}: {e}", flush=True)
            return
        rec = {k: req[k] for k in KEY}
        rec.update(p_i=pr[req["label_i"]], p_k=pr["B" if req["label_i"] == "A" else "A"])
        with lock, open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    return n_bad[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True, choices=["dev", "test", "test_unseen", "test_continuum", "test_idio"])
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--rpm", type=float, default=1150, help="request starts per minute")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-arrivals", type=int, default=None, help="judge only the first n arrivals of the roster")
    a = ap.parse_args()
    path = Path(a.out or f"calls_jev_{a.roster}.jsonl")
    for _ in range(3):                                   # failed calls are retried in up to 3 passes
        if run(a.roster, path, a.workers, a.rpm, a.n_arrivals) == 0:
            break
    else:
        sys.exit(f"[{a.roster}] calls still failing after 3 passes")


if __name__ == "__main__":
    main()
