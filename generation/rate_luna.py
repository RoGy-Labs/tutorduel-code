#!/usr/bin/env python
"""The rater of the feature map lung (used with dataset v3.1J): gpt-6-luna as a user-independent rater. One chat call per question
(reasoning off, temperature 0): the system prompt describes the six tutoring styles (written from the persona system prompts in
generation/data/generation_prompts.json) and asks for a 0-10 rating of how BADLY each would serve the student's message, as a
JSON object constrained by a schema; the reply's top-5 logprobs are read at the first token of each rating value.

    python generation/rate_luna.py --split dev [--workers 128] [--out luna_badly_dev.jsonl]   # needs OPENAI_API_KEY
    gzip -n luna_badly_dev.jsonl && mv luna_badly_dev.jsonl.gz dataset/rater/

One output row per question: {qid, top_logprobs}: qid indexes dataset/texts/<split>/packs.json["questions"]; top_logprobs holds,
for each style in persona order P0_proof ... P5_stepbystep, the [token, logprob] pairs returned at the first token of that
style's rating. generation/build_features.py turns dataset/rater/luna_badly_{dev,test}.jsonl.gz into phi_lung. Resumable:
questions already in --out are skipped.
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-6-luna"
KEYS = ["rigorous", "examples", "socratic", "intuition", "concise", "steps"]       # persona order P0..P5
STYLES = {
    "rigorous": "Rigorous and formal: starts from precise definitions, states the result as a claim and proves it or sketches the "
                "proof, uses standard notation, and points out which hypotheses are needed. No analogies, no chatty language.",
    "examples": "Worked examples: immediately works through one or two specific numerical or concrete examples in full detail, then "
                "states the general pattern briefly at the end. Light notation.",
    "socratic": "Guided questions: does not give the answer directly; leads the student through a short sequence of leading "
                "questions and small hints so they reach the answer themselves, and states the key idea only at the end in one "
                "sentence.",
    "intuition": "Intuition and big picture: explains the idea in plain language with a vivid analogy or mental picture, says why "
                 "the idea matters, and keeps formulas to a minimum.",
    "concise": "Concise summary: a very short bullet-point answer (under about 90 words) giving the key definition, the key formula "
               "or rule, and one line on when it applies. No analogies, no examples beyond one line.",
    "steps": "Step-by-step procedure: a clearly numbered sequence of steps the student can follow, explaining the purpose of each "
             "step in simple words, warning about common mistakes, and ending with a short encouraging remark.",
}
ORDER = ["socratic", "concise", "rigorous", "steps", "examples", "intuition"]       # order of the styles in the prompt and schema
SYSTEM = (
    "You advise a tutoring app on how to answer a student's message. The app can write its answer in one of six tutoring "
    "styles:\n\n{styles}\n\nRead the student's message. For a student you know nothing else about, rate each style from 0 to 10 "
    "for how BADLY an answer written in that style would serve what this message asks for. Take into account everything the "
    "message says about what the student wants, needs or already knows. 10 = the worst possible style for this message: it would "
    "frustrate this student or miss what they ask for; 5 = neither a problem nor a good fit; 0 = no problem at all, this style "
    "serves the message well. Use the whole scale and differentiate between the styles.\n\nReply with a JSON object only, mapping "
    "each style name ({keys}) to an integer from 0 to 10."
).format(styles="\n".join(f"- {k}: {STYLES[k]}" for k in ORDER), keys=", ".join(ORDER))
USER = "Student message:\n<message>\n{q}\n</message>"
SCHEMA = {"type": "json_schema", "json_schema": {"name": "ratings", "strict": True, "schema": {
          "type": "object", "properties": {k: {"type": "integer"} for k in ORDER}, "required": ORDER, "additionalProperties": False}}}


def value_lp(tokens):
    """Top logprobs at the first token of each style's rating value in the JSON reply."""
    out, text = {}, ""
    for t in tokens:
        if t.token.strip().isdigit():
            m = re.search(r'"(\w+)"\s*:\s*$', text)
            if m and m.group(1) in KEYS and m.group(1) not in out:
                out[m.group(1)] = [[x.token, x.logprob] for x in t.top_logprobs]
        text += t.token
    return out


def call(cl, qid, q):
    for attempt in range(8):
        try:
            r = cl.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": USER.format(q=q)}], temperature=0, max_completion_tokens=120,
                                           response_format=SCHEMA, logprobs=True, top_logprobs=5, reasoning_effort="none")
            c = r.choices[0]
            assert set(json.loads(c.message.content)) == set(KEYS)
            lp = value_lp(c.logprobs.content)
            assert set(lp) == set(KEYS)
            return dict(qid=int(qid), top_logprobs=[lp[k] for k in KEYS])
        except Exception as e:
            if "400" in str(e):
                raise
            time.sleep(min(60, 2 ** attempt) * (0.5 + np.random.rand()))
    raise RuntimeError("8 failed attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from openai import OpenAI
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    cl = OpenAI(max_retries=0, timeout=60)
    qs = json.load(open(ROOT / "dataset" / "texts" / a.split / "packs.json"))["questions"]
    path = Path(a.out or f"luna_badly_{a.split}.jsonl")
    for _ in range(3):                                              # failed questions are retried in up to 3 passes
        done = {json.loads(l)["qid"] for l in open(path) if l.strip()} if path.exists() else set()
        todo = [i for i in range(len(qs)) if i not in done]
        print(f"[{a.split}] {len(done)} cached, {len(todo)} to call", flush=True)
        if not todo:
            break
        with ThreadPoolExecutor(a.workers) as ex, open(path, "a") as f:
            futs = {ex.submit(call, cl, i, qs[i]): i for i in todo}
            for fu in as_completed(futs):
                try:
                    f.write(json.dumps(fu.result()) + "\n"); f.flush()
                except Exception as e:
                    print("FAILED", futs[fu], repr(e)[:200], flush=True)


if __name__ == "__main__":
    main()
