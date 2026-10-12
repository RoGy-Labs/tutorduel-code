"""Load the frozen LLM preference tables.

The published files live in `dataset/` at the repo root. Override with
PERSONA_DUELING_DATA.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .replay import ArrivalEnv, arrival_oracle, calibrate_S, filter_arrivals
from .env import TableEnv

# Default dataset version: "v3.1L", the benchmark of the paper (dataset/v3.1L, judge gpt-6-luna); "v3.1J" is the same
# benchmark judged by TypeSafe Jev (the paper's second judge). Override with PERSONA_DATASET_VERSION.
DEFAULT_VERSION = os.environ.get("PERSONA_DATASET_VERSION", "v3.1L")

TYPES_M = {
    2: ("T1_major", "T2_anxious"),
    4: ("T1_major", "T2_anxious", "T3_socratic", "T5_busy"),
    8: (
        "T0_exam", "T1_major", "T2_anxious", "T3_socratic",
        "T4_intuition", "T5_busy", "T6_coach", "T7_cs",
    ),
}


def dataset_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    env = os.environ.get("PERSONA_DUELING_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "dataset"


def types_dir(root: str | Path | None = None) -> Path:
    return dataset_root(root) / "types"


def shared_dir(root: str | Path | None = None) -> Path:
    return dataset_root(root) / "shared"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Set PERSONA_DUELING_DATA to the unpacked dataset folder."
        )
    return path


def meta(root: str | Path | None = None) -> dict:
    return json.loads(_require(dataset_root(root) / "meta.json").read_text())


def manifest(root: str | Path | None = None, version: str = "v2") -> dict:
    return json.loads(_require(dataset_root(root) / version / "MANIFEST.json").read_text())


def load_types(root: str | Path | None = None, M: int | None = None, version: str | None = None, utility: str | None = None,
               roster: str = "dev") -> dict:
    """Private-pack replay tensor. M in {2,4,8} filters the cartoon roster.

    version  "v3.1L" (the benchmark) or "v3.1J" (second judge); the code paths of earlier internal versions are kept.
    utility  "stretch": favourite = 1, worst = 0 on every question (v1 default). "global": U = 1 - (v_max - v) / R with
             one declared R for the whole dataset (v2 default; for v1 it is read from dataset/v2a).
    roster   "dev" (the 300 original arrivals), "test" (v2: 200 new arrivals; v2.1: 350), and for v2.1 also
             "test_unseen" (350 arrivals from 8 profiles never used in design) and "test_mixed" (half known, half new).
    """
    version = version or DEFAULT_VERSION
    if utility is None:
        utility = "stretch" if version == "v1" else "global"
    if utility not in ("stretch", "global"):
        raise ValueError(utility)
    Pcbt = None
    if version == "v1":
        if roster != "dev":
            raise ValueError("v1 has no test roster")
        d = types_dir(root)
        U = np.load(_require(d / "U.npy"))
        a = dataset_root(root) / "v2a" / "types"
        if utility == "global":
            U = np.load(_require(a / "U.npy"))
        if (a / "Pcbt.npy").exists():
            Pcbt = np.load(a / "Pcbt.npy")
    else:
        d = dataset_root(root) / version / "types" / roster
        if not d.exists() and manifest(root, version).get("base"):   # the second-judge versions reuse their base's dev roster
            d = dataset_root(root) / manifest(root, version)["base"] / "types" / roster
        U = np.load(_require(d / ("U.npy" if utility == "global" else "U_stretch.npy")))
        # Pcbt (Bradley-Terry click table) is shipped where the paper uses it (click="cbt" worlds of the development roster);
        # generation/build_benchmark.py --out writes it for every roster
        Pcbt = np.load(d / "Pcbt.npy") if (d / "Pcbt.npy").exists() else None
    P = np.load(_require(d / "P.npy"))
    qids = np.load(_require(d / "qids.npy"))
    if (d / "Phi.npy").exists():
        Phi = np.load(d / "Phi.npy")
    else:
        # the released rosters carry no feature tensor: a world takes its persona-question features from
        # results/features/<version>/<split>/phi_<map>.npy (world name suffix _f<map>, results/worlds.py). Until then the
        # environment holds a zero-width placeholder with one row per question.
        Phi = np.zeros((int(qids.max()) + 1, P.shape[-1], 0))
    index = json.loads(_require(d / "index.json").read_text())
    profiles = list(index["profiles"])
    kept = None
    arrival_ids = np.arange(len(profiles))
    if M is not None and int(M) != 8:
        keep = TYPES_M[int(M)]
        if Pcbt is not None:
            Pcbt = Pcbt[np.array([p in set(keep) for p in profiles])]
        P, U, qids, mask = filter_arrivals(P, U, qids, profiles, keep)
        arrival_ids = arrival_ids[mask]
        profiles = [p for p, m in zip(profiles, mask) if m]
        kept = list(keep)
        index = dict(index)
        index["profiles"] = profiles
        index["M"] = int(M)
        index["cartoons"] = kept
    own = dataset_root(root) / version / "types" / "cartoons.json"                   # v2.1+: profiles live with the dataset version
    cartoons = json.loads((own if own.exists() else _require(types_dir(root) / "cartoons.json")).read_text())
    return {
        "P": P, "U": U, "Phi": Phi, "qids": qids, "index": index,
        "profiles": profiles, "cartoons": cartoons, "kept": kept,
        "root": d, "Pcbt": Pcbt, "arrival_ids": arrival_ids, "version": version, "utility": utility, "roster": roster,
    }


def load_shared(root: str | Path | None = None, pool: str | None = "types") -> dict:
    """Overlap slice: 56 profiles × 24 questions. pool is types/unique/unstructured or None for all."""
    d = shared_dir(root)
    P = np.load(_require(d / "P.npy"))
    U = np.load(_require(d / "U.npy"))
    Phi = np.load(_require(d / "Phi.npy"))
    index = json.loads(_require(d / "index.json").read_text())
    students = json.loads(_require(d / "students.json").read_text())
    questions = json.loads(_require(d / "questions.json").read_text())
    n_miss = int(index.get("n_missing_logprob_tokens") or 0)
    if n_miss > 0:
        raise ValueError(f"shared tensor has n_missing_logprob_tokens={n_miss}")
    pools = np.array(index["pools"])
    if pool is not None:
        mask = pools == pool
        if not mask.any():
            raise ValueError(f"no students in pool {pool!r}")
        P, U = P[mask], U[mask]
        keys = [k for k, m in zip(index["students"], mask) if m]
    else:
        keys = list(index["students"])
    return {
        "P": P, "U": U, "Phi": Phi, "index": index, "students": students,
        "questions": questions, "keys": keys, "root": d,
    }


def types_env(root: str | Path | None = None, M: int | None = 8, S: float | None = None, version: str | None = None,
              utility: str | None = None, roster: str = "dev", tau: float = 1.0, click: str = "raw") -> ArrivalEnv:
    """Replay environment (default: dataset v2, dev roster, dataset-level utilities; version="v1" reproduces v1 exactly).
    tau multiplies the click logits (what-if sharpness); click="cbt" draws clicks from sigma(v_i - v_j)."""
    version = version or DEFAULT_VERSION
    pack = load_types(root, M=M, version=version, utility=utility, roster=roster)
    if S is None:                                       # replaced by the feature map's scale in feature-map worlds
        S = types_scale(root) if version == "v1" else float(manifest(root, version).get("S", 1.0))
    # the learner is told beta = tau, as it is told beta in the simulator (S is calibrated on the raw tau = 1 table)
    return ArrivalEnv(Phi=pack["Phi"], P=pack["P"], U=pack["U"], qids=pack["qids"], S=S, beta=float(tau), tau=tau, click=click,
                      Pcbt=pack["Pcbt"], profiles=pack["profiles"], arrival_ids=pack["arrival_ids"])


def shared_env(root: str | Path | None = None, pool: str = "types", S: float | None = None) -> TableEnv:
    pack = load_shared(root, pool=pool)
    if S is None:
        S = types_scale(root)
    chains = pack["questions"]["chain_ids"]
    env = TableEnv(Phi=pack["Phi"], P=pack["P"], U=pack["U"], chains=chains, S=S, beta=1.0)
    env.profile_weights = np.full(env.P.shape[0], 1.0 / env.P.shape[0])
    env.N = env.Sn
    env.b_priv = env.Q
    return env


def types_scale(root: str | Path | None = None) -> float:
    """S = median ||w|| of linear fits on the full eight-cartoon types tensor."""
    pack = load_types(root, M=8, version="v1")
    env = ArrivalEnv(Phi=pack["Phi"], P=pack["P"], U=pack["U"], qids=pack["qids"], S=1.0, beta=1.0)
    _, atoms = arrival_oracle(env)
    return calibrate_S(atoms)
