"""World names of the benchmark and the simulator. World draw and CRN share the seed."""
from __future__ import annotations

import os

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persona_bench import make_world, types_env  # noqa: E402
from persona_bench.env import SyntheticEnv  # noqa: E402
from persona_bench.replay import arrival_oracle  # noqa: E402

SIM = {
    "w01_types_M2_90_b4": dict(population="types", M=2, theta_deg=90, beta=4),
    "w02_types_M4_60_b4": dict(population="types", M=4, theta_deg=60, beta=4),
    "w03_types_M4_60_b12": dict(population="types", M=4, theta_deg=60, beta=12),
    "w04_types_M8_90_b12": dict(population="types", M=8, theta_deg=90, beta=12),
    "w05_gauss_s03_b4": dict(population="gaussian", s=0.3, beta=4),
    "w06_lowrank_r2_s1_b4": dict(population="lowrank", rank=2, s=1.0, beta=4),
}
# name -> (base world, run_sequence kwargs, SyntheticEnv kwargs)
EXTRA = {
    "w07_w3_chain": ("w03_types_M4_60_b12", dict(query_kind="chain", query_kw={"rho": 0.9}), {}),
    "w08_w3_lapse": ("w03_types_M4_60_b12", {}, dict(link="lapse", lapse=0.15)),
    "w09_w6_chain": ("w06_lowrank_r2_s1_b4", dict(query_kind="chain", query_kw={"rho": 0.9}), {}),
    "w10_w6_lapse": ("w06_lowrank_r2_s1_b4", {}, dict(link="lapse", lapse=0.15)),
}
LLM = {"w11_llm_M8": 8, "w12_llm_M4": 4, "w13_llm_M2": 2}

CORE = list(SIM) + list(EXTRA) + list(LLM)
GROUPS = {
    "core": CORE,
    "sim": list(SIM) + list(EXTRA),
    "llm": list(LLM),
    "fast": ["w13_llm_M2", "w12_llm_M4", "w01_types_M2_90_b4", "w02_types_M4_60_b4"],
    "types": ["w01_types_M2_90_b4", "w02_types_M4_60_b4", "w03_types_M4_60_b12", "w04_types_M8_90_b12"],
    "cloud": ["w05_gauss_s03_b4", "w06_lowrank_r2_s1_b4"],
    "robust": list(EXTRA),
    # benchmark v2 headline cells (dev roster): new judge, dataset-level utilities, raw clicks, personas relabelled per seed
    "v2dev": [f"pl_v2_dev_M{M}_t1_b12_ug_raw_rl1" for M in (8, 4, 2)],
    "v21dev": [f"pl_v2.1_dev_M{M}_t1_b12_ug_raw_rl1" for M in (8, 4, 2)],
    "v3dev": [f"pl_v3_dev_M{M}_t1_b12_ug_raw_rl1" for M in (8, 4, 2)],
    "v31dev": [f"pl_v3.1_dev_M{M}_t1_b24_ug_raw_rl1" for M in (8, 4, 2)],      # v3.1 = v3 with 24 questions per arrival
    "v32dev": [f"pl_v3.2_dev_M{M}_t1_b48_ug_raw_rl1" for M in (8, 4, 2)],      # v3.2 = v3.1 + 24 questions on 4 new STEM topics
}


def resolve(spec: str) -> list[str]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in GROUPS:
            out += GROUPS[tok]
        elif tok in CORE:
            out.append(tok)
        else:
            m = [w for w in CORE if w.startswith(tok) or w.split("_")[0] == tok]
            if not m:
                raise KeyError(tok)
            out += m
    seen, res = set(), []
    for w in out:
        if w not in seen:
            seen.add(w); res.append(w)
    return res


_env_cache: dict = {}
_orc_cache: dict = {}
_lib_cache: dict = {}


# ----------------------------------------------------------------------------- parametric worlds (benchmark v2)
# The run cache is keyed by (tag, world, seed, policy), so every knob that changes the numbers is part of the name.
#   LLM replay:  pl_<version>_<roster>_M<M>_t<tau>_b<b>_u<g|s>_<raw|cbt>_rl<0|1>[_scr|_syn][_f<map>]
#                version v1|v2|v3, roster dev|test|test_unseen|test_mixed|test_continuum|test_idio (the last two: the 300
#                dev packs with a different template profile per arrival, M8 only), tau click sharpness, b rounds (> 12 repeats the pack), ug global /
#                us stretched utilities, click table, rl1 = personas relabelled per seed, _scr = scrambled null roster,
#                _scrl = click-relabelling null roster (every arrival's clicks and utilities relabelled by its own persona
#                permutation, features untouched: the paper's scrambled development roster),
#                _syn = semi-synthetic bridge (real features and roster, but every student IS their type's fitted
#                linear model on [style features, persona identity]: well-specified, no within-type variation),
#                _f<map> = the feature tensor (and S) replaced by results/features/<version>/<split>/phi_<map>.npy (the paper:
#                _fjevgw on v3.1L, _flung on v3.1J); clicks, utilities, packs, arrivals and the per-seed relabelling are those
#                of the base world (base_world(name)), so rows of the two pair by seed.
#   simulator:   ps_types_M<M>_th<deg>_be<beta>_b<b> | ps_gauss_s<s>_be.. | ps_lowrank_r<r>_s<s>_be.. | ps_null_be<beta>_b<b>
def parse_world(name: str):
    import re
    m = re.fullmatch(r"pl_(v[0-9.]+[A-Za-z]*)_(dev|test|test_unseen|test_mixed|test_continuum|test_idio)_M(2|4|8)_t([0-9.]+)_b(\d+)_u(g|s)_(raw|cbt)_rl(0|1)(_scrl|_scr|_syn)?(_rsp|_rss)?(?:_f([A-Za-z0-9._-]+))?", name)
    if m:
        return dict(kind="llm", version=m[1], roster=m[2], M=int(m[3]), tau=float(m[4]), b=int(m[5]),
                    utility="global" if m[6] == "g" else "stretch", click=m[7], relabel=bool(int(m[8])), scramble=(m[9] == "_scr"), synthetic=(m[9] == "_syn"),
                    scramble_clicks=(m[9] == "_scrl"),            # _scrl: per-arrival click relabelling null (features untouched)
                    resample=(m[10] or "")[1:] or None,          # rsp / rss: the roster is redrawn per seed (resample_roster)
                    fmap=m[11], feat=m[11])                      # the feature map (both names are used)
    m = re.fullmatch(r"ps_(types|gauss|lowrank|null)(?:_M(\d+))?(?:_th(\d+))?(?:_r(\d+))?(?:_s([0-9.]+))?_be([0-9.]+)_b(\d+)", name)
    if m:
        kw = dict(population={"types": "types", "gauss": "gaussian", "lowrank": "lowrank", "null": "null"}[m[1]], beta=float(m[6]))
        if m[2]: kw["M"] = int(m[2])
        if m[3]: kw["theta_deg"] = float(m[3])
        if m[4]: kw["rank"] = int(m[4])
        if m[5]: kw["s"] = float(m[5])
        return dict(kind="sim", b=int(m[7]), make=kw)
    return None


def llm_name(version="v2", roster="dev", M=8, tau=1, b=12, utility="g", click="raw", relabel=1, scramble=False, synthetic=False, fmap=None):
    return (f"pl_{version}_{roster}_M{M}_t{tau:g}_b{b}_u{utility}_{click}_rl{int(relabel)}" + ("_scr" if scramble else "_syn" if synthetic else "")
            + (f"_f{fmap}" if fmap else ""))


def base_world(name: str) -> str:
    """The world name without its `_f<map>` suffix (unchanged when there is none): the world whose clicks, utilities and
    arrivals a feature-map world shares, hence where its comparator rows and cell references live."""
    p = parse_world(name)
    if p and p.get("fmap"):
        return name[: -len("_f" + p["fmap"])]
    return name


SIM_POPS = {"t90M2": "types_M2_th90", "t90M4": "types_M4_th90", "t90M8": "types_M8_th90", "t60M2": "types_M2_th60", "t60M4": "types_M4_th60",
            "t60M8": "types_M8_th60", "gauss": "gauss_s0.3", "lowrank": "lowrank_r2_s1", "null": "null"}
GRID_B, GRID_BETA = (3, 6, 12, 24, 48), (1, 2, 4, 8)


def grid(tok: str) -> list[str]:
    """grid:sim[:<pop>]          simulator phase grid (b x beta), one population or all, plus the b=20 anchors
       grid:llm:<v1|v2>[:<M>]    replay dev grid: tau x b (cbt clicks whenever tau != 1), personas relabelled per seed
       grid:llmnull:<v1|v2>      scrambled-roster null panel
       grid:bridge:<v1|v2>       the bridging rows (one change at a time)
       grid:test:v2              the held-out test roster, headline cells only"""
    parts = tok.split(":")
    if parts[1] == "sim":
        pops = [parts[2]] if len(parts) > 2 else list(SIM_POPS)
        out = [f"ps_{SIM_POPS[p]}_be{be}_b{b}" for p in pops for b in GRID_B for be in GRID_BETA]
        return out + [f"ps_{SIM_POPS[p]}_be{be}_b20" for p in pops for be in (4, 12)]
    if parts[1] in ("llm", "llmnull"):
        ver = parts[2]; Ms = [int(parts[3])] if len(parts) > 3 else [8, 4, 2]
        scr = parts[1] == "llmnull"
        taus = (1, 2, 4, 8) if ver == "v1" else (0.5, 1, 2, 4) if ver != "v2" else (0.25, 0.5, 1, 2)
        return [llm_name(ver, "dev", M, tau, b, "g", "raw" if tau == 1 else "cbt", 1, scr) for M in Ms for tau in taus
                for b in ((3, 6, 9, 12) if scr else (3, 6, 9, 12, 24, 48))]
    if parts[1] == "bridge":
        ver = parts[2]
        rows = [("s", "raw", 0), ("s", "raw", 1), ("g", "raw", 1), ("g", "cbt", 1)]
        return [llm_name(ver, "dev", M, 1, 12, u, c, r) for M in (8, 4, 2) for u, c, r in rows]
    if parts[1] == "llmsyn":       # semi-synthetic bridge, tau sweep at b = 12
        return [llm_name(parts[2], "dev", M, tau, 12, "g", "raw", 1, False, True) for M in (8, 4, 2) for tau in ((1, 2, 4, 8) if parts[2] == "v1" else (0.25, 0.5, 1, 2))]
    if parts[1] == "test":
        return [llm_name(parts[2], "test", M, 1, 12, "g", "raw", 1) for M in (8, 4, 2)]
    if parts[1] in ("unseen", "mixed"):      # grid:unseen:v2.1 -> the 8-type worlds on the unseen / mixed rosters
        return [llm_name(parts[2], "test_" + parts[1], 8, 1, 12, "g", "raw", 1)]
    raise KeyError(tok)


def _resolve_one(tok):
    if tok.startswith("grid:"):
        return grid(tok)
    if tok in GROUPS:
        return list(GROUPS[tok])
    if tok in CORE or parse_world(tok):
        return [tok]
    m = [w for w in CORE if w.startswith(tok) or w.split("_")[0] == tok]
    if not m:
        raise KeyError(tok)
    return m


def resolve(spec: str) -> list[str]:        # noqa: F811  (supersedes the legacy resolver above; legacy names still work)
    out = []
    for tok in spec.split(","):
        if tok.strip():
            out += _resolve_one(tok.strip())
    return list(dict.fromkeys(out))


def _llm_env(p):
    key = (p["version"], p["roster"], p["M"], p["tau"], p["utility"], p["click"])
    if key not in _env_cache:
        if p["roster"].startswith("test") and os.environ.get("PERSONA_ALLOW_TEST") != "1":
            raise PermissionError("test roster: run results/bench.py with --allow-test ")
        _env_cache[key] = types_env(M=p["M"], version=p["version"], roster=p["roster"], utility=p["utility"], tau=p["tau"], click=p["click"])
    return _env_cache[key]


def _feat_env(p):
    """The base env of p with its feature tensor replaced by the map p['fmap'] (cached per base env and map)."""
    key = (p["version"], p["roster"], p["M"], p["tau"], p["utility"], p["click"], "f", p["fmap"])
    if key not in _env_cache:
        _env_cache[key] = _with_features(_llm_env(p), p["fmap"], p["roster"], p["version"])
    return _env_cache[key]


def _with_features(env, name, roster, version="v3.1L"):
    """Same world with the feature tensor replaced by results/features/<version>/<split>/phi_<name>.npy and S by the scale
    calibrated on dev for that map (S_<name>.json). The test_* rosters share the test questions, hence the test tensor.
    Clicks, utilities, packs and arrivals are unchanged."""
    import json
    from persona_bench.replay import ArrivalEnv
    F = ROOT / "results" / "features" / version
    split = "test" if roster.startswith("test_") else roster
    Phi, meta = np.load(F / split / f"phi_{name}.npy"), json.load(open(F / "dev" / f"S_{name}.json"))
    assert Phi.shape[0] > int(env.qids.max()) and Phi.shape[1] == env.K, (Phi.shape, env.Phi.shape)
    return ArrivalEnv(Phi=Phi, P=env.P, U=env.U, qids=env.qids, S=float(meta["S"]), beta=env.beta, rating_noise=env.rating_noise,
                      tau=env.tau, click=env.click, Pcbt=env.Pcbt, profiles=env.profiles, arrival_ids=env.arrival_ids)


def resample_indices(env, seed, mode):
    """Arrivals of the roster redrawn for one seed (a bootstrap replicate of the roster; the policies are rerun on it).
    rsp: the roster's profiles are drawn with replacement (as many draws as profiles) and every arrival of each drawn
    profile is kept (a profile drawn twice appears twice); rss: the arrivals themselves are drawn with replacement."""
    rng = np.random.default_rng(880_000 + seed)
    if mode == "rss":
        return rng.integers(0, env.N, env.N)
    prof = np.asarray(env.profiles)
    names = np.unique(prof)
    return np.concatenate([np.flatnonzero(prof == q) for q in rng.choice(names, len(names))])


def _resample(env, seed, mode):
    from persona_bench.replay import ArrivalEnv
    idx = resample_indices(env, seed, mode)
    q = env.qids[idx]                                   # a drawn-twice arrival needs its own question ids: copy the rows
    Phi = env.Phi[q.reshape(-1)]; qids = np.arange(q.size).reshape(q.shape)
    return ArrivalEnv(Phi=Phi, P=env.P[idx], U=env.U[idx], qids=qids, S=env.S, beta=env.beta, rating_noise=env.rating_noise,
                      tau=env.tau, click=env.click, Pcbt=None if env.Pcbt is None else env.Pcbt[idx],
                      profiles=None if env.profiles is None else [env.profiles[i] for i in idx], arrival_ids=np.arange(len(idx)))


def _scramble(env, rng, base_d=None, rng_feat=None):
    """Null roster: every arrival gets its own persona relabelling and feature rotation, which destroys everything
    that is shared between students while keeping each student's own problem unchanged.
    base_d / rng_feat (feature-map worlds only): the per-arrival permutations are drawn exactly as for the base world
    (whose rotation of dimension base_d is drawn from `rng` and discarded), so clicks and utilities stay those of the base
    null world; the rotation of the new tensor comes from its own stream `rng_feat`."""
    from persona_bench.world import haar_rotation
    from persona_bench.replay import ArrivalEnv
    Phi, P, U = env.Phi.copy(), env.P.copy(), env.U.copy()
    Pc = None if env.Pcbt is None else env.Pcbt.copy()
    for s_ in range(env.N):
        perm = rng.permutation(env.K)
        if base_d is None or base_d == env.d:
            R = haar_rotation(rng, env.d)
        else:
            haar_rotation(rng, base_d); R = haar_rotation(rng_feat, env.d)
        for t in range(env.b_priv):
            q = int(env.qids[s_, t]); Phi[q] = (env.Phi[q] @ R.T)[perm]
        P[s_] = env.P[s_][:, perm][:, :, perm]; U[s_] = env.U[s_][:, perm]
        if Pc is not None:
            Pc[s_] = env.Pcbt[s_][:, perm][:, :, perm]
    return ArrivalEnv(Phi=Phi, P=P, U=U, qids=env.qids, S=env.S, beta=env.beta, tau=env.tau, click=env.click, Pcbt=Pc,
                      profiles=env.profiles, arrival_ids=env.arrival_ids)


def _scramble_clicks(env, rng):
    """Null roster for feature maps whose rows carry persona identity (one-hot persona part, e.g. CODuel's rating maps), where
    _scramble's per-arrival rotation would destroy the one-hot: every arrival's clicks and utilities are relabelled by its own
    random persona permutation and the features are left unchanged. Nothing is shared between students (neither persona
    preferences nor how a feature maps to utility), while each student's own problem stays learnable from their own clicks."""
    from persona_bench.replay import ArrivalEnv
    P, U = env.P.copy(), env.U.copy()
    Pc = None if env.Pcbt is None else env.Pcbt.copy()
    for s_ in range(env.N):
        perm = rng.permutation(env.K)
        P[s_] = env.P[s_][:, perm][:, :, perm]; U[s_] = env.U[s_][:, perm]
        if Pc is not None:
            Pc[s_] = env.Pcbt[s_][:, perm][:, :, perm]
    return ArrivalEnv(Phi=env.Phi, P=P, U=U, qids=env.qids, S=env.S, beta=env.beta, tau=env.tau, click=env.click, Pcbt=Pc,
                      profiles=env.profiles, arrival_ids=env.arrival_ids)


def _semi_synthetic(env):
    """Same roster, questions and features; clicks and utilities come from the type-level linear model fitted on this
    (dev) roster. Separates 'too little information' from 'wrong model' when compared with the real world."""
    import numpy as np
    from scipy.special import expit
    from persona_bench.replay import ArrivalEnv, type_oracle
    from persona_bench.utility import top_anchored
    orc = type_oracle(env)
    aug, K = orc["aug"], env.K
    P, U = np.empty_like(env.P), np.empty_like(env.U)
    for u in range(env.N):
        w = orc["atoms"][orc["names"].index(env.profiles[u])]
        for t in range(env.b_priv):
            ph = np.concatenate([env.Phi[int(env.qids[u, t])], aug * np.eye(K)], axis=1) / np.sqrt(1 + aug * aug)
            v = ph @ w
            P[u, t] = expit(v[:, None] - v[None, :]); U[u, t] = top_anchored(v)
    return ArrivalEnv(Phi=env.Phi, P=P, U=U, qids=env.qids, S=env.S, beta=env.beta, tau=env.tau, click="raw", Pcbt=P,
                      profiles=env.profiles, arrival_ids=env.arrival_ids)


def type_library(env):
    """Hidden-information reference input: per profile type, the mean click table and mean utilities over its arrivals
    and rounds (question-independent). Built from the DESIGN roster only."""
    import numpy as np
    names = sorted(set(env.profiles))
    idx = [[i for i, q in enumerate(env.profiles) if q == nm] for nm in names]
    return {"names": names, "P": np.array([env._Pclick[ix].mean((0, 1)) for ix in idx]), "U": np.array([env.U[ix].mean((0, 1)) for ix in idx]),
            "weights": np.array([len(ix) for ix in idx], dtype=float) / env.N}


def _diag_rk(rk, env, seed, N):
    """With PERSONA_DIAG_TYPES=1 (results/bench.py --diag-types) the arrivals' true types, in order, are handed to diagnostic
    reference policies (refx4 / refx6) through extra_context; admissible policies never read it."""
    rk = dict(rk)
    if os.environ.get("PERSONA_DIAG_TYPES") == "1":
        n_run = env.N if N is None else min(N, env.N)
        s_students = np.random.SeedSequence([seed, 1]).spawn(4)[0]
        students = env.sample_students(np.random.default_rng(s_students), n_run)
        rk["extra_context"] = dict(rk.get("extra_context", {}), diag_types=[int(x) for x in np.asarray(env.type_ids())[np.asarray(students)]])
    return rk


def build(name: str, seed: int, N: int | None = None, with_oracle: bool = True):
    """Return (env, run_kwargs, N, b, oracle_dict_or_None). run_kwargs may carry extra_context for reference policies."""
    import numpy as np
    if name in SIM or name in EXTRA:
        if name in SIM:
            kw, rk, ek = SIM[name], {}, {}
        else:
            base, rk, ek = EXTRA[name]
            kw = SIM[base]
        w = make_world(seed, d=8, K=6, S=1.0, lam=1.0, **kw)
        env = SyntheticEnv(w, **ek)
        return env, dict(rk), (300 if N is None else N), 20, (w.population.true_prior() if with_oracle else None)
    if name in LLM:
        M = LLM[name]
        if ("legacy", M) not in _env_cache:
            _env_cache[("legacy", M)] = types_env(M=M, version="v1")
        env = _env_cache[("legacy", M)]
        if with_oracle and M not in _orc_cache:          # env and oracle are cached separately (an oracle-less first call used to poison the cache)
            _orc_cache[M] = arrival_oracle(env)[0]
        rk = _diag_rk(dict(query_kind="chain"), env, seed, N)
        return env, rk, (env.N if N is None else min(N, env.N)), 12, (_orc_cache[M] if with_oracle else None)
    p = parse_world(name)
    if p is None:
        raise KeyError(name)
    if p["kind"] == "sim":
        mk = dict(p["make"])
        null = mk["population"] == "null"
        if null:
            mk.update(population="gaussian", s=0.3)
        w = make_world(seed, d=8, K=6, S=1.0, lam=1.0, **mk)
        if null:                                           # population = exactly the no-transfer prior: nothing to transfer
            from persona_bench.world import Population
            w.population = Population("gaussian", w.d, w.S, mu=np.zeros(w.d), Sigma=(w.S ** 2 / w.d) * np.eye(w.d))
        return SyntheticEnv(w), {}, (300 if N is None else N), p["b"], (w.population.true_prior() if with_oracle else None)
    from persona_bench.replay import relabel_personas
    if not p["fmap"] and p["version"].startswith("v3.1"):
        raise ValueError(f"{name}: the tutoring worlds take their features from results/features/<version>/<split>/phi_<map>.npy; "
                         "name the map with the suffix _f<map>, e.g. _fjevgw (v3.1L) or _flung (v3.1J)")
    base = _llm_env(p)
    env = _feat_env(p) if p["fmap"] else base           # feature map: swap the tensor and S BEFORE any relabelling / scramble
    fkey = p["fmap"] if p["synthetic"] else None        # the type library reads P and U only; the semi-synthetic fit reads Phi
    lib_key = (p["version"], p["M"], p["tau"], p["utility"], p["click"], p["synthetic"], fkey)
    if p["synthetic"]:
        skey = ("syn",) + lib_key + (p["roster"],)
        if skey not in _env_cache:
            _env_cache[skey] = _semi_synthetic(env)
        env = _env_cache[skey]
    if lib_key not in _lib_cache:
        pd = dict(p, roster="dev")
        dev = _feat_env(pd) if fkey else _llm_env(pd)
        _lib_cache[lib_key] = type_library(_semi_synthetic(dev) if p["synthetic"] else dev)
    lib = _lib_cache[lib_key]
    if p.get("resample"):                                # redraw the roster for this seed, before relabelling
        env = _resample(env, seed, p["resample"])
    if p.get("scramble_clicks"):
        env = _scramble_clicks(env, np.random.default_rng(778_000 + seed)); lib = None
    elif p["scramble"]:
        env = _scramble(env, np.random.default_rng(777_000 + seed), base_d=base.d if p["fmap"] else None,
                        rng_feat=np.random.default_rng(779_000 + seed) if p["fmap"] else None); lib = None
    elif p["relabel"]:
        perm = np.random.default_rng(10_000 + seed).permutation(env.K)
        env = relabel_personas(env, perm)
        lib = dict(lib, P=lib["P"][:, perm][:, :, perm], U=lib["U"][:, perm])
    rk = dict(query_kind="chain")
    if lib is not None:
        rk["extra_context"] = {"typelib": lib}
    rk = _diag_rk(rk, env, seed, N)
    return env, rk, (env.N if N is None else min(N, env.N)), p["b"], None
