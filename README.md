# TutorDuel and CODuel

Code for **Online LLM Personalization from Pairwise Preferences Across Users** (Romain Puech\*, Georgy Salakhutdinov\*, Dimitris
Bertsimas; MIT; \*equal contribution).

[Website](https://rogy-labs.github.io/tutorduel/) | [Dataset on Hugging Face](https://huggingface.co/datasets/romainpuech/tutorduel)

The repository contains the benchmark (TutorDuel), the numerical simulator, the proposed method (CODuel), every comparison
method, the tuning grids and selection rule, the per-user linear oracle, the result rows behind every table of the paper, the
scripts that compute the paper's numbers from them, and the commands and scripts that rerun every experiment and rebuild the
benchmark from the raw judge calls. The dataset (both judges' rosters, the texts, the raw judge and rater outputs, the human
labels and the persona-question features) is on Hugging Face; `scripts/download_data.py` puts it where the code reads it.

Everything runs on CPU with Python >= 3.10, NumPy and SciPy. No GPU and no API key are needed to run policies or to reproduce the
tables; an API key is only needed to re-query the LLMs (`generation/`).

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py                  # the dataset (about 170 MB) -> dataset/ and results/features/
python results/paper_stats.py all                # every results table of the paper, from the shipped rows (under a minute)
python results/paper_stats.py main               # Table 1 and the paired comparisons only
python results/agreement.py all                  # agreement with the authors' choices and between the two judges
python -m pytest -q persona_bench/tests          # unit tests (11 are skipped: they need earlier data versions, not shipped)
```

## Layout

| path | content |
|---|---|
| `persona_bench/` | The benchmark: policy interface (`protocol.py`), runner with common random numbers (`runner.py`), replay environment for the tutoring data (`replay.py`, `dataset.py`), linear-utility simulator (`world.py`, `env.py`), Bradley-Terry utilities (`utility.py`), metrics, comparison methods (`baselines/`: RUCB, CoLSTIM, COLDB and COLDB-LR, pooled baseline, random pairs, the no-transfer learner), unit tests (`tests/`). |
| `persona_v3/` | **CODuel**: `mix.py` (population model: EM mixture prior with births, per-user Laplace posterior, lookahead pair rule), `policy.py` (evidence gate around the population learner and the no-transfer learner; spec `v3:`), `recal.py` (recalibration of the rater's score from completed sessions; spec `qo:`, CODuel on the tutoring benchmark). |
| `persona_lab/` | Shared Laplace and mixture inference (`infer.py`, `hb.py`, `kde.py`), the pooled baseline (`pop.py`), and Meta-TS, HierTS and AdaTS adapted to pairwise choices (`metats.py`). |
| `results/bench.py` | The harness: runs (world, seed, policy) triples in parallel, caches one JSON row per run under `results/runs/<tag>/`, prints mean regret and paired differences. |
| `results/worlds.py` | World names (below). `results/refs.py`: closed-form references (random pairs, floor, fixed pairs) and the type-library reference; `results/refs_v3.py`: pair rules given the development profiles' tables (diagnostic); `results/stats.py`: t-intervals, crossed bootstrap, Benjamini-Hochberg. |
| `results/paper_stats.py` | The paper's numbers from the run rows: one report per table (list below). `results/agreement.py`: agreement of the judges with the authors' choices and with each other. |
| `results/experiments.py` | Every run of the paper, by experiment group: worlds, seeds and policies; prints or runs the `results/bench.py` commands. |
| `results/tune.py`, `results/tuning_grids.json` | Tuning of the comparison methods: the grids (after the edge rule), the tuning worlds and seeds, the selection rule, the selected configurations. |
| `results/oracle.py`, `results/oracle/` | The per-user linear oracle and its per-user excess regret on every roster of both datasets. |
| `results/rows/` | The result rows behind the paper: one `.npz` per experiment group, one row per (world, seed, policy) (format below). |
| `scripts/download_data.py` | Downloads the dataset from Hugging Face into `dataset/` and `results/features/`. |
| `results/features/` | (downloaded) The persona-question features $\phi(p,q)=[e_p, g(p,q)]$ of the questions of each dataset and their scale `S`. |
| `dataset/` | (downloaded by `scripts/download_data.py`) The data, in the folders below. |
| `dataset/v3.1L/` | The benchmark: every roster's choice probabilities and utilities, judge `gpt-6-luna`. |
| `dataset/v3.1J/` | The same rosters judged by TypeSafe Jev (the second judge of the paper). |
| `dataset/texts/` | Questions and tutor answers of the development and test question sets, and the question packs. |
| `dataset/judge_calls_luna/`, `dataset/judge_calls_jev/` | Raw judge outputs of every roster (one row per prompt), from which `generation/build_benchmark.py` rebuilds the tables exactly. |
| `dataset/rater/` | Raw rater outputs for every question (Jev for v3.1L, `gpt-6-luna` for v3.1J), from which `generation/build_features.py` rebuilds the features exactly. |
| `dataset/human_labels/` | The 204 comparisons labeled by both authors, and the picks of two larger chat judges on them (`results/agreement.py`). |
| `generation/` | The LLM clients (judges: `judge_luna.py`, `judge_jev.py`; raters: `rate_jev.py`, `rate_luna.py`), the table and feature builders, the tutor persona prompts and answer instruction (`data/generation_prompts.json`) and the question cue bank (`data/cue_bank.json`). |

## Data

### Rosters

Both datasets have the same five rosters, students, question packs and tutor answers; only the judge differs.

| paper | roster | arrivals | seeds in the paper |
|---|---|---|---|
| held-out profiles | `test_unseen` | 350 | 0-63 |
| Composite | `test_continuum` | 350 | 0-31 |
| Quirky | `test_idio` | 350 | 0-31 |
| held-out topics | `test` | 350 | 0-63 |
| development | `dev` | 300 | 100-131 (tuning of the comparison methods: 300-315) |

`dataset/<version>/types/<roster>/` holds `P.npy` (judge choice probabilities, arrival x question x 6 x 6, `P[a, j, p, p']` = probability
that persona p's answer is chosen over p'), `U.npy` (evaluation utilities $U = 1 - (v_{\max} - v)/\kappa$ from ridge Bradley-Terry scores
$v$, $\kappa = 8$, arrival x question x 6), `qids.npy` (question id of each arrival's questions) and `index.json` (profile of each
arrival). `Pcbt.npy` ($\sigma(v_p - v_{p'})$, the Bradley-Terry choice table of the click-sharpness grid) is shipped for the
development roster of v3.1L, where the paper uses it. Every student's profile text is in `types/cartoons.json`, keyed by the profile id of `index.json` (the eight development
profiles `T*`, the eight held-out profiles `A*`, the Composite students `C*` and the Quirky students `I*`). `MANIFEST.json` documents the judge and the choice-probability law. The test rosters use the test question
set: `qids` index `dataset/texts/test/` for every `test*` roster and `dataset/texts/dev/` for `dev`.

Personas are indexed 0-5 in the order proof (rigorous prover), examples (worked-examples tutor), socratic (Socratic guide),
intuition (intuition-and-analogy tutor), concise (concise cheat-sheet), steps (patient step-by-step coach).

### Texts

`dataset/texts/<split>/answers.jsonl`: one row per (question, persona) with `qid`, `persona`, `question` (with its cue), `text` (the
tutor answer) and `model` (`gpt-5.6-terra`); `packs.json`: the questions (`questions[qid]`), each arrival's pack of 24 questions,
and the generation metadata. The answers were requested at temperature 0.7, which the API ignores for this reasoning model; they
are sampled at its default temperature 1, as the paper states. All texts are synthetic.

### Features

`results/features/<version>/<split>/phi_<map>.npy` (question x 6 x 7): $\phi(p,q)=[e_p, g(p,q)]$, the persona indicator and the
rater's score. The paper's maps are `jevgw` on v3.1L (rater TypeSafe Jev, $g = -\log\max\{\pi(p\mid q), 10^{-3}\}$ from the probability
that style $p$ serves the question worst, centered over the personas and divided by its standard deviation on the development
questions) and `lung` on v3.1J (rater `gpt-6-luna`, so that rater and judge differ; $g$ = minus the expected 0-10 rating of how badly
style $p$ would serve the question, centered and scaled in the same way). `dev/S_<map>.json` holds the map's scale $S$ (median norm of per-user linear fits of
the choice logits on the feature contrasts, development roster), from which the learners' prior scale follows.

### Judge and rater outputs

`dataset/judge_calls_luna/<roster>/calls.jsonl.gz`: one row per (arrival, question `j`, persona pair `i<k`, display order `pos`, label
`lab`): `pos = 1` shows persona i's answer first, `lab = 1` labels the first-shown answer A; `n_i` of `n` valid picks (4 samples) chose
persona i. `dataset/judge_calls_jev/<roster>/calls.jsonl.gz`: the same keys with `p_i`, `p_k`, Jev's probabilities of the labels
carried by persona i's and persona k's answers. `dataset/rater/jev_worst_<split>.jsonl.gz`: `{qid, p_worst}` (six probabilities, persona
order); `luna_badly_<split>.jsonl.gz`: `{qid, top_logprobs}` (for each persona, the token / log-probability pairs of its 0-10 rating).

### Human labels

`dataset/human_labels/labels.jsonl`: one row per comparison, `{id, cell, roster, arrival, j, qid, i, k, left, choice}`: the student
(arrival of the roster), question j of the arrival and its qid, the persona pair i < k, the persona shown first (as answer A) to
author 1 and to author 2, and the persona each author chose. The texts they saw are the profile (`types/cartoons.json` via
`index.json`), the question and the two answers (`dataset/texts/`). `chat_judges.jsonl`: `{id, gpt-5.6-sol, claude-opus-5}`, each the
picks of persona i among 4 sampled replies to the dataset judge's four prompts, in the order (i first, i is A), (i first, k is A),
(k first, k is A), (k first, i is A). The dataset judges' probabilities are read from `P.npy` of v3.1L and v3.1J.

### World names

A tutoring world is named `pl_<version>_<roster>_M8_t1_b24_ug_raw_rl1_f<map>`, e.g.
`pl_v3.1L_test_unseen_M8_t1_b24_ug_raw_rl1_fjevgw` (m = 8 profiles, click sharpness 1, 24 questions per session, utilities `U`,
choices drawn from the stored probabilities `P`, personas relabeled per seed, features `phi_<map>`) and
`pl_v3.1J_test_unseen_M8_t1_b24_ug_raw_rl1_flung` for the second judge. Variants: `_scrl` (scrambled null roster: every arrival's
choice probabilities and utilities relabeled by its own persona permutation, features unchanged), `_rsp` / `_rss` (the roster redrawn
for every seed, by profile / by arrival), `t<tau>_b<b>_ug_cbt` (click sharpness tau and session length b, Bradley-Terry choices; the
grid of the appendix). Simulator worlds: `ps_<population>_be<beta>_b<b>`, e.g. `ps_types_M4_th60_be4_b20`, `ps_gauss_s0.3_be12_b20`,
`ps_lowrank_r2_s1_be4_b20`, `ps_null_be4_b20`.

## Methods (policy specs)

`S2O` below stands for CODuel's population-learner options
`rule=kg:mstep=pooled:comp_var=0:n_comp=16:birth_margin=0:births=3:alpha=20:peg=0.5:peg_min=8:w_iso=0:n_init=1:em_late=1:cap=0:cap_v0=1000`.

| paper | tutoring, v3.1L | tutoring, v3.1J (second judge) | simulator |
|---|---|---|---|
| **CODuel** | `qo:S2O:xcols=1:recal=1:rfull=1:xpass=0:lama=10:aug=0:sa=1.6996` | same with `sa=2.5019` | `v3:S2O:aug=0` |
| no-transfer ablation | `none:emax:rt` | same | same |
| CODuel without recalibration | `v3:S2O:aug=0` | | |
| RUCB | `rucb:a0.51` | `rucb:a0.51` | `rucb:a1` |
| Pooled | `lab:pop:aug=0` | same | same |
| CoLSTIM | `colstim:c1:l0.1:g0.3` | `colstim:c3:l0.03:g0.3` | `colstim:c0.3:l3:g0.01` |
| COLDB | `coldb:a0.03:c0.03` | `coldb:a0.03:c0.03` | `coldb:a0.1:c0` |
| COLDB-LR | `coldbl:t2:c0.3` | `coldbl:t0.5:c0.3` | `coldbl:t0.5:c0.3` |
| Meta-TS | `lab:metats:s0=0.001:sq=0.1:aug=0` | `lab:metats:s0=0.0003:sq=1:aug=0` | `lab:metats:s0=0.03:sq=1:aug=0` |
| HierTS | `lab:hierts:s0=0.001:sq=1:aug=0` | `lab:hierts:s0=0.0003:sq=1:aug=0` | `lab:hierts:s0=0.01:sq=0.1:aug=0` |
| AdaTS (appendix) | `lab:adats:s0=0.0003:sq=10:aug=0` | `lab:adats:s0=0.003:sq=1:aug=0` | `lab:adats:s0=0.03:sq=1:aug=0` |
| random pairs | `random` | same | same |
| type-library reference (hidden information) | `ref:typelib` | same | |
| population oracle (hidden information) | | | `oracle:emax` |

`sa` is CODuel's prior scale: its learners' benchmark scale is $S' = \mathrm{sa}\sqrt{7/6}$, which equals the map's $S$ (the paper's
$B_\theta$ = 1.84 on v3.1L and 2.70 on v3.1J). In the simulator, whose features contain no rater score, CODuel runs without the
recalibration step. CODuel's ablations append one option to its spec: `:rule=emax` (greedy pairs), `:rule=lts` / `:rule=lucb:lucb_c=0.5`
(latent-bandit Thompson sampling / UCB), `:peg=0` (no evidence gate), `:alpha=5`, `:mstep=moment`, `:births=0:n_comp=1` (one tight type),
`:births=0:n_comp=1:mstep=moment` (one Gaussian with learned covariance), `:warm=1000000000:n_init=1000000000:peg=0` (no population);
the sensitivity study appends `:peg=`, `:peg_min=`, `:n_comp=`, `:bank=`, `:alpha=`, `:lama=` (recalibration ridge), `:refit=`.
The pair rules given the development profiles' tables are `refx:typelib:{greedy,kg,lts,lucb}`. The comparison methods were tuned on
the development roster and on the 16 structured simulator anchors, seeds 300-315 (`results/tuning_grids.json`).

## Reproducing the paper

### From the shipped rows

`python results/paper_stats.py <report>` prints the numbers of one table (and the numbers the text quotes from it):

| report | paper |
|---|---|
| `main` | Table 1 (benchmark results on the evaluation rosters) and the paired comparisons |
| `topic` | held-out-topic and development rosters, with their paired comparisons |
| `references` | diagnostic references |
| `profiles` | held-out-profile roster by profile, and the bootstrap over profiles |
| `phases` | excess regret in the first and last quarter of the arrivals |
| `redrawn` | rosters redrawn for every seed |
| `ablations`, `ablations-eval` | one-component changes on the development and evaluation rosters |
| `sensitivity` | CODuel's settings changed one at a time |
| `controls` | negative controls, the evidence gate, pair rules with the profiles known |
| `grid` | click sharpness x session length (development and scrambled rosters, type-library reference) |
| `sim` | simulator anchors, the b x beta grid, COLDB with long sessions |
| `tuning`, `tuning-eval` | tuning of the comparison methods; settings chosen on the evaluation rosters; the double-sample pair rule |
| `second-judge` | every roster judged by TypeSafe Jev |
| `python results/agreement.py human` | agreement of the judge with the authors' choices (and of the two larger chat judges) |
| `python results/agreement.py judges` | agreement of the two judges with each other (the second-judge appendix) |

Rows: `results/rows/<group>.npz` holds `worlds`, `specs` (string tables) and, per row, `world`, `spec` (indices), `seed`, `excess`
(excess strong regret $R_T - G_T$), `floor` ($G_T$), `total` ($R_T$), `n_users`; groups used for user-level statistics add `per_user`
(each user's excess regret, in arrival order), `arrival_ids` and `type_ids` (profile of each user); the grids add `prefix_excess`
(cumulative excess after 25, 50, 100, 200, 300 users). Scores, references (random pairs' expected regret, the floor and the fixed
pairs are closed-form, `results/refs.py`; the oracle is read from `results/oracle/`) and intervals are computed by
`results/paper_stats.py` exactly as in the paper.

### Rerunning the experiments

```bash
python results/experiments.py --list               # the experiment groups and their number of runs
python results/experiments.py main --print         # the results/bench.py commands of a group
python results/experiments.py main --jobs 32       # run them: rows in results/runs/main/ (cached; an interrupted run resumes)
python results/paper_stats.py main --from-runs     # the tables from your own rows
python results/tune.py --env v3.1L                 # tuning grids (also --env v3.1J, --env sim), then the selection
python results/oracle.py --version v3.1L --all --out results/oracle    # the per-user linear oracle (about a minute per dataset)
```

Every policy sees the same users, questions and uniform draws for a given seed, and a run is deterministic on a given machine. The
shipped rows were produced with this code on Linux clusters (NumPy 1.24 and 2.3, several CPU models). Across NumPy builds and CPUs,
the comparison methods reproduce the shipped rows exactly or to the last bits of floating point; CODuel's EM and lookahead can turn
such last-bit differences into different choices later in the run, so a rerun of a CODuel row on another machine typically differs
from the shipped row by about 1% of its excess regret (we measured 0.01% to 3.5% on 16 rows), in either direction. CODuel takes
about 5 minutes per run of 350 users on one core; the `main` group is about 40 CPU-hours, and every group together several hundred. Held-out rosters (`test*`) need `--allow-test` in `results/bench.py` (the experiment script passes it), a
guard against accidental looks at held-out data during development.

Other `results/bench.py` uses: `--worlds` takes comma-separated world names, `--policies` comma-separated specs, `--seeds` ranges
such as `0-63`; for each world it prints every policy's mean regret and the paired difference to `--competitor` (default the
no-transfer learner).

### Rebuilding the benchmark and the features

```bash
python generation/build_benchmark.py                 # rebuilds P, U (and V, U_stretch, Pcbt) of every v3.1L roster from the luna calls
python generation/build_benchmark.py --judge jev     # the same for v3.1J from the Jev calls
python generation/build_features.py                  # rebuilds phi_jevgw and phi_lung (and S) from the rater outputs
```

Each prints the largest difference between the rebuilt and the shipped arrays: 0 with NumPy 2.1 and SciPy 1.15 (the shipped
tables are reproduced exactly), and at most a few units of the last bit (about 1e-15) with other versions;
`--out <dir>` writes the rebuilt arrays. Re-querying the LLMs (`generation/judge_luna.py`, `judge_jev.py`, `rate_jev.py`,
`rate_luna.py`; keys from `OPENAI_API_KEY` / `TYPESAFE_API_KEY`) samples new outputs and therefore gives statistically equivalent, not
identical, tables. The script docstrings document the prompts and the row formats.

## Notes

- The student profiles, questions, answers and judgments are synthetic (LLM-generated); the dataset contains no data about real
  people. The only human data are the two authors' choices in `dataset/human_labels/`.
- Simulated preferences can differ from those of real students, and the style a student prefers need not be the one that best
  supports learning: TutorDuel is a testbed for learning algorithms, not evidence of what real students prefer.

## License

Code: MIT (`LICENSE`). Dataset: CC BY 4.0 (see the dataset card on Hugging Face).

## Citation

```bibtex
@misc{puech2026tutorduel,
  title  = {Online {LLM} Personalization from Pairwise Preferences Across Users},
  author = {Puech, Romain and Salakhutdinov, Georgy and Bertsimas, Dimitris},
  year   = {2026},
  url    = {https://github.com/RoGy-Labs/tutorduel-code}
}
```
