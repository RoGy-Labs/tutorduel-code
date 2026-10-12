# persona_lab

Shared inference code and the population-learning comparison methods of the paper. The proposed method, CODuel, is in
`persona_v3/`. Importing the package registers the policies with spec prefix `lab:` (`results/bench.py` does this).

| file | content |
|---|---|
| `infer.py` | Batched Laplace inference for pairwise (Bradley-Terry) feedback and Gaussian-mixture beliefs, used by CODuel and the comparison methods. |
| `hb.py`, `kde.py` | The mixture-prior learners that CODuel's `persona_v3/mix.py` builds on (`MixV3` extends `KDEPrior`, which extends `HBMix`), and the option parser of the `lab:`, `v3:` and `qo:` specs. |
| `pop.py` | The pooled baseline (`lab:pop`). |
| `metats.py` | Meta-TS, HierTS and AdaTS adapted to pairwise choices (`lab:metats`, `lab:hierts`, `lab:adats`). |
