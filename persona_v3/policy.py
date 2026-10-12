"""CODuel: gated transfer policy: `v3:<key=value...>`.

Two learners see the same log (Phi and the policy's own clicks only):
  base  an exact replica of the no-transfer learner `none:emax:rt` (matched-isotropic prior on Phi, within-student
        Laplace, two highest posterior means, exact ties broken at random);
  mix   the population mixture learner `MixV3` (EM with births; identification-aware pair rule).

Prequential evidence gate: every round both learners predict the click on the pair that is shown; the per-student sum
of log-likelihood differences (mix minus base) is recorded when the student leaves. The population is used only while
the mean difference over past students is significantly positive (one-sided z > `peg`, at least `peg_min` students).
With nothing to transfer the gate stays shut and the policy IS the no-transfer learner; the mixture keeps learning
from the same clicks so that it can be switched on as soon as its predictions are better.
"""
from __future__ import annotations

import numpy as np

from persona_bench.baselines.laplace_policy import LaplaceDueling
from persona_bench.protocol import Policy
from persona_lab.hb import parse_opts
from persona_lab.infer import probit_click_prob

from .mix import MixV3

GATE_DEFAULTS = dict(peg=1.0, peg_min=8, mode="full", gate="z", peg_sum=4.0, peg_kg=4.0, base="raw", ref="none", peg_ref=2.0, ref_rounds=100, veto=-1.0, z_swap=2.0, n_min=3)


class GatedV3(Policy):
    def __init__(self, name="v3", **opts):
        self.name = name
        self.gate = {k: opts.pop(k, v) for k, v in GATE_DEFAULTS.items()}
        self.mix = MixV3(name=name + "/mix", **opts)
        if self.gate["base"] == "aug":
            # no-transfer learner on the same augmented features (isotropic prior, never learns a population):
            # the gate then measures the value of the POPULATION alone, not of the persona-index feature
            bopts = dict(opts); bopts.update(rule="emax", warm=10 ** 9, n_init=10 ** 9)
            self.base = MixV3(name=name + "/base", **bopts)
        else:
            self.base = LaplaceDueling(population="none", rule="emax", random_ties=True, name=name + "/base")
        self.ref = None
        if self.gate["ref"] == "aug":
            # second reference: a no-transfer learner on the augmented features that only observes. The population is
            # used only if the mixture predicts the clicks better than BOTH no-transfer learners (the persona one-hot
            # alone is a legitimate within-session gain on some worlds and a loss on others; it is not transfer)
            ropts = dict(opts); ropts.update(rule="emax", warm=10 ** 9, n_init=10 ** 9)
            self.ref = MixV3(name=name + "/ref", **ropts)

    def reset(self, rng, context):
        super().reset(rng, context)
        seeds = rng.integers(0, 2 ** 31 - 1, size=2)
        self.base.reset(np.random.default_rng(int(seeds[0])), context)
        self.mix.reset(np.random.default_rng(int(seeds[1])), context)
        if self.ref is not None:
            self.ref.reset(np.random.default_rng(int(seeds[1]) + 1), context)
        self.S2 = 0.0
        self.ev: list[float] = []
        self.gated = self.gate["gate"] != "z" or self.gate["peg"] > 0      # peg <= 0 with the z-gate = always open
        self.open = not self.gated
        self.n_open = 0
        self.n_students = 0
        self.z = 0.0
        self.S = 0.0                 # cumulative log-likelihood ratio (mixture vs no-transfer replica) over past students
        self.mix.rule_override = None

    def start_student(self):
        self.base.start_student()
        self.mix.start_student()
        if self.ref is not None:
            self.ref.start_student()
        self._ev = 0.0
        self._ev2 = 0.0

    @staticmethod
    def _mix_prob(k, i, j, Phi):
        Fk = k._aug(Phi) * k.scale[None, :]
        xk = k.beta * (Fk[i] - Fk[j])
        return float(k.post.pi @ probit_click_prob(k.post.m @ xk, np.einsum("d,mde,e->m", xk, k.post.C, xk)))

    def _vote_gain(self, Phi, c, bj):
        """Weighted mean and z-score, across the population components (session posterior weights), of the predicted
        utility of persona c minus persona bj on this question (persona_lab.safe.SafeKDE's vote gain)."""
        k = self.mix
        F = k._aug(Phi) * k.scale[None, :]
        U = k.post.m @ F.T
        has_op = (U.max(axis=1) - U.min(axis=1)) > 1e-9
        w = k.post.pi * has_op
        if w.sum() <= 0:
            return 0.0, 0.0
        w = w / w.sum()
        delta = U[:, c] - U[:, bj]
        m = float(w @ delta); v = float(w @ (delta - m) ** 2)
        n_eff = 1.0 / float((w ** 2).sum())
        return m, m / np.sqrt(max(v, 1e-12) / max(n_eff, 1.0))

    def _click_probs(self, i, j, Phi):
        b_ = self.base
        if isinstance(b_, MixV3):
            pb = self._mix_prob(b_, i, j, Phi)
        else:
            x = b_.beta * (Phi[i] - Phi[j])
            pb = float(b_.post.pi @ probit_click_prob(b_.post.m @ x, np.einsum("d,mde,e->m", x, b_.post.C, x)))
        k = self.mix
        Fk = k._aug(Phi) * k.scale[None, :]
        xk = k.beta * (Fk[i] - Fk[j])
        pk = float(k.post.pi @ probit_click_prob(k.post.m @ xk, np.einsum("d,mde,e->m", xk, k.post.C, xk)))
        return pb, pk

    def select(self, Phi):
        bi, bj = self.base.select(Phi)
        if self.open and self.gate["mode"] == "hedgez":
            # the incumbent's mechanism with this population: keep the replica's leader; replace its second slot by the
            # population's preferred partner only if the components' votes for the swap are significant (z > z_swap)
            i, j = bi, bj
            if self.mix.post.M >= 2 and len(self.mix.sess_X) >= self.gate["n_min"]:
                ci, cj = self.mix.select(Phi, fixed=bi)
                c = cj if ci == bi else ci
                if c != bj:
                    m, z = self._vote_gain(Phi, c, bj)
                    if z > self.gate["z_swap"] and m > 0:
                        j = c
            self.mix._last = (i, j, self.mix._aug(Phi))
            self.base._last = (i, j, Phi)
        elif self.open:
            i, j = self.mix.select(Phi, fixed=(bi if self.gate["mode"] == "hedge" else None))
            self.base._last = (i, j, self.base._aug(Phi)) if isinstance(self.base, MixV3) else (i, j, Phi)
        else:
            i, j = bi, bj
            self.mix.observe(self.mix._aug(Phi) * self.mix.scale[None, :])
            self.mix._last = (i, j, self.mix._aug(Phi))
        self._p = self._click_probs(i, j, Phi)
        if self.ref is not None:
            self.ref.observe(self.ref._aug(Phi) * self.ref.scale[None, :])
            self.ref._last = (i, j, self.ref._aug(Phi))
            self._pr = self._mix_prob(self.ref, i, j, Phi)
        return i, j

    def update(self, outcome):
        pb, pk = (min(max(p, 1e-6), 1 - 1e-6) for p in self._p)
        self._ev += float(np.log(pk if outcome else 1 - pk) - np.log(pb if outcome else 1 - pb))
        self.base.update(outcome)
        self.mix.update(outcome)
        if self.ref is not None:
            pr = min(max(self._pr, 1e-6), 1 - 1e-6)
            if self.mix.t < self.gate["ref_rounds"]:       # the population's value is in the first rounds of a session
                self._ev2 += float(np.log(pk if outcome else 1 - pk) - np.log(pr if outcome else 1 - pr))
            self.ref.update(outcome)

    def end_student(self):
        self.base.end_student()
        self.mix.end_student()
        if self.ref is not None:
            self.ref.end_student()
            self.S2 += self._ev2
        self.n_students += 1
        self.n_open += int(self.open)
        if self.gated:
            self.ev.append(self._ev)
            n = len(self.ev)
            self.S += self._ev
            if n >= self.gate["peg_min"]:
                e = np.asarray(self.ev)
                self.z = float(e.mean() / max(e.std(ddof=1) / np.sqrt(n), 1e-12))
                if self.gate["gate"] == "or":
                    # either evidence route opens the gate: the anytime-valid dual test (fast where the population is
                    # click-predictive early, e.g. few sharp types) or the z-test on the raw evidence (many types)
                    okA = self.S > self.gate["peg_sum"] and (self.ref is None or self.S2 > self.gate["peg_ref"])
                    okZ = self.z > self.gate["peg"]
                    self.open = bool(okA or okZ)
                    self.mix.rule_override = None if (self.open and (self.S > self.gate["peg_kg"] or okZ)) else "emax"
                elif self.gate["gate"] == "sum":
                    # anytime-valid gate: the product of predictive likelihood ratios is a test martingale under the
                    # no-transfer model, so P(sum ever exceeds c) <= exp(-c) (Ville); the exploring rule is allowed only
                    # once the evidence is larger still (peg_kg), the greedy rule in between
                    if self.gate["veto"] >= 0:      # hybrid: open on the raw evidence unless the augmented reference clearly wins
                        ok2 = self.ref is None or self.S2 > -self.gate["veto"]
                    else:
                        ok2 = (self.ref is None or self.S2 > self.gate["peg_ref"]) and self.mix.mass >= self.mix.opt["mass_min"]
                    self.open = bool(self.S > self.gate["peg_sum"] and ok2)
                    self.mix.rule_override = None if (self.S > self.gate["peg_kg"] and ok2) else "emax"
                else:
                    self.open = bool(self.z > self.gate["peg"])

    def predict_best(self, Phi):
        return self.mix.predict_best(Phi) if self.open else self.base.predict_best(Phi)

    def diagnostics(self):
        d = {"gate_open": bool(self.open), "open_share": self.n_open / max(self.n_students, 1), "z": self.z, "S": self.S, "S2": self.S2, "ev": (self.ev[-1] if self.ev else 0.0)}
        d.update({f"mix_{k}": v for k, v in self.mix.diagnostics().items()})
        return d


def make_v3(spec: str) -> Policy:
    parts = spec.split(":")
    defaults = dict(MixV3.DEFAULTS); defaults.update(GATE_DEFAULTS)
    opts = parse_opts(parts[1:], defaults)
    return GatedV3(name=spec, **opts)
