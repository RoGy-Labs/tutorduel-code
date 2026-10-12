"""Tests for the fully online benchmark (adapted from simulator_specifications.md §13)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from persona_bench.baselines.simple import MaxDiameter, RandomPairs
from persona_bench.env import SyntheticEnv, TableEnv
from persona_bench.infer import Mixture, condition, expected_max_pair, fit_laplace
from persona_bench.registry import make_policy
from persona_bench.runner import run_sequence
from persona_bench.world import make_type_atoms, make_world, sample_sphere, sigmoid


class TestWorld(unittest.TestCase):
    def test_lambda0_query_independent(self):
        w = make_world(1, lam=0.0)
        z1, z2 = sample_sphere(np.random.default_rng(0), w.d, 2)
        np.testing.assert_allclose(w.phi(z1), w.phi(z2))

    def test_lambda1_query_dependent_and_unit_norm(self):
        w = make_world(1, lam=1.0)
        z1, z2 = sample_sphere(np.random.default_rng(0), w.d, 2)
        self.assertFalse(np.allclose(w.phi(z1), w.phi(z2)))
        np.testing.assert_allclose(np.linalg.norm(w.phi(z1), axis=1), 1.0)

    def test_type_atoms_orthogonal_at_90(self):
        rng = np.random.default_rng(0)
        for M in (2, 4, 8):
            W = make_type_atoms(rng, 8, M, 90.0, 1.0)
            G = W @ W.T
            np.testing.assert_allclose(np.diag(G), 1.0)
            off = G - np.diag(np.diag(G))
            self.assertLess(np.abs(off).max(), 1e-6)

    def test_type_atoms_angle_60(self):
        rng = np.random.default_rng(0)
        W = make_type_atoms(rng, 8, 4, 60.0, 1.0)
        G = W @ W.T
        off = G[~np.eye(4, dtype=bool)]
        np.testing.assert_allclose(off, np.cos(np.deg2rad(60)), atol=1e-6)

    def test_diagnostics_present_and_ok(self):
        for pop in ("types", "gaussian", "lowrank"):
            w = make_world(3, population=pop)
            for k in ("query_disagreement_rate", "type_pair_disagreement_rate", "eps", "mean_gap", "diagnostics_ok"):
                self.assertIn(k, w.diagnostics)
            self.assertTrue(w.diagnostics["diagnostics_ok"])

    def test_degenerate_types_rejected(self):
        # force all atoms equal: disagreement is zero -> must be flagged
        w = make_world(5, population="types", M=4, theta_deg=60.0)
        w.population.atoms[:] = w.population.atoms[0]
        from persona_bench.world import _diagnostics
        d = _diagnostics(w, np.random.default_rng(0))
        self.assertLess(d["query_disagreement_rate"], 0.05)
        self.assertLess(d["type_pair_disagreement_rate"], 0.05)

    def test_lowrank_users_in_subspace(self):
        w = make_world(7, population="lowrank", rank=2, s=1.0)
        W, _ = w.population.sample(np.random.default_rng(0), 200)
        U = w.population.subspace
        resid = W - (W @ U) @ U.T
        self.assertLess(np.linalg.norm(resid, axis=1).mean(), 0.15)

    def test_click_nearly_deterministic_at_high_beta(self):
        self.assertGreater(sigmoid(50 * 0.5), 0.999)


class TestQueries(unittest.TestCase):
    def test_chain_is_correlated(self):
        w = make_world(1)
        env = SyntheticEnv(w)
        st = env.sample_students(np.random.default_rng(0), 1)[0]
        s = env.query_stream("chain", np.random.default_rng(1), st, 20, rho=0.9)
        Z = np.array([s.next(t, None) for t in range(20)])
        self.assertGreater(np.mean(np.sum(Z[1:] * Z[:-1], axis=1)), 0.7)

    def test_pool_cycles(self):
        w = make_world(1)
        env = SyntheticEnv(w)
        st = env.sample_students(np.random.default_rng(0), 1)[0]
        s = env.query_stream("pool", np.random.default_rng(1), st, 9, pool_size=3)
        Z = np.array([s.next(t, None) for t in range(9)])
        np.testing.assert_allclose(Z[0], Z[3]); np.testing.assert_allclose(Z[1], Z[7])

    def test_adversary_hurts_a_fixed_guess(self):
        w = make_world(1)
        env = SyntheticEnv(w)
        st = env.sample_students(np.random.default_rng(0), 1)[0]

        class Dumb:
            def predict_best(self, Phi):
                return 0

        adv = env.query_stream("adversary", np.random.default_rng(1), st, 10)
        iid = env.query_stream("iid", np.random.default_rng(1), st, 10)
        la = np.mean([env.utilities(st, adv.next(t, Dumb())).max() - env.utilities(st, adv.next(t, Dumb()))[0] for t in range(10)])
        li = np.mean([env.utilities(st, iid.next(t, Dumb())).max() - env.utilities(st, iid.next(t, Dumb()))[0] for t in range(10)])
        self.assertGreater(la, li)


class TestRunner(unittest.TestCase):
    def test_regret_nonnegative_pairs_distinct_and_crn(self):
        w = make_world(11, population="types", M=4, beta=8.0)
        env = SyntheticEnv(w)
        r1 = run_sequence(env, ["random", "max_diameter", "none:greedy_ts"], N=5, b=10, seed=3)
        r2 = run_sequence(env, ["random", "max_diameter", "none:greedy_ts"], N=5, b=10, seed=3)
        for k in r1:
            self.assertGreaterEqual(r1[k].regret.min(), -1e-12)
            self.assertTrue((r1[k].pairs[:, :, 0] != r1[k].pairs[:, :, 1]).all())
            np.testing.assert_allclose(r1[k].regret, r2[k].regret, atol=1e-12)

    def test_max_diameter_deterministic(self):
        w = make_world(1)
        p = MaxDiameter(); p.reset(np.random.default_rng(0), {"K": w.K, "d": w.d})
        Phi = w.phi(sample_sphere(np.random.default_rng(2), w.d))
        self.assertEqual(p.select(Phi), p.select(Phi))

    def test_policy_never_sees_utilities(self):
        # the policy interface receives only Phi and clicks; the runner passes no w
        p = make_policy("none:emax")
        self.assertFalse(hasattr(p, "w_true"))
        w = make_world(1)
        env = SyntheticEnv(w)
        run_sequence(env, [p], N=3, b=5, seed=0)
        for attr in vars(p):
            self.assertNotIn("true", attr)

    def test_rating_mode_runs(self):
        w = make_world(2)
        env = SyntheticEnv(w)
        r = run_sequence(env, ["none:emax", "random"], N=4, b=6, seed=0, feedback="rating")
        for k in r:
            self.assertGreaterEqual(r[k].regret.min(), -1e-12)

    def test_table_env_runs(self):
        rng = np.random.default_rng(0)
        Q, K, d, Sn = 5, 4, 3, 2
        Phi = rng.normal(size=(Q, K, d))
        U = rng.random(size=(Sn, Q, K))
        P = sigmoid(4 * (U[:, :, :, None] - U[:, :, None, :]))
        env = TableEnv(Phi=Phi, P=P, U=U, chains=[[0, 1, 2], [3, 4]])
        for q in ("iid", "chain", "pool", "adversary"):
            r = run_sequence(env, ["random", "none:emax"], N=3, b=4, seed=0, query_kind=q)
            for k in r:
                self.assertGreaterEqual(r[k].regret.min(), -1e-12)


class TestMisspecificationAndBaselines(unittest.TestCase):
    def test_links_differ_from_logistic_as_intended(self):
        w = make_world(4, population="types", M=4, beta=12.0)
        st = SyntheticEnv(w).sample_students(np.random.default_rng(0), 1)[0]
        q = sample_sphere(np.random.default_rng(1), w.d)
        p_logit = SyntheticEnv(w).click_prob(st, q, 0, 1)
        p_lapse = SyntheticEnv(w, link="lapse", lapse=0.15).click_prob(st, q, 0, 1)
        p_probit = SyntheticEnv(w, link="probit").click_prob(st, q, 0, 1)
        np.testing.assert_allclose(p_lapse, 0.85 * p_logit + 0.075, atol=1e-12)
        self.assertTrue(0.0 < p_probit < 1.0)
        # probit tails are lighter: a strong preference is more deterministic than under the logit
        V = SyntheticEnv(w).utilities(st, q)
        big = np.argsort(V)
        z = 12.0 * (V[big[-1]] - V[big[0]])
        if z > 3:
            self.assertGreater(SyntheticEnv(w, link="probit").click_prob(st, q, int(big[-1]), int(big[0])),
                               SyntheticEnv(w).click_prob(st, q, int(big[-1]), int(big[0])))
        with self.assertRaises(ValueError):
            SyntheticEnv(w, link="cauchy")

    def test_eps_min_below_eps(self):
        w = make_world(6, population="types", M=4, beta=12.0)
        self.assertIn("eps_min", w.diagnostics)
        self.assertLessEqual(w.diagnostics["eps_min"], w.diagnostics["eps"] + 1e-12)

    def test_external_baselines_run_and_never_see_utilities(self):
        w = make_world(8, population="types", M=4, beta=8.0)
        env = SyntheticEnv(w)
        res = run_sequence(env, ["colstim", "counts", "onboarding"], N=6, b=8, seed=1)
        for k, r in res.items():
            self.assertGreaterEqual(r.regret.min(), -1e-12)
            self.assertTrue((r.pairs[:, :, 0] != r.pairs[:, :, 1]).all())
            self.assertGreaterEqual(r.avg_regret.min(), r.weak_regret.min() - 1e-12)
            self.assertTrue((r.avg_regret + 1e-12 >= r.weak_regret).all())
        for spec in ("colstim", "counts", "onboarding"):
            p = make_policy(spec)
            run_sequence(env, [p], N=2, b=4, seed=0)
            for attr in vars(p):
                self.assertNotIn("true", attr)

    def test_onboarding_gets_a_stated_favourite_and_counts_learn(self):
        w = make_world(9, population="types", M=2, theta_deg=90.0, beta=12.0)
        env = SyntheticEnv(w)
        st = env.sample_students(np.random.default_rng(0), 1)[0]
        f = env.stated_favourite(st)
        self.assertTrue(0 <= f < env.K)
        res = run_sequence(env, ["counts", "random"], N=40, b=20, seed=2)
        self.assertLess(res["counts"].regret[20:].mean(), res["random"].regret[20:].mean())


class TestLaplace(unittest.TestCase):
    def test_fit_matches_scipy_on_one_problem(self):
        rng = np.random.default_rng(0)
        b, d = 30, 4
        X = rng.normal(size=(1, b, d)); w_true = rng.normal(size=d)
        o = (rng.random(b) < sigmoid(X[0] @ w_true)).astype(float)[None]
        m0 = np.zeros((1, d)); P0 = np.eye(d)[None] * 2.0
        w, H, f, ev = fit_laplace(X, o, m0, P0)
        from scipy.optimize import minimize
        obj = lambda v: np.sum(np.logaddexp(0, X[0] @ v) - o[0] * (X[0] @ v)) + 0.5 * 2.0 * v @ v
        ref = minimize(obj, np.zeros(d)).x
        np.testing.assert_allclose(w[0], ref, atol=1e-4)

    def test_evidence_prefers_true_component(self):
        rng = np.random.default_rng(1)
        d, b = 6, 40
        atoms = np.eye(d)[:3]
        prior = Mixture.atoms(atoms, var=1e-3)
        X = 6.0 * rng.normal(size=(b, d))
        o = (rng.random(b) < sigmoid(X @ atoms[1])).astype(float)
        post, _ = condition(prior, X, o)
        self.assertEqual(int(post.pi.argmax()), 1)

    def test_expected_max_pair_monotone(self):
        self.assertGreater(expected_max_pair(0.0, 0.0, 1.0), 0.0)
        np.testing.assert_allclose(expected_max_pair(1.0, 0.0, 1e-12), 1.0, atol=1e-6)

    def test_matched_iso_prior_scale(self):
        p = make_policy("none:emax")
        p.reset(np.random.default_rng(0), {"K": 6, "d": 8, "S": 1.0, "beta": 1.0, "b": 10, "N": 1})
        np.testing.assert_allclose(np.trace(p.prior.C[0]), 1.0)
        np.testing.assert_allclose(p.prior.C[0], np.eye(8) / 8)


class TestPooled(unittest.TestCase):
    def test_pooled_runs_and_refits(self):
        w = make_world(1, population="types", M=2, theta_deg=90.0, beta=12.0)
        env = SyntheticEnv(w)
        res = run_sequence(env, ["pooled", "none:emax"], N=8, b=6, seed=3)
        self.assertGreaterEqual(res["pooled"].regret.min(), -1e-12)
        p = make_policy("pooled")
        run_sequence(env, [p], N=4, b=5, seed=4)
        self.assertEqual(len(p.log_X), 4)
        self.assertFalse(np.allclose(p.w, 0.0))


class TestStrongRegret(unittest.TestCase):
    def test_best_worst_costs_more_than_best_second(self):
        V = np.array([1.0, 0.8, 0.0, 0.2, 0.4, 0.6])
        r_star = 2 * V.max() - V[0] - V[1]
        r_bad = 2 * V.max() - V[0] - V[2]
        self.assertGreater(r_bad, r_star)
        self.assertAlmostEqual(r_star, 0.2)
        self.assertAlmostEqual(r_bad, 1.0)

    def test_floor_and_weak_identities(self):
        w = make_world(1, population="types", M=2, theta_deg=90.0, beta=12.0)
        env = SyntheticEnv(w)
        res = run_sequence(env, ["none:emax", "random"], N=6, b=8, seed=0)
        for r in res.values():
            self.assertTrue((r.regret + 1e-12 >= r.weak_regret).all())
            self.assertTrue((r.regret + 1e-12 >= r.floor).all())
            self.assertTrue((r.weak_regret + 1e-12 >= 0).all())
            self.assertTrue((r.floor + 1e-12 >= 0).all())

    def test_emax_picks_two_highest_means(self):
        p = make_policy("none:emax")
        p.reset(np.random.default_rng(0), {"K": 6, "d": 8, "S": 1.0, "beta": 1.0, "b": 10, "N": 1})
        p.start_student()
        rng = np.random.default_rng(0)
        Phi = rng.normal(size=(6, 8))
        Phi /= np.linalg.norm(Phi, axis=1, keepdims=True)
        w = np.array([3.0, 2.0, -1.0, 0.0, 0.5, 1.0, -0.2, 0.3])
        p._mean_utility = lambda Phi, _w=w: Phi @ _w
        i, j = p.select(Phi)
        order = np.argsort(-(Phi @ w))
        self.assertEqual({i, j}, {int(order[0]), int(order[1])})

    def test_pooled_picks_two_highest(self):
        p = make_policy("pooled")
        p.reset(np.random.default_rng(0), {"K": 6, "d": 3, "S": 1.0, "beta": 1.0, "b": 4, "N": 2})
        Phi = np.eye(6, 3)
        p.w = np.array([5.0, 3.0, 1.0])
        i, j = p.select(Phi)
        self.assertEqual({i, j}, {0, 1})


if __name__ == "__main__":
    unittest.main()


class TestColdb(unittest.TestCase):
    def test_coldb_runs_clusters_and_never_sees_utilities(self):
        w = make_world(8, population="types", M=4, theta_deg=60.0, beta=12.0)
        env = SyntheticEnv(w)
        res = run_sequence(env, ["coldb", "coldb:a0.03", "coldbc:a0.05"], N=12, b=10, seed=1)
        for k, r in res.items():
            self.assertGreaterEqual(r.regret.min(), -1e-12)
            self.assertTrue((r.pairs[:, :, 0] != r.pairs[:, :, 1]).all())
            self.assertIn("n_clusters", r.diagnostics[-1])
        # with the paper's constant no edge is ever deleted at b=10: a single cluster
        self.assertEqual(res["coldb"].diagnostics[-1]["n_clusters"], 1)
        p = make_policy("coldb:a0.03")
        run_sequence(env, [p], N=3, b=4, seed=0)
        for attr in vars(p):
            self.assertNotIn("true", attr)

    def test_coldb_recovers_clusters_with_long_sessions(self):
        w = make_world(9, population="types", M=2, theta_deg=90.0, beta=12.0)
        env = SyntheticEnv(w)
        res = run_sequence(env, ["coldb:a0.05"], N=16, b=80, seed=2)
        self.assertEqual(res["coldb:a0.05"].diagnostics[-1]["n_clusters"], 2)
