"""Tests for the v2 additions: utility scales, click sharpness, relabeling, pack cycling, random tie-breaking."""
import unittest

import numpy as np

from persona_bench import relabel_personas, run_sequence, subset_arrivals, type_oracle, types_env
from persona_bench.dataset import dataset_root, load_types
from persona_bench.report import excess_curves
from persona_bench.utility import R_LOGIT, bt_scores, cbt_table, sharpen, stretch, top_anchored

HAVE = (dataset_root() / "types" / "P.npy").exists() and (dataset_root() / "v2a" / "types" / "U.npy").exists()


@unittest.skipUnless(HAVE, "dataset or dataset/v2a missing (python tools/build_dataset.py derive)")
class TestV2(unittest.TestCase):
    def test_utility_regression(self):
        pack = load_types(version="v1")
        V = bt_scores(pack["P"])
        self.assertLess(np.abs(stretch(V) - pack["U"]).max(), 1e-4)            # shipped v1 utilities are stretched BT scores
        U = top_anchored(V)
        self.assertTrue(np.allclose(U.max(-1), 1.0))
        self.assertEqual(int((U < 0).sum()), 0)
        self.assertTrue(np.allclose(U, load_types(version="v1", utility="global")["U"]))
        # regret in the global scale is the logit gap divided by R, whatever the question
        self.assertTrue(np.allclose((1 - U) * R_LOGIT, V.max(-1, keepdims=True) - V))

    def test_tau_one_is_identity_and_P_stays_raw(self):
        e1, e4 = types_env(M=2, version="v1"), types_env(M=2, version="v1", tau=4.0)
        self.assertIs(e1._Pclick, e1.P)
        self.assertTrue(np.array_equal(e1.P, e4.P))                             # oracle atoms and S are fitted on raw P
        self.assertGreater(np.abs(e4._Pclick - 0.5).mean(), 2 * np.abs(e1.P - 0.5).mean())
        self.assertTrue(np.allclose(sharpen(e1.P, 1.0), e1.P))
        a = run_sequence(types_env(M=2, version="v1"), ["none:emax"], N=10, b=12, seed=3, query_kind="chain")["none:emax"]
        b = run_sequence(types_env(M=2, version="v1", tau=1.0, click="raw"), ["none:emax"], N=10, b=12, seed=3, query_kind="chain")["none:emax"]
        self.assertTrue(np.array_equal(a.pairs, b.pairs) and np.array_equal(a.outcomes, b.outcomes))

    def test_cbt_clicks_agree_with_utilities(self):
        env = types_env(M=2, version="v1", click="cbt", utility="global")
        iu = np.triu_indices(6, 1)
        self.assertTrue(np.all((env._Pclick[..., iu[0], iu[1]] > 0.5) == (env.U[..., iu[0]] > env.U[..., iu[1]])))
        self.assertTrue(np.allclose(cbt_table(bt_scores(env.P)), env.Pcbt))

    def test_relabel_invariance(self):
        env = types_env(M=4, version="v1", utility="global")
        perm = np.random.default_rng(0).permutation(6)
        env2 = relabel_personas(env, perm)
        self.assertTrue(np.allclose(np.sort(env.U, -1), np.sort(env2.U, -1)))
        r1 = run_sequence(env, ["random"], N=30, b=12, seed=1, query_kind="chain")["random"]
        r2 = run_sequence(env2, ["random"], N=30, b=12, seed=1, query_kind="chain")["random"]
        self.assertAlmostEqual(r1.G_T, r2.G_T, places=9)                        # the floor does not depend on labels
        inv = np.argsort(perm)                                                  # same shown answers under the other labelling
        self.assertTrue(np.allclose(env.U[0, 0, perm[2]], env2.U[0, 0, 2]) and inv[perm[2]] == 2)

    def test_pack_cycles_beyond_b(self):
        env = types_env(M=2, version="v1")
        st = env.query_stream("chain", None, 0, 24)
        seq = [st.next(t, None) for t in range(24)]
        self.assertEqual(seq[:12], seq[12:])
        r = run_sequence(env, ["random"], N=5, b=24, seed=0, query_kind="chain")["random"]
        self.assertEqual(r.regret.shape, (5, 24))

    def test_rt_breaks_only_ties(self):
        env = types_env(M=2, version="v1")
        a = run_sequence(env, ["none:emax"], N=40, b=12, seed=5, query_kind="chain")["none:emax"]
        b = run_sequence(env, ["none:emax:rt"], N=40, b=12, seed=5, query_kind="chain")["none:emax:rt"]
        self.assertTrue(np.all(np.sort(a.pairs[:, 0], -1) == [0, 1]))           # the legacy rule always opens with (0, 1)
        self.assertGreater(len({tuple(sorted(p)) for p in b.pairs[:, 0]}), 5)   # rt opens with a random pair

    def test_type_oracle_and_subset(self):
        env = types_env(M=4, version="v1")
        half = subset_arrivals(env, np.arange(0, env.N, 2))
        orc = type_oracle(half)
        self.assertEqual(orc["atoms"].shape, (4, env.d + env.K))
        self.assertEqual(sorted(orc["names"]), sorted(set(env.profiles)))
        self.assertTrue(np.array_equal(half.arrival_ids, env.arrival_ids[::2]))

    def test_excess_curves(self):
        r = run_sequence(types_env(M=2, version="v1", utility="global"), ["random"], N=30, b=12, seed=0, query_kind="chain")["random"]
        c = excess_curves(r)
        self.assertAlmostEqual(c["excess01"], r.total - r.G_T, places=9)
        self.assertEqual(len(c["by_round01"]), 12)
        self.assertGreaterEqual(min(c["excess_per_student01"]), -1e-12)


if __name__ == "__main__":
    unittest.main()
