"""Tests for replay helpers, reporting, leaderboard, and a third-party Policy."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from persona_bench.env import SyntheticEnv
from persona_bench.leaderboard import write_leaderboard
from persona_bench.protocol import Policy
from persona_bench.replay import ArrivalEnv, calibrate_S, filter_arrivals, no_transfer_oracle
from persona_bench.report import i_conv, result_row
from persona_bench.runner import run_sequence
from persona_bench.world import make_world


class StubPolicy(Policy):
    name = "stub"

    def start_student(self):
        self._last = None

    def select(self, Phi):
        i, j = 0, 1
        self._last = (i, j)
        return i, j

    def update(self, outcome):
        pass


class TestReplayHelpers(unittest.TestCase):
    def test_filter_arrivals(self):
        rng = np.random.default_rng(0)
        N, b, K, Q, d = 10, 3, 4, 12, 3
        P = rng.random((N, b, K, K))
        U = rng.random((N, b, K))
        qids = rng.integers(0, Q, size=(N, b))
        profiles = [f"T{i % 4}" for i in range(N)]
        Pf, Uf, qf, mask = filter_arrivals(P, U, qids, profiles, keep=("T0", "T1"))
        self.assertEqual(int(mask.sum()), Pf.shape[0])
        self.assertTrue(all(p in ("T0", "T1") for p, m in zip(profiles, mask) if m))
        self.assertLess(Pf.shape[0], N)

    def test_calibrate_S_is_median_norm(self):
        atoms = np.array([[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
        self.assertAlmostEqual(calibrate_S(atoms), 3.0)

    def test_no_transfer_oracle_isotropic(self):
        o = no_transfer_oracle(4, 2.0)
        self.assertEqual(o["kind"], "gaussian")
        np.testing.assert_allclose(o["mu"], 0)
        np.testing.assert_allclose(np.diag(o["Sigma"]), 1.0)

    def test_arrival_env_roundtrip(self):
        rng = np.random.default_rng(1)
        K, d, N, b = 4, 3, 5, 2
        Phi = rng.normal(size=(N * b, K, d))
        Phi /= np.linalg.norm(Phi, axis=2, keepdims=True)
        U = rng.random((N, b, K))
        P = 1 / (1 + np.exp(-(U[:, :, :, None] - U[:, :, None, :])))
        qids = np.arange(N * b).reshape(N, b)
        env = ArrivalEnv(Phi=Phi, P=P, U=U, qids=qids, S=1.0, beta=1.0)
        res = run_sequence(env, [StubPolicy()], N=N, b=b, seed=0, query_kind="chain")
        self.assertEqual(list(res)[0], "stub")
        self.assertEqual(res["stub"].regret.shape, (N, b))


class TestIConvAndLeaderboard(unittest.TestCase):
    def test_i_conv_finds_crossing(self):
        learner = np.array([5.0, 4.0, 3.0, 1.0, 1.0, 1.0])
        oracle = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(i_conv(learner, oracle, window=2), 5)

    def test_i_conv_never(self):
        learner = np.ones(10) * 5
        oracle = np.ones(10)
        self.assertIsNone(i_conv(learner, oracle, window=3))

    def test_stub_policy_on_leaderboard(self):
        w = make_world(1, population="types", M=2, theta_deg=90.0, beta=12.0)
        env = SyntheticEnv(w)
        rows = []
        for seed in (0, 1):
            res = run_sequence(env, [StubPolicy(), "random", "none:emax"], N=4, b=4, seed=seed)
            for r in res.values():
                rows.append(result_row(r, extra={"world": "toy", "seed": seed}))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "leaderboard.json"
            table = write_leaderboard(rows, path, competitor="none:emax")
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".md").exists())
            names = [r["policy"] for r in table["worlds"]["toy"]["ranking"]]
            self.assertIn("stub", names)
            self.assertIn("random", names)
            json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
