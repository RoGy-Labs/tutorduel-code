"""Load the published LLM dataset if it is present."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from persona_bench.dataset import dataset_root, load_types, types_env
from persona_bench.runner import run_sequence
import persona_bench  # noqa: F401  registers none:emax


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.root = dataset_root()
        if not (self.root / "types" / "P.npy").exists():
            self.skipTest("dataset/ not packed")

    def test_types_shapes_and_no_nan(self):
        pack = load_types(M=8, version="v1")
        self.assertEqual(pack["P"].shape, (300, 12, 6, 6))
        self.assertEqual(pack["U"].shape, (300, 12, 6))
        self.assertEqual(pack["Phi"].shape[1:], (6, 8))
        self.assertFalse(np.isnan(pack["P"]).any())
        self.assertTrue(np.allclose(pack["U"].max(-1), 1.0))
        self.assertEqual(pack["index"]["n_missing_logprob_tokens"], 0)

    def test_m2_filter(self):
        pack = load_types(M=2, version="v1")
        self.assertTrue(set(pack["profiles"]) <= {"T1_major", "T2_anxious"})
        self.assertEqual(pack["P"].shape[0], pack["U"].shape[0])
        self.assertGreater(pack["P"].shape[0], 0)

    def test_env_smoke(self):
        env = types_env(M=8, version="v1")
        res = run_sequence(env, ["none:emax"], N=2, b=2, seed=0, query_kind="chain")
        self.assertEqual(res["none:emax"].regret.shape, (2, 2))
