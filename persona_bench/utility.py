"""Utilities from pairwise click tables: Bradley-Terry scores and the two utility scales.

`stretch` is the v1 scale (favourite = 1, worst = 0 on every question). `top_anchored` is the v2 scale: the favourite
is 1 and score gaps are divided by one declared constant R (logits), the same for every student and question, so
that a question on which the student hardly cares costs little regret.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

R_LOGIT = 4.0


def bt_scores(P: np.ndarray, ridge: float = 0.05, iters: int = 50) -> np.ndarray:
    """Ridge Bradley-Terry scores, centred, for click tables P[..., K, K] (P[i, j] = Pr(i chosen over j)).
    Vectorised Newton; identical (to ~1e-6) to minimising the ridge negative log-likelihood cell by cell."""
    P = np.asarray(P, dtype=float)
    K = P.shape[-1]
    off = 1.0 - np.eye(K)
    v = np.zeros(P.shape[:-1])
    for _ in range(iters):
        s = expit(v[..., :, None] - v[..., None, :])
        g = -((P - s) * off).sum(-1) + ridge * v
        w = s * (1 - s) * off
        H = -w + np.eye(K) * (w.sum(-1)[..., None, :] + ridge)
        step = np.linalg.solve(H, g[..., None])[..., 0]
        v = v - step
        if np.abs(step).max() < 1e-12:
            break
    return v - v.mean(-1, keepdims=True)


def stretch(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(-1, keepdims=True), v.max(-1, keepdims=True)
    return (v - lo) / np.maximum(hi - lo, 1e-9)


def top_anchored(v: np.ndarray, R: float = R_LOGIT) -> np.ndarray:
    return 1.0 - (v.max(-1, keepdims=True) - v) / R


def cbt_table(v: np.ndarray) -> np.ndarray:
    """Bradley-Terry-consistent click table sigma(v_i - v_j)."""
    return expit(v[..., :, None] - v[..., None, :])


def sharpen(P: np.ndarray, tau: float) -> np.ndarray:
    """Click table with logits multiplied by tau (tau = 1 returns P unchanged, bit for bit)."""
    if tau == 1.0:
        return P
    p = np.clip(P, 1e-9, 1 - 1e-9)
    return expit(tau * np.log(p / (1 - p)))
