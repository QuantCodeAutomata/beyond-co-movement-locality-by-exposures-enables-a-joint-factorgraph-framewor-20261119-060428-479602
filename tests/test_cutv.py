"""Unit tests for Adaptive CutV spectral partitioning and Contagion Cut
(src/cutv.py) on synthetic block-structured graphs."""

from __future__ import annotations

import numpy as np
import pytest

from src.cutv import adaptive_cutv, contagion_cut_weights, cutv_equal_weights


def _block_graph(n_blocks=4, block_size=5, intra=0.9, inter=0.01, seed=0):
    rng = np.random.default_rng(seed)
    p = n_blocks * block_size
    W = np.full((p, p), inter)
    for b in range(n_blocks):
        sl = slice(b * block_size, (b + 1) * block_size)
        W[sl, sl] = intra
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    W += 1e-3 * np.abs(rng.standard_normal((p, p)))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def test_adaptive_cutv_recovers_block_structure():
    W = _block_graph(n_blocks=4, block_size=5)
    labels, selected = adaptive_cutv(W, n_clusters=4, is_graph=True)
    assert labels.shape == (20,)
    assert len(np.unique(labels)) <= 4
    # Each true block of 5 assets should mostly fall in a single cluster.
    for b in range(4):
        block_labels = labels[b * 5:(b + 1) * 5]
        values, counts = np.unique(block_labels, return_counts=True)
        assert counts.max() >= 4
    assert all(idx >= 1 for idx in selected)


def test_adaptive_cutv_n_clusters_upper_bound():
    W = _block_graph(n_blocks=2, block_size=3)
    labels, _ = adaptive_cutv(W, n_clusters=24, is_graph=True)
    assert len(np.unique(labels)) <= 6  # cannot exceed the number of assets


def test_cutv_equal_weights_sum_to_one_and_nonnegative():
    labels = np.array([0, 0, 1, 1, 1, 2])
    weights = cutv_equal_weights(labels)
    assert weights.shape == (6,)
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_contagion_cut_weights_favour_peripheral_assets():
    labels = np.array([0, 0, 0])
    W = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.1], [0.1, 0.1, 0.0]])
    weights = contagion_cut_weights(labels, W)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    # Asset 2 has the lowest intra-cluster degree (0.2) and should get the
    # largest weight (Eq. 9: p_i proportional to 1 / sum_j w_ij).
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]
