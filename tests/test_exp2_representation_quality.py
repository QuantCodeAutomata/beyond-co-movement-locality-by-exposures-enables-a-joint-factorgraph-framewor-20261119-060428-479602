"""Lightweight unit tests for exp/exp2_representation_quality.py helpers.

These tests use small synthetic inputs and do not call the network/data
loader; they must not call ``run()`` (which needs live market data).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import normalized_mutual_info_score

from exp.exp2_representation_quality import sector_alignment_ratio


def test_sector_alignment_ratio_above_one_for_block_structured_matrix() -> None:
    # Two sectors of 3 assets each: strong within-sector edges, weak
    # between-sector edges.
    sector_labels = ["tech", "tech", "tech", "energy", "energy", "energy"]
    matrix = np.full((6, 6), 0.01)
    matrix[:3, :3] = 0.9
    matrix[3:, 3:] = 0.9
    np.fill_diagonal(matrix, 0.0)
    ratio = sector_alignment_ratio(matrix, sector_labels)
    assert ratio > 1.0


def test_sector_alignment_ratio_near_one_for_uniform_matrix() -> None:
    sector_labels = ["tech", "tech", "energy", "energy"]
    matrix = np.full((4, 4), 0.5)
    np.fill_diagonal(matrix, 0.0)
    ratio = sector_alignment_ratio(matrix, sector_labels)
    assert ratio == pytest.approx(1.0, abs=1e-8)


def test_sector_alignment_ratio_handles_single_sector() -> None:
    # No between-sector pairs at all: ratio must be well-defined (inf), not
    # raise a division error.
    sector_labels = ["tech", "tech", "tech"]
    matrix = np.full((3, 3), 0.5)
    np.fill_diagonal(matrix, 0.0)
    ratio = sector_alignment_ratio(matrix, sector_labels)
    assert ratio == float("inf")


def test_nmi_identical_labels_is_one() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert normalized_mutual_info_score(labels, labels) == 1.0


def test_nmi_singleton_clusters_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    labels_a = np.arange(10)  # every asset its own cluster
    labels_b = rng.permutation(10)
    nmi = normalized_mutual_info_score(labels_a, labels_b)
    assert 0.0 <= nmi <= 1.0 + 1e-9
