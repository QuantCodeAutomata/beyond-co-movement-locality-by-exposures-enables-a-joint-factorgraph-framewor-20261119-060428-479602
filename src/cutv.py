"""Adaptive CutV spectral partitioning and the Contagion Cut allocation rule.

The paper attributes the recursive-bipartitioning spectral-cut algorithm with
an "information-theoretic criterion" for eigenvector selection to reference
[11] (Dionysiou et al., "Adaptive Spectral Graph Partitioning for Portfolio
Optimisation", ICASSP 2026), which is not reproduced in this paper and whose
full selection rule is not available here. This module implements a custom
approximation (Context7 found no portfolio-clustering library offering this
exact recursive-eigenvector-selection spectral cut; networkx/scipy provide
only generic spectral clustering, not this selection rule): at each recursive
split, it builds the graph Laplacian, examines the smallest
``config.CUTV_CANDIDATE_EIGENVECTORS`` non-trivial eigenvectors, and selects
the one whose sign-based bipartition minimises the normalised cut
(conductance) -- an information-theoretic-style criterion that avoids relying
on the Fiedler vector alone when nearby eigengaps make it unreliable (Sec.
2.1, citing Davis-Kahan sensitivity, ref. [7]). This is a custom
implementation of the paper's described behaviour, not of the unavailable
original algorithm.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from src import config


def _normalised_cut(adjacency: np.ndarray, labels: np.ndarray) -> float:
    """Normalised cut value for a binary partition of a weighted graph."""
    mask0 = labels == 0
    mask1 = ~mask0
    if mask0.sum() == 0 or mask1.sum() == 0:
        return np.inf
    cut = adjacency[np.ix_(mask0, mask1)].sum()
    assoc0 = adjacency[mask0, :].sum()
    assoc1 = adjacency[mask1, :].sum()
    if assoc0 == 0 or assoc1 == 0:
        return np.inf
    return cut / assoc0 + cut / assoc1


def _best_bipartition(adjacency: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, int]:
    """Split the subgraph on ``indices`` using the eigenvector (among the
    smallest ``CUTV_CANDIDATE_EIGENVECTORS`` non-trivial ones) whose sign
    pattern minimises the normalised cut; returns (labels over ``indices``,
    selected eigenvalue index, 1-based, Fiedler = 1)."""
    sub = adjacency[np.ix_(indices, indices)]
    degree = np.diag(sub.sum(axis=1))
    laplacian = degree - sub
    n = len(indices)
    n_candidates = min(config.CUTV_CANDIDATE_EIGENVECTORS + 1, n)
    eigenvalues, eigenvectors = eigh(laplacian, subset_by_index=[0, n_candidates - 1])
    best_cut, best_labels, best_index = np.inf, None, 1
    for eig_idx in range(1, n_candidates):  # skip the trivial constant eigenvector (index 0)
        vec = eigenvectors[:, eig_idx]
        labels = (vec >= np.median(vec)).astype(int)
        cut_value = _normalised_cut(sub, labels)
        if cut_value < best_cut:
            best_cut, best_labels, best_index = cut_value, labels, eig_idx
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
    return best_labels, best_index


def adaptive_cutv(
    Sigma_or_W: np.ndarray, n_clusters: int = config.N_CLUSTERS, is_graph: bool = False
) -> tuple[np.ndarray, list[int]]:
    """Recursively bipartition the assets into ``n_clusters`` leaf clusters.

    ``Sigma_or_W`` is either a covariance matrix (``is_graph=False``, edge
    weights are |Sigma_ij|, i != j) or an already-built graph weight matrix
    ``W`` (``is_graph=True``). Returns the leaf-cluster label per asset and
    the list of selected eigenvalue indices (1 = Fiedler vector) used at
    every recursive cut, needed for the Sec. 5.1 Fiedler-selection-frequency
    analysis.
    """
    p = Sigma_or_W.shape[0]
    if is_graph:
        adjacency = Sigma_or_W.copy()
    else:
        adjacency = np.abs(Sigma_or_W.copy())
    np.fill_diagonal(adjacency, 0.0)

    clusters = [np.arange(p)]
    selected_indices: list[int] = []
    while len(clusters) < n_clusters:
        sizes = [len(c) for c in clusters]
        split_pos = int(np.argmax(sizes))
        if sizes[split_pos] < 2:
            break
        target = clusters.pop(split_pos)
        labels, eig_idx = _best_bipartition(adjacency, target)
        selected_indices.append(eig_idx)
        clusters.append(target[labels == 0])
        clusters.append(target[labels == 1])
        clusters = [c for c in clusters if len(c) > 0]

    assignment = np.zeros(p, dtype=int)
    for cluster_id, members in enumerate(clusters):
        assignment[members] = cluster_id
    return assignment, selected_indices


def cutv_equal_weights(labels: np.ndarray) -> np.ndarray:
    """Equal weight across clusters, equal weight within each cluster."""
    weights = np.zeros(len(labels), dtype=float)
    unique_labels = np.unique(labels)
    per_cluster_weight = 1.0 / len(unique_labels)
    for label in unique_labels:
        members = labels == label
        weights[members] = per_cluster_weight / members.sum()
    return weights


def contagion_cut_weights(labels: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Contagion Cut (Eq. 9): within each cluster C, weight asset i by
    p_i ~ 1/(sum_{j in C} w_ij), favouring peripheral (low-degree) assets;
    clusters themselves are weighted equally."""
    weights = np.zeros(len(labels), dtype=float)
    unique_labels = np.unique(labels)
    per_cluster_weight = 1.0 / len(unique_labels)
    for label in unique_labels:
        members = np.where(labels == label)[0]
        intra_degree = W[np.ix_(members, members)].sum(axis=1)
        inv_degree = 1.0 / np.maximum(intra_degree, 1e-10)
        normalised = inv_degree / inv_degree.sum()
        weights[members] = per_cluster_weight * normalised
    return weights
