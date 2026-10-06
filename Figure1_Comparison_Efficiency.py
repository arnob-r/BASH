#!/usr/bin/env python3
"""
Figure 1: Comparison efficiency relative to sequential Holm.

This script is self-contained. It implements the algorithms used in the
manuscript and reproduces the Section 9 decision-stage comparison experiment:

  * Holm
  * Flat BASH with k = 16, 64, 256
  * Naive top-down recursive certification
  * Galloping BASH

The comparison-count convention matches the manuscript: one comparison is one
distinct inequality between an ordered p-value and a critical value. Repeated
evaluation of an identical inequality is cached and is not counted twice.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from numpy.random import Generator, PCG64
from scipy.special import ndtr

# ---------------------------------------------------------------------
# Experiment settings: manuscript Section 9
# ---------------------------------------------------------------------
ALPHA = 0.05
SEED = 20260913
M = 5000
REPLICATES = 500
BLOCK_SIZES = (16, 64, 256)

OUTPUT_PDF = "Figure1_Comparison_Efficiency.pdf"
OUTPUT_PNG = "Figure1_Comparison_Efficiency.png"
OUTPUT_CSV = "Figure1_Comparison_Efficiency_results.csv"

SCENARIOS = (
    ("global_null_independent", "Null, independent", 0.00, 0.0, 0.0),
    ("global_null_rho_05", r"Null, $\rho=0.5$", 0.00, 0.0, 0.5),
    ("sparse_independent", "Sparse, independent", 0.01, 3.5, 0.0),
    ("sparse_rho_05", r"Sparse, $\rho=0.5$", 0.01, 3.5, 0.5),
    ("moderate_independent", "Moderate, independent", 0.10, 3.5, 0.0),
    ("dense_independent", "Dense, independent", 0.50, 4.0, 0.0),
)

METHOD_COLUMNS = (
    ("BASH\n$k=16$", "bash_k16_mean"),
    ("BASH\n$k=64$", "bash_k64_mean"),
    ("BASH\n$k=256$", "bash_k256_mean"),
    ("Naive\nrecursion", "naive_mean"),
    ("Galloping\nBASH", "galloping_mean"),
)


# ---------------------------------------------------------------------
# Reproducible simulation
# ---------------------------------------------------------------------
def make_rng(*parts: object) -> Generator:
    material = "|".join([str(SEED), *(str(x) for x in parts)])
    digest = hashlib.blake2b(material.encode(), digest_size=16).digest()
    words = np.frombuffer(digest, dtype=np.uint32)
    return Generator(PCG64(np.random.SeedSequence(words)))


def simulate_sorted_pvalues(
    signal_fraction: float,
    mu: float,
    rho: float,
    rng: Generator,
) -> np.ndarray:
    means = np.zeros(M)
    means[: int(round(M * signal_fraction))] = mu

    common = rng.standard_normal() if rho else 0.0
    z = means + math.sqrt(rho) * common
    z += math.sqrt(1.0 - rho) * rng.standard_normal(M)

    # one-sided p_i = 1 - Phi(Z_i) = Phi(-Z_i)
    return np.sort(ndtr(-z), kind="stable")


# ---------------------------------------------------------------------
# Comparison counter and algorithms
# ---------------------------------------------------------------------
class Counter:
    """
    Counts distinct threshold inequalities.

    Indexing is zero-based:
      value_rank = r-1 and threshold_rank = s-1
    corresponds to p_(r) <= alpha/(m-s+1).
    """

    def __init__(self, p_sorted: np.ndarray):
        self.p = p_sorted
        self.cache: dict[tuple[int, int], bool] = {}

    def passes(self, value_rank: int, threshold_rank: int) -> bool:
        key = (value_rank, threshold_rank)
        if key not in self.cache:
            threshold = ALPHA / (self.p.size - threshold_rank)
            self.cache[key] = bool(self.p[value_rank] <= threshold)
        return self.cache[key]

    @property
    def comparisons(self) -> int:
        return len(self.cache)


def holm(p_sorted: np.ndarray) -> tuple[int, int]:
    counter = Counter(p_sorted)
    for r in range(p_sorted.size):
        if not counter.passes(r, r):
            return r, counter.comparisons
    return p_sorted.size, counter.comparisons


def flat_bash(p_sorted: np.ndarray, block_size: int) -> tuple[int, int]:
    counter = Counter(p_sorted)
    m = p_sorted.size
    start = 0

    while start < m:
        end = min(start + block_size - 1, m - 1)

        # Stop certificate: p_(start) > c_start
        if not counter.passes(start, start):
            return start, counter.comparisons

        # Singleton block
        if end == start:
            start += 1
            continue

        # Pass certificate: p_(end) <= c_start
        if counter.passes(end, start):
            start = end + 1
            continue

        # Ambiguous block: rankwise Holm fallback for remaining ranks
        for rank in range(start + 1, end + 1):
            if not counter.passes(rank, rank):
                return rank, counter.comparisons

        start = end + 1

    return m, counter.comparisons


def recursive_certify(counter: Counter, start: int, end: int) -> int | None:
    """
    Exact recursive certificate from Definition 1 / Theorem 3.

    Returns None if all ranks in [start,end] are certified to pass;
    otherwise returns the smallest failing zero-based rank.
    """
    if counter.passes(end, start):
        return None

    if start == end:
        return start

    midpoint = (start + end) // 2
    failure = recursive_certify(counter, start, midpoint)
    if failure is not None:
        return failure
    return recursive_certify(counter, midpoint + 1, end)


def naive_recursion(p_sorted: np.ndarray) -> tuple[int, int]:
    counter = Counter(p_sorted)
    failure = recursive_certify(counter, 0, p_sorted.size - 1)
    rejected = p_sorted.size if failure is None else failure
    return rejected, counter.comparisons


def galloping_bash(p_sorted: np.ndarray) -> tuple[int, int]:
    counter = Counter(p_sorted)
    left = 0
    step = 1
    m = p_sorted.size

    while left < m:
        end = min(left + step, m) - 1
        failure = recursive_certify(counter, left, end)

        if failure is not None:
            return failure, counter.comparisons

        left = end + 1
        step *= 2

    return m, counter.comparisons


# ---------------------------------------------------------------------
# Experiment and plotting
# ---------------------------------------------------------------------
def main() -> None:
    summary_rows = []

    for key, display_label, fraction, mu, rho in SCENARIOS:
        rng = make_rng("comparisons", M, key)

        holm_counts = []
        bash_counts = {k: [] for k in BLOCK_SIZES}
        naive_counts = []
        galloping_counts = []
        mismatches = 0

        for _ in range(REPLICATES):
            p_sorted = simulate_sorted_pvalues(fraction, mu, rho, rng)

            holm_r, holm_c = holm(p_sorted)
            holm_counts.append(holm_c)

            for k in BLOCK_SIZES:
                r, c = flat_bash(p_sorted, k)
                bash_counts[k].append(c)
                mismatches += int(r != holm_r)

            naive_r, naive_c = naive_recursion(p_sorted)
            gallop_r, gallop_c = galloping_bash(p_sorted)

            naive_counts.append(naive_c)
            galloping_counts.append(gallop_c)
            mismatches += int(naive_r != holm_r)
            mismatches += int(gallop_r != holm_r)

        if mismatches != 0:
            raise AssertionError(
                f"{key}: {mismatches} decision mismatches found; "
                "certificate implementation is inconsistent with Holm."
            )

        row = {
            "scenario": key,
            "scenario_label": display_label,
            "replicates": REPLICATES,
            "holm_mean": float(np.mean(holm_counts)),
            "bash_k16_mean": float(np.mean(bash_counts[16])),
            "bash_k64_mean": float(np.mean(bash_counts[64])),
            "bash_k256_mean": float(np.mean(bash_counts[256])),
            "naive_mean": float(np.mean(naive_counts)),
            "galloping_mean": float(np.mean(galloping_counts)),
            "mismatches": mismatches,
        }
        summary_rows.append(row)

    # Save the numerical values used in the figure.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Ratios relative to Holm.
    ratios = np.empty((len(SCENARIOS), len(METHOD_COLUMNS)))
    for i, row in enumerate(summary_rows):
        baseline = row["holm_mean"]
        for j, (_, column) in enumerate(METHOD_COLUMNS):
            ratios[i, j] = row[column] / baseline

    # Use log2 ratios so equal multiplicative deviations from 1 have
    # symmetric visual magnitude.
    log_ratios = np.log2(ratios)
    extent = max(1.0, math.ceil(float(np.max(np.abs(log_ratios)))))
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    figure, axis = plt.subplots(figsize=(7.4, 3.4), constrained_layout=True)
    image = axis.imshow(log_ratios, cmap="RdYlBu_r", norm=norm, aspect="auto")

    axis.set_xticks(
        np.arange(len(METHOD_COLUMNS)),
        [label for label, _ in METHOD_COLUMNS],
    )
    axis.set_yticks(
        np.arange(len(SCENARIOS)),
        [label for _, label, *_ in SCENARIOS],
    )
    axis.tick_params(length=0)
    axis.set_title(r"Comparison cost relative to sequential Holm at $m=5000$")

    for i in range(ratios.shape[0]):
        for j in range(ratios.shape[1]):
            text_color = "white" if abs(log_ratios[i, j]) > 0.62 * extent else "black"
            axis.text(
                j,
                i,
                f"{ratios[i, j]:.2f}x",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    colorbar = figure.colorbar(image, ax=axis, shrink=0.88, pad=0.025)
    colorbar.set_label(r"$\log_2$(method comparisons / Holm comparisons)",fontsize=9)

    axis.text(
        0.0,
        -0.18,
        "Blue: fewer comparisons than Holm; red: more comparisons than Holm.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_CSV}")
    print("All certificate methods matched Holm on every simulated realization.")


if __name__ == "__main__":
    main()
