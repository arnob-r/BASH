#!/usr/bin/env python3
"""
Figure 2: Scalability of sequential Holm, flat BASH, and Galloping BASH.

This script is self-contained and implements the manuscript algorithms directly.
It reproduces the Section 9 scalability design:

  m = 10^3, 10^4, 10^5, 10^6
  independent Gaussian model
  sparse 1%, moderate 10%, dense 50% signal regimes
  flat BASH block size k = 64
  replicate counts 20, 20, 15, 10 per regime

Error bars are +/- one Monte Carlo standard error of the mean comparison count.
"""

from __future__ import annotations

import csv
import hashlib
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext
from numpy.random import Generator, PCG64
from scipy.special import ndtr

ALPHA = 0.05
SEED = 20260913
BLOCK_SIZE = 64

REPLICATES_BY_M = {
    1_000: 20,
    10_000: 20,
    100_000: 15,
    1_000_000: 10,
}

REGIMES = (
    ("sparse_independent", "Sparse signals (1%)", 0.01, 3.5),
    ("moderate_independent", "Moderate signals (10%)", 0.10, 3.5),
    ("dense_independent", "Dense signals (50%)", 0.50, 4.0),
)

OUTPUT_PDF = "Figure2_Scalability.pdf"
OUTPUT_PNG = "Figure2_Scalability.png"
OUTPUT_CSV = "Figure2_Scalability_results.csv"


def make_rng(*parts: object) -> Generator:
    material = "|".join([str(SEED), *(str(x) for x in parts)])
    digest = hashlib.blake2b(material.encode(), digest_size=16).digest()
    words = np.frombuffer(digest, dtype=np.uint32)
    return Generator(PCG64(np.random.SeedSequence(words)))


def simulate_sorted_pvalues(
    m: int,
    signal_fraction: float,
    mu: float,
    rng: Generator,
) -> np.ndarray:
    means = np.zeros(m)
    means[: int(round(m * signal_fraction))] = mu
    z = means + rng.standard_normal(m)
    return np.sort(ndtr(-z), kind="stable")


class Counter:
    """
    Counts distinct inequalities p_(r) <= alpha/(m-s+1).

    Zero-based threshold_rank=s-1 gives denominator m-threshold_rank.
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


def flat_bash(p_sorted: np.ndarray, block_size: int = BLOCK_SIZE) -> tuple[int, int]:
    counter = Counter(p_sorted)
    m = p_sorted.size
    start = 0

    while start < m:
        end = min(start + block_size - 1, m - 1)

        # First-rank stop check
        if not counter.passes(start, start):
            return start, counter.comparisons

        if end == start:
            start += 1
            continue

        # Endpoint pass certificate
        if counter.passes(end, start):
            start = end + 1
            continue

        # Rankwise Holm fallback within the ambiguous block
        for rank in range(start + 1, end + 1):
            if not counter.passes(rank, rank):
                return rank, counter.comparisons

        start = end + 1

    return m, counter.comparisons


def recursive_certify(counter: Counter, start: int, end: int) -> int | None:
    if counter.passes(end, start):
        return None

    if start == end:
        return start

    midpoint = (start + end) // 2
    failure = recursive_certify(counter, start, midpoint)
    if failure is not None:
        return failure
    return recursive_certify(counter, midpoint + 1, end)


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


def mcse(values: list[int]) -> float:
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return 0.0
    return float(a.std(ddof=1) / math.sqrt(a.size))


def main() -> None:
    results = []

    for m, replicates in REPLICATES_BY_M.items():
        for key, title, fraction, mu in REGIMES:
            rng = make_rng("scalability", m, key)

            holm_counts = []
            bash_counts = []
            gallop_counts = []
            mismatches = 0

            for _ in range(replicates):
                p_sorted = simulate_sorted_pvalues(m, fraction, mu, rng)

                holm_r, holm_c = holm(p_sorted)
                bash_r, bash_c = flat_bash(p_sorted)
                gallop_r, gallop_c = galloping_bash(p_sorted)

                mismatches += int(bash_r != holm_r)
                mismatches += int(gallop_r != holm_r)

                holm_counts.append(holm_c)
                bash_counts.append(bash_c)
                gallop_counts.append(gallop_c)

            if mismatches != 0:
                raise AssertionError(
                    f"{key}, m={m}: {mismatches} decision mismatches found."
                )

            results.append(
                {
                    "m": m,
                    "scenario": key,
                    "scenario_label": title,
                    "replicates": replicates,
                    "holm_mean_comparisons": float(np.mean(holm_counts)),
                    "holm_mcse": mcse(holm_counts),
                    "bash_mean_comparisons": float(np.mean(bash_counts)),
                    "bash_mcse": mcse(bash_counts),
                    "bash_to_holm_ratio": float(np.mean(bash_counts) / np.mean(holm_counts)),
                    "galloping_mean_comparisons": float(np.mean(gallop_counts)),
                    "galloping_mcse": mcse(gallop_counts),
                    "galloping_to_holm_ratio": float(
                        np.mean(gallop_counts) / np.mean(holm_counts)
                    ),
                    "mismatches": mismatches,
                }
            )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )

    # Manuscript figure: 1 x 3 panels, shared logarithmic y-axis.
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.4, 2.8),
        sharey=True,
        constrained_layout=True,
    )

    method_specs = (
        ("Holm", "holm_mean_comparisons", "holm_mcse", "o", "-"),
        ("Flat BASH", "bash_mean_comparisons", "bash_mcse", "s", "--"),
        ("Galloping BASH", "galloping_mean_comparisons", "galloping_mcse", "^", "-."),
    )

    for panel, (key, title, _, _) in enumerate(REGIMES):
        axis = axes[panel]
        selected = sorted(
            (row for row in results if row["scenario"] == key),
            key=lambda row: row["m"],
        )

        x = np.asarray([row["m"] for row in selected], dtype=float)

        for label, mean_col, error_col, marker, linestyle in method_specs:
            y = np.asarray([row[mean_col] for row in selected], dtype=float)
            yerr = np.asarray([row[error_col] for row in selected], dtype=float)

            axis.errorbar(
                x,
                y,
                yerr=yerr,
                label=label,
                marker=marker,
                linestyle=linestyle,
                markerfacecolor="white" if label != "Holm" else None,
                linewidth=1.7,
                capsize=2,
            )

        axis.set_xscale("log", base=10)
        axis.set_yscale("log", base=10)
        axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.grid(True, which="major", linewidth=0.55)
        axis.grid(True, which="minor", linewidth=0.35)
        axis.set_title(title)
        axis.set_xlabel(r"Number of hypotheses, $m$")
        axis.text(
            0.02,
            0.97,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    axes[0].set_ylabel("Mean post-sorting comparisons")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )

    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(figure)

    total_realizations = sum(REPLICATES_BY_M.values()) * len(REGIMES)
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_CSV}")
    print(f"Total simulated realizations: {total_realizations}")
    print("No BASH/Galloping decision mismatches with Holm.")


if __name__ == "__main__":
    main()
