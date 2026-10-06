#!/usr/bin/env python3
"""Complete standalone code for manuscript Section 9.4, Table 2."""

import csv
import hashlib
import math

import numpy as np
from numpy.random import Generator, PCG64
from scipy.special import ndtr

ALPHA, SEED, M, REPLICATES = 0.05, 20260913, 5000, 500
BLOCK_SIZES = (16, 64, 256)
SCENARIOS = (
    ("global_null_independent", "Global null, independent", 0.00, 0.0, 0.0),
    ("global_null_rho_05", "Global null, rho=0.5", 0.00, 0.0, 0.5),
    ("sparse_independent", "Sparse 1%, independent", 0.01, 3.5, 0.0),
    ("sparse_rho_05", "Sparse 1%, rho=0.5", 0.01, 3.5, 0.5),
    ("moderate_independent", "Moderate 10%, independent", 0.10, 3.5, 0.0),
    ("dense_independent", "Dense 50%, independent", 0.50, 4.0, 0.0),
)


def make_rng(*parts: object) -> Generator:
    text = "|".join([str(SEED), *(str(x) for x in parts)])
    digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
    return Generator(PCG64(np.random.SeedSequence(np.frombuffer(digest, dtype=np.uint32))))


def simulate(fraction: float, mu: float, rho: float, rng: Generator):
    means = np.zeros(M)
    means[: int(round(M * fraction))] = mu
    common = rng.standard_normal() if rho else 0.0
    z = means + math.sqrt(rho) * common
    z += math.sqrt(1.0 - rho) * rng.standard_normal(M)
    return np.sort(ndtr(-z), kind="stable")


class Counter:
    def __init__(self, p_sorted):
        self.p = p_sorted
        self.cache = {}

    def passes(self, value_rank: int, threshold_rank: int) -> bool:
        key = (value_rank, threshold_rank)
        if key not in self.cache:
            self.cache[key] = bool(
                self.p[value_rank] <= ALPHA / (self.p.size - threshold_rank)
            )
        return self.cache[key]

    @property
    def comparisons(self):
        return len(self.cache)


def holm(p_sorted):
    counter = Counter(p_sorted)
    for r in range(p_sorted.size):
        if not counter.passes(r, r):
            return r, counter.comparisons
    return p_sorted.size, counter.comparisons


def flat_bash(p_sorted, block_size: int):
    counter, m, start = Counter(p_sorted), p_sorted.size, 0
    while start < m:
        end = min(start + block_size - 1, m - 1)
        if not counter.passes(start, start):
            return start, counter.comparisons
        if end == start:
            start += 1
            continue
        if counter.passes(end, start):
            start = end + 1
            continue
        for rank in range(start + 1, end + 1):
            if not counter.passes(rank, rank):
                return rank, counter.comparisons
        start = end + 1
    return m, counter.comparisons


def recursive_certify(counter: Counter, start: int, end: int):
    if counter.passes(end, start):
        return None
    if start == end:
        return start
    midpoint = (start + end) // 2
    failure = recursive_certify(counter, start, midpoint)
    return failure if failure is not None else recursive_certify(counter, midpoint + 1, end)


def naive_recursion(p_sorted):
    counter = Counter(p_sorted)
    failure = recursive_certify(counter, 0, p_sorted.size - 1)
    return (p_sorted.size if failure is None else failure), counter.comparisons


def galloping_bash(p_sorted):
    counter, left, step, m = Counter(p_sorted), 0, 1, p_sorted.size
    while left < m:
        end = min(left + step, m) - 1
        failure = recursive_certify(counter, left, end)
        if failure is not None:
            return failure, counter.comparisons
        left, step = end + 1, step * 2
    return m, counter.comparisons


def main():
    raw, summary = [], []
    for key, label, fraction, mu, rho in SCENARIOS:
        rng = make_rng("comparisons", M, key)
        group = []
        for replicate in range(1, REPLICATES + 1):
            p_sorted = simulate(fraction, mu, rho, rng)
            holm_r, holm_c = holm(p_sorted)
            row = {
                "scenario": key,
                "replicate": replicate,
                "holm_comparisons": holm_c,
            }
            mismatch = False
            for block_size in BLOCK_SIZES:
                rejected, comparisons = flat_bash(p_sorted, block_size)
                row[f"bash_k{block_size}_comparisons"] = comparisons
                mismatch |= rejected != holm_r
            recursive_r, recursive_c = naive_recursion(p_sorted)
            galloping_r, galloping_c = galloping_bash(p_sorted)
            row["naive_recursive_comparisons"] = recursive_c
            row["galloping_comparisons"] = galloping_c
            mismatch |= recursive_r != holm_r or galloping_r != holm_r
            row["mismatch"] = int(mismatch)
            raw.append(row)
            group.append(row)

        result = {
            "scenario": key,
            "scenario_label": label,
            "replicates": REPLICATES,
            "holm_mean_comparisons": np.mean([x["holm_comparisons"] for x in group]),
        }
        for block_size in BLOCK_SIZES:
            result[f"bash_k{block_size}_mean_comparisons"] = np.mean(
                [x[f"bash_k{block_size}_comparisons"] for x in group]
            )
        result["naive_recursive_mean_comparisons"] = np.mean(
            [x["naive_recursive_comparisons"] for x in group]
        )
        result["galloping_mean_comparisons"] = np.mean(
            [x["galloping_comparisons"] for x in group]
        )
        result["mismatches"] = sum(x["mismatch"] for x in group)
        summary.append(result)

    for filename, rows in (("Table2_raw.csv", raw), ("Table2_results.csv", summary)):
        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nTABLE 2: MEAN POST-SORTING COMPARISONS")
    print(f"m={M}, replicates={REPLICATES}, alpha={ALPHA}\n")
    print(f"{'Scenario':32s} {'Holm':>8s} {'k=16':>8s} {'k=64':>8s} "
          f"{'k=256':>8s} {'Naive':>8s} {'Gallop':>8s}")
    for row in summary:
        print(
            f'{row["scenario_label"]:32s} {row["holm_mean_comparisons"]:8.2f} '
            f'{row["bash_k16_mean_comparisons"]:8.2f} '
            f'{row["bash_k64_mean_comparisons"]:8.2f} '
            f'{row["bash_k256_mean_comparisons"]:8.2f} '
            f'{row["naive_recursive_mean_comparisons"]:8.2f} '
            f'{row["galloping_mean_comparisons"]:8.2f}'
        )
    print(f"\nTotal mismatches: {sum(x['mismatches'] for x in summary)}")
    print("Saved Table2_results.csv and Table2_raw.csv")


if __name__ == "__main__":
    main()

