#!/usr/bin/env python3
"""Complete standalone code for manuscript Section 9.5, Table 3."""

import csv
import hashlib
import json
import math
from fractions import Fraction

import numpy as np
from numpy.random import Generator, PCG64
from scipy.special import ndtr

ALPHA, SEED, M, REPLICATES, BLOCK_SIZE = 0.05, 20260913, 4900, 400, 70
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
        self.p, self.cache = p_sorted, {}

    def passes(self, value_rank, threshold_rank):
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
    for rank in range(p_sorted.size):
        if not counter.passes(rank, rank):
            return rank, counter.comparisons
    return p_sorted.size, counter.comparisons


def flat_bash(p_sorted):
    counter, start, m = Counter(p_sorted), 0, p_sorted.size
    while start < m:
        end = min(start + BLOCK_SIZE - 1, m - 1)
        if not counter.passes(start, start):
            return start, counter.comparisons
        if end == start:
            start += 1
        elif counter.passes(end, start):
            start = end + 1
        else:
            for rank in range(start + 1, end + 1):
                if not counter.passes(rank, rank):
                    return rank, counter.comparisons
            start = end + 1
    return m, counter.comparisons


def direct_two_level(p_sorted, block_size=BLOCK_SIZE):
    m, n, comparisons = p_sorted.size, p_sorted.size // block_size, 0
    if m % block_size:
        raise ValueError("block_size must divide m")
    for block in range(n):
        start, end = block * block_size, (block + 1) * block_size - 1
        direct_threshold = ALPHA / (n * (n - block))
        comparisons += 1
        if p_sorted[end] <= direct_threshold:
            continue
        for rank in range(start, end + 1):
            comparisons += 1
            if p_sorted[rank] > ALPHA / (m - rank):
                return rank, comparisons
    return m, comparisons


def recursive_certify(counter, start, end):
    if counter.passes(end, start):
        return None
    if start == end:
        return start
    midpoint = (start + end) // 2
    failure = recursive_certify(counter, start, midpoint)
    return failure if failure is not None else recursive_certify(counter, midpoint + 1, end)


def galloping_bash(p_sorted):
    counter, left, step, m = Counter(p_sorted), 0, 1, p_sorted.size
    while left < m:
        end = min(left + step, m) - 1
        failure = recursive_certify(counter, left, end)
        if failure is not None:
            return failure, counter.comparisons
        left, step = end + 1, step * 2
    return m, counter.comparisons


def boundary_counterexample():
    m, k, n = 16, 8, 2
    holm_threshold = ALPHA / m
    direct_threshold = ALPHA / (n * n)
    midpoint = (holm_threshold + direct_threshold) / 2
    p_sorted = np.r_[np.full(k, midpoint), np.ones(m - k)]
    holm_result = holm(p_sorted)
    direct_result = direct_two_level(p_sorted, k)
    galloping_result = galloping_bash(p_sorted)

    errors = []
    alpha_fraction = Fraction(str(ALPHA))
    for test_m in (16, 36, 64, 100, 4900):
        for test_k in range(1, test_m + 1):
            if test_m % test_k:
                continue
            test_n = test_m // test_k
            for block in range(test_n):
                start = block * test_k
                a_j = alpha_fraction / (test_n * (test_n - block))
                c_start = alpha_fraction / (test_m - start)
                errors.append(abs(a_j - Fraction(test_k, test_n) * c_start))
    return {
        "m": m,
        "k": k,
        "n": n,
        "holm_first_threshold": holm_threshold,
        "direct_first_threshold": direct_threshold,
        "pvalue_midpoint": midpoint,
        "holm_rejections": holm_result[0],
        "direct_two_level_rejections": direct_result[0],
        "galloping_rejections": galloping_result[0],
        "identity_max_absolute_error": float(max(errors)),
    }


def main():
    raw, summary = [], []
    for key, label, fraction, mu, rho in SCENARIOS:
        rng, group = make_rng("direct", M, key), []
        for replicate in range(1, REPLICATES + 1):
            p_sorted = simulate(fraction, mu, rho, rng)
            holm_r, holm_c = holm(p_sorted)
            bash_r, bash_c = flat_bash(p_sorted)
            direct_r, direct_c = direct_two_level(p_sorted)
            gallop_r, gallop_c = galloping_bash(p_sorted)
            row = {
                "scenario": key,
                "replicate": replicate,
                "holm_comparisons": holm_c,
                "bash_comparisons": bash_c,
                "direct_two_level_comparisons": direct_c,
                "galloping_comparisons": gallop_c,
                "mismatch": int(any(x != holm_r for x in (bash_r, direct_r, gallop_r))),
            }
            raw.append(row)
            group.append(row)
        summary.append(
            {
                "scenario": key,
                "scenario_label": label,
                "holm_mean_comparisons": np.mean([x["holm_comparisons"] for x in group]),
                "bash_mean_comparisons": np.mean([x["bash_comparisons"] for x in group]),
                "direct_two_level_mean_comparisons": np.mean(
                    [x["direct_two_level_comparisons"] for x in group]
                ),
                "galloping_mean_comparisons": np.mean(
                    [x["galloping_comparisons"] for x in group]
                ),
                "mismatches": sum(x["mismatch"] for x in group),
            }
        )

    for filename, rows in (("Table3_raw.csv", raw), ("Table3_results.csv", summary)):
        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    boundary = boundary_counterexample()
    with open("Table3_counterexample.json", "w", encoding="utf-8") as stream:
        json.dump(boundary, stream, indent=2)

    print("\nTABLE 3: DIRECT CONSTRUCTION AT THE SQUARE BOUNDARY")
    print(f"m={M}, k={BLOCK_SIZE}, replicates={REPLICATES}\n")
    print(f"{'Scenario':32s} {'Holm':>8s} {'BASH':>8s} {'Direct':>8s} {'Gallop':>8s}")
    for row in summary:
        print(
            f'{row["scenario_label"]:32s} {row["holm_mean_comparisons"]:8.2f} '
            f'{row["bash_mean_comparisons"]:8.2f} '
            f'{row["direct_two_level_mean_comparisons"]:8.2f} '
            f'{row["galloping_mean_comparisons"]:8.2f}'
        )
    print(f"\nTotal mismatches: {sum(x['mismatches'] for x in summary)}")
    print("Boundary counterexample:", boundary)
    print("Saved Table3_results.csv, Table3_raw.csv, and Table3_counterexample.json")


if __name__ == "__main__":
    main()

