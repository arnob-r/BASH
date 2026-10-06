#!/usr/bin/env python3
"""Complete standalone code for manuscript Section 9.6, Table 4."""

import csv
import hashlib
import math

import numpy as np
from numpy.random import Generator, PCG64
from scipy.special import ndtr

ALPHA, SEED, BLOCK_SIZE = 0.05, 20260913, 64
REPLICATES_BY_M = {1000: 20, 10000: 20, 100000: 15, 1000000: 10}
SCENARIOS = (
    ("sparse_independent", "sparse", 0.01, 3.5),
    ("moderate_independent", "moderate", 0.10, 3.5),
    ("dense_independent", "dense", 0.50, 4.0),
)


def make_rng(*parts: object) -> Generator:
    text = "|".join([str(SEED), *(str(x) for x in parts)])
    digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
    return Generator(PCG64(np.random.SeedSequence(np.frombuffer(digest, dtype=np.uint32))))


def simulate(m: int, fraction: float, mu: float, rng: Generator):
    means = np.zeros(m)
    means[: int(round(m * fraction))] = mu
    return np.sort(ndtr(-(means + rng.standard_normal(m))), kind="stable")


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


def standard_error(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / math.sqrt(values.size))


def main():
    raw, summary = [], []
    for m, replicates in REPLICATES_BY_M.items():
        for key, label, fraction, mu in SCENARIOS:
            rng, group = make_rng("scalability", m, key), []
            for replicate in range(1, replicates + 1):
                p_sorted = simulate(m, fraction, mu, rng)
                holm_r, holm_c = holm(p_sorted)
                bash_r, bash_c = flat_bash(p_sorted)
                gallop_r, gallop_c = galloping_bash(p_sorted)
                row = {
                    "m": m,
                    "scenario": key,
                    "replicate": replicate,
                    "holm_comparisons": holm_c,
                    "bash_comparisons": bash_c,
                    "galloping_comparisons": gallop_c,
                    "mismatch": int(bash_r != holm_r or gallop_r != holm_r),
                }
                raw.append(row)
                group.append(row)

            holm_values = [x["holm_comparisons"] for x in group]
            bash_values = [x["bash_comparisons"] for x in group]
            gallop_values = [x["galloping_comparisons"] for x in group]
            holm_mean, bash_mean, gallop_mean = map(
                np.mean, (holm_values, bash_values, gallop_values)
            )
            summary.append(
                {
                    "m": m,
                    "scenario": key,
                    "regime": label,
                    "replicates": replicates,
                    "holm_mean_comparisons": holm_mean,
                    "holm_mcse": standard_error(holm_values),
                    "bash_mean_comparisons": bash_mean,
                    "bash_mcse": standard_error(bash_values),
                    "bash_to_holm_ratio": bash_mean / holm_mean,
                    "galloping_mean_comparisons": gallop_mean,
                    "galloping_mcse": standard_error(gallop_values),
                    "galloping_to_holm_ratio": gallop_mean / holm_mean,
                    "mismatches": sum(x["mismatch"] for x in group),
                }
            )

    for filename, rows in (("Table4_raw.csv", raw), ("Table4_results.csv", summary)):
        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nTABLE 4: SCALABILITY")
    print(f"{'m':>8s} {'Regime':>10s} {'Holm':>10s} {'BASH':>10s} {'Ratio':>8s} "
          f"{'Gallop':>10s} {'Ratio':>8s}")
    for row in summary:
        print(
            f'{row["m"]:8d} {row["regime"]:>10s} '
            f'{row["holm_mean_comparisons"]:10.2f} '
            f'{row["bash_mean_comparisons"]:10.2f} {row["bash_to_holm_ratio"]:8.3f} '
            f'{row["galloping_mean_comparisons"]:10.2f} '
            f'{row["galloping_to_holm_ratio"]:8.3f}'
        )
    print(f"\nTotal mismatches: {sum(x['mismatches'] for x in summary)}")
    print("Saved Table4_results.csv and Table4_raw.csv")


if __name__ == "__main__":
    main()

