#!/usr/bin/env python3
"""Complete standalone code for manuscript Section 9.3, Table 1."""

import csv
import hashlib
import math

import numpy as np
from numpy.random import Generator, PCG64
from scipy.special import ndtr

ALPHA = 0.05
SEED = 20260913
M = 5000
REPLICATES = 1000
REPAIR_BLOCK_SIZE = 64

SCENARIOS = (
    ("global_null_independent", "Global null, independent", 0.00, 0.0, 0.0),
    ("global_null_rho_05", "Global null, rho=0.5", 0.00, 0.0, 0.5),
    ("sparse_independent", "Sparse 1%, independent", 0.01, 3.5, 0.0),
    ("sparse_rho_05", "Sparse 1%, rho=0.5", 0.01, 3.5, 0.5),
    ("moderate_independent", "Moderate 10%, independent", 0.10, 3.5, 0.0),
    ("dense_independent", "Dense 50%, independent", 0.50, 4.0, 0.0),
)


def make_rng(*parts: object) -> Generator:
    material = "|".join([str(SEED), *(str(x) for x in parts)])
    digest = hashlib.blake2b(material.encode(), digest_size=16).digest()
    words = np.frombuffer(digest, dtype=np.uint32)
    return Generator(PCG64(np.random.SeedSequence(words)))


def simulate_pvalues(signal_fraction: float, mu: float, rho: float, rng: Generator):
    n_false = int(round(M * signal_fraction))
    false_null = np.zeros(M, dtype=bool)
    false_null[:n_false] = True
    means = np.zeros(M)
    means[:n_false] = mu
    common = rng.standard_normal() if rho else 0.0
    z = means + math.sqrt(rho) * common
    z += math.sqrt(1.0 - rho) * rng.standard_normal(M)
    return ndtr(-z), false_null


def sort_pvalues(pvalues: np.ndarray):
    indices = np.arange(pvalues.size)
    order = np.lexsort((indices, pvalues))
    return pvalues[order], order


def holm_rejections(p_sorted: np.ndarray) -> int:
    for r, value in enumerate(p_sorted):
        if value > ALPHA / (p_sorted.size - r):
            return r
    return p_sorted.size


def conservative_repair_rejections(p_sorted: np.ndarray, n_blocks: int) -> int:
    for r, value in enumerate(p_sorted):
        if value > ALPHA / (n_blocks * (p_sorted.size - r)):
            return r
    return p_sorted.size


def decision_metrics(order: np.ndarray, rejected: int, false_null: np.ndarray):
    rejected_flags = false_null[order[:rejected]]
    fwer = int(np.any(~rejected_flags))
    n_false = int(false_null.sum())
    power = rejected_flags.sum() / n_false if n_false else math.nan
    return fwer, power


def mean(values):
    values = np.asarray(list(values), dtype=float)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else math.nan


def mcse_binary(estimate: float) -> float:
    return math.sqrt(estimate * (1.0 - estimate) / REPLICATES)


def zero_event_upper_95() -> float:
    return 1.0 - 0.05 ** (1.0 / REPLICATES)


def main():
    n_blocks = math.ceil(M / REPAIR_BLOCK_SIZE)
    raw_rows = []
    summary_rows = []

    for key, label, fraction, mu, rho in SCENARIOS:
        rng = make_rng("inferential", M, key)
        group = []
        for replicate in range(1, REPLICATES + 1):
            pvalues, false_null = simulate_pvalues(fraction, mu, rho, rng)
            p_sorted, order = sort_pvalues(pvalues)
            exact_r = holm_rejections(p_sorted)
            repair_r = conservative_repair_rejections(p_sorted, n_blocks)
            exact_fwer, exact_power = decision_metrics(order, exact_r, false_null)
            repair_fwer, repair_power = decision_metrics(order, repair_r, false_null)
            row = {
                "scenario": label,
                "replicate": replicate,
                "exact_rejections": exact_r,
                "exact_fwer": exact_fwer,
                "exact_power": exact_power,
                "repaired_rejections": repair_r,
                "repaired_fwer": repair_fwer,
                "repaired_power": repair_power,
            }
            raw_rows.append(row)
            group.append(row)

        exact_fwer = mean(x["exact_fwer"] for x in group)
        repair_fwer = mean(x["repaired_fwer"] for x in group)
        summary_rows.append(
            {
                "scenario": label,
                "exact_fwer": exact_fwer,
                "exact_fwer_mcse": mcse_binary(exact_fwer),
                "exact_power": mean(x["exact_power"] for x in group),
                "repaired_fwer": repair_fwer,
                "repaired_fwer_mcse": mcse_binary(repair_fwer),
                "repaired_zero_event_upper_95": (
                    zero_event_upper_95() if repair_fwer == 0.0 else math.nan
                ),
                "repaired_power": mean(x["repaired_power"] for x in group),
            }
        )

    for filename, rows in (("Table1_raw.csv", raw_rows), ("Table1_results.csv", summary_rows)):
        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nTABLE 1: INFERENTIAL PERFORMANCE")
    print(f"m={M}, replicates={REPLICATES}, alpha={ALPHA}, n_blocks={n_blocks}\n")
    header = (
        f"{'Scenario':32s} {'Exact FWER':>11s} {'Power':>8s} "
        f"{'Repair FWER':>12s} {'Power':>8s}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        exact_power = "--" if math.isnan(row["exact_power"]) else f'{row["exact_power"]:.3f}'
        repair_power = (
            "--" if math.isnan(row["repaired_power"]) else f'{row["repaired_power"]:.3f}'
        )
        print(
            f'{row["scenario"]:32s} {row["exact_fwer"]:.3f} '
            f'({row["exact_fwer_mcse"]:.4f}) {exact_power:>8s} '
            f'{row["repaired_fwer"]:.3f} ({row["repaired_fwer_mcse"]:.4f}) '
            f'{repair_power:>8s}'
        )
    print("\nSaved Table1_results.csv and Table1_raw.csv")


if __name__ == "__main__":
    main()
