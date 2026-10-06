# BASHâ€“Holm Range Certificates

Reproducible Python implementation of **Block-Accelerated Sequential Holm (BASH)**, recursive range certificates, the completed two-level direct construction, and **Galloping BASH** for exact post-sorting recovery of Holmâ€™s step-down decisions.

This repository accompanies the manuscript:

> **Exact Range Certificates for Holmâ€™s Step-Down Procedure: Fixed Blocks and Adaptive Recursive Search**

The main computational question studied here is:

> Given an already ordered vector of p-values, how many rankwise threshold evaluations are required to recover Holmâ€™s exact stopping rank?

The methods in this repository **do not define a new multiple-testing criterion**. They are computational wrappers around Holmâ€™s step-down procedure. The valid certificate-based methods are designed to recover the **same rejection set as Holm** while potentially reducing the number of post-sorting threshold comparisons.

---

## 1. Overview

Holmâ€™s step-down procedure compares ordered p-values

\[
p_{(1)} \le p_{(2)} \le \cdots \le p_{(m)}
\]

against the critical values

\[
c_r^{\mathrm H} = \frac{\alpha}{m-r+1},
\qquad r=1,\ldots,m.
\]

The central observation behind BASH is that a consecutive interval can sometimes be certified using only endpoint comparisons.

For a block of ranks

\[
B_j=\{s_j,s_j+1,\ldots,e_j\},
\]

if

\[
p_{(e_j)}\le c_{s_j}^{\mathrm H},
\]

then every rank in the block necessarily satisfies its own Holm inequality:

\[
p_{(r)}
\le
p_{(e_j)}
\le
c_{s_j}^{\mathrm H}
\le
c_r^{\mathrm H}.
\]

Thus, the entire block may be certified without evaluating every interior rank individually.

---

## 2. Implemented Methods

### Sequential Holm

The ordinary sequential implementation of Holmâ€™s step-down procedure is used as the baseline.

### Flat BASH

BASH partitions the ordered p-values into blocks of fixed size \(k\). For each visited block, it uses:

1. a **stop certificate** at the first rank,
2. a **pass certificate** at the upper endpoint,
3. rankwise Holm fallback when the block is ambiguous.

The block size \(k\) is a computational tuning parameter and does not alter the rejection set.

### Naive Recursive Certification

A top-down recursive certificate is applied to the full ordered range. Ambiguous intervals are recursively subdivided until a certificate is obtained or singleton ranks are reached.

This method is exact but may incur a logarithmic initialization cost when Holm stops very early.

### Galloping BASH

Galloping BASH uses exponentially growing candidate ranges and recursive refinement only when needed.

It is tuning-free with respect to block size and remains pathwise equivalent to Holm.

### Completed Two-Level Direct Construction

The repository also includes the completed two-level direct construction studied at the square-boundary setting

\[
m=4900,\qquad k=70=\sqrt m.
\]

The code additionally reproduces the explicit counterexample beyond the equivalence boundary.

---

## 3. Repository Contents

The repository is organized around the four numerical tables and two figures in the manuscript.

### Table scripts

#### `Table1_Inferential_Validation_final.py`

Reproduces the inferential validation study at

\[
m=5000,\qquad N_{\mathrm{rep}}=1000,\qquad \alpha=0.05.
\]

Outputs include:

- exact FWER,
- Monte Carlo standard error,
- exact power,
- conservative-repair FWER,
- conservative-repair power,
- one-sided 95% exact binomial upper bound when zero FWER events are observed.

Generated files:

```text
Table1_results.csv
Table1_raw.csv
```

#### `Table2_Comparison_Counts_final.py`

Reproduces the mean post-sorting comparison study at

\[
m=5000,\qquad N_{\mathrm{rep}}=500.
\]

Methods compared:

- Holm,
- BASH with \(k=16\),
- BASH with \(k=64\),
- BASH with \(k=256\),
- naive recursive certification,
- Galloping BASH.

Generated files:

```text
Table2_results.csv
Table2_raw.csv
```

The script also checks decision agreement with Holm.

#### `Table3_Direct_Construction_final.py`

Reproduces the square-boundary direct-construction experiment at

\[
m=4900=70^2,\qquad k=70,\qquad N_{\mathrm{rep}}=400.
\]

Methods compared:

- Holm,
- flat BASH,
- completed two-level direct construction,
- Galloping BASH.

It also reproduces the explicit counterexample at

\[
m=16,\qquad k=8,\qquad n=2,
\]

where the direct construction is outside the equivalence boundary.

Generated files:

```text
Table3_results.csv
Table3_raw.csv
Table3_counterexample.json
```

#### `Table4_Scalability_final.py`

Reproduces the scalability study for

\[
m\in\{10^3,10^4,10^5,10^6\}.
\]

Signal regimes:

- sparse: 1% signals,
- moderate: 10% signals,
- dense: 50% signals.

Methods compared:

- Holm,
- flat BASH with \(k=64\),
- Galloping BASH.

Generated files:

```text
Table4_results.csv
Table4_raw.csv
```

### Figure scripts

#### `Figure1_Comparison_Efficiency.py`

Generates the comparison-efficiency heatmap relative to sequential Holm.

Each cell reports

\[
\frac{\text{mean method comparisons}}{\text{mean Holm comparisons}}.
\]

Values below 1 indicate fewer post-sorting comparisons than Holm, while values above 1 indicate additional comparison cost.

Generated files:

```text
Figure1_Comparison_Efficiency.pdf
Figure1_Comparison_Efficiency.png
Figure1_Comparison_Efficiency_results.csv
```

#### `Figure2_Scalability.py`

Generates the scalability figure for

\[
m=10^3,10^4,10^5,10^6
\]

under sparse, moderate, and dense signal regimes.

The figure compares:

- Holm,
- flat BASH,
- Galloping BASH.

Error bars represent \(\pm 1\) Monte Carlo standard error of the mean comparison count.

Generated files:

```text
Figure2_Scalability.pdf
Figure2_Scalability.png
Figure2_Scalability_results.csv
```

---

## 4. Simulation Model

The numerical studies use a one-sided equicorrelated Gaussian model:

\[
Z_i
=
\mu_i
+
\sqrt{\rho}\,G
+
\sqrt{1-\rho}\,\varepsilon_i,
\]

with

\[
p_i = 1-\Phi(Z_i)=\Phi(-Z_i),
\]

where

\[
G,\varepsilon_1,\ldots,\varepsilon_m
\]

are independent standard normal variables.

True null hypotheses use

\[
\mu_i=0,
\]

while false nulls use \(\mu_i>0\).

The principal scenarios are:

| Scenario | Signal fraction | \(\mu\) | \(\rho\) |
|---|---:|---:|---:|
| Global null, independent | 0% | 0.0 | 0.0 |
| Global null, correlated | 0% | 0.0 | 0.5 |
| Sparse, independent | 1% | 3.5 | 0.0 |
| Sparse, correlated | 1% | 3.5 | 0.5 |
| Moderate, independent | 10% | 3.5 | 0.0 |
| Dense, independent | 50% | 4.0 | 0.0 |

All reported simulations use

\[
\alpha=0.05.
\]

---

## 5. Reproducibility

The simulation code is deterministic.

A common master seed is used:

```text
20260913
```

Each experiment cell derives its own reproducible random-number stream using:

- NumPy `PCG64`,
- `SeedSequence`,
- a deterministic `BLAKE2b` hash of the experiment identifiers.

This design makes each simulation cell reproducible independently of execution order.

---

## 6. Comparison-Count Convention

The repository studies the **post-sorting decision stage**.

One comparison is defined as one distinct inequality evaluation between an ordered p-value and a critical value.

For example,

\[
p_{(r)}\le c_r^{\mathrm H}
\]

is one Holm comparison, while

\[
p_{(e)}\le c_s^{\mathrm H}
\]

is one certificate comparison.

Repeated evaluation of the same ordered-p-value / threshold pair is cached and is not counted twice.

The comparison count excludes:

- sorting,
- loop control,
- index arithmetic,
- array allocation,
- data-structure construction,
- general Python overhead.

Therefore, the reported quantity is a hardware-independent algorithmic measure of the post-sorting decision stage rather than end-to-end wall-clock runtime.

---

## 7. Requirements

Python 3.10 or later is recommended.

Required packages:

```text
numpy
scipy
matplotlib
```

Install them with:

```bash
pip install numpy scipy matplotlib
```

---

## 8. Running the Code

Clone the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd bash-holm-range-certificates
```

Run any table script directly:

```bash
python Table1_Inferential_Validation_final.py
python Table2_Comparison_Counts_final.py
python Table3_Direct_Construction_final.py
python Table4_Scalability_final.py
```

Generate the figures:

```bash
python Figure1_Comparison_Efficiency.py
python Figure2_Scalability.py
```

The scripts write their output files to the current working directory.

---

## 9. Expected Numerical Checks

The scripts include internal agreement checks where appropriate.

For the valid certificate procedures, the rejection count should agree with sequential Holm on every simulated realization.

Representative manuscript results include:

### Dense-signal comparison experiment

At \(m=5000\), the mean post-sorting comparison counts are approximately:

```text
Holm                 1038.92
BASH, k = 64           69.05
Naive recursion        19.67
Galloping BASH         26.26
```

### Dense-signal scalability experiment

At \(m=10^6\), the mean comparison counts are approximately:

```text
Holm                46885.90
Flat BASH            1496.70
Galloping BASH         40.40
```

These values are intended as reproducibility checks rather than hard-coded outputs.

---

## 10. Statistical Interpretation

The certificate algorithms should not be interpreted as alternative multiple-testing procedures.

For the valid implementations,

\[
\text{certificate output}
=
\text{Holm rejection set}
\]

pathwise, under the common deterministic ordering convention.

Hence the inferential guarantee is inherited from Holm.

The computational objective is only to alter the route by which the stopping rank is recovered.

---

## 11. Scope and Limitations

The repository focuses on comparison complexity after the p-values have already been ordered.

Important limitations include:

1. sorting cost is not included;
2. fewer threshold comparisons do not automatically imply shorter wall-clock time;
3. the access-count interpretation is most relevant when ordered ranks can be queried directly;
4. the numerical experiments use the Gaussian models described in the manuscript;
5. the reported methods are not claimed to be comparison-optimal;
6. the direct-construction equivalence boundary applies to that specific block-count-indexed construction, not to BASH in general.

---

## 12. Suggested Repository Structure

```text
bash-holm-range-certificates/
â”‚
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”‚
â”œâ”€â”€ Table1_Inferential_Validation_final.py
â”œâ”€â”€ Table2_Comparison_Counts_final.py
â”œâ”€â”€ Table3_Direct_Construction_final.py
â”œâ”€â”€ Table4_Scalability_final.py
â”‚
â”œâ”€â”€ Figure1_Comparison_Efficiency.py
â”œâ”€â”€ Figure2_Scalability.py
â”‚
â””â”€â”€ outputs/
    â”œâ”€â”€ tables/
    â””â”€â”€ figures/
```

Generated CSV, PNG, PDF, and JSON files may either be committed under `outputs/` or regenerated locally.

---

## 13. Suggested GitHub Topics

```text
multiple-testing
holm-procedure
family-wise-error-rate
fwer
multiple-comparisons
statistical-computing
algorithms
python
reproducible-research
adaptive-search
```

---

## 14. Citation

If you use this software, please cite the associated manuscript:

```text
Sarkar, R. and Ray, A.
Exact Range Certificates for Holmâ€™s Step-Down Procedure:
Fixed Blocks and Adaptive Recursive Search.
```

A BibTeX entry may be added once the final journal publication details or preprint identifier are available.

Example placeholder:

```bibtex
@article{SarkarRayBASH,
  author  = {Rishiraj Sarkar and Arnob Ray},
  title   = {Exact Range Certificates for Holm's Step-Down Procedure:
             Fixed Blocks and Adaptive Recursive Search},
  year    = {2026},
  note    = {Manuscript}
}
```

---

## 15. Authors

**Rishiraj Sarkar**  
Department of Statistics  
University of Calcutta  
Ballygunge Science College Campus  
Kolkata, India

**Arnob Ray**  
Department of Mathematics  
SRM Institute of Science and Technology  
Kattankulathur Campus, Tamil Nadu, India

---

## 16. License

Add the license appropriate for your intended distribution.

A common choice for an academic reproducibility repository is the **MIT License**.

If you select MIT, add a `LICENSE` file to the repository and replace this section with:

```text
This project is licensed under the MIT License. See the LICENSE file for details.
```

---

## 17. Contact

For questions about the manuscript or computational implementation, please open a GitHub issue or contact the corresponding author.

---

## 18. Disclaimer

This repository is intended for research reproducibility and methodological study.

The algorithms implemented here concern exact computational recovery of Holmâ€™s step-down decision from ordered p-values. They should not be interpreted as replacing Holmâ€™s inferential criterion or as providing a new multiplicity adjustment.

