# BASH-Holm Range Certificates

Reproducible Python code for the manuscript **“Exact Range Certificates for Holm’s Step-Down Procedure: Fixed Blocks and Adaptive Recursive Search.”**

This repository contains implementations of Holm, BASH, recursive certification, the completed two-level direct construction, and Galloping BASH.

The goal is computational: recover Holm’s exact rejection set while reducing the number of post-sorting threshold comparisons.

## Repository Contents

### Tables

- `Table1_Inferential_Validation_final.py`  
  Reproduces the inferential validation study, including FWER, Monte Carlo error, power, and the conservative repair.

- `Table2_Comparison_Counts_final.py`  
  Reproduces the comparison-count study for Holm, BASH with different block sizes, naive recursion, and Galloping BASH.

- `Table3_Direct_Construction_final.py`  
  Reproduces the square-boundary direct-construction experiment and the explicit counterexample beyond the equivalence boundary.

- `Table4_Scalability_final.py`  
  Reproduces the scalability study from 1,000 to 1,000,000 hypotheses.

### Figures

- `Figure1_Comparison_Efficiency.py`  
  Generates the comparison-efficiency heatmap relative to sequential Holm.

- `Figure2_Scalability.py`  
  Generates the scalability plots for sparse, moderate, and dense signal regimes.

## Reproducibility

All simulations use:

- significance level 0.05
- master seed `20260913`
- NumPy PCG64 random-number generation
- deterministic BLAKE2b-derived streams for each experiment

Each table script writes both summary and replicate-level CSV files.

## Requirements

```text
Python >= 3.10
NumPy
SciPy
Matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

## Running the Code

Run the table scripts:

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

## Interpretation

The valid certificate methods are computational implementations of Holm’s procedure. They are not new multiple-testing criteria.

The reported comparison counts refer only to the post-sorting decision stage. Sorting cost and total wall-clock time are not included.

## Suggested Repository Structure

```text
bash-holm-range-certificates/
├── README.md
├── Table1_Inferential_Validation_final.py
├── Table2_Comparison_Counts_final.py
├── Table3_Direct_Construction_final.py
├── Table4_Scalability_final.py
├── Figure1_Comparison_Efficiency.py
└── Figure2_Scalability.py
```

## Citation

If you use this code, please cite the associated manuscript:

**Rishiraj Sarkar and Arnob Ray**  
*Exact Range Certificates for Holm’s Step-Down Procedure: Fixed Blocks and Adaptive Recursive Search.*

## Authors

**Rishiraj Sarkar**  
Department of Statistics, University of Calcutta

**Arnob Ray**  
Department of Mathematics, SRM Institute of Science and Technology

## License

Add your preferred open-source license before public release. The MIT License is a common choice for academic reproducibility repositories.
