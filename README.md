# Reproducibility Package for Time-Aware Botnet Detection Experiments

This repository contains the code, configuration, and generated outputs required to reproduce the main experimental results reported in the associated manuscript on time-aware botnet detection under operational reliability and throughput constraints.

## Overview

The experiment evaluates binary botnet detection on CICIDS2018-derived traffic files using chronological train/test windowing within Friday traffic, forward validation inside the training window, multi-objective hyperparameter optimization with NSGA-II, representative Pareto model selection, reliability-constrained threshold calibration, negative-day reliability analysis on Thursday traffic, and end-to-end benchmarking that includes both DMatrix construction and prediction time.

## Repository Structure

```text
project_root/
├── data/
│   ├── Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
│   └── Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
├── experiment_results/
│   ├── diagnostics/
│   └── paper_figures/
├── run_experiment.py
├── requirements.txt
├── README.md
├── LICENSE
└── CITATION.cff
```

## Required Input Files

Place the following CSV files in the `data/` directory before running the experiment:

`Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`

`Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv`

These files are not redistributed in this repository. Users should obtain them independently from the appropriate benchmark source and place them in the `data/` directory with the exact filenames shown above.

## Environment

The experiment was developed and tested in Python 3.10. A virtual environment is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Experiment

Run the full pipeline with:

```bash
python run_experiment.py
```

## Main Outputs

After execution, the script generates diagnostic files, final result tables, and publication-ready figures.

The main diagnostic outputs are saved in `experiment_results/diagnostics/` and include:

- `experimental_setup.csv`
- `config_used.json`
- `fri_time_coverage.json`
- `thu_time_coverage.json` (if Thursday data are available)
- `split_summaries.csv`
- `pareto_points.csv`
- `selected_ids.json`
- `calibrated_models.json`

The final result tables are saved as:

- `experiment_results/final_calibrated_results.csv`
- `experiment_results/table1_selected_pareto_configs.csv`
- `experiment_results/table2_article_results.csv`
- `experiment_results/table2_article_results_full.csv`
- `experiment_results/article_caption_helper.txt`

The generated figures are saved in `experiment_results/paper_figures/` and include:

- `fig1_pareto_frontier.png`
- `fig2_e2e_time_decomposition.png`
- `fig3_s2_pr_roc_curves.png`
- `fig4_thursday_fpr_vs_threshold_all.png` (if Thursday data are available)
- `fig5_s2_score_distributions.png`
- `fig6_s2_confmat_panel.png`

## Labeling Convention

In this repository, the binary target is defined as follows:

- `Label = 1` only for the exact class `Bot`
- `Label = 0` for `Benign`

The Thursday file is used as a negative-day reliability check. If available, its benign flows are used to derive an FPR-constrained threshold and to assess alert behavior under negative-day conditions.

## Reproducibility Notes

The code records software versions, platform metadata, CPU metadata, optional RAM and physical core information, SHA256 hashes of the input files, and the full experiment configuration used in the run.

Random seeds and thread counts are fixed to improve reproducibility across repeated executions.

## Data Availability

This repository provides the full experimental protocol, code, configuration, and generated outputs, but does not redistribute the raw benchmark CSV files. Users must obtain the required benchmark files independently and place them in the `data/` directory before running the experiment.

## Citation

If you use this reproducibility package, please cite the associated article and, where appropriate, the repository release.
