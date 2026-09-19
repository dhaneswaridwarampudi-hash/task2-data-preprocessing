# Task 2 — Data Collection & Preprocessing

Real-world census income data from the **UCI Machine Learning Repository**, cleaned and documented for modeling.

| Requirement | This repository |
| --- | --- |
| ≥ 1,000 public samples | 48,842 raw rows (Adult train + test) |
| Cleaning | Missing `?` values, duplicates, types, outliers |
| Feature engineering | Income target, capital summaries, age/hours bins, encodings, scaling |
| Clean dataset | `data/processed/adult_clean.csv` |
| Script / notebook | `src/preprocess.py`, `notebooks/01_data_collection_preprocessing.ipynb` |
| Data docs | `docs/DATA_DOCUMENTATION.md` |
| Screenshots / output | `outputs/figures/` and `outputs/reports/` |

## Dataset

[Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) — 1994 US Census extract. Target: whether income is `>50K`.

Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.* Proceedings of KDD.

## Pipeline

```
UCI adult.data + adult.test
        │
        ▼
 strip / types / '?' → NA
        │
        ▼
 impute Unknown  ·  drop duplicates
        │
        ▼
 clip age to 17–90  ·  flag IQR hours  ·  flag top-coded capital-gain
        │
        ▼
 engineer features  ·  encode  ·  scale
        │
        ▼
 adult_clean.csv  +  adult_model_ready.csv  +  figures
```

## Setup

Python 3.10+ with pandas, numpy, matplotlib, seaborn, scikit-learn (see `requirements.txt`).

```bash
python src/preprocess.py
```

This downloads the public UCI files into `data/raw/` (if needed), writes the clean tables, and regenerates reports and figures.

## Publish to GitHub

Git was not available on this machine when the project was created (the Git for Windows installer needs an Administrator prompt). After Git and [GitHub CLI](https://cli.github.com/) are installed:

```bash
cd C:\Users\Intel\task2-data-preprocessing
git init
git add .
git commit -m "Add UCI Adult collection and preprocessing pipeline."
gh repo create task2-data-preprocessing --public --source=. --remote=origin --push
```

```
data/raw/            original UCI files
data/processed/      clean and model-ready CSVs
src/preprocess.py    full reproducible pipeline
notebooks/           walkthrough with outputs
docs/                data dictionary and decisions
outputs/figures/     plots (assignment screenshots)
outputs/reports/     before/after JSON + markdown report
```
