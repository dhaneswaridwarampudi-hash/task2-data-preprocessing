"""
UCI Adult Census Income — collection and preprocessing pipeline.

Source: UCI Machine Learning Repository, Adult (Census Income) dataset.
https://archive.ics.uci.edu/dataset/2/adult
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

UCI_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

CATEGORICAL = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def ensure_dirs() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, FIG_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _fetch(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as fh:
        fh.write(resp.read())


def download_raw() -> tuple[Path, Path]:
    """Download original UCI train/test files if they are not already present."""
    ensure_dirs()
    train_path = RAW_DIR / "adult.data"
    test_path = RAW_DIR / "adult.test"
    if not train_path.exists() or train_path.stat().st_size < 1_000_000:
        _fetch(UCI_TRAIN, train_path)
    if not test_path.exists() or test_path.stat().st_size < 500_000:
        _fetch(UCI_TEST, test_path)
    return train_path, test_path


def load_raw() -> pd.DataFrame:
    train_path, test_path = download_raw()
    train = pd.read_csv(train_path, header=None, names=COLUMNS)
    first_line = test_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    skip = 1 if first_line.startswith("|") else 0
    test = pd.read_csv(test_path, header=None, names=COLUMNS, skiprows=skip)
    train["split"] = "train"
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def profile_frame(df: pd.DataFrame, label: str) -> dict:
    return {
        "label": label,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items() if v},
    }


def clean_types_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in CATEGORICAL + ["income"]:
        out[col] = out[col].astype(str).str.strip()

    # Test file suffixes income with a period: "<=50K."
    out["income"] = out["income"].str.replace(r"\.$", "", regex=True)

    # UCI encodes missing categoricals as "?"
    out = out.replace("?", np.nan)
    out = out.replace({"nan": np.nan, "None": np.nan})

    numeric_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop_duplicates().reset_index(drop=True)

    # Workclass and occupation missingness are strongly aligned; treat as "Unknown".
    for col in ["workclass", "occupation", "native_country"]:
        out[col] = out[col].fillna("Unknown")

    out = out.dropna(subset=numeric_cols + ["income"]).reset_index(drop=True)
    return out


def handle_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Treat implausible values without erasing legitimate census tails."""
    out = df.copy()
    notes: dict = {}

    # Age 17-90 is the published Adult range; clip anything outside it.
    notes["age_before_minmax"] = [int(out["age"].min()), int(out["age"].max())]
    out["age"] = out["age"].clip(lower=17, upper=90)

    # Capital-gain of 99999 is a documented top-code, not a measurement error.
    out["capital_gain_topcoded"] = (out["capital_gain"] == 99999).astype(int)

    # Hours-per-week: flag 1.5*IQR tails, but only clip to the survey domain [1, 99].
    # Part-time and overtime hours are real labor-supply values, not errors.
    q1 = out["hours_per_week"].quantile(0.25)
    q3 = out["hours_per_week"].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out["hours_iqr_outlier"] = (
        (out["hours_per_week"] < low) | (out["hours_per_week"] > high)
    ).astype(int)
    notes["hours_per_week_iqr_bounds"] = [float(low), float(high)]
    notes["hours_per_week_flagged"] = int(out["hours_iqr_outlier"].sum())
    notes["hours_outside_survey_bounds"] = int(
        ((out["hours_per_week"] < 1) | (out["hours_per_week"] > 99)).sum()
    )
    out["hours_per_week"] = out["hours_per_week"].clip(lower=1, upper=99)

    # fnlwgt is a census sampling weight, not a person-level feature.
    w_high = out["fnlwgt"].quantile(0.99)
    out["high_sampling_weight"] = (out["fnlwgt"] > w_high).astype(int)
    notes["fnlwgt_p99"] = float(w_high)

    out = out.drop_duplicates().reset_index(drop=True)
    return out, notes


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["income_binary"] = (out["income"] == ">50K").astype(int)
    out["net_capital"] = out["capital_gain"] - out["capital_loss"]
    out["has_capital_gain"] = (out["capital_gain"] > 0).astype(int)
    out["has_capital_loss"] = (out["capital_loss"] > 0).astype(int)
    out["log_capital_gain"] = np.log1p(out["capital_gain"])
    out["log_fnlwgt"] = np.log1p(out["fnlwgt"])

    out["age_group"] = pd.cut(
        out["age"],
        bins=[16, 24, 34, 44, 54, 64, 90],
        labels=["17-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    )
    out["hours_group"] = pd.cut(
        out["hours_per_week"],
        bins=[0, 20, 35, 40, 50, 99],
        labels=["part_time", "reduced", "standard_40", "overtime", "very_high"],
        include_lowest=True,
    )
    out["is_united_states"] = (out["native_country"] == "United-States").astype(int)
    out["is_married"] = out["marital_status"].str.startswith("Married").astype(int)
    out["workclass_simplified"] = out["workclass"].replace(
        {
            "Federal-gov": "Government",
            "Local-gov": "Government",
            "State-gov": "Government",
            "Self-emp-inc": "Self-employed",
            "Self-emp-not-inc": "Self-employed",
            "Without-pay": "Unpaid/Unknown",
            "Never-worked": "Unpaid/Unknown",
            "Unknown": "Unpaid/Unknown",
        }
    )
    return out


def encode_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Produce a modeling table with encodings plus a human-readable clean table."""
    model_df = df.copy()
    encoders: dict[str, dict] = {}
    cat_cols = CATEGORICAL + ["age_group", "hours_group", "workclass_simplified"]
    for col in cat_cols:
        model_df[col] = model_df[col].astype(str)
        le = LabelEncoder()
        model_df[f"{col}_code"] = le.fit_transform(model_df[col])
        encoders[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}

    scale_cols = [
        "age",
        "education_num",
        "hours_per_week",
        "log_capital_gain",
        "log_fnlwgt",
        "net_capital",
    ]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_df[scale_cols])
    for i, col in enumerate(scale_cols):
        model_df[f"{col}_z"] = scaled[:, i]

    artifacts = {
        "label_encoders": encoders,
        "scaler_mean": {c: float(m) for c, m in zip(scale_cols, scaler.mean_)},
        "scaler_scale": {c: float(s) for c, s in zip(scale_cols, scaler.scale_)},
    }
    return model_df, artifacts


def save_figures(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    raw_missing = raw.copy()
    for col in CATEGORICAL + ["income"]:
        stripped = raw_missing[col].astype(str).str.strip()
        raw_missing[col] = stripped.replace({"?": np.nan, "nan": np.nan, "None": np.nan})
    miss = raw_missing.isna().mean().sort_values(ascending=False).head(10)
    miss.plot(kind="bar", ax=axes[0], color="#c44e52")
    axes[0].set_title("Missingness before cleaning (share of rows)")
    axes[0].set_ylabel("Fraction missing")
    axes[0].tick_params(axis="x", rotation=45)

    sns.histplot(clean["age"], bins=30, ax=axes[1], color="#4c72b0")
    axes[1].set_title("Age distribution after cleaning")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_missingness_and_age.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=clean, x="income", y="hours_per_week", ax=axes[0])
    axes[0].set_title("Hours per week by income (survey domain 1–99)")
    sns.countplot(data=clean, x="age_group", hue="income", ax=axes[1], order=clean["age_group"].cat.categories)
    axes[1].set_title("Income by engineered age group")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_hours_and_age_groups.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=clean.groupby("workclass_simplified", as_index=False)["income_binary"].mean(),
        x="workclass_simplified",
        y="income_binary",
        ax=ax,
        color="#55a868",
    )
    ax.set_ylabel("Share earning >50K")
    ax.set_title("High-income rate by simplified workclass")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_workclass_income_rate.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    num_cols = ["age", "education_num", "hours_per_week", "net_capital", "income_binary"]
    corr = clean[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlations after engineering")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def write_pipeline_report(before: dict, after: dict, outlier_notes: dict, n_final: int) -> None:
    lines = [
        "# Preprocessing pipeline report",
        "",
        "## Source",
        "- Dataset: UCI Adult (Census Income)",
        "- URL: https://archive.ics.uci.edu/dataset/2/adult",
        "- License: UCI repository terms (public research use)",
        f"- Combined train + test samples collected: {before['n_rows']}",
        "",
        "## Before cleaning",
        f"- Rows: {before['n_rows']}",
        f"- Columns: {before['n_cols']}",
        f"- Duplicate rows: {before['duplicate_rows']}",
        f"- Missing cells (after treating '?' as NA): {before['missing_cells']}",
        "",
        "## After cleaning + feature engineering",
        f"- Rows: {after['n_rows']}",
        f"- Columns: {after['n_cols']}",
        f"- Duplicate rows: {after['duplicate_rows']}",
        f"- Missing cells: {after['missing_cells']}",
        f"- Modeling/clean table rows written: {n_final}",
        "",
        "## Outlier handling",
        f"- Age clipped to [17, 90]; original min/max: {outlier_notes['age_before_minmax']}",
        f"- Hours-per-week IQR bounds: {outlier_notes['hours_per_week_iqr_bounds']}",
        f"- Hours-per-week IQR outliers flagged (not dropped): {outlier_notes['hours_per_week_flagged']}",
        f"- Hours outside survey bounds [1, 99] clipped: {outlier_notes['hours_outside_survey_bounds']}",
        f"- fnlwgt 99th percentile (high-weight flag): {outlier_notes['fnlwgt_p99']:.0f}",
        "- capital_gain == 99999 retained and flagged as top-coded, not dropped",
        "",
        "## Feature engineering",
        "- income_binary, net_capital, has_capital_gain/loss, log transforms",
        "- age_group, hours_group, is_united_states, is_married, workclass_simplified",
        "- label encodings (*_code) and z-scored numeric columns (*_z) for modeling",
        "",
    ]
    (REPORT_DIR / "pipeline_report.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "before_profile.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    (REPORT_DIR / "after_profile.json").write_text(json.dumps(after, indent=2), encoding="utf-8")


def run() -> pd.DataFrame:
    ensure_dirs()
    raw = load_raw()
    raw_for_missing = raw.copy()
    for col in CATEGORICAL + ["income"]:
        raw_for_missing[col] = raw_for_missing[col].astype(str).str.strip().replace("?", np.nan)

    before = profile_frame(raw_for_missing, "raw")
    cleaned = clean_types_and_missing(raw)
    cleaned, outlier_notes = handle_outliers(cleaned)
    featured = engineer_features(cleaned)
    after = profile_frame(featured, "clean")
    model_df, artifacts = encode_and_scale(featured)

    featured.to_csv(PROCESSED_DIR / "adult_clean.csv", index=False)
    model_df.to_csv(PROCESSED_DIR / "adult_model_ready.csv", index=False)
    (REPORT_DIR / "encoding_artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    save_figures(raw, featured)
    write_pipeline_report(before, after, outlier_notes, len(featured))

    summary = {
        "raw_rows": before["n_rows"],
        "clean_rows": after["n_rows"],
        "clean_cols": after["n_cols"],
        "missing_after": after["missing_cells"],
    }
    print(json.dumps(summary, indent=2))
    return featured


if __name__ == "__main__":
    run()
