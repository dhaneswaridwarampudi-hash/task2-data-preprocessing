# Data documentation — UCI Adult Census Income

## 1. Collection

| Field | Detail |
| --- | --- |
| Dataset | Adult (Census Income) |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) |
| Original files | `adult.data` (train), `adult.test` (test) |
| Direct URLs | `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data` and `.../adult.test` |
| Citation | Kohavi, R. (1996). *Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid.* KDD. |
| Collection method | `src/preprocess.py` downloads both files into `data/raw/` if they are missing, then concatenates them. |
| Minimum sample requirement | Combined table has **48,842** rows before cleaning (well above 1,000). |

This is a real 1994 US Census extract. Each row is one adult. The original prediction task is whether annual income exceeds $50,000.

## 2. Raw schema

| Column | Type (raw) | Meaning |
| --- | --- | --- |
| `age` | integer | Age in years |
| `workclass` | string | Employer type (Private, Self-emp-*, *-gov, …) |
| `fnlwgt` | integer | Census final sampling weight |
| `education` | string | Highest education label |
| `education_num` | integer | Ordered education level (maps 1:1 to `education`) |
| `marital_status` | string | Marital status |
| `occupation` | string | Occupation group |
| `relationship` | string | Role in household |
| `race` | string | Race category as recorded in the census extract |
| `sex` | string | Sex |
| `capital_gain` | integer | Capital gains (USD); `99999` is a top-code |
| `capital_loss` | integer | Capital losses (USD) |
| `hours_per_week` | integer | Usual hours worked per week |
| `native_country` | string | Country of origin |
| `income` | string | `<=50K` or `>50K` (test file includes a trailing `.`) |
| `split` | string | Added here: `train` or `test` from the UCI files |

Missing categoricals in the public files are stored as `?`, not as empty cells.

## 3. Cleaning decisions

1. **Whitespace** — UCI fields are space-padded (`" Private"`). All categoricals are stripped.
2. **Incorrect types** — Numeric columns are coerced with `pd.to_numeric`. Income periods on the test file are removed.
3. **Missing values** — `?` is converted to NA. `workclass`, `occupation`, and `native_country` are imputed as `Unknown` (dropping those rows would discard ~7% of people, mostly without recorded jobs). Remaining NA numeric rows are dropped.
4. **Duplicates** — Exact duplicate rows are removed.
5. **Outliers**
   - `age` clipped to the published Adult range **17–90**.
   - `hours_per_week` values outside the census survey domain **[1, 99]** are clipped. Rows in the **1.5×IQR** tails are flagged as `hours_iqr_outlier` but **kept**, because part-time and overtime hours are real.
   - `capital_gain == 99999` is **not** treated as an error; it is a census top-code and is flagged as `capital_gain_topcoded`.
   - Extreme `fnlwgt` values (above the 99th percentile) are flagged as `high_sampling_weight` rather than deleted, because `fnlwgt` is a survey weight, not a personal attribute.
   - Duplicates created by clipping are dropped.

## 4. Feature engineering

| New column | How it is built | Why |
| --- | --- | --- |
| `income_binary` | 1 if `income == >50K` | Modeling target |
| `net_capital` | gain − loss | Single capital summary |
| `has_capital_gain` / `has_capital_loss` | indicators | Most people have zero capital; the flag is more stable than the raw dollar amount |
| `log_capital_gain` / `log_fnlwgt` | `log1p` | Reduce right skew |
| `age_group` | bins 17–24 … 65+ | Non-linear age effects |
| `hours_group` | part-time / reduced / 40 / overtime / very high | Labor-supply categories |
| `is_united_states` | native country flag | Country is extremely imbalanced |
| `is_married` | marital status starts with `Married` | Strong income correlate |
| `workclass_simplified` | gov / self-employed / unpaid-unknown / private | Collapse rare workclass levels |
| `*_code` | `LabelEncoder` | Numeric codes for tree models |
| `*_z` | `StandardScaler` | Z-scored numerics for linear models |

## 5. Delivered tables

| File | Description |
| --- | --- |
| `data/raw/adult.data` | Original UCI training file |
| `data/raw/adult.test` | Original UCI test file |
| `data/processed/adult_clean.csv` | Cleaned, documented, human-readable table with engineered features |
| `data/processed/adult_model_ready.csv` | Same rows plus encodings and scaled columns |
| `outputs/reports/pipeline_report.md` | Before/after counts from the last run |
| `outputs/reports/encoding_artifacts.json` | Label maps and scaler statistics |
| `outputs/figures/*.png` | Plots used as screenshots / notebook output |

## 6. How to reproduce

```bash
python src/preprocess.py
```

Or run all cells in `notebooks/01_data_collection_preprocessing.ipynb`.

Expected result: **>1,000 rows**, **0 missing cells** in the clean table, duplicate-free, typed correctly, with engineered features and saved figures.
