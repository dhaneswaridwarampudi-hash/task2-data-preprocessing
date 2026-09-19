# Preprocessing pipeline report

## Source
- Dataset: UCI Adult (Census Income)
- URL: https://archive.ics.uci.edu/dataset/2/adult
- License: UCI repository terms (public research use)
- Combined train + test samples collected: 48842

## Before cleaning
- Rows: 48842
- Columns: 16
- Duplicate rows: 29
- Missing cells (after treating '?' as NA): 6465

## After cleaning + feature engineering
- Rows: 48813
- Columns: 30
- Duplicate rows: 0
- Missing cells: 0
- Modeling/clean table rows written: 48813

## Outlier handling
- Age clipped to [17, 90]; original min/max: [17, 90]
- Hours-per-week IQR bounds: [32.5, 52.5]
- Hours-per-week IQR outliers flagged (not dropped): 13489
- Hours outside survey bounds [1, 99] clipped: 0
- fnlwgt 99th percentile (high-weight flag): 509495
- capital_gain == 99999 retained and flagged as top-coded, not dropped

## Feature engineering
- income_binary, net_capital, has_capital_gain/loss, log transforms
- age_group, hours_group, is_united_states, is_married, workclass_simplified
- label encodings (*_code) and z-scored numeric columns (*_z) for modeling
