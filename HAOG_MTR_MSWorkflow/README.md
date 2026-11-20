# HAOG-MTR: six-pollutant emission prediction

This folder contains the code and a processed data subset supporting the
HAOG-MTR experiments for one-minute-ahead prediction of six pollutants from a
municipal solid-waste incineration process.

## Contents

- `code/run_full_period_validation.py`: forward temporal validation, external
  testing, PM/HF error analysis, and split-conformal uncertainty evaluation.
- `code/original_HAOG_MTR_implementation.py`: model implementation used by the
  validation script.
- `data/external_test_sample_30min.csv`: every 30th record from the independent
  external test period, including observations, predictions, and 90%/95%
  prediction intervals for HCl, SO2, NOx, PM, CO, and HF.
- `data/External_Test_Metrics_95CI.csv`: external-test R2, RMSE, MAE, and 95%
  confidence intervals.
- `data/Forward_Validation_Summary_95CI.csv`: five forward-validation folds
  summarized with 95% confidence intervals.
- `data/Conformal_Uncertainty_Metrics.csv`: PICP, MPIW, and Winkler scores for
  90% and 95% split-conformal prediction intervals.
- `data/PM_HF_Concentration_Strata_Metrics.csv`: concentration-stratified PM and
  HF error analysis.
- `data/PM_HF_Error_by_Concentration_Decile.csv`: error by concentration decile.
- `data/PM_HF_Residual_Autocorrelation.csv`: residual autocorrelation results.
- `DATA_DICTIONARY.md`: variable definitions and units.
- `SHA256SUMS.txt`: SHA-256 checksums for integrity verification.

## Validation design

The full study period runs from 10 May to 7 August 2024 at one-minute
resolution. Five forward temporal folds were used, followed by a fully held-out
external test period covering the final two weeks. Prediction intervals were
calibrated only on data preceding the external test period.

## Data availability

The full raw operational data cannot be released because they contain
confidential plant-level process information. The processed subset included
here is sufficient to verify the reported external-test predictions,
uncertainty intervals, and summary statistics, but it is not intended to
reconstruct the complete training set.

## Reproducibility note

The scripts expect the original feature store and raw process records, which
are not included. Paths are defined near the top of the validation script and
must be adapted for an authorized local copy of the data. Set
`HAOG_DATA_ROOT` to the folder containing `horizon1_result.joblib`; optionally
set `HAOG_OUTPUT_DIR` to select the output folder. For example, in PowerShell:

```powershell
$env:HAOG_DATA_ROOT = "D:\authorized_data"
$env:HAOG_OUTPUT_DIR = "D:\haog_results"
python code/run_full_period_validation.py
```

Install the packages listed in `requirements.txt` before running the script.

## Suggested citation statement

> The code and a processed data subset supporting the reported analyses are
> available at https://github.com/haolupeng/data/tree/1/HAOG_MTR_MSWorkflow.
> The complete raw operational dataset is not publicly available because of
> confidentiality restrictions imposed by the industrial data provider.

## Integrity

Run `Get-FileHash -Algorithm SHA256 <file>` on Windows or `sha256sum <file>`
on Linux/macOS and compare the output with `SHA256SUMS.txt`.



