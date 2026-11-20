# Data dictionary

## `external_test_sample_30min.csv`

| Field | Meaning | Unit |
|---|---|---|
| `time` | Timestamp of the one-minute record; the public subset retains every 30th record | local plant time |
| `date` | Calendar date | YYYY-MM-DD |
| `observed_<pollutant>` | Observed concentration | mg m^-3 |
| `predicted_<pollutant>` | HAOG-MTR point prediction | mg m^-3 |
| `lower_<pollutant>_90`, `upper_<pollutant>_90` | 90% split-conformal interval bounds | mg m^-3 |
| `lower_<pollutant>_95`, `upper_<pollutant>_95` | 95% split-conformal interval bounds | mg m^-3 |

Pollutant labels are `HCl`, `SO2`, `NOx`, `PM`, `CO`, and `HF`.

## Metric files

| Metric | Meaning |
|---|---|
| `R2` | Coefficient of determination |
| `RMSE` | Root-mean-square error (mg m^-3) |
| `MAE` | Mean absolute error (mg m^-3) |
| `PICP` | Prediction interval coverage probability |
| `MPIW` | Mean prediction interval width (mg m^-3) |
| `Winkler` | Winkler interval score; lower is better |
| `bias` | Mean observed-minus-predicted residual (mg m^-3) |
| `ACF` | Residual autocorrelation at the stated lag |

The concentration-stratified tables may show negative R2 values within very
narrow concentration bands. In these bands, MAE and bias are more informative
because the observed within-band variance is small.


