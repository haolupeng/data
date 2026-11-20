from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


SEED = 20261118
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = Path(os.environ.get("HAOG_DATA_ROOT", PROJECT_ROOT / "data_private"))
SOURCE = DATA_ROOT / "horizon1_result.joblib"
ORIGINAL = CODE_DIR / "original_HAOG_MTR_implementation.py"
OUT = Path(os.environ.get("HAOG_OUTPUT_DIR", PROJECT_ROOT / "results"))
OUT.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["HCl", "SO2", "NOx", "PM", "CO", "HF"]
TARGET_CODES = ["Xa", "Xb", "Xc", "Xd", "Xe", "Xf"]
TEST_START = np.datetime64("2024-07-24T00:00:00", "s")
TEST_END = np.datetime64("2024-08-07T00:00:00", "s")
CAL_START = np.datetime64("2024-07-10T00:00:00", "s")

FOLD_WINDOWS = [
    ("2024-06-15", "2024-06-23"),
    ("2024-06-23", "2024-07-01"),
    ("2024-07-01", "2024-07-09"),
    ("2024-07-09", "2024-07-17"),
    ("2024-07-17", "2024-07-24"),
]


def load_original_module():
    spec = importlib.util.spec_from_file_location("haog_original", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_row(y, p, pollutant, split, fold, start, end, model="HAOG-MTR"):
    return {
        "split": split,
        "fold": fold,
        "model": model,
        "pollutant": pollutant,
        "start": str(start),
        "end": str(end),
        "n": int(len(y)),
        "R2": float(r2_score(y, p)),
        "RMSE": float(mean_squared_error(y, p, squared=False)),
        "MAE": float(mean_absolute_error(y, p)),
    }


def select_fold_features(x0, y, train_idx, current_idx, base_names, seed):
    """Training-only feature selection: 6 current targets + 12 ranked base inputs."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(train_idx)
    if len(idx) > 50000:
        idx = np.sort(rng.choice(idx, 50000, replace=False))
    selector = ExtraTreesRegressor(
        n_estimators=40,
        max_depth=16,
        min_samples_leaf=4,
        max_features=0.60,
        n_jobs=-1,
        random_state=seed,
    )
    selector.fit(x0[idx], np.log1p(np.maximum(y[idx], 0)))
    current_set = set(current_idx)
    ranked = np.argsort(selector.feature_importances_)[::-1]
    other = [int(i) for i in ranked if int(i) not in current_set][:12]
    base_keep = list(current_idx) + other
    selected = [base_names[i] for i in base_keep]
    return base_keep, selected


def build_design(mod, x0, y, times, train_idx, current_idx, base_names, seed):
    xfull, extra_names = mod.add_ar_features(x0, times.astype("int64"), current_idx)
    base_keep, selected_base = select_fold_features(
        x0, y, train_idx, current_idx, base_names, seed
    )
    extra_start = x0.shape[1]
    keep = base_keep + list(range(extra_start, xfull.shape[1]))
    x = xfull[:, keep].astype(np.float32, copy=False)
    weighted_start = xfull.shape[1] - 6
    current_positions = [keep.index(weighted_start + j) for j in range(6)]
    names = selected_base + extra_names
    assert x.shape[1] == 120, x.shape
    return x, current_positions, names


def fit_knn_pm(xtr, ytr, xte):
    # Same KNN definition as the original seven-model experiment.
    model = make_pipeline(
        RobustScaler(),
        KNeighborsRegressor(n_neighbors=9, weights="distance", p=2, n_jobs=-1),
    )
    model.fit(xtr, ytr[:, 3])
    # Chunking bounds peak memory for the full-period exact-distance search.
    pred = np.empty(len(xte), dtype=float)
    for start in range(0, len(xte), 256):
        stop = min(start + 256, len(xte))
        pred[start:stop] = model.predict(xte[start:stop])
    return pred


def t_ci(values):
    a = np.asarray(values, dtype=float)
    mean = float(np.mean(a))
    sd = float(np.std(a, ddof=1))
    half = 2.7764451051977987 * sd / math.sqrt(len(a))
    return mean, sd, mean - half, mean + half


def daily_block_bootstrap(y, p, dates, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    unique = np.unique(dates)
    blocks = [np.flatnonzero(dates == d) for d in unique]
    vals = np.empty((n_boot, 3), dtype=float)
    for b in range(n_boot):
        chosen = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in chosen])
        vals[b, 0] = r2_score(y[idx], p[idx])
        vals[b, 1] = mean_squared_error(y[idx], p[idx], squared=False)
        vals[b, 2] = mean_absolute_error(y[idx], p[idx])
    return np.quantile(vals, [0.025, 0.975], axis=0)


def conformal_quantile(scores, alpha):
    scores = np.sort(np.asarray(scores, dtype=float))
    rank = int(math.ceil((len(scores) + 1) * (1 - alpha)))
    rank = min(max(rank, 1), len(scores))
    return float(scores[rank - 1])


def interval_metrics(y, lo, hi, alpha):
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    score = width.copy()
    score[y < lo] += (2 / alpha) * (lo[y < lo] - y[y < lo])
    score[y > hi] += (2 / alpha) * (y[y > hi] - hi[y > hi])
    return float(np.mean(covered)), float(np.mean(width)), float(np.mean(score))


def residual_acf(residual, lags=(1, 5, 15, 60)):
    out = {}
    residual = np.asarray(residual, dtype=float)
    for lag in lags:
        if len(residual) <= lag:
            out[lag] = np.nan
        else:
            out[lag] = float(np.corrcoef(residual[:-lag], residual[lag:])[0, 1])
    return out


def main():
    started = time.time()
    print("Loading frozen full-period feature store...", flush=True)
    saved = joblib.load(SOURCE)
    mod = load_original_module()
    times = np.asarray(saved["target_times"], dtype=np.int64).astype("datetime64[s]")
    x0 = np.asarray(saved["Xall"], dtype=np.float32)
    y_raw = np.asarray(saved["y"], dtype=np.float64)
    base_names = list(saved["feature_names"])
    current_idx = [base_names.index(f"{c}[t]") for c in TARGET_CODES]

    # Same causal four-sample target definition used by the seven-model study.
    y = y_raw.copy()
    y[3:] = (y_raw[3:] + y_raw[2:-1] + y_raw[1:-2] + 0.60 * y_raw[:-3]) / 3.60
    valid = np.arange(len(y)) >= 15
    x0, y, times = x0[valid], y[valid], times[valid]
    print(f"Usable full-period samples after lag warm-up: {len(y):,}", flush=True)

    fold_records, prediction_frames, feature_audit = [], [], []
    checkpoint = OUT / "forward_fold_metrics_checkpoint.csv"

    for fold, (start_s, end_s) in enumerate(FOLD_WINDOWS, 1):
        start = np.datetime64(start_s, "s")
        end = np.datetime64(end_s, "s")
        tr = np.flatnonzero(times < start)
        te = np.flatnonzero((times >= start) & (times < end))
        print(f"Fold {fold}: train={len(tr):,}, test={len(te):,}, {start}--{end}", flush=True)
        x, current_positions, selected_names = build_design(
            mod, x0, y, times, tr, current_idx, base_names, SEED + fold
        )
        pred, weights = mod.fit_haog(
            x[tr], y[tr], x[te], current_positions, SEED + 1000 * fold
        )
        knn_pm = fit_knn_pm(x[tr], y[tr], x[te])
        for j, pollutant in enumerate(POLLUTANTS):
            fold_records.append(metric_row(
                y[te, j], pred[:, j], pollutant, "forward_fold", fold,
                times[te].min(), times[te].max(), "HAOG-MTR"
            ))
        fold_records.append(metric_row(
            y[te, 3], knn_pm, "PM", "forward_fold", fold,
            times[te].min(), times[te].max(), "KNN"
        ))
        feature_audit.append({
            "fold": fold,
            "train_start": str(times[tr].min()),
            "train_end": str(times[tr].max()),
            "test_start": str(times[te].min()),
            "test_end": str(times[te].max()),
            "train_n": int(len(tr)),
            "test_n": int(len(te)),
            "selected_base_features": selected_names[:18],
            "gate_parameters": {POLLUTANTS[j]: weights[j].tolist() for j in range(6)},
        })
        frame = pd.DataFrame({"time": times[te].astype(str), "fold": fold})
        for j, pol in enumerate(POLLUTANTS):
            frame[f"observed_{pol}"] = y[te, j]
            frame[f"predicted_{pol}"] = pred[:, j]
        frame["predicted_PM_KNN"] = knn_pm
        prediction_frames.append(frame)
        pd.DataFrame(fold_records).to_csv(checkpoint, index=False, encoding="utf-8-sig")
        print(f"Fold {fold} completed; macro R2={np.mean([r2_score(y[te,j], pred[:,j]) for j in range(6)]):.4f}", flush=True)

    fold_metrics = pd.DataFrame(fold_records)
    fold_metrics.to_csv(OUT / "Forward_Validation_Fold_Metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUT / "Forward_Validation_OOF_Predictions.csv", index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    for (model, pollutant), block in fold_metrics.groupby(["model", "pollutant"]):
        row = {"model": model, "pollutant": pollutant, "folds": int(len(block))}
        for metric in ["R2", "RMSE", "MAE"]:
            mean, sd, lo, hi = t_ci(block[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_CI95_low"] = lo
            row[f"{metric}_CI95_high"] = hi
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUT / "Forward_Validation_Summary_95CI.csv", index=False, encoding="utf-8-sig"
    )

    # Final chronological train/calibration/external-test experiment.
    core = np.flatnonzero(times < CAL_START)
    cal = np.flatnonzero((times >= CAL_START) & (times < TEST_START))
    test = np.flatnonzero((times >= TEST_START) & (times < TEST_END))
    print(f"Final split: core={len(core):,}, calibration={len(cal):,}, external={len(test):,}", flush=True)
    x, current_positions, selected_names = build_design(
        mod, x0, y, times, core, current_idx, base_names, SEED + 90
    )
    both = np.concatenate([cal, test])
    pred_both, weights = mod.fit_haog(
        x[core], y[core], x[both], current_positions, SEED + 90000
    )
    pred_cal = pred_both[:len(cal)]
    pred_test = pred_both[len(cal):]
    knn_pm_test = fit_knn_pm(x[core], y[core], x[test])

    external_rows = []
    dates = times[test].astype("datetime64[D]")
    for j, pollutant in enumerate(POLLUTANTS):
        row = metric_row(
            y[test, j], pred_test[:, j], pollutant, "external_test", 0,
            times[test].min(), times[test].max(), "HAOG-MTR"
        )
        ci = daily_block_bootstrap(y[test, j], pred_test[:, j], dates, seed=SEED+j)
        for k, metric in enumerate(["R2", "RMSE", "MAE"]):
            row[f"{metric}_CI95_low"] = float(ci[0, k])
            row[f"{metric}_CI95_high"] = float(ci[1, k])
        external_rows.append(row)
    pm_knn_row = metric_row(
        y[test, 3], knn_pm_test, "PM", "external_test", 0,
        times[test].min(), times[test].max(), "KNN"
    )
    ci = daily_block_bootstrap(y[test, 3], knn_pm_test, dates, seed=SEED+33)
    for k, metric in enumerate(["R2", "RMSE", "MAE"]):
        pm_knn_row[f"{metric}_CI95_low"] = float(ci[0, k])
        pm_knn_row[f"{metric}_CI95_high"] = float(ci[1, k])
    external_rows.append(pm_knn_row)
    pd.DataFrame(external_rows).to_csv(
        OUT / "External_Test_Metrics_95CI.csv", index=False, encoding="utf-8-sig"
    )

    # Time-ordered split conformal intervals.
    uncertainty_rows = []
    interval_data = pd.DataFrame({"time": times[test].astype(str)})
    interval_data["date"] = dates.astype(str)
    for j, pollutant in enumerate(POLLUTANTS):
        interval_data[f"observed_{pollutant}"] = y[test, j]
        interval_data[f"predicted_{pollutant}"] = pred_test[:, j]
        scores = np.abs(y[cal, j] - pred_cal[:, j])
        for confidence in [0.90, 0.95]:
            alpha = 1 - confidence
            q = conformal_quantile(scores, alpha)
            lo = np.maximum(pred_test[:, j] - q, 0)
            hi = pred_test[:, j] + q
            picp, mpiw, winkler = interval_metrics(y[test, j], lo, hi, alpha)
            uncertainty_rows.append({
                "pollutant": pollutant,
                "confidence": confidence,
                "alpha": alpha,
                "calibration_n": int(len(cal)),
                "test_n": int(len(test)),
                "conformal_q": q,
                "PICP": picp,
                "MPIW": mpiw,
                "Winkler_score": winkler,
            })
            suffix = str(int(confidence * 100))
            interval_data[f"lower_{pollutant}_{suffix}"] = lo
            interval_data[f"upper_{pollutant}_{suffix}"] = hi
    pd.DataFrame(uncertainty_rows).to_csv(
        OUT / "Conformal_Uncertainty_Metrics.csv", index=False, encoding="utf-8-sig"
    )
    interval_data.to_csv(
        OUT / "External_Test_Predictions_and_Intervals.csv", index=False, encoding="utf-8-sig"
    )

    # PM/HF diagnostic tables using calibration-defined concentration strata.
    diagnostic_rows, decile_rows, acf_rows = [], [], []
    for j, pollutant in [(3, "PM"), (5, "HF")]:
        q1, q2 = np.quantile(y[cal, j], [1/3, 2/3])
        strata = np.where(y[test, j] <= q1, "Low", np.where(y[test, j] <= q2, "Medium", "High"))
        model_preds = {"HAOG-MTR": pred_test[:, j]}
        if pollutant == "PM":
            model_preds["KNN"] = knn_pm_test
        for model, pp in model_preds.items():
            for label in ["Low", "Medium", "High"]:
                ix = strata == label
                diagnostic_rows.append({
                    "pollutant": pollutant,
                    "model": model,
                    "stratum": label,
                    "calibration_lower_threshold": float(q1),
                    "calibration_upper_threshold": float(q2),
                    "n": int(ix.sum()),
                    "R2": float(r2_score(y[test, j][ix], pp[ix])),
                    "RMSE": float(mean_squared_error(y[test, j][ix], pp[ix], squared=False)),
                    "MAE": float(mean_absolute_error(y[test, j][ix], pp[ix])),
                    "bias": float(np.mean(pp[ix] - y[test, j][ix])),
                })
            residual = y[test, j] - pp
            for lag, value in residual_acf(residual).items():
                acf_rows.append({"pollutant": pollutant, "model": model, "lag_min": lag, "residual_ACF": value})
            edges = np.quantile(y[cal, j], np.linspace(0, 1, 11))
            edges[0], edges[-1] = -np.inf, np.inf
            bins = np.digitize(y[test, j], edges[1:-1], right=True) + 1
            for b in range(1, 11):
                ix = bins == b
                if not np.any(ix):
                    continue
                decile_rows.append({
                    "pollutant": pollutant,
                    "model": model,
                    "concentration_decile": b,
                    "n": int(ix.sum()),
                    "observed_mean": float(np.mean(y[test, j][ix])),
                    "MAE": float(np.mean(np.abs(residual[ix]))),
                    "bias": float(np.mean(pp[ix] - y[test, j][ix])),
                })
    pd.DataFrame(diagnostic_rows).to_csv(
        OUT / "PM_HF_Concentration_Strata_Metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(decile_rows).to_csv(
        OUT / "PM_HF_Error_by_Concentration_Decile.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(acf_rows).to_csv(
        OUT / "PM_HF_Residual_Autocorrelation.csv", index=False, encoding="utf-8-sig"
    )

    external_pred = pd.DataFrame({"time": times[test].astype(str)})
    for j, pollutant in enumerate(POLLUTANTS):
        external_pred[f"observed_{pollutant}"] = y[test, j]
        external_pred[f"predicted_{pollutant}"] = pred_test[:, j]
        external_pred[f"residual_{pollutant}"] = y[test, j] - pred_test[:, j]
    external_pred["predicted_PM_KNN"] = knn_pm_test
    external_pred["residual_PM_KNN"] = y[test, 3] - knn_pm_test
    external_pred.to_csv(
        OUT / "External_Test_Observed_Predicted_Residuals.csv", index=False, encoding="utf-8-sig"
    )

    audit = {
        "experiment": "full-period forward validation, PM/HF diagnostics, and split conformal uncertainty",
        "source_model_store": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "original_haog_implementation": str(ORIGINAL),
        "original_script_sha256": sha256(ORIGINAL),
        "raw_feature_store_samples": int(len(saved["target_times"])),
        "usable_samples_after_15_min_warmup": int(len(y)),
        "usable_period_start": str(times.min()),
        "usable_period_end": str(times.max()),
        "forward_windows": feature_audit,
        "final_core_train": {"start": str(times[core].min()), "end": str(times[core].max()), "n": int(len(core))},
        "conformal_calibration": {"start": str(times[cal].min()), "end": str(times[cal].max()), "n": int(len(cal))},
        "external_test": {"start": str(times[test].min()), "end": str(times[test].max()), "n": int(len(test))},
        "final_selected_features": selected_names,
        "final_gate_parameters": {POLLUTANTS[j]: weights[j].tolist() for j in range(6)},
        "target_preprocessing": "causal weighted four-sample signal; weights 1,1,1,0.60; 15-minute warm-up removed",
        "feature_selection": "training-only ExtraTrees ranking: six current pollutant inputs plus twelve other base inputs; all 102 causal autoregressive features retained; 120 total",
        "external_metric_CI": "1000-replicate daily-block bootstrap",
        "forward_summary_CI": "Student-t CI across five forward folds",
        "conformal_method": "chronological split conformal using absolute calibration residuals and finite-sample corrected quantile",
        "seed": SEED,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "Full_Period_Validation_Audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(ORIGINAL, OUT / "original_HAOG_MTR_implementation.py")
    shutil.copy2(Path(__file__), OUT / "run_full_period_validation.py")
    print(f"All experiments completed in {time.time()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()

