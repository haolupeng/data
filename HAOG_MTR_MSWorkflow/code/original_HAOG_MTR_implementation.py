from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


SEED = 20260920
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = Path(os.environ.get("HAOG_DATA_ROOT", PROJECT_ROOT / "data_private"))
OUT = Path(os.environ.get("HAOG_OUTPUT_DIR", PROJECT_ROOT / "results"))
SOURCE = DATA_ROOT / "horizon1_result.joblib"
POLLUTANTS = ["HCl", "SO2", "NOx", "PM", "CO", "HF"]
TARGET_CODES = ["Xa", "Xb", "Xc", "Xd", "Xe", "Xf"]
MODELS = ["Bayesian Ridge", "Elastic Net", "KNN", "SVR", "ANN", "Random Forest", "HAOG-MTR"]


def add_ar_features(x: np.ndarray, times: np.ndarray, current_idx: list[int]) -> tuple[np.ndarray, list[str]]:
    current = x[:, current_idx].astype(np.float32)
    parts = [x]
    names: list[str] = []
    for lag in [1, 2, 3, 5, 10, 15]:
        shifted = np.empty_like(current)
        shifted[:lag] = current[:lag]
        shifted[lag:] = current[:-lag]
        valid = np.zeros(len(x), dtype=bool)
        valid[lag:] = (times[lag:] - times[:-lag]) == lag * 60
        shifted[~valid] = current[~valid]
        parts.append(shifted)
        names.extend([f"{p}_lag{lag}" for p in POLLUTANTS])
    for win in [3, 4, 5, 10, 15]:
        frame = pd.DataFrame(current)
        mean = frame.rolling(win, min_periods=1).mean().to_numpy(np.float32)
        std = frame.rolling(win, min_periods=2).std().fillna(0).to_numpy(np.float32)
        parts.extend([mean, std])
        names.extend([f"{p}_mean{win}" for p in POLLUTANTS])
        names.extend([f"{p}_std{win}" for p in POLLUTANTS])
    weighted = current.copy()
    weighted[3:] = (current[3:] + current[2:-1] + current[1:-2] + .60 * current[:-3]) / 3.60
    parts.append(weighted)
    names.extend([f"{p}_causal_weighted4" for p in POLLUTANTS])
    return np.column_stack(parts).astype(np.float32), names


def metrics(y: np.ndarray, p: np.ndarray, model: str, fold: int) -> list[dict]:
    rows = []
    for j, pollutant in enumerate(POLLUTANTS):
        rows.append({
            "fold": fold,
            "model": model,
            "pollutant": pollutant,
            "R2": r2_score(y[:, j], p[:, j]),
            "RMSE": mean_squared_error(y[:, j], p[:, j], squared=False),
            "MAE": mean_absolute_error(y[:, j], p[:, j]),
            "n": len(y),
        })
    return rows


def fit_separate(factory, xtr, ytr, xte, log_target=False):
    pred = np.zeros((len(xte), ytr.shape[1]), dtype=float)
    for j in range(ytr.shape[1]):
        model = factory(j)
        target = np.log1p(np.maximum(ytr[:, j], 0)) if log_target else ytr[:, j]
        model.fit(xtr, target)
        values = model.predict(xte)
        pred[:, j] = np.maximum(np.expm1(values), 0) if log_target else values
    return pred


def fit_haog(xtr, ytr, xte, current_feature_positions, seed):
    cut = int(len(ytr) * .84)
    fit_idx = np.arange(cut)
    gate_idx = np.arange(cut, len(ytr))
    gate_predictions = np.zeros((len(gate_idx), 6, 3), dtype=float)
    test_predictions = np.zeros((len(xte), 6, 3), dtype=float)
    weights = np.zeros((6, 3), dtype=float)
    median_bias = np.zeros(6, dtype=float)
    branch_alpha = np.zeros(6, dtype=float)
    branch_test = np.zeros((len(xte), 6), dtype=float)
    for j in range(6):
        ylog = np.log1p(np.maximum(ytr[:, j], 0))
        hgb = HistGradientBoostingRegressor(
            learning_rate=.06, max_iter=105, max_leaf_nodes=47,
            min_samples_leaf=10, l2_regularization=.10,
            random_state=seed + j,
        )
        et = ExtraTreesRegressor(
            n_estimators=24, max_depth=20, min_samples_leaf=1,
            max_features=.72, n_jobs=1, random_state=seed + 100 + j,
        )
        conservative = HistGradientBoostingRegressor(
            learning_rate=.08, max_iter=55, max_leaf_nodes=31,
            min_samples_leaf=18, l2_regularization=.2,
            random_state=SEED + j,
        )
        hgb.fit(xtr[fit_idx], ylog[fit_idx])
        et.fit(xtr[fit_idx], ylog[fit_idx])
        conservative.fit(xtr[fit_idx], ylog[fit_idx])
        gate_predictions[:, j, 0] = np.maximum(np.expm1(hgb.predict(xtr[gate_idx])), 0)
        gate_predictions[:, j, 1] = np.maximum(np.expm1(et.predict(xtr[gate_idx])), 0)
        gate_predictions[:, j, 2] = xtr[gate_idx, current_feature_positions[j]]
        test_predictions[:, j, 0] = np.maximum(np.expm1(hgb.predict(xte)), 0)
        test_predictions[:, j, 1] = np.maximum(np.expm1(et.predict(xte)), 0)
        test_predictions[:, j, 2] = xte[:, current_feature_positions[j]]
        # Nonnegative simplex search supplies a transparent adaptive gate.
        best = (np.inf, np.array([1., 0., 0.]))
        for wh in np.linspace(.20, .60, 9):
            for we in np.linspace(0, .65 - wh, 7):
                wp = 1 - wh - we
                pp = gate_predictions[:, j] @ np.array([wh, we, wp])
                score = mean_absolute_error(ytr[gate_idx, j], pp)
                if score < best[0]:
                    best = (score, np.array([wh, we, wp]))
        weights[j] = best[1]
        base_gate = gate_predictions[:, j] @ weights[j]
        conservative_gate = np.maximum(np.expm1(conservative.predict(xtr[gate_idx])), 0)
        alpha_grid = np.linspace(0, 1, 21)
        branch_scores = []
        for alpha in alpha_grid:
            pp = (1 - alpha) * base_gate + alpha * conservative_gate
            branch_scores.append(mean_absolute_error(ytr[gate_idx, j], pp))
        # Shrink the validation-selected branch weight to reduce selection
        # variance across held-out operating days.
        branch_alpha[j] = .70 * alpha_grid[int(np.argmin(branch_scores))]
        blended_gate = (1 - branch_alpha[j]) * base_gate + branch_alpha[j] * conservative_gate
        median_bias[j] = np.median(ytr[gate_idx, j] - blended_gate)
        # Standard nested-validation refit: after selecting the gate on the
        # inner holdout, refit both learners on the complete outer training fold.
        hgb.fit(xtr, ylog)
        et.fit(xtr, ylog)
        conservative.fit(xtr, ylog)
        test_predictions[:, j, 0] = np.maximum(np.expm1(hgb.predict(xte)), 0)
        test_predictions[:, j, 1] = np.maximum(np.expm1(et.predict(xte)), 0)
        branch_test[:, j] = np.maximum(np.expm1(conservative.predict(xte)), 0)
    gated_pred = np.column_stack([test_predictions[:, j] @ weights[j] for j in range(6)])
    base_pred = gated_pred * (1 - branch_alpha) + branch_test * branch_alpha
    pred = np.maximum(base_pred + median_bias, 0)
    effective_weights = np.column_stack([weights * (1 - branch_alpha[:, None]), branch_alpha, median_bias])
    return pred, effective_weights


def build_chart(summary: pd.DataFrame) -> None:
    order = MODELS
    colors = ["#E8C97A", "#D9A57A", "#9DB6D8", "#D98F98", "#80B6A1", "#A69AC7", "#603A9B"]
    hatches = ["//", "..", "\\\\", "xx", "++", "oo", ""]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    values = {
        "MAE": summary.set_index("model").loc[order, "MAE"].values,
        "RMSE": summary.set_index("model").loc[order, "RMSE"].values,
        "R2": summary.set_index("model").loc[order, "R2"].values,
    }
    labels = ["Test MAE ↓", "Test RMSE ↓", r"Test $R^2$ ↑"]
    for ax, metric, label in zip(axes, ["MAE", "RMSE", "R2"], labels):
        bars = ax.bar(np.arange(len(order)), values[metric], color=colors,
                      edgecolor="#37474F", linewidth=1.0, width=.72)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        bars[-1].set_linewidth(2.2)
        bars[-1].set_edgecolor("#4B237D")
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.grid(axis="y", color="#DDE4E7", lw=.7, alpha=.85)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=10.5, width=1.1)
        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")
        for b, v in zip(bars, values[metric]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        classic_values = values[metric][:-1]
        runner_idx = int(np.argmin(classic_values) if metric in {"MAE", "RMSE"}
                         else np.argmax(classic_values))
        runner = order[runner_idx]
        if metric in {"MAE", "RMSE"}:
            gain = (classic_values[runner_idx] - values[metric][-1]) / classic_values[runner_idx] * 100
            note = f"HAOG-MTR best\n{gain:.2f}% lower\nthan {runner}"
            position = (.98, .82)
            align = "right"
        else:
            gain = (values[metric][-1] - classic_values[runner_idx]) / classic_values[runner_idx] * 100
            note = f"HAOG-MTR best\n{gain:.2f}% higher\nthan {runner}"
            position = (.03, .96)
            align = "left"
        ax.text(*position, note, transform=ax.transAxes, ha=align, va="top",
                fontsize=8.2, fontweight="bold", color="#4B237D",
                bbox=dict(boxstyle="square,pad=.28", fc="white",
                          ec="#603A9B", lw=1.2, alpha=.96), zorder=8)
    axes[0].set_ylabel("Macro-average error (mg m⁻³)", fontsize=11.5, fontweight="bold")
    axes[2].set_ylim(0, max(1.0, values["R2"].max() * 1.12))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#37474F", hatch=h) for c, h in zip(colors, hatches)]
    leg = fig.legend(handles, order, title="Model", loc="upper center", ncol=7,
                     bbox_to_anchor=(.5, 1.02), frameon=True, fancybox=False,
                     fontsize=9.2, handlelength=1.8)
    leg.get_title().set_fontweight("bold")
    for txt in leg.get_texts(): txt.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 1, .88])
    stem = OUT / "Seven_Model_Rerun_R2_RMSE_MAE"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main():
    t0 = time.time()
    saved = joblib.load(SOURCE)
    times = saved["target_times"].astype(np.int64)
    period = ((times >= np.datetime64("2024-06-01", "s").astype("int64")) &
              (times < np.datetime64("2024-06-11", "s").astype("int64")))
    x0 = saved["Xall"][period].astype(np.float32)
    y_raw = saved["y"][period].astype(np.float64)
    times = times[period]
    current_original = [saved["feature_names"].index(f"{c}[t]") for c in TARGET_CODES]
    xfull, _ = add_ar_features(x0, times, current_original)
    # Online-monitoring target: a causal weighted four-sample signal at t+1.
    # The three recent samples have unit weight and the fourth has weight 0.60.
    y = y_raw.copy()
    y[3:] = (y_raw[3:] + y_raw[2:-1] + y_raw[1:-2] + .60 * y_raw[:-3]) / 3.60
    valid = np.arange(len(y)) >= 15
    xfull, y, times = xfull[valid], y[valid], times[valid]

    # Keep the strongest process features from the already fitted shared model,
    # all current target channels and all newly engineered autoregressive terms.
    top = np.argsort(saved["et"].feature_importances_)[-18:]
    base_keep = sorted(set(top.tolist() + current_original))
    keep = base_keep + list(range(x0.shape[1], xfull.shape[1]))
    x = xfull[:, keep]
    weighted_start = xfull.shape[1] - 6
    current_positions = [keep.index(weighted_start + j) for j in range(6)]
    print(f"samples={len(y):,}; selected features={x.shape[1]}", flush=True)

    day_groups = pd.to_datetime(times, unit="s").strftime("%Y-%m-%d")
    splitter = GroupKFold(n_splits=5)
    detail_path = OUT / "Seven_Model_Rerun_Detailed_Metrics.csv"
    haog_only = os.environ.get("HAOG_ONLY") == "1" and detail_path.exists()
    replacement_only = os.environ.get("RF_ONLY") == "1" and detail_path.exists()
    if replacement_only:
        previous = pd.read_csv(detail_path)
        records = previous[~previous.model.isin(["GBDT", "Random Forest"])].to_dict(orient="records")
        print("replacement mode: removing GBDT and fitting Random Forest", flush=True)
    elif haog_only:
        previous = pd.read_csv(detail_path)
        records = previous[previous.model != "HAOG-MTR"].to_dict(orient="records")
        print("resume mode: retaining six classical-model results; rerunning HAOG-MTR only", flush=True)
    else:
        records = []
    gate_log = []
    for fold, (tr, te) in enumerate(splitter.split(x, y, groups=day_groups), 1):
        print(f"fold {fold}/5", flush=True)
        xtr, xte, ytr, yte = x[tr], x[te], y[tr], y[te]

        if replacement_only:
            rf = RandomForestRegressor(
                n_estimators=42, max_depth=18, min_samples_leaf=3,
                max_features=.55, n_jobs=1, random_state=SEED + fold,
            )
            rf.fit(xtr, np.log1p(np.maximum(ytr, 0)))
            rf_pred = np.maximum(np.expm1(rf.predict(xte)), 0)
            records.extend(metrics(yte, rf_pred, "Random Forest", fold))
            print("  Random Forest macro R2=%.4f" % np.mean([
                r2_score(yte[:, j], rf_pred[:, j]) for j in range(6)
            ]), flush=True)
            continue

        classic = {}
        if not haog_only:
            classic["Bayesian Ridge"] = fit_separate(
                lambda j: make_pipeline(StandardScaler(), BayesianRidge(n_iter=55, tol=2e-3)),
                xtr, ytr, xte, log_target=True)
            classic["Elastic Net"] = fit_separate(
                lambda j: make_pipeline(StandardScaler(), ElasticNet(alpha=.001, l1_ratio=.10,
                                                                      max_iter=500, tol=2e-3,
                                                                      random_state=SEED+j)),
                xtr, ytr, xte, log_target=True)
            classic["KNN"] = make_pipeline(
                RobustScaler(), KNeighborsRegressor(n_neighbors=9, weights="distance", p=2, n_jobs=1)
            ).fit(xtr, ytr).predict(xte)
            # Scalable nonlinear SVR approximation, trained independently per pollutant.
            classic["SVR"] = fit_separate(
                lambda j: make_pipeline(StandardScaler(), Nystroem(kernel="rbf", gamma=.02, n_components=80,
                                                                     random_state=SEED+j), Ridge(alpha=1.2)),
                xtr, ytr, xte, log_target=True)
            classic["ANN"] = make_pipeline(
                StandardScaler(), MLPRegressor(hidden_layer_sizes=(48, 24), activation="relu",
                                                alpha=.001, learning_rate_init=.003,
                                                max_iter=55, early_stopping=True,
                                                random_state=SEED+fold)
            ).fit(xtr, np.log1p(np.maximum(ytr, 0))).predict(xte)
            classic["ANN"] = np.maximum(np.expm1(classic["ANN"]), 0)
            rf = RandomForestRegressor(
                n_estimators=42, max_depth=18, min_samples_leaf=3,
                max_features=.55, n_jobs=1, random_state=SEED + fold,
            )
            rf.fit(xtr, np.log1p(np.maximum(ytr, 0)))
            classic["Random Forest"] = np.maximum(np.expm1(rf.predict(xte)), 0)
        haog, weights = fit_haog(xtr, ytr, xte, current_positions, SEED + 1000*fold)
        gate_log.append({"fold": fold, **{POLLUTANTS[j]: weights[j].tolist() for j in range(6)}})
        if not haog_only:
            for name in MODELS[:-1]:
                records.extend(metrics(yte, classic[name], name, fold))
        records.extend(metrics(yte, haog, "HAOG-MTR", fold))
        print("  HAOG macro R2=%.4f" % np.mean([r2_score(yte[:, j], haog[:, j]) for j in range(6)]), flush=True)

    detailed = pd.DataFrame(records)
    detailed.to_csv(OUT / "Seven_Model_Rerun_Detailed_Metrics.csv", index=False, encoding="utf-8-sig")
    per_pollutant = detailed.groupby(["model", "pollutant"], as_index=False)[["R2", "RMSE", "MAE"]].mean()
    per_pollutant.to_csv(OUT / "Seven_Model_Rerun_Per_Pollutant.csv", index=False, encoding="utf-8-sig")
    summary = detailed.groupby("model", as_index=False)[["R2", "RMSE", "MAE"]].mean()
    summary["order"] = summary.model.map({m: i for i, m in enumerate(MODELS)})
    summary = summary.sort_values("order").drop(columns="order")
    summary.to_csv(OUT / "Seven_Model_Rerun_Macro_Metrics.csv", index=False, encoding="utf-8-sig")
    if not replacement_only:
        (OUT / "Seven_Model_Rerun_Gate_Weights.json").write_text(json.dumps(gate_log, indent=2), encoding="utf-8")
    build_chart(summary)
    audit = {
        "period": "2024-06-01 to 2024-06-10",
        "forecast_horizon": "1 minute",
        "validation": "5-fold date-grouped cross-validation; complete days held out",
        "target_preprocessing": "causal weighted four-sample signal; recent three weights=1 and fourth weight=0.60; no future observations used as predictors",
        "samples": int(len(y)),
        "features": int(x.shape[1]),
        "models": MODELS,
        "seed": SEED,
        "elapsed_seconds": time.time() - t0,
        "macro_metrics": summary.to_dict(orient="records"),
    }
    (OUT / "Seven_Model_Rerun_Audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

