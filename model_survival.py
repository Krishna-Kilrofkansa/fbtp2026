"""
model_survival.py — Survival Analysis for Event Duration (Method B)

The single biggest differentiator vs. other teams:
- Treats active events as RIGHT-CENSORED, not missing
- Uses proper time-to-event models (Weibull AFT, CoxPH)
- Outputs expected resolution time + confidence intervals
- Handles the 21.3% of events that hit the 24h duration cap

Why this matters:
    Most teams will either:
    1. Drop the 937 active events (biasing toward short durations), or
    2. Run naive linear regression on duration (ignoring censoring)
    
    Both are statistically wrong. Survival analysis is the correct framework
    for "time until event" data with incomplete observations.

Usage:
    from model_survival import train_survival_models, predict_duration
    results = train_survival_models(train_df, test_df)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DURATION_CAP = 1440  # 24 hours in minutes

# Features for survival model — subset that's most relevant to duration
SURVIVAL_FEATURES = [
    "event_cause", "corridor_clean", "priority_binary",
    "requires_road_closure_int", "hour", "day_of_week",
    "is_weekend", "is_rush_hour", "is_on_corridor",
    "events_in_corridor_1h", "events_in_corridor_24h",
    "cause_severity",
]


# ──────────────────────────────────────────────────────────────────────
# Data preparation for survival analysis
# ──────────────────────────────────────────────────────────────────────

def prepare_survival_data(df):
    """
    Prepare data for survival analysis.
    
    Key decisions:
    1. Duration = time from start to close/resolve (for observed events)
    2. For censored events (active): duration = time from start to last observation
    3. Event indicator: 1 = observed (closed/resolved), 0 = censored (active)
    4. Cap at 24h — events beyond this are likely administrative, not operational
    5. Events with 0 or negative duration are excluded
    
    Returns:
        DataFrame with columns: duration_hours, event_observed, + features
    """
    sdf = df.copy()
    
    # Duration in hours (more interpretable than minutes for survival curves)
    sdf["duration_hours"] = sdf["duration_minutes_capped"] / 60.0
    
    # Event indicator: 1 = closed/resolved, 0 = still active (censored)
    sdf["event_observed"] = (sdf["is_censored"] == 0).astype(int)
    
    # Convert boolean to int for the model
    sdf["requires_road_closure_int"] = sdf["requires_road_closure"].astype(int)
    
    # Filter: need valid duration > 0
    valid_mask = sdf["duration_hours"].notna() & (sdf["duration_hours"] > 0)
    sdf = sdf[valid_mask].copy()
    
    print(f"[survival_prep] {len(sdf)} events with valid duration")
    print(f"  Observed (closed/resolved): {sdf['event_observed'].sum()}")
    print(f"  Censored (active):          {(sdf['event_observed'] == 0).sum()}")
    print(f"  Duration range: {sdf['duration_hours'].min():.2f}h - {sdf['duration_hours'].max():.2f}h")
    print(f"  Median duration: {sdf['duration_hours'].median():.2f}h")
    
    return sdf


def encode_survival_features(df, fit_encoders=None):
    """
    One-hot encode categoricals for survival models.
    lifelines doesn't handle categoricals natively like LightGBM,
    so we need explicit encoding.
    
    Returns:
        X: Feature DataFrame (numeric only)
        encoders: dict of category mappings (for reuse on test data)
    """
    X = pd.DataFrame(index=df.index)
    encoders = fit_encoders or {}
    
    cat_cols = ["event_cause", "corridor_clean"]
    num_cols = [c for c in SURVIVAL_FEATURES if c not in cat_cols and c in df.columns]
    
    # Add numeric features directly
    for col in num_cols:
        X[col] = df[col].fillna(0).astype(float)
    
    # One-hot encode categoricals (top N categories + other)
    for col in cat_cols:
        if col not in df.columns:
            continue
            
        if col in encoders:
            # Reuse existing categories from training
            top_cats = encoders[col]
        else:
            # Fit: keep top 8 categories, collapse rest to "other"
            top_cats = df[col].value_counts().head(8).index.tolist()
            encoders[col] = top_cats
        
        for cat in top_cats:
            X[f"{col}_{cat}"] = (df[col] == cat).astype(int)
    
    return X, encoders


# ──────────────────────────────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────────────────────────────

def train_survival_models(train_df, test_df, save_path=None):
    """
    Train three survival models and compare:
    1. Weibull AFT — parametric, most interpretable
    2. Log-Normal AFT — often fits real-world durations well
    3. Cox PH — semi-parametric, most flexible
    
    Returns dict with models, metrics, and predictions.
    """
    from lifelines import WeibullAFTFitter, LogNormalAFTFitter, CoxPHFitter
    from lifelines.utils import concordance_index
    
    print("\n" + "=" * 60)
    print("METHOD B — Survival Analysis (Duration/Impact)")
    print("=" * 60)
    
    # Prepare data
    train_surv = prepare_survival_data(train_df)
    test_surv = prepare_survival_data(test_df)
    
    X_train, encoders = encode_survival_features(train_surv)
    X_test, _ = encode_survival_features(test_surv, fit_encoders=encoders)
    
    # Add duration and event columns
    X_train["duration_hours"] = train_surv["duration_hours"].values
    X_train["event_observed"] = train_surv["event_observed"].values
    X_test["duration_hours"] = test_surv["duration_hours"].values
    X_test["event_observed"] = test_surv["event_observed"].values
    
    # Align columns between train and test
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Ensure duration and event columns are present
    assert "duration_hours" in X_train.columns
    assert "event_observed" in X_train.columns
    
    feature_cols = [c for c in common_cols if c not in ["duration_hours", "event_observed"]]
    
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  Train: {len(X_train)} rows")
    print(f"  Test:  {len(X_test)} rows")
    
    results = {}
    
    # ── Model 1: Weibull AFT ──
    print("\n  --- Weibull AFT ---")
    try:
        wf = WeibullAFTFitter(penalizer=0.05)
        wf.fit(X_train, duration_col="duration_hours", event_col="event_observed")
        
        # Concordance index on test
        pred_median_w = wf.predict_median(X_test[feature_cols])
        ci_w = concordance_index(
            X_test["duration_hours"],
            pred_median_w,
            X_test["event_observed"]
        )
        
        # MAE on observed events only (can't measure error on censored)
        observed_mask = X_test["event_observed"] == 1
        if observed_mask.sum() > 0:
            mae_w = np.abs(
                pred_median_w[observed_mask] - X_test.loc[observed_mask, "duration_hours"]
            ).median()
        else:
            mae_w = np.nan
        
        print(f"  C-index: {ci_w:.4f}")
        print(f"  Median AE (observed only): {mae_w:.2f}h")
        print(f"  AIC: {wf.AIC_:.1f}")
        
        results["weibull"] = {
            "model": wf,
            "c_index": float(ci_w),
            "median_ae": float(mae_w),
            "aic": float(wf.AIC_),
        }
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # ── Model 2: Log-Normal AFT ──
    print("\n  --- Log-Normal AFT ---")
    try:
        lnf = LogNormalAFTFitter(penalizer=0.05)
        lnf.fit(X_train, duration_col="duration_hours", event_col="event_observed")
        
        pred_median_ln = lnf.predict_median(X_test[feature_cols])
        ci_ln = concordance_index(
            X_test["duration_hours"],
            pred_median_ln,
            X_test["event_observed"]
        )
        
        if observed_mask.sum() > 0:
            mae_ln = np.abs(
                pred_median_ln[observed_mask] - X_test.loc[observed_mask, "duration_hours"]
            ).median()
        else:
            mae_ln = np.nan
        
        print(f"  C-index: {ci_ln:.4f}")
        print(f"  Median AE (observed only): {mae_ln:.2f}h")
        print(f"  AIC: {lnf.AIC_:.1f}")
        
        results["lognormal"] = {
            "model": lnf,
            "c_index": float(ci_ln),
            "median_ae": float(mae_ln),
            "aic": float(lnf.AIC_),
        }
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # ── Model 3: Cox PH ──
    print("\n  --- Cox Proportional Hazards ---")
    try:
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(X_train, duration_col="duration_hours", event_col="event_observed")
        
        ci_cox = cph.score(X_test, scoring_method="concordance_index")
        
        print(f"  C-index: {ci_cox:.4f}")
        
        # Cox doesn't give direct point predictions easily,
        # but the C-index tells us ranking quality
        results["cox"] = {
            "model": cph,
            "c_index": float(ci_cox),
        }
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # ── Select best model ──
    print("\n  --- Model Comparison ---")
    best_name = None
    best_ci = 0
    for name, res in results.items():
        ci = res.get("c_index", 0)
        aic = res.get("aic", "N/A")
        mae = res.get("median_ae", "N/A")
        print(f"    {name:12s}  C-index={ci:.4f}  AIC={aic}  MedAE={mae}")
        if ci > best_ci:
            best_ci = ci
            best_name = name
    
    print(f"\n  Best model: {best_name} (C-index: {best_ci:.4f})")
    results["best"] = best_name
    
    # ── Print covariate summary for best AFT model ──
    best_model = results[best_name]["model"]
    if hasattr(best_model, "summary"):
        print(f"\n  === {best_name.upper()} — Significant Covariates ===")
        summary = best_model.summary
        # Show only significant covariates (p < 0.05)
        significant = summary[summary["p"] < 0.05].copy()
        if len(significant) > 0:
            significant = significant.sort_values("p")
            for idx, row in significant.head(15).iterrows():
                coef = row["coef"]
                p = row["p"]
                # For AFT: positive coef = longer duration
                direction = "LONGER" if coef > 0 else "SHORTER"
                var_name = idx[1] if isinstance(idx, tuple) else idx
                print(f"    {var_name:40s}  coef={coef:+.4f}  p={p:.4f}  ({direction})")
    
    # ── Generate predictions for test set ──
    if best_name in ["weibull", "lognormal"]:
        best_aft = results[best_name]["model"]
        
        # Predict median + confidence band
        # NOTE: In lifelines, predict_percentile(p) gives the time at which
        # survival probability = p. So p=0.9 → 10th percentile of failure time (early),
        # p=0.5 → median, p=0.1 → 90th percentile (late).
        pred_median = best_aft.predict_median(X_test[feature_cols])
        pred_10 = best_aft.predict_percentile(X_test[feature_cols], p=0.9)   # 10th pctile (fast)
        pred_90 = best_aft.predict_percentile(X_test[feature_cols], p=0.1)   # 90th pctile (slow)
        
        results["test_predictions"] = pd.DataFrame({
            "actual_hours": X_test["duration_hours"].values,
            "event_observed": X_test["event_observed"].values,
            "predicted_median_hours": pred_median.values,
            "predicted_10th_hours": pred_10.values,
            "predicted_90th_hours": pred_90.values,
        })
        
        # Prediction quality stats (observed events only)
        obs = results["test_predictions"][results["test_predictions"]["event_observed"] == 1]
        if len(obs) > 0:
            abs_err = np.abs(obs["predicted_median_hours"] - obs["actual_hours"])
            print(f"\n  === Prediction Quality (observed events, n={len(obs)}) ===")
            print(f"    Median Absolute Error: {abs_err.median():.2f}h")
            print(f"    Mean Absolute Error:   {abs_err.mean():.2f}h")
            # Calibration: what % of actuals fall within 10-90th predicted interval?
            in_interval = (
                (obs["actual_hours"] >= obs["predicted_10th_hours"]) &
                (obs["actual_hours"] <= obs["predicted_90th_hours"])
            ).mean()
            print(f"    Calibration (actual in 10-90th interval): {in_interval:.1%}")
            results["calibration_80"] = float(in_interval)
    
    # ── Save ──
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_out = {}
        for name, res in results.items():
            if isinstance(res, dict) and "model" in res:
                metrics_out[name] = {k: v for k, v in res.items() if k != "model"}
        metrics_out["best"] = best_name
        
        with open(save_path / "survival_metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2, default=str)
        
        if "test_predictions" in results:
            results["test_predictions"].to_csv(
                save_path / "survival_predictions.csv", index=False
            )
        
        print(f"\n  Results saved to {save_path}")
    
    results["encoders"] = encoders
    results["feature_cols"] = feature_cols
    
    return results


# ──────────────────────────────────────────────────────────────────────
# Prediction function for new events
# ──────────────────────────────────────────────────────────────────────

def predict_duration(event_df, model, encoders, feature_cols):
    """
    Predict expected resolution time for new events.
    
    Args:
        event_df: DataFrame with event features
        model: Trained AFT model (Weibull or LogNormal)
        encoders: Category encoders from training
        feature_cols: Feature column names
    
    Returns:
        DataFrame with predicted median, 25th, 75th percentile durations
    """
    # Prepare features
    event_df = event_df.copy()
    event_df["requires_road_closure_int"] = event_df["requires_road_closure"].astype(int)
    
    X, _ = encode_survival_features(event_df, fit_encoders=encoders)
    
    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]
    
    predictions = pd.DataFrame({
        "predicted_median_hours": model.predict_median(X).values,
        "predicted_10th_hours": model.predict_percentile(X, p=0.9).values,   # fast resolution
        "predicted_90th_hours": model.predict_percentile(X, p=0.1).values,   # slow resolution
    })
    
    # Convert to minutes for operational use
    predictions["predicted_median_minutes"] = predictions["predicted_median_hours"] * 60
    predictions["predicted_10th_minutes"] = predictions["predicted_10th_hours"] * 60
    predictions["predicted_90th_minutes"] = predictions["predicted_90th_hours"] * 60
    
    return predictions


# ──────────────────────────────────────────────────────────────────────
# Kaplan-Meier analysis by group (for EDA / notebook)
# ──────────────────────────────────────────────────────────────────────

def kaplan_meier_by_group(df, group_col, duration_col="duration_hours",
                          event_col="event_observed", top_n=5):
    """
    Fit Kaplan-Meier curves for each group and return summary stats.
    Useful for EDA: "How does resolution time differ by event_cause?"
    """
    from lifelines import KaplanMeierFitter
    
    sdf = prepare_survival_data(df)
    
    results = {}
    kmf = KaplanMeierFitter()
    
    top_groups = sdf[group_col].value_counts().head(top_n).index
    
    for group in top_groups:
        mask = sdf[group_col] == group
        group_data = sdf[mask]
        
        if len(group_data) < 10:
            continue
        
        kmf.fit(
            group_data[duration_col],
            group_data[event_col],
            label=str(group)
        )
        
        results[group] = {
            "n": len(group_data),
            "n_censored": (group_data[event_col] == 0).sum(),
            "median_survival": float(kmf.median_survival_time_),
            "mean_survival": float(kmf.survival_function_.mean().iloc[0]),
        }
        
        print(f"  {group:25s}  n={len(group_data):4d}  "
              f"censored={results[group]['n_censored']:3d}  "
              f"median={results[group]['median_survival']:.1f}h")
    
    return results


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_and_prepare_data, get_temporal_split
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    df = load_and_prepare_data(csv_path)
    train_df, test_df = get_temporal_split(df)
    
    save_dir = Path(__file__).parent / "models"
    
    # Train survival models
    results = train_survival_models(train_df, test_df, save_path=save_dir)
    
    # Kaplan-Meier by event cause
    print("\n" + "=" * 60)
    print("KAPLAN-MEIER — Median Resolution Time by Cause")
    print("=" * 60)
    km_cause = kaplan_meier_by_group(df, "event_cause", top_n=8)
    
    print("\n" + "=" * 60)
    print("KAPLAN-MEIER — Median Resolution Time by Corridor")
    print("=" * 60)
    km_corridor = kaplan_meier_by_group(df, "corridor_clean", top_n=8)
