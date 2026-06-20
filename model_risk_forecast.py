"""
model_risk_forecast.py — Spatio-Temporal Risk Forecasting (Method C)

The PROACTIVE layer: predict corridor risk BEFORE events happen.

Instead of reacting to a single logged event, this model answers:
    "What is the expected event volume and severity for each corridor
     in the next time window, given historical patterns and recent load?"

This is what actually enables pre-positioning manpower — the literal
pain point from the problem statement.

Approach:
    1. Aggregate events to a corridor × time_slot grid
    2. For each cell, compute: event count, mean severity, closure rate
    3. Model with Negative Binomial GLM (handles count overdispersion)
       + LightGBM for comparison
    4. Output: risk score per corridor per time slot

Usage:
    from model_risk_forecast import train_risk_model, predict_corridor_risk
    results = train_risk_model(train_df, test_df)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Time slots: 4-hour windows (6 per day)
TIME_SLOTS = {
    0: "00-04",  # midnight
    1: "04-08",  # early morning
    2: "08-12",  # morning
    3: "12-16",  # afternoon
    4: "16-20",  # evening rush
    5: "20-24",  # night
}


# ──────────────────────────────────────────────────────────────────────
# Aggregation: event-level → corridor × time_slot grid
# ──────────────────────────────────────────────────────────────────────

def aggregate_to_grid(df):
    """
    Aggregate event-level data into corridor × date × time_slot cells.
    
    Each cell represents: "What happened on corridor X during time slot Y on date Z?"
    
    Returns DataFrame with one row per (corridor, date, time_slot) cell.
    """
    agg_df = df.copy()
    
    # Assign time slot (4-hour buckets)
    agg_df["time_slot"] = (agg_df["hour"] // 4).astype(int)
    agg_df["time_slot_name"] = agg_df["time_slot"].map(TIME_SLOTS)
    agg_df["date"] = agg_df["start_datetime_ist"].dt.date
    
    # Only use named corridors (non-corridor is meaningless to aggregate)
    # But keep it as a special group for overall city-level risk
    corridors = agg_df["corridor_clean"].unique()
    
    # Group by corridor × date × time_slot
    grid = agg_df.groupby(["corridor_clean", "date", "time_slot"]).agg(
        event_count=("id", "size") if "id" in agg_df.columns else ("event_cause", "size"),
        high_priority_count=("priority_binary", "sum"),
        closure_count=("requires_road_closure", "sum"),
        mean_severity=("cause_severity", "mean"),
        
        # Dominant cause
        dominant_cause=("event_cause", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"),
        
        # Vehicle breakdown count (most common event)
        breakdown_count=("event_cause", lambda x: (x == "vehicle_breakdown").sum()),
        
        # Unique event causes
        cause_diversity=("event_cause", "nunique"),
    ).reset_index()
    
    # Add temporal features for the time slot
    grid["date"] = pd.to_datetime(grid["date"])
    grid["day_of_week"] = grid["date"].dt.dayofweek
    grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(int)
    grid["month"] = grid["date"].dt.month
    grid["time_slot_name"] = grid["time_slot"].map(TIME_SLOTS)
    
    # Derived features
    grid["high_priority_rate"] = grid["high_priority_count"] / grid["event_count"].clip(lower=1)
    grid["closure_rate"] = grid["closure_count"] / grid["event_count"].clip(lower=1)
    grid["breakdown_rate"] = grid["breakdown_count"] / grid["event_count"].clip(lower=1)
    
    # ── Lag features ──
    # For each corridor, add rolling stats from previous time slots
    grid = grid.sort_values(["corridor_clean", "date", "time_slot"])
    
    for corridor, group in grid.groupby("corridor_clean"):
        idx = group.index
        
        # Events in previous slot (lag-1)
        grid.loc[idx, "lag_1_count"] = group["event_count"].shift(1).fillna(0)
        
        # Events in same slot yesterday (lag-6, since 6 slots per day)
        grid.loc[idx, "lag_6_count"] = group["event_count"].shift(6).fillna(0)
        
        # Rolling 7-day average for same corridor
        grid.loc[idx, "rolling_7d_mean"] = (
            group["event_count"].rolling(window=42, min_periods=1).mean()  # 42 = 7 days × 6 slots
        )
        
        # Rolling 7-day max event count
        grid.loc[idx, "rolling_7d_max"] = (
            group["event_count"].rolling(window=42, min_periods=1).max()
        )
    
    grid = grid.fillna(0)
    
    print(f"[grid] Aggregated to {len(grid)} corridor×date×slot cells")
    print(f"  Corridors: {grid['corridor_clean'].nunique()}")
    print(f"  Date range: {grid['date'].min()} → {grid['date'].max()}")
    print(f"  Event count stats: mean={grid['event_count'].mean():.1f}, "
          f"max={grid['event_count'].max()}, "
          f"zeros={( grid['event_count'] == 0).sum()}")
    
    return grid


def create_complete_grid(df):
    """
    Create a COMPLETE grid with all corridor × date × time_slot combinations,
    including zero-event cells. This is important for proper count modeling.
    """
    agg_df = df.copy()
    agg_df["time_slot"] = (agg_df["hour"] // 4).astype(int)
    agg_df["date"] = agg_df["start_datetime_ist"].dt.date
    
    # Get all unique corridors (excluding non-corridor for grid — too noisy)
    named_corridors = agg_df.loc[
        agg_df["corridor_clean"] != "non_corridor", "corridor_clean"
    ].unique()
    
    # Create all dates in range
    min_date = agg_df["date"].min()
    max_date = agg_df["date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D")
    all_slots = list(range(6))
    
    # Full grid
    from itertools import product
    full_idx = pd.DataFrame(
        list(product(named_corridors, all_dates, all_slots)),
        columns=["corridor_clean", "date", "time_slot"]
    )
    
    # Aggregate actual events
    actual = aggregate_to_grid(agg_df)
    actual["date"] = pd.to_datetime(actual["date"])
    
    # Merge: left join to keep all grid cells, fill missing with 0
    grid = full_idx.merge(actual, on=["corridor_clean", "date", "time_slot"], how="left")
    
    # Fill missing counts with 0
    count_cols = ["event_count", "high_priority_count", "closure_count",
                  "breakdown_count", "cause_diversity"]
    for col in count_cols:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0).astype(int)
    
    # Fill derived features
    grid["mean_severity"] = grid["mean_severity"].fillna(0)
    grid["high_priority_rate"] = grid["high_priority_rate"].fillna(0)
    grid["closure_rate"] = grid["closure_rate"].fillna(0)
    grid["breakdown_rate"] = grid["breakdown_rate"].fillna(0)
    grid["dominant_cause"] = grid["dominant_cause"].fillna("none")
    
    # Temporal features
    grid["day_of_week"] = grid["date"].dt.dayofweek
    grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(int)
    grid["month"] = grid["date"].dt.month
    grid["time_slot_name"] = grid["time_slot"].map(TIME_SLOTS)
    
    # Re-compute lag features on the complete grid
    grid = grid.sort_values(["corridor_clean", "date", "time_slot"])
    
    for corridor, group in grid.groupby("corridor_clean"):
        idx = group.index
        grid.loc[idx, "lag_1_count"] = group["event_count"].shift(1).fillna(0)
        grid.loc[idx, "lag_6_count"] = group["event_count"].shift(6).fillna(0)
        grid.loc[idx, "rolling_7d_mean"] = (
            group["event_count"].rolling(window=42, min_periods=1).mean()
        )
        grid.loc[idx, "rolling_7d_max"] = (
            group["event_count"].rolling(window=42, min_periods=1).max()
        )
    
    grid = grid.fillna(0)
    
    print(f"[complete_grid] {len(grid)} total cells "
          f"({len(named_corridors)} corridors × {len(all_dates)} days × 6 slots)")
    print(f"  Non-zero cells: {(grid['event_count'] > 0).sum()} "
          f"({(grid['event_count'] > 0).mean()*100:.1f}%)")
    
    return grid


# ──────────────────────────────────────────────────────────────────────
# Feature matrix for risk model
# ──────────────────────────────────────────────────────────────────────

GRID_FEATURES = [
    "time_slot", "day_of_week", "is_weekend", "month",
    "lag_1_count", "lag_6_count", "rolling_7d_mean", "rolling_7d_max",
]

GRID_CATEGORICAL = ["corridor_clean"]


def prepare_grid_features(grid, target="event_count"):
    """Build X, y for the grid-level model."""
    from sklearn.preprocessing import LabelEncoder
    
    X = grid[GRID_FEATURES + GRID_CATEGORICAL].copy()
    y = grid[target].copy()
    
    # Label-encode corridor
    le = LabelEncoder()
    X["corridor_clean"] = le.fit_transform(X["corridor_clean"].astype(str))
    
    return X, y, le


# ──────────────────────────────────────────────────────────────────────
# Model training
# ──────────────────────────────────────────────────────────────────────

def train_risk_model(train_df, test_df, save_path=None):
    """
    Train corridor × time_slot risk forecasting model.
    
    Two approaches:
    1. LightGBM with Poisson objective (handles count data)
    2. Corridor baseline comparison (historical average)
    
    Returns dict with model, metrics, risk profiles.
    """
    print("\n" + "=" * 60)
    print("METHOD C — Spatio-Temporal Risk Forecasting")
    print("=" * 60)
    
    # Create complete grid from training data
    print("\n  Building training grid...")
    train_grid = create_complete_grid(train_df)
    
    print("\n  Building test grid...")
    test_grid = create_complete_grid(test_df)
    
    # Prepare features
    X_train, y_train, corridor_encoder = prepare_grid_features(train_grid)
    
    # Encode test with same encoder
    X_test = test_grid[GRID_FEATURES + GRID_CATEGORICAL].copy()
    # Handle unseen corridors
    known = set(corridor_encoder.classes_)
    X_test["corridor_clean"] = X_test["corridor_clean"].astype(str).apply(
        lambda x: x if x in known else "unknown"
    )
    if "unknown" not in corridor_encoder.classes_:
        corridor_encoder.classes_ = np.append(corridor_encoder.classes_, "unknown")
    X_test["corridor_clean"] = corridor_encoder.transform(X_test["corridor_clean"])
    y_test = test_grid["event_count"].copy()
    
    print(f"\n  Train grid: {len(X_train)} cells, mean count: {y_train.mean():.2f}")
    print(f"  Test grid:  {len(X_test)} cells, mean count: {y_test.mean():.2f}")
    
    results = {}
    
    # ── Baseline: historical corridor × slot average ──
    print("\n  --- Baseline (Historical Average) ---")
    baseline_avg = train_grid.groupby(["corridor_clean", "time_slot"])["event_count"].mean()
    test_grid_with_key = test_grid.set_index(["corridor_clean", "time_slot"])
    baseline_pred = test_grid_with_key.index.map(
        lambda x: baseline_avg.get(x, y_train.mean())
    )
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    print(f"  MAE:  {baseline_mae:.3f}")
    print(f"  RMSE: {baseline_rmse:.3f}")
    results["baseline"] = {"mae": float(baseline_mae), "rmse": float(baseline_rmse)}
    
    # ── LightGBM with Poisson objective ──
    print("\n  --- LightGBM (Poisson) ---")
    params = {
        "objective": "poisson",
        "metric": "poisson",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train,
                              categorical_feature=["corridor_clean"],
                              free_raw_data=False)
    valid_data = lgb.Dataset(X_test, label=y_test,
                              categorical_feature=["corridor_clean"],
                              free_raw_data=False,
                              reference=train_data)
    
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30),
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    
    y_pred = model.predict(X_test)
    lgb_mae = mean_absolute_error(y_test, y_pred)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"  MAE:  {lgb_mae:.3f}")
    print(f"  RMSE: {lgb_rmse:.3f}")
    print(f"  Improvement over baseline: {(1 - lgb_mae/baseline_mae)*100:.1f}% MAE reduction")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": GRID_FEATURES + GRID_CATEGORICAL,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    
    print("\n  Feature importance:")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:25s} {row['importance']:,.0f}")
    
    results["lgb_poisson"] = {
        "model": model,
        "mae": float(lgb_mae),
        "rmse": float(lgb_rmse),
        "improvement_pct": float((1 - lgb_mae/baseline_mae)*100),
        "importance": importance,
    }
    
    # ── Build corridor risk profiles ──
    print("\n  --- Corridor Risk Profiles ---")
    risk_profiles = build_risk_profiles(train_grid, model, corridor_encoder)
    results["risk_profiles"] = risk_profiles
    
    # ── Top risk corridors for each time slot ──
    print("\n  Top-risk corridors by time slot:")
    for slot_id, slot_name in TIME_SLOTS.items():
        slot_risks = risk_profiles[risk_profiles["time_slot"] == slot_id]
        top = slot_risks.nlargest(3, "predicted_risk")
        corridors_str = ", ".join(
            f"{r['corridor_clean']}({r['predicted_risk']:.1f})"
            for _, r in top.iterrows()
        )
        print(f"    {slot_name}: {corridors_str}")
    
    results["corridor_encoder"] = corridor_encoder
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        
        model.save_model(str(save_path / "risk_model.txt"))
        risk_profiles.to_csv(str(save_path / "risk_profiles.csv"), index=False)
        
        metrics_out = {
            "baseline": results["baseline"],
            "lgb_poisson": {k: v for k, v in results["lgb_poisson"].items() 
                           if k not in ["model", "importance"]},
        }
        with open(save_path / "risk_metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2, default=str)
        
        print(f"\n  Results saved to {save_path}")
    
    return results


def build_risk_profiles(grid, model, corridor_encoder):
    """
    Build a risk profile for each corridor × time_slot combination.
    
    This is the "forward-looking risk map" output:
    "Mysore Road, weekday evenings, has 3x baseline risk of high-priority disruption"
    """
    corridors = grid["corridor_clean"].unique()
    profiles = []
    
    for corridor in corridors:
        if corridor == "non_corridor":
            continue
        
        for slot_id in range(6):
            # Historical stats
            mask = (grid["corridor_clean"] == corridor) & (grid["time_slot"] == slot_id)
            hist = grid[mask]
            
            if len(hist) == 0:
                continue
            
            # Build prediction input (average conditions)
            avg_features = {
                "time_slot": slot_id,
                "day_of_week": 2,  # Wednesday (typical weekday)
                "is_weekend": 0,
                "month": 3,  # March (middle of data range)
                "lag_1_count": hist["event_count"].mean(),
                "lag_6_count": hist["event_count"].mean(),
                "rolling_7d_mean": hist["event_count"].mean(),
                "rolling_7d_max": hist["event_count"].max(),
                "corridor_clean": corridor,
            }
            
            pred_df = pd.DataFrame([avg_features])
            known = set(corridor_encoder.classes_)
            pred_df["corridor_clean"] = pred_df["corridor_clean"].apply(
                lambda x: x if x in known else "unknown"
            )
            pred_df["corridor_clean"] = corridor_encoder.transform(pred_df["corridor_clean"])
            
            predicted_risk = model.predict(pred_df[GRID_FEATURES + GRID_CATEGORICAL])[0]
            
            profiles.append({
                "corridor_clean": corridor,
                "time_slot": slot_id,
                "time_slot_name": TIME_SLOTS[slot_id],
                "historical_mean_events": float(hist["event_count"].mean()),
                "historical_max_events": int(hist["event_count"].max()),
                "historical_high_priority_rate": float(hist["high_priority_rate"].mean()),
                "historical_closure_rate": float(hist["closure_rate"].mean()),
                "predicted_risk": float(predicted_risk),
            })
    
    profiles_df = pd.DataFrame(profiles)
    
    # Normalize risk to 0-100 scale
    if len(profiles_df) > 0:
        risk_max = profiles_df["predicted_risk"].quantile(0.95)
        profiles_df["risk_score"] = (
            profiles_df["predicted_risk"] / max(risk_max, 0.01) * 100
        ).clip(0, 100)
    
    return profiles_df


# ──────────────────────────────────────────────────────────────────────
# Prediction for new time windows
# ──────────────────────────────────────────────────────────────────────

def predict_corridor_risk(corridor, time_slot, day_of_week, month,
                          recent_events_1h, recent_events_24h,
                          model, corridor_encoder):
    """
    Predict event risk for a specific corridor at a specific time.
    
    Args:
        corridor: Corridor name (string)
        time_slot: 0-5 (4-hour window)
        day_of_week: 0-6 (Mon-Sun)
        month: 1-12
        recent_events_1h: Events in this corridor in last hour
        recent_events_24h: Events in last 24 hours
        model: Trained LightGBM model
        corridor_encoder: LabelEncoder for corridors
    
    Returns:
        Predicted event count for this time slot
    """
    features = pd.DataFrame([{
        "time_slot": time_slot,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "month": month,
        "lag_1_count": recent_events_1h,
        "lag_6_count": recent_events_24h / 6,
        "rolling_7d_mean": recent_events_24h / 6,
        "rolling_7d_max": recent_events_24h,
        "corridor_clean": corridor,
    }])
    
    known = set(corridor_encoder.classes_)
    features["corridor_clean"] = features["corridor_clean"].apply(
        lambda x: x if x in known else "unknown"
    )
    if "unknown" not in corridor_encoder.classes_:
        corridor_encoder.classes_ = np.append(corridor_encoder.classes_, "unknown")
    features["corridor_clean"] = corridor_encoder.transform(features["corridor_clean"])
    
    return float(model.predict(features[GRID_FEATURES + GRID_CATEGORICAL])[0])


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
    
    results = train_risk_model(train_df, test_df, save_path=save_dir)
