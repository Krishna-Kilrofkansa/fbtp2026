"""
data_loader.py — Core data ingestion, cleaning, and feature engineering
for Flipkart Grid 7.0 PS2 (Event-Driven Congestion).

Usage:
    from data_loader import load_and_prepare_data, get_temporal_split
    df = load_and_prepare_data("path/to/csv")
    train_df, test_df = get_temporal_split(df)
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

CSV_FILENAME = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

# Timezone note: timestamps have +00 UTC suffix BUT the hour distribution
# peaks at 21:00 UTC / troughs at 13-16 UTC. Converting to IST (+5:30) makes
# the peak 2:30am which is WORSE. Conclusion: data is already in IST, stored
# with an incorrect +00 UTC offset. We strip the timezone and treat as IST.

# Duration outlier cap — 90th percentile is already 12 days;
# anything beyond 24h is almost certainly a data-entry / batch-close artifact
DURATION_CAP_MINUTES = 1440  # 24 hours

# Cause severity mapping (for composite impact score downstream)
CAUSE_SEVERITY = {
    "accident": 1.0,
    "protest": 0.95,
    "public_event": 0.9,
    "vip_movement": 0.85,
    "procession": 0.8,
    "tree_fall": 0.8,
    "water_logging": 0.7,
    "construction": 0.6,
    "congestion": 0.55,
    "pot_holes": 0.4,
    "road_conditions": 0.4,
    "vehicle_breakdown": 0.35,
    "others": 0.5,
}

# Temporal split boundary
TRAIN_END = pd.Timestamp("2024-02-29", tz="Asia/Kolkata")

# ──────────────────────────────────────────────────────────────────────
# Haversine distance (km) between two lat/lng points
# ──────────────────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km. Returns NaN where inputs are invalid."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# ──────────────────────────────────────────────────────────────────────
# Core loading & cleaning
# ──────────────────────────────────────────────────────────────────────

def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV with proper NULL handling."""
    df = pd.read_csv(
        csv_path,
        na_values=["NULL", "null", "None", "", "[]"],
        low_memory=False,
    )
    print(f"[load_raw] Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


def clean_core_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing critical fields, normalize categoricals."""
    initial = len(df)

    # Drop rows missing the absolute essentials
    df = df.dropna(subset=["event_cause", "start_datetime", "latitude", "longitude"])
    print(f"[clean] Dropped {initial - len(df)} rows missing core fields → {len(df)} remaining")

    # Normalize string columns
    str_cols = ["event_type", "event_cause", "status", "priority",
                "corridor", "veh_type", "zone", "police_station", "junction"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan, "null": np.nan})

    # Standardize requires_road_closure to boolean
    df["requires_road_closure"] = df["requires_road_closure"].map(
        {True: True, False: False, "TRUE": True, "FALSE": False,
         "true": True, "false": False, 1: True, 0: False}
    ).fillna(False).astype(bool)

    # Standardize priority to binary
    df["priority_binary"] = (df["priority"] == "high").astype(int)

    return df.copy()


# ──────────────────────────────────────────────────────────────────────
# Timestamp processing
# ──────────────────────────────────────────────────────────────────────

def process_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamps, investigate timezone, compute duration.
    
    Key insight: timestamps have +00 UTC suffix but the hour distribution
    peaks at 21:00 UTC and troughs at 13-16 UTC. Converting to IST (+5:30)
    would put the peak at 2:30am — nonsensical. The data is ALREADY in IST
    but stored with an incorrect +00 UTC offset. We strip the timezone
    and localize as Asia/Kolkata.
    """
    time_cols = ["start_datetime", "closed_datetime", "resolved_datetime",
                 "end_datetime", "modified_datetime", "created_date"]

    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            # Strip UTC and re-localize as IST (data is already IST with wrong offset)
            df[col] = df[col].dt.tz_localize(None).dt.tz_localize("Asia/Kolkata")

    # ── Timezone investigation log ──
    hours = df["start_datetime"].dt.hour
    peak_hour = hours.mode().iloc[0] if len(hours.mode()) > 0 else 12
    print(f"[timestamps] Peak hour (treated as IST): {peak_hour}:00")
    
    # Create IST alias (same column now, since we re-localized)
    df["start_datetime_ist"] = df["start_datetime"]

    # ── Duration computation ──
    # Use closed_datetime first, fall back to resolved_datetime
    close_time = df["closed_datetime"].fillna(df["resolved_datetime"])
    df["duration_minutes"] = (close_time - df["start_datetime"]).dt.total_seconds() / 60

    # ── Censoring flag ──
    # Active events have no close time → right-censored, NOT missing
    df["is_censored"] = (df["status"] == "active").astype(int)
    # For censored events, duration = time from start to last known observation
    # Use modified_datetime as the last-observation proxy
    if "modified_datetime" in df.columns:
        last_obs = df["modified_datetime"].fillna(df["start_datetime"])
        censored_mask = df["is_censored"] == 1
        df.loc[censored_mask, "duration_minutes"] = (
            (last_obs[censored_mask] - df.loc[censored_mask, "start_datetime"])
            .dt.total_seconds() / 60
        )

    # ── Duration cleaning ──
    # Negative durations -> data error, set to NaN
    neg_count = (df["duration_minutes"] < 0).sum()
    df.loc[df["duration_minutes"] < 0, "duration_minutes"] = np.nan
    # Very short durations (< 1 min) -> likely auto-close artifacts
    df["is_instant_close"] = (df["duration_minutes"] < 1) & (df["is_censored"] == 0)
    # Cap extreme outliers at 24h (90th percentile is already ~12 days)
    df["duration_minutes_capped"] = df["duration_minutes"].clip(upper=DURATION_CAP_MINUTES)
    df["duration_log"] = np.log1p(df["duration_minutes_capped"])
    if neg_count > 0:
        print(f"[timestamps] Fixed {neg_count} negative durations")

    # Duration stats
    valid_dur = df["duration_minutes"].dropna()
    print(f"[timestamps] Duration stats: median={valid_dur.median():.0f}min, "
          f"75th={valid_dur.quantile(0.75):.0f}min, "
          f"99th={valid_dur.quantile(0.99):.0f}min, "
          f"max={valid_dur.max():.0f}min")
    print(f"[timestamps] Censored events: {df['is_censored'].sum()}")

    return df


# ──────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from start_datetime (IST)."""
    dt = df["start_datetime_ist"]

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek          # 0=Mon, 6=Sun
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Rush hour: 7-10am and 5-8pm IST
    df["is_rush_hour"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 10)) |
        ((df["hour"] >= 17) & (df["hour"] <= 20))
    ).astype(int)

    # Time buckets (coarse — more robust than raw hour given timezone uncertainty)
    bins = [-1, 5, 11, 16, 20, 24]
    labels = ["night", "morning", "afternoon", "evening", "late_night"]
    df["time_bucket"] = pd.cut(df["hour"], bins=bins, labels=labels)

    return df


def engineer_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Spatial features from coordinates, corridor, zone."""

    # Clean corridor: consolidate NULL/nan to "non_corridor"
    df["corridor_clean"] = df["corridor"].fillna("non_corridor")
    df.loc[df["corridor_clean"].isin(["nan", "none", "null", ""]), "corridor_clean"] = "non_corridor"

    # Flag: is this on a named corridor?
    df["is_on_corridor"] = (df["corridor_clean"] != "non_corridor").astype(int)

    # Displacement: haversine between start and end coordinates
    has_end = (
        df["endlatitude"].notna() &
        df["endlongitude"].notna() &
        (df["endlatitude"] != 0) &
        (df["endlongitude"] != 0)
    )
    df["has_end_coords"] = has_end.astype(int)
    df["displacement_km"] = np.nan
    if has_end.any():
        df.loc[has_end, "displacement_km"] = haversine_km(
            df.loc[has_end, "latitude"],
            df.loc[has_end, "longitude"],
            df.loc[has_end, "endlatitude"],
            df.loc[has_end, "endlongitude"],
        )

    # Cause severity (pre-computed mapping for impact score)
    df["cause_severity"] = df["event_cause"].map(CAUSE_SEVERITY).fillna(0.5)

    return df


def engineer_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling event-load features per corridor.
    
    These capture: how busy is this corridor right now / recently?
    Critical for duration prediction (congested corridor = slower resolution).
    """
    # Sort by time for rolling calculations
    df = df.sort_values("start_datetime").reset_index(drop=True)

    # ── Events in corridor in last 1h and 24h ──
    # For each event, count how many events occurred in the same corridor
    # within the preceding 1h and 24h windows.
    # This is an O(n²) operation in pure pandas — we use a groupby + rolling approach.

    df["start_ts"] = df["start_datetime"].astype(np.int64) // 10**9  # unix seconds

    corridor_load_1h = []
    corridor_load_24h = []

    # Group by corridor for efficiency
    for corridor, group in df.groupby("corridor_clean"):
        if corridor == "non_corridor":
            # Don't compute load for non-corridor (meaningless aggregation)
            corridor_load_1h.extend([0] * len(group))
            corridor_load_24h.extend([0] * len(group))
            continue

        times = group["start_ts"].values
        counts_1h = []
        counts_24h = []

        for i, t in enumerate(times):
            # Count events in same corridor within [t-3600, t) and [t-86400, t)
            mask_1h = (times[:i] >= t - 3600) & (times[:i] < t)
            mask_24h = (times[:i] >= t - 86400) & (times[:i] < t)
            counts_1h.append(mask_1h.sum())
            counts_24h.append(mask_24h.sum())

        corridor_load_1h.extend(counts_1h)
        corridor_load_24h.extend(counts_24h)

    # Reorder to match original df order
    # Since we iterated groupby, we need to map back by index
    load_df = pd.DataFrame({
        "events_in_corridor_1h": corridor_load_1h,
        "events_in_corridor_24h": corridor_load_24h,
    })
    # The groupby preserves original indices, but extend doesn't.
    # Safer approach: compute on sorted df, assign directly.
    # Actually, groupby iteration gives us groups in key-order, not original order.
    # Let's use a proper index-based approach.

    # Redo with proper index tracking
    df["events_in_corridor_1h"] = 0
    df["events_in_corridor_24h"] = 0

    for corridor, group in df.groupby("corridor_clean"):
        if corridor == "non_corridor":
            continue

        idxs = group.index.values
        times = group["start_ts"].values

        for j, (idx, t) in enumerate(zip(idxs, times)):
            mask_1h = (times[:j] >= t - 3600) & (times[:j] < t)
            mask_24h = (times[:j] >= t - 86400) & (times[:j] < t)
            df.at[idx, "events_in_corridor_1h"] = int(mask_1h.sum())
            df.at[idx, "events_in_corridor_24h"] = int(mask_24h.sum())

    # ── Repeat-vehicle flag ──
    # Some veh_no values appear 4-7 times → fleet reliability signal
    if "veh_no" in df.columns:
        veh_counts = df["veh_no"].value_counts()
        df["vehicle_event_count"] = df["veh_no"].map(veh_counts).fillna(0).astype(int)
        df["is_repeat_vehicle"] = (df["vehicle_event_count"] >= 3).astype(int)
        repeat = df["is_repeat_vehicle"].sum()
        print(f"[load_features] Repeat vehicles (3+ events): {repeat} events")

    # Clean up temp column
    df = df.drop(columns=["start_ts"], errors="ignore")

    return df


def deduplicate_events(df: pd.DataFrame, time_window_min: int = 15) -> pd.DataFrame:
    """
    Remove near-duplicate events: same (lat, lng rounded to 4dp, event_cause)
    within a time window. Keep the first occurrence.
    """
    initial = len(df)

    df["lat_round"] = df["latitude"].round(4)
    df["lon_round"] = df["longitude"].round(4)

    df = df.sort_values("start_datetime")
    
    # Mark duplicates within time window
    df["dedup_key"] = (
        df["lat_round"].astype(str) + "_" +
        df["lon_round"].astype(str) + "_" +
        df["event_cause"].astype(str)
    )
    
    to_drop = set()
    for key, group in df.groupby("dedup_key"):
        if len(group) <= 1:
            continue
        times = group["start_datetime"].values
        idxs = group.index.values
        for i in range(1, len(times)):
            delta = (times[i] - times[i-1]) / np.timedelta64(1, "m")
            if delta < time_window_min:
                to_drop.add(idxs[i])

    df = df.drop(index=to_drop)
    df = df.drop(columns=["lat_round", "lon_round", "dedup_key"])

    print(f"[dedup] Removed {initial - len(df)} near-duplicates → {len(df)} remaining")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Drop useless columns
# ──────────────────────────────────────────────────────────────────────

DROP_COLS = [
    "map_file", "direction", "cargo_material", "reason_breakdown",
    "age_of_truck", "route_path", "meta_data", "citizen_accident_id",
    "comment", "gba_identifier", "client_id", "created_by_id",
    "last_modified_by_id", "assigned_to_police_id", "closed_by_id",
    "resolved_by_id", "authenticated", "kgid",
]


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with 99%+ nulls or no ML signal."""
    existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[drop_cols] Dropped {len(existing)} useless columns → {len(df.columns)} remaining")
    return df


# ──────────────────────────────────────────────────────────────────────
# Master pipeline
# ──────────────────────────────────────────────────────────────────────

def load_and_prepare_data(csv_path: str | Path = None) -> pd.DataFrame:
    """
    Full data preparation pipeline:
    1. Load raw CSV
    2. Clean core fields
    3. Drop useless columns
    4. Process timestamps (timezone, duration, censoring, batch-close)
    5. Deduplicate near-identical events
    6. Engineer temporal features
    7. Engineer spatial features
    8. Engineer load/recency features
    
    Returns a fully-featured DataFrame ready for modeling.
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / CSV_FILENAME

    print("=" * 60)
    print("PS2 DATA PIPELINE — Starting")
    print("=" * 60)

    df = load_raw(csv_path)
    df = clean_core_fields(df)
    df = drop_useless_columns(df)
    df = process_timestamps(df)
    df = deduplicate_events(df)
    df = engineer_temporal_features(df)
    df = engineer_spatial_features(df)
    df = engineer_load_features(df)

    print("=" * 60)
    print(f"PIPELINE COMPLETE — {len(df)} rows × {len(df.columns)} columns")
    print("=" * 60)

    # Summary stats
    print(f"\n  Event types: {df['event_type'].value_counts().to_dict()}")
    print(f"  Top causes: {df['event_cause'].value_counts().head(5).to_dict()}")
    print(f"  Priority: {df['priority'].value_counts().to_dict()}")
    print(f"  Road closure: {df['requires_road_closure'].value_counts().to_dict()}")
    print(f"  Status: {df['status'].value_counts().to_dict()}")
    print(f"  Censored: {df['is_censored'].sum()}")
    print(f"  Corridors: {df['corridor_clean'].nunique()} unique")
    print(f"  Date range: {df['start_datetime_ist'].min()} → {df['start_datetime_ist'].max()}")

    return df


# ──────────────────────────────────────────────────────────────────────
# Temporal train/test split
# ──────────────────────────────────────────────────────────────────────

def get_temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time, NOT randomly.
    Train: Nov 2023 – Feb 2024
    Test:  Mar 2024 – Apr 2024
    
    Random splits would leak future information in operational time-series data.
    """
    train = df[df["start_datetime_ist"] <= TRAIN_END].copy()
    test = df[df["start_datetime_ist"] > TRAIN_END].copy()

    print(f"[split] Train: {len(train)} rows ({train['start_datetime_ist'].min().date()} → "
          f"{train['start_datetime_ist'].max().date()})")
    print(f"[split] Test:  {len(test)} rows ({test['start_datetime_ist'].min().date()} → "
          f"{test['start_datetime_ist'].max().date()})")

    return train, test


# ──────────────────────────────────────────────────────────────────────
# Feature matrix builder (for sklearn/lightgbm models)
# ──────────────────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "event_cause", "corridor_clean", "zone", "police_station",
    "veh_type", "event_type", "time_bucket",
]

NUMERICAL_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "is_on_corridor", "has_end_coords", "displacement_km",
    "cause_severity", "events_in_corridor_1h", "events_in_corridor_24h",
    "vehicle_event_count", "is_repeat_vehicle",
]


def build_feature_matrix(df: pd.DataFrame, target: str = "priority_binary"):
    """
    Build X, y matrices for classification/regression.
    
    Args:
        df: Prepared DataFrame from load_and_prepare_data()
        target: Column name to predict
    
    Returns:
        X: Feature DataFrame with categoricals label-encoded
        y: Target Series
        feature_names: List of feature names in X
    """
    from sklearn.preprocessing import LabelEncoder

    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    existing = [f for f in features if f in df.columns]

    X = df[existing].copy()
    y = df[target].copy()

    # Label-encode categoricals (LightGBM handles them natively too)
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = X[col].astype(str).fillna("unknown")
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

    # Fill NaN in numericals
    for col in NUMERICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    return X, y, existing, label_encoders


# ──────────────────────────────────────────────────────────────────────
# CLI test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_and_prepare_data(csv)
    train, test = get_temporal_split(df)
    
    print("\n--- Feature matrix test (priority) ---")
    X, y, feat_names, _ = build_feature_matrix(train, target="priority_binary")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Features: {feat_names}")
    print(f"y distribution: {y.value_counts().to_dict()}")
