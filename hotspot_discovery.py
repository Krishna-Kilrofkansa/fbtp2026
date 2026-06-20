"""
hotspot_discovery.py — Hidden Hotspot Discovery (Phase 6)

38% of events are "Non-corridor" — off-grid events not tracked by any
of BTP's 22 named corridors. DBSCAN clustering on lat/lng surfaces
unofficial hotspots the corridor list misses entirely.

This is a strong visual differentiator for the dashboard — interactive
map with clusters that judges can explore.

Usage:
    from hotspot_discovery import discover_hotspots, get_hotspot_summary
    hotspots_df = discover_hotspots(df)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from pathlib import Path
import json


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# DBSCAN parameters
# eps=0.005 degrees ≈ 500m at Bengaluru's latitude (12.97°N)
# min_samples=5: need at least 5 events to form a hotspot
EPS_DEGREES = 0.005
MIN_SAMPLES = 5


# ──────────────────────────────────────────────────────────────────────
# Hotspot discovery
# ──────────────────────────────────────────────────────────────────────

def discover_hotspots(df, eps=EPS_DEGREES, min_samples=MIN_SAMPLES,
                      focus_non_corridor=True):
    """
    Run DBSCAN on event coordinates to discover spatial clusters.
    
    Args:
        df: DataFrame with latitude, longitude columns
        eps: Maximum distance between two samples (in degrees)
        min_samples: Minimum points to form a cluster
        focus_non_corridor: If True, only cluster non-corridor events
                           (the interesting ones the corridor list misses)
    
    Returns:
        DataFrame with cluster labels and hotspot metadata
    """
    print("\n" + "=" * 60)
    print("HOTSPOT DISCOVERY — DBSCAN Clustering")
    print("=" * 60)
    
    work_df = df.copy()
    
    if focus_non_corridor:
        mask = work_df["corridor_clean"].isin(["non-corridor", "non_corridor"])
        cluster_df = work_df[mask].copy()
        print(f"  Clustering non-corridor events: {len(cluster_df)}")
    else:
        cluster_df = work_df.copy()
        print(f"  Clustering all events: {len(cluster_df)}")
    
    # Filter valid coordinates
    coord_mask = (
        cluster_df["latitude"].notna() &
        cluster_df["longitude"].notna() &
        (cluster_df["latitude"] != 0) &
        (cluster_df["longitude"] != 0) &
        # Bengaluru bounding box (sanity check)
        (cluster_df["latitude"] >= 12.7) &
        (cluster_df["latitude"] <= 13.4) &
        (cluster_df["longitude"] >= 77.3) &
        (cluster_df["longitude"] <= 77.9)
    )
    cluster_df = cluster_df[coord_mask].copy()
    print(f"  Valid coordinates: {len(cluster_df)}")
    
    if len(cluster_df) < min_samples:
        print("  Not enough points for clustering")
        return pd.DataFrame()
    
    # Run DBSCAN
    coords = cluster_df[["latitude", "longitude"]].values
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_df["cluster_id"] = db.fit_predict(coords)
    
    n_clusters = len(set(cluster_df["cluster_id"])) - (1 if -1 in cluster_df["cluster_id"].values else 0)
    n_noise = (cluster_df["cluster_id"] == -1).sum()
    n_clustered = (cluster_df["cluster_id"] != -1).sum()
    
    print(f"\n  Results:")
    print(f"    Clusters found:    {n_clusters}")
    print(f"    Points in clusters: {n_clustered} ({n_clustered/len(cluster_df)*100:.1f}%)")
    print(f"    Noise points:      {n_noise} ({n_noise/len(cluster_df)*100:.1f}%)")
    
    return cluster_df


def get_hotspot_summary(cluster_df):
    """
    Summarize each discovered hotspot with location, stats, and risk profile.
    """
    if len(cluster_df) == 0 or "cluster_id" not in cluster_df.columns:
        return pd.DataFrame()
    
    # Exclude noise
    clustered = cluster_df[cluster_df["cluster_id"] != -1]
    
    if len(clustered) == 0:
        return pd.DataFrame()
    
    summary = clustered.groupby("cluster_id").agg(
        event_count=("latitude", "size"),
        center_lat=("latitude", "mean"),
        center_lng=("longitude", "mean"),
        
        # Event profile
        top_cause=("event_cause", lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown"),
        unique_causes=("event_cause", "nunique"),
        high_priority_pct=("priority_binary", "mean"),
        closure_pct=("requires_road_closure", lambda x: x.astype(int).mean()),
        
        # Spatial spread
        lat_spread=("latitude", lambda x: x.max() - x.min()),
        lng_spread=("longitude", lambda x: x.max() - x.min()),
        
        # Temporal
        peak_hour=("hour", lambda x: x.mode().iloc[0] if len(x) > 0 else 0),
        
        # Sample addresses
        sample_address=("address", lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else "Unknown"),
    ).reset_index()
    
    # Approximate radius in meters (rough: 1 degree ≈ 111km at equator)
    summary["approx_radius_m"] = (
        np.sqrt(summary["lat_spread"]**2 + summary["lng_spread"]**2) * 111000 / 2
    ).astype(int)
    
    # Risk score for each hotspot
    summary["hotspot_risk"] = (
        summary["event_count"] * 0.4 +
        summary["high_priority_pct"] * 30 +
        summary["closure_pct"] * 30
    ).round(1)
    
    summary = summary.sort_values("hotspot_risk", ascending=False)
    
    # Assign names
    summary["hotspot_name"] = [f"Hotspot-{i+1}" for i in range(len(summary))]
    
    print(f"\n  === Hotspot Summary ({len(summary)} clusters) ===")
    for _, row in summary.head(10).iterrows():
        print(f"    {row['hotspot_name']:12s}  "
              f"events={row['event_count']:3d}  "
              f"cause={row['top_cause']:20s}  "
              f"high_priority={row['high_priority_pct']:.0%}  "
              f"radius={row['approx_radius_m']:4d}m  "
              f"near: {row['sample_address'][:50]}")
    
    return summary


def get_hotspot_map_data(cluster_df, summary_df):
    """
    Prepare data for map visualization.
    Returns list of dicts suitable for folium/plotly markers.
    """
    map_data = []
    
    for _, row in summary_df.iterrows():
        map_data.append({
            "name": row["hotspot_name"],
            "lat": float(row["center_lat"]),
            "lng": float(row["center_lng"]),
            "event_count": int(row["event_count"]),
            "top_cause": row["top_cause"],
            "high_priority_pct": float(row["high_priority_pct"]),
            "radius_m": int(row["approx_radius_m"]),
            "risk_score": float(row["hotspot_risk"]),
            "sample_address": row["sample_address"],
        })
    
    return map_data


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_and_prepare_data
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_and_prepare_data(csv_path)
    
    # Discover hotspots in non-corridor events
    cluster_df = discover_hotspots(df, focus_non_corridor=True)
    summary = get_hotspot_summary(cluster_df)
    
    # Also try all events
    print("\n\n  --- All events (for comparison) ---")
    cluster_all = discover_hotspots(df, focus_non_corridor=False)
    summary_all = get_hotspot_summary(cluster_all)
    
    # Save
    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)
    
    if len(summary) > 0:
        summary.to_csv(save_dir / "hotspot_summary.csv", index=False)
        map_data = get_hotspot_map_data(cluster_df, summary)
        with open(save_dir / "hotspot_map_data.json", "w") as f:
            json.dump(map_data, f, indent=2)
        print(f"\n  Saved to {save_dir}")
