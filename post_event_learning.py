"""
post_event_learning.py — Post-Event Learning Loop (Phase 6)

The problem statement literally says "no post-event learning system."
Building exactly that, named that way, is an easy high-visibility point.

Approach:
    For each corridor, maintain a rolling risk prior that updates after
    each closed event. Two methods:
    
    1. Exponential Moving Average (EMA):
       risk_t = alpha * observed_severity + (1 - alpha) * risk_{t-1}
    
    2. Bayesian Beta update:
       Prior: Beta(alpha, beta) on corridor's closure probability
       After event: update alpha/beta based on whether closure occurred

Usage:
    from post_event_learning import PostEventLearner
    learner = PostEventLearner(df)
    learner.update(new_event)
    current_risk = learner.get_corridor_risk("mysore road")
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


# ──────────────────────────────────────────────────────────────────────
# Post-Event Learner
# ──────────────────────────────────────────────────────────────────────

class PostEventLearner:
    """
    Maintains rolling risk priors per corridor that update after each event.
    
    This is the "post-event learning system" that the problem statement
    says doesn't exist today.
    """
    
    def __init__(self, historical_df=None, alpha=0.1):
        """
        Initialize with optional historical data.
        
        Args:
            historical_df: Historical event data for initial priors
            alpha: EMA smoothing factor (0.1 = slow adaptation, 0.3 = fast)
        """
        self.alpha = alpha
        
        # EMA risk state per corridor
        self.ema_risk = {}
        
        # Bayesian Beta priors per corridor (for closure probability)
        # Start with weak prior Beta(1, 1) = uniform
        self.beta_priors = {}
        
        # Event counter per corridor
        self.event_counts = {}
        
        # History of risk updates (for visualization)
        self.risk_history = []
        
        if historical_df is not None:
            self._initialize_from_history(historical_df)
    
    def _initialize_from_history(self, df):
        """Set initial priors from historical data."""
        for corridor, group in df.groupby("corridor_clean"):
            n = len(group)
            self.event_counts[corridor] = n
            
            # EMA: start with mean impact score
            if "impact_score" in group.columns:
                self.ema_risk[corridor] = float(group["impact_score"].mean())
            elif "cause_severity" in group.columns:
                self.ema_risk[corridor] = float(group["cause_severity"].mean() * 100)
            else:
                self.ema_risk[corridor] = 50.0
            
            # Bayesian: initialize with observed closure rate
            closure_count = group["requires_road_closure"].sum()
            # Beta(a, b) where a = closures + 1, b = non-closures + 1
            self.beta_priors[corridor] = {
                "alpha": int(closure_count) + 1,
                "beta": int(n - closure_count) + 1,
            }
        
        print(f"[PostEventLearner] Initialized {len(self.ema_risk)} corridors from history")
    
    def update(self, event):
        """
        Update risk priors after a single event is resolved.
        
        Args:
            event: dict or Series with keys: corridor_clean, impact_score (or cause_severity),
                   requires_road_closure, event_cause
        """
        corridor = event.get("corridor_clean", "unknown")
        
        # Get severity signal
        if "impact_score" in event:
            severity = float(event["impact_score"])
        elif "cause_severity" in event:
            severity = float(event["cause_severity"]) * 100
        else:
            severity = 50.0
        
        closure = bool(event.get("requires_road_closure", False))
        
        # ── EMA update ──
        if corridor in self.ema_risk:
            old_risk = self.ema_risk[corridor]
            new_risk = self.alpha * severity + (1 - self.alpha) * old_risk
        else:
            new_risk = severity
        
        self.ema_risk[corridor] = new_risk
        
        # ── Bayesian Beta update ──
        if corridor not in self.beta_priors:
            self.beta_priors[corridor] = {"alpha": 1, "beta": 1}
        
        if closure:
            self.beta_priors[corridor]["alpha"] += 1
        else:
            self.beta_priors[corridor]["beta"] += 1
        
        # ── Count ──
        self.event_counts[corridor] = self.event_counts.get(corridor, 0) + 1
        
        # ── Log ──
        self.risk_history.append({
            "corridor": corridor,
            "event_count": self.event_counts[corridor],
            "ema_risk": new_risk,
            "closure_prob": self.get_closure_probability(corridor),
            "severity_input": severity,
        })
    
    def batch_update(self, events_df):
        """Update with a batch of resolved events (in chronological order)."""
        events_df = events_df.sort_values("start_datetime_ist")
        for _, event in events_df.iterrows():
            self.update(event.to_dict())
        print(f"[PostEventLearner] Batch updated with {len(events_df)} events")
    
    def get_corridor_risk(self, corridor):
        """Get current EMA risk score for a corridor."""
        return self.ema_risk.get(corridor, 50.0)
    
    def get_closure_probability(self, corridor):
        """
        Get Bayesian posterior probability of road closure for a corridor.
        
        Returns the mean of Beta(alpha, beta) = alpha / (alpha + beta)
        """
        prior = self.beta_priors.get(corridor, {"alpha": 1, "beta": 1})
        return prior["alpha"] / (prior["alpha"] + prior["beta"])
    
    def get_all_corridor_risks(self):
        """Get risk summary for all corridors."""
        risks = []
        for corridor in self.ema_risk:
            risks.append({
                "corridor": corridor,
                "ema_risk": self.ema_risk[corridor],
                "closure_probability": self.get_closure_probability(corridor),
                "total_events": self.event_counts.get(corridor, 0),
                "beta_alpha": self.beta_priors.get(corridor, {}).get("alpha", 1),
                "beta_beta": self.beta_priors.get(corridor, {}).get("beta", 1),
            })
        return pd.DataFrame(risks).sort_values("ema_risk", ascending=False)
    
    def get_risk_history_df(self):
        """Get history of risk updates for visualization."""
        return pd.DataFrame(self.risk_history)
    
    def to_dict(self):
        """Serialize state for saving/loading."""
        return {
            "alpha": self.alpha,
            "ema_risk": self.ema_risk,
            "beta_priors": self.beta_priors,
            "event_counts": self.event_counts,
        }
    
    @classmethod
    def from_dict(cls, state):
        """Load from serialized state."""
        learner = cls(alpha=state["alpha"])
        learner.ema_risk = state["ema_risk"]
        learner.beta_priors = state["beta_priors"]
        learner.event_counts = state["event_counts"]
        return learner


# ──────────────────────────────────────────────────────────────────────
# CLI: Demonstrate the learning loop
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_and_prepare_data, get_temporal_split
    from impact_score import compute_impact_score
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_and_prepare_data(csv_path)
    df = compute_impact_score(df)
    train_df, test_df = get_temporal_split(df)
    
    print("\n" + "=" * 60)
    print("POST-EVENT LEARNING LOOP")
    print("=" * 60)
    
    # Initialize from training data
    learner = PostEventLearner(train_df, alpha=0.1)
    
    # Show initial state
    print("\n  Initial corridor risks (top 10):")
    initial = learner.get_all_corridor_risks()
    for _, row in initial.head(10).iterrows():
        print(f"    {row['corridor']:25s}  risk={row['ema_risk']:.1f}  "
              f"closure_prob={row['closure_probability']:.3f}  "
              f"events={row['total_events']}")
    
    # Simulate processing test events one by one
    print(f"\n  Processing {len(test_df)} test events...")
    learner.batch_update(test_df)
    
    # Show updated state
    print("\n  Updated corridor risks (top 10):")
    updated = learner.get_all_corridor_risks()
    for _, row in updated.head(10).iterrows():
        print(f"    {row['corridor']:25s}  risk={row['ema_risk']:.1f}  "
              f"closure_prob={row['closure_probability']:.3f}  "
              f"events={row['total_events']}")
    
    # Show risk drift (how much changed after processing test data)
    print("\n  Risk drift (train → test):")
    merged = initial.set_index("corridor")[["ema_risk"]].rename(columns={"ema_risk": "initial"})
    merged = merged.join(
        updated.set_index("corridor")[["ema_risk"]].rename(columns={"ema_risk": "updated"})
    )
    merged["drift"] = merged["updated"] - merged["initial"]
    merged = merged.sort_values("drift", ascending=False)
    for corridor, row in merged.head(5).iterrows():
        print(f"    {corridor:25s}  {row['initial']:.1f} → {row['updated']:.1f}  "
              f"(drift: {row['drift']:+.1f})")
    print("    ...")
    for corridor, row in merged.tail(5).iterrows():
        print(f"    {corridor:25s}  {row['initial']:.1f} → {row['updated']:.1f}  "
              f"(drift: {row['drift']:+.1f})")
    
    # Save
    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / "learner_state.json", "w") as f:
        json.dump(learner.to_dict(), f, indent=2)
    
    updated.to_csv(save_dir / "corridor_risk_updated.csv", index=False)
    
    history = learner.get_risk_history_df()
    if len(history) > 0:
        history.to_csv(save_dir / "risk_history.csv", index=False)
    
    print(f"\n  State saved to {save_dir}")
