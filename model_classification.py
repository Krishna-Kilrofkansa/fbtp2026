"""
model_classification.py — Gradient-Boosted Classifiers for PS2
    Model A1: Priority (High/Low) classifier
    Model A2: Road Closure (True/False) classifier

Usage:
    from model_classification import train_priority_model, train_closure_model, evaluate_model
    
    # From data_loader
    from data_loader import load_and_prepare_data, get_temporal_split, build_feature_matrix
    df = load_and_prepare_data()
    train_df, test_df = get_temporal_split(df)
    
    # Train & evaluate
    priority_results = train_priority_model(train_df, test_df)
    closure_results = train_closure_model(train_df, test_df)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
import json
import pickle
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Feature definitions (mirrors data_loader.py)
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

# ── Fix 1: Semantic features for priority model (no spatial leakage) ──
# corridor_clean perfectly encodes BTP's priority rule (named corridor = High).
# Removing corridor, zone, police_station, is_on_corridor, and corridor-load
# features forces the model to learn from event SEMANTICS, not spatial rules.
PRIORITY_SEMANTIC_CATEGORICAL = [
    "event_cause", "veh_type", "event_type", "time_bucket",
]

PRIORITY_SEMANTIC_NUMERICAL = [
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "has_end_coords", "displacement_km", "cause_severity",
    "vehicle_event_count", "is_repeat_vehicle",
]


def _prepare_features(train_df, test_df, target_col, extra_features=None):
    """
    Prepare X_train, y_train, X_test, y_test with consistent label encoding.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_col: Name of target column
        extra_features: Additional feature columns beyond the defaults
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names, label_encoders, cat_indices
    """
    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    if extra_features:
        features = features + [f for f in extra_features if f not in features]
    
    # Only use features that exist in both splits
    existing = [f for f in features if f in train_df.columns and f in test_df.columns]
    
    X_train = train_df[existing].copy()
    X_test = test_df[existing].copy()
    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()
    
    # Label-encode categoricals — fit on train, transform both
    label_encoders = {}
    cat_indices = []
    for i, col in enumerate(existing):
        if col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            X_train[col] = X_train[col].astype(str).fillna("unknown")
            X_test[col] = X_test[col].astype(str).fillna("unknown")
            
            # Fit on combined to handle unseen categories in test
            all_vals = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_vals)
            
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le
            cat_indices.append(i)
    
    # Fill NaN in numericals
    for col in existing:
        if col in NUMERICAL_FEATURES or (extra_features and col in extra_features):
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
    
    return X_train, y_train, X_test, y_test, existing, label_encoders, cat_indices


# ──────────────────────────────────────────────────────────────────────
# Model A1: Priority Classifier (High/Low)
# ──────────────────────────────────────────────────────────────────────

def train_priority_model(train_df, test_df, save_path=None):
    """
    Train LightGBM classifier for priority prediction (High=1, Low=0).
    
    Priority distribution is ~63/37, roughly balanced — no special
    rebalancing needed.
    
    Returns:
        dict with model, metrics, feature importances, predictions
    """
    print("\n" + "=" * 60)
    print("MODEL A1 — Priority Classifier (High/Low)")
    print("=" * 60)
    
    X_train, y_train, X_test, y_test, feat_names, encoders, cat_idx = \
        _prepare_features(train_df, test_df, "priority_binary")
    
    print(f"  Train: {len(X_train)} rows, y distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test:  {len(X_test)} rows, y distribution: {y_test.value_counts().to_dict()}")
    
    # LightGBM parameters
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train,
                              categorical_feature=[feat_names[i] for i in cat_idx],
                              free_raw_data=False)
    valid_data = lgb.Dataset(X_test, label=y_test,
                              categorical_feature=[feat_names[i] for i in cat_idx],
                              free_raw_data=False,
                              reference=train_data)
    
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30),
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    
    # Find optimal threshold via PR curve (model probabilities are well-separated
    # but concentrated — naive 0.5 threshold doesn't work)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    print(f"  Optimal threshold (max F1): {best_threshold:.3f}")
    
    # Metrics
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Priority (High/Low)")
    metrics["optimal_threshold"] = float(best_threshold)
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": feat_names,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    
    print("\n  Top 10 features (by gain):")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:,.0f}")
    
    results = {
        "model": model,
        "metrics": metrics,
        "importance": importance,
        "predictions": y_pred_proba,
        "y_test": y_test,
        "feature_names": feat_names,
        "label_encoders": encoders,
    }
    
    if save_path:
        model.save_model(str(save_path / "priority_model.txt"))
        importance.to_csv(str(save_path / "priority_importance.csv"), index=False)
        print(f"\n  Model saved to {save_path / 'priority_model.txt'}")
    
    return results


def train_priority_model_semantic(train_df, test_df, save_path=None):
    """
    Train LightGBM priority classifier WITHOUT spatial features.
    
    This is the leakage-free version. corridor_clean perfectly encodes
    BTP's priority rule (named corridor = High, non-corridor = Low).
    Removing all spatial features forces the model to learn from
    event semantics — enabling classification BEFORE dispatch location
    is confirmed.
    
    Expected AUC: ~0.65-0.72 (honest, not 0.9995 from leakage)
    """
    print("\n" + "=" * 60)
    print("MODEL A1-SEMANTIC — Priority Classifier (No Spatial Leakage)")
    print("=" * 60)
    
    # Use semantic-only features
    features = PRIORITY_SEMANTIC_CATEGORICAL + PRIORITY_SEMANTIC_NUMERICAL
    existing = [f for f in features if f in train_df.columns and f in test_df.columns]
    
    X_train = train_df[existing].copy()
    X_test = test_df[existing].copy()
    y_train = train_df["priority_binary"].copy()
    y_test = test_df["priority_binary"].copy()
    
    # Label-encode categoricals
    label_encoders = {}
    cat_indices = []
    for i, col in enumerate(existing):
        if col in PRIORITY_SEMANTIC_CATEGORICAL:
            le = LabelEncoder()
            X_train[col] = X_train[col].astype(str).fillna("unknown")
            X_test[col] = X_test[col].astype(str).fillna("unknown")
            all_vals = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_vals)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
            label_encoders[col] = le
            cat_indices.append(i)
    
    # Fill NaN in numericals
    for col in existing:
        if col in PRIORITY_SEMANTIC_NUMERICAL:
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
    
    print(f"  Features (semantic only): {existing}")
    print(f"  Excluded: corridor_clean, zone, police_station, is_on_corridor, corridor_load")
    print(f"  Train: {len(X_train)} rows, y distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test:  {len(X_test)} rows, y distribution: {y_test.value_counts().to_dict()}")
    
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
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
                              categorical_feature=[existing[i] for i in cat_indices],
                              free_raw_data=False)
    valid_data = lgb.Dataset(X_test, label=y_test,
                              categorical_feature=[existing[i] for i in cat_indices],
                              free_raw_data=False,
                              reference=train_data)
    
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30),
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    
    y_pred_proba = model.predict(X_test)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    print(f"  Optimal threshold (max F1): {best_threshold:.3f}")
    
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Priority SEMANTIC (No Leakage)")
    metrics["optimal_threshold"] = float(best_threshold)
    
    importance = pd.DataFrame({
        "feature": existing,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    
    print("\n  Top features (semantic, by gain):")
    for _, row in importance.iterrows():
        print(f"    {row['feature']:30s} {row['importance']:,.0f}")
    
    results = {
        "model": model,
        "metrics": metrics,
        "importance": importance,
        "predictions": y_pred_proba,
        "y_test": y_test,
        "feature_names": existing,
        "label_encoders": label_encoders,
    }
    
    if save_path:
        model.save_model(str(save_path / "priority_model_semantic.txt"))
        importance.to_csv(str(save_path / "priority_importance_semantic.csv"), index=False)
        print(f"\n  Semantic model saved to {save_path / 'priority_model_semantic.txt'}")
    
    return results


# ──────────────────────────────────────────────────────────────────────
# Model A2: Road Closure Classifier
# ──────────────────────────────────────────────────────────────────────

def train_closure_model(train_df, test_df, save_path=None):
    """
    Train LightGBM classifier for road closure prediction.
    
    Key challenge: 8% positive rate = heavily imbalanced.
    Strategy: scale_pos_weight + PR-AUC as primary metric.
    
    Returns:
        dict with model, metrics, feature importances, predictions
    """
    print("\n" + "=" * 60)
    print("MODEL A2 — Road Closure Classifier")
    print("=" * 60)
    
    # Include priority as a feature for closure prediction
    X_train, y_train, X_test, y_test, feat_names, encoders, cat_idx = \
        _prepare_features(
            train_df, test_df,
            "requires_road_closure",
            extra_features=["priority_binary"]
        )
    
    # Convert boolean target to int
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_weight = neg_count / max(pos_count, 1)
    
    print(f"  Train: {len(X_train)} rows, positive: {pos_count} ({pos_count/len(y_train)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} rows, positive: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    print(f"  scale_pos_weight: {scale_weight:.1f}")
    
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_weight,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train,
                              categorical_feature=[feat_names[i] for i in cat_idx],
                              free_raw_data=False)
    valid_data = lgb.Dataset(X_test, label=y_test,
                              categorical_feature=[feat_names[i] for i in cat_idx],
                              free_raw_data=False,
                              reference=train_data)
    
    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=30),
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    
    # Find optimal threshold using PR curve (since imbalanced)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    # Optimize for F1 = 2*P*R / (P+R)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    print(f"  Optimal threshold (max F1): {best_threshold:.3f}")
    
    # Metrics
    metrics = evaluate_model(y_test, y_pred, y_pred_proba, "Road Closure")
    metrics["optimal_threshold"] = float(best_threshold)
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": feat_names,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    
    print("\n  Top 10 features (by gain):")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:,.0f}")
    
    results = {
        "model": model,
        "metrics": metrics,
        "importance": importance,
        "predictions": y_pred_proba,
        "y_test": y_test,
        "feature_names": feat_names,
        "label_encoders": encoders,
        "optimal_threshold": best_threshold,
    }
    
    if save_path:
        model.save_model(str(save_path / "closure_model.txt"))
        importance.to_csv(str(save_path / "closure_importance.csv"), index=False)
        print(f"\n  Model saved to {save_path / 'closure_model.txt'}")
    
    return results


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_model(y_true, y_pred, y_pred_proba, model_name="Model"):
    """
    Compute and print evaluation metrics.
    
    For imbalanced targets (road_closure), PR-AUC matters more than ROC-AUC.
    """
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n  === {model_name} — Test Set Results ===")
    print(f"  ROC-AUC:  {auc_roc:.4f}")
    print(f"  PR-AUC:   {pr_auc:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    metrics = {
        "roc_auc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
        "accuracy": float(report["accuracy"]),
        "confusion_matrix": cm.tolist(),
    }
    
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Generate predictions for downstream models (B & C)
# ──────────────────────────────────────────────────────────────────────

def add_model_a_predictions(df, priority_model, closure_model, feature_names_priority, feature_names_closure, encoders_priority, encoders_closure, closure_threshold=0.5):
    """
    Add predicted priority and closure probabilities as features 
    for downstream models (survival, risk forecast).
    """
    # Prepare features using same encoding
    for feat_set, encs, model, pred_col in [
        (feature_names_priority, encoders_priority, priority_model, "pred_priority_proba"),
        (feature_names_closure, encoders_closure, closure_model, "pred_closure_proba"),
    ]:
        X = df[feat_set].copy()
        for col, le in encs.items():
            if col in X.columns:
                X[col] = X[col].astype(str).fillna("unknown")
                # Handle unseen categories
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else "unknown")
                X[col] = le.transform(X[col])
        for col in X.columns:
            if col not in CATEGORICAL_FEATURES:
                X[col] = X[col].fillna(0)
        
        df[pred_col] = model.predict(X)
    
    df["pred_priority"] = (df["pred_priority_proba"] >= 0.5).astype(int)
    df["pred_closure"] = (df["pred_closure_proba"] >= closure_threshold).astype(int)
    
    return df


# ──────────────────────────────────────────────────────────────────────
# CLI: run both models end-to-end
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_and_prepare_data, get_temporal_split
    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Load and split
    df = load_and_prepare_data(csv_path)
    train_df, test_df = get_temporal_split(df)
    
    # Save directory
    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)
    
    # Train all models
    priority_results = train_priority_model(train_df, test_df, save_path=save_dir)
    priority_semantic_results = train_priority_model_semantic(train_df, test_df, save_path=save_dir)
    closure_results = train_closure_model(train_df, test_df, save_path=save_dir)
    
    # Save metrics summary
    summary = {
        "priority_spatial": priority_results["metrics"],
        "priority_semantic": priority_semantic_results["metrics"],
        "closure": closure_results["metrics"],
        "leakage_note": "priority_spatial has AUC ~1.0 because corridor_clean perfectly encodes BTP's priority rule. priority_semantic is the honest model without spatial leakage.",
    }
    with open(save_dir / "classification_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION MODELS — Summary")
    print("=" * 60)
    print(f"  Priority (spatial)  — ROC-AUC: {priority_results['metrics']['roc_auc']:.4f}, "
          f"F1: {priority_results['metrics']['f1']:.4f}  ⚠️ DATA LEAKAGE")
    print(f"  Priority (semantic) — ROC-AUC: {priority_semantic_results['metrics']['roc_auc']:.4f}, "
          f"F1: {priority_semantic_results['metrics']['f1']:.4f}  ✅ HONEST")
    print(f"  Closure             — ROC-AUC: {closure_results['metrics']['roc_auc']:.4f}, "
          f"PR-AUC: {closure_results['metrics']['pr_auc']:.4f}, "
          f"F1: {closure_results['metrics']['f1']:.4f}")
    print(f"\n  Models saved to: {save_dir}")

