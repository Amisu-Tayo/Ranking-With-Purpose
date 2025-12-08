# rwp_recommender.py

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "student_success_percentile",
    "affordability_percentile",
    "resources_percentile",
    "equity_percentile",
]


def apply_filters(
    df_in: pd.DataFrame,
    thresholds: dict | None = None,
    states_preferred: list[str] | None = None,
    states_excluded: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply metric thresholds + state filters.
      thresholds: {column_name: minimum_percentile}
      states_preferred: list of state codes to keep (if not empty)
      states_excluded: list of state codes to drop
    """
    df_f = df_in.copy()

    thresholds = thresholds or {}
    states_preferred = states_preferred or []
    states_excluded = states_excluded or []

    # State filters
    if states_preferred and "State" in df_f.columns:
        df_f = df_f[df_f["State"].isin(states_preferred)]

    if states_excluded and "State" in df_f.columns:
        df_f = df_f[~df_f["State"].isin(states_excluded)]

    # Metric thresholds
    for col, min_val in thresholds.items():
        if col in df_f.columns:
            try:
                min_val = float(min_val)
                df_f = df_f[df_f[col] >= min_val]
            except (TypeError, ValueError):
                continue

    return df_f


def normalize_weights(weights: dict) -> dict:
    """
    Ensure weights for FEATURE_COLS sum to 1.0.
    If all zero/missing, fallback to equal weights.
    """
    w = {}
    for col in FEATURE_COLS:
        try:
            w[col] = float(weights.get(col, 0.0))
        except (TypeError, ValueError):
            w[col] = 0.0

    total = sum(w.values())
    if total == 0:
        # fallback: equal weights
        w = {col: 1.0 for col in FEATURE_COLS}
        total = float(len(FEATURE_COLS))

    return {col: val / total for col, val in w.items()}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def recommend_weighted_sum(
    df_in: pd.DataFrame,
    weights: dict,
    thresholds: dict | None = None,
    states_preferred: list[str] | None = None,
    states_excluded: list[str] | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Model 1: Weighted sum over FEATURE_COLS after filtering.
    Returns a dataframe with an extra 'score_weighted' column, sorted descending.
    """
    df_f = apply_filters(df_in, thresholds, states_preferred, states_excluded)
    if df_f.empty:
        df_f = df_in.copy()

    w = normalize_weights(weights)

    def compute_score(row):
        s = 0.0
        for col in FEATURE_COLS:
            if col in row and pd.notna(row[col]):
                try:
                    s += w[col] * float(row[col])
                except (TypeError, ValueError):
                    continue
        return s

    df_f = df_f.copy()
    df_f["score_weighted"] = df_f.apply(compute_score, axis=1)
    df_f = df_f.sort_values("score_weighted", ascending=False)

    return df_f.head(top_k)


def recommend_cosine(
    df_in: pd.DataFrame,
    weights: dict,
    thresholds: dict | None = None,
    states_preferred: list[str] | None = None,
    states_excluded: list[str] | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Model 2: Cosine similarity between school vector (metrics) and weight vector.
    Returns a dataframe with an extra 'score_cosine' column, sorted descending.
    """
    df_f = apply_filters(df_in, thresholds, states_preferred, states_excluded)
    if df_f.empty:
        df_f = df_in.copy()

    w_norm = normalize_weights(weights)
    pref_vec = np.array([w_norm[col] for col in FEATURE_COLS], dtype=float)

    def compute_sim(row):
        vals = []
        for col in FEATURE_COLS:
            if col in row and pd.notna(row[col]):
                vals.append(float(row[col]))
            else:
                vals.append(0.0)
        school_vec = np.array(vals, dtype=float)
        return cosine_similarity(pref_vec, school_vec)

    df_f = df_f.copy()
    df_f["score_cosine"] = df_f.apply(compute_sim, axis=1)
    df_f = df_f.sort_values("score_cosine", ascending=False)

    return df_f.head(top_k)


def recommend_cluster_aware(
    df_in: pd.DataFrame,
    base_model: str,
    weights: dict,
    thresholds: dict | None = None,
    states_preferred: list[str] | None = None,
    states_excluded: list[str] | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Model 3: Cluster-aware re-ranking.

    Steps:
      1. Compute base model scores ('weighted' or 'cosine').
      2. Within each cluster, sort by that score.
      3. Build final list by round-robin across clusters to promote diversity.
    """
    if "cluster_name" not in df_in.columns:
        raise ValueError("cluster_name column not found in dataframe.")

    # Score with base model, but keep all rows
    if base_model == "weighted":
        df_scored = recommend_weighted_sum(
            df_in, weights, thresholds, states_preferred, states_excluded, top_k=len(df_in)
        )
        score_col = "score_weighted"
    elif base_model == "cosine":
        df_scored = recommend_cosine(
            df_in, weights, thresholds, states_preferred, states_excluded, top_k=len(df_in)
        )
        score_col = "score_cosine"
    else:
        raise ValueError("base_model must be 'weighted' or 'cosine'.")

    # Group by cluster and sort within each cluster
    cluster_groups = {}
    for cluster, group in df_scored.groupby("cluster_name"):
        cluster_groups[cluster] = group.sort_values(score_col, ascending=False)

    picked_rows = []
    cluster_names = list(cluster_groups.keys())
    idx_per_cluster = {c: 0 for c in cluster_names}

    # Round-robin
    while len(picked_rows) < top_k:
        made_progress = False
        for c in cluster_names:
            group = cluster_groups[c]
            idx = idx_per_cluster[c]
            if idx < len(group):
                picked_rows.append(group.iloc[idx])
                idx_per_cluster[c] += 1
                made_progress = True
                if len(picked_rows) >= top_k:
                    break
        if not made_progress:
            break

    if not picked_rows:
        return df_scored.head(top_k)

    result = pd.DataFrame(picked_rows).reset_index(drop=True)
    result = result.rename(columns={score_col: "score_cluster_base"})
    result["score_cluster_aware"] = np.arange(len(result), 0, -1)

    return result
