"""Evaluation metrics for rating prediction and top-N recommendation quality."""
import numpy as np


# ---------------------------------------------------------------------------
# Rating prediction
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Ranking quality (Top-N)
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of all relevant items captured in top-K recommendations."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Relevance is binary (1 if item in relevant, else 0).
    """
    if not relevant or k == 0:
        return 0.0
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def catalog_coverage(all_recommendations: list[list], n_items: int) -> float:
    """Fraction of the total item catalog that appears in any recommendation list."""
    if n_items == 0:
        return 0.0
    recommended_items = {item for recs in all_recommendations for item in recs}
    return len(recommended_items) / n_items


# ---------------------------------------------------------------------------
# Aggregate over all users
# ---------------------------------------------------------------------------

def evaluate_ranking(
    recommendations: dict[int, list],
    test_relevant: dict[int, set],
    k: int = 10,
) -> dict:
    """Compute mean Precision, Recall, and NDCG at K across all users.

    Args:
        recommendations: {user_id: [item_id, ...]} sorted by predicted score desc.
        test_relevant: {user_id: set of relevant item_ids (rating >= threshold)}.
        k: cut-off rank.
    """
    p_scores, r_scores, n_scores = [], [], []
    for user, recs in recommendations.items():
        relevant = test_relevant.get(user, set())
        p_scores.append(precision_at_k(recs, relevant, k))
        r_scores.append(recall_at_k(recs, relevant, k))
        n_scores.append(ndcg_at_k(recs, relevant, k))
    return {
        f"Precision@{k}": float(np.mean(p_scores)),
        f"Recall@{k}": float(np.mean(r_scores)),
        f"NDCG@{k}": float(np.mean(n_scores)),
    }


def evaluate_predictions(
    test_df,
    pred_col: str = "predicted_rating",
    true_col: str = "rating",
) -> dict:
    """Compute RMSE and MAE from a DataFrame with true and predicted ratings."""
    valid = test_df[[true_col, pred_col]].dropna()
    return {
        "RMSE": rmse(valid[true_col].values, valid[pred_col].values),
        "MAE": mae(valid[true_col].values, valid[pred_col].values),
    }
