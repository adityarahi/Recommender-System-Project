"""Item-Based Collaborative Filtering using adjusted cosine similarity."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    """K-nearest-neighbour item-based collaborative filter.

    Uses adjusted cosine similarity (mean-centred by user) which accounts for
    individual user rating scales and is generally more stable than raw cosine
    on sparse user-item matrices.

    Args:
        k: number of most-similar items to use for prediction.
        similarity: 'cosine' or 'adjusted_cosine'.
    """

    def __init__(self, k: int = 20, similarity: str = "adjusted_cosine"):
        assert similarity in ("cosine", "adjusted_cosine")
        self.k = k
        self.similarity = similarity

        self._matrix: pd.DataFrame | None = None       # users × items
        self._sim_matrix: np.ndarray | None = None     # items × items
        self._item_index: dict | None = None           # item_id → col position
        self._user_means: pd.Series | None = None

    # ------------------------------------------------------------------
    def fit(self, train_matrix: pd.DataFrame):
        """Compute and cache the item-item similarity matrix.

        Args:
            train_matrix: users × items pivot table (NaN = unrated).
        """
        self._matrix = train_matrix.copy()
        self._user_means = self._matrix.mean(axis=1)
        self._item_index = {iid: j for j, iid in enumerate(self._matrix.columns)}

        if self.similarity == "adjusted_cosine":
            centred = self._matrix.sub(self._user_means, axis=0).fillna(0).values.astype(float)
        else:
            centred = self._matrix.fillna(0).values.astype(float)

        # Items are columns; transpose so each row = one item vector
        item_vectors = centred.T
        self._sim_matrix = cosine_similarity(item_vectors)
        np.fill_diagonal(self._sim_matrix, 0)
        return self

    # ------------------------------------------------------------------
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the rating user_id would give to item_id."""
        if item_id not in self._item_index:
            return float(self._user_means.mean())

        if user_id not in self._matrix.index:
            item_means = self._matrix.mean(axis=0)
            return float(item_means.get(item_id, self._user_means.mean()))

        j = self._item_index[item_id]
        user_row = self._matrix.loc[user_id]

        # Only consider items the user has rated
        rated_mask = user_row.notna().values
        sim_row = self._sim_matrix[j].copy()
        sim_row[~rated_mask] = 0.0

        top_k_idx = np.argsort(sim_row)[::-1][: self.k]
        top_k_sims = sim_row[top_k_idx]

        nonzero_mask = top_k_sims != 0
        if not nonzero_mask.any():
            return float(user_row.mean())

        top_k_idx = top_k_idx[nonzero_mask]
        top_k_sims = top_k_sims[nonzero_mask]

        neighbour_ratings = user_row.iloc[top_k_idx].values.astype(float)
        numerator = np.dot(top_k_sims, neighbour_ratings)
        denominator = np.sum(np.abs(top_k_sims))

        if denominator == 0:
            return float(user_row.mean())

        prediction = numerator / denominator
        return float(np.clip(prediction, 1.0, 5.0))

    # ------------------------------------------------------------------
    def predict_batch(self, user_item_pairs: pd.DataFrame) -> np.ndarray:
        """Predict ratings for a DataFrame with columns [user_id, item_id]."""
        return np.array(
            [self.predict(row.user_id, row.item_id) for row in user_item_pairs.itertuples()]
        )

    # ------------------------------------------------------------------
    def recommend(self, user_id: int, n: int = 10, exclude_rated: bool = True) -> list[int]:
        """Return the top-N item IDs predicted to be most liked by user_id."""
        if user_id not in self._matrix.index:
            return []

        rated = set()
        if exclude_rated:
            rated = set(self._matrix.columns[self._matrix.loc[user_id].notna()])

        candidates = [iid for iid in self._matrix.columns if iid not in rated]
        scores = [(iid, self.predict(user_id, iid)) for iid in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in scores[:n]]
