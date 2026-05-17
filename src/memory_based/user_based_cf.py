"""User-Based Collaborative Filtering using cosine or Pearson similarity."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """K-nearest-neighbour user-based collaborative filter.

    Args:
        k: number of nearest neighbours to use for prediction.
        similarity: 'cosine' or 'pearson'.
        min_common: minimum number of co-rated items required to compute similarity.
    """

    def __init__(self, k: int = 20, similarity: str = "cosine", min_common: int = 3):
        assert similarity in ("cosine", "pearson"), "similarity must be 'cosine' or 'pearson'"
        self.k = k
        self.similarity = similarity
        self.min_common = min_common

        self._matrix: pd.DataFrame | None = None   # users × items (NaN = not rated)
        self._sim_matrix: np.ndarray | None = None  # users × users
        self._user_means: pd.Series | None = None
        self._user_index: dict | None = None        # user_id → row position

    # ------------------------------------------------------------------
    def fit(self, train_matrix: pd.DataFrame):
        """Compute and cache the user-user similarity matrix.

        Args:
            train_matrix: pivot-table from data_loader.build_user_item_matrix().
        """
        self._matrix = train_matrix.copy()
        self._user_means = self._matrix.mean(axis=1)
        self._user_index = {uid: i for i, uid in enumerate(self._matrix.index)}

        filled = self._matrix.fillna(0).values.astype(float)

        if self.similarity == "cosine":
            self._sim_matrix = cosine_similarity(filled)
        else:  # pearson — mean-centre each user before cosine
            centred = self._matrix.sub(self._user_means, axis=0).fillna(0).values.astype(float)
            self._sim_matrix = cosine_similarity(centred)

        # Zero out diagonal (a user is not their own neighbour)
        np.fill_diagonal(self._sim_matrix, 0)
        return self

    # ------------------------------------------------------------------
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the rating user_id would give to item_id."""
        if user_id not in self._user_index:
            return float(self._user_means.mean())

        u_idx = self._user_index[user_id]
        u_mean = self._user_means.iloc[u_idx]

        if item_id not in self._matrix.columns:
            return float(u_mean)

        item_col = self._matrix[item_id]
        item_raters_mask = item_col.notna().values

        sim_row = self._sim_matrix[u_idx].copy()
        sim_row[~item_raters_mask] = 0.0

        top_k_idx = np.argsort(sim_row)[::-1][: self.k]
        top_k_sims = sim_row[top_k_idx]

        nonzero_mask = top_k_sims != 0
        if not nonzero_mask.any():
            return float(u_mean)

        top_k_idx = top_k_idx[nonzero_mask]
        top_k_sims = top_k_sims[nonzero_mask]

        neighbour_ratings = item_col.iloc[top_k_idx].values.astype(float)
        neighbour_means = self._user_means.iloc[top_k_idx].values.astype(float)

        numerator = np.dot(top_k_sims, neighbour_ratings - neighbour_means)
        denominator = np.sum(np.abs(top_k_sims))

        if denominator == 0:
            return float(u_mean)

        prediction = u_mean + numerator / denominator
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
        if user_id not in self._user_index:
            return []

        rated = set()
        if exclude_rated:
            rated = set(self._matrix.columns[self._matrix.loc[user_id].notna()])

        candidates = [iid for iid in self._matrix.columns if iid not in rated]
        scores = [(iid, self.predict(user_id, iid)) for iid in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in scores[:n]]
