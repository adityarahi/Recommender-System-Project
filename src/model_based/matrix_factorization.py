"""Matrix Factorization via SGD (pure NumPy implementation).

Factorises R ≈ μ + b_u + b_i + U @ V.T
where:
  μ      = global mean rating
  b_u    = user bias vector
  b_i    = item bias vector
  U, V   = latent factor matrices (n_users × n_factors, n_items × n_factors)
"""
import numpy as np
import pandas as pd


class MatrixFactorization:
    """SVD-style Matrix Factorization with SGD optimisation and bias terms.

    Args:
        n_factors: dimensionality of the latent space.
        n_epochs: number of passes over the training data.
        lr: learning rate for SGD.
        reg: L2 regularisation coefficient applied to all parameters.
        random_state: seed for reproducibility.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        self._global_mean: float = 0.0
        self._U: np.ndarray | None = None   # user latent factors
        self._V: np.ndarray | None = None   # item latent factors
        self._b_u: np.ndarray | None = None # user biases
        self._b_i: np.ndarray | None = None # item biases
        self._user_index: dict = {}
        self._item_index: dict = {}
        self.train_loss_: list[float] = []
        self.val_loss_: list[float] = []

    # ------------------------------------------------------------------
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None):
        """Train on a ratings DataFrame with columns [user_id, item_id, rating].

        Args:
            train: training ratings.
            val: optional validation set for loss tracking per epoch.
        """
        rng = np.random.default_rng(self.random_state)

        users = train["user_id"].unique()
        items = train["item_id"].unique()
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {it: j for j, it in enumerate(items)}

        n_users, n_items = len(users), len(items)
        self._global_mean = float(train["rating"].mean())

        scale = 0.1
        self._U = rng.normal(0, scale, (n_users, self.n_factors))
        self._V = rng.normal(0, scale, (n_items, self.n_factors))
        self._b_u = np.zeros(n_users)
        self._b_i = np.zeros(n_items)

        records = train[["user_id", "item_id", "rating"]].values

        for epoch in range(self.n_epochs):
            rng.shuffle(records)
            epoch_loss = 0.0

            for user_id, item_id, r in records:
                u = self._user_index.get(user_id)
                i = self._item_index.get(item_id)
                if u is None or i is None:
                    continue

                pred = (
                    self._global_mean
                    + self._b_u[u]
                    + self._b_i[i]
                    + self._U[u] @ self._V[i]
                )
                err = float(r) - pred
                epoch_loss += err ** 2

                # SGD updates
                self._b_u[u] += self.lr * (err - self.reg * self._b_u[u])
                self._b_i[i] += self.lr * (err - self.reg * self._b_i[i])

                u_vec = self._U[u].copy()
                self._U[u] += self.lr * (err * self._V[i] - self.reg * self._U[u])
                self._V[i] += self.lr * (err * u_vec - self.reg * self._V[i])

            train_rmse = np.sqrt(epoch_loss / len(records))
            self.train_loss_.append(train_rmse)

            if val is not None:
                val_preds = self.predict_batch(val)
                val_rmse = float(np.sqrt(np.mean((val["rating"].values - val_preds) ** 2)))
                self.val_loss_.append(val_rmse)

            if (epoch + 1) % 5 == 0:
                msg = f"Epoch {epoch + 1}/{self.n_epochs}  train RMSE={train_rmse:.4f}"
                if val is not None:
                    msg += f"  val RMSE={self.val_loss_[-1]:.4f}"
                print(msg)

        return self

    # ------------------------------------------------------------------
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict the rating user_id would give to item_id."""
        u = self._user_index.get(user_id)
        i = self._item_index.get(item_id)

        b_u = self._b_u[u] if u is not None else 0.0
        b_i = self._b_i[i] if i is not None else 0.0
        dot = (self._U[u] @ self._V[i]) if (u is not None and i is not None) else 0.0

        pred = self._global_mean + b_u + b_i + dot
        return float(np.clip(pred, 1.0, 5.0))

    # ------------------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict ratings for a DataFrame with columns [user_id, item_id]."""
        return np.array(
            [self.predict(row.user_id, row.item_id) for row in df.itertuples()]
        )

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        all_item_ids: list[int],
        rated_item_ids: set[int] | None = None,
        n: int = 10,
    ) -> list[int]:
        """Return top-N item IDs with highest predicted score for user_id."""
        rated = rated_item_ids or set()
        candidates = [iid for iid in all_item_ids if iid not in rated]
        scores = [(iid, self.predict(user_id, iid)) for iid in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in scores[:n]]
