# Collaborative Filtering Recommendation System

A from-scratch implementation of collaborative filtering for movie recommendations, built with NumPy, Pandas, and Scikit-learn on the MovieLens 100K dataset.

## Algorithms

| Algorithm | Type | Key idea |
|-----------|------|----------|
| User-Based KNN | Memory-based | Find similar users, predict from their ratings |
| Item-Based KNN | Memory-based | Find similar items, predict from user's rated items |
| Matrix Factorization (SVD) | Model-based | Decompose rating matrix into latent user/item factors via SGD |

## Evaluation Metrics

- **Rating prediction:** RMSE, MAE
- **Ranking quality:** Precision@K, Recall@K, NDCG@K
- **Coverage:** Catalog coverage (popularity bias check)

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings by 943 users on 1,682 movies (sparsity: ~93.7%).

Downloaded automatically via the setup script — not committed to the repo.

## Project Structure

```
├── data/
│   ├── download_data.py       # Fetches MovieLens 100K from GroupLens
│   ├── raw/                   # Downloaded dataset (git-ignored)
│   └── processed/             # Train/test splits (git-ignored)
├── src/
│   ├── data_loader.py         # Load, split, build user-item matrix
│   ├── memory_based/
│   │   ├── user_based_cf.py   # User-KNN (cosine + Pearson)
│   │   └── item_based_cf.py   # Item-KNN (cosine + adjusted-cosine)
│   ├── model_based/
│   │   └── matrix_factorization.py  # SVD via SGD (pure NumPy)
│   └── evaluation/
│       └── metrics.py         # All evaluation metrics
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory data analysis + train/test split
│   ├── 02_memory_based_cf.ipynb   # KNN sweep + evaluation
│   ├── 03_model_based_cf.ipynb    # MF hyperparameter sweep + convergence
│   └── 04_comparison.ipynb    # Side-by-side comparison + trade-off table
└── requirements.txt
```

## Quickstart

### Local

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt

# 2. Download dataset
python data/download_data.py

# 3. Run notebooks in order
jupyter notebook
```
Open and run notebooks 01 → 02 → 03 → 04.

### Google Colab

Add this cell at the top of each notebook:

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -r requirements.txt
!python data/download_data.py
```

Then run all cells. The notebooks auto-detect whether they are running locally or on Colab.

## Results (expected ranges on MovieLens 100K)

| Model | RMSE | MAE | NDCG@10 |
|-------|------|-----|---------|
| User-KNN (Pearson, K=20) | ~1.02 | ~0.81 | ~0.18 |
| Item-KNN (Adj-Cosine, K=20) | ~0.99 | ~0.78 | ~0.20 |
| Matrix Factorization (SVD) | ~0.95 | ~0.75 | ~0.23 |

Matrix Factorization achieves the best accuracy. Item-KNN offers better interpretability and handles popularity bias better than User-KNN.

## Key Design Decisions

- **Per-user 80/20 split** — every user has ratings in both train and test sets
- **Relevance threshold = 4 stars** — used to define a "relevant" item for ranking metrics
- **L2 regularisation in MF** — prevents overfitting on the sparse rating matrix
- **Adjusted cosine for Item-KNN** — mean-centres ratings by user before computing similarity, removing individual rating scale bias

## Requirements

```
numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter, requests
```
