"""
Day 2: Building the Recommendation Model
Movie Recommendation System - MovieLens 100k Dataset

Builds and compares two models:
  1. Item-based Collaborative Filtering (cosine similarity)
  2. Matrix Factorization (SGD-based latent factors, "FunkSVD" style)

Run from your recsys-project folder (after Day 1):
    python train_model.py

This will take 1-3 minutes to run, mostly for the matrix factorization
training loop. That's normal -- don't worry if it looks "stuck", it's
printing progress per epoch.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------
# 1. Load data (saved from Day 1)
# ----------------------------
ratings = pd.read_csv("clean_ratings.csv")
movies = pd.read_csv("clean_movies.csv")

print("Loaded", len(ratings), "ratings for", ratings["user_id"].nunique(), "users and",
      ratings["item_id"].nunique(), "items")

# ----------------------------
# 2. Train / test split
# ----------------------------
train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=RANDOM_STATE)
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# ----------------------------
# 3. Build ID mappings (user_id/item_id -> contiguous matrix index)
# ----------------------------
user_ids = ratings["user_id"].unique()
item_ids = ratings["item_id"].unique()

user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {m: i for i, m in enumerate(item_ids)}
idx_to_item = {i: m for m, i in item_to_idx.items()}

n_users = len(user_ids)
n_items = len(item_ids)
global_mean = train_df["rating"].mean()

print(f"n_users={n_users}, n_items={n_items}, global_mean_rating={global_mean:.3f}")

# ============================================================
# MODEL 1: Item-Based Collaborative Filtering
# ============================================================
print("\n" + "=" * 50)
print("MODEL 1: Item-Based Collaborative Filtering")
print("=" * 50)

train_matrix = np.zeros((n_users, n_items))
for row in train_df.itertuples():
    u = user_to_idx[row.user_id]
    i = item_to_idx[row.item_id]
    train_matrix[u, i] = row.rating

item_similarity = cosine_similarity(train_matrix.T)
print("Item similarity matrix shape:", item_similarity.shape)


def predict_item_based(u_idx, i_idx):
    user_ratings = train_matrix[u_idx]
    rated_mask = user_ratings > 0
    if not rated_mask.any():
        return global_mean

    sims = item_similarity[i_idx, rated_mask]
    rated_values = user_ratings[rated_mask]

    sim_sum = np.abs(sims).sum()
    if sim_sum == 0:
        return global_mean

    return np.dot(sims, rated_values) / sim_sum


start = time.time()
preds, actuals = [], []
for row in test_df.itertuples():
    if row.user_id not in user_to_idx or row.item_id not in item_to_idx:
        continue
    u_idx = user_to_idx[row.user_id]
    i_idx = item_to_idx[row.item_id]
    preds.append(predict_item_based(u_idx, i_idx))
    actuals.append(row.rating)

item_cf_rmse = np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))
print(f"Item-Based CF RMSE: {item_cf_rmse:.4f}  (evaluated in {time.time()-start:.1f}s)")

# ============================================================
# MODEL 2: Matrix Factorization (SGD / "FunkSVD" style)
# ============================================================
print("\n" + "=" * 50)
print("MODEL 2: Matrix Factorization (SGD)")
print("=" * 50)

N_FACTORS = 20
LEARNING_RATE = 0.01
REG = 0.02
N_EPOCHS = 15

user_factors = np.random.normal(scale=0.1, size=(n_users, N_FACTORS))
item_factors = np.random.normal(scale=0.1, size=(n_items, N_FACTORS))
user_bias = np.zeros(n_users)
item_bias = np.zeros(n_items)

train_tuples = [
    (user_to_idx[row.user_id], item_to_idx[row.item_id], row.rating)
    for row in train_df.itertuples()
]


def mf_predict(u_idx, i_idx):
    return (
        global_mean
        + user_bias[u_idx]
        + item_bias[i_idx]
        + np.dot(user_factors[u_idx], item_factors[i_idx])
    )


start = time.time()
for epoch in range(N_EPOCHS):
    np.random.shuffle(train_tuples)
    sq_err_sum = 0.0
    for u_idx, i_idx, r in train_tuples:
        pred = mf_predict(u_idx, i_idx)
        err = r - pred
        sq_err_sum += err ** 2

        user_bias[u_idx] += LEARNING_RATE * (err - REG * user_bias[u_idx])
        item_bias[i_idx] += LEARNING_RATE * (err - REG * item_bias[i_idx])

        uf_old = user_factors[u_idx].copy()
        user_factors[u_idx] += LEARNING_RATE * (err * item_factors[i_idx] - REG * user_factors[u_idx])
        item_factors[i_idx] += LEARNING_RATE * (err * uf_old - REG * item_factors[i_idx])

    train_rmse = np.sqrt(sq_err_sum / len(train_tuples))
    print(f"Epoch {epoch+1:2d}/{N_EPOCHS} - train RMSE: {train_rmse:.4f}")

print(f"Training took {time.time()-start:.1f}s")

preds, actuals = [], []
for row in test_df.itertuples():
    if row.user_id not in user_to_idx or row.item_id not in item_to_idx:
        continue
    u_idx = user_to_idx[row.user_id]
    i_idx = item_to_idx[row.item_id]
    preds.append(mf_predict(u_idx, i_idx))
    actuals.append(row.rating)

mf_rmse = np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))
print(f"Matrix Factorization RMSE: {mf_rmse:.4f}")

# ============================================================
# Compare and save the better model
# ============================================================
print("\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"Item-Based CF RMSE:        {item_cf_rmse:.4f}")
print(f"Matrix Factorization RMSE: {mf_rmse:.4f}")

if mf_rmse <= item_cf_rmse:
    print("-> Matrix Factorization wins. Saving this as the production model.")
    best_model = "matrix_factorization"
else:
    print("-> Item-Based CF wins. Saving this as the production model.")
    best_model = "item_based_cf"

artifacts = {
    "best_model": best_model,
    "user_to_idx": user_to_idx,
    "item_to_idx": item_to_idx,
    "idx_to_item": idx_to_item,
    "global_mean": global_mean,
    "train_matrix": train_matrix,
    "item_similarity": item_similarity,
    "user_factors": user_factors,
    "item_factors": item_factors,
    "user_bias": user_bias,
    "item_bias": item_bias,
    "item_cf_rmse": item_cf_rmse,
    "mf_rmse": mf_rmse,
}
joblib.dump(artifacts, "model_artifacts.pkl")
print("\nSaved all model artifacts -> model_artifacts.pkl")


def get_top_n_recommendations(user_id, n=10, use_model="matrix_factorization"):
    if user_id not in user_to_idx:
        return []
    u_idx = user_to_idx[user_id]
    already_rated = set(np.where(train_matrix[u_idx] > 0)[0])

    scores = []
    for i_idx in range(n_items):
        if i_idx in already_rated:
            continue
        if use_model == "matrix_factorization":
            score = mf_predict(u_idx, i_idx)
        else:
            score = predict_item_based(u_idx, i_idx)
        scores.append((i_idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_items = scores[:n]

    movie_lookup = movies.set_index("item_id")["title"].to_dict()
    return [
        (movie_lookup.get(idx_to_item[i_idx], "Unknown"), round(score, 2))
        for i_idx, score in top_items
    ]


sample_user = ratings["user_id"].iloc[0]
print(f"\nSample recommendations for user_id={sample_user}:")
for title, score in get_top_n_recommendations(sample_user, n=10, use_model=best_model):
    print(f"  {title}  (predicted rating: {score})")