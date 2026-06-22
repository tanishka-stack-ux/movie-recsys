import time
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("recsys")

logger.info("Loading model artifacts...")
artifacts = joblib.load("model_artifacts.pkl")

user_to_idx = artifacts["user_to_idx"]
item_to_idx = artifacts["item_to_idx"]
idx_to_item = artifacts["idx_to_item"]
global_mean = artifacts["global_mean"]
train_matrix = artifacts["train_matrix"]
user_factors = artifacts["user_factors"]
item_factors = artifacts["item_factors"]
user_bias = artifacts["user_bias"]
item_bias = artifacts["item_bias"]
n_items = len(item_to_idx)

movies = pd.read_csv("clean_movies.csv")
movie_lookup = movies.set_index("item_id")["title"].to_dict()

ratings = pd.read_csv("clean_ratings.csv")
popular_items = (
    ratings.groupby("item_id").size().sort_values(ascending=False).head(20).index.tolist()
)
popular_titles = [movie_lookup.get(i, "Unknown") for i in popular_items]

logger.info(f"Loaded model. {len(user_to_idx)} users, {n_items} items.")

recommendation_cache: dict = {}
cache_hits = 0
cache_misses = 0


def mf_predict(u_idx: int, i_idx: int) -> float:
    return float(
        global_mean
        + user_bias[u_idx]
        + item_bias[i_idx]
        + np.dot(user_factors[u_idx], item_factors[i_idx])
    )


def compute_recommendations(user_id: int, n: int = 10):
    if user_id not in user_to_idx:
        logger.info(f"user_id={user_id} not found -- using popularity fallback")
        return [{"title": t, "predicted_rating": None} for t in popular_titles[:n]], True

    u_idx = user_to_idx[user_id]
    already_rated = set(np.where(train_matrix[u_idx] > 0)[0])

    scores = []
    for i_idx in range(n_items):
        if i_idx in already_rated:
            continue
        score = mf_predict(u_idx, i_idx)
        scores.append((i_idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_items = scores[:n]

    results = [
        {"title": movie_lookup.get(idx_to_item[i_idx], "Unknown"), "predicted_rating": round(score, 2)}
        for i_idx, score in top_items
    ]
    return results, False


def get_recommendations_cached(user_id: int, n: int = 10):
    global cache_hits, cache_misses
    cache_key = (user_id, n)

    if cache_key in recommendation_cache:
        cache_hits += 1
        recommendations, is_cold_start = recommendation_cache[cache_key]
        return recommendations, is_cold_start, True

    cache_misses += 1
    recommendations, is_cold_start = compute_recommendations(user_id, n)
    recommendation_cache[cache_key] = (recommendations, is_cold_start)
    return recommendations, is_cold_start, False


app = FastAPI(title="Movie Recommendation API", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "n_users": len(user_to_idx), "n_items": n_items}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    start = time.time()

    if n < 1 or n > 50:
        raise HTTPException(status_code=400, detail="n must be between 1 and 50")

    recommendations, is_cold_start, was_cached = get_recommendations_cached(user_id, n)
    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        f"user_id={user_id} n={n} cold_start={is_cold_start} "
        f"cached={was_cached} latency_ms={elapsed_ms:.1f}"
    )

    return {
        "user_id": user_id,
        "cold_start": is_cold_start,
        "cached": was_cached,
        "count": len(recommendations),
        "latency_ms": round(elapsed_ms, 2),
        "recommendations": recommendations,
    }


@app.get("/cache-stats")
def cache_stats():
    total = cache_hits + cache_misses
    hit_rate = (cache_hits / total * 100) if total > 0 else 0
    return {
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "hit_rate_pct": round(hit_rate, 2),
        "cached_keys": len(recommendation_cache),
    }