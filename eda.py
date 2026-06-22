"""
Day 1: Data Loading + Exploratory Data Analysis
Movie Recommendation System - MovieLens 100k Dataset

Run this from your recsys-project folder, after downloading and
unzipping ml-100k.zip into the same folder, with:

    python eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it saves files instead of needing a GUI popup
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load the data
# ----------------------------
# MovieLens 100k ratings file is tab-separated: user_id, item_id, rating, timestamp
ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=ratings_cols)

# Movie metadata file: item_id, title, release_date, ... + 19 genre flag columns
movie_cols = (
    ["item_id", "title", "release_date", "video_release_date", "imdb_url"]
    + [f"genre_{i}" for i in range(19)]
)
movies = pd.read_csv("ml-100k/u.item", sep="|", names=movie_cols, encoding="latin-1")

print("=" * 50)
print("RAW DATA PREVIEW")
print("=" * 50)
print("\nRatings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("\nFirst 5 ratings:")
print(ratings.head())

# ----------------------------
# 2. Basic stats
# ----------------------------
n_users = ratings["user_id"].nunique()
n_items = ratings["item_id"].nunique()
n_ratings = len(ratings)

print("\n" + "=" * 50)
print("BASIC STATS")
print("=" * 50)
print(f"Unique users:  {n_users}")
print(f"Unique movies: {n_items}")
print(f"Total ratings: {n_ratings}")

# Sparsity = how empty the user-item matrix is.
# Real-world recsys systems are 99%+ sparse -- this number matters
# because it's WHY collaborative filtering is hard, and it's a great
# thing to mention in an interview.
sparsity = 1 - (n_ratings / (n_users * n_items))
print(f"Matrix sparsity: {sparsity:.4%}  (i.e. {sparsity:.4%} of user-movie pairs have NO rating)")

# ----------------------------
# 3. Rating distribution
# ----------------------------
print("\n" + "=" * 50)
print("RATING DISTRIBUTION")
print("=" * 50)
print(ratings["rating"].value_counts().sort_index())

plt.figure(figsize=(6, 4))
ratings["rating"].value_counts().sort_index().plot(kind="bar", color="steelblue")
plt.title("Distribution of Ratings")
plt.xlabel("Rating (1-5)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("rating_distribution.png")
print("\nSaved chart -> rating_distribution.png")

# ----------------------------
# 4. Most-rated movies (popularity baseline)
# ----------------------------
rating_counts = ratings.groupby("item_id").size().reset_index(name="num_ratings")
top_movies = rating_counts.merge(movies[["item_id", "title"]], on="item_id")
top_movies = top_movies.sort_values("num_ratings", ascending=False).head(10)

print("\n" + "=" * 50)
print("TOP 10 MOST-RATED MOVIES")
print("=" * 50)
print(top_movies[["title", "num_ratings"]].to_string(index=False))

# ----------------------------
# 5. Ratings per user (user activity distribution)
# ----------------------------
ratings_per_user = ratings.groupby("user_id").size()

print("\n" + "=" * 50)
print("USER ACTIVITY")
print("=" * 50)
print(f"Avg ratings per user: {ratings_per_user.mean():.1f}")
print(f"Min: {ratings_per_user.min()}, Max: {ratings_per_user.max()}")

plt.figure(figsize=(6, 4))
ratings_per_user.hist(bins=30, color="darkorange")
plt.title("Number of Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig("ratings_per_user.png")
print("\nSaved chart -> ratings_per_user.png")

# ----------------------------
# 6. Save a clean file for Day 2
# ----------------------------
ratings.to_csv("clean_ratings.csv", index=False)
movies.to_csv("clean_movies.csv", index=False)
print("\n" + "=" * 50)
print("Saved clean_ratings.csv and clean_movies.csv")
print("You're ready for Day 2 (building the model).")
print("=" * 50)