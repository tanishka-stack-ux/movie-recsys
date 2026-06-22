# 🎬 Scalable Movie Recommendation API

A production-grade movie recommendation system built with Matrix Factorization 
and served via a REST API. Built to demonstrate ML systems design concepts 
relevant to FAANG-level SDE interviews.

## 📊 Results
- **Model RMSE:** 0.93 (Matrix Factorization vs 1.01 Item-Based CF)
- **API Latency:** ~16ms uncached, ~0ms cached (p99)
- **Cache Hit Rate:** 66%+ in testing
- **Dataset:** MovieLens 100k (943 users, 1,682 movies, 100,000 ratings)
- **Matrix Sparsity:** 93.7%

## 🏗️ Architecture
Raw Data (MovieLens 100k)

↓

Data Pipeline (eda.py) → clean_ratings.csv, clean_movies.csv

↓

Model Training (train_model.py)

├── Item-Based Collaborative Filtering (RMSE: 1.01)

└── Matrix Factorization/SGD (RMSE: 0.93) ← winner

↓

REST API (main.py - FastAPI)

├── GET /recommend/{user_id} → Top-N recommendations

├── GET /health → API status

└── GET /cache-stats → Cache performance metrics

↓

In-Memory Cache (dict-based, Redis-ready)
## 🚀 Features

- **Matrix Factorization** with SGD optimization (20 latent factors)
- **Cold-start handling** — new users get popularity-based fallback
- **In-memory caching** — repeat requests served in ~0ms
- **Request logging** — every request logged with latency
- **Automated tests** — 6 pytest tests covering all endpoints
- **Interactive API docs** — auto-generated Swagger UI

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | NumPy, Scikit-learn |
| API | FastAPI, Uvicorn |
| Testing | Pytest, HTTPX |
| Data | Pandas, MovieLens 100k |

## ⚙️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/tanishka-stack-ux/recsys-project.git
cd recsys-project

# Set up environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Download data and train model
python build.py

# Start the API
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to test interactively.

## 🧪 Run Tests

```bash
pytest test_main.py -v
```

All 6 tests should pass.

## 📈 Key Design Decisions

**Why Matrix Factorization over Item-Based CF?**
Matrix Factorization captures latent patterns (e.g. hidden genre preferences) 
that similarity-based methods miss, resulting in 8% lower RMSE.

**Why in-memory cache over Redis?**
For a single-instance deployment, a Python dict avoids network overhead. 
In a multi-instance production setup, this would be replaced with Redis 
for shared state across instances.

**Cold-start problem?**
New users with no rating history get popularity-based recommendations 
as a fallback — the same approach used by Netflix and YouTube for 
new user onboarding.

## 📁 Project Structure
recsys-project/

├── eda.py              # Data pipeline and EDA

├── train_model.py      # Model training (CF + MF)

├── main.py             # FastAPI serving layer

├── test_main.py        # Automated tests

├── build.py            # Deployment startup script

└── requirements.txt    # Dependencies


