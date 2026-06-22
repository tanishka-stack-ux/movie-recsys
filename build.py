"""
build.py -- runs on Render during deployment.
Downloads the MovieLens dataset and trains the model
so model_artifacts.pkl exists before the API starts.
"""

import os
import urllib.request
import zipfile

# Download dataset if not already present
if not os.path.exists("ml-100k"):
    print("Downloading MovieLens 100k dataset...")
    urllib.request.urlretrieve(
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "ml-100k.zip"
    )
    with zipfile.ZipFile("ml-100k.zip", "r") as z:
        z.extractall(".")
    print("Dataset downloaded and extracted.")
else:
    print("Dataset already exists, skipping download.")

# Train model if artifacts not already present
if not os.path.exists("model_artifacts.pkl"):
    print("Training model...")
    import train_model
    print("Model trained and saved.")
else:
    print("Model artifacts already exist, skipping training.")