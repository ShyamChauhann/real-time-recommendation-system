import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


os.makedirs(
    "models",
    exist_ok=True
)


# ==============================
# LOAD PRODUCTS
# ==============================

products = pd.read_csv(
    "data/processed/products.csv"
)


# ==============================
# COMBINE TEXT FEATURES
# ==============================

products["combined_text"] = (
    products["item_name"].fillna("")
    + " "
    + products["category"].fillna("")
    + " "
    + products["brand"].fillna("")
    + " "
    + products["description"].fillna("")
)


# ==============================
# TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    stop_words="english"
)


tfidf_matrix = vectorizer.fit_transform(
    products["combined_text"]
)


print(
    "TF-IDF matrix:",
    tfidf_matrix.shape
)


# ==============================
# SIMILARITY
# ==============================

content_similarity = cosine_similarity(
    tfidf_matrix
)


# ==============================
# SAVE
# ==============================

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    tfidf_matrix,
    "models/tfidf_matrix.pkl"
)

joblib.dump(
    content_similarity,
    "models/content_similarity.pkl"
)


print(
    "Content-based model saved."
)