import os
import joblib

from src.recommendation_engine import (
    RecommendationEngine
)


os.makedirs(
    "models",
    exist_ok=True
)


engine = (
    RecommendationEngine()
)


joblib.dump(
    engine.user_item,
    "models/user_item_matrix.pkl"
)

joblib.dump(
    engine.item_similarity,
    "models/item_similarity.pkl"
)

joblib.dump(
    engine.item_ids,
    "models/item_ids.pkl"
)

joblib.dump(
    engine.vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    engine.tfidf_matrix,
    "models/tfidf_matrix.pkl"
)

joblib.dump(
    engine.content_similarity,
    "models/content_similarity.pkl"
)


print(
    "All recommendation artifacts saved."
)