import os
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


os.makedirs(
    "models",
    exist_ok=True
)


# ==============================
# LOAD DATA
# ==============================

interactions = pd.read_csv(
    "data/processed/train.csv"
)


# ==============================
# USER-ITEM MATRIX
# ==============================

user_item_matrix = (
    interactions
    .pivot_table(
        index="user_id",
        columns="item_id",
        values="interaction_score",
        aggfunc="sum",
        fill_value=0
    )
)


print(
    "User-item matrix:",
    user_item_matrix.shape
)


# ==============================
# ITEM-ITEM SIMILARITY
# ==============================

item_user_matrix = (
    user_item_matrix.T
)


item_similarity = cosine_similarity(
    item_user_matrix
)


item_ids = (
    item_user_matrix.index.tolist()
)


# ==============================
# SAVE MODEL
# ==============================

joblib.dump(
    user_item_matrix,
    "models/user_item_matrix.pkl"
)

joblib.dump(
    item_similarity,
    "models/item_similarity.pkl"
)

joblib.dump(
    item_ids,
    "models/item_ids.pkl"
)


print(
    "Collaborative filtering model saved."
)