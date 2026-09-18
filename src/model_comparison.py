import numpy as np
import pandas as pd

# from src.recommendation_engine import (
#     RecommendationEngine
# )
from src.recommendation_engine import RecommendationEngine


K = 10


train = pd.read_csv(
    "data/processed/train.csv"
)

test = pd.read_csv(
    "data/processed/test.csv"
)


engine = (
    RecommendationEngine()
)


# ==========================================
# POPULARITY
# ==========================================

popularity = (
    train
    .groupby("item_id")[
        "interaction_score"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)


def popular_recommendations(
    user_id
):

    already_seen = set(
        train.loc[
            train["user_id"] == user_id,
            "item_id"
        ]
    )


    products = (
        engine.products.copy()
    )


    products["score"] = (
        products["item_id"]
        .map(popularity)
        .fillna(0)
    )


    products = products[
        ~products[
            "item_id"
        ].isin(
            already_seen
        )
    ]


    return (
        products
        .sort_values(
            "score",
            ascending=False
        )
        .head(K)
    )


# ==========================================
# METRIC FUNCTION
# ==========================================

def metrics(
    recommended,
    relevant
):

    items = recommended[
        "item_id"
    ].tolist()


    hits = sum(
        item in relevant
        for item in items
    )


    precision = hits / K

    recall = (
        hits / len(relevant)
        if relevant
        else 0
    )


    hit_rate = (
        1
        if hits > 0
        else 0
    )


    dcg = 0

    for rank, item in enumerate(
        items,
        1
    ):

        if item in relevant:

            dcg += (
                1
                / np.log2(
                    rank + 1
                )
            )


    ideal = min(
        len(relevant),
        K
    )


    idcg = sum(
        1 / np.log2(i + 1)
        for i in range(
            1,
            ideal + 1
        )
    )


    ndcg = (
        dcg / idcg
        if idcg
        else 0
    )


    return (
        precision,
        recall,
        hit_rate,
        ndcg
    )


# ==========================================
# EVALUATE BOTH
# ==========================================

results = []


for user_id, group in test.groupby(
    "user_id"
):

    relevant = set(
        group["item_id"]
    )


    # Popularity

    popular = (
        popular_recommendations(
            user_id
        )
    )


    p, r, h, n = metrics(
        popular,
        relevant
    )


    results.append({
        "model": "Popularity",
        "precision": p,
        "recall": r,
        "hit_rate": h,
        "ndcg": n
    })


    # Hybrid

    hybrid = (
        engine.recommend(
            user_id,
            K
        )
    )


    p, r, h, n = metrics(
        hybrid,
        relevant
    )


    results.append({
        "model": "Hybrid",
        "precision": p,
        "recall": r,
        "hit_rate": h,
        "ndcg": n
    })


results_df = pd.DataFrame(
    results
)


summary = (
    results_df
    .groupby("model")
    .mean(numeric_only=True)
)


print(
    "\n========== MODEL COMPARISON =========="
)

print(summary)


summary.to_csv(
    "reports/model_comparison.csv"
)