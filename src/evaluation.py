import numpy as np
import pandas as pd

from src.recommendation_engine import (
    RecommendationEngine
)


K = 10

# ==========================================
# METRICS
# ==========================================

def calculate_metrics(
    recommendations,
    relevant_items
):

    recommended_items = (
        recommendations[
            "item_id"
        ].tolist()
    )


    hits = [
        item
        for item in recommended_items
        if item in relevant_items
    ]


    hit_count = len(hits)


    # Precision@K

    precision = (
        hit_count / K
    )


    # Recall@K

    recall = (
        hit_count
        / len(relevant_items)
        if len(relevant_items) > 0
        else 0
    )


    # Hit Rate@K

    hit_rate = (
        1
        if hit_count > 0
        else 0
    )


    # NDCG@K

    dcg = 0.0

    for rank, item in enumerate(
        recommended_items,
        start=1
    ):

        if item in relevant_items:

            dcg += (
                1
                / np.log2(
                    rank + 1
                )
            )


    ideal_count = min(
        len(relevant_items),
        K
    )


    idcg = sum(
        1 / np.log2(i + 1)
        for i in range(
            1,
            ideal_count + 1
        )
    )


    ndcg = (
        dcg / idcg
        if idcg > 0
        else 0
    )


    return (
        precision,
        recall,
        hit_rate,
        ndcg
    )


# ==========================================
# EVALUATION
# ==========================================

def evaluate():
    
    train = pd.read_csv(
        "data/processed/train.csv"
    )

    test = pd.read_csv(
        "data/processed/test.csv"
    )


    engine = (
        RecommendationEngine()
    )


    results = []


    for user_id, group in test.groupby(
        "user_id"
    ):

        relevant_items = set(
            group["item_id"]
        )


        recommendations = (
            engine.recommend(
                user_id,
                K
            )
        )


        precision, recall, hit_rate, ndcg = (
            calculate_metrics(
                recommendations,
                relevant_items
            )
        )


        results.append({
            "user_id": user_id,
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
            "ndcg": ndcg
        })


    results_df = pd.DataFrame(
        results
    )


    print("\n========== RESULTS ==========")

    print(
        "Precision@10:",
        results_df["precision"].mean()
    )

    print(
        "Recall@10:",
        results_df["recall"].mean()
    )

    print(
        "HitRate@10:",
        results_df["hit_rate"].mean()
    )

    print(
        "NDCG@10:",
        results_df["ndcg"].mean()
    )


    results_df.to_csv(
        "reports/evaluation_results.csv",
        index=False
    )


if __name__ == "__main__":

    evaluate()
