import pandas as pd


interactions = pd.read_csv(
    "data/processed/train.csv"
)

products = pd.read_csv(
    "data/processed/products.csv"
)


popularity = (
    interactions
    .groupby("item_id")[
        "interaction_score"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)


print("\nMost Popular Products:")

popular_products = (
    products[
        products["item_id"].isin(
            popularity.head(10).index
        )
    ]
    .copy()
)

popular_products["score"] = (
    popular_products["item_id"]
    .map(popularity)
)

popular_products = (
    popular_products
    .sort_values(
        "score",
        ascending=False
    )
)


print(
    popular_products[
        [
            "item_id",
            "item_name",
            "category",
            "score"
        ]
    ]
)