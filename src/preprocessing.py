import os
import pandas as pd


os.makedirs(
    "data/processed",
    exist_ok=True
)


# ==============================
# LOAD DATA
# ==============================

users = pd.read_csv(
    "data/raw/users.csv"
)

products = pd.read_csv(
    "data/raw/products.csv"
)

interactions = pd.read_csv(
    "data/raw/interactions.csv"
)


# ==============================
# CLEAN INTERACTIONS
# ==============================

interactions["timestamp"] = pd.to_datetime(
    interactions["timestamp"],
    errors="coerce"
)


# Remove invalid rows

interactions = interactions.dropna(
    subset=[
        "user_id",
        "item_id",
        "timestamp"
    ]
)


# Remove duplicates

interactions = interactions.drop_duplicates()


# Recalculate interaction scores

score_map = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5
}

interactions[
    "interaction_score"
] = interactions[
    "interaction"
].map(
    score_map
).fillna(1)


# Sort chronologically

interactions = interactions.sort_values(
    "timestamp"
)


# ==============================
# SAVE CLEAN DATA
# ==============================

users.to_csv(
    "data/processed/users.csv",
    index=False
)

products.to_csv(
    "data/processed/products.csv",
    index=False
)

interactions.to_csv(
    "data/processed/interactions.csv",
    index=False
)


# ==============================
# TIME-BASED TRAIN/TEST SPLIT
# ==============================

train_parts = []
test_parts = []


for user_id, group in interactions.groupby(
    "user_id"
):

    group = group.sort_values(
        "timestamp"
    )

    if len(group) < 5:

        train_parts.append(
            group
        )

        continue

    split_index = int(
        len(group) * 0.8
    )

    train_parts.append(
        group.iloc[:split_index]
    )

    test_parts.append(
        group.iloc[split_index:]
    )


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)


# ==============================
# SAVE TRAIN/TEST
# ==============================

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)


print("=" * 60)

print("PREPROCESSING COMPLETED")

print("=" * 60)

print(
    "Train:",
    train_df.shape
)

print(
    "Test:",
    test_df.shape
)