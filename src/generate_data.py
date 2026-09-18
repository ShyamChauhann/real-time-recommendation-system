import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ==============================
# CONFIGURATION
# ==============================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

NUM_USERS = 500
NUM_PRODUCTS = 1000
NUM_INTERACTIONS = 25000


# ==============================
# CREATE FOLDERS
# ==============================

os.makedirs("real-time recommendation system/data/raw", exist_ok=True)


# ==============================
# PRODUCT INFORMATION
# ==============================

categories = [
    "Electronics",
    "Fashion",
    "Sports",
    "Books",
    "Home",
    "Beauty",
    "Gaming",
    "Accessories"
]

brands = [
    "Nike",
    "Adidas",
    "Samsung",
    "Apple",
    "Sony",
    "Puma",
    "Dell",
    "HP",
    "Lenovo",
    "Amazon Basics"
]

product_names = {
    "Electronics": [
        "Smartphone",
        "Laptop",
        "Tablet",
        "Smart TV",
        "Wireless Speaker",
        "Bluetooth Earphones",
        "Power Bank"
    ],

    "Fashion": [
        "T-Shirt",
        "Jeans",
        "Jacket",
        "Hoodie",
        "Sneakers",
        "Shirt",
        "Track Pants"
    ],

    "Sports": [
        "Running Shoes",
        "Gym Bag",
        "Sports Watch",
        "Yoga Mat",
        "Football",
        "Cricket Bat",
        "Fitness Gloves"
    ],

    "Books": [
        "Python Programming",
        "Machine Learning",
        "Data Science",
        "Deep Learning",
        "Database Systems",
        "Algorithms",
        "Artificial Intelligence"
    ],

    "Home": [
        "Table Lamp",
        "Office Chair",
        "Bedsheet",
        "Wall Clock",
        "Storage Box",
        "Coffee Mug",
        "Cushion"
    ],

    "Beauty": [
        "Face Wash",
        "Shampoo",
        "Perfume",
        "Moisturizer",
        "Sunscreen",
        "Face Cream",
        "Hair Oil"
    ],

    "Gaming": [
        "Gaming Laptop",
        "Gaming Mouse",
        "Gaming Keyboard",
        "Gaming Headset",
        "Game Controller",
        "Gaming Chair",
        "Mouse Pad"
    ],

    "Accessories": [
        "Backpack",
        "Wallet",
        "Sunglasses",
        "Smart Watch",
        "Phone Case",
        "Travel Bag",
        "Belt"
    ]
}


# ==============================
# GENERATE USERS
# ==============================

users = []

for i in range(1, NUM_USERS + 1):

    users.append({
        "user_id": f"U{i:04d}"
    })

users_df = pd.DataFrame(users)


# ==============================
# GENERATE PRODUCTS
# ==============================

products = []

for i in range(1, NUM_PRODUCTS + 1):

    category = random.choice(categories)
    brand = random.choice(brands)

    base_name = random.choice(
        product_names[category]
    )

    item_name = f"{brand} {base_name}"

    price = round(
        random.uniform(200, 100000),
        2
    )

    description = (
        f"{brand} {base_name} "
        f"designed for {category.lower()} users"
    )

    products.append({
        "item_id": f"P{i:04d}",
        "item_name": item_name,
        "category": category,
        "brand": brand,
        "price": price,
        "description": description
    })

products_df = pd.DataFrame(products)


# ==============================
# CREATE USER PREFERENCES
# ==============================

user_preferences = {}

for user_id in users_df["user_id"]:

    preferred_categories = random.sample(
        categories,
        random.choice([1, 2])
    )

    user_preferences[user_id] = preferred_categories


# ==============================
# GENERATE INTERACTIONS
# ==============================

interaction_types = [
    "view",
    "click",
    "cart",
    "purchase"
]

interaction_scores = {
    "view": 1,
    "click": 2,
    "cart": 3,
    "purchase": 5
}

start_date = datetime(2026, 1, 1)

interactions = []


for _ in range(NUM_INTERACTIONS):

    user_id = random.choice(
        users_df["user_id"].tolist()
    )

    preferred_categories = user_preferences[user_id]

    # Most interactions come from preferred categories
    if random.random() < 0.82:

        candidate_products = products_df[
            products_df["category"].isin(
                preferred_categories
            )
        ]

    else:

        candidate_products = products_df

    product = candidate_products.sample(
        1
    ).iloc[0]

    interaction = random.choices(
        interaction_types,
        weights=[60, 25, 10, 5],
        k=1
    )[0]

    random_days = random.randint(0, 240)

    random_seconds = random.randint(
        0,
        86399
    )

    timestamp = (
        start_date
        + timedelta(days=random_days)
        + timedelta(seconds=random_seconds)
    )

    interactions.append({
        "user_id": user_id,
        "item_id": product["item_id"],
        "interaction": interaction,
        "timestamp": timestamp,
        "interaction_score": interaction_scores[
            interaction
        ]
    })


interactions_df = pd.DataFrame(
    interactions
)


# ==============================
# SAVE DATA
# ==============================

users_df.to_csv(
    "data/raw/users.csv",
    index=False
)

products_df.to_csv(
    "data/raw/products.csv",
    index=False
)

interactions_df.to_csv(
    "data/raw/interactions.csv",
    index=False
)


# ==============================
# OUTPUT
# ==============================

print("=" * 60)
print("DATA GENERATION COMPLETED")
print("=" * 60)

print(
    f"Users        : {len(users_df)}"
)

print(
    f"Products     : {len(products_df)}"
)

print(
    f"Interactions : {len(interactions_df)}"
)

print("\nInteraction distribution:")

print(
    interactions_df["interaction"]
    .value_counts()
)

print("\nFiles created:")

print("data/raw/users.csv")
print("data/raw/products.csv")
print("data/raw/interactions.csv")