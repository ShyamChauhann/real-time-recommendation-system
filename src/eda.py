import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


users = pd.read_csv(
    "data/raw/users.csv"
)

products = pd.read_csv(
    "data/raw/products.csv"
)

interactions = pd.read_csv(
    "data/raw/interactions.csv"
)


print("\n========== DATASET INFORMATION ==========")

print(
    "Users:",
    users.shape
)

print(
    "Products:",
    products.shape
)

print(
    "Interactions:",
    interactions.shape
)


print("\n========== FIRST 5 USERS ==========")

print(
    users.head()
)


print("\n========== FIRST 5 PRODUCTS ==========")

print(
    products.head()
)


print("\n========== FIRST 5 INTERACTIONS ==========")

print(
    interactions.head()
)


print("\n========== INTERACTION DISTRIBUTION ==========")

print(
    interactions["interaction"]
    .value_counts()
)


print("\n========== PRODUCT CATEGORIES ==========")

print(
    products["category"]
    .value_counts()
)


print("\n========== MOST POPULAR PRODUCTS ==========")

print(
    interactions["item_id"]
    .value_counts()
    .head(10)
)


# Interaction plot

plt.figure(figsize=(8, 5))

sns.countplot(
    data=interactions,
    x="interaction"
)

plt.title(
    "User Interaction Distribution"
)

plt.tight_layout()

plt.savefig(
    "reports/interaction_distribution.png"
)

plt.show()


# Category distribution

plt.figure(figsize=(10, 5))

sns.countplot(
    data=products,
    y="category"
)

plt.title(
    "Product Category Distribution"
)

plt.tight_layout()

plt.savefig(
    "reports/category_distribution.png"
)

plt.show()