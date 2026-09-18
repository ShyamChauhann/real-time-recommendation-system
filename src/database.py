import os
import sqlite3
import pandas as pd


DB_PATH = (
    "data/recommendations.db"
)


def get_connection():

    os.makedirs(
        "data",
        exist_ok=True
    )

    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )


def initialize_database():

    connection = (
        get_connection()
    )

    cursor = connection.cursor()


    # Users

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY
        )
    """)


    # Products

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            item_id TEXT PRIMARY KEY,
            item_name TEXT,
            category TEXT,
            brand TEXT,
            price REAL,
            description TEXT
        )
    """)


    # Interactions

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            item_id TEXT,
            interaction TEXT,
            timestamp TEXT,
            interaction_score REAL
        )
    """)


    connection.commit()


    # Insert initial data only if empty

    users_count = cursor.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]


    if users_count == 0:

        users = pd.read_csv(
            "data/processed/users.csv"
        )

        users.to_sql(
            "users",
            connection,
            if_exists="append",
            index=False
        )


    products_count = cursor.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0]


    if products_count == 0:

        products = pd.read_csv(
            "data/processed/products.csv"
        )

        products.to_sql(
            "products",
            connection,
            if_exists="append",
            index=False
        )


    interactions_count = cursor.execute(
        "SELECT COUNT(*) FROM interactions"
    ).fetchone()[0]


    if interactions_count == 0:

        interactions = pd.read_csv(
            "data/processed/interactions.csv"
        )


        interactions[
            [
                "user_id",
                "item_id",
                "interaction",
                "timestamp",
                "interaction_score"
            ]
        ].to_sql(
            "interactions",
            connection,
            if_exists="append",
            index=False
        )


    connection.commit()

    connection.close()


def add_interaction(
    user_id,
    item_id,
    interaction,
    score
):

    connection = (
        get_connection()
    )


    connection.execute(
        """
        INSERT INTO interactions
        (
            user_id,
            item_id,
            interaction,
            timestamp,
            interaction_score
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            item_id,
            interaction,
            pd.Timestamp.utcnow().isoformat(),
            score
        )
    )


    connection.commit()

    connection.close()


def get_recent_interactions(
    user_id,
    limit=10
):

    connection = (
        get_connection()
    )


    query = """
        SELECT *
        FROM interactions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """


    result = pd.read_sql_query(
        query,
        connection,
        params=(
            user_id,
            limit
        )
    )


    connection.close()


    return result