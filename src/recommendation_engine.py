import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:

    def __init__(self):

        # ==========================
        # LOAD DATA
        # ==========================

        self.products = pd.read_csv(
            "data/processed/products.csv"
        )

        self.interactions = pd.read_csv(
            "data/processed/train.csv"
        )

        self.interactions[
            "timestamp"
        ] = pd.to_datetime(
            self.interactions["timestamp"]
        )


        # ==========================
        # USER-ITEM MATRIX
        # ==========================

        self.user_item = (
            self.interactions
            .pivot_table(
                index="user_id",
                columns="item_id",
                values="interaction_score",
                aggfunc="sum",
                fill_value=0
            )
        )


        # ==========================
        # ITEM SIMILARITY
        # ==========================

        item_user = self.user_item.T

        self.item_ids = (
            item_user.index.tolist()
        )

        self.item_similarity = (
            cosine_similarity(item_user)
        )


        self.item_index = {
            item_id: index
            for index, item_id
            in enumerate(self.item_ids)
        }


        # ==========================
        # CONTENT FEATURES
        # ==========================

        self.products["combined_text"] = (
            self.products[
                "item_name"
            ].fillna("")
            + " "
            + self.products[
                "category"
            ].fillna("")
            + " "
            + self.products[
                "brand"
            ].fillna("")
            + " "
            + self.products[
                "description"
            ].fillna("")
        )


        from sklearn.feature_extraction.text import (
            TfidfVectorizer
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                self.products[
                    "combined_text"
                ]
            )
        )

        self.content_similarity = (
            cosine_similarity(
                self.tfidf_matrix
            )
        )


        # ==========================
        # POPULARITY
        # ==========================

        self.popularity = (
            self.interactions
            .groupby("item_id")[
                "interaction_score"
            ]
            .sum()
        )


    # ==================================================
    # NORMALIZATION
    # ==================================================

    def normalize(self, values):

        values = np.asarray(
            values,
            dtype=float
        )

        if values.max() == values.min():

            return np.zeros_like(
                values
            )

        return (
            values - values.min()
        ) / (
            values.max() - values.min()
        )


    # ==================================================
    # COLLABORATIVE SCORE
    # ==================================================

    def collaborative_scores(
        self,
        user_id
    ):

        scores = np.zeros(
            len(self.products)
        )


        if user_id not in self.user_item.index:

            return scores


        user_vector = (
            self.user_item.loc[user_id]
        )


        for item_id, interaction_score in (
            user_vector[
                user_vector > 0
            ].items()
        ):

            if item_id not in self.item_index:

                continue


            item_index = (
                self.item_index[item_id]
            )


            similarity_vector = (
                self.item_similarity[
                    item_index
                ]
            )


            for index, candidate_id in enumerate(
                self.item_ids
            ):

                product_index = (
                    self.products.index[
                        self.products[
                            "item_id"
                        ] == candidate_id
                    ]
                )


                if len(product_index) == 0:

                    continue


                product_index = (
                    product_index[0]
                )


                scores[
                    product_index
                ] += (
                    similarity_vector[index]
                    * interaction_score
                )


        return scores


    # ==================================================
    # CONTENT SCORE
    # ==================================================

    def content_scores(
        self,
        user_id
    ):

        scores = np.zeros(
            len(self.products)
        )


        user_history = (
            self.interactions[
                self.interactions[
                    "user_id"
                ] == user_id
            ]
        )


        for _, row in user_history.iterrows():

            item_id = row["item_id"]


            product_index = (
                self.products.index[
                    self.products[
                        "item_id"
                    ] == item_id
                ]
            )


            if len(product_index) == 0:

                continue


            product_index = (
                product_index[0]
            )


            scores += (
                self.content_similarity[
                    product_index
                ]
                * row[
                    "interaction_score"
                ]
            )


        return scores


    # ==================================================
    # RECENCY SCORE
    # ==================================================

    def recency_scores(
        self,
        user_id
    ):

        scores = np.zeros(
            len(self.products)
        )


        user_history = (
            self.interactions[
                self.interactions[
                    "user_id"
                ] == user_id
            ]
        )


        if user_history.empty:

            return scores


        latest_time = (
            user_history[
                "timestamp"
            ].max()
        )


        for _, row in user_history.iterrows():

            days_old = (
                latest_time
                - row["timestamp"]
            ).total_seconds() / 86400


            recency_weight = np.exp(
                -days_old / 30
            )


            product_index = (
                self.products.index[
                    self.products[
                        "item_id"
                    ] == row["item_id"]
                ]
            )


            if len(product_index) == 0:

                continue


            product_index = (
                product_index[0]
            )


            scores[
                product_index
            ] += (
                recency_weight
                * row[
                    "interaction_score"
                ]
            )


        return scores


    # ==================================================
    # RECOMMEND
    # ==================================================

    def recommend(
        self,
        user_id,
        top_n=10
    ):

        # --------------------------
        # COLD START
        # --------------------------

        if user_id not in self.user_item.index:

            popularity = (
                self.products[
                    "item_id"
                ].map(
                    self.popularity
                ).fillna(0)
            )

            result = self.products.copy()

            result["score"] = (
                popularity
            )

            return (
                result
                .sort_values(
                    "score",
                    ascending=False
                )
                .head(top_n)
            )


        # --------------------------
        # CALCULATE SCORES
        # --------------------------

        collaborative = (
            self.collaborative_scores(
                user_id
            )
        )

        content = (
            self.content_scores(
                user_id
            )
        )

        recency = (
            self.recency_scores(
                user_id
            )
        )


        # --------------------------
        # NORMALIZE
        # --------------------------

        collaborative = (
            self.normalize(
                collaborative
            )
        )

        content = (
            self.normalize(
                content
            )
        )

        recency = (
            self.normalize(
                recency
            )
        )


        # --------------------------
        # HYBRID SCORE
        # --------------------------

        final_score = (
            0.60 * collaborative
            +
            0.30 * content
            +
            0.10 * recency
        )


        result = self.products.copy()

        result["score"] = final_score


        # --------------------------
        # REMOVE ALREADY INTERACTED
        # --------------------------

        interacted_items = set(
            self.interactions.loc[
                self.interactions[
                    "user_id"
                ] == user_id,
                "item_id"
            ]
        )


        result = result[
            ~result[
                "item_id"
            ].isin(
                interacted_items
            )
        ]


        # --------------------------
        # RANK
        # --------------------------

        result = (
            result
            .sort_values(
                "score",
                ascending=False
            )
            .head(top_n)
        )


        return result


# ======================================================
# TEST
# ======================================================

if __name__ == "__main__":

    engine = (
        RecommendationEngine()
    )

    recommendations = (
        engine.recommend(
            "U0001",
            10
        )
    )

    print(
        recommendations[
            [
                "item_id",
                "item_name",
                "category",
                "brand",
                "score"
            ]
        ]
    )