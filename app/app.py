import requests
import pandas as pd
import streamlit as st


API_URL = (
    "http://127.0.0.1:8000"
)


# ==========================================
# PAGE
# ==========================================

st.set_page_config(
    page_title="Recommendation System",
    page_icon="🛍️",
    layout="wide"
)


st.title(
    "🛍️ Real-Time Personalized Recommendation System"
)

st.write(
    "Hybrid recommendation using collaborative filtering, "
    "content-based filtering and recency."
)


# ==========================================
# LOAD DATA
# ==========================================

users = pd.read_csv(
    "/Users/shyamchauhan/Desktop/home/codes/real-time recommendation system/data/processed/users.csv"
)

products = pd.read_csv(
    "/Users/shyamchauhan/Desktop/home/codes/real-time recommendation system/data/processed/products.csv"
)

# ==========================================
# USER
# ==========================================

user_id = st.selectbox(
    "Select User",
    users["user_id"].tolist()
)


top_n = st.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# ==========================================
# RECOMMENDATIONS
# ==========================================

if st.button(
    "Get Recommendations"
):

    try:

        response = requests.get(
            f"{API_URL}/recommendations/{user_id}",
            params={
                "n": top_n
            },
            timeout=10
        )


        response.raise_for_status()


        recommendations = (
            response.json()
        )


        if not recommendations:

            st.warning(
                "No recommendations found."
            )


        else:

            df = pd.DataFrame(
                recommendations
            )


            df["score"] = (
                df["score"] * 100
            ).round(2)


            st.subheader(
                "Recommended For You"
            )


            st.dataframe(
                df[
                    [
                        "item_id",
                        "item_name",
                        "category",
                        "brand",
                        "price",
                        "score"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


    except Exception as error:

        st.error(
            "Could not connect to FastAPI. "
            "Make sure the API is running."
        )

        st.error(
            str(error)
        )


# ==========================================
# INTERACTION
# ==========================================

st.divider()

st.subheader(
    "Record User Interaction"
)


item_id = st.selectbox(
    "Select Product",
    products["item_id"].tolist()
)


interaction = st.selectbox(
    "Interaction Type",
    [
        "view",
        "click",
        "cart",
        "purchase"
    ]
)


if st.button(
    "Record Interaction"
):

    try:

        response = requests.post(
            f"{API_URL}/interaction",

            json={
                "user_id": user_id,
                "item_id": item_id,
                "interaction": interaction
            },

            timeout=10
        )


        response.raise_for_status()


        st.success(
            "Interaction recorded successfully."
        )


        st.info(
            "Click 'Get Recommendations' again "
            "to see the updated recommendations."
        )


    except Exception as error:

        st.error(
            "Could not record interaction."
        )

        st.error(
            str(error)
        )