from fastapi import (
    FastAPI,
    HTTPException
)

from pydantic import BaseModel

from src.database import (
    initialize_database,
    add_interaction
)

from src.recommendation_engine import (
    RecommendationEngine
)


# ==========================================
# APPLICATION
# ==========================================

app = FastAPI(
    title="Real-Time Recommendation API",
    version="1.0"
)


# ==========================================
# INITIALIZE
# ==========================================

initialize_database()

engine = (
    RecommendationEngine()
)


# ==========================================
# REQUEST MODEL
# ==========================================

class InteractionRequest(
    BaseModel
):

    user_id: str
    item_id: str
    interaction: str


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ==========================================
# RECOMMENDATIONS
# ==========================================

@app.get(
    "/recommendations/{user_id}"
)
def recommendations(
    user_id: str,
    n: int = 10
):

    try:

        result = (
            engine.recommend(
                user_id,
                n
            )
        )


        return result[
            [
                "item_id",
                "item_name",
                "category",
                "brand",
                "price",
                "score"
            ]
        ].to_dict(
            orient="records"
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ==========================================
# ADD INTERACTION
# ==========================================

@app.post("/interaction")
def record_interaction(
    request: InteractionRequest
):

    score_map = {
        "view": 1,
        "click": 2,
        "cart": 3,
        "purchase": 5
    }


    if (
        request.interaction
        not in score_map
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid interaction"
        )


    try:

        # Update in-memory recommendation engine

        engine.interactions = (
            __import__(
                "pandas"
            ).concat(
                [
                    engine.interactions,

                    __import__(
                        "pandas"
                    ).DataFrame(
                        [
                            {
                                "user_id":
                                    request.user_id,

                                "item_id":
                                    request.item_id,

                                "interaction":
                                    request.interaction,

                                "timestamp":
                                    __import__(
                                        "pandas"
                                    ).Timestamp.utcnow(),

                                "interaction_score":
                                    score_map[
                                        request.interaction
                                    ]
                            }
                        ]
                    )
                ],
                ignore_index=True
            )
        )


        # Store in database

        add_interaction(
            request.user_id,
            request.item_id,
            request.interaction,
            score_map[
                request.interaction
            ]
        )


        return {
            "message":
                "Interaction recorded successfully"
        }


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )