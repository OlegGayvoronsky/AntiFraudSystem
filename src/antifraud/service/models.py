from typing import Literal

from pydantic import BaseModel, Field


class Features(BaseModel):
    model_config = {"extra": "forbid"}

    step: int = Field(ge=0)
    customer: str
    age: Literal["0", "1", "2", "3", "4", "5", "6", "U"]
    gender: Literal["M", "F", "E", "U"]
    merchant: str
    category: Literal[
        'es_transportation', 'es_food', 'es_health',
        'es_wellnessandbeauty', 'es_fashion', 'es_barsandrestaurants',
        'es_hyper', 'es_sportsandtoys', 'es_tech',
        'es_home', 'es_hotelservices', 'es_otherservices',
       'es_contents', 'es_travel', 'es_leisure'
    ]
    amount: float = Field(ge=0)


class BatchFeatures(BaseModel):
    rows: list[Features] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    score: float
    decision: Literal["ФРОД", "НЕ ФРОД", "НЕОПРЕДЕЛЕНО"]
    model_version: str
    request_id: str
    latency_ms: float