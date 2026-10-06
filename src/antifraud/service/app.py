import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from antifraud import db
from antifraud.config import settings
from antifraud.preprocessing import FeatureBuilder, LateTransactionError


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
    fraud: bool
    model_version: str
    request_id: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(settings.model_path)
    app.state.model = bundle["model"]
    app.state.meta = bundle["metadata"]
    app.state.version = bundle["metadata"]["model_version"]
    app.state.features = FeatureBuilder()

    db.init()
    yield
    app.state.model = None


app = FastAPI(title="antifraud-system", version="1.0", lifespan=lifespan)


logger = logging.getLogger(__name__)


def _safe_payload(body) -> dict:
    if isinstance(body, dict):
        return body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")

    return {"raw": str(body)[:2000]}


@app.exception_handler(LateTransactionError)
async def late_transaction_handler(request: Request, exc: LateTransactionError):
    request_id = str(uuid.uuid4())

    body = await request.json()

    response = JSONResponse(
        status_code=422,
        content={"detail": str(exc), "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )

    response.background = BackgroundTask(
        db.save_prediction,
        request_id,
        getattr(app.state, "version", "unknown"),
        _safe_payload(body),
        None,
        None,
        409,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    request_id = str(uuid.uuid4())

    response = await request_validation_exception_handler(request, exc)
    response.headers["X-Request-ID"] = request_id

    response.background = BackgroundTask(
        db.save_prediction,
        request_id,
        getattr(app.state, "version", "unknown"),
        _safe_payload(exc.body),
        None,
        None,
        422,
    )
    return response


@app.middleware("http")
async def unhandled_error_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        request_id = str(uuid.uuid4())
        logger.exception("Unhandled error, request_id=%s path=%s", request_id, request.url.path)

        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )


        if request.url.path.startswith("/v1/predict"):
            try:
                raw = await request.body()
                body = json.loads(raw)
            except Exception:
                body = raw if "raw" in locals() else b""
            response.background = BackgroundTask(
                db.save_prediction,
                request_id,
                getattr(app.state, "version", "unknown"),
                _safe_payload(body),
                None,
                None,
                500,
            )
        return response


@app.get("/health")
def health():
    return {"status": "ok", "model_version": getattr(app.state, "version", "unknown")}


@app.get("/ready")
def ready():
    if getattr(app.state, "model", "None") is  None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "ready"}


@app.post("/v1/predict")
def predict(x: Features, bg: BackgroundTasks) -> Prediction:
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    
    payload = app.state.features.build([x.model_dump()])[0]
    
    frame = pd.DataFrame([payload]).reindex(columns=app.state.meta["features"])

    score = float(app.state.model.predict_proba(frame)[0, 1])

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    bg.add_task(
        db.save_prediction,
        request_id,
        app.state.version,
        payload,
        score,
        latency_ms,
        200
    )

    fraud = score >= app.state.meta["threshold"]

    return Prediction(
        score=score,
        fraud=fraud,
        model_version=app.state.version,
        request_id=request_id,
        latency_ms=latency_ms
    )


@app.post("/v1/predict/batch")
def predict_batch(x: BatchFeatures, bg: BackgroundTasks) -> list[Prediction]:
    t0 = time.perf_counter()

    payloads = app.state.features.build([row.model_dump() for row in x.rows])

    frame = (
        pd.DataFrame(payloads)
        .reindex(columns=app.state.meta["features"])
    )

    scores = app.state.model.predict_proba(frame)[:, 1]
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    results = []

    for payload, score in zip(payloads, scores, strict=True):
        request_id = str(uuid.uuid4())
        score = float(score)
        fraud = score >= app.state.meta["threshold"]
        bg.add_task(
            db.save_prediction,
            request_id,
            app.state.version,
            payload,
            score,
            latency_ms,
            200
        )

        results.append(
            Prediction(
                score=score,
                fraud=fraud,
                model_version=app.state.version,
                request_id=request_id,
                latency_ms=latency_ms,
            )
        )

    return results
