import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask

from antifraud import db
from antifraud.config import settings
from antifraud.model_store import load_model
from antifraud.service.explain import Explainer
from antifraud.service.make_decision import decide
from antifraud.service.models import BatchFeatures, Features, Prediction
from antifraud.service.preprocessing import FeatureBuilder
from antifraud.service.web import router as web_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.meta, app.state.version = load_model()
    app.state.features = FeatureBuilder()
    app.state.explainer = Explainer(app.state.model, app.state.meta["features"])
    app.state.explainer_top_n = settings.explainer_top_n

    db.init()
    yield
    app.state.model = None


app = FastAPI(title="antifraud-system", version="1.0", lifespan=lifespan)
app.include_router(web_router)

logger = logging.getLogger(__name__)


def _safe_payload(body) -> dict:
    if isinstance(body, dict):
        return body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="replace")

    return {"raw": str(body)[:2000]}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    transaction_id = str(uuid.uuid4())

    response = await request_validation_exception_handler(request, exc)
    response.headers["X-Transaction-ID"] = transaction_id

    response.background = BackgroundTask(
        db.save_prediction,
        transaction_id,
        getattr(app.state, "version", "unknown"),
        _safe_payload(exc.body),
        None,
        None,
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
        transaction_id = str(uuid.uuid4())
        logger.exception(
            "Unhandled error, transaction_id=%s path=%s",
            transaction_id,
            request.url.path
        )

        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "transaction_id": transaction_id},
            headers={"X-Transaction-ID": transaction_id},
        )


        if request.url.path.startswith("/v1/predict"):
            try:
                raw = await request.body()
                body = json.loads(raw)
            except Exception:
                body = raw if "raw" in locals() else b""
            response.background = BackgroundTask(
                db.save_prediction,
                transaction_id,
                getattr(app.state, "version", "unknown"),
                _safe_payload(body),
                None,
                None,
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


def log_shap(explainer, frame, transaction_ids, model_version, top_n):
    try:
        pairs = explainer.explain(frame, top_n=top_n)
        db.save_shap(
            transaction_ids,
            [{name: float(v) for name, v in row} for row in pairs],
            model_version,
        )
    except Exception:
        logger.exception("shap logging failed")


@app.post("/v1/predict")
def predict(x: Features, bg: BackgroundTasks) -> Prediction:
    t0 = time.perf_counter()
    transaction_id = str(uuid.uuid4())

    row = x.model_dump()
    row["transaction_id"] = transaction_id
    payloads = app.state.features.build([row])
    
    frame = pd.DataFrame(payloads)
    frame = frame.drop("transaction_id", axis=1).reindex(columns=app.state.meta["features"])

    model = app.state.model
    T_low = app.state.meta["T_low"]
    T_hight = app.state.meta["T_hight"]
    top_n = app.state.explainer_top_n

    score = float(model.predict_proba(frame)[0, 1])
    decision = decide(score, T_low=T_low, T_hight=T_hight)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    bg.add_task(
        db.save_prediction,
        transaction_id,
        app.state.version,
        frame.iloc[0].to_dict(),
        score,
        decision,
        True if decision == "REVIEW" else None,
        latency_ms,
        200
    )

    bg.add_task(
        log_shap,
        app.state.explainer,
        frame.copy(),
        [transaction_id],
        app.state.version,
        top_n
    )
    if decision != "REVIEW":
        bg.add_task(
            db.save_fraud_label,
            transaction_id=transaction_id,
            fraud=(decision == "BLOCK"),
            fraud_type="MODEL"
        )

    return Prediction(
        score=score,
        decision=decision,
        model_version=app.state.version,
        transaction_id=transaction_id,
        latency_ms=latency_ms
    )


@app.post("/v1/predict/batch")
def predict_batch(x: BatchFeatures, bg: BackgroundTasks) -> list[Prediction]:
    t0 = time.perf_counter()

    rows = []
    ids = []
    for row in x.rows:
        transaction_id = str(uuid.uuid4())
        ids.append(transaction_id)
        
        row = row.model_dump()
        row["transaction_id"] = transaction_id
        rows.append(row)

    payloads = app.state.features.build(rows)
    
    frame = pd.DataFrame(payloads)
    frame = frame.drop("transaction_id", axis=1).reindex(columns=app.state.meta["features"])

    model = app.state.model
    T_low = app.state.meta["T_low"]
    T_hight = app.state.meta["T_hight"]
    top_n = app.state.explainer_top_n

    scores = model.predict_proba(frame)[:, 1]
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    results = []
    for transaction_id, (_, f_row), score in zip(
        ids,
        frame.iterrows(),
        scores,
        strict=True,
    ):
        score = float(score)
        decision = decide(score, T_low=T_low, T_hight=T_hight)
        bg.add_task(
            db.save_prediction,
            transaction_id,
            app.state.version,
            f_row.to_dict(),
            score,
            decision,
            True if decision == "REVIEW" else None,
            latency_ms,
            200
        )

        if decision != "REVIEW":
            bg.add_task(
                db.save_fraud_label,
                transaction_id=transaction_id,
                fraud=(decision == "BLOCK"),
                fraud_type="MODEL"
            )

        results.append(
            Prediction(
                score=score,
                decision=decision,
                model_version=app.state.version,
                transaction_id=transaction_id,
                latency_ms=latency_ms,
            )
        )

    bg.add_task(
        log_shap,
        app.state.explainer,
        frame.copy(),
        ids,
        app.state.version,
        top_n,
    )

    return results
