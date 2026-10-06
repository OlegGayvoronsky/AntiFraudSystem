"""
Веб-интерфейс ручного и batch-скоринга поверх /v1/predict-логики.

Ожидается, что в app.state уже есть (как в lifespan основного API):
    app.state.pipeline  -- обученный CatBoostClassifier (без sklearn Pipeline)
    app.state.meta      -- {"features": [...], "threshold": b, "model_version": ...}
    app.state.features  -- antifraud.preprocessing.FeatureBuilder
    app.state.explainer -- antifraud.explain.Explainer (создать рядом с pipeline в lifespan)
"""
from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from antifraud.config import settings
from antifraud.service.explain import feature_label
from antifraud.service.make_decision import MANUAL_REVIEW_THRESHOLD_A, decide
from antifraud.service.models import Features

router = APIRouter()

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

RAW_BATCH_COLUMNS = ["step", "customer", "age", "gender", "merchant", "category", "amount"]


def _final_frame(request: Request, rows: list[dict]) -> pd.DataFrame:
    """Строит фрейм с историческими признаками в порядке app.state.meta['features']."""
    enriched = request.app.state.features.build(rows)
    features = request.app.state.meta["features"]
    return pd.DataFrame(enriched).reindex(columns=features), enriched


@router.get("/ui", response_class=HTMLResponse)
def scoring_page(request: Request):
    return templates.TemplateResponse(
        "scoring.html",
        {"request": request, "single_result": None, "batch_result": None, "error": None},
    )


@router.post("/ui/score", response_class=HTMLResponse)
def score_single(
    request: Request,
    step: int = Form(...),
    customer: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    merchant: str = Form(...),
    category: str = Form(...),
    amount: float = Form(...),
):
    raw = {
        "step": step, "customer": customer, "age": age, "gender": gender,
        "merchant": merchant, "category": category, "amount": amount,
    }

    try:
        Features(**raw)
    except ValidationError as exc:
        return templates.TemplateResponse(
            "scoring.html",
            {"request": request, "single_result": None, "batch_result": None,
             "error": f"Некорректные данные: {exc.errors()[0]['msg']}"},
        )

    t0 = time.perf_counter()
    frame, enriched_rows = _final_frame(request, [raw])
    
    model = request.app.state.model
    threshold_b = request.app.state.meta["threshold"]
    top_n = request.app.state.explainer_top_n

    
    score = float(model.predict_proba(frame)[0, 1])
    cls, decision, css_class = decide(score, threshold_b)

    reasons = request.app.state.explainer.explain(frame, top_n=top_n)[0]
    reasons_fmt = [(feature_label(name), value) for name, value in reasons]

    final_features = {
        feature_label(k): v for k, v in enriched_rows[0].items()
        if k in request.app.state.meta["features"]
    }

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    single_result = {
        "score": score,
        "cls": cls,
        "decision": decision,
        "css_class": css_class,
        "threshold_a": MANUAL_REVIEW_THRESHOLD_A,
        "threshold_b": threshold_b,
        "reasons": reasons_fmt,
        "final_features": final_features,
        "latency_ms": latency_ms,
    }

    return templates.TemplateResponse(
        "scoring.html",
        {"request": request, "single_result": single_result, "batch_result": None, "error": None},
    )


@router.post("/ui/score/batch", response_class=HTMLResponse)
async def score_batch(request: Request, file: UploadFile = File(...)):
    content = await file.read()

    try:
        raw_df = pd.read_csv(io.BytesIO(content), dtype={"age": str, "gender": str})
    except Exception as exc:
        return templates.TemplateResponse(
            "scoring.html",
            {"request": request, "single_result": None, "batch_result": None,
             "error": f"Не удалось прочитать CSV: {exc}"},
        )

    missing = [c for c in RAW_BATCH_COLUMNS if c not in raw_df.columns]
    if missing:
        return templates.TemplateResponse(
            "scoring.html",
            {"request": request, "single_result": None, "batch_result": None,
             "error": f"В CSV не хватает колонок: {', '.join(missing)}"},
        )

    rows = raw_df[RAW_BATCH_COLUMNS].to_dict("records")
    frame, enriched_rows = _final_frame(request, rows)

    model = request.app.state.pipeline
    threshold_b = request.app.state.meta["threshold"]
    top_n = request.app.state.explainer_top_n

    scores = float(model.predict_proba(frame)[:, 1])
    reasons_per_row = request.app.state.explainer.explain(frame, top_n=top_n)

    decisions, classes, reasons_fmt = [], [], []
    for score, reasons in zip(scores, reasons_per_row, strict=True):
        cls, decision, _ = decide(float(score), threshold_b)
        classes.append(cls)
        decisions.append(decision)
        reasons_fmt.append(
            request.app.state.explainer.format_reasons(reasons)
        )

    result_df = raw_df.copy()
    result_df["p_fraud"] = scores
    result_df["class"] = classes
    result_df["decision"] = decisions
    result_df["reasons"] = reasons_fmt

    uploads_dir = Path(getattr(settings, "uploads_dir", "uploads"))
    uploads_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"predictions_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out_path = uploads_dir / out_name
    result_df.to_csv(out_path, index=False)

    counts = result_df["decision"].value_counts()

    batch_result = {
        "total": len(result_df),
        "block": int(counts.get("ЗАБЛОКИРОВАТЬ", 0)),
        "approve": int(counts.get("ПРОПУСТИТЬ", 0)),
        "review": int(counts.get("НА РУЧНУЮ ПРОВЕРКУ", 0)),
        "saved_path": str(out_path),
        "rows": result_df.head(50).to_dict("records"),
    }

    return templates.TemplateResponse(
        "scoring.html",
        {"request": request, "single_result": None, "batch_result": batch_result, "error": None},
    )
