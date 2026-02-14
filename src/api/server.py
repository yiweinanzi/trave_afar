"""
FastAPI unified service entrypoint.
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.recommendation import RecommendationRequest, RecommendationResponse
from service.pipeline import get_pipeline


app = FastAPI(title="GoAfar API", version="2.0.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    pipeline = get_pipeline()
    return {"status": "ready", "embedding_ready": bool(getattr(pipeline, "_embedding_ready", False))}


@app.post("/v1/recommend/itinerary", response_model=RecommendationResponse)
def recommend_itinerary(request: RecommendationRequest):
    pipeline = get_pipeline()
    response = pipeline.recommend(request)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    return response
