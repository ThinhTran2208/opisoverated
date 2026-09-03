# -*- coding: utf-8 -*-
"""FastAPI router serving individual images directly from immutable ZIPs."""

from __future__ import annotations

try:
    from fastapi import APIRouter, HTTPException, Response
except ModuleNotFoundError:  # Keep core recommendation usable without FastAPI.
    APIRouter = None
    HTTPException = None
    Response = None


def create_image_router(image_resolver):
    if APIRouter is None or HTTPException is None or Response is None:
        raise RuntimeError("FastAPI is required to create the ZIP image router")

    router = APIRouter(prefix="/recommendation", tags=["recommendation"])

    @router.get("/images/{item_id}")
    def get_recommendation_image(item_id: str):
        try:
            payload = image_resolver.read_bytes(item_id)
            media_type = image_resolver.media_type(item_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="item image not found") from error
        return Response(
            content=payload,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return router

