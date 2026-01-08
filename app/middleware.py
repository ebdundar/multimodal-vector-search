"""Middleware utilities under app.middleware for package imports."""
import time
import uuid
from typing import Callable
from fastapi import Request
from starlette.types import ASGIApp
from app.logger import get_logger, set_request_id, clear_request_id

log = get_logger("middleware")


def request_id_middleware_factory(app: ASGIApp) -> Callable:
    async def middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(rid)
        start = time.time()
        try:
            response = await call_next(request)
            duration_ms = int((time.time() - start) * 1000)
            log.info("request_complete", extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "request_id": rid,
            })
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            clear_request_id()

    return middleware

