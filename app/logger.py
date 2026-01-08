"""Structured JSON logger exported under `app.logger` for package imports."""
import logging
import os
import json
import contextvars
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import LOG_LEVEL as CONFIG_LOG_LEVEL

LOG_LEVEL = os.getenv("LOG_LEVEL", CONFIG_LOG_LEVEL).upper()

request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


def set_request_id(rid: Optional[str]) -> None:
    request_id_ctx.set(rid)


def get_request_id() -> Optional[str]:
    return request_id_ctx.get()


def clear_request_id() -> None:
    request_id_ctx.set(None)


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        try:
            rid = get_request_id()
        except Exception:
            rid = None
        if rid:
            payload["request_id"] = rid
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(LOG_LEVEL)
        handler = logging.StreamHandler()
        handler.setLevel(LOG_LEVEL)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger

